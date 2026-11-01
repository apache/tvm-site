..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Symbolic Integer Analysis
=========================

Before combining four scalar accesses into one vector access, a compiler
pass needs to check the index calculation and make sure all four elements
are within the tensor. When the tensor length is known only at runtime,
these questions involve symbolic expressions. We will use
:py:class:`~tvm_tirx.sym.Analyzer` to simplify those expressions, find their
bounds, and prove conditions using facts from loops and branches. We will
then use the same groups of four to examine coordinate mappings and
accessed regions.

The core types separate the facts an analysis knows, the analyzer that
uses them, and the results of its queries:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Type
     - Purpose
   * - :py:class:`~tvm_tirx.sym.Analyzer`
     - Uses known facts to simplify expressions, infer bounds, and prove conditions.
   * - :py:class:`~tvm_tirx.sym.ConstraintContext`
     - Holds facts valid at a particular point in the program.
   * - :py:class:`~tvm_tirx.sym.RangeConstraint`
     - Records a variable's range of possible values.
   * - :py:class:`~tvm_tirx.sym.PredicateConstraint`
     - Records a condition known to hold at that program point.
   * - :py:class:`~tvm_tirx.sym.BindingConstraint`
     - Records the value captured when a variable is bound.
   * - :py:class:`~tvm_tirx.sym.IntSet`, :py:class:`~tvm_tirx.sym.IntervalSet`
     - Describe possible integer values. ``IntervalSet`` represents an inclusive
       interval whose endpoints can contain symbolic expressions.
   * - :py:class:`~tvm_tirx.sym.ConstIntBound`
     - Gives inclusive numeric lower and upper bounds.
   * - :py:class:`~tvm_tirx.sym.ModularSet`
     - Describes integer divisibility and remainder patterns.

We will first record a loop's range in a constraint context, then use an
analyzer to simplify and check its indices. The queries will show how to
interpret the different result types.

Starting from a Loop
--------------------

The following function adds one to each element of ``A`` and writes the
result to ``B``. ``T.dynamic("n")`` creates a symbolic length shared by the
tensor shapes and loop bound:

.. code-block:: python

   from tvm_tirx import ir, sym, tirx
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + 1

Inside the loop, ``i`` satisfies ``0 <= i < n``. We can obtain that
variable from the loop's IR and ask an analyzer whether ``i < n`` holds:

.. code-block:: python

   loop = add_one.body[0]  # The for loop is the function's first statement.
   i = loop.var
   analyzer = sym.Analyzer()
   print(analyzer.can_prove(i < n))

The output is ``False``, meaning the analyzer has not proved the condition.
It has only received the expression ``i < n``. The variable ``i`` records
its name and type, while its bounds belong to the surrounding loop. The
analyzer needs those bounds as facts before it can use them in a proof.

A ``ConstraintContext`` holds such facts. Add the loop range, then
synchronize the analyzer with the context:

.. code-block:: python

   loop_context = sym.ConstraintContext()
   loop_context.append(sym.RangeConstraint(i, ir.Range(loop.begin, loop.end)))
   analyzer.sync_constraint_context(loop_context)
   print(analyzer.can_prove(i < n))

The output is now ``True``. ``ir.Range(begin, end)`` describes a half-open
interval, including ``begin`` and excluding ``end``. Here it is ``[0, n)``,
matching the values taken by ``i``. Subsequent queries on this analyzer can
use the same range fact.

Splitting and Recombining an Index
----------------------------------

To group consecutive elements in fours, split an index into its group's
starting position and its position within that group, called a lane.
For example, index 6 belongs to the group starting at 4 and occupies
lane 2. Division and remainder express this split:

.. code-block:: python

   group_start = (i // 4) * 4
   lane = i % 4
   recombined = group_start + lane
   print(analyzer.simplify(recombined))
   print(analyzer.can_prove_equal(recombined, i))

The output is:

.. code-block:: text

   i
   True

``simplify`` returns a simpler IR expression that a pass can use in place
of the longer calculation. ``can_prove_equal`` instead returns a boolean:
it confirms that the two expressions compute the same value, even though
their node structures differ.

Checking a Group's Bounds
--------------------------

Recombining the group start and lane recovers the scalar index ``i``.
To read the whole group, we also need to check the lane positions,
the group's start, and whether its last element is within the tensor.

First, find the lane's numeric bounds with ``const_int_bound``:

.. code-block:: python

   lane_bounds = analyzer.const_int_bound(lane)
   print(lane_bounds.min_value, lane_bounds.max_value)

The output is ``0 3``. Both endpoints are inclusive, so the lane selects
one of four positions. We can also check whether each group starts at a
multiple of four:

.. code-block:: python

   starts = analyzer.modular_set(group_start)
   print(starts.coeff, starts.base)

The output is ``4 0``. A modular set describes values of the form
``coeff*k + base`` for integer ``k``, so the group starts at ``4*k``.
This establishes the element offset's divisibility. Aligned vector
accesses also depend on the tensor's base-address alignment.

The loop index has symbolic bounds because the tensor length is not fixed.
``int_set`` can express those bounds in terms of ``n``:

.. code-block:: python

   indices = analyzer.int_set(i)
   print(analyzer.simplify(indices.min_value))
   print(analyzer.simplify(indices.max_value))

The output is:

.. code-block:: text

   0
   n - 1

For this expression, the result is an ``IntervalSet`` with inclusive
endpoints. Unlike ``const_int_bound``, these endpoints are IR expressions
and can contain variables. In general, a computed bound covers the
possible values and may also include values that never occur.

Every scalar index ``i`` is within those bounds. A full group ends at
``group_start + 3``, which can be larger than ``i``. We can ask whether
that last index is also in bounds:

.. code-block:: python

   group_last = group_start + 3
   print(analyzer.can_prove(i < n))
   print(analyzer.can_prove(group_last < n))

The output is:

.. code-block:: text

   True
   False

For example, when ``n = 6`` and ``i = 5``, the group starts at 4 and ends
at 7. The original scalar access is valid, but indices 6 and 7 are outside
the tensor. A pass must handle this partial group, for example by using
scalar accesses for the tail.

A ``False`` proof result means the analyzer did not establish the condition.
It does not prove the opposite. In this case we have a concrete
counterexample, but other queries may need additional facts or stronger
reasoning.

.. _symbolic-constraint-management:

Following Facts into a Branch
------------------------------

One way to handle partial groups is to test ``group_last < n`` before
taking a path that accesses all four elements. Inside the true branch,
that test becomes an additional fact. If the last element is in bounds,
the earlier element at ``group_start + 2`` is also safe. We can represent
the branch with a snapshot of the loop context and ask the analyzer to
prove this consequence:

.. code-block:: python

   third_index = group_start + 2
   branch_context = loop_context.snapshot()
   branch_context.append(sym.PredicateConstraint(group_last < n))
   analyzer.sync_constraint_context(branch_context)
   print(analyzer.can_prove(third_index < n))

   analyzer.sync_constraint_context(loop_context)
   print(analyzer.can_prove(third_index < n))

The output is:

.. code-block:: text

   True
   False

The first query uses the branch condition to prove that the third element
is in bounds.
Synchronizing with ``loop_context`` again models leaving the branch and
removes the branch-only fact. The original context remains unchanged
because we extended a snapshot.

Adding the predicate represents a test that has already succeeded at this
program point. It does not show that the test is true on every iteration.
Likewise, appending a fact to a context affects an analyzer only after
``sync_constraint_context`` is called. Keeping the context synchronized is
what makes the same analyzer usable at different points in a program.

.. _ir-traversal-with-constraints:

Getting Facts During Traversal
------------------------------

In a pass over a complete function, manually reconstructing the facts for
every statement would be tedious. TIRx provides
:py:func:`~tvm_tirx.tirx.structural.structural_walk_with_constraint_context` to collect
them while walking the IR. Its callback receives both the current value
and the facts valid there.

The group check above protects accesses that may extend beyond the tensor.
We will now examine a scalar check, ``i < n``, which repeats a condition
already guaranteed by the loop. We can use the traversal to identify this
unnecessary check:

.. code-block:: python

   @T.function
   def guarded_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           if i < n:
               B[i] = A[i] + 1

   guard_analyzer = sym.Analyzer()
   proven_guards = []

   def inspect_guard(value, constraints):
       if isinstance(value, ir.If):
           guard_analyzer.sync_constraint_context(constraints)
           proven_guards.append(guard_analyzer.can_prove(value.condition))

   tirx.structural_walk_with_constraint_context(
       guarded_add_one, inspect_guard, order="pre"
   )
   print(proven_guards)

The output is ``[True]``. At the ``ir.If`` node, the callback synchronizes
the analyzer with the surrounding loop's range. That range is enough to
prove ``i < n``. A pass can therefore remove this check.

The pre-order callback runs before entering either branch. This matters
because using the condition as a fact inside its own true branch would
only show that it holds there, not that the test is redundant. Synchronize
at each callback that makes a query so facts from a previous program point
are replaced by the ones valid now.

Do not append facts to the context passed to the callback or modify the
facts it contains. To explore an additional assumption, extend a snapshot
as in the branch example. The
:doc:`TIRx traversal API </reference/python/tirx>`
describes the collection rules for bindings, loops, branches, and other
constructs. :doc:`writing_passes` uses the same guarded function in a
transformation pipeline.

.. _symbolic-iteration-maps:

Mapping Groups and Lanes
------------------------

Bounds tell us whether a group's accesses fit. To replace a linear loop
with group and lane loops, we also need to check that the coordinate mapping
covers each original index exactly once. Consider a fixed length of 16,
which forms four complete groups. ``k // 4`` selects a group, and
``k % 4`` selects a lane within it.

``detect_iter_map`` analyzes these coordinates over the input domain.
``check_level="bijective"`` asks it to check that every group-and-lane pair
has exactly one input index. We use a fresh analyzer for this fixed domain:

.. code-block:: python

   k = ir.Var("k", "int32")
   map_analyzer = sym.Analyzer()
   detected = sym.detect_iter_map(
       [k // 4, k % 4],
       {k: ir.Range(0, 16)},
       check_level="bijective",
       analyzer=map_analyzer,
   )
   assert len(detected.indices) == 2, detected.errors

The two entries in ``detected.indices`` describe the group and lane
coordinates. If detection fails, the array is empty and ``errors`` gives
diagnostics. With a successful bijective mapping, we can recover the
original index from the output coordinates:

.. code-block:: python

   group = ir.Var("group", "int32")
   lane_index = ir.Var("lane_index", "int32")
   inverse = sym.inverse_affine_iter_map(detected.indices, [group, lane_index])
   print(inverse[k])
   print(map_analyzer.can_prove_equal(inverse[k], group * 4 + lane_index))

The output is:

.. code-block:: text

   lane_index + group * 4
   True

The recovered expression gives the original index. For example, group 1
and lane 2 give index 6. The equality query confirms the expected formula
regardless of the printed operand order.

The detector checks an index calculation over a specified domain.
:py:class:`~tvm_tirx.tirx.IndexMap` instead records a coordinate rule that a
pass can apply to indices and shapes. We can define the same split into
groups and lanes as an index map:

.. code-block:: python

   split = tirx.IndexMap.from_func(lambda index: (index // 4, index % 4))
   print([int(v) for v in split.map_indices([6])])
   print([int(v) for v in split.map_shape([16])])

The output is:

.. code-block:: text

   [1, 2]
   [4, 4]

``map_indices`` maps index 6 to group 1, lane 2. ``map_shape`` maps the
16-element shape to four groups of four lanes. These queries describe
the coordinate transformation. A pass uses that information when rewriting
loops and tensor accesses.

.. _symbolic-region-analysis:

Covering an Accessed Region
---------------------------

The mapping tells us how to name individual elements. We may also need to
find the whole region accessed by a loop, for example to size a scratch
tensor. Suppose group 0 accesses elements 0 through 3, group 1 accesses
4 through 7, and so on. Each group's access is a range starting at
``group * 4`` with extent 4:

.. code-block:: python

   region = [ir.Range.from_min_extent(group * 4, 4)]
   covered = sym.estimate_region_strict_bound(
       region,
       {group: ir.Range(0, 4)},
       predicate=True,
       analyzer=map_analyzer,
   )
   assert covered is not None
   print(map_analyzer.simplify(covered[0].min_value))
   print(map_analyzer.simplify(covered[0].max_value))

The output is:

.. code-block:: text

   0
   15

``from_min_extent`` takes a starting value and a size. The estimator combines
that range over all four group indices. ``predicate=True`` includes every
iteration, and the result covers all sixteen elements. The input ranges
are half-open, while the returned integer-set endpoints are inclusive.
A scratch tensor covering indices 0 through 15 therefore needs 16 elements.

Here the groups fill one contiguous interval. If each group accessed only
two elements, the accessed indices would instead be ``[0, 1]``, ``[4, 5]``,
``[8, 9]``, and ``[12, 13]``. An upper bound of ``[0, 13]`` would cover all
those accesses but also include the gaps. The estimator's guarantee matters
when using its result:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Estimator
     - Guarantee
   * - ``estimate_region_upper_bound``
     - Includes every accessed point, and may include additional points.
   * - ``estimate_region_lower_bound``
     - Includes only accessed points, but may omit some of them.
   * - ``estimate_region_strict_bound``
     - Represents the accessed region without adding or omitting points.

A lower bound alone does not guarantee enough storage for every access.
Strict and lower-bound estimation can return ``None`` when they cannot
produce the required result. The
:doc:`symbolic analysis API </reference/python/sym>` covers other region
queries and iteration-map operations.

Stronger Proofs
----------------

The default proof strength can be inconclusive even when a condition follows
from the known facts. Consider scratch storage rounded up to a multiple of
four. It has ``(n + 3) // 4`` complete groups. When ``n = 6``, for example,
there are two groups and room for eight elements.

We want to prove that any group and lane select an index within this padded
allocation. ``g`` is the group number and ``lane_id`` is its lane:

.. code-block:: python

   group_count = (n + 3) // 4
   padded_length = group_count * 4
   g = ir.Var("g", "int32")
   lane_id = ir.Var("lane_id", "int32")

   padded_context = sym.ConstraintContext()
   padded_context.append(sym.PredicateConstraint(n >= 0))
   padded_context.append(sym.RangeConstraint(g, ir.Range(0, group_count)))
   padded_context.append(sym.RangeConstraint(lane_id, ir.Range(0, 4)))
   prover = sym.Analyzer()
   prover.sync_constraint_context(padded_context)

   fits_allocation = g * 4 + lane_id < padded_length
   print(prover.can_prove(fits_allocation))
   print(prover.can_prove(fits_allocation, sym.ProofStrength.SYMBOLIC_BOUND))

The following output is from a build without Z3:

.. code-block:: text

   False
   True

``SYMBOLIC_BOUND`` requests additional reasoning with symbolic bounds.
Here, native analysis combines the group and lane ranges to prove
``g * 4 + lane_id < padded_length``. With ``n = 6``, the final two positions
are still outside the original tensor, so accesses to that tensor still
need tail handling.

In a build with ``TVM_TIRX_USE_Z3=ON``, both ``DEFAULT`` and
``SYMBOLIC_BOUND`` try Z3 after native reasoning fails. A default query may
already succeed through this fallback.
``prover.is_z3_enabled`` reports whether the current build supports it.
The :py:class:`~tvm_tirx.sym.Analyzer` reference documents solver timeouts,
resource limits, and proof diagnostics. A ``False`` result still means
unproved, including when the solver cannot handle a condition or reaches
a limit.
