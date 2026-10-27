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

In this chapter, we will introduce symbolic integer analysis for tensor
shapes, loop bounds, and element indices. A pass that combines four scalar
accesses into one vector access needs to preserve their positions and
establish that the whole group is within the tensor. The length may only
be known at runtime, so these decisions require reasoning about symbolic
expressions. We will start from a loop's known range, then examine grouped
indices and a neighboring-element access.

The Analyzer
------------

:class:`sym.Analyzer <tvm_tirx.sym.Analyzer>` answers queries about integer
IR expressions. It combines algebraic rules with facts supplied by a pass,
such as a loop variable's range. The same analyzer can use those facts
across several queries:

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Query
     - Arguments
     - Result
   * - ``simplify(expr)``
     - An IR expression.
     - An equivalent IR expression in a simpler form.
   * - ``int_set(expr)``
     - An integer IR expression.
     - A ``sym.IntSet`` covering its possible values, including symbolic bounds.
   * - ``const_int_bound(expr)``
     - An integer IR expression.
     - A ``sym.ConstIntBound`` with inclusive numeric bounds.
   * - ``modular_set(expr)``
     - An integer IR expression.
     - A ``sym.ModularSet`` describing divisibility and remainders.
   * - ``can_prove(expr)``
     - A boolean IR expression.
     - ``True`` if the condition is established from the known facts.
   * - ``can_prove_equal(lhs, rhs)``
     - Two integer IR expressions.
     - ``True`` if their values are established to be equal.

These calls show the required arguments. The expressions describe
calculations in the program, and the analyzer uses its known facts to
reason about their runtime values.

Facts from the Program
----------------------

The analyzer needs facts from the point in the program where a rewrite
would occur. In a loop over a contiguous tensor of symbolic length ``n``,
the index ``i`` satisfies ``0 <= i < n``. We can represent the length and
index directly with integer IR variables:

.. code-block:: python

   from tvm_tirx import ir, sym

   n = ir.Var("n", "int32")
   i = ir.Var("i", "int32")
   analyzer = sym.Analyzer()

The loop bounds belong to the loop statement, while the variable ``i``
records a name and type. A query on ``i`` therefore does not automatically
include those bounds. The pass supplies them through a ``RangeConstraint``;
a ``ConstraintContext`` collects the facts for the current program point:

.. code-block:: python

   range_fact = sym.RangeConstraint(i, ir.Range(0, n))
   context = sym.ConstraintContext()
   context.append(range_fact)
   analyzer.sync_constraint_context(context)

The range starts at zero and has extent ``n``:

.. code-block:: text

   range_fact: sym.RangeConstraint
   ├── var → ir.Var                     i
   └── range: ir.Range
       ├── begin: ir.IntImm              0
       └── extent: ir.Var                n

``sync_constraint_context`` makes these facts available to subsequent
queries. The analyzer captures the context at synchronization, so later
additions need another synchronization. A pass moving to another part
of the program must supply the context valid there.

Simplifying Index Expressions
-----------------------------

A vectorization pass can split the loop index ``i`` into a group start
and a lane within a group of four. The address calculation may later
recombine these parts, leaving division and remainder operations that can
be removed. The group starts at ``(i // 4) * 4``, and ``i % 4`` gives its
lane. For example, index 6 belongs to the group starting at 4 and occupies
lane 2. The analyzer can simplify the recombined expression:

.. code-block:: python

   group_start = (i // 4) * 4
   lane = i % 4
   simplified = analyzer.simplify(group_start + lane)
   print(simplified)

The output is:

.. code-block:: text

   i

``simplified`` is the IR expression a pass can use to replace the longer
index calculation.

Bounds and Divisibility
-----------------------

The loop range guarantees that ``A[i]`` is in bounds. A computation that
also reads ``A[i + 1]`` needs to establish a bound for that new index.
``int_set`` can bound its possible values before the pass decides whether
the access needs a guard:

.. code-block:: python

   next_index = i + 1
   possible_values = analyzer.int_set(next_index)
   print(analyzer.simplify(possible_values.min_value))
   print(analyzer.simplify(possible_values.max_value))

The output is:

.. code-block:: text

   1
   n

For this expression, ``int_set`` returns a ``sym.IntervalSet``. Its
``min_value`` and ``max_value`` fields are integer IR expressions, so
the endpoints can depend on ``n``. The interval is inclusive: ``i + 1``
can reach ``n`` on the last iteration. This matters because the final
valid tensor index is ``n - 1``.

The grouping calculation gives another bounds query: the lane must select
one of the group's four positions. Its bounds are numeric, so
``const_int_bound`` can report them directly:

.. code-block:: python

   lane_bounds = analyzer.const_int_bound(lane)
   print(lane_bounds.min_value, lane_bounds.max_value)

The output is:

.. code-block:: text

   0 3

These endpoints are also inclusive, describing the four lane positions.
Bounds cover every possible value; for more complex expressions, an
inferred interval may also contain values that never occur.

The start offset also matters for a vector access. A pass can use
``modular_set`` to check whether every group starts on a multiple of four:

.. code-block:: python

   multiples = analyzer.modular_set(group_start)
   print(multiples.coeff, multiples.base)

The output is:

.. code-block:: text

   4 0

``coeff`` and ``base`` describe values of the form ``coeff*k + base``
for integer ``k``. Every group therefore starts on a four-element boundary.
A vectorization pass can use this offset information together with the
base-address alignment and access bounds.

Proving Conditions
------------------

A compiler pass often needs a yes-or-no answer before it removes a check
or replaces an index. A proof query asks whether the required condition
holds for every value allowed by the known facts. The following calls
check the original index, the neighboring index, and the regrouped index:

.. code-block:: python

   print(analyzer.can_prove(i < n))
   print(analyzer.can_prove(next_index < n))
   print(analyzer.can_prove_equal(group_start + lane, i))

The output is:

.. code-block:: text

   True
   False
   True

The known range proves ``i < n``, so a redundant check of this condition
inside the loop can be removed. The neighboring index can reach ``n``,
so ``i + 1 < n`` cannot be established for every iteration. That access
needs a guard or an iteration range that excludes the last element.

In general, ``False`` means that the analyzer did not prove a condition;
it does not prove the opposite condition. A pass cannot use that result
to justify removing a bounds check.

``can_prove_equal`` compares expression values. Here, its ``True`` result
confirms that a pass can replace ``group_start + lane`` with ``i`` while
preserving the element index, even though their IR structures differ.

The analysis infrastructure and its APIs are under active development.
The :doc:`tvm_tirx.sym </reference/python/sym>` reference describes the
current interfaces.
