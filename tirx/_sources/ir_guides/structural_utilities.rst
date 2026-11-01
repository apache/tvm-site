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

Structural Traversal, Rewriting, and Comparison
===============================================

Most compiler analyses and transformations come down to two patterns:
traversing a program to collect information, or rewriting part of it. TIRx
provides first-class support for both through its foundation, TVM-FFI.

Every TIRx IR node registers **structural hooks** that tell these utilities
how to traverse and rewrite it, including which fields define variables.
The hooks are compiled, so traversal runs at native speed. Objects without
hooks fall back to their reflected fields, and new IR types can
:ref:`register the same hooks <ffi-structural-hooks>` to get the same
support. On top of these hooks, TVM-FFI provides:

- **Walk**: call our function on every node of a given type, to collect
  information or search for a match.
- **Map**: replace matching nodes and get back a new program, rebuilt
  around the replacements.
- **Equality and Hashing**: compare and hash programs by structure rather
  than object identity, to check results in tests or to deduplicate equal
  programs.
- **Visit and Mutate**: let our function decide when and whether to
  process a node's children, for advanced use cases that need fine-grained
  control over the traversal order.

We start with walk and map, which cover most everyday tasks, then compare
programs, look at how variables are defined and used, and finish with visit
and mutate.

Throughout this chapter, we use the ``add_one`` function from Quick Start:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

.. note::

   The runnable code in this chapter is under
   `examples/ir_guides <https://github.com/apache/tvm-tirx/tree/main/examples/ir_guides>`_
   in the repository.

Structural Walk
---------------

Let's start by inspecting a program. Before we traverse ``add_one``, it
helps to see what we will be traversing.
:py:func:`ir.format_tree <tvm_tirx.ir.format_tree>` prints the tree of IR
nodes that ``add_one`` is made of:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [tree.begin]
   :end-before: # [tree.end]

The output is:

.. code-block:: text

   Function
   ├── params[0]: A = Var(ty=TensorType(shape=[n], dtype=float32))
   ├── params[1]: B = Var(ty=TensorType(shape=[n], dtype=float32))
   └── body: SeqStmt
       └── seq[0]: For(begin=IntImm(value=0), end=n)
           ├── var: i = Var(ty=int32)
           └── body: SeqStmt
               └── seq[0]: TensorStore(dest=B, indices=[i])
                   └── value: Add
                       ├── a: TensorLoad(source=A, indices=[i])
                       └── b: FloatImm(value=1.0)

Most analyses care about only a few kinds of node in this tree. To find
the tensors that ``add_one`` writes, for example, we only need the
``TensorStore`` nodes. ``tvm_ffi.structural_walk`` visits every node under
a root and calls our function on the nodes that match a type, so we
pair the store type with a callback:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [walk.begin]
   :end-before: # [walk.end]

The output is:

.. code-block:: text

   ['B']

The walker descends through the whole tree on its own, from the function
down to the loop body. Our callback runs only once, on the ``TensorStore``
node for ``B[i] = A[i] + 1``.

Matching Nodes to Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we matched a single type, but an analysis often needs several kinds
of node at once, or every node. The callback argument accepts these forms:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Argument
     - Calls
   * - ``(ir.TensorStore, on_store)``
     - ``on_store`` on each write.
   * - ``((ir.TensorStore, ir.TensorLoad), f)``
     - ``f`` on each write and each read, one node per call.
   * - ``[(ir.For, on_loop), (ir.TensorStore, on_store)]``
     - ``on_loop`` on each loop and ``on_store`` on each write. A tuple of
       pairs works the same way.
   * - ``f``
     - ``f`` on every value the walk reaches.

Matching follows the class hierarchy, so a type also matches its
subclasses: ``(ir.Stmt, f)`` receives every statement. When several pairs
match a node, they are tried in order and only the first one runs. A
callback that returns ``None`` lets the walk continue; the next sections
show what else it can return.

Traversal Order
~~~~~~~~~~~~~~~

Our first walk did not depend on the order in which nodes are reached,
but many analyses do. An analysis that tracks state, for example, may need
to see a loop before the statements inside it. By default, the walker calls
a callback after visiting the node's children (post-order), and
``order="pre"`` calls it before them. Let's record every statement both
ways:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [order.begin]
   :end-before: # [order.end]

The output is:

.. code-block:: text

   pre ['SeqStmt', 'For', 'SeqStmt', 'TensorStore']
   post ['TensorStore', 'SeqStmt', 'For', 'SeqStmt']

The function body and the loop body are each a ``SeqStmt``, a sequence of
statements. Pre-order reaches the loop before the write inside it;
post-order reaches the write first.

Skipping and Stopping
~~~~~~~~~~~~~~~~~~~~~

So far, every walk has covered the whole tree. Often we can do less: an
analysis may not care what is inside a loop, and a search can end as soon
as it finds a match. A callback steers the walk through its return value.
In pre-order, returning ``WalkResult.SKIP`` tells the walker not to descend
into the node's children:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [skip.begin]
   :end-before: # [skip.end]

The output is:

.. code-block:: text

   ['SeqStmt', 'For']

The walk records the loop but never enters it, so the loop body and the
write are not reached.

For a search, a callback returns ``VisitInterrupt(payload)`` instead. This
stops the whole walk, and ``structural_walk`` returns the interrupt, so the
payload carries the result back to us. Here we look for the first write:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [stop.begin]
   :end-before: # [stop.end]

The output is:

.. code-block:: text

   B

The walk ends at the first write, and ``found.value`` holds it. A walk that
finishes without an interrupt returns ``None``, so the result also tells us
whether anything was found.

Structural Map
--------------

Walks let us inspect a program; a transformation also needs to rewrite it.
``tvm_ffi.structural_map`` takes the same callbacks as a walk, but each one
returns a replacement for its node. The mapper rebuilds the parents of
everything that changed and returns the new root.

Suppose ``add_one`` is always called on tensors of 128 elements. Knowing
``n`` ahead of time, we can specialize the function by replacing ``n`` with
a constant:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [specialize.begin]
   :end-before: # [specialize.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
       for i in range(128):
           B[i] = A[i] + T.float32(1.0)

Every occurrence of ``n``, in both parameter shapes and the loop bound,
becomes ``128``. Because IR is immutable, the map returns a new function
and leaves ``add_one`` unchanged. Note that a map callback must always
return a node, even when it has nothing to change: returning its input
keeps that node as it is, which is what ``specialize`` does for every other
variable.

Immutability also keeps rewrites cheap: a map does not copy the whole
program. It rebuilds only the nodes on the path from each replacement up to
the root, and the new program shares every unchanged part with the
original. The original program remains valid, so we can keep it to inspect
or compare.

Structural Equality and Hashing
-------------------------------

Once we can rewrite programs, we need a way to check the result. To test a
transformation, we write down the program we expect and compare it with the
one the transformation produced. The two programs are built separately, so
they are different objects even when they describe the same computation,
and comparing them by identity cannot work. Instead, we compare them by
structure. Let's build the same function twice with a small helper:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [equal.begin]
   :end-before: # [equal.end]

The output is:

.. code-block:: text

   False
   True
   True

The two functions are different objects, yet ``tvm_ffi.structural_equal``
finds them equal, and ``tvm_ffi.structural_hash`` gives them the same hash.
Each call to ``make_add`` also creates its own ``n``, ``A``, ``B``, and
``i``. The comparison matches each variable defined in one program with its
counterpart in the other, so these different variable objects do not make
the programs unequal.

A variable that is not defined anywhere in the compared values is free and
has no definition to match. By default, a free variable is equal only to
itself. ``map_free_vars=True`` pairs free variables up as well:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [free_vars.begin]
   :end-before: # [free_vars.end]

The output is:

.. code-block:: text

   False
   True

In a test, we usually want more than a yes-or-no answer.
``ir.assert_structural_equal`` raises an error that shows where the
programs differ. Let's compare ``first`` with ``make_add(2)``, its ``+ 2``
variant:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [assert_equal.begin]
   :end-before: # [assert_equal.end]
   :dedent: 4

The error reads, in part:

.. code-block:: text

   StructuralEqual check failed, caused by lhs at <root>.body.seq[0].body.seq[0].value.b.value:
   ...
           B[i] = A[i] + T.float32(1.0)
                         ^^^^^^^^^^^^^^
   ...
   and rhs at <root>.body.seq[0].body.seq[0].value.b.value:
   ...
           B[i] = A[i] + T.float32(2.0)
                         ^^^^^^^^^^^^^^

The error names the access path of the first difference, the chain of
fields from the function down to the constant, and underlines that
constant in both programs.

Equality and hashing are also useful inside the compiler itself, whenever
we need to detect that two functions or subtrees are the same program.
``tvm_ffi.StructuralKey`` wraps a value so that a Python set or dictionary
compares it by structure, which lets us deduplicate equal programs:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [dedup.begin]
   :end-before: # [dedup.end]

The output is:

.. code-block:: text

   2

The two ``make_add(1)`` functions collapse into one entry, and
``make_add(2)`` keeps its own.

Definitions and Uses
--------------------

A variable, represented by :py:class:`ir.Var <tvm_tirx.ir.Var>`, can
appear in a program for one of two purposes. A **definition site**
introduces the variable, and a **use site** reads its value. In
``add_one``, the parameters are the definition sites of ``A``, ``B``, and
the length ``n``, and the loop is the definition site of ``i``. The loop
bound ``range(n)`` is a use site of ``n``, and ``B[i] = A[i] + 1``
contains use sites of ``A``, ``B``, and ``i``. Each variable has exactly
one definition site and can have any number of use sites.

In the IR, every site refers to the same ``ir.Var`` object. Many tasks need
to tell the two purposes apart. To rename ``i``, we create the new variable
once at its definition site and point every use site at it. To find an
unused variable, we look for a definition site with no use sites. To check
whether two programs are the same, we match each definition site in one
with its counterpart in the other. The variable object alone does not say
which purpose it serves at a given site, but its position in the program
does.

TIRx IR nodes use their structural hooks to mark which of their fields are
definition sites. As the utilities traverse the program, they report the
role at each position as a **definition region kind**:

- ``NONE`` marks a use site, such as ``n`` in ``range(n)`` and ``A``,
  ``B``, and ``i`` in the body.
- ``DEF_SIMPLE`` defines exactly the variable at the site, such as the
  loop variable ``i``. A ``Bind`` statement defines its variable the same
  way.
- ``DEF_PATTERN`` defines every variable in the site by matching,
  including those in its type.

A ``DEF_PATTERN`` site is a pattern that is matched against the values
passed in. In ``add_one``, the parameter ``A: T.Tensor((n,), T.float32)``
is matched against the tensor passed for ``A``. This defines both ``A`` and
its length ``n``, so ``n`` needs no separate declaration. A variable can
appear more than once in a pattern. The first appearance defines it, and
each later appearance adds a constraint that is checked at runtime. Here,
``n`` also appears in ``B``'s shape, so ``B`` must have the same length.
When we call ``add_one`` with an ``A`` of length 100, ``n`` is bound to
100, and passing a ``B`` of length 50 raises an error.

Importantly, the structural utilities are themselves aware of definitions
and uses. When ``structural_map`` rewrites the type of a variable at its
definition site, it creates a new variable with the new type and remaps
every use site to it. For example, replacing ``n`` with 128 in ``add_one``
changes the type of ``A`` from ``T.Tensor((n,), T.float32)`` to
``T.Tensor((128,), T.float32)``, so the result has a new ``A``, and
``A[i]`` in the body refers to that new ``A``. Our callback only replaced
``n``; the mapper kept every definition and its uses consistent.

To receive the kind in our own function, we register it through
``with_def_region_kind`` instead of ``callbacks``, and the function is
called as ``fn(var, kind)``. The kind gives our function additional context
about each site, which analyses and rewrites can use to treat definitions
and uses differently. The following walk prints each definition site in
``add_one``:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [definitions.begin]
   :end-before: # [definitions.end]

The output is:

.. code-block:: text

   A DEF_PATTERN
   n DEF_PATTERN
   B DEF_PATTERN
   n DEF_PATTERN
   i DEF_SIMPLE

``n`` appears twice, once for each shape: the first defines it, and the
second is the constraint. Every other site is a use site, reported as
``NONE``.

``structural_map`` accepts the same form. A rename leaves each variable's
type unchanged, so we create the new variables ourselves: a new variable at
each definition site, returned as the same object at every use site:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [rename.begin]
   :end-before: # [rename.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one[n1](A1: T.Tensor((n1,), T.float32), B1: T.Tensor((n1,), T.float32)):
       for i1 in range(n1):
           B1[i1] = A1[i1] + T.float32(1.0)

Pre-order traversal reaches each definition site before its use sites, so
the new variable exists by the time a use needs it. The
``var not in new_vars`` check keeps the first replacement for ``n``, so both
shapes share ``n1``.

Advanced Control: Visit and Mutate
----------------------------------

Walk and map recurse for us, and that covers most tasks. Some analyses,
however, need to know where they are in the program, not just which node
they are looking at. To report the loops around each write, for example, a
callback must know when the traversal enters a loop and when it leaves
again. ``tvm_ffi.structural_visit`` and ``tvm_ffi.structural_mutate`` hand
that control to us: a matching callback also receives the visitor or
mutator and decides when, or whether, to process the node's children. If it
never does, the children are skipped. Let's record the loops around each
write:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [visit.begin]
   :end-before: # [visit.end]

The output is:

.. code-block:: text

   [('B', ['i'])]

``visit_for`` pushes the loop variable, lets ``default_visit`` visit the
loop's children, then pops the variable, so ``loops`` always names the
loops around the current node. ``default_visit`` processes a node's
children with their usual traversal, without calling the node's own
callback again; ``default_mutate`` does the same for a mutator.

Because the callback now drives the recursion, it also takes on what the
walker used to do for us. ``visit_for`` returns the result of
``default_visit``, so that a ``VisitInterrupt`` from a child reaches the
caller.

Mutate brings the same control to rewriting. A mutate callback returns the
new node, as in a map, but it can also set up state before its children are
rewritten. Let's reverse the loop by replacing each use of ``i`` with
``n - 1 - i``:

.. literalinclude:: ../../examples/ir_guides/structural_utilities.py
   :language: python
   :start-after: # [mutate.begin]
   :end-before: # [mutate.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[n - 1 - i] = A[n - 1 - i] + T.float32(1.0)

When the mutate reaches the loop, ``reverse_loop`` records the replacement
``n - 1 - i`` for ``i`` in the ``reversed_index`` dictionary, then lets
``default_mutate`` rewrite the loop's children, so the replacement is ready
before any use of ``i`` is reached. The loop's own ``i`` is a definition
and must stay a variable, so ``reverse_uses`` replaces ``i`` only where
``def_region_kind()`` is ``NONE``, that is, at uses.

These utilities cover the common ways to traverse, rewrite, and compare IR,
but they are a convenience, not a requirement. IR objects are ordinary
objects whose fields we can read directly, so we can still write our own
recursive visitors and mutators by hand when a task calls for it, and mix
them with the structural utilities as needed.
