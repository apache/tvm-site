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

Working with IR in C++
======================

The IR objects we have used from Python are TVM-FFI objects implemented in
C++. A C++ function can receive ``add_one`` from Python, inspect or rewrite
it, and hand the result back, with no conversion in between. This lets us
move a hot analysis or a large transformation to C++ while keeping the rest
of the workflow in Python.

This chapter assumes the concepts of the earlier chapters and shows their
C++ form:

- **Accessing and creating IR nodes**: read fields, check node types, and
  build new nodes.
- **Structural utilities**: walk, map, and compare programs, with callbacks
  selected by their parameter types.
- **Visitor and mutator classes**: C++ classes with one hook per node kind,
  for passes that customize many kinds of node.

.. note::

   The runnable code in this chapter is under
   `examples/ir_guides <https://github.com/apache/tvm-tirx/tree/main/examples/ir_guides>`_
   in the repository.

Setting Up
----------

Throughout this chapter, we use the ``add_one`` function from Quick Start:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

The examples are packaged as ``tirx_ir_guide_examples`` under
`examples/ir_guides <https://github.com/apache/tvm-tirx/tree/main/examples/ir_guides>`__.
Installing the package builds the C++ source, ``cpp/working_with_ir.cc``,
into a library; ``CMakeLists.txt`` and ``pyproject.toml`` in the same
directory show how. Install TIRx from the same source tree to keep the IR
APIs aligned. We install without build isolation to build against that
installation instead of fetching one from PyPI:

.. code-block:: bash

   pip install scikit-build-core
   pip install --no-build-isolation -e examples/ir_guides

Each example is a C++ function that takes ``add_one`` as its argument. A
``TVM_FFI_STATIC_INIT_BLOCK`` at the end of the source registers each one as
a global function under ``tirx_ir_guide_examples.cpp``:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-at: refl::GlobalDef()
   :end-at: .def("tirx_ir_guide_examples.cpp.inspect", Inspect)
   :dedent: 2

The package's ``tirx_ir_guide_examples.cpp`` module loads the library and
exposes these functions, so Python calls them by name:

.. code-block:: python

   import tirx_ir_guide_examples.cpp

   tirx_ir_guide_examples.cpp.inspect(add_one)

The driver ``working_with_ir_in_cpp.py`` calls each example this way and
prints the result.

Accessing and Creating IR Nodes
-------------------------------

Before reading any IR, it helps to know where the classes live. The
headers relevant to this chapter sit in five folders under ``tvm/``; see
:ref:`cpp-header-organization` for the full list:

.. list-table::
   :header-rows: 1
   :widths: 18 47 35

   * - Folder
     - Contents
     - Included by our examples
   * - ``tvm/ffi/``
     - From TVM-FFI: ``Object`` and ``ObjectRef``, containers, functions,
       and the structural utilities.
     - ``extra/structural_visit.h``, ``extra/structural_mutate.h``,
       ``extra/structural_equal.h``, ``extra/structural_hash.h``,
       ``reflection/registry.h``
   * - ``tvm/ir/``
     - Core IR nodes, such as ``Expr``, ``Var``, ``Stmt``, ``For``, and
       ``TensorStore``, and the expression visitors in ``expr_functor.h``.
     - (through ``tvm/tirx/function.h``)
   * - ``tvm/sym/``
     - Symbolic analysis, such as ``sym::Analyzer`` and
       ``sym::ConstraintContext``.
     - None
   * - ``tvm/tirx/``
     - Tensor programs: ``tirx::Function``, the statement visitors in
       ``stmt_functor.h``, and the constraint-context variants.
     - ``function.h``, ``stmt_functor.h``
   * - ``tvm/script/``
     - The IR builder and the TVMScript printer.
     - None

IR classes live in the ``tvm`` namespace, TIRx-specific ones such as
:cpp:class:`tvm::tirx::Function` in ``tvm::tirx``, and the TVM-FFI classes
in ``tvm::ffi``. The examples start with ``using namespace tvm;``, so these
appear as ``Var``, ``tirx::Function``, and
:cpp:func:`ffi::StructuralWalk <tvm::ffi::StructuralWalk>`.

**Each IR node has two C++ classes.** The node class, named with an
``Obj`` suffix such as :cpp:class:`tvm::ForObj`, holds the fields. The
reference class without the suffix, such as :cpp:class:`tvm::For`, is a
reference-counted handle to a node, and is what functions take and return.
The ``->`` operator on a reference reaches the node's fields.

Let's read ``add_one`` from its parameters down to the store in its loop:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [inspect.begin]
   :end-before: // [inspect.end]

The output is:

.. code-block:: text

   param A
   param B
   loop over i up to n
     store to B: A[i] + T.float32(1.0)

Fields have the types we know from Python, and writing an expression to a
stream prints it in TVMScript. The elements of a ``SeqStmt`` are general
``Stmt`` references, so we check the actual node type before reading a
loop or a store. |ObjectRef::as|_ and its relatives offer three forms:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Returns
   * - ``stmt.as<ForObj>()``
     - A pointer to the node, or ``nullptr`` if it is not a loop.
   * - ``stmt.as<For>()``
     - A ``std::optional<For>`` reference.
   * - ``stmt.as_or_throw<For>()``
     - The ``For`` reference, or raises an error.

.. note::

   A function's ``body`` is always a ``SeqStmt``. The loop variable is stored
   as a ``PrimVar``, which views a ``Var`` as a ``PrimExpr``, so we read its
   name through ``as<VarObj>()``.

To create a node, we call the constructor of its reference class, and
arithmetic operators on ``PrimExpr`` build new expressions. Let's build a
store that writes twice the value of the store in ``add_one``:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [create.begin]
   :end-before: // [create.end]

The output is:

.. code-block:: python

   n = T.dynamic("n", dtype="int32")
   B = T.Var("B", T.Tensor((n,), T.float32))
   i = T.dynamic("i", dtype="int32")
   A = T.Var("A", T.Tensor((n,), T.float32))
   B[i] = (A[i] + T.float32(1.0)) * T.float32(2.0)

The new store reuses the destination and indices of the old one, so the
printer declares them as free variables first. ``store->value * 2`` turns
the literal into a constant of the value's type, ``T.float32(2.0)``.
Constructing nodes creates new objects; ``add_one`` itself is not modified.

Structural Utilities in C++
---------------------------

The structural utilities from Python are available in C++ with the same
behavior: :cpp:func:`ffi::StructuralWalk <tvm::ffi::StructuralWalk>`,
:cpp:func:`ffi::StructuralMap <tvm::ffi::StructuralMap>`,
:cpp:class:`ffi::StructuralEqual <tvm::ffi::StructuralEqual>`, and
:cpp:class:`ffi::StructuralHash <tvm::ffi::StructuralHash>`. **What changes
is how callbacks are written**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Python
     - C++
   * - Nodes a callback receives
     - The type paired with it, such as ``(ir.TensorStore, f)``
     - The type of the callback's first parameter
   * - Walk callback result
     - ``None``, ``WalkResult.SKIP``, or a ``VisitInterrupt``
     - ``ffi::Expected<WalkResult>``
   * - Traversal order
     - ``order="pre"`` or the default post-order
     - A required template argument, ``WalkOrder::kPreOrder`` or
       ``kPostOrder``
   * - Keep a node in a map
     - Return the input node
     - Return ``ffi::Unchanged()`` or the node
   * - Result of a map
     - The new value
     - An ``ffi::Any``, cast back with ``cast<T>()``
   * - Definition region kind
     - Pass the callback through ``with_def_region_kind``
     - A trailing ``TVMFFIDefRegionKind`` parameter

Structural Walk
~~~~~~~~~~~~~~~

As in Python, a walk calls our callback on each node of the type it asks
for. To find the tensors that ``add_one`` writes, the callback takes a
``const TensorStore&``:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [walk.begin]
   :end-before: // [walk.end]

The output is:

.. code-block:: text

   ['B']

Only the store reaches the callback, because its parameter is
``const TensorStore&``.

A walk callback returns an
``ffi::Expected<``\ :cpp:class:`ffi::WalkResult <tvm::ffi::WalkResult>`\ ``>``.
``WalkResult::Advance()`` continues the walk, ``WalkResult::Skip()`` skips
the node's children in pre-order, and ``WalkResult::Interrupt(...)`` stops
the walk.

To handle several node kinds in one walk, we pass one callback per kind as
extra arguments. They are tried in
declaration order, and only the first one whose parameter type matches a
node runs; a node that matches none is still traversed but not reported.
This walk records the loops and stores in pre-order:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [dispatch.begin]
   :end-before: // [dispatch.end]

The output is:

.. code-block:: text

   ['loop i', 'store B']

The loop and the store each reach their own callback, and pre-order
reports the loop before the store inside it. Matching follows the class
hierarchy, so a callback that takes a base class, such as
``const Stmt&``, would also take loops and stores that no earlier callback
claims.

To stop at the first match and return it, a callback wraps the node in an
:cpp:class:`ffi::VisitInterrupt <tvm::ffi::VisitInterrupt>`.
``StructuralWalk`` returns that interrupt, or ``std::nullopt`` if the walk
finished without one:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [stop.begin]
   :end-before: // [stop.end]

The output is:

.. code-block:: text

   B

The interrupt's ``value`` field carries the payload as an ``ffi::Any``, and
``cast<TensorStore>()`` converts it back to a store.

Definitions and Uses
~~~~~~~~~~~~~~~~~~~~

A callback receives the definition region kind of each site when it takes
a ``TVMFFIDefRegionKind`` as its last parameter. The kinds match Python's:
``kTVMFFIDefRegionKindNone`` for a use, ``kTVMFFIDefRegionKindSimple`` for
``DEF_SIMPLE``, and ``kTVMFFIDefRegionKindPattern`` for ``DEF_PATTERN``. Let's list the definition sites in
``add_one``:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [definitions.begin]
   :end-before: // [definitions.end]

The output is:

.. code-block:: text

   A pattern
   n pattern
   B pattern
   n pattern
   i simple

These are the same sites as in Python: the parameters define ``A``, ``B``,
and ``n`` by pattern, and the loop defines ``i``.

Structural Map
~~~~~~~~~~~~~~

``ffi::StructuralMap`` takes the same callbacks, but each one returns a
replacement for its node. Let's specialize ``add_one`` by replacing ``n``
with 128, passing ``n`` in from Python:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [specialize.begin]
   :end-before: // [specialize.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
       for i in range(128):
           B[i] = A[i] + T.float32(1.0)

For every other variable, the callback returns
:cpp:struct:`ffi::Unchanged <tvm::ffi::Unchanged>`\ ``()``, which keeps the
node as it is. The constant takes its type from the variable, ``var->ty``.
As in Python, the mapper keeps definitions and uses consistent, so the new
``A`` and ``B`` and every use of them carry the specialized shape.

Equality and Hashing
~~~~~~~~~~~~~~~~~~~~

``ffi::StructuralEqual::Equal`` and ``ffi::StructuralHash::Hash`` compare
and hash by structure. Let's compare two ``add_one`` functions built
separately in Python by ``make_add(1)``, the helper from the structural
utilities chapter:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [equal.begin]
   :end-before: // [equal.end]

The output is:

.. code-block:: text

   [False, True, True]

The functions are different objects, yet structurally equal with equal
hashes. ``Equal`` and ``Hash`` also take an optional ``map_free_vars``
argument, which defaults to ``false`` as in Python.

Visit and Mutate
~~~~~~~~~~~~~~~~

When a callback must decide which children to visit, we use
:cpp:func:`ffi::StructuralVisit <tvm::ffi::StructuralVisit>` and
:cpp:func:`ffi::StructuralMutate <tvm::ffi::StructuralMutate>`, as in
Python. Their callbacks also receive
the visitor or mutator, such as
:cpp:class:`ffi::StructuralVisitorObj* <tvm::ffi::StructuralVisitorObj>`,
and descend into a node's children with ``DefaultVisitExpected`` or
``DefaultMutateExpected``.

Visitor and Mutator Classes
---------------------------

Walk and map cover most tasks. A pass that handles many kinds of node,
however, reads better as a class with one method per kind:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Tool
     - Covers
     - Use it for
   * - Structural utilities
     - Any TVM-FFI object
     - A few node kinds, written as callbacks
   * - :cpp:class:`tirx::StmtExprVisitor <tvm::tirx::StmtExprVisitor>`,
       :cpp:class:`tirx::StmtExprMutator <tvm::tirx::StmtExprMutator>`
     - TIRx statements and expressions (``tvm/tirx/stmt_functor.h``)
     - Passes that customize many node kinds
   * - :cpp:class:`ExprVisitor <tvm::ExprVisitor>`,
       :cpp:class:`ExprMutator <tvm::ExprMutator>`
     - Expressions alone (``tvm/ir/expr_functor.h``)
     - Expression analyses and rewrites
   * - :cpp:class:`ExprFunctor <tvm::ExprFunctor>`,
       :cpp:class:`tirx::StmtFunctor <tvm::tirx::StmtFunctor>`
     - Dispatch on the node kind with a signature we choose, such as
       ``int(const Expr&)``, without traversing children
     - Computations where each hook decides which children to recurse into

Each class declares a virtual hook for every node kind, named ``Visit_``
for a visitor and ``Mutate_`` for a mutator, and a subclass overrides only
the kinds it cares about.

**These classes are built on the structural utilities.** A node kind
without a hook, such as a function, falls back to its structural traversal,
which hands each child back to our hooks.

.. note::

   ``ExprVisitor`` ultimately derives from
   :cpp:class:`ffi::StructuralVisitorObj <tvm::ffi::StructuralVisitorObj>`,
   and ``ExprMutator`` from the mutator base behind ``StructuralMap``. Any
   hook can read the definition region kind with ``def_region_kind()``.

Let's record the loops around each write, as the structural visit example
in Python did:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [visit.begin]
   :end-before: // [visit.end]

The output is:

.. code-block:: text

   ['B in i']

Our ``For`` hook pushes the loop variable, calls the base class hook to
visit the loop's bounds and body, then pops the variable. Calling the base
hook plays the role of ``default_visit`` in Python, and returning its
result passes an interrupt from a child up to the caller. Visitors are
objects, so we allocate them with ``ffi::make_object``.

.. note::

   The ``using tirx::StmtExprVisitor::Visit_;`` line keeps the hooks we did
   not override visible, as usual for overloaded virtual functions in C++.

A mutator works the same way, with hooks that return the replacement node.
Let's reverse the loop by replacing each use of ``i`` with ``n - 1 - i``,
built with arithmetic operators as before:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [mutate.begin]
   :end-before: // [mutate.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[n - 1 - i] = A[n - 1 - i] + T.float32(1.0)

A mutator hook returns an
:cpp:class:`UnchangedOr <tvm::ffi::UnchangedOr>`\ ``<T>``: either a
replacement node or ``ffi::Unchanged()``, which keeps the original and lets
the parents above it stay unchanged too. Our ``For`` hook records the
replacement before calling the base class hook, so it is ready before any
use of ``i`` is reached.

.. note::

   ``StmtExprMutator``'s ``For`` hook rewrites the loop's bounds and body but
   not the loop variable itself, so our ``Var`` hook sees only the uses of
   ``i``, and the loop keeps its variable. The ``InplaceMode`` argument
   tells a hook whether it may update a node that no one else references in
   place; our hooks forward it to the base class unchanged.

Constraint Contexts
~~~~~~~~~~~~~~~~~~~

TIRx adds variants of the structural utilities and the visitor and mutator
classes that also pass the facts valid at each node, such as loop ranges
and branch conditions:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Utility
     - Works like
     - Facts reach the callback as
   * - :cpp:func:`tirx::StructuralWalkWithConstraintContext <tvm::tirx::StructuralWalkWithConstraintContext>`
     - ``ffi::StructuralWalk``
     - A ``const sym::ConstraintContext&`` after the node
   * - :cpp:func:`tirx::StructuralMapWithConstraintContext <tvm::tirx::StructuralMapWithConstraintContext>`
     - ``ffi::StructuralMap``
     - A ``const sym::ConstraintContext&`` after the node
   * - :cpp:func:`tirx::StructuralVisitWithConstraintContext <tvm::tirx::StructuralVisitWithConstraintContext>`
     - ``ffi::StructuralVisit``
     - The ``constraints`` member of the visitor the callback receives
   * - :cpp:func:`tirx::StructuralMutateWithConstraintContext <tvm::tirx::StructuralMutateWithConstraintContext>`
     - ``ffi::StructuralMutate``
     - The ``constraints`` member of the mutator the callback receives
   * - :cpp:class:`tirx::StmtExprVisitorWithConstraintContext <tvm::tirx::StmtExprVisitorWithConstraintContext>`,
       :cpp:class:`tirx::StmtExprMutatorWithConstraintContext <tvm::tirx::StmtExprMutatorWithConstraintContext>`
     - ``tirx::StmtExprVisitor``, ``tirx::StmtExprMutator``
     - The inherited ``constraints`` member

The functions are in ``tvm/tirx/structural_visit_with_constraint_context.h``
and ``tvm/tirx/structural_mutate_with_constraint_context.h``, and the classes
in ``tvm/tirx/stmt_functor_with_constraint_context.h``.

These tools follow the same rules as in Python: IR is immutable, rewrites
return new programs that share unchanged parts, and every utility respects
definitions and uses. Because C++ and Python share the same objects, we can
prototype a pass in Python and move parts of it to C++ as needed.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - C++
     - What it does
     - Python equivalent
   * - ``loop->var``, ``func->params``
     - Reads a node's fields
     - ``loop.var``, ``func.params``
   * - ``stmt.as<ForObj>()``, ``stmt.as<For>()``,
       ``stmt.as_or_throw<For>()``
     - Checks the node type and downcasts
     - ``isinstance(stmt, ir.For)``
   * - ``TensorStore(dest, indices, value)``, ``store->value * 2``
     - Creates a node; builds an expression with operators
     - ``ir.TensorStore(dest, indices, value)``, ``store.value * 2``
   * - :cpp:func:`ffi::StructuralWalk <tvm::ffi::StructuralWalk>`\ ``<ffi::WalkOrder::kPreOrder>(func, ...)``
     - Walks with typed callbacks that return
       ``ffi::Expected<ffi::WalkResult>``
     - ``tvm_ffi.structural_walk(func, callbacks, order="pre")``
   * - :cpp:func:`ffi::StructuralMap <tvm::ffi::StructuralMap>`\ ``<order>(func, ...)``,
       ``ffi::Unchanged()``
     - Rebuilds with replacements and returns an ``ffi::Any``
     - ``tvm_ffi.structural_map(func, callbacks)``
   * - A trailing ``TVMFFIDefRegionKind`` parameter
     - Receives the definition region kind
     - ``with_def_region_kind=``
   * - :cpp:func:`ffi::StructuralEqual::Equal <tvm::ffi::StructuralEqual::Equal>`,
       :cpp:func:`ffi::StructuralHash::Hash <tvm::ffi::StructuralHash::Hash>`
     - Compares and hashes by structure
     - ``tvm_ffi.structural_equal``, ``tvm_ffi.structural_hash``
   * - :cpp:class:`tirx::StmtExprVisitor <tvm::tirx::StmtExprVisitor>`,
       :cpp:class:`tirx::StmtExprMutator <tvm::tirx::StmtExprMutator>`,
       :cpp:class:`ExprVisitor <tvm::ExprVisitor>`,
       :cpp:class:`ExprMutator <tvm::ExprMutator>`
     - One ``Visit_`` or ``Mutate_`` hook per node kind
     - ``tvm_ffi.structural_visit``, ``tvm_ffi.structural_mutate``
   * - ``TVM_FFI_STATIC_INIT_BLOCK``, ``refl::GlobalDef().def(...)``
     - Registers C++ functions as global functions
     - ``tvm_ffi.register_global_func``

.. TVM-FFI C++ API page for ObjectRef, which the C++ reference does not include.

.. |ObjectRef::as| replace:: ``ObjectRef::as``
.. _`ObjectRef::as`: https://tvm.apache.org/ffi/reference/cpp/generated/classtvm_1_1ffi_1_1ObjectRef.html
