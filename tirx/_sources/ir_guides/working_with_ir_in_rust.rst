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

Working with IR in Rust
=======================

The IR objects we have used so far in Python are TVM-FFI objects, and the
same objects are available in Rust through the
`tvm-tirx <https://crates.io/crates/tvm-tirx>`_ crate. A Rust function can
receive a function written in TVMScript, read and rewrite it, and return
the result to Python. Nothing is converted or copied along the way: Python
and Rust hold references to the same native objects.

This chapter assumes the concepts of the earlier chapters and shows their
Rust form:

- **Accessing and creating IR nodes**: read fields, branch on node types,
  and build new nodes.
- **Structural utilities**: walk, map, and compare programs, with
  callbacks selected by their Rust parameter types.
- **Visitors and mutators**: keep state across callbacks and control the
  traversal, with ``#[dispatch]`` structs and ``VisitCallbacks``.

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
When cargo is available, installing the package builds the Rust crate in
``rust/`` into a library; ``CMakeLists.txt`` and ``pyproject.toml`` in the
same directory show how. The crate uses the bindings from this checkout,
so install TIRx from the same source tree. We install without build isolation
to build against that installation instead of fetching one from PyPI:

.. code-block:: bash

   pip install scikit-build-core
   pip install --no-build-isolation -e examples/ir_guides

Each example is a Rust function that takes ``add_one`` as its argument. A
|static_init_block!|_ at the end of the crate registers each one as a
global function under ``tirx_ir_guide_examples.rust``:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-at: tvm_ffi::Function::register_global_typed(
   :end-at: )?;
   :dedent: 4

The package's ``tirx_ir_guide_examples.rust`` module loads the library and
exposes these functions, so Python calls them by name:

.. code-block:: python

   import tirx_ir_guide_examples.rust

   tirx_ir_guide_examples.rust.inspect(add_one)

The driver ``working_with_ir_in_rust.py`` calls each example this way and
prints the result.

Accessing and Creating IR Nodes
-------------------------------

**Every IR node has two Rust types**: a handle such as |For|_, a
reference-counted pointer that is cheap to clone, and the object it points
to, such as |ForObj|_, which holds the fields. A handle dereferences to its
object, so we read fields as ordinary struct fields.

Let's read ``add_one`` from its parameters down to the store in its loop.
The statements of a body are general ``Stmt`` handles, and |match_any!|_
branches on their actual node type, as ``match`` does on an enum:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [inspect.begin]
   :end-before: // [inspect.end]

The output is:

.. code-block:: text

   param A
   param B
   loop over i up to n
     store to B: A[i] + T.float32(1.0)

Each arm names a handle type and binds the matching node to it; the arms
are tried in order, and ``_`` takes everything else. Fields have the types
we know from Python: ``params`` is an ``Array<Var>``, and the body of a
loop is a ``SeqStmt`` whose ``seq`` holds its statements. |script|_ prints
any node as TVMScript.

.. note::

   A function's ``body`` is always a ``SeqStmt``. ``get`` on an array returns
   an error for an index out of range, and ``?`` passes it on.

To create a node, we call the constructor of its handle type, such as
|TensorStore::new|_. |ex|_ wraps an expression so that the usual Rust
operators build IR; constructors accept the result directly. Let's build a store
that writes twice the value of the store in ``add_one``:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
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
printer declares them as free variables first. Constructing nodes creates
new objects; ``add_one`` itself is not modified. Here |try_cast|_ downcasts
to a single known type instead of branching.

Structural Utilities
--------------------

The structural utilities are part of TVM-FFI, and the ``tvm_ffi`` crate,
re-exported as ``tvm_tirx::tvm_ffi``, provides them in Rust:
|structural_walk|_, |structural_map|_, |structural_visit|_, and
|structural_mutate|_. They run on the same structural hooks as in Python,
so they reach the same nodes and keep definitions and uses consistent in
the same way. **What changes is the Rust form of each piece**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Python
     - Rust
   * - Nodes a callback receives
     - The type paired with it, such as ``(ir.TensorStore, f)``
     - The type of the callback's parameter
   * - Walk callback result
     - ``None``, ``WalkResult.SKIP``, or a ``VisitInterrupt``
     - ``WalkResult``, or ``Result<WalkResult>`` to use ``?``
   * - Traversal order
     - ``order="pre"`` or the default post-order
     - A required argument, ``WalkOrder::PreOrder`` or ``PostOrder``
   * - Result of a map
     - The new value
     - An ``Any``, converted back with ``try_into``
   * - Definition region kind
     - Pass the callback through ``with_def_region_kind``
     - A trailing ``DefRegionKind`` parameter
   * - Errors
     - Exceptions
     - ``Result`` values
   * - Structural equality
     - ``structural_equal``
     - The global function ``ffi.StructuralEqual``

Structural Walk
~~~~~~~~~~~~~~~

As in Python, a walk calls our callback on each node of the type it asks
for. Let's find the tensors that ``add_one`` writes:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [walk.begin]
   :end-before: // [walk.end]

The output is:

.. code-block:: text

   ['B']

``on_store`` takes a ``TensorStore``, so the walk calls it only on
stores. ``Strings`` is our alias for ``Array<tvm_ffi::String>``, which
Python receives as a list of strings.

|WalkResult|_\ ``::Advance`` continues into the node's children, ``Skip``
does not descend into them, and ``Interrupt`` stops the whole walk.

The parameter list of a callback decides what it receives:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Callback
     - Receives
   * - ``|store: TensorStore|``
     - Each write, as a handle.
   * - ``|store: &TensorStoreObj|``
     - Each write, as a borrowed object, without changing its reference
       count.
   * - ``|value: &StructuralView|``
     - Every value the walk reaches.
   * - ``|var: Var, kind: DefRegionKind|``
     - Each variable, with its definition region kind.
   * - ``(on_loop, on_store)``
     - A tuple of callbacks, tried in order; only the first one whose type
       matches a node runs. Tuples hold up to 12 callbacks and can be
       nested.

``WalkResult::interrupt_with`` stops the walk and returns a value, which
lets a walk search for a node:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [stop.begin]
   :end-before: // [stop.end]

Python prints the ``dest.name`` of the store it receives. The output is:

.. code-block:: text

   B

``structural_walk`` returns ``Ok(Some(interrupt))`` when a callback stops
it, with the payload in ``interrupt.value``, and ``Ok(None)`` otherwise.

Definitions and Uses
~~~~~~~~~~~~~~~~~~~~

A callback receives the definition region kind when it takes a
|DefRegionKind|_ as its last parameter. This walk lists the definition
sites of ``add_one``:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [definitions.begin]
   :end-before: // [definitions.end]

The output is:

.. code-block:: text

   A Pattern
   n Pattern
   B Pattern
   n Pattern
   i Simple

The kinds are the ones we saw in Python, under Rust names:
``DefRegionKind::None`` for a use site, ``Pattern`` for ``DEF_PATTERN``,
and ``Simple`` for ``DEF_SIMPLE``. ``structural_map`` callbacks take the
same trailing parameter.

Structural Map
~~~~~~~~~~~~~~

A map callback returns the replacement for its node. Let's specialize
``add_one`` for ``n = 128``, with ``n`` passed in from Python:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
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

This is the same result as in Python. |make_const|_ builds the constant
``128`` with the type of ``n``, and every other variable is returned as it
is. ``structural_map`` takes ownership of the handle we pass and returns
the new root as an ``Any``; Python still holds ``add_one``, which is
unchanged.

Structural Equality
~~~~~~~~~~~~~~~~~~~

The crate has no Rust function for structural equality yet, so we call the
global function that Python's ``structural_equal`` uses,
``ffi.StructuralEqual``, with its four arguments: the two values,
``map_free_vars``, and ``skip_tensor_content``. Let's specialize
``add_one`` twice and compare the results:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [equal.begin]
   :end-before: // [equal.end]

The output is:

.. code-block:: text

   [False, True]

The two functions are different objects, so ``same_as`` is false, but they
are structurally equal.

Visitors and Mutators in Rust
-----------------------------

Rust has no separate layer of visitor and mutator classes. Stateful and
typed traversals use the same four structural utilities, and the crate
provides two conveniences to build them: |dispatch|_ turns the methods
of a struct into typed callbacks, and |VisitCallbacks|_ and
|MutateCallbacks|_ attach state to a tuple of callbacks.

Typed Dispatch on a Struct
~~~~~~~~~~~~~~~~~~~~~~~~~~

The closures in a callback tuple cannot borrow the same state mutably at
the same time, so callbacks that share state are written as methods
instead. ``#[dispatch(walk)]`` on an ``impl`` block turns its ``walk_*``
methods into callbacks: each method receives the nodes of its parameter
type, and the methods are tried in source order. The struct holds the
state. Let's record the loops and writes of ``add_one`` in the order the
walk reaches them:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [dispatch_walk.begin]
   :end-before: // [dispatch_walk.end]

Called with ``pre_order`` true and then false, the output is:

.. code-block:: text

   ['loop i', 'write B']
   ['write B', 'loop i']

Pre-order reports the loop before the write inside it, and post-order the
reverse. A method may also take a trailing ``DefRegionKind``, as a closure
can. ``#[dispatch(map)]``, ``#[dispatch(visit)]``, and
``#[dispatch(mutate)]`` work the same way for ``map_*``, ``visit_*``, and
``mutate_*`` methods.

Visit and Mutate
~~~~~~~~~~~~~~~~

A visit or mutate callback takes control of its node's children, as in
Python. Let's record the loops around each write. With ``VisitCallbacks``,
each callback receives the node and a ``VisitContext``, which holds the
shared state:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [visit.begin]
   :end-before: // [visit.end]

The output is:

.. code-block:: text

   ['B inside [i]']

``visit_for`` pushes the loop variable, lets ``visit_children`` visit the
loop's children with their usual traversal, then pops the variable. It
returns the result of ``visit_children``, so that an interrupt from a child
reaches the caller. The context also provides ``visit`` for a selected
child and ``def_region_kind`` for the current node.

For the mutation we use the struct form instead; either form works for
both visits and mutations. A ``#[dispatch(mutate)]`` method that takes a
trailing ``&mut Mutator`` gets the mutation counterpart of
``VisitContext``, with ``default_mutate`` for the children and
``def_region_kind()``. Let's reverse the loop
by replacing each use of ``i`` with ``n - 1 - i``, built with ``ex`` as
before:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
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

``mutate_for`` records the replacement, then ``default_mutate`` rewrites
the loop's children through the same struct, so ``mutate_var`` sees the
replacement before any use of ``i``.

.. note::

   ``mutate_var`` replaces ``i`` only where ``def_region_kind()`` is
   ``None``, so the loop's own ``i`` stays a variable. |ObjectIdentity|_
   keys the map by object identity, as the Python version keys its
   dictionary by variable.

.. note::

   TIRx also has traversals that track the facts valid at each node, such
   as loop ranges and branch conditions. In Rust they are
   |StructuralVisitorWithConstraintContext|_ and
   |StructuralMutatorWithConstraintContext|_; ``constraints()`` returns the
   current facts.

With fields, typed callbacks, and the structural utilities, a Rust function
works with the same IR that Python builds and prints, so an analysis or a
transformation can move between the two languages one function at a time.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Rust
     - What it does
     - Python equivalent
   * - |match_any!|_
     - Branches on the node type; the first matching arm runs
     - ``isinstance`` checks
   * - |try_cast|_, |as_node|_
     - Downcasts to one type, as a ``Result`` handle or a borrowed
       ``Option``
     - ``isinstance(stmt, ir.For)``
   * - |TensorStore::new|_, |ex|_\ ``(...)``, ``.expr()``
     - Creates a node; builds an expression with operators
     - ``ir.TensorStore(dest, indices, value)``, ``store.value * 2``
   * - |structural_walk|_\ ``(&func, callbacks, WalkOrder::PreOrder)``
     - Walks with typed callbacks that return ``WalkResult``
     - ``tvm_ffi.structural_walk(func, callbacks, order="pre")``
   * - |structural_map|_\ ``(func, callbacks, order)``
     - Rebuilds with replacements and returns an ``Any``
     - ``tvm_ffi.structural_map(func, callbacks)``
   * - A trailing |DefRegionKind|_ parameter
     - Receives the definition region kind
     - ``with_def_region_kind=``
   * - ``Function::get_global("ffi.StructuralEqual")``
     - Compares by structure
     - ``tvm_ffi.structural_equal``
   * - |dispatch|_\ ``(walk)``, ``(map)``, ``(visit)``, ``(mutate)``
     - Turns the methods of a struct into typed callbacks
     - A list of ``(type, callback)`` pairs
   * - |VisitCallbacks|_, |MutateCallbacks|_
     - Attach state to a tuple of visit or mutate callbacks
     - ``tvm_ffi.structural_visit``, ``tvm_ffi.structural_mutate``
   * - |static_init_block!|_, |register_global_typed|_
     - Registers Rust functions as global functions
     - ``tvm_ffi.register_global_func``

.. API pages of the tvm-tirx and tvm-ffi crates for the names above.

.. |static_init_block!| replace:: ``static_init_block!``
.. _static_init_block!: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/macro.static_init_block.html
.. |For| replace:: ``For``
.. _For: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/ir/stmt/struct.For.html
.. |ForObj| replace:: ``ForObj``
.. _ForObj: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/ir/stmt/struct.ForObj.html
.. |script| replace:: ``script``
.. _script: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/script/fn.script.html
.. |try_cast| replace:: ``try_cast``
.. _try_cast: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/object/trait.ObjectRefCast.html
.. |structural_walk| replace:: ``structural_walk``
.. _structural_walk: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_visit/fn.structural_walk.html
.. |structural_map| replace:: ``structural_map``
.. _structural_map: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_mutate/fn.structural_map.html
.. |structural_visit| replace:: ``structural_visit``
.. _structural_visit: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_visit/fn.structural_visit.html
.. |structural_mutate| replace:: ``structural_mutate``
.. _structural_mutate: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_mutate/fn.structural_mutate.html
.. |WalkResult| replace:: ``WalkResult``
.. _WalkResult: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_visit/enum.WalkResult.html
.. |DefRegionKind| replace:: ``DefRegionKind``
.. _DefRegionKind: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_visit/enum.DefRegionKind.html
.. |make_const| replace:: ``make_const``
.. _make_const: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/ir/prim/op/fn.make_const.html
.. |dispatch| replace:: ``#[dispatch]``
.. _dispatch: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/attr.dispatch.html
.. |VisitCallbacks| replace:: ``VisitCallbacks``
.. _VisitCallbacks: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_visit/struct.VisitCallbacks.html
.. |MutateCallbacks| replace:: ``MutateCallbacks``
.. _MutateCallbacks: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/extra/structural_mutate/struct.MutateCallbacks.html
.. |ex| replace:: ``ex``
.. _ex: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/ir/prim/op/fn.ex.html
.. |ObjectIdentity| replace:: ``ObjectIdentity``
.. _ObjectIdentity: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/object/struct.ObjectIdentity.html
.. |StructuralVisitorWithConstraintContext| replace:: ``StructuralVisitorWithConstraintContext``
.. _StructuralVisitorWithConstraintContext: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/tirx/struct.StructuralVisitorWithConstraintContext.html
.. |StructuralMutatorWithConstraintContext| replace:: ``StructuralMutatorWithConstraintContext``
.. _StructuralMutatorWithConstraintContext: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/tirx/struct.StructuralMutatorWithConstraintContext.html
.. |match_any!| replace:: ``match_any!``
.. _match_any!: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/macro.match_any.html
.. |TensorStore::new| replace:: ``TensorStore::new``
.. _TensorStore::new: https://docs.rs/tvm-tirx/0.1.0-alpha.5/tvm_tirx/ir/stmt/struct.TensorStore.html#method.new
.. |as_node| replace:: ``as_node``
.. _as_node: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/object/trait.ObjectRefCore.html#method.as_node
.. |register_global_typed| replace:: ``Function::register_global_typed``
.. _register_global_typed: https://docs.rs/tvm-ffi/0.1.15-post2/tvm_ffi/function/struct.Function.html#method.register_global_typed
