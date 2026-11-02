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

The `tvm-tirx <https://crates.io/crates/tvm-tirx>`_ crate lets a Rust
function read, build, and rewrite TIRx IR. The function can receive a
program written in TVMScript and return the result, and nothing is
converted or copied along the way. This guide covers:

- **TVM-FFI object system**: how IR nodes are represented and used in Rust.
- **Structural utilities**: walk, map, and compare programs.
- **Visitors and mutators**: keep state and control the traversal.

.. note::

   The runnable code in this guide is under
   `examples/ir_guides <https://github.com/apache/tvm-tirx/tree/main/examples/ir_guides>`_
   in the repository.

Setting Up
----------

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

A |static_init_block!|_ at the end of the crate registers each example as
a global function under ``tirx_ir_guide_examples.rust``:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-at: tvm_ffi::Function::register_global_typed(
   :end-at: )?;
   :dedent: 4

The package's ``tirx_ir_guide_examples.rust`` module loads the library and
exposes these functions, so Python calls them by name. The driver
``working_with_ir_in_rust.py`` calls each example this way and prints the
result.

The object system examples take no arguments: each builds its own IR in
Rust. The later sections receive the ``add_one`` function from Quick
Start, written in TVMScript:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

.. code-block:: python

   import tirx_ir_guide_examples.rust

   tirx_ir_guide_examples.rust.construct()
   tirx_ir_guide_examples.rust.written_tensors(add_one)

TVM-FFI Object System
---------------------

Every TIRx IR node is a TVM-FFI object. Rust IRs are typically built as
enums and matched with ``match``, which works well when every node type is
known in advance. TIRx is designed for **dynamic extension** instead: a
library can add a new node type, for example a new statement for a
hardware feature, and existing code keeps working on programs that contain
it. Generic utilities such as walking, rewriting, comparing, and printing
work on every node through its registered fields.

TVM-FFI objects also work across languages. Python, C++, and Rust share
the same native, immutable, reference-counted objects, so a program moves
between them without conversion.

**We designed the TVM-FFI object system to be ergonomic in Rust.** The
key constructs are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Construct
     - Purpose
   * - Constructors, ``ex()``
     - Build expressions with ordinary Rust operators
   * - ``FooObj`` / ``Foo``
     - Underlying object and its named managed reference: read fields
       directly, clone cheaply
   * - ``.into()``
     - Convert to a parent type
   * - |try_cast|_
     - Convert to a child type, with a checked ``Result``
   * - |match_any!|_
     - Branch on the node type
   * - |dispatch|_
     - Process each node type in its own method, with shared state

The subsections below introduce each construct.

Constructing Objects
~~~~~~~~~~~~~~~~~~~~

**We build IR with constructors and ordinary Rust operators.** Writing
``x * 2 + 1`` as nested constructor calls would bury the expression, so
the crate lets us write it as we would in Rust. Let's build it:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [construct.begin]
   :end-before: // [construct.end]

The output is:

.. code-block:: text

   x * 2 + 1

|Var::new|_ takes a name, a type, and a source location; ``UNKNOWN_LOC``
means there is none. |ex|_ wraps ``x`` so that ``*`` and ``+`` build IR,
and the literals ``2`` and ``1`` become constants of the type of ``x``.

An operator can fail, for example on mismatched types, so ``ex`` carries
the first error along, much like a chain of ``Result``\ s. ``.expr()``
takes the result out and ``?`` passes an error on. |script|_ prints the
node as TVMScript.

We now have an expression; next, let's see what kind of Rust value it is.

Objects and Managed References
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Every node type comes as a pair: an underlying object and a named
managed reference.** The top of
``x * 2 + 1`` is an addition, so let's convert the expression to ``Add``
with ``try_cast``, which the next subsection explains, and read its two
operands:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [objects.begin]
   :end-before: // [objects.end]

The output is:

.. code-block:: text

   a = x * 2
   b = 1
   same object: true

``alias`` is the same object as ``add``: ``clone`` copies only the managed
reference, and ``same_as`` compares identity, like ``Arc::ptr_eq``.

|AddObj|_ is the **underlying object**: the struct that holds the fields. |Add|_ is a **named managed reference**
to it: an owning, reference-counted smart pointer, like ``Arc<AddObj>``,
so cloning it is cheap. Functions take and return managed references, and
a managed reference dereferences to its object, so fields read as ordinary
struct fields: ``add.a``.

We define a separate named type ``Add`` instead of using a plain pointer
to ``AddObj`` because the named type carries the interface: constructors,
methods, conversions to parent types, and the traits that let it pass
through TVM-FFI functions. It can also impose additional constraints on
the underlying object that the object alone does not express.

This is the newtype pattern. Simplified from the crate, the pair is:

.. code-block:: rust

   #[repr(C)]
   #[derive(Object)]
   #[type_key = "prim.Add"]
   pub struct AddObj {
       base: ExprObj,
       pub a: Expr,
       pub b: Expr,
   }

   #[repr(C)]
   #[derive(ObjectRef, Clone)]
   pub struct Add {
       base: ObjectArc<AddObj>,
   }

Rust has no inheritance; instead, ``#[repr(C)]`` puts the parent's object,
``ExprObj``, first in ``AddObj``, so every addition is also an expression.
``Add`` implements ``Deref`` to ``AddObj``, which is why ``add.a`` works.

**Nodes are immutable.** As with data behind an ``Arc``, a managed
reference gives read access only; to change a node, we build a new one.

We reached ``Add`` with a cast; next, let's look at casts on their own.

Casts
~~~~~

**Casts move a managed reference up and down the type hierarchy.** A
function that takes any expression wants an ``Expr``, while reading the
operands needs an ``Add``. ``.into()`` converts to a parent type through
``From``/``Into`` and cannot fail. ``.try_cast()`` converts to a child
type: like ``TryFrom``, it checks the node's actual type and returns a
``Result``.

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [casts.begin]
   :end-before: // [casts.end]

The output is:

.. code-block:: text

   back to Add: x * 2 + 1
   as Mul: Cannot convert from type `prim.Add` to `prim.Mul`

``value`` is an ``Expr`` again, but the node behind it is still an
addition, so the cast back to ``Add`` succeeds.

What if we guess wrong? The cast to ``Mul`` returns an ``Err``, and its
message names both type keys: the actual type ``prim.Add`` and the
requested ``prim.Mul``.

.. note::

   ``try_cast`` consumes the managed reference it converts, so we clone
   ``value`` before the first cast; the clone costs one reference-count
   increment.

Casting to one type at a time gets tedious when a node can be one of
many; that is what matching is for.

Matching Node Types
~~~~~~~~~~~~~~~~~~~

|match_any!|_ **branches on the actual type of a node**, as ``match`` does
on an enum. Let's print an expression with explicit parentheses by
matching each node and recursing into its operands:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [describe.begin]
   :end-before: // [describe.end]

We call it on ``x * 2 + 1`` and on ``x - 1``:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [matching.begin]
   :end-before: // [matching.end]

The output is:

.. code-block:: text

   ((x * 2) + 1)
   <other>

Each arm names a managed reference type and binds the matching node to it,
and the first matching arm runs. The ``Add`` and ``Mul`` arms recurse into
the operands, the ``Var`` and ``IntImm`` arms end the recursion, and
``x - 1`` falls to the ``_`` arm because ``describe`` has no arm for
subtraction.

**The catch-all arm keeps existing code working as node types are added.**
The set of node types is open, like a ``#[non_exhaustive]`` enum from
another crate, so the match needs a ``_`` arm. A function handles the
types it knows, and ``_`` lets programs with new types pass through.

.. note::

   Like ``try_cast``, ``match_any!`` consumes what it matches, so
   ``describe`` passes it ``expr.clone()``. ``match_any!`` works on any
   TVM-FFI object, such as tensors, shapes, and typed arrays, and on
   ``Any`` values. Plain values such as integers take the ``_`` arm.

A match looks at one node at a time; to process every node in a program
while keeping state, we dispatch.

Dispatching on Node Types
~~~~~~~~~~~~~~~~~~~~~~~~~

|dispatch|_ **lets one struct process each node type in its own method,
with shared state.** A compiler pass often needs one piece of logic per
node type and state shared across them; ``#[dispatch]`` puts both on one
struct. It works like a trait whose methods all have default bodies: we
write methods for the node types we care about, and every other type gets
the default behavior.

``#[dispatch(walk)]`` on an ``impl`` block turns its ``walk_*`` methods
into callbacks for |structural_walk|_, which visits every node. Each
method borrows the nodes of its parameter type, and the methods are tried
in source order. Let's count the variables and additions:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [counter.begin]
   :end-before: // [counter.end]

We walk ``x * 2 + 1`` with it. ``WalkResult::Advance`` continues into
the node's children, and ``WalkOrder::PreOrder`` visits a node before its
children:

.. literalinclude:: ../../examples/ir_guides/rust/src/lib.rs
   :language: rust
   :start-after: // [count.begin]
   :end-before: // [count.end]

The output is:

.. code-block:: text

   variables: 1, additions: 1

**Node types without a method fall back to the default traversal**: the
walk continues into their children, so new types pass through. Here the
multiplication and the constants have no method, and the walk still
reaches ``x`` inside the multiplication.

Structural Utilities
--------------------

The structural utilities are part of TVM-FFI, and the ``tvm_ffi`` crate,
re-exported as ``tvm_tirx::tvm_ffi``, provides them in Rust:
|structural_walk|_, |structural_map|_, |structural_visit|_, and
|structural_mutate|_. They work on every node through its registered
fields and keep definitions and uses consistent. **Each piece takes a Rust
form**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Piece
     - Rust form
   * - Nodes a callback receives
     - Selected by the type of the callback's parameter
   * - Walk callback result
     - ``WalkResult``, or ``Result<WalkResult>`` to use ``?``
   * - Traversal order
     - A required argument, ``WalkOrder::PreOrder`` or ``PostOrder``
   * - Result of a map
     - An ``Any``, converted back with ``try_into``
   * - Definition region kind
     - A trailing ``DefRegionKind`` parameter
   * - Errors
     - ``Result`` values
   * - Structural equality
     - The global function ``ffi.StructuralEqual``

Structural Walk
~~~~~~~~~~~~~~~

A walk calls our callback on each node of the type it asks for. Let's find
the tensors that ``add_one`` writes:

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
     - Each write, as a managed reference.
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

``DefRegionKind::None`` marks a use site; ``Pattern`` and ``Simple`` are
the two kinds of definition site. ``structural_map`` callbacks take the
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

|make_const|_ builds the constant ``128`` with the type of ``n``, and
every other variable is returned as it is. ``structural_map`` takes
ownership of the managed reference we pass and returns the new root as an
``Any``; Python still holds ``add_one``, which is unchanged.

Structural Equality
~~~~~~~~~~~~~~~~~~~

The crate has no Rust function for structural equality yet, so we call the
global function ``ffi.StructuralEqual`` with its four arguments: the two values,
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

Stateful traversals use the same four structural utilities. The state
lives either on a |dispatch|_ struct, as in Dispatching on Node Types, or
in |VisitCallbacks|_ and |MutateCallbacks|_, which attach it to a tuple of
callbacks. ``#[dispatch(map)]``, ``#[dispatch(visit)]``, and
``#[dispatch(mutate)]`` work like ``#[dispatch(walk)]`` for ``map_*``,
``visit_*``, and ``mutate_*`` methods.

**A visit or mutate callback takes control of its node's children.**
Let's record the loops around each write. With ``VisitCallbacks``, each
callback receives the node and a ``VisitContext``, which holds the shared
state:

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
   keys the map by object identity, so only this loop's ``i`` matches.

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
   :widths: 45 55

   * - Construct
     - What it does
   * - |match_any!|_
     - Branches on the node type; the first matching arm runs
   * - |try_cast|_, |as_node|_
     - Downcasts to one type, as a managed reference in a ``Result`` or a
       borrowed object in an ``Option``
   * - ``.into()``
     - Upcasts a managed reference to a parent type
   * - |Var::new|_, |ex|_\ ``(...)``, ``.expr()``
     - Creates a node; builds an expression with operators
   * - |structural_walk|_\ ``(&func, callbacks, WalkOrder::PreOrder)``
     - Walks with typed callbacks that return ``WalkResult``
   * - |structural_map|_\ ``(func, callbacks, order)``
     - Rebuilds with replacements and returns an ``Any``
   * - A trailing |DefRegionKind|_ parameter
     - Receives the definition region kind
   * - ``Function::get_global("ffi.StructuralEqual")``
     - Compares by structure
   * - |dispatch|_\ ``(walk)``, ``(map)``, ``(visit)``, ``(mutate)``
     - Turns the methods of a struct into typed callbacks
   * - |VisitCallbacks|_, |MutateCallbacks|_
     - Attach state to a tuple of visit or mutate callbacks
   * - |static_init_block!|_, |register_global_typed|_
     - Registers Rust functions as global functions

.. API pages of the tvm-tirx and tvm-ffi crates for the names above.

.. |static_init_block!| replace:: ``static_init_block!``
.. _static_init_block!: ../reference/rust/generated/tvm_ffi/macro.static_init_block.html
.. |AddObj| replace:: ``AddObj``
.. _AddObj: ../reference/rust/generated/tvm_tirx/ir/prim/expr/struct.AddObj.html
.. |Add| replace:: ``Add``
.. _Add: ../reference/rust/generated/tvm_tirx/ir/prim/expr/struct.Add.html
.. |script| replace:: ``script``
.. _script: ../reference/rust/generated/tvm_tirx/script/fn.script.html
.. |try_cast| replace:: ``.try_cast()``
.. _try_cast: ../reference/rust/generated/tvm_ffi/object/trait.ObjectRefCast.html
.. |structural_walk| replace:: ``structural_walk``
.. _structural_walk: ../reference/rust/generated/tvm_ffi/extra/structural_visit/fn.structural_walk.html
.. |structural_map| replace:: ``structural_map``
.. _structural_map: ../reference/rust/generated/tvm_ffi/extra/structural_mutate/fn.structural_map.html
.. |structural_visit| replace:: ``structural_visit``
.. _structural_visit: ../reference/rust/generated/tvm_ffi/extra/structural_visit/fn.structural_visit.html
.. |structural_mutate| replace:: ``structural_mutate``
.. _structural_mutate: ../reference/rust/generated/tvm_ffi/extra/structural_mutate/fn.structural_mutate.html
.. |WalkResult| replace:: ``WalkResult``
.. _WalkResult: ../reference/rust/generated/tvm_ffi/extra/structural_visit/enum.WalkResult.html
.. |DefRegionKind| replace:: ``DefRegionKind``
.. _DefRegionKind: ../reference/rust/generated/tvm_ffi/extra/structural_visit/enum.DefRegionKind.html
.. |make_const| replace:: ``make_const``
.. _make_const: ../reference/rust/generated/tvm_tirx/ir/prim/op/fn.make_const.html
.. |dispatch| replace:: ``#[dispatch]``
.. _dispatch: ../reference/rust/generated/tvm_ffi/attr.dispatch.html
.. |VisitCallbacks| replace:: ``VisitCallbacks``
.. _VisitCallbacks: ../reference/rust/generated/tvm_ffi/extra/structural_visit/struct.VisitCallbacks.html
.. |MutateCallbacks| replace:: ``MutateCallbacks``
.. _MutateCallbacks: ../reference/rust/generated/tvm_ffi/extra/structural_mutate/struct.MutateCallbacks.html
.. |ex| replace:: ``ex``
.. _ex: ../reference/rust/generated/tvm_tirx/ir/prim/op/fn.ex.html
.. |ObjectIdentity| replace:: ``ObjectIdentity``
.. _ObjectIdentity: ../reference/rust/generated/tvm_ffi/object/struct.ObjectIdentity.html
.. |StructuralVisitorWithConstraintContext| replace:: ``StructuralVisitorWithConstraintContext``
.. _StructuralVisitorWithConstraintContext: ../reference/rust/generated/tvm_tirx/tirx/struct.StructuralVisitorWithConstraintContext.html
.. |StructuralMutatorWithConstraintContext| replace:: ``StructuralMutatorWithConstraintContext``
.. _StructuralMutatorWithConstraintContext: ../reference/rust/generated/tvm_tirx/tirx/struct.StructuralMutatorWithConstraintContext.html
.. |match_any!| replace:: ``match_any!``
.. _match_any!: ../reference/rust/generated/tvm_ffi/macro.match_any.html
.. |Var::new| replace:: ``Var::new``
.. _Var::new: ../reference/rust/generated/tvm_tirx/ir/expr/struct.Var.html#method.new
.. |as_node| replace:: ``as_node``
.. _as_node: ../reference/rust/generated/tvm_ffi/object/trait.ObjectRefCore.html#method.as_node
.. |register_global_typed| replace:: ``Function::register_global_typed``
.. _register_global_typed: ../reference/rust/generated/tvm_ffi/function/struct.Function.html#method.register_global_typed
