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

TIRx is implemented in C++, and a C++ function can read, build, and
rewrite its IR directly. The function can receive a program written in
TVMScript, inspect or rewrite it, and return the result, with no
conversion in between, which lets us move a hot analysis or a large
transformation to C++. This guide covers:

- **TVM-FFI object system**: how IR nodes are represented and used in C++.
- **Structural utilities**: walk, map, and compare programs.
- **Visitor and mutator classes**: one hook per node type, for larger passes.

.. note::

   The runnable code in this guide is under
   `examples/ir_guides <https://github.com/apache/tvm-tirx/tree/main/examples/ir_guides>`_
   in the repository.

Setting Up
----------

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

A ``TVM_FFI_STATIC_INIT_BLOCK`` at the end of the source registers each
example as a global function under ``tirx_ir_guide_examples.cpp``:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-at: refl::GlobalDef()
   :end-at: .def("tirx_ir_guide_examples.cpp.construct", Construct)
   :dedent: 2

The package's ``tirx_ir_guide_examples.cpp`` module loads the library and
exposes these functions, so Python calls them by name. The driver
``working_with_ir_in_cpp.py`` calls each example this way and prints the
result.

The object system examples take no arguments: each builds its own IR in
C++. The later sections receive the ``add_one`` function from Quick Start,
written in TVMScript:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

.. code-block:: python

   import tirx_ir_guide_examples.cpp

   tirx_ir_guide_examples.cpp.construct()
   tirx_ir_guide_examples.cpp.tensors_written(add_one)

The headers relevant to this guide sit in five folders under ``tvm/``;
see :ref:`cpp-header-organization` for the full list:

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
       ``TensorStore``, and the expression functors in ``expr_functor.h``.
     - ``expr_functor.h``
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

IR classes live in the ``tvm`` namespace, primitive expressions such as
:cpp:class:`tvm::prim::Add` in ``tvm::prim``, TIRx-specific ones such as
:cpp:class:`tvm::tirx::Function` in ``tvm::tirx``, and the TVM-FFI classes
in ``tvm::ffi``. The examples start with ``using namespace tvm;``, so these
appear as ``Var``, ``prim::Add``, ``tirx::Function``, and
:cpp:func:`ffi::StructuralWalk <tvm::ffi::StructuralWalk>`.

TVM-FFI Object System
---------------------

Every TIRx IR node is a TVM-FFI object. TIRx is designed for **dynamic
extension**: a library can add a new node type, for example a new
statement for a hardware feature, and existing code keeps working on
programs that contain it, which a fixed class hierarchy compiled into one
library cannot offer. Generic utilities such as walking, rewriting, comparing, and
printing work on every node through its registered fields.

TVM-FFI objects also work across languages. Python, C++, and Rust share
the same native, immutable, reference-counted objects, so a program moves
between them without conversion.

**In C++, TVM-FFI objects are ordinary C++ classes, used through a few
key constructs:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Construct
     - Purpose
   * - Constructors, ``PrimExpr`` operators
     - Build expressions with ordinary C++ operators
   * - ``FooObj`` / ``Foo``
     - Underlying object and its named managed reference: read fields with
       ``->``, copy cheaply
   * - ``CopyOnWrite()``
     - Change a field, copying the object only if it is shared
   * - Implicit upcast
     - Pass a node wherever a parent type is expected
   * - ``as<T>()`` / ``as_or_throw<T>()``
     - Convert to a child type, with a checked result
   * - Typed views (``PrimExpr``, ``IntExpr``, ``PrimVar``)
     - Require an expression of a given type, checked on conversion
   * - ``ExprFunctor`` / ``StmtFunctor``
     - Process each node type in its own method

The subsections below introduce each construct.

Constructing Objects
~~~~~~~~~~~~~~~~~~~~

**We build IR with constructors and ordinary C++ operators.** Writing
``x * 2 + 1`` as nested constructor calls would bury the expression, so
arithmetic operators on :cpp:class:`PrimExpr <tvm::PrimExpr>` build the
nodes for us. Let's build it:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [construct.begin]
   :end-before: // [construct.end]

The output is:

.. code-block:: text

   x * 2 + 1

:cpp:class:`PrimVar <tvm::PrimVar>` creates an integer variable ``x``, a
variable known to have a primitive type.
``*`` and ``+`` turn the literals ``2`` and ``1`` into constants of the
type of ``x``, fold constants, and otherwise call the node constructors,
such as ``prim::Add(a, b)``. Writing a node to a stream prints it as
TVMScript; ``Print`` is our small helper that does so.

We now have an expression; next, let's see what kind of C++ value it is.

Objects and Managed References
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Every node type comes as a pair: an underlying object and a named
managed reference.** The top of
``x * 2 + 1`` is an addition, so let's view the expression as an addition,
with a cast the next subsection explains, and read its two operands:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [objects.begin]
   :end-before: // [objects.end]

The output is:

.. code-block:: text

   a = x * 2
   b = 1
   same object: true

``alias`` is the same object as ``add``: copying a managed reference copies
only the pointer, and ``same_as`` compares identity.

:cpp:class:`AddObj <tvm::prim::AddObj>` is the **underlying object**: the
class that holds the fields. :cpp:class:`Add <tvm::prim::Add>` is a
**named managed reference** to it: an owning, reference-counted pointer,
like ``std::shared_ptr<AddObj>`` except that it is never null, so copying
it is cheap and ``->`` is always safe. Functions take and return managed
references, and ``->`` on a managed reference reaches the object's fields:
``add->a``.

We define a separate named class ``Add`` instead of using a plain pointer
to ``AddObj`` because the named class carries the interface: constructors,
methods, and the type information that lets it pass through TVM-FFI
functions. It can also impose additional constraints on the underlying
object that the object alone does not express.

**Nodes are immutable once shared.** To change a field, ``CopyOnWrite()``
returns a mutable pointer to the object, copying the object first if
anyone else holds it:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [cow.begin]
   :end-before: // [cow.end]

The output is:

.. code-block:: text

   add = x * 2 + 1
   changed = x * 2 + 3

``changed`` shared its object with ``add``, so ``CopyOnWrite()`` copied it
before the write, and ``add`` is untouched.

We reached ``Add`` with a cast; next, let's look at casts on their own.

Casts
~~~~~

**Casts move a managed reference up and down the class hierarchy.** A
function that takes any expression wants an ``Expr``, while reading the
operands needs an ``Add``. Converting up is implicit, through C++
inheritance. Converting down checks the node's actual type, with
|ObjectRef::as|_ and its relatives in one of three forms:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Returns
   * - ``value.as<AddObj>()``
     - A non-owning ``const AddObj*``, valid while a reference keeps the
       object alive, or ``nullptr`` if it is not an addition.
   * - ``value.as<Add>()``
     - A ``std::optional<Add>``.
   * - ``value.as_or_throw<Add>()``
     - The ``Add``, or raises an error.

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [casts.begin]
   :end-before: // [casts.end]

The output is:

.. code-block:: text

   as<AddObj>: pointer
   as<Add>: Add
   as<Mul>: nullopt
   as_or_throw<Mul>: Cannot treat type `prim.Add` as type `prim.Mul`

``value`` is an ``Expr``, but the node behind it is still an addition, so
the casts to ``AddObj`` and ``Add`` succeed and the cast to ``Mul`` finds
nothing. ``as<AddObj>()`` works like ``dynamic_cast<const AddObj*>``, and
``as<Add>()`` and ``as_or_throw<Add>()`` do the same check but return a
managed reference. ``as_or_throw`` turns the same mismatch into an error
that names both type keys: the actual ``prim.Add`` and the requested
``prim.Mul``.

Casts check the class of the object; next, let's look at references that
also check the type of the expression.

Typed Views
~~~~~~~~~~~

**A typed view constrains an expression by its type, not by its object
class.** Arithmetic needs expressions whose result is a primitive value,
but an addition, a variable, and a call can all produce one, so no single
object class captures "primitive". :cpp:class:`PrimExpr <tvm::PrimExpr>`
is a named managed reference that accepts any ``Expr`` whose ``ty`` is a
``PrimType``, and has no object class of its own.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - View
     - Accepts
   * - :cpp:class:`PrimExpr <tvm::PrimExpr>`
     - Any expression whose ``ty`` is a ``PrimType``
   * - :cpp:class:`IntExpr <tvm::IntExpr>`
     - A scalar integer expression, signed or unsigned
   * - :cpp:class:`PrimVar <tvm::PrimVar>`
     - A ``Var`` whose ``ty`` is a ``PrimType``

The same casts check the view. Let's try two variables, an integer and a
float:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [views.begin]
   :end-before: // [views.end]

The output is:

.. code-block:: text

   i as PrimExpr: yes
   i as IntExpr: yes
   f as IntExpr: no
   as_or_throw<IntExpr>: Cannot treat type `ir.Var[ty=float32]` as type `ir.IntExpr`

Both variables are ``VarObj`` objects; only their ``ty`` differs, and the
views check it. The error shows the type in brackets, ``ir.Var[ty=float32]``.
This is why ``x`` is a ``PrimVar``: the operators take ``PrimExpr``, and a
plain ``Var`` would first need a checked cast.

.. note::

   A view is not a subclass of the node's own reference class:
   ``PrimVar`` derives from ``PrimExpr`` and converts to ``Var`` through a
   conversion operator.

A cast or a view looks at one node at a time; to process every node type
in its own method, we dispatch.

Dispatching on Node Types
~~~~~~~~~~~~~~~~~~~~~~~~~

**A functor picks one of its own methods by the node's runtime type, as
in the visitor pattern.** :cpp:class:`ExprFunctor <tvm::ExprFunctor>`
declares a virtual ``Dispatch_`` overload for each expression class, and a
subclass overrides the ones it cares about. ``Dispatch`` looks up the
node's type in a table and calls the matching overload, so the nodes need
no ``accept()`` method. Let's print an expression with
explicit parentheses:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [describe.begin]
   :end-before: // [describe.end]

We call it on ``x * 2 + 1`` and on ``x - 1``:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [dispatch_use.begin]
   :end-before: // [dispatch_use.end]

The output is:

.. code-block:: text

   ((x * 2) + 1)
   <other>

The template argument chooses the signature, here ``std::string`` from an
``Expr``. A functor does not visit children on its own: the ``Add`` and
``Mul`` overloads recurse by calling ``Dispatch`` on the operands, and the
``Var`` and ``IntImm`` overloads end the recursion.

**Unhandled node types fall back to a default method.** They go to
``DispatchDefault_``, which raises an error unless we override it. Our
override returns ``<other>``, so ``x - 1`` passes through instead of
failing.
:cpp:class:`tirx::StmtFunctor <tvm::tirx::StmtFunctor>` does the same for
statements.

.. note::

   The ``using ExprFunctor::Dispatch_;`` line keeps the overloads we did
   not override visible, as usual for overloaded virtual functions in C++.

Dispatch looks at one node and recurses by hand; the structural utilities
below walk whole programs for us.

Structural Utilities in C++
---------------------------

The structural utilities are part of TVM-FFI and work on every node
through its registered fields:
:cpp:func:`ffi::StructuralWalk <tvm::ffi::StructuralWalk>`,
:cpp:func:`ffi::StructuralMap <tvm::ffi::StructuralMap>`,
:cpp:class:`ffi::StructuralEqual <tvm::ffi::StructuralEqual>`, and
:cpp:class:`ffi::StructuralHash <tvm::ffi::StructuralHash>`. **Each piece
takes a C++ form**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Piece
     - C++ form
   * - Nodes a callback receives
     - Selected by the type of the callback's first parameter
   * - Walk callback result
     - ``ffi::Expected<WalkResult>``
   * - Traversal order
     - A required template argument, ``WalkOrder::kPreOrder`` or
       ``kPostOrder``
   * - Keep a node in a map
     - Return ``ffi::Unchanged()`` or the node
   * - Result of a map
     - An ``ffi::Any``, cast back with ``cast<T>()``
   * - Definition region kind
     - A trailing ``TVMFFIDefRegionKind`` parameter

Structural Walk
~~~~~~~~~~~~~~~

A walk calls our callback on each node of the type it asks for. To find
the tensors that ``add_one`` writes, the callback takes a
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

To handle several node types in one walk, we pass one callback per type as
extra arguments. They are tried in declaration order, and only the first
one whose parameter type matches a node runs; a node that matches none is
still traversed but not reported. This walk records the loops and stores
in pre-order:

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
a ``TVMFFIDefRegionKind`` as its last parameter:
``kTVMFFIDefRegionKindNone`` for a use, and ``kTVMFFIDefRegionKindSimple``
and ``kTVMFFIDefRegionKindPattern`` for the two kinds of definition site.
Let's list the definition sites in ``add_one``:

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

The parameters define ``A``, ``B``, and ``n`` by pattern, and the loop
defines ``i``.

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
The mapper keeps definitions and uses consistent, so the new
``A`` and ``B`` and every use of them carry the specialized shape.

Equality and Hashing
~~~~~~~~~~~~~~~~~~~~

``ffi::StructuralEqual::Equal`` and ``ffi::StructuralHash::Hash`` compare
and hash by structure. Let's compare two ``add_one`` functions built
separately in Python by ``make_add(1)``, the helper from the structural
utilities guide:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [equal.begin]
   :end-before: // [equal.end]

The output is:

.. code-block:: text

   [False, True, True]

The functions are different objects, yet structurally equal with equal
hashes. ``Equal`` and ``Hash`` also take an optional ``map_free_vars``
argument, which defaults to ``false``.

Visit and Mutate
~~~~~~~~~~~~~~~~

When a callback must decide which children to visit, we use
:cpp:func:`ffi::StructuralVisit <tvm::ffi::StructuralVisit>` and
:cpp:func:`ffi::StructuralMutate <tvm::ffi::StructuralMutate>`. Their
callbacks also receive
the visitor or mutator, such as
:cpp:class:`ffi::StructuralVisitorObj* <tvm::ffi::StructuralVisitorObj>`,
and descend into a node's children with ``DefaultVisitExpected`` or
``DefaultMutateExpected``.

Visitor and Mutator Classes
---------------------------

Walk and map cover most tasks. A pass that handles many node types,
however, reads better as a class with one method per type:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Tool
     - Covers
     - Use it for
   * - Structural utilities
     - Any TVM-FFI object
     - A few node types, written as callbacks
   * - :cpp:class:`tirx::StmtExprVisitor <tvm::tirx::StmtExprVisitor>`,
       :cpp:class:`tirx::StmtExprMutator <tvm::tirx::StmtExprMutator>`
     - TIRx statements and expressions (``tvm/tirx/stmt_functor.h``)
     - Passes that customize many node types
   * - :cpp:class:`ExprVisitor <tvm::ExprVisitor>`,
       :cpp:class:`ExprMutator <tvm::ExprMutator>`
     - Expressions alone (``tvm/ir/expr_functor.h``)
     - Expression analyses and rewrites
   * - :cpp:class:`ExprFunctor <tvm::ExprFunctor>`,
       :cpp:class:`tirx::StmtFunctor <tvm::tirx::StmtFunctor>`
     - Dispatch on the node type with a signature we choose, such as
       ``int(const Expr&)``, without traversing children
     - Computations where each hook decides which children to recurse into

Each class declares a virtual hook for every node type, named ``Visit_``
for a visitor and ``Mutate_`` for a mutator, and a subclass overrides only
the types it cares about. Unlike the functors, these classes traverse
children for us.

**These classes are built on the structural utilities.** A node type
without a hook, such as a function, falls back to its structural traversal,
which hands each child back to our hooks.

.. note::

   ``ExprVisitor`` ultimately derives from
   :cpp:class:`ffi::StructuralVisitorObj <tvm::ffi::StructuralVisitorObj>`,
   and ``ExprMutator`` from the mutator base behind ``StructuralMap``. Any
   hook can read the definition region kind with ``def_region_kind()``.

Let's record the loops around each write:

.. literalinclude:: ../../examples/ir_guides/cpp/working_with_ir.cc
   :language: cpp
   :start-after: // [visit.begin]
   :end-before: // [visit.end]

The output is:

.. code-block:: text

   ['B in i']

Our ``For`` hook pushes the loop variable, calls the base class hook to
visit the loop's bounds and body, then pops the variable. Calling the base
hook visits the children with their usual traversal, and returning its
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

Across these tools, IR is immutable, rewrites return new programs that
share unchanged parts, and every utility respects definitions and uses.
Because C++ and Python share the same objects, a pass can move between the
two languages one function at a time.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Construct
     - What it does
   * - ``PrimVar x("x", PrimType::Int(32))``, ``x * 2 + 1``
     - Creates a node; builds an expression with operators
   * - ``add->a``, ``loop->var``
     - Reads a node's fields
   * - ``value.as<AddObj>()``, ``value.as<Add>()``,
       ``value.as_or_throw<Add>()``
     - Checks the node type and downcasts
   * - :cpp:class:`PrimExpr <tvm::PrimExpr>`, :cpp:class:`IntExpr <tvm::IntExpr>`,
       :cpp:class:`PrimVar <tvm::PrimVar>`
     - Typed views that check an expression's ``ty``
   * - ``ref.CopyOnWrite()->field = value``
     - Changes a field, copying the object first if it is shared
   * - :cpp:class:`ExprFunctor <tvm::ExprFunctor>`,
       :cpp:class:`tirx::StmtFunctor <tvm::tirx::StmtFunctor>`
     - One ``Dispatch_`` overload per node type, with explicit recursion
   * - :cpp:func:`ffi::StructuralWalk <tvm::ffi::StructuralWalk>`\ ``<ffi::WalkOrder::kPreOrder>(func, ...)``
     - Walks with typed callbacks that return
       ``ffi::Expected<ffi::WalkResult>``
   * - :cpp:func:`ffi::StructuralMap <tvm::ffi::StructuralMap>`\ ``<order>(func, ...)``,
       ``ffi::Unchanged()``
     - Rebuilds with replacements and returns an ``ffi::Any``
   * - A trailing ``TVMFFIDefRegionKind`` parameter
     - Receives the definition region kind
   * - :cpp:func:`ffi::StructuralEqual::Equal <tvm::ffi::StructuralEqual::Equal>`,
       :cpp:func:`ffi::StructuralHash::Hash <tvm::ffi::StructuralHash::Hash>`
     - Compares and hashes by structure
   * - :cpp:class:`tirx::StmtExprVisitor <tvm::tirx::StmtExprVisitor>`,
       :cpp:class:`tirx::StmtExprMutator <tvm::tirx::StmtExprMutator>`,
       :cpp:class:`ExprVisitor <tvm::ExprVisitor>`,
       :cpp:class:`ExprMutator <tvm::ExprMutator>`
     - One ``Visit_`` or ``Mutate_`` hook per node type
   * - ``TVM_FFI_STATIC_INIT_BLOCK``, ``refl::GlobalDef().def(...)``
     - Registers C++ functions as global functions

.. TVM-FFI C++ API page for ObjectRef, which the C++ reference does not include.

.. |ObjectRef::as| replace:: ``ObjectRef::as``
.. _`ObjectRef::as`: https://tvm.apache.org/ffi/reference/cpp/generated/classtvm_1_1ffi_1_1ObjectRef.html
