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

C++ API
=======

This page contains the API reference for the C++ API. The full API index below
can be a bit dense, so we recommend the following tips first:

- The :ref:`cpp-key-classes` lists the key classes that are most commonly used.
- See :ref:`cpp-header-organization` for how the modules are organized.
- You can go to the Full API Index at the bottom of this page to access the full list of APIs.

  - We usually group the APIs by files. You can look at the file hierarchy in the
    full API index and navigate to the specific file to find the APIs in that file.

.. _cpp-header-organization:

Header Organization
-------------------

The C++ headers are organized into the following folders, in dependency order:
each folder builds on those above it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Folder
     - Description
   * - ``tvm/support/``
     - Compiler-independent common utilities.
   * - ``tvm/runtime/``
     - Minimal runtime foundation for executing compiled code.
   * - ``tvm/ir/``
     - Core IR that language variants build upon, extend, and customize.
   * - ``tvm/sym/``
     - Symbolic analysis: integer arithmetic, integer sets, and iterator affine maps.
   * - ``tvm/script/``
     - Extensible IR builder, printer, and Python parser support for language variants.
   * - ``tvm/target/``
     - Target configuration schema and tag registry.
   * - ``tvm/tirx/``
     - Tensor program types, layouts, and operations.
   * - ``tvm/codegen/``
     - Generic host code generation and intrinsic lowering rules.
   * - ``tvm/backend/``
     - Backend-specific attributes, intrinsics, passes, and code generation.


.. _cpp-key-classes:

Key Classes
-----------

This section lists the key classes of the core IR and tensor programs.

The IR follows a common convention: each IR node is a TVM FFI ``Object`` whose
class name carries the ``Obj`` suffix, such as :cpp:class:`tvm::prim::AddObj`,
paired with a named managed reference class, such as :cpp:class:`tvm::prim::Add`.
C++ extensions written against the former ``FooNode`` spelling must use
``FooObj`` and rebuild. Managed reference names, runtime type keys, and serialized
identities remain unchanged.
A reference class can define its own constructors, operators, and member functions,
and inherit from other reference classes to share and specialize its interface.
It can also impose additional constraints on the node it holds, as described below.
The tables list the reference classes; to find the node class behind a reference,
check its ``ContainerType`` member. IR reference classes are non-nullable; use
``ffi::Optional<T>`` when a value may be absent. See also `Object and Class
<https://tvm.apache.org/ffi/concepts/object_and_class.html>`__.

Besides the named reference for each node, some reference types add constraints on
the node they hold. For example, ``TypedExpr<T>`` holds any
:cpp:class:`tvm::ExprObj` whose ``ty`` field is a ``T``, and
:cpp:class:`tvm::PrimExpr`, defined as ``TypedExpr<PrimType>``, holds any
``ExprObj`` whose ``ty`` is a :cpp:class:`tvm::PrimType`. These constraints are
defined by the TVM FFI ``TypeTraits`` specialization of the reference type.

Type
^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :cpp:class:`tvm::Type`
     - The base class of all types.
   * - :cpp:class:`tvm::MissingType`
     - Type information that has not been supplied or computed.
   * - :cpp:class:`tvm::OpaqueType`
     - Type marker for opaque values that must be removed from finished IR.
   * - :cpp:class:`tvm::PrimType`
     - A primitive scalar or vector data type, including the void sentinel.
   * - :cpp:class:`tvm::NoneType`
     - The concrete type whose only value is None.
   * - :cpp:class:`tvm::AnyType`
     - The top type, which admits any value.
   * - :cpp:class:`tvm::TensorRegionType`
     - The type of a multi-dimensional tensor region.
   * - :cpp:class:`tvm::StringType`
     - Semantic string type, independent of the function's calling convention.
   * - :cpp:class:`tvm::PtrType`
     - A pointer to an element type in a specified memory scope.
   * - :cpp:class:`tvm::TupleType`
     - The type of tuple values.
   * - :cpp:class:`tvm::FuncType`
     - Function type.

Expr
^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :cpp:class:`tvm::Expr`
     - Base class of typed IR expressions.
   * - :cpp:class:`tvm::StagingExpr`
     - A traversable expression eliminated before executable IR.
   * - :cpp:class:`tvm::OpaqueExpr`
     - Base class for opaque values that must be removed from finished IR.
   * - :cpp:class:`tvm::Constant`
     - Base class of literal constants.
   * - :cpp:class:`tvm::GenericConst`
     - A literal payload with an explicit expression type.
   * - :cpp:class:`tvm::StringImm`
     - A string literal with StringType.
   * - :cpp:class:`tvm::DataTypeImm`
     - A data type literal whose expression type is AnyType.
   * - :cpp:class:`tvm::IntImm`
     - Integer constant.
   * - :cpp:class:`tvm::FloatImm`
     - Floating-point constant.
   * - :cpp:class:`tvm::Tuple`
     - Tuple expression that groups several fields together.
   * - :cpp:class:`tvm::TupleGetItem`
     - Get the index-th item from a tuple.
   * - :cpp:class:`tvm::TensorLoad`
     - Load an element or vector from an indexed expression source.
   * - :cpp:class:`tvm::Var`
     - A local variable in the IR.
   * - :cpp:class:`tvm::LambdaExpr`
     - A typed staging expression representing a lambda computation.
   * - :cpp:class:`tvm::GlobalVar`
     - A global variable that refers to a function in the module.
   * - :cpp:class:`tvm::Call`
     - Call a registered operator or another callable IR expression.
   * - :cpp:class:`tvm::TensorRegion`
     - A region of an indexed expression source.
   * - :cpp:class:`tvm::BaseFunc`
     - Base class for callable IR function definitions.
   * - :cpp:class:`tvm::Op`
     - A registered primitive operator used as a Call callee.
   * - :cpp:class:`tvm::OpAttrMap`
     - The values of one operator attribute, keyed by operator.

Stmt
^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :cpp:class:`tvm::Stmt`
     - Base class of all statements.
   * - :cpp:class:`tvm::SeqStmt`
     - Sequence of statements.
   * - :cpp:class:`tvm::Bind`
     - Bind a variable to a value in the enclosing scope.
   * - :cpp:class:`tvm::RegionStmt`
     - An operation with enclosing-scope operands and one lexical body.
   * - :cpp:class:`tvm::AssertStmt`
     - Check a condition and raise the specified error when it is false.
   * - :cpp:class:`tvm::TensorStore`
     - Store a primitive value into a tensor destination.
   * - :cpp:class:`tvm::Evaluate`
     - Evaluate an expression for its effects and discard its result.
   * - :cpp:class:`tvm::If`
     - Execute one branch according to a boolean condition.
   * - :cpp:class:`tvm::For`
     - Loop over an integer range.
   * - :cpp:class:`tvm::While`
     - Repeat the body while the condition evaluates to true.
   * - :cpp:class:`tvm::Return`
     - Exit the enclosing function with a return value.
   * - :cpp:class:`tvm::Break`
     - Exit the innermost enclosing loop.
   * - :cpp:class:`tvm::Continue`
     - Skip to the next iteration of the innermost enclosing loop.

Location, IRModule, and Supporting Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :cpp:class:`tvm::SourceName`
     - An identifier for a source location.
   * - :cpp:class:`tvm::Location`
     - Source-location metadata attached to IR nodes.
   * - :cpp:class:`tvm::UnknownLoc`
     - The location used when source information is unavailable.
   * - :cpp:class:`tvm::SourceLoc`
     - A source range in frontend coordinates.
   * - :cpp:class:`tvm::CallSiteLoc`
     - A callee's source location together with its caller's location.
   * - :cpp:class:`tvm::IRModule`
     - A collection of global functions and module-level metadata; the basic unit of IR transformations.
   * - :cpp:class:`tvm::GlobalInfo`
     - Base class for module-level information referenced by the IR.
   * - :cpp:class:`tvm::Attrs`
     - Base class for structured operator and function attributes.
   * - :cpp:class:`tvm::DictAttrs`
     - A string-keyed attribute map for functions, modules, and calls.
   * - :cpp:class:`tvm::EnvFunc`
     - A registered global function referenced and serialized by name.
   * - :cpp:class:`tvm::Range`
     - A half-open interval stored as begin and extent.

Tensor Program
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :cpp:class:`tvm::tirx::TensorType`
     - Type of a tensor value: shape, dtype, and memory scope.
   * - :cpp:class:`tvm::tirx::TensorMapType`
     - Type of a hardware tensor map (TMA descriptor).
   * - :cpp:class:`tvm::tirx::ThreadScope`
     - Canonical thread-scope identity, named as backend.scope.
   * - :cpp:class:`tvm::tirx::IndexMap`
     - Mapping between index spaces used for layout transforms.
   * - :cpp:class:`tvm::tirx::CallFFIKernelAttr`
     - Attributes describing an FFI kernel call and its launch values.
   * - :cpp:class:`tvm::tirx::Layout`
     - Base reference for tensor-program coordinate layouts.
   * - :cpp:class:`tvm::tirx::TileLayout`
     - Tiled layout over hardware and memory axes.
   * - :cpp:class:`tvm::tirx::ComposeLayout`
     - Tile layout composed with an XOR swizzle on the memory coordinate.
   * - :cpp:class:`tvm::tirx::TensorVar`
     - Variable bound to a tensor.
   * - :cpp:class:`tvm::tirx::Function`
     - TIRx tensor-program function.


.. _cpp-full-api-index:

Full API Index
--------------

`Browse the full API index <generated/index.html>`__
