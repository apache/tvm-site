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

IR Overview
===========

One primary design goal of TIRx is to allow developers and agents to quickly
access every stage of compilation, not only its input and output, and to
work with every stage in the same way, so they can customize the compiler.

TIRx represents a program as a tree of intermediate representation (IR)
nodes, such as functions, statements, and values. The tree is similar to the
abstract syntax tree (AST) of Python code, but in a more structured, typed
form. A compiler pipeline analyzes and transforms this tree step by step and
maps it to the target program. At every stage, the tree can be printed as
Python code, in a syntax we call TVMScript (provided by the
``tvm_tirx.script`` module). For example, the following TVMScript program
defines ``add_one`` for a GPU:

.. literalinclude:: ../../examples/quick_start/add_one_cuda.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at: B[i] = A[i] + 1

``add_one`` adds one to every element of a tensor ``A`` of length ``n`` and
writes the result to ``B``. Each GPU thread computes its index ``i`` and
handles one element. At a high level, TIRx IR
contains the following elements:

- **Types** (:py:class:`ir.Type <tvm_tirx.ir.Type>`) describe the values that
  expressions produce, such as the scalar type ``T.float32`` and the tensor
  type ``T.Tensor((n,), T.float32)``.
- **Expressions** (:py:class:`ir.Expr <tvm_tirx.ir.Expr>`) compute values,
  such as ``A[i] + 1`` and the call ``T.cuda.thread_idx("x")`` to a registered
  operator (:py:class:`ir.Op <tvm_tirx.ir.Op>`).
- **Statements** (:py:class:`ir.Stmt <tvm_tirx.ir.Stmt>`) describe what the
  program does and in what order, such as the ``if`` branch and the write
  ``B[i] = A[i] + 1``.
- **Functions** (:py:class:`ir.BaseFunc <tvm_tirx.ir.BaseFunc>`) wrap a body
  of statements with parameters, such as ``add_one``.
- **Modules** (:py:class:`ir.IRModule <tvm_tirx.ir.IRModule>`) group named
  functions into a complete program.

Class Hierarchy
---------------

The following diagram shows the class hierarchy of the IR nodes:

.. code-block:: text

   ffi.Object                   TVM-FFI base object
   ├── ir.Expr                  Expressions
   │   ├── ir.Constant          Literal constants
   │   │   ├── ir.IntImm        Integer constants
   │   │   ├── ir.FloatImm      Floating-point constants
   │   │   ├── ir.StringImm     String constants
   │   │   ├── ir.DataTypeImm   Dtype constants
   │   │   └── ir.GenericConst  Constants with arbitrary payloads
   │   ├── ir.Var               Local variables
   │   ├── ir.GlobalVar         Global function references
   │   ├── ir.Tuple             Tuple construction
   │   ├── ir.TupleGetItem      Tuple field access
   │   ├── ir.Call              Calls
   │   ├── ir.Op                Registered operators
   │   ├── ir.TensorLoad        Tensor element reads
   │   ├── ir.TensorRegion      Tensor regions
   │   ├── ir.BaseFunc          Function definitions
   │   │   └── tirx.Function    Tensor-program functions
   │   ├── ir.StagingExpr       Expressions removed during compilation
   │   │   └── ir.LambdaExpr    Construction-time lambdas
   │   ├── ir.OpaqueExpr        Opaque construction-time expressions
   │   ├── prim.Cast            Primitive dtype conversion
   │   ├── prim.Add             Primitive addition, a + b
   │   ├── prim.Sub             Primitive subtraction, a - b
   │   ├── prim.Mul             Primitive multiplication, a * b
   │   ├── prim.Div             Primitive division in C semantics
   │   ├── prim.Mod             Primitive remainder in C semantics
   │   ├── prim.FloorDiv        Primitive floor division, a // b
   │   ├── prim.FloorMod        Primitive floor remainder, a % b
   │   ├── prim.Min             Primitive minimum, min(a, b)
   │   ├── prim.Max             Primitive maximum, max(a, b)
   │   ├── prim.LShift          Primitive left shift, a << b
   │   ├── prim.RShift          Primitive right shift, a >> b
   │   ├── prim.BitwiseAnd      Primitive bitwise AND, a & b
   │   ├── prim.BitwiseOr       Primitive bitwise OR, a | b
   │   ├── prim.BitwiseXor      Primitive bitwise XOR, a ^ b
   │   ├── prim.BitwiseNot      Primitive bitwise NOT, ~a
   │   ├── prim.EQ              Primitive equal, a == b
   │   ├── prim.NE              Primitive not equal, a != b
   │   ├── prim.LT              Primitive less than, a < b
   │   ├── prim.LE              Primitive less or equal, a <= b
   │   ├── prim.GT              Primitive greater than, a > b
   │   ├── prim.GE              Primitive greater or equal, a >= b
   │   ├── prim.And             Primitive logical AND
   │   ├── prim.Or              Primitive logical OR
   │   ├── prim.Not             Primitive logical NOT
   │   ├── prim.Select          Primitive conditional value selection
   │   ├── prim.Let             Primitive expression-local binding
   │   ├── prim.Ramp            Primitive vector arithmetic progression
   │   ├── prim.Broadcast       Primitive scalar broadcast to vector
   │   └── prim.Shuffle         Primitive vector lane selection
   ├── ir.Type                  Types
   │   ├── ir.MissingType       Unresolved type information
   │   ├── ir.NoneType          The type of None
   │   ├── ir.AnyType           Any value
   │   ├── ir.OpaqueType        Opaque construction-time types
   │   ├── ir.PrimType          Scalar and vector dtypes
   │   ├── ir.StringType        String values
   │   ├── ir.PtrType           Pointer types
   │   ├── ir.TupleType         Tuple field types
   │   ├── ir.FuncType          Function signatures
   │   ├── ir.TensorRegionType  Tensor region values
   │   ├── tirx.TensorType      Tensor types
   │   └── tirx.TensorMapType   Tensor map types
   ├── ir.Stmt                  Statements
   │   ├── ir.SeqStmt           Statement sequences
   │   ├── ir.Bind              Variable definitions
   │   ├── ir.RegionStmt        Operator-defined lexical regions
   │   ├── ir.AssertStmt        Program assertions
   │   ├── ir.TensorStore       Tensor element writes
   │   ├── ir.Evaluate          Evaluation with an unused result
   │   ├── ir.If                Conditional branches
   │   ├── ir.For               Counted loops
   │   ├── ir.While             Conditional loops
   │   ├── ir.Return            Function returns
   │   ├── ir.Break             Loop exits
   │   └── ir.Continue          Next loop iterations
   ├── ir.Location              Source locations
   │   ├── ir.UnknownLoc        Unavailable source information
   │   ├── ir.SourceLoc         Source ranges
   │   └── ir.CallSiteLoc       Callee and caller locations
   └── ir.IRModule              Global function definitions

These IR nodes are designed to be generic, so that new IR variants can be built
on top of them, and their namespaces show their roles. The ``ir`` namespace
holds the common nodes shared by all variants. Within it, ``ir.prim`` (shown as
``prim``) holds primitive expressions, which compute scalar or vector values of
a primitive type (:py:class:`ir.PrimType <tvm_tirx.ir.PrimType>`), such as
``float32`` or ``int32``. These expressions concisely represent dynamic indices
and element computations, such as ``A[i] + 1`` in ``add_one``. The ``tirx``
namespace builds on ``ir`` with the types and operations specific to tensor
programs, such as ``tirx.Function`` and ``tirx.TensorType``. More generally,
TIRx is designed for extension: downstream projects can build on the IR in the
same way, registering new operators, defining new types, and adding their own
TVMScript namespaces from Python, C++, or Rust, while reusing TIRx as a
foundation.

Every IR node can also carry a source location
(:py:class:`ir.Location <tvm_tirx.ir.Location>`) in its ``loc`` field, which
records where the node comes from in the source code and can be used for
diagnostics.

.. _choose-a-chapter:

What to Read Next
-----------------

- :doc:`types_and_expressions` explains how values, computations, and tensor
  shapes are represented.
- :doc:`statements` covers how a body binds variables, writes tensors,
  branches, and loops.
- :doc:`operators_and_calls` introduces registered operators and function
  calls.
- :doc:`functions_and_modules` shows how functions are defined and organized
  in modules.
- :doc:`printing_and_parsing` covers utilities for parsing and printing
  TVMScript.
