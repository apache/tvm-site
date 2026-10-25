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

In :doc:`/get_started/example`, we wrote and ran our first TIRx program. To
improve efficiency, we need to convert the program into a common intermediate
representation (IR) and apply compiler passes to it. In this section, we'll
explore the IR system in tvm-tirx in detail.

The main components of TIRx IR are:

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - Component
     - Main classes
     - Purpose
     - Example
   * - Expressions
     - ``ir.Expr``
     - Represent values and computations.
     - ``A[i] + 1``
   * - Statements
     - ``ir.Stmt``
     - Describe execution order and control flow.
     - ``B[i] = A[i] + 1``
   * - Types
     - ``ir.Type``
     - Describe value properties, such as dtypes and tensor shapes.
     - ``T.float32``
   * - Functions
     - ``ir.BaseFunc``, ``tirx.Function``
     - Define parameters and a computation body.
     - ``add_one`` defined with ``@T.function``.
   * - Modules
     - ``ir.IRModule``
     - Organize named functions into a program.
     - ``ir.IRModule({"add_one": add_one})``
   * - Source locations
     - ``ir.Location``
     - Record where IR nodes originate in the source.
     - The filename and line number of ``A[i] + 1``.

The IR makes values, computations, control flow, and types explicit, allowing
analyses and transformations to work with the program's structure independently
of the syntax used to construct it.

TIRx's C++, Python, and Rust interfaces share a common IR object system through
TVM FFI, allowing compiler components written in different languages to work
together. Built on shared C++ IR infrastructure, the IR can be extended with
custom operators and types, and its existing interfaces can be used to build
custom analyses, transformation passes, and backends.

The following tree shows the detailed inheritance hierarchy of the core IR classes.

.. code-block:: text

   ffi.Object                                          TVM FFI base object
   ├── ir.Expr                                         Expressions
   │   ├── ir.Constant                                 Literal constants
   │   │   ├── ir.IntImm                               Integer constants
   │   │   ├── ir.FloatImm                             Floating-point constants
   │   │   ├── ir.StringImm                            String constants
   │   │   ├── ir.DataTypeImm                          Dtype constants
   │   │   └── ir.GenericConst                         Constants with arbitrary payloads
   │   ├── ir.Var                                      Local variables
   │   ├── ir.GlobalVar                                Global function references
   │   ├── ir.Tuple                                    Tuple construction
   │   ├── ir.TupleGetItem                             Tuple field access
   │   ├── ir.Call                                     Calls
   │   ├── ir.Op                                       Registered operators
   │   ├── ir.TensorLoad                               Tensor element reads
   │   ├── ir.TensorRegion                             Tensor regions
   │   ├── ir.BaseFunc                                 Function definitions
   │   │   └── tirx.Function                           Tensor-program functions
   │   ├── ir.StagingExpr                              Expressions removed during compilation
   │   │   └── ir.LambdaExpr                           Construction-time lambdas
   │   ├── ir.OpaqueExpr                               Opaque construction-time expressions
   │   ├── prim.Cast                                   Dtype conversion
   │   ├── prim.Add                                    Addition
   │   ├── prim.Sub                                    Subtraction
   │   ├── prim.Mul                                    Multiplication
   │   ├── prim.Div                                    Division
   │   ├── prim.Mod                                    Remainder
   │   ├── prim.FloorDiv                               Floor division
   │   ├── prim.FloorMod                               Floor remainder
   │   ├── prim.Min                                    Minimum
   │   ├── prim.Max                                    Maximum
   │   ├── prim.LShift                                 Left bit shift
   │   ├── prim.RShift                                 Right bit shift
   │   ├── prim.BitwiseAnd                             Bitwise AND
   │   ├── prim.BitwiseOr                              Bitwise OR
   │   ├── prim.BitwiseXor                             Bitwise XOR
   │   ├── prim.BitwiseNot                             Bitwise NOT
   │   ├── prim.EQ                                     Equal
   │   ├── prim.NE                                     Not equal
   │   ├── prim.LT                                     Less than
   │   ├── prim.LE                                     Less than or equal
   │   ├── prim.GT                                     Greater than
   │   ├── prim.GE                                     Greater than or equal
   │   ├── prim.And                                    Logical AND
   │   ├── prim.Or                                     Logical OR
   │   ├── prim.Not                                    Logical NOT
   │   ├── prim.Select                                 Conditional value selection
   │   ├── prim.Let                                    Expression-local binding
   │   ├── prim.Ramp                                   Vector arithmetic progression
   │   ├── prim.Broadcast                              Scalar broadcast to vector lanes
   │   └── prim.Shuffle                                Vector lane selection
   ├── ir.Type                                         Types
   │   ├── ir.MissingType                              Unresolved type information
   │   ├── ir.NoneType                                 The type of None
   │   ├── ir.AnyType                                  Any value
   │   ├── ir.OpaqueType                               Opaque construction-time types
   │   ├── ir.PrimType                                 Scalar and vector dtypes
   │   ├── ir.StringType                               String values
   │   ├── ir.PtrType                                  Pointer types
   │   ├── ir.TupleType                                Tuple field types
   │   ├── ir.FuncType                                 Function signatures
   │   ├── ir.TensorRegionType                         Tensor region values
   │   ├── tirx.TensorType                             Tensor types
   │   └── tirx.TensorMapType                          Tensor map types
   ├── ir.Stmt                                         Statements
   │   ├── ir.SeqStmt                                  Statement sequences
   │   ├── ir.Bind                                     Variable definitions
   │   ├── ir.RegionStmt                               Operator-defined lexical regions
   │   ├── ir.AssertStmt                               Program assertions
   │   ├── ir.TensorStore                              Tensor element writes
   │   ├── ir.Evaluate                                 Evaluation with an unused result
   │   ├── ir.If                                       Conditional branches
   │   ├── ir.For                                      Counted loops
   │   ├── ir.While                                    Conditional loops
   │   ├── ir.Return                                   Function returns
   │   ├── ir.Break                                    Loop exits
   │   └── ir.Continue                                 Next loop iterations
   ├── ir.Location                                     Source locations
   │   ├── ir.UnknownLoc                               Unavailable source information
   │   ├── ir.SourceLoc                                Source ranges
   │   └── ir.CallSiteLoc                              Callee and caller locations
   └── ir.IRModule                                     Global function definitions

.. _choose-a-chapter:

What to Read Next
-----------------

After the add-one example in :doc:`/get_started/example`, this walkthrough
uses a bias + sigmoid program to introduce symbolic shapes and scalar computations.
It computes ``Y[i, j] = 1 / (1 + exp(-(X[i, j] + bias[j])))``.
Download :download:`tvmscript_sigmoid.py <examples/tvmscript_sigmoid.py>`
and run ``python tvmscript_sigmoid.py`` with an LLVM-enabled TIRx build.

* The :doc:`types_and_expressions` chapter explains how values, computations,
  and tensor shapes are represented.
* The :doc:`statements` chapter covers statement bodies with bindings, tensor
  writes, branches, and loops.
* The :doc:`operators_and_calls` chapter introduces registered operators and
  function calls.
* The :doc:`functions_and_modules` chapter shows how functions are defined
  and organized in modules.
* The :doc:`printing_and_parsing` chapter covers TVMScript definitions,
  program generation, and conversion between source text and IR.
