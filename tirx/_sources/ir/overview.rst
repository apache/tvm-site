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

An intermediate representation (IR) describes a program in a form that compiler
components can inspect and transform. Throughout this walkthrough, we'll use
the CPU and GPU examples from :doc:`/get_started/quick_start` to introduce
the main IR objects and show how they fit together.

TVMScript constructs IR objects to describe the computation in these examples.
PyTorch tensors hold the input and output data, and the compiled functions
perform the computation when called.

TIRx shares its IR object system across C++, Python, and Rust through TVM FFI,
so compiler components written in different languages can work together. The
same infrastructure supports custom types and operators and can be reused to
build analyses, transformation passes, and backends.

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
   * - Types
     - ``ir.Type``
     - Describe value properties, such as dtypes and tensor shapes.
     - ``T.float32``
   * - Statements
     - ``ir.Stmt``
     - Describe execution order and control flow.
     - ``B[i] = A[i] + 1``
   * - Source locations
     - ``ir.Location``
     - Record where IR nodes originate in the source.
     - The filename and line number of ``A[i] + 1``.
   * - Functions
     - ``ir.BaseFunc``, ``tirx.Function``
     - Define parameters and a computation body.
     - ``add_one`` defined with ``@T.function``.
   * - Modules
     - ``ir.IRModule``
     - Organize named functions into a program.
     - ``ir.IRModule({"add_one": add_one})``

The hierarchy below shows the inheritance relationships between these classes:

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

The hierarchy shows inheritance between classes. The smaller diagrams in
the following chapters show selected fields of individual objects. Each
branch names a field, list entry, or mapping entry; for example,
``seq[0]`` can refer to an ``ir.Bind`` stored inside an ``ir.SeqStmt``.

.. _choose-a-chapter:

What to Read Next
-----------------

The CPU example provides the tensor types, arithmetic, and loop; the GPU
example adds a variable binding, a conditional, and calls for thread
coordinates. Short additional examples introduce scalar and vector
values through the IR API.

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
