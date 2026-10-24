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
components can inspect and transform. It makes values, computations, control
flow, and types explicit, allowing analyses and transformations to work with the
program's structure independently of the syntax used to construct it.

TIRx's C++, Python, and Rust interfaces share a common IR object system through
TVM FFI, allowing compiler components written in different languages to work
together. The IR can be extended with custom operators and types, and its
existing interfaces can be used to build custom analyses, transformation passes,
and backends.

The following tree shows the inheritance relationships between IR object types.
Registered names can differ from Python API names. For example, ``prim.Add``
is available as ``ir.prim.Add`` in Python.

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
   │   ├── prim.Shuffle                                Vector lane selection
   │   ├── sym.IterMapExpr                             Iterator mapping expressions
   │   │   ├── sym.IterSplitExpr                       Iterator splitting
   │   │   └── sym.IterSumExpr                         Sums of iterator expressions
   │   └── sym.CanonicalExpr                           Internal simplifier expressions
   │       ├── sym.SplitExpr                           Internal split expressions
   │       └── sym.SumExpr                             Internal sum expressions
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
   ├── ir.IRModule                                     Global function definitions
   ├── ir.Attrs                                        Structured attributes
   │   ├── ir.DictAttrs                                Named attribute maps
   │   ├── tirx.CallFFIKernelAttr                       FFI kernel call attributes
   │   ├── tirx.tensor.ScopeAttrs                      Tensor instruction scopes
   │   ├── tirx.tensor.TrnAttrs                        Tensor instruction attributes
   │   ├── tirx.cuda.TensorMapEncodeTiledAttr           CUDA tensor map encoding attributes
   │   ├── tirx.cuda.TCGen05InstrDescriptorAttrs         CUDA instruction descriptors
   │   └── tirx.cuda.TCGen05InstrDescriptorBlockScaledAttrs  Block-scaled CUDA descriptors
   ├── ir.Location                                     Source locations
   │   ├── ir.UnknownLoc                               Unavailable source information
   │   ├── ir.SourceLoc                                Source ranges
   │   └── ir.CallSiteLoc                              Callee and caller locations
   ├── ir.SourceName                                   Source identifiers
   ├── ir.Range                                        Begin and end bounds
   ├── ir.ArgumentInfo                                 Operator argument descriptions
   ├── ir.GlobalInfo                                   Module-level metadata
   ├── ir.EnvFunc                                      Named global function references
   ├── ir.UniqueNameSupply                             Unique-name generation
   ├── tirx.Layout                                     Index-to-axis mappings
   │   ├── tirx.TileLayout                             Tile layouts
   │   └── tirx.ComposeLayout                          Composed layouts
   ├── tirx.Iter                                       Layout iterators
   ├── tirx.IndexMap                                   Index transformations
   ├── sym.IterMark                                    Iterator source and extent
   └── ffi.Enum                                        Registered enumeration objects
       ├── tirx.Axis                                   Layout axis identities
       └── tirx.ThreadScope                            Execution scope identities

.. _choose-a-chapter:

What to Read Next
-----------------

The following chapters explain how to work with each part of the IR:

* To understand how values, computations, and tensor shapes are represented,
  see :doc:`types_and_expressions`.
* To build statement bodies with bindings, tensor writes, branches, and loops,
  see :doc:`statements`.
* To work with registered operators and function calls, see
  :doc:`operators_and_calls`.
* To define functions and organize them into modules, see
  :doc:`functions_and_modules`.
* To write or generate TVMScript and print or parse IR, see
  :doc:`printing_and_parsing`.
