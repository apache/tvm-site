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

IR
==

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.ir
-----------

.. currentmodule:: tvm_tirx.ir

.. autosummary::
   :toctree: generated/

   AnyType
   Array
   AssertStmt
   Attrs
   BaseFunc
   Bind
   Break
   Call
   CallEffectKind
   CallSiteLoc
   CallingConv
   Constant
   Continue
   DataTypeImm
   DictAttrs
   EnvFunc
   Evaluate
   Expr
   ExprOperand
   ExprWithOp
   FloatImm
   For
   ForKind
   FuncType
   GenericConst
   GlobalInfo
   GlobalVar
   IRModule
   If
   IntImm
   LambdaExpr
   Location
   Map
   MissingType
   Node
   Op
   OpaqueExpr
   OpaqueType
   PrimType
   PtrType
   Range
   RegionStmt
   Return
   Scriptable
   SeqStmt
   SourceLoc
   SourceName
   StagingExpr
   Stmt
   StringImm
   StringType
   TensorLoad
   TensorRegion
   TensorRegionType
   TensorStore
   Tuple
   TupleGetItem
   TupleType
   Type
   UNKNOWN_LOC
   UnknownLoc
   Var
   While
   assert_structural_equal
   const
   is_prim_expr
   is_prim_var
   load_json
   make_node
   register_op_attr
   reinfer_type
   save_json
   stmt_list
   stmt_seq


tvm_tirx.ir.base
----------------

.. currentmodule:: tvm_tirx.ir.base

.. autosummary::
   :toctree: generated/

   register_dialect_script_method
   register_script_printer

* :obj:`EnvFunc <tvm_tirx.ir.EnvFunc>`
* :obj:`Node <tvm_tirx.ir.Node>`
* :obj:`Scriptable <tvm_tirx.ir.Scriptable>`
* :obj:`assert_structural_equal <tvm_tirx.ir.assert_structural_equal>`
* :obj:`load_json <tvm_tirx.ir.load_json>`
* :obj:`save_json <tvm_tirx.ir.save_json>`

tvm_tirx.ir.prim
----------------

.. currentmodule:: tvm_tirx.ir.prim

.. autosummary::
   :toctree: generated/

   Add
   And
   BinaryOpExpr
   BitwiseAnd
   BitwiseNot
   BitwiseOr
   BitwiseXor
   Broadcast
   Cast
   CmpExpr
   Div
   EQ
   FloorDiv
   FloorMod
   GE
   GT
   LE
   LShift
   LT
   Let
   LogicalExpr
   Max
   Min
   Mod
   Mul
   NE
   Not
   Or
   RShift
   Ramp
   Select
   Shuffle
   Sub
   abs
   acos
   acosh
   all
   any
   asin
   asinh
   assume
   atan
   atan2
   atanh
   bitwise_and
   bitwise_not
   bitwise_or
   bitwise_xor
   ceil
   ceildiv
   clz
   convert
   copysign
   cos
   cosh
   div
   erf
   exp
   exp10
   exp2
   expr_deep_equal
   floor
   floordiv
   floormod
   fma
   fmod
   hypot
   if_then_else
   indexdiv
   indexmod
   infinity
   isfinite
   isinf
   isnan
   ldexp
   likely
   log
   log10
   log1p
   log2
   max
   max_value
   min
   min_value
   nearbyint
   nextafter
   popcount
   pow
   power
   round
   rsqrt
   shift_left
   shift_right
   sigmoid
   sin
   sinh
   sqrt
   tan
   tanh
   trunc
   truncdiv
   truncmod
   vscale

* :obj:`Expr <tvm_tirx.ir.Expr>`
* :obj:`const <tvm_tirx.ir.const>`

tvm_tirx.ir.transform
---------------------

.. currentmodule:: tvm_tirx.ir.transform

.. autosummary::
   :toctree: generated/

   ModulePass
   Pass
   PassContext
   PassInfo
   PassInstrument
   PrintIR
   Sequential
   module_pass
   pass_instrument


tvm_tirx.ir.transform.instrument
--------------------------------

.. currentmodule:: tvm_tirx.ir.transform.instrument

.. autosummary::
   :toctree: generated/

   DumpIR
   PassPrintingInstrument
   PassTimingInstrument
   PrintAfterAll
   PrintBeforeAll


tvm_tirx.ir.unique_name_supply
------------------------------

.. currentmodule:: tvm_tirx.ir.unique_name_supply

.. autosummary::
   :toctree: generated/

   UniqueNameSupply


tvm_tirx.ir.json_compact
------------------------

.. currentmodule:: tvm_tirx.ir.json_compact

.. autosummary::
   :toctree: generated/

   create_updater
   get_version
   upgrade_json
