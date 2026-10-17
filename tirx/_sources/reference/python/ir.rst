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

tvm_tirx.ir
===========

.. automodule:: tvm_tirx.ir
   :no-members:

.. currentmodule:: tvm_tirx.ir

Common IR infrastructure
------------------------

.. autosummary::
   :toctree: generated/

   Array
   Attrs
   CallEffectKind
   CallSiteLoc
   CallingConv
   DictAttrs
   EnvFunc
   ExprOperand
   ForKind
   GlobalInfo
   IRModule
   Location
   Map
   Node
   Range
   Scriptable
   SourceLoc
   SourceName
   UnknownLoc

Type
----

.. autosummary::
   :toctree: generated/

   AnyType
   FuncType
   MissingType
   OpaqueType
   PrimType
   PtrType
   StringType
   TensorRegionType
   TupleType
   Type

Expr
----

.. autosummary::
   :toctree: generated/

   BaseFunc
   Call
   Constant
   DataTypeImm
   Expr
   ExprWithOp
   FloatImm
   GenericConst
   GlobalVar
   IntImm
   LambdaExpr
   Op
   OpaqueExpr
   StagingExpr
   StringImm
   TensorLoad
   TensorRegion
   Tuple
   TupleGetItem
   Var

Stmt
----

.. autosummary::
   :toctree: generated/

   AssertStmt
   Bind
   Break
   Continue
   Evaluate
   For
   If
   RegionStmt
   Return
   SeqStmt
   Stmt
   TensorStore
   While

Operator functions (tvm_tirx.ir.prim)
-------------------------------------

.. currentmodule:: tvm_tirx.ir.prim

.. autosummary::

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
   copysign
   cos
   cosh
   div
   erf
   exp
   exp10
   exp2
   floor
   floordiv
   floormod
   fma
   fmod
   hypot
   if_then_else
   indexdiv
   indexmod
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
   min
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

.. currentmodule:: tvm_tirx.ir

IR helpers
----------

.. autosummary::
   :toctree: generated/

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

Constants
---------

.. autosummary::
   :toctree: generated/

   UNKNOWN_LOC

Namespaces
----------

.. toctree::
   :maxdepth: 1

   ir.prim
   ir.base
   ir.json_compact
   ir.unique_name_supply

Namespace links
---------------

.. raw:: html

   <span id="tvm-tirx-ir-base"></span>

:doc:`tvm_tirx.ir.base <ir.base>`

.. raw:: html

   <span id="tvm-tirx-ir-prim"></span>

:doc:`tvm_tirx.ir.prim <ir.prim>`

.. raw:: html

   <span id="tvm-tirx-ir-transform"></span>

:doc:`tvm_tirx.ir.transform <ir.transform>`

.. raw:: html

   <span id="tvm-tirx-ir-transform-instrument"></span>

:doc:`tvm_tirx.ir.transform.instrument <ir.transform.instrument>`

.. raw:: html

   <span id="tvm-tirx-ir-unique-name-supply"></span>

:doc:`tvm_tirx.ir.unique_name_supply <ir.unique_name_supply>`

.. raw:: html

   <span id="tvm-tirx-ir-json-compact"></span>

:doc:`tvm_tirx.ir.json_compact <ir.json_compact>`
