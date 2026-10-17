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

tvm_tirx.ir.prim
================

.. automodule:: tvm_tirx.ir.prim
   :no-members:

.. currentmodule:: tvm_tirx.ir.prim

Classes
-------

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

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Expr``
     - :obj:`tvm_tirx.ir.Expr`

Functions
---------

.. autosummary::
   :toctree: generated/

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

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``const``
     - :obj:`tvm_tirx.ir.const`
