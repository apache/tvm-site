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

tvm_tirx.tirx
=============

.. automodule:: tvm_tirx.tirx
   :no-members:

.. currentmodule:: tvm_tirx.tirx

Classes
-------

.. autosummary::
   :toctree: generated/

   CallFFIKernelAttr
   ComposeLayout
   Function
   IndexMap
   Layout
   TensorMapType
   TensorType
   ThreadScope
   TileLayout

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Add``
     - :obj:`tvm_tirx.ir.prim.Add`
   * - ``And``
     - :obj:`tvm_tirx.ir.prim.And`
   * - ``BitwiseAnd``
     - :obj:`tvm_tirx.ir.prim.BitwiseAnd`
   * - ``BitwiseNot``
     - :obj:`tvm_tirx.ir.prim.BitwiseNot`
   * - ``BitwiseOr``
     - :obj:`tvm_tirx.ir.prim.BitwiseOr`
   * - ``BitwiseXor``
     - :obj:`tvm_tirx.ir.prim.BitwiseXor`
   * - ``Cast``
     - :obj:`tvm_tirx.ir.prim.Cast`
   * - ``Div``
     - :obj:`tvm_tirx.ir.prim.Div`
   * - ``EQ``
     - :obj:`tvm_tirx.ir.prim.EQ`
   * - ``FloorDiv``
     - :obj:`tvm_tirx.ir.prim.FloorDiv`
   * - ``FloorMod``
     - :obj:`tvm_tirx.ir.prim.FloorMod`
   * - ``GE``
     - :obj:`tvm_tirx.ir.prim.GE`
   * - ``GT``
     - :obj:`tvm_tirx.ir.prim.GT`
   * - ``LE``
     - :obj:`tvm_tirx.ir.prim.LE`
   * - ``LShift``
     - :obj:`tvm_tirx.ir.prim.LShift`
   * - ``LT``
     - :obj:`tvm_tirx.ir.prim.LT`
   * - ``Let``
     - :obj:`tvm_tirx.ir.prim.Let`
   * - ``Max``
     - :obj:`tvm_tirx.ir.prim.Max`
   * - ``Min``
     - :obj:`tvm_tirx.ir.prim.Min`
   * - ``Mod``
     - :obj:`tvm_tirx.ir.prim.Mod`
   * - ``Mul``
     - :obj:`tvm_tirx.ir.prim.Mul`
   * - ``NE``
     - :obj:`tvm_tirx.ir.prim.NE`
   * - ``Not``
     - :obj:`tvm_tirx.ir.prim.Not`
   * - ``Or``
     - :obj:`tvm_tirx.ir.prim.Or`
   * - ``RShift``
     - :obj:`tvm_tirx.ir.prim.RShift`
   * - ``Select``
     - :obj:`tvm_tirx.ir.prim.Select`
   * - ``Sub``
     - :obj:`tvm_tirx.ir.prim.Sub`
   * - ``Var``
     - :obj:`tvm_tirx.ir.Var`

Functions
---------

.. autosummary::
   :toctree: generated/

   TensorLoad
   abi_field_get
   abi_field_set
   address_of
   alloc_workspace
   assume_aligned
   call_cpacked
   call_cpacked_lowered
   call_extern
   call_ffi_kernel
   call_intrin
   call_llvm_intrin
   call_llvm_pure_intrin
   call_packed
   call_packed_lowered
   call_pure_extern
   compile
   cpu_parallel_barrier
   decl_tensor
   free_workspace
   get_active_lane_mask
   get_default_tir_pipeline
   get_tir_pipeline
   get_vscale_expr
   gpu_dp4a
   gpu_fill_fragment
   gpu_load_matrix_sync
   gpu_mma_sync
   gpu_store_matrix_sync
   gpu_thread_allreduce
   handle_add_byte_offset
   ignore_loop_partition
   is_tensor_var
   isnullptr
   logaddexp
   ptr_byte_offset
   register_intrin_lowering
   register_tir_pipeline
   reinterpret
   renew_def
   stack_alloca
   stack_make_dltensor
   stack_make_shape
   tensor_data_ptr
   throw_last_error
   undef
   vector_combine
   vector_high
   vector_low

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``abs``
     - :obj:`tvm_tirx.ir.prim.abs`
   * - ``acos``
     - :obj:`tvm_tirx.ir.prim.acos`
   * - ``acosh``
     - :obj:`tvm_tirx.ir.prim.acosh`
   * - ``all``
     - :obj:`tvm_tirx.ir.prim.all`
   * - ``any``
     - :obj:`tvm_tirx.ir.prim.any`
   * - ``asin``
     - :obj:`tvm_tirx.ir.prim.asin`
   * - ``asinh``
     - :obj:`tvm_tirx.ir.prim.asinh`
   * - ``assume``
     - :obj:`tvm_tirx.ir.prim.assume`
   * - ``atan``
     - :obj:`tvm_tirx.ir.prim.atan`
   * - ``atan2``
     - :obj:`tvm_tirx.ir.prim.atan2`
   * - ``atanh``
     - :obj:`tvm_tirx.ir.prim.atanh`
   * - ``bitwise_and``
     - :obj:`tvm_tirx.ir.prim.bitwise_and`
   * - ``bitwise_not``
     - :obj:`tvm_tirx.ir.prim.bitwise_not`
   * - ``bitwise_or``
     - :obj:`tvm_tirx.ir.prim.bitwise_or`
   * - ``bitwise_xor``
     - :obj:`tvm_tirx.ir.prim.bitwise_xor`
   * - ``ceil``
     - :obj:`tvm_tirx.ir.prim.ceil`
   * - ``ceildiv``
     - :obj:`tvm_tirx.ir.prim.ceildiv`
   * - ``clz``
     - :obj:`tvm_tirx.ir.prim.clz`
   * - ``convert``
     - :obj:`tvm_tirx.ir.prim.convert`
   * - ``copysign``
     - :obj:`tvm_tirx.ir.prim.copysign`
   * - ``cos``
     - :obj:`tvm_tirx.ir.prim.cos`
   * - ``cosh``
     - :obj:`tvm_tirx.ir.prim.cosh`
   * - ``div``
     - :obj:`tvm_tirx.ir.prim.div`
   * - ``erf``
     - :obj:`tvm_tirx.ir.prim.erf`
   * - ``exp``
     - :obj:`tvm_tirx.ir.prim.exp`
   * - ``exp10``
     - :obj:`tvm_tirx.ir.prim.exp10`
   * - ``exp2``
     - :obj:`tvm_tirx.ir.prim.exp2`
   * - ``floor``
     - :obj:`tvm_tirx.ir.prim.floor`
   * - ``floordiv``
     - :obj:`tvm_tirx.ir.prim.floordiv`
   * - ``floormod``
     - :obj:`tvm_tirx.ir.prim.floormod`
   * - ``fmod``
     - :obj:`tvm_tirx.ir.prim.fmod`
   * - ``hypot``
     - :obj:`tvm_tirx.ir.prim.hypot`
   * - ``if_then_else``
     - :obj:`tvm_tirx.ir.prim.if_then_else`
   * - ``indexdiv``
     - :obj:`tvm_tirx.ir.prim.indexdiv`
   * - ``indexmod``
     - :obj:`tvm_tirx.ir.prim.indexmod`
   * - ``infinity``
     - :obj:`tvm_tirx.ir.prim.infinity`
   * - ``isfinite``
     - :obj:`tvm_tirx.ir.prim.isfinite`
   * - ``isinf``
     - :obj:`tvm_tirx.ir.prim.isinf`
   * - ``isnan``
     - :obj:`tvm_tirx.ir.prim.isnan`
   * - ``ldexp``
     - :obj:`tvm_tirx.ir.prim.ldexp`
   * - ``likely``
     - :obj:`tvm_tirx.ir.prim.likely`
   * - ``log``
     - :obj:`tvm_tirx.ir.prim.log`
   * - ``log10``
     - :obj:`tvm_tirx.ir.prim.log10`
   * - ``log1p``
     - :obj:`tvm_tirx.ir.prim.log1p`
   * - ``log2``
     - :obj:`tvm_tirx.ir.prim.log2`
   * - ``max``
     - :obj:`tvm_tirx.ir.prim.max`
   * - ``max_value``
     - :obj:`tvm_tirx.ir.prim.max_value`
   * - ``min``
     - :obj:`tvm_tirx.ir.prim.min`
   * - ``min_value``
     - :obj:`tvm_tirx.ir.prim.min_value`
   * - ``nearbyint``
     - :obj:`tvm_tirx.ir.prim.nearbyint`
   * - ``nextafter``
     - :obj:`tvm_tirx.ir.prim.nextafter`
   * - ``popcount``
     - :obj:`tvm_tirx.ir.prim.popcount`
   * - ``pow``
     - :obj:`tvm_tirx.ir.prim.pow`
   * - ``power``
     - :obj:`tvm_tirx.ir.prim.power`
   * - ``round``
     - :obj:`tvm_tirx.ir.prim.round`
   * - ``rsqrt``
     - :obj:`tvm_tirx.ir.prim.rsqrt`
   * - ``shift_left``
     - :obj:`tvm_tirx.ir.prim.shift_left`
   * - ``shift_right``
     - :obj:`tvm_tirx.ir.prim.shift_right`
   * - ``sigmoid``
     - :obj:`tvm_tirx.ir.prim.sigmoid`
   * - ``sin``
     - :obj:`tvm_tirx.ir.prim.sin`
   * - ``sinh``
     - :obj:`tvm_tirx.ir.prim.sinh`
   * - ``sqrt``
     - :obj:`tvm_tirx.ir.prim.sqrt`
   * - ``tan``
     - :obj:`tvm_tirx.ir.prim.tan`
   * - ``tanh``
     - :obj:`tvm_tirx.ir.prim.tanh`
   * - ``trunc``
     - :obj:`tvm_tirx.ir.prim.trunc`
   * - ``truncdiv``
     - :obj:`tvm_tirx.ir.prim.truncdiv`
   * - ``truncmod``
     - :obj:`tvm_tirx.ir.prim.truncmod`
   * - ``vscale``
     - :obj:`tvm_tirx.ir.prim.vscale`

Namespaces
----------

.. toctree::
   :maxdepth: 1

   tirx.analysis
   tirx.tile
   tirx.script
   tirx.bench
   tirx.compilation_pipeline
   tirx.layout
   tirx.op

Namespace links
---------------

.. raw:: html

   <span id="tvm-tirx-tirx-op"></span>

:doc:`tvm_tirx.tirx.op <tirx.op>`

.. raw:: html

   <span id="tvm-tirx-tirx-analysis"></span>

:doc:`tvm_tirx.tirx.analysis <tirx.analysis>`

.. raw:: html

   <span id="tvm-tirx-tirx-transform"></span>

:doc:`tvm_tirx.tirx.transform <tirx.transform>`

.. raw:: html

   <span id="tvm-tirx-tirx-layout"></span>

:doc:`tvm_tirx.tirx.layout <tirx.layout>`

.. raw:: html

   <span id="tvm-tirx-tirx-compilation-pipeline"></span>

:doc:`tvm_tirx.tirx.compilation_pipeline <tirx.compilation_pipeline>`

.. raw:: html

   <span id="tvm-tirx-tirx-tile"></span>

:doc:`tvm_tirx.tirx.tile <tirx.tile>`

.. raw:: html

   <span id="tvm-tirx-tirx-tile-dispatch"></span>

:doc:`tvm_tirx.tirx.tile.dispatch <tirx.tile.dispatch>`

.. raw:: html

   <span id="tvm-tirx-tirx-tile-instruction"></span>

:doc:`tvm_tirx.tirx.tile.instruction <tirx.tile.instruction>`

.. raw:: html

   <span id="tvm-tirx-tirx-tile-views"></span>

:doc:`tvm_tirx.tirx.tile.views <tirx.tile.views>`

.. raw:: html

   <span id="tvm-tirx-tirx-bench"></span>

:doc:`tvm_tirx.tirx.bench <tirx.bench>`
