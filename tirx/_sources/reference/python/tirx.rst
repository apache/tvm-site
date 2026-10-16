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

TIRx
====

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.tirx
-------------

.. currentmodule:: tvm_tirx.tirx

.. autosummary::
   :toctree: generated/

   CallFFIKernelAttr
   ComposeLayout
   Function
   IndexMap
   Layout
   TensorLoad
   TensorMapType
   TensorType
   ThreadScope
   TileLayout
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

* :obj:`Add <tvm_tirx.ir.prim.Add>`
* :obj:`And <tvm_tirx.ir.prim.And>`
* :obj:`BitwiseAnd <tvm_tirx.ir.prim.BitwiseAnd>`
* :obj:`BitwiseNot <tvm_tirx.ir.prim.BitwiseNot>`
* :obj:`BitwiseOr <tvm_tirx.ir.prim.BitwiseOr>`
* :obj:`BitwiseXor <tvm_tirx.ir.prim.BitwiseXor>`
* :obj:`Cast <tvm_tirx.ir.prim.Cast>`
* :obj:`Div <tvm_tirx.ir.prim.Div>`
* :obj:`EQ <tvm_tirx.ir.prim.EQ>`
* :obj:`FloorDiv <tvm_tirx.ir.prim.FloorDiv>`
* :obj:`FloorMod <tvm_tirx.ir.prim.FloorMod>`
* :obj:`GE <tvm_tirx.ir.prim.GE>`
* :obj:`GT <tvm_tirx.ir.prim.GT>`
* :obj:`LE <tvm_tirx.ir.prim.LE>`
* :obj:`LShift <tvm_tirx.ir.prim.LShift>`
* :obj:`LT <tvm_tirx.ir.prim.LT>`
* :obj:`Let <tvm_tirx.ir.prim.Let>`
* :obj:`Max <tvm_tirx.ir.prim.Max>`
* :obj:`Min <tvm_tirx.ir.prim.Min>`
* :obj:`Mod <tvm_tirx.ir.prim.Mod>`
* :obj:`Mul <tvm_tirx.ir.prim.Mul>`
* :obj:`NE <tvm_tirx.ir.prim.NE>`
* :obj:`Not <tvm_tirx.ir.prim.Not>`
* :obj:`Or <tvm_tirx.ir.prim.Or>`
* :obj:`RShift <tvm_tirx.ir.prim.RShift>`
* :obj:`Select <tvm_tirx.ir.prim.Select>`
* :obj:`Sub <tvm_tirx.ir.prim.Sub>`
* :obj:`Var <tvm_tirx.ir.Var>`
* :obj:`abs <tvm_tirx.ir.prim.abs>`
* :obj:`acos <tvm_tirx.ir.prim.acos>`
* :obj:`acosh <tvm_tirx.ir.prim.acosh>`
* :obj:`all <tvm_tirx.ir.prim.all>`
* :obj:`any <tvm_tirx.ir.prim.any>`
* :obj:`asin <tvm_tirx.ir.prim.asin>`
* :obj:`asinh <tvm_tirx.ir.prim.asinh>`
* :obj:`assume <tvm_tirx.ir.prim.assume>`
* :obj:`atan <tvm_tirx.ir.prim.atan>`
* :obj:`atan2 <tvm_tirx.ir.prim.atan2>`
* :obj:`atanh <tvm_tirx.ir.prim.atanh>`
* :obj:`bitwise_and <tvm_tirx.ir.prim.bitwise_and>`
* :obj:`bitwise_not <tvm_tirx.ir.prim.bitwise_not>`
* :obj:`bitwise_or <tvm_tirx.ir.prim.bitwise_or>`
* :obj:`bitwise_xor <tvm_tirx.ir.prim.bitwise_xor>`
* :obj:`ceil <tvm_tirx.ir.prim.ceil>`
* :obj:`ceildiv <tvm_tirx.ir.prim.ceildiv>`
* :obj:`clz <tvm_tirx.ir.prim.clz>`
* :obj:`convert <tvm_tirx.ir.prim.convert>`
* :obj:`copysign <tvm_tirx.ir.prim.copysign>`
* :obj:`cos <tvm_tirx.ir.prim.cos>`
* :obj:`cosh <tvm_tirx.ir.prim.cosh>`
* :obj:`div <tvm_tirx.ir.prim.div>`
* :obj:`erf <tvm_tirx.ir.prim.erf>`
* :obj:`exp <tvm_tirx.ir.prim.exp>`
* :obj:`exp10 <tvm_tirx.ir.prim.exp10>`
* :obj:`exp2 <tvm_tirx.ir.prim.exp2>`
* :obj:`floor <tvm_tirx.ir.prim.floor>`
* :obj:`floordiv <tvm_tirx.ir.prim.floordiv>`
* :obj:`floormod <tvm_tirx.ir.prim.floormod>`
* :obj:`fmod <tvm_tirx.ir.prim.fmod>`
* :obj:`hypot <tvm_tirx.ir.prim.hypot>`
* :obj:`if_then_else <tvm_tirx.ir.prim.if_then_else>`
* :obj:`indexdiv <tvm_tirx.ir.prim.indexdiv>`
* :obj:`indexmod <tvm_tirx.ir.prim.indexmod>`
* :obj:`infinity <tvm_tirx.ir.prim.infinity>`
* :obj:`isfinite <tvm_tirx.ir.prim.isfinite>`
* :obj:`isinf <tvm_tirx.ir.prim.isinf>`
* :obj:`isnan <tvm_tirx.ir.prim.isnan>`
* :obj:`ldexp <tvm_tirx.ir.prim.ldexp>`
* :obj:`likely <tvm_tirx.ir.prim.likely>`
* :obj:`log <tvm_tirx.ir.prim.log>`
* :obj:`log10 <tvm_tirx.ir.prim.log10>`
* :obj:`log1p <tvm_tirx.ir.prim.log1p>`
* :obj:`log2 <tvm_tirx.ir.prim.log2>`
* :obj:`max <tvm_tirx.ir.prim.max>`
* :obj:`max_value <tvm_tirx.ir.prim.max_value>`
* :obj:`min <tvm_tirx.ir.prim.min>`
* :obj:`min_value <tvm_tirx.ir.prim.min_value>`
* :obj:`nearbyint <tvm_tirx.ir.prim.nearbyint>`
* :obj:`nextafter <tvm_tirx.ir.prim.nextafter>`
* :obj:`popcount <tvm_tirx.ir.prim.popcount>`
* :obj:`pow <tvm_tirx.ir.prim.pow>`
* :obj:`power <tvm_tirx.ir.prim.power>`
* :obj:`round <tvm_tirx.ir.prim.round>`
* :obj:`rsqrt <tvm_tirx.ir.prim.rsqrt>`
* :obj:`shift_left <tvm_tirx.ir.prim.shift_left>`
* :obj:`shift_right <tvm_tirx.ir.prim.shift_right>`
* :obj:`sigmoid <tvm_tirx.ir.prim.sigmoid>`
* :obj:`sin <tvm_tirx.ir.prim.sin>`
* :obj:`sinh <tvm_tirx.ir.prim.sinh>`
* :obj:`sqrt <tvm_tirx.ir.prim.sqrt>`
* :obj:`tan <tvm_tirx.ir.prim.tan>`
* :obj:`tanh <tvm_tirx.ir.prim.tanh>`
* :obj:`trunc <tvm_tirx.ir.prim.trunc>`
* :obj:`truncdiv <tvm_tirx.ir.prim.truncdiv>`
* :obj:`truncmod <tvm_tirx.ir.prim.truncmod>`
* :obj:`vscale <tvm_tirx.ir.prim.vscale>`

tvm_tirx.tirx.op
----------------

.. currentmodule:: tvm_tirx.tirx.op

.. autosummary::
   :toctree: generated/

   gpu_active_thread_selector
   gpu_storage_sync
   gpu_thread_filter
   gpu_thread_invariant
   gpu_thread_return
   gpu_warp_activemask
   gpu_warp_shuffle
   gpu_warp_shuffle_down
   gpu_warp_shuffle_up
   gpu_warp_shuffle_xor
   kernel_replace_point
   masked_load
   masked_store

* :obj:`CallFFIKernelAttr <tvm_tirx.tirx.CallFFIKernelAttr>`
* :obj:`TensorLoad <tvm_tirx.ir.TensorLoad>`
* :obj:`TensorMapType <tvm_tirx.tirx.TensorMapType>`
* :obj:`abi_field_get <tvm_tirx.tirx.abi_field_get>`
* :obj:`abi_field_set <tvm_tirx.tirx.abi_field_set>`
* :obj:`address_of <tvm_tirx.tirx.address_of>`
* :obj:`alloc_workspace <tvm_tirx.tirx.alloc_workspace>`
* :obj:`assume_aligned <tvm_tirx.tirx.assume_aligned>`
* :obj:`call_cpacked <tvm_tirx.tirx.call_cpacked>`
* :obj:`call_cpacked_lowered <tvm_tirx.tirx.call_cpacked_lowered>`
* :obj:`call_extern <tvm_tirx.tirx.call_extern>`
* :obj:`call_ffi_kernel <tvm_tirx.tirx.call_ffi_kernel>`
* :obj:`call_intrin <tvm_tirx.tirx.call_intrin>`
* :obj:`call_llvm_intrin <tvm_tirx.tirx.call_llvm_intrin>`
* :obj:`call_llvm_pure_intrin <tvm_tirx.tirx.call_llvm_pure_intrin>`
* :obj:`call_packed <tvm_tirx.tirx.call_packed>`
* :obj:`call_packed_lowered <tvm_tirx.tirx.call_packed_lowered>`
* :obj:`call_pure_extern <tvm_tirx.tirx.call_pure_extern>`
* :obj:`cpu_parallel_barrier <tvm_tirx.tirx.cpu_parallel_barrier>`
* :obj:`free_workspace <tvm_tirx.tirx.free_workspace>`
* :obj:`get_active_lane_mask <tvm_tirx.tirx.get_active_lane_mask>`
* :obj:`get_vscale_expr <tvm_tirx.tirx.get_vscale_expr>`
* :obj:`gpu_dp4a <tvm_tirx.tirx.gpu_dp4a>`
* :obj:`gpu_fill_fragment <tvm_tirx.tirx.gpu_fill_fragment>`
* :obj:`gpu_load_matrix_sync <tvm_tirx.tirx.gpu_load_matrix_sync>`
* :obj:`gpu_mma_sync <tvm_tirx.tirx.gpu_mma_sync>`
* :obj:`gpu_store_matrix_sync <tvm_tirx.tirx.gpu_store_matrix_sync>`
* :obj:`gpu_thread_allreduce <tvm_tirx.tirx.gpu_thread_allreduce>`
* :obj:`handle_add_byte_offset <tvm_tirx.tirx.handle_add_byte_offset>`
* :obj:`ignore_loop_partition <tvm_tirx.tirx.ignore_loop_partition>`
* :obj:`is_tensor_var <tvm_tirx.tirx.is_tensor_var>`
* :obj:`isnullptr <tvm_tirx.tirx.isnullptr>`
* :obj:`logaddexp <tvm_tirx.tirx.logaddexp>`
* :obj:`ptr_byte_offset <tvm_tirx.tirx.ptr_byte_offset>`
* :obj:`register_intrin_lowering <tvm_tirx.tirx.register_intrin_lowering>`
* :obj:`reinterpret <tvm_tirx.tirx.reinterpret>`
* :obj:`stack_alloca <tvm_tirx.tirx.stack_alloca>`
* :obj:`stack_make_dltensor <tvm_tirx.tirx.stack_make_dltensor>`
* :obj:`stack_make_shape <tvm_tirx.tirx.stack_make_shape>`
* :obj:`tensor_data_ptr <tvm_tirx.tirx.tensor_data_ptr>`
* :obj:`throw_last_error <tvm_tirx.tirx.throw_last_error>`
* :obj:`undef <tvm_tirx.tirx.undef>`
* :obj:`vector_combine <tvm_tirx.tirx.vector_combine>`
* :obj:`vector_high <tvm_tirx.tirx.vector_high>`
* :obj:`vector_low <tvm_tirx.tirx.vector_low>`

tvm_tirx.tirx.analysis
----------------------

.. currentmodule:: tvm_tirx.tirx.analysis

.. autosummary::
   :toctree: generated/

   undefined_vars
   verify_memory
   verify_well_formed


tvm_tirx.tirx.transform
-----------------------

.. currentmodule:: tvm_tirx.tirx.transform

.. autosummary::
   :toctree: generated/

   AnnotateEntryFunc
   Apply
   BF16ComputeLegalize
   BF16StorageLegalize
   BindTarget
   CommonSubexprElim
   ConvertSSA
   DecorateDeviceScope
   FP8ComputeLegalize
   FP8StorageLegalize
   Filter
   FlattenTensor
   ForceNarrowIndexToInt32
   FunctionPass
   HoistIf
   LowerIntrin
   LowerTIRx
   LowerTVMBuiltin
   LowerThreadAllreduce
   MakePackedAPI
   NarrowDataType
   PointerValueTypeRewrite
   RemoveNoOp
   RemoveNoOpConfig
   RenormalizeSplitPattern
   SkipAssert
   SplitHostDevice
   StmtSimplify
   StorageRewrite
   TileDispatch
   UnrollLoop
   UnrollLoopConfig
   VectorizeLoop
   VerifyMemory
   function_pass


tvm_tirx.tirx.layout
--------------------

.. currentmodule:: tvm_tirx.tirx.layout

.. autosummary::
   :toctree: generated/

   Axis
   Iter

* :obj:`ComposeLayout <tvm_tirx.tirx.ComposeLayout>`
* :obj:`Layout <tvm_tirx.tirx.Layout>`
* :obj:`TileLayout <tvm_tirx.tirx.TileLayout>`

tvm_tirx.tirx.compilation_pipeline
----------------------------------

.. currentmodule:: tvm_tirx.tirx.compilation_pipeline

.. autosummary::
   :toctree: generated/

   default_tir_pipeline
   finalize_device_passes
   finalize_device_passes_tirx
   finalize_host_passes
   finalize_tir_pipeline
   register_default_tir_pipeline_selector
   tirx_pipeline

* :obj:`get_default_tir_pipeline <tvm_tirx.tirx.get_default_tir_pipeline>`
* :obj:`get_tir_pipeline <tvm_tirx.tirx.get_tir_pipeline>`
* :obj:`register_tir_pipeline <tvm_tirx.tirx.register_tir_pipeline>`

tvm_tirx.tirx.tile
------------------

.. currentmodule:: tvm_tirx.tirx.tile

.. autosummary::
   :toctree: generated/

   DispatchContext
   Instruction
   Operand
   TensorCall
   fail


tvm_tirx.tirx.tile.dispatch
---------------------------

.. currentmodule:: tvm_tirx.tirx.tile.dispatch

.. autosummary::
   :toctree: generated/

   DispatchFail
   run_dispatch

* :obj:`DispatchContext <tvm_tirx.tirx.tile.DispatchContext>`
* :obj:`fail <tvm_tirx.tirx.tile.fail>`

tvm_tirx.tirx.tile.instruction
------------------------------

.. currentmodule:: tvm_tirx.tirx.tile.instruction

.. autosummary::
   :toctree: generated/

   namespace

* :obj:`Instruction <tvm_tirx.tirx.tile.Instruction>`
* :obj:`Operand <tvm_tirx.tirx.tile.Operand>`
* :obj:`TensorCall <tvm_tirx.tirx.tile.TensorCall>`

tvm_tirx.tirx.tile.views
------------------------

.. currentmodule:: tvm_tirx.tirx.tile.views

.. autosummary::
   :toctree: generated/

   Add
   BinaryChain
   BinaryOp
   BinaryReduce
   Cast
   Copy
   CopyAsync
   Exp
   Exp2
   Exp2WithScaleBias
   ExpWithScaleBias
   FDiv
   FMA
   Fill
   Gemm
   GemmAsync
   Log2
   Log2WithScaleBias
   Max
   Maximum
   Memset
   Min
   Minimum
   Mul
   Reciprocal
   ReduceNegate
   ReduceOp
   Select
   SiLU
   Sqrt
   SqrtWithScaleBias
   Sub
   Sum
   UnaryOp
   UnaryOpWithScaleBias
   UnaryReduce
   UnaryReduceWithScaleBias
   Zero


tvm_tirx.tirx.bench
-------------------

.. currentmodule:: tvm_tirx.tirx.bench

.. autosummary::
   :toctree: generated/

   CudaProfiler
   DistributedBenchContext
   EventType
   bench
   decode_tag
   export_to_perfetto_trace
   is_running_under_pytest
   setup
