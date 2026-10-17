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

tvm_tirx.tirx.script
====================

.. automodule:: tvm_tirx.tirx.script
   :no-members:

.. currentmodule:: tvm_tirx.tirx.script

Classes
-------

.. autosummary::
   :toctree: generated/

   LetAnnotation
   Optional

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Add``
     - :obj:`tvm_tirx.ir.prim.Add`
   * - ``AlreadyEmitted``
     - :obj:`tvm_tirx.script.ir_builder.AlreadyEmitted`
   * - ``And``
     - :obj:`tvm_tirx.ir.prim.And`
   * - ``Any``
     - :obj:`tvm_tirx.ir.AnyType`
   * - ``Axis``
     - :obj:`tvm_tirx.tirx.layout.Axis`
   * - ``BitwiseAnd``
     - :obj:`tvm_tirx.ir.prim.BitwiseAnd`
   * - ``BitwiseNot``
     - :obj:`tvm_tirx.ir.prim.BitwiseNot`
   * - ``BitwiseOr``
     - :obj:`tvm_tirx.ir.prim.BitwiseOr`
   * - ``BitwiseXor``
     - :obj:`tvm_tirx.ir.prim.BitwiseXor`
   * - ``Broadcast``
     - :obj:`tvm_tirx.ir.prim.Broadcast`
   * - ``Call``
     - :obj:`tvm_tirx.ir.Call`
   * - ``CallEffectKind``
     - :obj:`tvm_tirx.ir.CallEffectKind`
   * - ``CallFFIKernelAttr``
     - :obj:`tvm_tirx.tirx.CallFFIKernelAttr`
   * - ``Cast``
     - :obj:`tvm_tirx.ir.prim.Cast`
   * - ``ComposeLayout``
     - :obj:`tvm_tirx.tirx.ComposeLayout`
   * - ``DataTypeImm``
     - :obj:`tvm_tirx.ir.DataTypeImm`
   * - ``Div``
     - :obj:`tvm_tirx.ir.prim.Div`
   * - ``EQ``
     - :obj:`tvm_tirx.ir.prim.EQ`
   * - ``FloatImm``
     - :obj:`tvm_tirx.ir.FloatImm`
   * - ``FloorDiv``
     - :obj:`tvm_tirx.ir.prim.FloorDiv`
   * - ``FloorMod``
     - :obj:`tvm_tirx.ir.prim.FloorMod`
   * - ``Func``
     - :obj:`tvm_tirx.ir.FuncType`
   * - ``FuncType``
     - :obj:`tvm_tirx.ir.FuncType`
   * - ``GE``
     - :obj:`tvm_tirx.ir.prim.GE`
   * - ``GT``
     - :obj:`tvm_tirx.ir.prim.GT`
   * - ``GenericConst``
     - :obj:`tvm_tirx.ir.GenericConst`
   * - ``IRBuilder``
     - :obj:`tvm_tirx.script.ir_builder.IRBuilder`
   * - ``IRModuleFrame``
     - :obj:`tvm_tirx.script.ir_builder.IRModuleFrame`
   * - ``IntImm``
     - :obj:`tvm_tirx.ir.IntImm`
   * - ``Iter``
     - :obj:`tvm_tirx.tirx.layout.Iter`
   * - ``LE``
     - :obj:`tvm_tirx.ir.prim.LE`
   * - ``LShift``
     - :obj:`tvm_tirx.ir.prim.LShift`
   * - ``LT``
     - :obj:`tvm_tirx.ir.prim.LT`
   * - ``Layout``
     - :obj:`tvm_tirx.tirx.Layout`
   * - ``Max``
     - :obj:`tvm_tirx.ir.prim.Max`
   * - ``MemoryScope``
     - :obj:`tvm_tirx.runtime.MemoryScope`
   * - ``Min``
     - :obj:`tvm_tirx.ir.prim.Min`
   * - ``Missing``
     - :obj:`tvm_tirx.ir.MissingType`
   * - ``MissingType``
     - :obj:`tvm_tirx.ir.MissingType`
   * - ``Mod``
     - :obj:`tvm_tirx.ir.prim.Mod`
   * - ``Mul``
     - :obj:`tvm_tirx.ir.prim.Mul`
   * - ``NE``
     - :obj:`tvm_tirx.ir.prim.NE`
   * - ``Not``
     - :obj:`tvm_tirx.ir.prim.Not`
   * - ``Op``
     - :obj:`tvm_tirx.ir.Op`
   * - ``Opaque``
     - :obj:`tvm_tirx.ir.OpaqueType`
   * - ``Or``
     - :obj:`tvm_tirx.ir.prim.Or`
   * - ``Prim``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``PrimType``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``Ptr``
     - :obj:`tvm_tirx.ir.PtrType`
   * - ``RShift``
     - :obj:`tvm_tirx.ir.prim.RShift`
   * - ``Ramp``
     - :obj:`tvm_tirx.ir.prim.Ramp`
   * - ``Range``
     - :obj:`tvm_tirx.ir.Range`
   * - ``Select``
     - :obj:`tvm_tirx.ir.prim.Select`
   * - ``Shuffle``
     - :obj:`tvm_tirx.ir.prim.Shuffle`
   * - ``String``
     - :obj:`tvm_tirx.ir.StringType`
   * - ``StringImm``
     - :obj:`tvm_tirx.ir.StringImm`
   * - ``StringType``
     - :obj:`tvm_tirx.ir.StringType`
   * - ``Sub``
     - :obj:`tvm_tirx.ir.prim.Sub`
   * - ``Tensor``
     - :obj:`tvm_tirx.tirx.TensorType`
   * - ``TensorMap``
     - :obj:`tvm_tirx.tirx.TensorMapType`
   * - ``TensorRegion``
     - :obj:`tvm_tirx.ir.TensorRegion`
   * - ``TensorRegionType``
     - :obj:`tvm_tirx.ir.TensorRegionType`
   * - ``ThreadScope``
     - :obj:`tvm_tirx.tirx.ThreadScope`
   * - ``TileLayout``
     - :obj:`tvm_tirx.tirx.TileLayout`
   * - ``Type``
     - :obj:`tvm_tirx.ir.Type`
   * - ``Var``
     - :obj:`tvm_tirx.ir.Var`
   * - ``broadcast``
     - :obj:`tvm_tirx.ir.prim.Broadcast`
   * - ``ramp``
     - :obj:`tvm_tirx.ir.prim.Ramp`

Functions
---------

.. autosummary::
   :toctree: generated/

   Let
   Tuple
   abi_field_get
   abi_field_set
   abs
   acos
   acosh
   add_to_parent
   address_of
   alloc_local
   alloc_scalar
   alloc_shared
   alloc_tensor
   alloc_workspace
   and_
   arg_
   asin
   asinh
   assert_
   assign
   assume
   assume_aligned
   atan
   atan2
   atanh
   atomic_add
   bf16
   bind
   bind_
   bitwise_and
   bitwise_not
   bitwise_or
   bitwise_xor
   bool
   break_
   call_cpacked
   call_cpacked_lowered
   call_extern
   call_ffi_kernel
   call_global_var_
   call_intrin
   call_kernel
   call_llvm_intrin
   call_llvm_pure_intrin
   call_packed
   call_packed_lowered
   call_pure_extern
   call_spirv_pure_glsl450
   cast
   ceil
   ceildiv
   check_well_formed_
   clz
   compute_scope
   continue_
   copysign
   cos
   cosh
   cpu_parallel_barrier
   decl_mutable_cell_
   decl_scalar
   decl_tensor
   device_context
   device_entry
   device_scope
   else_
   emit
   eq_
   erf
   evaluate
   exp
   exp10
   exp2
   f16
   f32
   f64
   fabs
   float16x16
   float16x2
   float16x32
   float16x4
   float16x64
   float16x8
   float32x16
   float32x2
   float32x32
   float32x4
   float32x64
   float32x8
   float4_e2m1fn
   float4_e2m1fnx16
   float4_e2m1fnx2
   float4_e2m1fnx32
   float4_e2m1fnx4
   float4_e2m1fnx64
   float4_e2m1fnx8
   float64x16
   float64x2
   float64x32
   float64x4
   float64x64
   float64x8
   float6_e2m3fn
   float6_e2m3fnx16
   float6_e2m3fnx2
   float6_e2m3fnx32
   float6_e2m3fnx4
   float6_e2m3fnx64
   float6_e2m3fnx8
   float6_e3m2fn
   float6_e3m2fnx16
   float6_e3m2fnx2
   float6_e3m2fnx32
   float6_e3m2fnx4
   float6_e3m2fnx64
   float6_e3m2fnx8
   float8_e3m4
   float8_e3m4x16
   float8_e3m4x2
   float8_e3m4x32
   float8_e3m4x4
   float8_e3m4x64
   float8_e3m4x8
   float8_e4m3
   float8_e4m3b11fnuz
   float8_e4m3b11fnuzx16
   float8_e4m3b11fnuzx2
   float8_e4m3b11fnuzx32
   float8_e4m3b11fnuzx4
   float8_e4m3b11fnuzx64
   float8_e4m3b11fnuzx8
   float8_e4m3fn
   float8_e4m3fnuz
   float8_e4m3fnuzx16
   float8_e4m3fnuzx2
   float8_e4m3fnuzx32
   float8_e4m3fnuzx4
   float8_e4m3fnuzx64
   float8_e4m3fnuzx8
   float8_e4m3fnx16
   float8_e4m3fnx2
   float8_e4m3fnx32
   float8_e4m3fnx4
   float8_e4m3fnx64
   float8_e4m3fnx8
   float8_e4m3x16
   float8_e4m3x2
   float8_e4m3x32
   float8_e4m3x4
   float8_e4m3x64
   float8_e4m3x8
   float8_e5m2
   float8_e5m2fnuz
   float8_e5m2fnuzx16
   float8_e5m2fnuzx2
   float8_e5m2fnuzx32
   float8_e5m2fnuzx4
   float8_e5m2fnuzx64
   float8_e5m2fnuzx8
   float8_e5m2x16
   float8_e5m2x2
   float8_e5m2x32
   float8_e5m2x4
   float8_e5m2x64
   float8_e5m2x8
   float8_e8m0fnu
   float8_e8m0fnux16
   float8_e8m0fnux2
   float8_e8m0fnux32
   float8_e8m0fnux4
   float8_e8m0fnux64
   float8_e8m0fnux8
   floor
   floordiv
   floormod
   fma
   fmod
   for_
   frame_scope
   free_workspace
   func_attr
   func_name_
   func_ret
   function
   function_
   ge_
   get_active_lane_mask
   gpu_active_thread_selector
   gpu_dp4a
   gpu_fill_fragment
   gpu_load_matrix_sync
   gpu_mma_sync
   gpu_storage_sync
   gpu_store_matrix_sync
   gpu_thread_allreduce
   gpu_thread_filter
   gpu_thread_invariant
   gpu_thread_return
   gpu_warp_activemask
   gpu_warp_shuffle
   gpu_warp_shuffle_down
   gpu_warp_shuffle_up
   gpu_warp_shuffle_xor
   grid
   gt_
   handle
   handle_add_byte_offset
   hypot
   i16
   i32
   i64
   i8
   if_
   if_then_else
   if_then_else_
   ignore_loop_partition
   index_map
   infinity
   inline
   int16x16
   int16x2
   int16x32
   int16x4
   int16x64
   int16x8
   int32x16
   int32x2
   int32x32
   int32x4
   int32x64
   int32x8
   int64x16
   int64x2
   int64x32
   int64x4
   int64x64
   int64x8
   int8x16
   int8x2
   int8x32
   int8x4
   int8x64
   int8x8
   isfinite
   isinf
   isnan
   isnullptr
   jit
   kernel_replace_point
   launch_thread
   ldexp
   le_
   likely
   local_scalar
   log
   log10
   log1p
   log2
   logical_and
   logical_not
   logical_or
   lt_
   macro
   masked_load
   masked_store
   max
   max_value
   meta_class
   min
   min_value
   ne_
   nearbyint
   nextafter
   not_
   or_
   parallel
   parallel_launch
   popcount
   pow
   prefetch
   ptr
   ptr_byte_offset
   range_
   region
   register_script_namespace
   reinterpret
   resolve_global_info_
   return_
   round
   rsqrt
   select
   serial
   set_mutable_cell_
   setattr_
   setitem_
   shared_scalar
   shift_left
   shift_right
   sigmoid
   sin
   sinh
   sqrt
   stack_alloca
   stack_make_dltensor
   stack_make_shape
   static_assert
   static_handle
   tan
   tanh
   target
   tensor_data_ptr
   tensor_indices
   tensor_store
   then_
   throw_last_error
   tmem
   trunc
   truncdiv
   truncmod
   u16
   u32
   u64
   u8
   uint16x16
   uint16x2
   uint16x32
   uint16x4
   uint16x64
   uint16x8
   uint32x16
   uint32x2
   uint32x32
   uint32x4
   uint32x64
   uint32x8
   uint64x16
   uint64x2
   uint64x32
   uint64x4
   uint64x64
   uint64x8
   uint8x16
   uint8x2
   uint8x32
   uint8x4
   uint8x64
   uint8x8
   undef
   unpack_
   unroll
   vector_combine
   vector_high
   vector_low
   vectorized
   void
   vscale
   while_

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Lambda``
     - :obj:`tvm_tirx.script.ir_builder.Lambda`
   * - ``TensorLoad``
     - :obj:`tvm_tirx.tirx.TensorLoad`
   * - ``at_``
     - :obj:`tvm_tirx.script.ir_builder.at_`
   * - ``bfloat16``
     - :obj:`tvm_tirx.tirx.script.bf16`
   * - ``boolean``
     - :obj:`tvm_tirx.tirx.script.bool`
   * - ``constexpr``
     - :obj:`tvm_tirx.script.ir_builder.constexpr`
   * - ``decl_function``
     - :obj:`tvm_tirx.script.ir_builder.decl_function`
   * - ``dtype``
     - :obj:`tvm_tirx.script.ir_builder.dtype`
   * - ``dynamic``
     - :obj:`tvm_tirx.script.ir_builder.dynamic`
   * - ``emit_``
     - :obj:`tvm_tirx.tirx.script.emit`
   * - ``float16``
     - :obj:`tvm_tirx.tirx.script.f16`
   * - ``float32``
     - :obj:`tvm_tirx.tirx.script.f32`
   * - ``float64``
     - :obj:`tvm_tirx.tirx.script.f64`
   * - ``func_ret_type_``
     - :obj:`tvm_tirx.tirx.script.func_ret`
   * - ``int16``
     - :obj:`tvm_tirx.tirx.script.i16`
   * - ``int32``
     - :obj:`tvm_tirx.tirx.script.i32`
   * - ``int64``
     - :obj:`tvm_tirx.tirx.script.i64`
   * - ``int8``
     - :obj:`tvm_tirx.tirx.script.i8`
   * - ``ir_module``
     - :obj:`tvm_tirx.script.parser.ir_module`
   * - ``llvm_lookup_intrinsic_id``
     - :obj:`tvm_tirx.codegen.llvm.lookup_intrinsic_id`
   * - ``make_node``
     - :obj:`tvm_tirx.ir.make_node`
   * - ``meta_var``
     - :obj:`tvm_tirx.script.ir_builder.meta_var`
   * - ``module_attrs``
     - :obj:`tvm_tirx.script.ir_builder.module_attrs`
   * - ``module_get_attr``
     - :obj:`tvm_tirx.script.ir_builder.module_get_attr`
   * - ``module_global_infos``
     - :obj:`tvm_tirx.script.ir_builder.module_global_infos`
   * - ``module_member_``
     - :obj:`tvm_tirx.script.ir_builder.module_member_`
   * - ``module_set_attr``
     - :obj:`tvm_tirx.script.ir_builder.module_set_attr`
   * - ``pyfunc``
     - :obj:`tvm_tirx.script.parser.pyfunc`
   * - ``resolve_global_info_args``
     - :obj:`tvm_tirx.script.ir_builder.resolve_global_info_args`
   * - ``smem``
     - :obj:`tvm_tirx.tirx.script.alloc_shared`
   * - ``uint16``
     - :obj:`tvm_tirx.tirx.script.u16`
   * - ``uint32``
     - :obj:`tvm_tirx.tirx.script.u32`
   * - ``uint64``
     - :obj:`tvm_tirx.tirx.script.u64`
   * - ``uint8``
     - :obj:`tvm_tirx.tirx.script.u8`
   * - ``with_at_group_``
     - :obj:`tvm_tirx.script.ir_builder.with_at_group_`

Namespaces
----------

.. toctree::
   :maxdepth: 1

   tirx.script.ir_builder
   tirx.script.jit
   tirx.script.nvshmem
   tirx.script.ptx
   tirx.script.ptx_legacy
