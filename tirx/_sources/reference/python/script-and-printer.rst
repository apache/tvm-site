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

Script and Printer
==================

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.script
---------------

.. currentmodule:: tvm_tirx.script

.. autosummary::
   :toctree: generated/

   register_dialect
   register_module_validator


tvm_tirx.script.parser
----------------------

.. currentmodule:: tvm_tirx.script.parser

.. autosummary::
   :toctree: generated/

   from_source
   ir_module
   make_decorator
   make_macro_decorator
   pyfunc
   register_namespace
   register_namespace_initializer

* :obj:`parse <tvm_tirx.script.parser.from_source>`

tvm_tirx.script.parser.protocol_registry
----------------------------------------

.. currentmodule:: tvm_tirx.script.parser.protocol_registry

.. autosummary::
   :toctree: generated/

   register_mutable_decl
   register_scalar_annotation


tvm_tirx.script.ir_builder
--------------------------

.. currentmodule:: tvm_tirx.script.ir_builder

.. autosummary::
   :toctree: generated/

   AlreadyEmitted
   IRBuilder
   IRModuleFrame
   Lambda
   at_
   check_well_formed_
   constexpr
   decl_function
   dtype
   dynamic
   ir_module
   meta_var
   module_attrs
   module_get_attr
   module_global_infos
   module_member_
   module_set_attr
   resolve_global_info_
   resolve_global_info_args
   with_at_group_

* :obj:`Any <tvm_tirx.ir.AnyType>`
* :obj:`Call <tvm_tirx.ir.Call>`
* :obj:`DataTypeImm <tvm_tirx.ir.DataTypeImm>`
* :obj:`Func <tvm_tirx.ir.FuncType>`
* :obj:`FuncType <tvm_tirx.ir.FuncType>`
* :obj:`GenericConst <tvm_tirx.ir.GenericConst>`
* :obj:`Missing <tvm_tirx.ir.MissingType>`
* :obj:`MissingType <tvm_tirx.ir.MissingType>`
* :obj:`Op <tvm_tirx.ir.Op>`
* :obj:`Opaque <tvm_tirx.ir.OpaqueType>`
* :obj:`Prim <tvm_tirx.ir.PrimType>`
* :obj:`PrimType <tvm_tirx.ir.PrimType>`
* :obj:`Ptr <tvm_tirx.ir.PtrType>`
* :obj:`Range <tvm_tirx.ir.Range>`
* :obj:`String <tvm_tirx.ir.StringType>`
* :obj:`StringImm <tvm_tirx.ir.StringImm>`
* :obj:`StringType <tvm_tirx.ir.StringType>`
* :obj:`TensorRegion <tvm_tirx.ir.TensorRegion>`
* :obj:`TensorRegionType <tvm_tirx.ir.TensorRegionType>`
* :obj:`Tuple <tvm_tirx.ir.Tuple>`
* :obj:`Type <tvm_tirx.ir.Type>`
* :obj:`Var <tvm_tirx.ir.Var>`
* :obj:`make_node <tvm_tirx.ir.make_node>`

tvm_tirx.script.ir_builder.frame
--------------------------------

.. currentmodule:: tvm_tirx.script.ir_builder.frame

.. autosummary::
   :toctree: generated/

   AssertFrame
   ElseFrame
   ForFrame
   IfFrame
   RegionFrame
   StmtFrame
   ThenFrame
   WhileFrame

* :obj:`IRModuleFrame <tvm_tirx.script.ir_builder.IRModuleFrame>`

tvm_tirx.script.ir_builder.ir
-----------------------------

.. currentmodule:: tvm_tirx.script.ir_builder.ir

* :obj:`Lambda <tvm_tirx.script.ir_builder.Lambda>`
* :obj:`constexpr <tvm_tirx.script.ir_builder.constexpr>`
* :obj:`dtype <tvm_tirx.script.ir_builder.dtype>`
* :obj:`dynamic <tvm_tirx.script.ir_builder.dynamic>`
* :obj:`meta_var <tvm_tirx.script.ir_builder.meta_var>`
* :obj:`module_attrs <tvm_tirx.script.ir_builder.module_attrs>`
* :obj:`module_get_attr <tvm_tirx.script.ir_builder.module_get_attr>`
* :obj:`module_global_infos <tvm_tirx.script.ir_builder.module_global_infos>`
* :obj:`module_set_attr <tvm_tirx.script.ir_builder.module_set_attr>`

tvm_tirx.script.ir_builder.stmt
-------------------------------

.. currentmodule:: tvm_tirx.script.ir_builder.stmt

.. autosummary::
   :toctree: generated/

   bind

* :obj:`add_to_parent <tvm_tirx.tirx.script.add_to_parent>`
* :obj:`assert_ <tvm_tirx.tirx.script.assert_>`
* :obj:`break_ <tvm_tirx.tirx.script.break_>`
* :obj:`continue_ <tvm_tirx.tirx.script.continue_>`
* :obj:`else_ <tvm_tirx.tirx.script.else_>`
* :obj:`evaluate <tvm_tirx.tirx.script.evaluate>`
* :obj:`frame_scope <tvm_tirx.tirx.script.frame_scope>`
* :obj:`grid <tvm_tirx.tirx.script.grid>`
* :obj:`if_ <tvm_tirx.tirx.script.if_>`
* :obj:`region <tvm_tirx.tirx.script.region>`
* :obj:`return_ <tvm_tirx.tirx.script.return_>`
* :obj:`then_ <tvm_tirx.tirx.script.then_>`
* :obj:`unpack_ <tvm_tirx.tirx.script.unpack_>`
* :obj:`while_ <tvm_tirx.tirx.script.while_>`

tvm_tirx.script.ir_builder.tracing
----------------------------------

.. currentmodule:: tvm_tirx.script.ir_builder.tracing

.. autosummary::
   :toctree: generated/

   LineTraceScope


tvm_tirx.script.printer
-----------------------

.. currentmodule:: tvm_tirx.script.printer

.. autosummary::
   :toctree: generated/

   PrinterConfig


tvm_tirx.script.printer.doc
---------------------------

.. currentmodule:: tvm_tirx.script.printer.doc

.. autosummary::
   :toctree: generated/

   AssertDoc
   AssignDoc
   AttrAccessDoc
   CallDoc
   ClassDoc
   CommentDoc
   DictDoc
   Doc
   DocStringDoc
   ExprDoc
   ExprStmtDoc
   ForDoc
   FunctionDoc
   IdDoc
   IfDoc
   IndexDoc
   LambdaDoc
   ListDoc
   LiteralDoc
   OperationDoc
   OperationKind
   ReturnDoc
   ScopeDoc
   SliceDoc
   StmtBlockDoc
   StmtDoc
   TupleDoc
   WhileDoc


tvm_tirx.tirx.script
--------------------

.. currentmodule:: tvm_tirx.tirx.script

.. autosummary::
   :toctree: generated/

   Let
   LetAnnotation
   Optional
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

* :obj:`Add <tvm_tirx.ir.prim.Add>`
* :obj:`AlreadyEmitted <tvm_tirx.script.ir_builder.AlreadyEmitted>`
* :obj:`And <tvm_tirx.ir.prim.And>`
* :obj:`Any <tvm_tirx.ir.AnyType>`
* :obj:`Axis <tvm_tirx.tirx.layout.Axis>`
* :obj:`BitwiseAnd <tvm_tirx.ir.prim.BitwiseAnd>`
* :obj:`BitwiseNot <tvm_tirx.ir.prim.BitwiseNot>`
* :obj:`BitwiseOr <tvm_tirx.ir.prim.BitwiseOr>`
* :obj:`BitwiseXor <tvm_tirx.ir.prim.BitwiseXor>`
* :obj:`Broadcast <tvm_tirx.ir.prim.Broadcast>`
* :obj:`Call <tvm_tirx.ir.Call>`
* :obj:`CallEffectKind <tvm_tirx.ir.CallEffectKind>`
* :obj:`CallFFIKernelAttr <tvm_tirx.tirx.CallFFIKernelAttr>`
* :obj:`Cast <tvm_tirx.ir.prim.Cast>`
* :obj:`ComposeLayout <tvm_tirx.tirx.ComposeLayout>`
* :obj:`DataTypeImm <tvm_tirx.ir.DataTypeImm>`
* :obj:`Div <tvm_tirx.ir.prim.Div>`
* :obj:`EQ <tvm_tirx.ir.prim.EQ>`
* :obj:`FloatImm <tvm_tirx.ir.FloatImm>`
* :obj:`FloorDiv <tvm_tirx.ir.prim.FloorDiv>`
* :obj:`FloorMod <tvm_tirx.ir.prim.FloorMod>`
* :obj:`Func <tvm_tirx.ir.FuncType>`
* :obj:`FuncType <tvm_tirx.ir.FuncType>`
* :obj:`GE <tvm_tirx.ir.prim.GE>`
* :obj:`GT <tvm_tirx.ir.prim.GT>`
* :obj:`GenericConst <tvm_tirx.ir.GenericConst>`
* :obj:`IRBuilder <tvm_tirx.script.ir_builder.IRBuilder>`
* :obj:`IRModuleFrame <tvm_tirx.script.ir_builder.IRModuleFrame>`
* :obj:`IntImm <tvm_tirx.ir.IntImm>`
* :obj:`Iter <tvm_tirx.tirx.layout.Iter>`
* :obj:`LE <tvm_tirx.ir.prim.LE>`
* :obj:`LShift <tvm_tirx.ir.prim.LShift>`
* :obj:`LT <tvm_tirx.ir.prim.LT>`
* :obj:`Lambda <tvm_tirx.script.ir_builder.Lambda>`
* :obj:`Layout <tvm_tirx.tirx.Layout>`
* :obj:`Max <tvm_tirx.ir.prim.Max>`
* :obj:`MemoryScope <tvm_tirx.runtime.MemoryScope>`
* :obj:`Min <tvm_tirx.ir.prim.Min>`
* :obj:`Missing <tvm_tirx.ir.MissingType>`
* :obj:`MissingType <tvm_tirx.ir.MissingType>`
* :obj:`Mod <tvm_tirx.ir.prim.Mod>`
* :obj:`Mul <tvm_tirx.ir.prim.Mul>`
* :obj:`NE <tvm_tirx.ir.prim.NE>`
* :obj:`Not <tvm_tirx.ir.prim.Not>`
* :obj:`Op <tvm_tirx.ir.Op>`
* :obj:`Opaque <tvm_tirx.ir.OpaqueType>`
* :obj:`Or <tvm_tirx.ir.prim.Or>`
* :obj:`Prim <tvm_tirx.ir.PrimType>`
* :obj:`PrimType <tvm_tirx.ir.PrimType>`
* :obj:`Ptr <tvm_tirx.ir.PtrType>`
* :obj:`RShift <tvm_tirx.ir.prim.RShift>`
* :obj:`Ramp <tvm_tirx.ir.prim.Ramp>`
* :obj:`Range <tvm_tirx.ir.Range>`
* :obj:`Select <tvm_tirx.ir.prim.Select>`
* :obj:`Shuffle <tvm_tirx.ir.prim.Shuffle>`
* :obj:`String <tvm_tirx.ir.StringType>`
* :obj:`StringImm <tvm_tirx.ir.StringImm>`
* :obj:`StringType <tvm_tirx.ir.StringType>`
* :obj:`Sub <tvm_tirx.ir.prim.Sub>`
* :obj:`Tensor <tvm_tirx.tirx.TensorType>`
* :obj:`TensorLoad <tvm_tirx.tirx.TensorLoad>`
* :obj:`TensorMap <tvm_tirx.tirx.TensorMapType>`
* :obj:`TensorRegion <tvm_tirx.ir.TensorRegion>`
* :obj:`TensorRegionType <tvm_tirx.ir.TensorRegionType>`
* :obj:`ThreadScope <tvm_tirx.tirx.ThreadScope>`
* :obj:`TileLayout <tvm_tirx.tirx.TileLayout>`
* :obj:`Type <tvm_tirx.ir.Type>`
* :obj:`Var <tvm_tirx.ir.Var>`
* :obj:`at_ <tvm_tirx.script.ir_builder.at_>`
* :obj:`bfloat16 <tvm_tirx.tirx.script.bf16>`
* :obj:`boolean <tvm_tirx.tirx.script.bool>`
* :obj:`broadcast <tvm_tirx.ir.prim.Broadcast>`
* :obj:`constexpr <tvm_tirx.script.ir_builder.constexpr>`
* :obj:`decl_function <tvm_tirx.script.ir_builder.decl_function>`
* :obj:`dtype <tvm_tirx.script.ir_builder.dtype>`
* :obj:`dynamic <tvm_tirx.script.ir_builder.dynamic>`
* :obj:`emit_ <tvm_tirx.tirx.script.emit>`
* :obj:`float16 <tvm_tirx.tirx.script.f16>`
* :obj:`float32 <tvm_tirx.tirx.script.f32>`
* :obj:`float64 <tvm_tirx.tirx.script.f64>`
* :obj:`func_ret_type_ <tvm_tirx.tirx.script.func_ret>`
* :obj:`int16 <tvm_tirx.tirx.script.i16>`
* :obj:`int32 <tvm_tirx.tirx.script.i32>`
* :obj:`int64 <tvm_tirx.tirx.script.i64>`
* :obj:`int8 <tvm_tirx.tirx.script.i8>`
* :obj:`ir_module <tvm_tirx.script.parser.ir_module>`
* :obj:`llvm_lookup_intrinsic_id <tvm_tirx.codegen.llvm.lookup_intrinsic_id>`
* :obj:`make_node <tvm_tirx.ir.make_node>`
* :obj:`meta_var <tvm_tirx.script.ir_builder.meta_var>`
* :obj:`module_attrs <tvm_tirx.script.ir_builder.module_attrs>`
* :obj:`module_get_attr <tvm_tirx.script.ir_builder.module_get_attr>`
* :obj:`module_global_infos <tvm_tirx.script.ir_builder.module_global_infos>`
* :obj:`module_member_ <tvm_tirx.script.ir_builder.module_member_>`
* :obj:`module_set_attr <tvm_tirx.script.ir_builder.module_set_attr>`
* :obj:`pyfunc <tvm_tirx.script.parser.pyfunc>`
* :obj:`ramp <tvm_tirx.ir.prim.Ramp>`
* :obj:`resolve_global_info_args <tvm_tirx.script.ir_builder.resolve_global_info_args>`
* :obj:`smem <tvm_tirx.tirx.script.alloc_shared>`
* :obj:`uint16 <tvm_tirx.tirx.script.u16>`
* :obj:`uint32 <tvm_tirx.tirx.script.u32>`
* :obj:`uint64 <tvm_tirx.tirx.script.u64>`
* :obj:`uint8 <tvm_tirx.tirx.script.u8>`
* :obj:`with_at_group_ <tvm_tirx.script.ir_builder.with_at_group_>`

tvm_tirx.tirx.script.jit
------------------------

.. currentmodule:: tvm_tirx.tirx.script.jit

.. autosummary::
   :toctree: generated/

   TIRJit

* :obj:`OptionalAnnotation <tvm_tirx.tirx.script.Optional>`

tvm_tirx.tirx.script.ir_builder.external_kernel
-----------------------------------------------

.. currentmodule:: tvm_tirx.tirx.script.ir_builder.external_kernel

.. autosummary::
   :toctree: generated/

   BaseKernel
   SourceKernel
   call_kernel


tvm_tirx.tirx.script.ir_builder.triton
--------------------------------------

.. currentmodule:: tvm_tirx.tirx.script.ir_builder.triton

.. autosummary::
   :toctree: generated/

   TritonKernel
