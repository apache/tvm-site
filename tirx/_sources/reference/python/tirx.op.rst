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

tvm_tirx.tirx.op
================

.. automodule:: tvm_tirx.tirx.op
   :no-members:

.. currentmodule:: tvm_tirx.tirx.op

Classes
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``CallFFIKernelAttr``
     - :obj:`tvm_tirx.tirx.CallFFIKernelAttr`
   * - ``TensorLoad``
     - :obj:`tvm_tirx.ir.TensorLoad`
   * - ``TensorMapType``
     - :obj:`tvm_tirx.tirx.TensorMapType`

Functions
---------

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

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``abi_field_get``
     - :obj:`tvm_tirx.tirx.abi_field_get`
   * - ``abi_field_set``
     - :obj:`tvm_tirx.tirx.abi_field_set`
   * - ``address_of``
     - :obj:`tvm_tirx.tirx.address_of`
   * - ``alloc_workspace``
     - :obj:`tvm_tirx.tirx.alloc_workspace`
   * - ``assume_aligned``
     - :obj:`tvm_tirx.tirx.assume_aligned`
   * - ``call_cpacked``
     - :obj:`tvm_tirx.tirx.call_cpacked`
   * - ``call_cpacked_lowered``
     - :obj:`tvm_tirx.tirx.call_cpacked_lowered`
   * - ``call_extern``
     - :obj:`tvm_tirx.tirx.call_extern`
   * - ``call_ffi_kernel``
     - :obj:`tvm_tirx.tirx.call_ffi_kernel`
   * - ``call_intrin``
     - :obj:`tvm_tirx.tirx.call_intrin`
   * - ``call_llvm_intrin``
     - :obj:`tvm_tirx.tirx.call_llvm_intrin`
   * - ``call_llvm_pure_intrin``
     - :obj:`tvm_tirx.tirx.call_llvm_pure_intrin`
   * - ``call_packed``
     - :obj:`tvm_tirx.tirx.call_packed`
   * - ``call_packed_lowered``
     - :obj:`tvm_tirx.tirx.call_packed_lowered`
   * - ``call_pure_extern``
     - :obj:`tvm_tirx.tirx.call_pure_extern`
   * - ``cpu_parallel_barrier``
     - :obj:`tvm_tirx.tirx.cpu_parallel_barrier`
   * - ``free_workspace``
     - :obj:`tvm_tirx.tirx.free_workspace`
   * - ``get_active_lane_mask``
     - :obj:`tvm_tirx.tirx.get_active_lane_mask`
   * - ``get_vscale_expr``
     - :obj:`tvm_tirx.tirx.get_vscale_expr`
   * - ``gpu_dp4a``
     - :obj:`tvm_tirx.tirx.gpu_dp4a`
   * - ``gpu_fill_fragment``
     - :obj:`tvm_tirx.tirx.gpu_fill_fragment`
   * - ``gpu_load_matrix_sync``
     - :obj:`tvm_tirx.tirx.gpu_load_matrix_sync`
   * - ``gpu_mma_sync``
     - :obj:`tvm_tirx.tirx.gpu_mma_sync`
   * - ``gpu_store_matrix_sync``
     - :obj:`tvm_tirx.tirx.gpu_store_matrix_sync`
   * - ``gpu_thread_allreduce``
     - :obj:`tvm_tirx.tirx.gpu_thread_allreduce`
   * - ``handle_add_byte_offset``
     - :obj:`tvm_tirx.tirx.handle_add_byte_offset`
   * - ``ignore_loop_partition``
     - :obj:`tvm_tirx.tirx.ignore_loop_partition`
   * - ``is_tensor_var``
     - :obj:`tvm_tirx.tirx.is_tensor_var`
   * - ``isnullptr``
     - :obj:`tvm_tirx.tirx.isnullptr`
   * - ``logaddexp``
     - :obj:`tvm_tirx.tirx.logaddexp`
   * - ``ptr_byte_offset``
     - :obj:`tvm_tirx.tirx.ptr_byte_offset`
   * - ``register_intrin_lowering``
     - :obj:`tvm_tirx.tirx.register_intrin_lowering`
   * - ``reinterpret``
     - :obj:`tvm_tirx.tirx.reinterpret`
   * - ``stack_alloca``
     - :obj:`tvm_tirx.tirx.stack_alloca`
   * - ``stack_make_dltensor``
     - :obj:`tvm_tirx.tirx.stack_make_dltensor`
   * - ``stack_make_shape``
     - :obj:`tvm_tirx.tirx.stack_make_shape`
   * - ``tensor_data_ptr``
     - :obj:`tvm_tirx.tirx.tensor_data_ptr`
   * - ``throw_last_error``
     - :obj:`tvm_tirx.tirx.throw_last_error`
   * - ``undef``
     - :obj:`tvm_tirx.tirx.undef`
   * - ``vector_combine``
     - :obj:`tvm_tirx.tirx.vector_combine`
   * - ``vector_high``
     - :obj:`tvm_tirx.tirx.vector_high`
   * - ``vector_low``
     - :obj:`tvm_tirx.tirx.vector_low`
