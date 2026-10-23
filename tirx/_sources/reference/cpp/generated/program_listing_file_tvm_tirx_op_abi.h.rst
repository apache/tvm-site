
.. _program_listing_file_tvm_tirx_op_abi.h:

Program Listing for File abi.h
==============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_op_abi.h>` (``tvm/tirx/op/abi.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /*
    * Licensed to the Apache Software Foundation (ASF) under one
    * or more contributor license agreements.  See the NOTICE file
    * distributed with this work for additional information
    * regarding copyright ownership.  The ASF licenses this file
    * to you under the Apache License, Version 2.0 (the
    * "License"); you may not use this file except in compliance
    * with the License.  You may obtain a copy of the License at
    *
    *   http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing,
    * software distributed under the License is distributed on an
    * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    * KIND, either express or implied.  See the License for the
    * specific language governing permissions and limitations
    * under the License.
    */
   
   #ifndef TVM_TIRX_OP_ABI_H_
   #define TVM_TIRX_OP_ABI_H_
   
   #include <tvm/ir/attrs.h>
   #include <tvm/ir/op.h>
   #include <tvm/ir/prim/expr.h>
   
   namespace tvm {
   namespace tirx {
   
   TVM_TIRX_DLL const Op& call_extern_op();
   
   TVM_TIRX_DLL const Op& call_pure_extern_op();
   
   TVM_TIRX_DLL const Op& call_llvm_intrin_op();
   
   TVM_TIRX_DLL const Op& call_llvm_pure_intrin_op();
   
   TVM_TIRX_DLL const Op& call_spirv_pure_glsl450_op();
   
   enum TVMStructFieldKind : int {
     // DLTensor fields
     kDLTensorAddr,
     kDLTensorData,
     kDLTensorShape,
     kDLTensorStrides,
     kDLTensorNDim,
     kDLTensorTypeCode,
     kDLTensorTypeBits,
     kDLTensorTypeLanes,
     kDLTensorByteOffset,
     kDLTensorDeviceId,
     kDLTensorDeviceType,
     kDLTensorKindBound_,
     // TVMValue field
     kTVMValueContent,
     kTVMFFIAnyTypeIndex,
     kTVMFFIAnyZeroPadding,
     kTVMFFIAnyUnionValue,
     kTVMValueKindBound_,
     // Generic int64 array element access: ((int64_t*)buf)[index]
     kInt64ArrayElem,
   };
   
   TVM_TIRX_DLL const Op& abi_field_get_op();
   
   TVM_TIRX_DLL const Op& abi_field_set_op();
   
   TVM_TIRX_DLL const Op& throw_last_error_op();
   
   TVM_TIRX_DLL const Op& stack_alloca_op();
   
   TVM_TIRX_DLL const Op& stack_make_shape_op();
   
   TVM_TIRX_DLL const Op& stack_make_dltensor_op();
   
   TVM_TIRX_DLL const Op& call_packed_op();
   
   struct CallFFIKernelAttr : public AttrsObj {
     ffi::Array<ffi::String> launch_params;
     ffi::Array<ffi::String> launch_fields;
     int64_t num_kernel_args{-1};
     ffi::Map<ffi::String, int64_t> kernel_attrs;
   
     static void RegisterReflection();
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.CallFFIKernelAttr", CallFFIKernelAttr, AttrsObj);
   };
   
   TVM_TIRX_DLL const Op& call_ffi_kernel_op();
   
   TVM_TIRX_DLL const Op& call_cpacked_op();
   
   TVM_TIRX_DLL const Op& call_packed_lowered_op();
   
   TVM_TIRX_DLL const Op& call_cpacked_lowered_op();
   
   }  // namespace tirx
   }  // namespace tvm
   
   #endif  // TVM_TIRX_OP_ABI_H_
