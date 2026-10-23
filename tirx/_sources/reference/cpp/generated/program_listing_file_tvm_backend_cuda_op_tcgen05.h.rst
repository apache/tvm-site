
.. _program_listing_file_tvm_backend_cuda_op_tcgen05.h:

Program Listing for File tcgen05.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_backend_cuda_op_tcgen05.h>` (``tvm/backend/cuda/op/tcgen05.h``)

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
   
   #ifndef TVM_BACKEND_CUDA_OP_TCGEN05_H_
   #define TVM_BACKEND_CUDA_OP_TCGEN05_H_
   
   #include <tvm/ir/attrs.h>
   
   namespace tvm {
   namespace backend {
   namespace cuda {
   
   struct TCGen05InstrDescriptorAttrs : public AttrsNode {
     ffi::String d_dtype;
     ffi::String a_dtype;
     ffi::String b_dtype;
     int64_t M;
     int64_t N;
     int64_t K;
     bool trans_a;
     bool trans_b;
     int64_t n_cta_groups = 1;
     bool neg_a = false;
     bool neg_b = false;
     bool sat_d = false;
     bool is_sparse = false;
   
     static void RegisterReflection();
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.cuda.TCGen05InstrDescriptorAttrs",
                                       TCGen05InstrDescriptorAttrs, AttrsNode);
   };
   
   struct TCGen05InstrDescriptorBlockScaledAttrs : public AttrsNode {
     ffi::String d_dtype;
     ffi::String a_dtype;
     ffi::String b_dtype;
     ffi::String sfa_dtype;
     ffi::String sfb_dtype;
     int64_t M;
     int64_t N;
     int64_t K;
     bool trans_a;
     bool trans_b;
     int64_t n_cta_groups = 1;
     bool neg_a = false;
     bool neg_b = false;
     bool is_sparse = false;
   
     static void RegisterReflection();
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.cuda.TCGen05InstrDescriptorBlockScaledAttrs",
                                       TCGen05InstrDescriptorBlockScaledAttrs, AttrsNode);
   };
   
   }  // namespace cuda
   }  // namespace backend
   }  // namespace tvm
   
   #endif  // TVM_BACKEND_CUDA_OP_TCGEN05_H_
