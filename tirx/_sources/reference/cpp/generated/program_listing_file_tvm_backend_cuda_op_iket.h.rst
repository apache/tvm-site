
.. _program_listing_file_tvm_backend_cuda_op_iket.h:

Program Listing for File iket.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_backend_cuda_op_iket.h>` (``tvm/backend/cuda/op/iket.h``)

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
   
   #ifndef TVM_BACKEND_CUDA_OP_IKET_H_
   #define TVM_BACKEND_CUDA_OP_IKET_H_
   
   #include <tvm/ir/op.h>
   
   namespace tvm {
   namespace backend {
   namespace cuda {
   
   TVM_TIRX_DLL const Op& iket_mark_op();
   
   TVM_TIRX_DLL const Op& iket_range_start_op();
   
   TVM_TIRX_DLL const Op& iket_range_end_op();
   
   TVM_TIRX_DLL const Op& iket_range_push_op();
   
   TVM_TIRX_DLL const Op& iket_range_pop_op();
   
   TVM_TIRX_DLL const Op& iket_sentinel_token_op();
   
   }  // namespace cuda
   }  // namespace backend
   }  // namespace tvm
   
   #endif  // TVM_BACKEND_CUDA_OP_IKET_H_
