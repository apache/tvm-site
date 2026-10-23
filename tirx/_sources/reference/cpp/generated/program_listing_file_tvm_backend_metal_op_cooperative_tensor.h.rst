
.. _program_listing_file_tvm_backend_metal_op_cooperative_tensor.h:

Program Listing for File cooperative_tensor.h
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_backend_metal_op_cooperative_tensor.h>` (``tvm/backend/metal/op/cooperative_tensor.h``)

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
   
   #ifndef TVM_BACKEND_METAL_OP_COOPERATIVE_TENSOR_H_
   #define TVM_BACKEND_METAL_OP_COOPERATIVE_TENSOR_H_
   
   #include <tvm/ir/op.h>
   #include <tvm/ir/prim/expr.h>
   
   namespace tvm {
   namespace backend {
   namespace metal {
   
   TVM_TIRX_DLL const Op& cooperative_tensor_fill_op();
   
   TVM_TIRX_DLL const Op& cooperative_tensor_load_op();
   
   TVM_TIRX_DLL const Op& cooperative_tensor_store_op();
   
   TVM_TIRX_DLL const Op& cooperative_tensor_multiply_accumulate_op();
   
   }  // namespace metal
   }  // namespace backend
   }  // namespace tvm
   
   #endif  // TVM_BACKEND_METAL_OP_COOPERATIVE_TENSOR_H_
