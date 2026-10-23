
.. _program_listing_file_tvm_support_gpu_thread_tag.h:

Program Listing for File gpu_thread_tag.h
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_support_gpu_thread_tag.h>` (``tvm/support/gpu_thread_tag.h``)

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
   
   #ifndef TVM_SUPPORT_GPU_THREAD_TAG_H_
   #define TVM_SUPPORT_GPU_THREAD_TAG_H_
   
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/string.h>
   
   namespace tvm {
   namespace support {
   
   struct GPUThreadTag {
     int rank{0};
     int dim_index{0};
   
     static GPUThreadTag Parse(const ffi::String& tag) {
       if (tag.starts_with("blockIdx.")) {
         return {0, static_cast<int>(tag.c_str()[9] - 'x')};
       } else if (tag.starts_with("threadIdx.")) {
         return {1, static_cast<int>(tag.c_str()[10] - 'x')};
       } else if (tag.starts_with("clusterCtaIdx.")) {
         return {2, static_cast<int>(tag.c_str()[14] - 'x')};
       } else if (tag.starts_with("preferredClusterCtaIdx.")) {
         return {3, static_cast<int>(tag.c_str()[23] - 'x')};
       } else {
         TVM_FFI_THROW(InternalError) << "Unknown threadscope " << tag;
         return {};
       }
     }
   };
   
   }  // namespace support
   }  // namespace tvm
   #endif  // TVM_SUPPORT_GPU_THREAD_TAG_H_
