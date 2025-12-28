
.. _program_listing_file_tvm_ffi_extra_cuda_base.h:

Program Listing for File base.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_cuda_base.h>` (``tvm/ffi/extra/cuda/base.h``)

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
   #ifndef TVM_FFI_EXTRA_CUDA_BASE_H_
   #define TVM_FFI_EXTRA_CUDA_BASE_H_
   
   namespace tvm {
   namespace ffi {
   
   struct dim3 {
     unsigned int x;
     unsigned int y;
     unsigned int z;
   
     dim3() : x(1), y(1), z(1) {}
   
     explicit dim3(unsigned int x_) : x(x_), y(1), z(1) {}
   
     dim3(unsigned int x_, unsigned int y_) : x(x_), y(y_), z(1) {}
   
     dim3(unsigned int x_, unsigned int y_, unsigned int z_) : x(x_), y(y_), z(z_) {}
   };
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_EXTRA_CUDA_BASE_H_
