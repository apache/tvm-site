
.. _program_listing_file_tvm_ffi_extra_cuda_device_guard.h:

Program Listing for File device_guard.h
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_cuda_device_guard.h>` (``tvm/ffi/extra/cuda/device_guard.h``)

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
   #ifndef TVM_FFI_EXTRA_CUDA_DEVICE_GUARD_H_
   #define TVM_FFI_EXTRA_CUDA_DEVICE_GUARD_H_
   
   #include <tvm/ffi/extra/cuda/base.h>
   
   namespace tvm {
   namespace ffi {
   
   struct CUDADeviceGuard {
     CUDADeviceGuard() = delete;
     explicit CUDADeviceGuard(int device_index) {
       target_device_index_ = device_index;
       TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&original_device_index_));
       if (target_device_index_ != original_device_index_) {
         TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(device_index));
       }
     }
   
     ~CUDADeviceGuard() noexcept(false) {
       if (original_device_index_ != target_device_index_) {
         TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(original_device_index_));
       }
     }
   
    private:
     int original_device_index_;
     int target_device_index_;
   };
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_CUDA_DEVICE_GUARD_H_
