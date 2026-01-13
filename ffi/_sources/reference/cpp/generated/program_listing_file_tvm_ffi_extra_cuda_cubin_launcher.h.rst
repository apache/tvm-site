
.. _program_listing_file_tvm_ffi_extra_cuda_cubin_launcher.h:

Program Listing for File cubin_launcher.h
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_cuda_cubin_launcher.h>` (``tvm/ffi/extra/cuda/cubin_launcher.h``)

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
   #ifndef TVM_FFI_EXTRA_CUDA_CUBIN_LAUNCHER_H_
   #define TVM_FFI_EXTRA_CUDA_CUBIN_LAUNCHER_H_
   
   #include <cuda.h>  // NOLINT(clang-diagnostic-error)
   #include <cuda_runtime.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/extra/c_env_api.h>
   #include <tvm/ffi/extra/cuda/base.h>
   #include <tvm/ffi/extra/cuda/internal/unified_api.h>
   #include <tvm/ffi/string.h>
   
   #include <cstdint>
   #include <cstring>
   
   namespace tvm {
   namespace ffi {
   
   #define TVM_FFI_EMBED_CUBIN(name)                        \
     extern "C" const char __tvm_ffi__cubin_##name[];       \
     extern "C" const char __tvm_ffi__cubin_##name##_end[]; \
     namespace {                                            \
     struct EmbedCubinModule_##name {                       \
       tvm::ffi::CubinModule mod{__tvm_ffi__cubin_##name};  \
       static EmbedCubinModule_##name* Global() {           \
         static EmbedCubinModule_##name inst;               \
         return &inst;                                      \
       }                                                    \
     };                                                     \
     } /* anonymous namespace */
   
   #define TVM_FFI_EMBED_CUBIN_FROM_BYTES(name, imageBytes) \
     namespace {                                            \
     struct EmbedCubinModule_##name {                       \
       tvm::ffi::CubinModule mod{imageBytes};               \
       static EmbedCubinModule_##name* Global() {           \
         static EmbedCubinModule_##name inst;               \
         return &inst;                                      \
       }                                                    \
     };                                                     \
     } /* anonymous namespace */
   
   #define TVM_FFI_EMBED_CUBIN_GET_KERNEL(name, kernel_name) \
     (EmbedCubinModule_##name::Global()->mod[kernel_name])
   
   // Forward declaration
   class CubinKernel;
   
   class CubinModule {
    public:
     explicit CubinModule(const Bytes& bytes) {
       TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuda_api::LoadLibrary(&library_, bytes.data()));
     }
   
     explicit CubinModule(const char* code) {
       TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuda_api::LoadLibrary(&library_, code));
     }
   
     explicit CubinModule(const unsigned char* code) {
       TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuda_api::LoadLibrary(&library_, code));
     }
   
     ~CubinModule() {
       if (library_ != nullptr) {
         cuda_api::UnloadLibrary(library_);
       }
     }
   
     CubinKernel GetKernel(const char* name);
   
     CubinKernel GetKernelWithMaxDynamicSharedMemory(const char* name, int64_t dynamic_smem_max);
   
     CubinKernel operator[](const char* name);
   
     cuda_api::LibraryHandle GetHandle() const { return library_; }
   
     // Non-copyable
     CubinModule(const CubinModule&) = delete;
     CubinModule& operator=(const CubinModule&) = delete;
   
     CubinModule(CubinModule&& other) noexcept : library_(other.library_) { other.library_ = nullptr; }
   
     CubinModule& operator=(CubinModule&& other) noexcept {
       if (this != &other) {
         if (library_ != nullptr) {
           cuda_api::UnloadLibrary(library_);
         }
         library_ = other.library_;
         other.library_ = nullptr;
       }
       return *this;
     }
   
    private:
     cuda_api::LibraryHandle library_ = nullptr;
   };
   
   class CubinKernel {
    public:
     CubinKernel(cuda_api::LibraryHandle library, const char* name) {
       TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuda_api::GetKernel(&kernel_, library, name));
     }
   
     ~CubinKernel() = default;
   
     cuda_api::ResultType Launch(void** args, dim3 grid, dim3 block, cuda_api::StreamHandle stream,
                                 uint32_t dyn_smem_bytes = 0) {
       return cuda_api::LaunchKernel(kernel_, args, grid, block, stream, dyn_smem_bytes);
     }
   
     cuda_api::KernelHandle GetHandle() const { return kernel_; }
   
     // Non-copyable
     CubinKernel(const CubinKernel&) = delete;
     CubinKernel& operator=(const CubinKernel&) = delete;
   
     CubinKernel(CubinKernel&& other) noexcept : kernel_(other.kernel_) { other.kernel_ = nullptr; }
   
     CubinKernel& operator=(CubinKernel&& other) noexcept {
       if (this != &other) {
         kernel_ = other.kernel_;
         other.kernel_ = nullptr;
       }
       return *this;
     }
   
    private:
     void SetMaxDynamicSharedMemory(int64_t dynamic_smem_max = -1) {
       int device_count = 0;
       cuda_api::ResultType err = cuda_api::GetDeviceCount(&device_count);
       if (err != cuda_api::kSuccess || device_count == 0) {
         return;  // No devices available, nothing to configure
       }
   
       bool any_success = false;
       for (int device_id = 0; device_id < device_count; ++device_id) {
         auto device = cuda_api::GetDeviceHandle(device_id);
         // Query device's maximum shared memory per block
         int max_shared_mem = 0;
         err = cuda_api::GetDeviceAttribute(
             &max_shared_mem,
             /* CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK/cudaDevAttrMaxSharedMemoryPerBlock */
             cuda_api::DeviceAttrType(8), device);
         if (err != cuda_api::kSuccess) {
           continue;  // Skip this device if we can't get its attribute
         }
   
         int shared_mem_to_set;
         if (dynamic_smem_max == -1) {
           int static_shared;
           err = cuda_api::GetKernelSharedMem(kernel_, static_shared, device);
           if (err != cuda_api::kSuccess) {
             continue;  // Skip this device if we can't get kernel attributes
           }
   
           // Calculate available dynamic shared memory:
           // device max shared memory - static shared memory used by kernel
           int64_t max_shared = static_cast<int64_t>(max_shared_mem);
           int64_t available = max_shared - static_shared;
           shared_mem_to_set = (available > 0) ? static_cast<int>(available) : 0;
         } else {
           shared_mem_to_set = static_cast<int>(dynamic_smem_max);
         }
   
         // Set the maximum dynamic shared memory size for this device
         err = cuda_api::SetKernelMaxDynamicSharedMem(kernel_, shared_mem_to_set, device);
         if (err == cuda_api::kSuccess) {
           any_success = true;
         }
         // Don't error out for individual device failures - user may only use some GPUs
       }
   
       // Only error out if setting failed for ALL devices
       if (!any_success && device_count > 0) {
         TVM_FFI_THROW(RuntimeError) << "Failed to set dynamic shared memory attribute for any device";
       }
     }
   
     cuda_api::KernelHandle kernel_ = nullptr;
   
     friend class CubinModule;
   };
   
   // Implementation of CubinModule methods that return CubinKernel
   inline CubinKernel CubinModule::GetKernelWithMaxDynamicSharedMemory(const char* name,
                                                                       int64_t dynamic_smem_max = -1) {
     auto kernel = CubinKernel(library_, name);
     kernel.SetMaxDynamicSharedMemory(dynamic_smem_max);
     return kernel;
   }
   
   inline CubinKernel CubinModule::GetKernel(const char* name) {
     auto kernel = CubinKernel(library_, name);
     return kernel;
   }
   
   inline CubinKernel CubinModule::operator[](const char* name) { return GetKernel(name); }
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_EXTRA_CUDA_CUBIN_LAUNCHER_H_
