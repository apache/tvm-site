
.. _program_listing_file_tvm_runtime_device_api.h:

Program Listing for File device_api.h
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_runtime_device_api.h>` (``tvm/runtime/device_api.h``)

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
   
   #ifndef TVM_RUNTIME_DEVICE_API_H_
   #define TVM_RUNTIME_DEVICE_API_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/tensor.h>
   #include <tvm/ffi/device.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/string.h>
   #include <tvm/runtime/base.h>
   #include <tvm/runtime/memory_scope.h>
   
   #include <string>
   typedef void* TVMStreamHandle;
   
   namespace tvm {
   
   // alias DLDevice
   using Device = DLDevice;
   
   namespace runtime {
   
   #ifdef __cplusplus
   typedef enum : int32_t {
   #else
   typedef enum {
   #endif
     // To help avoid accidental conflicts between `DLDeviceType`
     // and this enumeration, start numbering the new enumerators
     // a little higher than (currently) seems necessary.
     TVMDeviceExtType_End = 36,  // sentinel value
   } TVMDeviceExtType;
   
   #ifdef TVM_KALLOC_ALIGNMENT
   constexpr int kAllocAlignment = TVM_KALLOC_ALIGNMENT;
   
   constexpr int kTempAllocaAlignment = TVM_KALLOC_ALIGNMENT;
   #else
   constexpr int kAllocAlignment = 64;
   
   constexpr int kTempAllocaAlignment = 64;
   #endif  // TVM_KALLOC_ALIGNMENT
   
   constexpr int kMaxStackAlloca = 1024;
   
   class TVM_RUNTIME_DLL DeviceAPI {
    public:
     virtual ~DeviceAPI() {}
     virtual void SetDevice(Device dev) = 0;
     virtual int GetDeviceIndex();
     virtual ffi::Any GetAttr(DLDevice dev, const ffi::String& name) = 0;
   
     virtual ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                               ffi::Optional<MemoryScope> mem_scope = std::nullopt) = 0;
   
     virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream);
     virtual TVMStreamHandle CreateStream(Device dev);
   
     virtual void FreeStream(Device dev, TVMStreamHandle stream);
   
     virtual void StreamSync(Device dev, TVMStreamHandle stream) = 0;
     virtual void SetStream(Device dev, TVMStreamHandle stream);
     virtual TVMStreamHandle GetCurrentStream(Device dev);
     virtual void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst);
     virtual void* AllocWorkspace(Device dev, size_t nbytes, DLDataType type_hint = {});
     virtual void FreeWorkspace(Device dev, void* ptr);
   
     static DeviceAPI* Get(Device dev, bool allow_missing = false);
   
     static bool NeedSetDevice(int device_type) { return device_type != kDLCPU; }
   
     virtual bool SupportsDevicePtrArithmeticOnHost() { return false; }
   
     virtual void* GetMappedHostPtrIfAvailable(void* data) { return nullptr; }
   
    protected:
     virtual void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                                 size_t num_bytes, Device dev_from, Device dev_to,
                                 DLDataType type_hint, TVMStreamHandle stream);
   };
   
   inline const char* DLDeviceType2Str(int type) {
     switch (type) {
       case kDLCPU:
         return "cpu";
       case kDLCUDA:
         return "cuda";
       case kDLCUDAHost:
         return "cuda_host";
       case kDLCUDAManaged:
         return "cuda_managed";
       case kDLOpenCL:
         return "opencl";
       case kDLVulkan:
         return "vulkan";
       case kDLMetal:
         return "metal";
       case kDLVPI:
         return "vpi";
       case kDLROCM:
         return "rocm";
       case kDLROCMHost:
         return "rocm_host";
       case kDLExtDev:
         return "ext_dev";
       case kDLOneAPI:
         return "oneapi";
       case kDLWebGPU:
         return "webgpu";
       case kDLHexagon:
         return "hexagon";
       case kDLTrn:
         return "trn";
       default:
         TVM_FFI_THROW(InternalError) << "unknown type = " << type;
     }
     throw;
   }
   
   inline std::ostream& operator<<(std::ostream& os, DLDevice dev) {  // NOLINT(*)
     os << DLDeviceType2Str(static_cast<int>(dev.device_type)) << ":" << dev.device_id;
     return os;
   }
   
   TVM_RUNTIME_DLL bool RuntimeEnabled(const ffi::String& target);
   
   }  // namespace runtime
   }  // namespace tvm
   
   #endif  // TVM_RUNTIME_DEVICE_API_H_
