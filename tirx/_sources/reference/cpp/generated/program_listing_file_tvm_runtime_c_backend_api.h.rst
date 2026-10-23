
.. _program_listing_file_tvm_runtime_c_backend_api.h:

Program Listing for File c_backend_api.h
========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_runtime_c_backend_api.h>` (``tvm/runtime/c_backend_api.h``)

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
   
   #ifndef TVM_RUNTIME_C_BACKEND_API_H_
   #define TVM_RUNTIME_C_BACKEND_API_H_
   
   #include <tvm/runtime/base.h>
   
   #ifdef __cplusplus
   extern "C" {
   #endif
   
   TVM_RUNTIME_DLL int TVMBackendSetDevice(int device_type, int device_index);
   
   TVM_RUNTIME_DLL int TVMBackendGetDeviceIndex(int device_type, int* out_device_index);
   
   TVM_RUNTIME_DLL void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                                                  int dtype_code_hint, int dtype_bits_hint);
   
   TVM_RUNTIME_DLL int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);
   
   typedef struct {
     void* sync_handle;
     int32_t num_task;
   } TVMParallelGroupEnv;
   
   typedef int (*FTVMParallelLambda)(int task_id, TVMParallelGroupEnv* penv, void* cdata);
   
   TVM_RUNTIME_DLL int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task);
   
   TVM_RUNTIME_DLL int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv);
   
   #ifdef __cplusplus
   }  // TVM_EXTERN_C
   #endif
   #endif  // TVM_RUNTIME_C_BACKEND_API_H_
