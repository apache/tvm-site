
.. _program_listing_file_tvm_ffi_extra_c_env_api.h:

Program Listing for File c_env_api.h
====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_c_env_api.h>` (``tvm/ffi/extra/c_env_api.h``)

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
   #ifndef TVM_FFI_EXTRA_C_ENV_API_H_
   #define TVM_FFI_EXTRA_C_ENV_API_H_
   
   #include <tvm/ffi/c_api.h>
   
   #ifdef __cplusplus
   extern "C" {
   #endif
   
   // ----------------------------------------------------------------------------
   // Stream context
   // Focusing on minimalistic thread-local context recording stream being used.
   // We explicitly not handle allocation/de-allocation of stream here.
   // ----------------------------------------------------------------------------
   typedef void* TVMFFIStreamHandle;
   
   TVM_FFI_DLL int TVMFFIEnvSetCurrentStream(int32_t device_type, int32_t device_id,
                                             TVMFFIStreamHandle stream,
                                             TVMFFIStreamHandle* opt_out_original_stream);
   
   TVM_FFI_DLL TVMFFIStreamHandle TVMFFIEnvGetCurrentStream(int32_t device_type, int32_t device_id);
   
   TVM_FFI_DLL int TVMFFIEnvCheckSignals();
   
   TVM_FFI_DLL int TVMFFIEnvRegisterCAPI(const char* name, void* symbol);
   
   // ----------------------------------------------------------------------------
   // Module symbol management in callee side
   // ----------------------------------------------------------------------------
   TVM_FFI_DLL int TVMFFIEnvModLookupFromImports(TVMFFIObjectHandle library_ctx, const char* func_name,
                                                 TVMFFIObjectHandle* out);
   
   TVM_FFI_DLL int TVMFFIEnvModRegisterContextSymbol(const char* name, void* symbol);
   
   TVM_FFI_DLL int TVMFFIEnvModRegisterSystemLibSymbol(const char* name, void* symbol);
   
   #ifdef __cplusplus
   }  // extern "C"
   #endif
   #endif  // TVM_FFI_EXTRA_C_ENV_API_H_
