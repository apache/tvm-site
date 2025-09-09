
.. _program_listing_file_tvm_ffi_extra_json.h:

Program Listing for File json.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_json.h>` (``tvm/ffi/extra/json.h``)

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
   #ifndef TVM_FFI_EXTRA_JSON_H_
   #define TVM_FFI_EXTRA_JSON_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/extra/base.h>
   
   namespace tvm {
   namespace ffi {
   namespace json {
   
   using Value = Any;
   
   using Object = ffi::Map<Any, Any>;
   
   using Array = ffi::Array<Any>;
   
   TVM_FFI_EXTRA_CXX_API json::Value Parse(const String& json_str, String* error_msg = nullptr);
   
   TVM_FFI_EXTRA_CXX_API String Stringify(const json::Value& value,
                                          Optional<int> indent = std::nullopt);
   
   }  // namespace json
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_JSON_H_
