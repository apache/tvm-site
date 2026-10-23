
.. _program_listing_file_tvm_target_target_tag_registry.h:

Program Listing for File target_tag_registry.h
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_target_target_tag_registry.h>` (``tvm/target/target_tag_registry.h``)

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
   
   #ifndef TVM_TARGET_TARGET_TAG_REGISTRY_H_
   #define TVM_TARGET_TARGET_TAG_REGISTRY_H_
   
   #include <tvm/ffi/container/dict.h>
   #include <tvm/target/target.h>
   
   namespace tvm {
   
   class TargetTagRegistry {
    public:
     using Config = ffi::Map<ffi::String, ffi::Any>;
   
     TVM_TIRX_DLL static TargetTagRegistry* Global();
   
     TVM_TIRX_DLL ffi::Optional<Target> Get(const ffi::String& name) const;
   
     TVM_TIRX_DLL ffi::Optional<Config> GetConfig(const ffi::String& name) const;
   
     TVM_TIRX_DLL ffi::Map<ffi::String, Target> ListTags() const;
   
     TVM_TIRX_DLL Target AddTag(ffi::String name, Config config, bool override);
   
    private:
     TargetTagRegistry() = default;
     TargetTagRegistry(const TargetTagRegistry&) = delete;
     TargetTagRegistry& operator=(const TargetTagRegistry&) = delete;
   
     ffi::Dict<ffi::String, Config> configs_;
   };
   
   }  // namespace tvm
   
   #endif  // TVM_TARGET_TARGET_TAG_REGISTRY_H_
