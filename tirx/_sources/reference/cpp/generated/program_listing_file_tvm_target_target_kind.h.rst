
.. _program_listing_file_tvm_target_target_kind.h:

Program Listing for File target_kind.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_target_target_kind.h>` (``tvm/target/target_kind.h``)

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
   
   #ifndef TVM_TARGET_TARGET_KIND_H_
   #define TVM_TARGET_TARGET_KIND_H_
   
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/runtime/base.h>
   #include <tvm/target/config_schema.h>
   
   #include <utility>
   #include <vector>
   
   namespace tvm {
   
   class Target;
   
   using FTargetCanonicalizer =
       ffi::TypedFunction<ffi::Map<ffi::String, ffi::Any>(ffi::Map<ffi::String, ffi::Any>)>;
   
   class TargetInternal;
   
   class TargetKindObj : public ffi::Object {
    public:
     ffi::String name;
     int default_device_type;
     ffi::Array<ffi::String> default_keys;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TargetKindObj>()
           .def_ro("name", &TargetKindObj::name)
           .def_ro("default_device_type", &TargetKindObj::default_device_type,
                   refl::AttachFieldFlag::SEqHashIgnore())
           .def_ro("default_keys", &TargetKindObj::default_keys,
                   refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("target.TargetKind", TargetKindObj, ffi::Object);
   
    private:
     ConfigSchema schema_;
     friend class TargetKindRegistry;
     friend class TargetKindDef;
     friend class TargetInternal;
   };
   
   class TargetKind : public ffi::ObjectRef {
    public:
     explicit TargetKind(ffi::ObjectPtr<TargetKindObj> data) : ffi::ObjectRef(data) {
       TVM_FFI_ICHECK(data != nullptr);
     }
     TVM_TIRX_DLL static ffi::Optional<TargetKind> Get(const ffi::String& target_kind_name);
     TargetKindObj* operator->() { return static_cast<TargetKindObj*>(data_.get()); }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TargetKind, ffi::ObjectRef, TargetKindObj);
   
    private:
     friend class TargetInternal;
   };
   
   class TargetKindRegistry {
    public:
     TVM_TIRX_DLL static TargetKindRegistry* Global();
     TVM_TIRX_DLL TargetKind RegisterOrGet(const ffi::String& name);
     TVM_TIRX_DLL ffi::Optional<TargetKind> Get(const ffi::String& name);
     TVM_TIRX_DLL ffi::Array<ffi::String> ListTargetKinds();
     TVM_TIRX_DLL ffi::Map<ffi::String, ffi::String> ListTargetKindOptions(const TargetKind& kind);
   
    private:
     ffi::Map<ffi::String, TargetKind> kinds_;
   };
   
   class TargetKindDef {
    public:
     explicit TargetKindDef(const ffi::String& name)
         : kind_(TargetKindRegistry::Global()->RegisterOrGet(name)) {}
     TargetKindDef& set_default_device_type(int device_type) {
       kind_->default_device_type = device_type;
       return *this;
     }
     TargetKindDef& set_default_keys(std::vector<ffi::String> keys) {
       kind_->default_keys = keys;
       return *this;
     }
     TargetKindDef& set_target_canonicalizer(FTargetCanonicalizer canonicalizer) {
       kind_->schema_.set_canonicalizer(std::move(canonicalizer));
       return *this;
     }
     template <typename ValueType, typename... Traits>
     TargetKindDef& def_option(const ffi::String& key, Traits&&... traits) {
       kind_->schema_.def_option<ValueType>(key, std::forward<Traits>(traits)...);
       return *this;
     }
   
    private:
     TargetKind kind_;
   };
   
   }  // namespace tvm
   
   #endif  // TVM_TARGET_TARGET_KIND_H_
