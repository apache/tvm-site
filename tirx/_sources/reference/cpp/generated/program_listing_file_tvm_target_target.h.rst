
.. _program_listing_file_tvm_target_target.h:

Program Listing for File target.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_target_target.h>` (``tvm/target/target.h``)

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
   
   #ifndef TVM_TARGET_TARGET_H_
   #define TVM_TARGET_TARGET_H_
   
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/function.h>
   #include <tvm/ir/with_context.h>
   #include <tvm/runtime/device_api.h>
   #include <tvm/target/target_kind.h>
   
   #include <algorithm>
   #include <string>
   
   namespace tvm {
   
   class TargetInternal;
   class Target;
   
   class TargetObj : public ffi::Object {
    public:
     explicit TargetObj(TargetKind kind) : kind(std::move(kind)) {}
     explicit TargetObj(ffi::UnsafeInit tag) : kind(tag) {}
     TargetKind kind;
     ffi::Optional<ffi::ObjectRef> host;
     ffi::String tag;
     ffi::Array<ffi::String> keys;
     ffi::Map<ffi::String, Any> attrs;
   
     TVM_TIRX_DLL ffi::String str() const;
     TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any> ToConfig() const;
     ffi::Optional<Target> GetHost() const;
     int GetTargetDeviceType() const {
       if (ffi::Optional<int64_t> device_type = GetAttr<int64_t>("target_device_type")) {
         return static_cast<int>(device_type.value());
       }
       return kind->default_device_type;
     }
   
     bool HasKey(const ffi::String& query_key) const {
       return std::any_of(keys.begin(), keys.end(),
                          [&query_key](const auto& key) { return key == query_key; });
     }
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TargetObj>()
           .def_ro("kind", &TargetObj::kind)
           .def_ro("tag", &TargetObj::tag)
           .def_ro("keys", &TargetObj::keys)
           .def_ro("attrs", &TargetObj::attrs)
           .def_ro("host", &TargetObj::host);
     }
   
     template <typename TObjectRef>
     ffi::Optional<TObjectRef> GetAttr(
         const std::string& attr_key,
         ffi::Optional<TObjectRef> default_value = ffi::Optional<TObjectRef>(std::nullopt)) const {
       auto it = attrs.find(attr_key);
       if (it != attrs.end()) {
         return (*it).second.as_or_throw<ffi::Optional<TObjectRef>>();
       } else {
         return default_value;
       }
     }
     template <typename TObjectRef>
     ffi::Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
       return GetAttr<TObjectRef>(attr_key, ffi::Optional<TObjectRef>(default_value));
     }
   
     template <typename T>
     ffi::Optional<T> GetAttr(ffi::reflection::AttrKey<T> key,
                              ffi::Optional<typename ffi::reflection::AttrKey<T>::ValueType>
                                  default_value = std::nullopt) const {
       return GetAttr<T>(key.name, std::move(default_value));
     }
   
     template <typename T>
     ffi::Optional<T> GetAttr(ffi::reflection::AttrKey<T> key,
                              typename ffi::reflection::AttrKey<T>::ValueType default_value) const {
       return GetAttr(key, ffi::Optional<T>(std::move(default_value)));
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("target.Target", TargetObj, ffi::Object);
   
    private:
     mutable ffi::String str_repr_;
   
     friend class TargetInternal;
   };
   
   class Target : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL explicit Target(const ffi::String& tag_or_config_or_target_str);
     TVM_TIRX_DLL explicit Target(const ffi::Map<ffi::String, ffi::Any>& config);
     TVM_TIRX_DLL static ffi::Optional<Target> Current(bool allow_not_defined = true);
     explicit Target(Target target, ffi::Optional<Target> host) {
       ffi::ObjectPtr<TargetObj> n = ffi::make_object<TargetObj>(*target.get());
       n->host =
           host.has_value() ? ffi::Optional<ffi::ObjectRef>(std::move(host).value()) : std::nullopt;
       data_ = std::move(n);
     }
     explicit Target(ffi::UnsafeInit tag, ffi::ObjectPtr<TargetObj> n) : ffi::ObjectRef(tag) {
       data_ = std::move(n);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Target, ffi::ObjectRef, TargetObj);
   
     TVM_TIRX_DLL static Target WithHost(const Target& target, const ffi::Optional<Target>& host);
   
     Target WithoutHost() const {
       if ((*this)->GetHost()) {
         auto output = ffi::make_object<TargetObj>(*get());
         output->host = std::nullopt;
         return Target(ffi::UnsafeInit{}, std::move(output));
       } else {
         return *this;
       }
     }
   
    private:
     // enable with syntax.
     friend class TargetInternal;
     friend class With<Target>;
     TVM_TIRX_DLL void EnterWithScope();
     TVM_TIRX_DLL void ExitWithScope();
   };
   
   inline ffi::Optional<Target> TargetObj::GetHost() const { return this->host.as<Target>(); }
   
   namespace attr {
   inline constexpr ffi::reflection::AttrKey<ffi::Map<ffi::String, ffi::Any>> kBackendConfig{
       "backend_config"};
   }  // namespace attr
   
   }  // namespace tvm
   #endif  // TVM_TARGET_TARGET_H_
