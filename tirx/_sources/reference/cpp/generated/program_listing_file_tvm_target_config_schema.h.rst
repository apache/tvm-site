
.. _program_listing_file_tvm_target_config_schema.h:

Program Listing for File config_schema.h
========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_target_config_schema.h>` (``tvm/target/config_schema.h``)

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
   
   #ifndef TVM_TARGET_CONFIG_SCHEMA_H_
   #define TVM_TARGET_CONFIG_SCHEMA_H_
   
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/reflection/registry.h>
   
   #include <sstream>
   #include <string>
   #include <type_traits>
   #include <unordered_map>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   
   class ConfigSchema {
    public:
     using ConfigMap = ffi::Map<ffi::String, ffi::Any>;
     using Canonicalizer = ffi::TypedFunction<ConfigMap(ConfigMap)>;
   
     struct OptionEntry {
       ffi::String key;
       ffi::String type_str;
       ffi::TypedFunction<ffi::Any(ffi::Any)> validator;
       bool has_default = false;
       bool default_from_factory = false;
       ffi::Any default_value;
     };
   
     template <typename T, typename... Traits>
     ConfigSchema& def_option(const ffi::String& key, Traits&&... traits) {
       std::string skey(key);
       if (key_to_index_.count(skey)) {
         TVM_FFI_THROW(ValueError) << "Duplicate config option key: '" << key << "'";
       }
       key_to_index_[skey] = options_.size();
       options_.push_back(MakeEntry<T>(key, std::forward<Traits>(traits)...));
       return *this;
     }
   
     void set_canonicalizer(Canonicalizer f) { canonicalizer_ = std::move(f); }
   
     struct AttrValidator {
       ffi::TypedFunction<ffi::Any(ffi::Any)> func;
       explicit AttrValidator(ffi::TypedFunction<ffi::Any(ffi::Any)> f) : func(std::move(f)) {}
     };
   
     void set_error_on_unknown(bool value) { error_on_unknown_ = value; }
   
     ConfigMap Resolve(ConfigMap config) const {
       ConfigMap result;
   
       // Step 1: validate/coerce and materialize options in declaration order
       for (const auto& e : options_) {
         auto it = config.find(e.key);
         if (it != config.end()) {
           result.Set(e.key, e.validator((*it).second));
         } else if (e.has_default) {
           if (e.default_from_factory) {
             result.Set(e.key, e.default_value.cast<ffi::Function>()());
           } else {
             result.Set(e.key, e.default_value);
           }
         }
         // else: missing non-required option, stays absent
       }
   
       // Step 2: unknown-key policy
       if (error_on_unknown_) {
         for (const auto& kv : config) {
           if (!key_to_index_.count(std::string(kv.first))) {
             std::ostringstream os;
             os << "Unknown config option '" << kv.first << "'. Known options: ";
             bool first = true;
             for (const auto& e : options_) {
               if (!first) os << ", ";
               os << "'" << e.key << "'";
               first = false;
             }
             TVM_FFI_THROW(ValueError) << os.str();
           }
         }
       }
   
       // Step 3: whole-object canonicalization (last)
       if (canonicalizer_ != nullptr) {
         result = canonicalizer_(result);
       }
   
       return result;
     }
   
     const std::vector<OptionEntry>& ListOptions() const { return options_; }
   
     bool HasOption(const ffi::String& key) const { return key_to_index_.count(std::string(key)) > 0; }
   
    private:
     template <typename T>
     static ffi::TypedFunction<ffi::Any(ffi::Any)> MakeValidator(const ffi::String& key) {
       return ffi::TypedFunction<ffi::Any(ffi::Any)>([key](ffi::Any val) -> ffi::Any {
         auto opt = val.try_cast<T>();
         if (!opt.has_value()) {
           TVM_FFI_THROW(TypeError) << "Option '" << key << "': expected type '"
                                    << ffi::TypeTraits<T>::TypeStr() << "' but got '"
                                    << val.GetTypeKey() << "'";
         }
         return ffi::Any(opt.value());
       });
     }
   
     template <typename Trait>
     static void ApplyTrait(OptionEntry* entry, ffi::reflection::FieldInfoBuilder* info,
                            Trait&& trait) {
       using T = std::decay_t<Trait>;
       if constexpr (std::is_same_v<T, AttrValidator>) {
         entry->validator = std::move(trait.func);
       } else if constexpr (std::is_base_of_v<ffi::reflection::InfoTrait, T>) {
         trait.Apply(info);
       } else if constexpr (std::is_same_v<T, const char*> || std::is_same_v<T, char*>) {
         const char* doc = trait;
         if (doc != nullptr && doc[0] != '\0') {
           info->doc = TVMFFIByteArray{doc, std::char_traits<char>::length(doc)};
         }
       }
     }
   
     template <typename T, typename... Traits>
     OptionEntry MakeEntry(const ffi::String& key, Traits&&... traits) {
       OptionEntry e;
       e.key = key;
       e.type_str = ffi::String(ffi::TypeTraits<T>::TypeStr());
       e.validator = MakeValidator<T>(key);
       // Apply traits through a temporary FieldInfoBuilder so existing
       // reflection traits (notably refl::DefaultValue) are reused unchanged.
       ffi::reflection::FieldInfoBuilder info{};
       info.flags = 0;
       info.default_value_or_factory = ffi::AnyView(nullptr).CopyToTVMFFIAny();
       info.doc = TVMFFIByteArray{nullptr, 0};
       (ApplyTrait(&e, &info, std::forward<Traits>(traits)), ...);
       if (info.flags & kTVMFFIFieldFlagBitMaskHasDefault) {
         e.has_default = true;
         e.default_from_factory = (info.flags & kTVMFFIFieldFlagBitMaskDefaultFromFactory) != 0;
         e.default_value = ffi::AnyView::CopyFromTVMFFIAny(info.default_value_or_factory);
         // Release the extra ref created by CopyToTVMFFIAny in Apply
         if (info.default_value_or_factory.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin) {
           ffi::detail::ObjectUnsafe::DecRefObjectHandle(info.default_value_or_factory.v_obj);
         }
       }
       return e;
     }
   
     std::vector<OptionEntry> options_;
     std::unordered_map<std::string, size_t> key_to_index_;
     Canonicalizer canonicalizer_{nullptr};
     bool error_on_unknown_ = true;
   };
   
   }  // namespace tvm
   
   #endif  // TVM_TARGET_CONFIG_SCHEMA_H_
