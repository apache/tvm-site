
.. _program_listing_file_tvm_ffi_reflection_enum_def.h:

Program Listing for File enum_def.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_reflection_enum_def.h>` (``tvm/ffi/reflection/enum_def.h``)

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
   
   #ifndef TVM_FFI_REFLECTION_ENUM_DEF_H_
   #define TVM_FFI_REFLECTION_ENUM_DEF_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/container/dict.h>
   #include <tvm/ffi/container/list.h>
   #include <tvm/ffi/enum.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/reflection/accessor.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ffi/string.h>
   
   #include <cstdint>
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   namespace reflection {
   
   template <typename EnumClsObj, typename = std::enable_if_t<std::is_base_of_v<EnumObj, EnumClsObj>>>
   class EnumDef : public ReflectionDefBase {
    public:
     explicit EnumDef(const char* instance_name)
         : type_index_(EnumClsObj::RuntimeTypeIndex()), name_(instance_name) {
       Dict<String, Enum> entries = EnsureEntriesDict();
       String name_str(name_);
       if (entries.count(name_str) != 0) {
         TVM_FFI_THROW(RuntimeError) << "Duplicate enum entry `" << name_ << "` for type `"
                                     << EnumClsObj::_type_key << "`";
       }
       ordinal_ = static_cast<int64_t>(entries.size());
       ObjectPtr<EnumClsObj> obj = make_object<EnumClsObj>();
       obj->_value = ordinal_;
       obj->_name = name_str;
       instance_ = Enum(ObjectPtr<EnumObj>(std::move(obj)));
       entries.Set(name_str, instance_);
       // Ensure the attrs dict exists so later ``set_attr`` calls can mutate it.
       EnsureAttrsDict();
     }
   
     template <typename T>
     EnumDef& set_attr(const char* attr_name, T value) {
       Dict<String, List<Any>> attrs = EnsureAttrsDict();
       String attr_key(attr_name);
       List<Any> column;
       auto it = attrs.find(attr_key);
       if (it == attrs.end()) {
         column = List<Any>();
         attrs.Set(attr_key, column);
       } else {
         column = (*it).second;
       }
       while (static_cast<int64_t>(column.size()) <= ordinal_) {
         column.push_back(Any(nullptr));
       }
       column.Set(ordinal_, Any(std::move(value)));
       return *this;
     }
   
     Enum instance() const { return instance_; }
   
     int64_t ordinal() const { return ordinal_; }
   
    private:
     Dict<String, Enum> EnsureEntriesDict() {
       return EnsureDict<Dict<String, Enum>>(type_attr::kEnumEntries);
     }
   
     Dict<String, List<Any>> EnsureAttrsDict() {
       return EnsureDict<Dict<String, List<Any>>>(type_attr::kEnumAttrs);
     }
   
     template <typename DictT>
     DictT EnsureDict(const char* attr_name) {
       TVMFFIByteArray name_array = {attr_name, std::char_traits<char>::length(attr_name)};
       const TVMFFITypeAttrColumn* column = TVMFFIGetTypeAttrColumn(&name_array);
       if (column != nullptr) {
         int32_t offset = type_index_ - column->begin_index;
         if (offset >= 0 && offset < column->size) {
           const TVMFFIAny* stored = &column->data[offset];
           if (stored->type_index != kTVMFFINone) {
             return AnyView::CopyFromTVMFFIAny(*stored).cast<DictT>();
           }
         }
       }
       DictT fresh;
       TVMFFIAny value_any = AnyView(fresh).CopyToTVMFFIAny();
       TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name_array, &value_any));
       return fresh;
     }
   
     int32_t type_index_;
     const char* name_;
     int64_t ordinal_;
     Enum instance_;
   };
   
   }  // namespace reflection
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_REFLECTION_ENUM_DEF_H_
