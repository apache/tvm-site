
.. _program_listing_file_tvm_ffi_enum.h:

Program Listing for File enum.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_enum.h>` (``tvm/ffi/enum.h``)

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
   
   #ifndef TVM_FFI_ENUM_H_
   #define TVM_FFI_ENUM_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/container/dict.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/reflection/accessor.h>
   #include <tvm/ffi/string.h>
   
   #include <cstdint>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   class Enum;
   
   class EnumObj : public Object {
    public:
     int64_t _value;
     String _name;
   
     EnumObj() = default;
     EnumObj(int64_t value, String name) : _value(value), _name(std::move(name)) {}
   
     template <typename EnumClsObj>
     static Enum Get(const String& name);
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
     TVM_FFI_DECLARE_OBJECT_INFO("ffi.Enum", EnumObj, Object);
   
    private:
     static const TVMFFITypeAttrColumn* GetEnumEntriesColumn() {
       constexpr TVMFFIByteArray kAttrName =
           reflection::AsByteArray(reflection::type_attr::kEnumEntries);
       static const TVMFFITypeAttrColumn* column = TVMFFIGetTypeAttrColumn(&kAttrName);
       return column;
     }
   };
   
   class Enum : public ObjectRef {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Enum, ObjectRef, EnumObj);
   };
   
   template <typename EnumClsObj>
   inline Enum EnumObj::Get(const String& name) {
     static_assert(std::is_base_of_v<EnumObj, EnumClsObj>,
                   "EnumObj::Get<T> requires T to be a subclass of EnumObj");
     const TVMFFITypeAttrColumn* column = GetEnumEntriesColumn();
     int32_t type_index = EnumClsObj::RuntimeTypeIndex();
     if (column != nullptr) {
       int32_t offset = type_index - column->begin_index;
       if (offset >= 0 && offset < column->size) {
         const TVMFFIAny* stored = &column->data[offset];
         if (stored->type_index != kTVMFFINone) {
           Dict<String, Enum> entries = AnyView::CopyFromTVMFFIAny(*stored).cast<Dict<String, Enum>>();
           auto it = entries.find(name);
           if (it != entries.end()) {
             return (*it).second;
           }
         }
       }
     }
     TVM_FFI_THROW(RuntimeError) << "Enum `" << EnumClsObj::_type_key << "` has no instance named `"
                                 << name << "`";
     TVM_FFI_UNREACHABLE();
   }
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_ENUM_H_
