
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
   #include <tvm/ffi/container/list.h>
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
   class IntEnum;
   class StrEnum;
   
   class EnumStateObj : public Object {
    public:
     List<ObjectRef> entries;
     Dict<Any, ObjectRef> indexes;
     Dict<String, Dict<ObjectRef, Any>> attrs;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.EnumState", EnumStateObj, Object);
   };
   
   class EnumState : public ObjectRef {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(EnumState, ObjectRef, EnumStateObj);
   };
   
   class EnumObj : public Object {
    public:
     int64_t _int_index = 0;
     String _str_index;
   
     EnumObj() = default;
     EnumObj(int64_t int_index, String str_index)
         : _int_index(int_index), _str_index(std::move(str_index)) {}
   
     // NOLINTBEGIN(bugprone-reserved-identifier)
     template <typename EnumClsObj>
     static Enum _GetByIntIndex(int64_t index);
     template <typename EnumClsObj>
     static Enum _GetByStrIndex(const String& index);
     static Enum _GetByIntIndex(int32_t type_index, int64_t index);
     static Enum _GetByStrIndex(int32_t type_index, const String& index);
     // NOLINTEND(bugprone-reserved-identifier)
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
     TVM_FFI_DECLARE_OBJECT_INFO("ffi.Enum", EnumObj, Object);
   
    private:
     template <typename Index>
     static Enum GetByIndex(int32_t type_index, const Index& index);
   };
   
   class Enum : public ObjectRef {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Enum, ObjectRef, EnumObj);
   };
   
   class IntEnumObj : public EnumObj {
    public:
     TVM_FFI_DECLARE_OBJECT_INFO("ffi.IntEnum", IntEnumObj, EnumObj);
   };
   
   class IntEnum : public Enum {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntEnum, Enum, IntEnumObj);
   };
   
   class StrEnumObj : public EnumObj {
    public:
     TVM_FFI_DECLARE_OBJECT_INFO("ffi.StrEnum", StrEnumObj, EnumObj);
   };
   
   class StrEnum : public Enum {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StrEnum, Enum, StrEnumObj);
   };
   
   template <typename Index>
   inline Enum EnumObj::GetByIndex(int32_t type_index, const Index& index) {
     static reflection::TypeAttrColumn state_column(reflection::type_attr::kEnumState);
     if (AnyView value = state_column[type_index]; value != nullptr) {
       EnumState state = value.cast<EnumState>();
       if (auto entry = state->indexes.Get(Any(index))) return entry->as_or_throw<Enum>();
     }
     TVM_FFI_THROW(ValueError) << "Enum `" << TypeIndexToTypeKey(type_index)
                               << "` has no instance with index " << index;
     TVM_FFI_UNREACHABLE();
   }
   
   // NOLINTBEGIN(bugprone-reserved-identifier)
   template <typename EnumClsObj>
   inline Enum EnumObj::_GetByIntIndex(int64_t index) {
     static_assert(std::is_base_of_v<EnumObj, EnumClsObj>);
     return _GetByIntIndex(EnumClsObj::_GetOrAllocRuntimeTypeIndex(), index);
   }
   
   template <typename EnumClsObj>
   inline Enum EnumObj::_GetByStrIndex(const String& index) {
     static_assert(std::is_base_of_v<EnumObj, EnumClsObj>);
     return _GetByStrIndex(EnumClsObj::_GetOrAllocRuntimeTypeIndex(), index);
   }
   
   inline Enum EnumObj::_GetByIntIndex(int32_t type_index, int64_t index) {
     return GetByIndex(type_index, index);
   }
   
   inline Enum EnumObj::_GetByStrIndex(int32_t type_index, const String& index) {
     return GetByIndex(type_index, index);
   }
   // NOLINTEND(bugprone-reserved-identifier)
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_ENUM_H_
