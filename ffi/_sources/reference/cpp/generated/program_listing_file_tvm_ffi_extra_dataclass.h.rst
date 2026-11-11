
.. _program_listing_file_tvm_ffi_extra_dataclass.h:

Program Listing for File dataclass.h
====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_dataclass.h>` (``tvm/ffi/extra/dataclass.h``)

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
   #ifndef TVM_FFI_EXTRA_DATACLASS_H_
   #define TVM_FFI_EXTRA_DATACLASS_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/variant.h>
   #include <tvm/ffi/extra/base.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/reflection/accessor.h>
   #include <tvm/ffi/string.h>
   
   #include <ostream>
   
   namespace tvm {
   namespace ffi {
   
   TVM_FFI_EXTRA_CXX_API Any DeepCopy(const Any& value);
   
   TVM_FFI_EXTRA_CXX_API String ReprPrint(const Any& value);
   
   TVM_FFI_EXTRA_CXX_API int64_t RecursiveHash(const Any& value);
   
   TVM_FFI_EXTRA_CXX_API bool RecursiveEq(const Any& lhs, const Any& rhs);
   
   TVM_FFI_EXTRA_CXX_API bool RecursiveLt(const Any& lhs, const Any& rhs);
   
   TVM_FFI_EXTRA_CXX_API bool RecursiveLe(const Any& lhs, const Any& rhs);
   
   TVM_FFI_EXTRA_CXX_API bool RecursiveGt(const Any& lhs, const Any& rhs);
   
   TVM_FFI_EXTRA_CXX_API bool RecursiveGe(const Any& lhs, const Any& rhs);
   
   // std::ostream overloads
   
   inline std::ostream& operator<<(std::ostream& os, const Any& value) {
     return os << ReprPrint(value);
   }
   
   inline std::ostream& operator<<(std::ostream& os, const ObjectRef& value) {
     return os << ReprPrint(Any(value));
   }
   
   template <typename... V>
   inline std::ostream& operator<<(std::ostream& os, const Variant<V...>& value) {
     return os << ReprPrint(Any(value));
   }
   
   template <typename T>
   inline std::ostream& operator<<(std::ostream& os, const Optional<T>& value) {
     return os << ReprPrint(Any(value));
   }
   
   namespace type_attr {
   inline constexpr reflection::AttrKey<
       TypedFunction<String(const Object*, const TypedFunction<String(AnyView)>&)>>
       kRepr{"__ffi_repr__"};
   
   inline constexpr reflection::AttrKey<
       TypedFunction<int64_t(const Object*, const TypedFunction<int64_t(AnyView)>&)>>
       kHash{"__ffi_hash__"};
   inline constexpr reflection::AttrKey<
       TypedFunction<bool(const Object*, const Object*, const TypedFunction<bool(AnyView, AnyView)>&)>>
       kEq{"__ffi_eq__"};
   inline constexpr reflection::AttrKey<TypedFunction<int32_t(
       const Object*, const Object*, const TypedFunction<int32_t(AnyView, AnyView)>&)>>
       kCompare{"__ffi_compare__"};
   }  // namespace type_attr
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_DATACLASS_H_
