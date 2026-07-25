
.. _program_listing_file_tvm_ffi_cast.h:

Program Listing for File cast.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_cast.h>` (``tvm/ffi/cast.h``)

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
   #ifndef TVM_FFI_CAST_H_
   #define TVM_FFI_CAST_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/optional.h>
   
   #include <type_traits>
   
   namespace tvm {
   namespace ffi {
   
   template <typename RefType, typename ObjectType>
   inline RefType GetRef(const ObjectType* ptr) {
     if constexpr (object_ref_contains_v<RefType, ObjectType>) {
       if constexpr (is_optional_type_v<RefType> || RefType::_type_is_nullable) {
         if (ptr == nullptr) {
           return details::ObjectUnsafe::ObjectRefFromObjectPtr<RefType>(nullptr);
         }
       } else {
         TVM_FFI_ICHECK_NOTNULL(ptr);
       }
       return details::ObjectUnsafe::ObjectRefFromObjectPtr<RefType>(
           details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(
               const_cast<Object*>(static_cast<const Object*>(ptr))));
     } else {
       static_assert(object_ref_contains_v<RefType, ObjectType>,
                     "GetRef requires RefType to contain every ObjectType instance; specialize "
                     "object_ref_contains_v for statically safe typed refs or use "
                     "ObjectRef::as<RefType>() for runtime-dependent checks");
       TVM_FFI_UNREACHABLE();
     }
   }
   
   template <typename BaseType, typename ObjectType>
   inline ObjectPtr<BaseType> GetObjectPtr(ObjectType* ptr) {
     static_assert(std::is_base_of_v<BaseType, ObjectType>,
                   "Can only cast to the ref of same container type");
     return details::ObjectUnsafe::ObjectPtrFromUnowned<BaseType>(ptr);
   }
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_CAST_H_
