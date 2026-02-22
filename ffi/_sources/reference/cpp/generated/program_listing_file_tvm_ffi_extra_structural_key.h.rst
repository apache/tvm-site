
.. _program_listing_file_tvm_ffi_extra_structural_key.h:

Program Listing for File structural_key.h
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_structural_key.h>` (``tvm/ffi/extra/structural_key.h``)

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
   #ifndef TVM_FFI_EXTRA_STRUCTURAL_KEY_H_
   #define TVM_FFI_EXTRA_STRUCTURAL_KEY_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/extra/structural_equal.h>
   #include <tvm/ffi/extra/structural_hash.h>
   
   #include <cstddef>
   #include <cstdint>
   #include <functional>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   class StructuralKeyObj : public Object {
    public:
     Any key;
     int64_t hash_i64{0};
   
     // Default constructor to support reflection-based initialization.
     StructuralKeyObj() = default;
     explicit StructuralKeyObj(Any key)
         : key(std::move(key)), hash_i64(static_cast<int64_t>(StructuralHash::Hash(this->key))) {}
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.StructuralKey", StructuralKeyObj, Object);
   };
   
   class StructuralKey : public ObjectRef {
    public:
     explicit StructuralKey(Any key) : ObjectRef(make_object<StructuralKeyObj>(std::move(key))) {}
   
     bool operator==(const StructuralKey& other) const {
       if (this->same_as(other)) {
         return true;
       }
       if (this->get()->hash_i64 != other->hash_i64) {
         return false;
       }
       return StructuralEqual::Equal(this->get()->key, other->key);
     }
     bool operator!=(const StructuralKey& other) const { return !(*this == other); }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralKey, ObjectRef, StructuralKeyObj);
   };
   
   }  // namespace ffi
   }  // namespace tvm
   
   namespace std {
   template <>
   struct hash<tvm::ffi::StructuralKey> {
     size_t operator()(const tvm::ffi::StructuralKey& key) const {
       return static_cast<size_t>(static_cast<uint64_t>(key->hash_i64));
     }
   };
   }  // namespace std
   
   #endif  // TVM_FFI_EXTRA_STRUCTURAL_KEY_H_
