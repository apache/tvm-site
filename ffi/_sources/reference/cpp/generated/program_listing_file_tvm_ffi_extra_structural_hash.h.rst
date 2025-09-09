
.. _program_listing_file_tvm_ffi_extra_structural_hash.h:

Program Listing for File structural_hash.h
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_structural_hash.h>` (``tvm/ffi/extra/structural_hash.h``)

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
   #ifndef TVM_FFI_EXTRA_STRUCTURAL_HASH_H_
   #define TVM_FFI_EXTRA_STRUCTURAL_HASH_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/extra/base.h>
   
   namespace tvm {
   namespace ffi {
   
   class StructuralHash {
    public:
     TVM_FFI_EXTRA_CXX_API static uint64_t Hash(const Any& value, bool map_free_vars = false,
                                                bool skip_tensor_content = false);
     TVM_FFI_INLINE uint64_t operator()(const Any& value) const { return Hash(value); }
   };
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_STRUCTURAL_HASH_H_
