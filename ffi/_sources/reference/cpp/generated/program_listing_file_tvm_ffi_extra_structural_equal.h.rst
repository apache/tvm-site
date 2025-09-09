
.. _program_listing_file_tvm_ffi_extra_structural_equal.h:

Program Listing for File structural_equal.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_structural_equal.h>` (``tvm/ffi/extra/structural_equal.h``)

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
   #ifndef TVM_FFI_EXTRA_STRUCTURAL_EQUAL_H_
   #define TVM_FFI_EXTRA_STRUCTURAL_EQUAL_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/extra/base.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/reflection/access_path.h>
   
   namespace tvm {
   namespace ffi {
   class StructuralEqual {
    public:
     TVM_FFI_EXTRA_CXX_API static bool Equal(const Any& lhs, const Any& rhs,
                                             bool map_free_vars = false,
                                             bool skip_tensor_content = false);
     TVM_FFI_EXTRA_CXX_API static Optional<reflection::AccessPathPair> GetFirstMismatch(
         const Any& lhs, const Any& rhs, bool map_free_vars = false, bool skip_tensor_content = false);
   
     /*
      * \brief Compare two Any values for structural equality.
      * \param lhs The left hand side Any object.
      * \param rhs The right hand side Any object.
      * \return True if the two Any values are structurally equal, false otherwise.
      */
     TVM_FFI_INLINE bool operator()(const Any& lhs, const Any& rhs) const {
       return Equal(lhs, rhs, false, true);
     }
   };
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_STRUCTURAL_EQUAL_H_
