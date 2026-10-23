
.. _program_listing_file_tvm_tirx_op_memory.h:

Program Listing for File memory.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_op_memory.h>` (``tvm/tirx/op/memory.h``)

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
   
   #ifndef TVM_TIRX_OP_MEMORY_H_
   #define TVM_TIRX_OP_MEMORY_H_
   
   #include <tvm/ir/op.h>
   #include <tvm/ir/prim/expr.h>
   
   namespace tvm {
   namespace tirx {
   
   TVM_TIRX_DLL const Op& alloc_tensor_op();
   
   TVM_TIRX_DLL const Op& decl_tensor_op();
   
   TVM_TIRX_DLL const Op& reinterpret_op();
   
   TVM_TIRX_DLL Expr reinterpret(Type target_ty, Expr value, Location loc = UnknownLoc());
   
   inline PrimExpr reinterpret(PrimType t, PrimExpr value, Location loc = UnknownLoc()) {
     return reinterpret(Type(std::move(t)), Expr(std::move(value)), std::move(loc))
         .as_or_throw<PrimExpr>();
   }
   
   inline Expr ConstHandle(int64_t value, Location loc = UnknownLoc()) {
     return reinterpret(PtrType::VoidPtrType(), IntImm(PrimType::UInt(64), value, loc), loc);
   }
   
   TVM_TIRX_DLL const Op& address_of_op();
   
   TVM_TIRX_DLL const Op& isnullptr_op();
   
   TVM_TIRX_DLL const Op& prefetch_op();
   
   TVM_TIRX_DLL const Op& ptr_byte_offset_op();
   
   TVM_TIRX_DLL const Op& static_handle_op();
   
   TVM_TIRX_DLL const Op& handle_add_byte_offset_op();
   
   TVM_TIRX_DLL const Op& atomic_add_op();
   
   TVM_TIRX_DLL const Op& assume_aligned_op();
   
   TVM_TIRX_DLL const Op& undef_op();
   
   TVM_TIRX_DLL const Op& masked_load_op();
   
   TVM_TIRX_DLL const Op& masked_store_op();
   
   TVM_TIRX_DLL const Op& tensor_data_ptr_op();
   
   }  // namespace tirx
   }  // namespace tvm
   
   #endif  // TVM_TIRX_OP_MEMORY_H_
