
.. _program_listing_file_tvm_ir_cow.h:

Program Listing for File cow.h
==============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_cow.h>` (``tvm/ir/cow.h``)

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
   
   #ifndef TVM_IR_COW_H_
   #define TVM_IR_COW_H_
   
   #include <tvm/ffi/object.h>
   
   #include <utility>
   
   namespace tvm {
   
   #ifndef TVM_DEFINE_OBJECT_REF_COW_METHOD
   #define TVM_DEFINE_OBJECT_REF_COW_METHOD(ObjectName)                       \
     static_assert(ObjectName::_type_final,                                   \
                   "TVM's CopyOnWrite may only be used for "                  \
                   "Object types that are declared as final, "                \
                   "using the TVM_FFI_DECLARE_OBJECT_INFO_FINAL macro.");     \
     ObjectName* CopyOnWrite() {                                              \
       TVM_FFI_ICHECK(data_ != nullptr);                                      \
       if (!data_.unique()) {                                                 \
         auto n = ::tvm::ffi::make_object<ObjectName>(*(operator->()));       \
         ::tvm::ffi::ObjectPtr<::tvm::ffi::Object>(std::move(n)).swap(data_); \
       }                                                                      \
       return static_cast<ObjectName*>(data_.get());                          \
     }
   #endif  // TVM_DEFINE_OBJECT_REF_COW_METHOD
   
   }  // namespace tvm
   
   #endif  // TVM_IR_COW_H_
