
.. _program_listing_file_tvm_ir_unique_name_supply.h:

Program Listing for File unique_name_supply.h
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_unique_name_supply.h>` (``tvm/ir/unique_name_supply.h``)

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
   
   #ifndef TVM_IR_UNIQUE_NAME_SUPPLY_H_
   #define TVM_IR_UNIQUE_NAME_SUPPLY_H_
   
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/expr.h>
   
   #include <cstdint>
   #include <string>
   #include <utility>
   
   namespace tvm {
   
   class UniqueNameSupplyObj : public ffi::Object {
    public:
     UniqueNameSupplyObj() = default;
   
     UniqueNameSupplyObj(const ffi::String& prefix, ffi::Map<ffi::String, int64_t> name_map)
         : prefix_(prefix), name_map(std::move(name_map)) {}
   
     TVM_TIRX_DLL ffi::String FreshName(const ffi::String& name, bool add_prefix = true,
                                        bool add_underscore = true);
   
     TVM_TIRX_DLL ffi::String ReserveName(const ffi::String& name, bool add_prefix = true);
   
     TVM_TIRX_DLL bool ContainsName(const ffi::String& name, bool add_prefix = true);
   
     // Prefix for all GlobalVar names. It can be empty.
     std::string prefix_;
   
     static constexpr const bool _type_mutable = true;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<UniqueNameSupplyObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.UniqueNameSupply", UniqueNameSupplyObj, ffi::Object);
   
    private:
     ffi::String AddPrefixToName(const ffi::String& name);
   
     std::string GetUniqueName(std::string name, bool add_underscore = true);
   
     ffi::Map<ffi::String, int64_t> name_map;
   };
   
   class UniqueNameSupply : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL explicit UniqueNameSupply(const ffi::String& prefix = "",
                                            ffi::Map<ffi::String, int64_t> name_map = {});
   
     template <typename Iter, typename Lambda>
     UniqueNameSupply(Iter begin, Iter end, Lambda f) : UniqueNameSupply("") {
       for (auto it = begin; it != end; ++it) {
         this->operator->()->ReserveName(f(*it), false);
       }
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(UniqueNameSupply, ffi::ObjectRef,
                                                   UniqueNameSupplyObj);
   };
   
   }  // namespace tvm
   
   #endif  // TVM_IR_UNIQUE_NAME_SUPPLY_H_
