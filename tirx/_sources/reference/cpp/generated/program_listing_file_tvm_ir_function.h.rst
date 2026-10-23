
.. _program_listing_file_tvm_ir_function.h:

Program Listing for File function.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_function.h>` (``tvm/ir/function.h``)

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
   
   #ifndef TVM_IR_FUNCTION_H_
   #define TVM_IR_FUNCTION_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/attrs.h>
   #include <tvm/ir/expr.h>
   
   #include <string>
   #include <type_traits>
   
   namespace tvm {
   
   class Target;
   
   enum class CallingConv : int {
     kDefault = 0,
     kCPackedFunc = 1,
     kDeviceKernelLaunch = 2,
   };
   
   enum class LinkageType : int {
     kInternal = 0,
     kExternal = 1
   };
   
   class BaseFuncNode : public ExprNode {
    public:
     DictAttrs attrs;
   
     template <typename TObjectRef>
     ffi::Optional<TObjectRef> GetAttr(const std::string& attr_key,
                                       ffi::Optional<TObjectRef> default_value = std::nullopt) const {
       return attrs.GetAttr(attr_key, default_value);
     }
     template <typename T>
     ffi::Optional<T> GetAttr(ffi::reflection::AttrKey<T> key,
                              ffi::Optional<typename ffi::reflection::AttrKey<T>::ValueType>
                                  default_value = std::nullopt) const {
       return attrs.GetAttr(key, std::move(default_value));
     }
     template <typename T>
     ffi::Optional<T> GetAttr(ffi::reflection::AttrKey<T> key,
                              typename ffi::reflection::AttrKey<T>::ValueType default_value) const {
       return GetAttr(key, ffi::Optional<T>(std::move(default_value)));
     }
   
     // variant that uses TObjectRef to enable implicit conversion to default value.
     template <typename TObjectRef>
     ffi::Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
       return GetAttr<TObjectRef>(attr_key, ffi::Optional<TObjectRef>(default_value));
     }
   
     bool HasNonzeroAttr(const std::string& attr_key) const { return attrs.HasNonzeroAttr(attr_key); }
     LinkageType GetLinkageType() const;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BaseFuncNode>().def_ro("attrs", &BaseFuncNode::attrs);
     }
   
     static constexpr const uint32_t _type_child_slots = 2;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.BaseFunc", BaseFuncNode, ExprNode);
   };
   
   class BaseFunc : public Expr {
    public:
     explicit BaseFunc(ffi::ObjectPtr<BaseFuncNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BaseFunc, Expr, BaseFuncNode);
   };
   
   namespace attr {
   inline constexpr ffi::reflection::AttrKey<CallingConv> kCallingConv{"calling_conv"};
   
   inline constexpr ffi::reflection::AttrKey<Target> kTarget{"target"};
   
   inline constexpr ffi::reflection::AttrKey<ffi::String> kGlobalSymbol{"global_symbol"};
   
   inline constexpr ffi::reflection::AttrKey<ffi::String> kScriptNamespace{"script.namespace"};
   
   }  // namespace attr
   
   inline LinkageType BaseFuncNode::GetLinkageType() const {
     if (GetAttr(tvm::attr::kGlobalSymbol))
       return LinkageType::kExternal;
     else
       return LinkageType::kInternal;
   }
   
   }  // namespace tvm
   #endif  // TVM_IR_FUNCTION_H_
