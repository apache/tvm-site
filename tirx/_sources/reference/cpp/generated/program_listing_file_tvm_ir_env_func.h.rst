
.. _program_listing_file_tvm_ir_env_func.h:

Program Listing for File env_func.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_env_func.h>` (``tvm/ir/env_func.h``)

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
   
   #ifndef TVM_IR_ENV_FUNC_H_
   #define TVM_IR_ENV_FUNC_H_
   
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/runtime/base.h>
   
   #include <string>
   #include <utility>
   
   namespace tvm {
   
   class EnvFuncObj : public ffi::Object {
    public:
     ffi::String name;
     ffi::Function func;
     EnvFuncObj() {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       // func do not participate in structural equal and hash.
       refl::ObjectDef<EnvFuncObj>()
           .def_ro("name", &EnvFuncObj::name)
           .def_ro("func", &EnvFuncObj::func, refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.EnvFunc", EnvFuncObj, ffi::Object);
   };
   
   class EnvFunc : public ffi::ObjectRef {
    public:
     static constexpr bool _type_is_nullable = false;
     explicit EnvFunc(ffi::UnsafeInit tag, ffi::ObjectPtr<ffi::Object> n) : ffi::ObjectRef(tag) {
       data_ = std::move(n);
     }
     explicit EnvFunc(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
     const EnvFuncObj* operator->() const { return static_cast<const EnvFuncObj*>(get()); }
     template <typename... Args>
     ffi::Any operator()(Args&&... args) const {
       const EnvFuncObj* n = operator->();
       TVM_FFI_ICHECK(n != nullptr);
       return n->func(std::forward<Args>(args)...);
     }
     TVM_TIRX_DLL static EnvFunc Get(const ffi::String& name);
     using ContainerType = EnvFuncObj;
   };
   
   template <typename FType>
   class TypedEnvFunc;
   
   template <typename R, typename... Args>
   class TypedEnvFunc<R(Args...)> : public ffi::ObjectRef {
    public:
     using TSelf = TypedEnvFunc<R(Args...)>;
     static constexpr bool _type_is_nullable = false;
     TypedEnvFunc(const EnvFunc& other) : ffi::ObjectRef(other) {}
     explicit TypedEnvFunc(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
     TSelf& operator=(const EnvFunc& other) {
       ffi::ObjectRef::operator=(other);
       return *this;
     }
     const EnvFuncObj* operator->() const { return static_cast<const EnvFuncObj*>(get()); }
     R operator()(Args... args) const {
       const EnvFuncObj* n = operator->();
       TVM_FFI_ICHECK(n != nullptr);
       if constexpr (std::is_same_v<R, void>) {
         n->func(std::forward<Args>(args)...);
       } else {
         ffi::Any res = n->func(std::forward<Args>(args)...);
         if constexpr (std::is_same_v<R, ffi::Any>) {
           return res;
         } else {
           return std::move(res).cast<R>();
         }
       }
     }
     using ContainerType = EnvFuncObj;
   };
   
   }  // namespace tvm
   #endif  // TVM_IR_ENV_FUNC_H_
