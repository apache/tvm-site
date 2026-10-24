
.. _program_listing_file_tvm_ir_transform.h:

Program Listing for File transform.h
====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_transform.h>` (``tvm/ir/transform.h``)

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
   
   #ifndef TVM_IR_TRANSFORM_H_
   #define TVM_IR_TRANSFORM_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/reflection/creator.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ffi/rvalue_ref.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/module.h>
   #include <tvm/ir/with_context.h>
   
   #include <functional>
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace transform {
   
   class PassInfo;
   class PassContextObj;
   
   class PassContext : public ffi::ObjectRef {
    public:
     static constexpr bool _type_is_nullable = false;
     explicit PassContext(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
     explicit PassContext(ffi::UnsafeInit tag, ffi::ObjectPtr<PassContextObj> n);
     const PassContextObj* operator->() const;
     PassContextObj* operator->();
   
     static PassContext Create();
     TVM_TIRX_DLL static PassContext Current();
   
     TVM_TIRX_DLL static ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::String>> ListConfigs();
   
     TVM_TIRX_DLL void InstrumentEnterPassContext();
   
     TVM_TIRX_DLL void InstrumentExitPassContext();
   
     TVM_TIRX_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
   
     void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
   
     TVM_TIRX_DLL bool PassEnabled(const PassInfo& info) const;
   
     template <typename ValueType>
     static void RegisterConfigOption(const char* key) {
       if constexpr (std::is_base_of_v<ffi::ObjectRef, ValueType>) {
         // Static-init blocks may run before the cached runtime type index is initialized.
         int32_t tindex = ValueType::ContainerType::_GetOrAllocRuntimeTypeIndex();
         auto type_key = ffi::TypeIndexToTypeKey(tindex);
         auto legalization = [=](ffi::Any value) -> ffi::Any {
           if (auto opt_map = value.try_cast<ffi::Map<ffi::String, ffi::Any>>()) {
             return ffi::reflection::ObjectCreator(type_key)(opt_map.value());
           } else {
             auto opt_val = value.try_cast<ValueType>();
             if (!opt_val.has_value()) {
               TVM_FFI_THROW(AttributeError)
                   << "Expect config " << key << " to have type " << type_key << ", but instead get "
                   << ffi::detail::AnyUnsafe::GetMismatchTypeInfo<ValueType>(value);
             }
             return *opt_val;
           }
         };
         RegisterConfigOption(key, type_key, legalization);
       } else {
         // non-object type, do not support implicit conversion from map
         std::string type_str = ffi::TypeTraits<ValueType>::TypeStr();
         auto legalization = [=](ffi::Any value) -> ffi::Any {
           auto opt_val = value.try_cast<ValueType>();
           if (!opt_val.has_value()) {
             TVM_FFI_THROW(AttributeError)
                 << "Expect config " << key << " to have type " << type_str << ", but instead get "
                 << ffi::detail::AnyUnsafe::GetMismatchTypeInfo<ValueType>(value);
             TVM_FFI_UNREACHABLE();
           } else {
             return *opt_val;
           }
         };
         RegisterConfigOption(key, type_str, legalization);
       }
     }
   
     // accessor.
     using ContainerType = PassContextObj;
     class Internal;
   
    private:
     // The entry of a pass context scope.
     TVM_TIRX_DLL void EnterWithScope();
     // The exit of a pass context scope.
     TVM_TIRX_DLL void ExitWithScope();
     // Register configuration key value type.
     TVM_TIRX_DLL static void RegisterConfigOption(
         const char* key, ffi::String value_type_str,
         ffi::TypedFunction<ffi::Any(ffi::Any)> legalization);
   
     // Classes to get the Python `with` like syntax.
     friend class Internal;
     friend class With<PassContext>;
   };
   
   template <typename TConfig>
   inline TConfig PassConfigWithDefaults() {
     static_assert(std::is_base_of_v<ffi::ObjectRef, TConfig>,
                   "Can only create ObjectRef-derived types");
     using ContainerType = typename TConfig::ContainerType;
     static auto finit_object = ffi::Function::GetGlobalRequired("ffi.MakeObjectFromPackedArgs");
     ffi::AnyView packed_args[1];
     packed_args[0] = ContainerType::RuntimeTypeIndex();
     ffi::Any rv;
     finit_object.CallPacked(ffi::PackedArgs(packed_args, 1), &rv);
     return rv.cast<TConfig>();
   }
   
   class PassInfoObj : public ffi::Object {
    public:
     int opt_level;
   
     ffi::String name;
   
     PassInfoObj() = default;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PassInfoObj>()
           .def_ro("opt_level", &PassInfoObj::opt_level)
           .def_ro("name", &PassInfoObj::name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("transform.PassInfo", PassInfoObj, ffi::Object);
   };
   
   class PassInfo : public ffi::ObjectRef {
    public:
     PassInfo(int opt_level, ffi::String name) {
       auto pass_info = ffi::make_object<PassInfoObj>();
       pass_info->opt_level = opt_level;
       pass_info->name = std::move(name);
       data_ = std::move(pass_info);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PassInfo, ffi::ObjectRef, PassInfoObj);
   };
   
   class PassObj : public ffi::Object {
    public:
     virtual ~PassObj() {}
     virtual PassInfo Info() const = 0;
   
     IRModule operator()(IRModule mod) const {
       return this->operator()(std::move(mod), PassContext::Current());
     }
   
     virtual IRModule operator()(IRModule mod, const PassContext& pass_ctx) const = 0;
     TVM_FFI_DECLARE_OBJECT_INFO("transform.Pass", PassObj, ffi::Object);
   };
   
   class Pass : public ffi::ObjectRef {
    public:
     IRModule operator()(IRModule mod) const {
       return this->operator()(std::move(mod), PassContext::Current());
     }
   
     TVM_TIRX_DLL IRModule operator()(IRModule mod, const PassContext& pass_ctx) const;
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Pass, ffi::ObjectRef, PassObj);
   
    private:
     IRModule static AssertImmutableModule(const IRModule& mod, const PassObj* node,
                                           const PassContext& pass_ctx);
   };
   
   class SequentialObj : public PassObj {
    public:
     explicit SequentialObj(PassInfo pass_info) : pass_info(std::move(pass_info)) {}
     explicit SequentialObj(ffi::UnsafeInit tag) : pass_info(tag) {}
     /* \brief The pass meta data.*/
     PassInfo pass_info;
   
     tvm::ffi::Array<Pass> passes;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<SequentialObj>()
           .def_ro("pass_info", &SequentialObj::pass_info)
           .def_ro("passes", &SequentialObj::passes);
     }
   
     PassInfo Info() const override { return pass_info; }
   
     TVM_TIRX_DLL IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("transform.Sequential", SequentialObj, PassObj);
   };
   
   class Sequential : public Pass {
    public:
     TVM_TIRX_DLL Sequential(ffi::Array<Pass> passes, PassInfo pass_info);
   
     TVM_TIRX_DLL Sequential(ffi::Array<Pass> passes, ffi::String name = "sequential");
   
     explicit Sequential(ffi::UnsafeInit tag) : Pass(tag) {}
   
     const SequentialObj* operator->() const { return static_cast<const SequentialObj*>(get()); }
     using ContainerType = SequentialObj;
   };
   
   TVM_TIRX_DLL Pass
   CreateModulePass(ffi::TypedFunction<IRModule(ffi::RValueRef<IRModule>, PassContext)> pass_func,
                    int opt_level, ffi::String name);
   
   inline Pass CreateModulePass(std::function<IRModule(IRModule, PassContext)> pass_func,
                                int opt_level, ffi::String name) {
     ffi::TypedFunction<IRModule(ffi::RValueRef<IRModule>, PassContext)> callback =
         [pass_func = std::move(pass_func)](ffi::RValueRef<IRModule> mod,
                                            PassContext ctx) -> IRModule {
       return pass_func(*std::move(mod), std::move(ctx));
     };
     return CreateModulePass(std::move(callback), opt_level, std::move(name));
   }
   
   TVM_TIRX_DLL Pass PrintIR(ffi::String header = "");
   
   TVM_TIRX_DLL ffi::Error EnrichPassErrorWithContext(
       const ffi::Error& err, const IRModule& mod, ffi::String pass_name,
       ffi::Optional<GlobalVar> func = ffi::Optional<GlobalVar>(std::nullopt));
   
   class PassInstrumentObj : public ffi::Object {
    public:
     ffi::String name;
   
     virtual ~PassInstrumentObj() {}
   
     virtual void EnterPassContext() const = 0;
   
     virtual void ExitPassContext() const = 0;
   
     virtual bool ShouldRun(const IRModule& mod, const transform::PassInfo& info) const = 0;
   
     virtual void RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const = 0;
   
     virtual void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const = 0;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PassInstrumentObj>().def_ro("name", &PassInstrumentObj::name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO("transform.PassInstrument", PassInstrumentObj, ffi::Object);
   };
   
   class PassInstrument : public ffi::ObjectRef {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PassInstrument, ffi::ObjectRef, PassInstrumentObj);
   };
   
   class PassContextObj : public ffi::Object {
    public:
     int opt_level{2};
   
     ffi::Array<ffi::String> required_pass;
     ffi::Array<ffi::String> disabled_pass;
     ffi::Map<ffi::String, Any> config;
   
     ffi::Array<PassInstrument> instruments;
   
     PassContextObj() = default;
   
     template <typename T>
     ffi::Optional<T> GetConfig(const std::string& key) const {
       if (config.defined()) {
         auto it = config.find(key);
         if (it != config.end()) {
           return (*it).second.as_or_throw<ffi::Optional<T>>();
         }
       }
       return std::nullopt;
     }
   
     template <typename TConfig>
     TConfig GetConfigOrDefault(const std::string& key) const {
       auto config = GetConfig<TConfig>(key);
       if (config.has_value()) {
         return config.value();
       }
       return PassConfigWithDefaults<TConfig>();
     }
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PassContextObj>()
           .def_ro("opt_level", &PassContextObj::opt_level)
           .def_ro("required_pass", &PassContextObj::required_pass)
           .def_ro("disabled_pass", &PassContextObj::disabled_pass)
           .def_ro("instruments", &PassContextObj::instruments)
           .def_ro("config", &PassContextObj::config);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("transform.PassContext", PassContextObj, ffi::Object);
   };
   
   inline PassContext::PassContext(ffi::UnsafeInit tag, ffi::ObjectPtr<PassContextObj> n)
       : ffi::ObjectRef(tag) {
     data_ = std::move(n);
   }
   
   inline const PassContextObj* PassContext::operator->() const {
     TVM_FFI_ICHECK(get() != nullptr);
     return static_cast<const PassContextObj*>(get());
   }
   
   inline PassContextObj* PassContext::operator->() {
     TVM_FFI_ICHECK(get() != nullptr);
     return static_cast<PassContextObj*>(get_mutable());
   }
   
   inline PassContext PassContext::Create() {
     return PassContext(ffi::UnsafeInit{}, ffi::make_object<PassContextObj>());
   }
   
   inline void PassContext::InstrumentAfterPass(const IRModule& ir_module,
                                                const PassInfo& pass_info) const {
     auto pass_ctx_node = this->operator->();
     if (pass_ctx_node->instruments.defined()) {
       for (PassInstrument pi : pass_ctx_node->instruments) {
         pi->RunAfterPass(ir_module, pass_info);
       }
     }
   }
   
   }  // namespace transform
   }  // namespace tvm
   
   #endif  // TVM_IR_TRANSFORM_H_
