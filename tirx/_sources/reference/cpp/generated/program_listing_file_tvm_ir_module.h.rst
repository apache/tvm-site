
.. _program_listing_file_tvm_ir_module.h:

Program Listing for File module.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_module.h>` (``tvm/ir/module.h``)

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
   
   #ifndef TVM_IR_MODULE_H_
   #define TVM_IR_MODULE_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/container/tensor.h>
   #include <tvm/ffi/extra/module.h>
   #include <tvm/ffi/extra/structural_equal.h>
   #include <tvm/ffi/extra/structural_hash.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/cow.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/function.h>
   #include <tvm/ir/global_info.h>
   #include <tvm/ir/location.h>
   #include <tvm/ir/type.h>
   
   #include <string>
   #include <unordered_map>
   #include <unordered_set>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   
   class IRModule;
   
   class IRModuleNode : public ffi::Object {
    public:
     ffi::Map<GlobalVar, BaseFunc> functions;
     /* \brief Additional attributes storing meta-data about the module. */
     DictAttrs attrs;
     ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos;
     ffi::Map<ffi::String, GlobalVar> global_var_map_;
   
     template <typename TObjectRef>
     ffi::Optional<TObjectRef> GetAttr(
         const std::string& attr_key,
         ffi::Optional<TObjectRef> default_value = ffi::Optional<TObjectRef>(std::nullopt)) const {
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
   
     DictAttrs GetAttrs() const { return attrs; }
   
     bool HasNonzeroAttr(const std::string& attr_key) const { return attrs.HasNonzeroAttr(attr_key); }
   
     IRModuleNode() = default;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IRModuleNode>()
           .def_ro("functions", &IRModuleNode::functions)
           .def_ro("global_var_map_", &IRModuleNode::global_var_map_)
           .def_ro("attrs", &IRModuleNode::attrs)
           .def_ro("global_infos", &IRModuleNode::global_infos);
       // register custom structural equal and hash.
       refl::TypeAttrDef<IRModuleNode>()
           .def(tvm::ffi::type_attr::kSEqual.name, &IRModuleNode::SEqual)
           .def(tvm::ffi::type_attr::kSHash.name, &IRModuleNode::SHash);
     }
   
     TVM_TIRX_DLL bool SEqual(const IRModuleNode* other,
                              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const;
     TVM_TIRX_DLL int64_t SHash(int64_t init_hash,
                                ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const;
   
     void Add(const GlobalVar& var, const BaseFunc& func, bool update = false) {
       BaseFunc checked_func = func;
       AddUnchecked(var, checked_func);
     }
   
     TVM_TIRX_DLL void AddUnchecked(const GlobalVar& var, const BaseFunc& func);
   
     void Update(const GlobalVar& var, const BaseFunc& func) { this->Add(var, func, true); }
   
     void UpdateGlobalInfo(const ffi::String& name, const ffi::Array<GlobalInfo>& info) {
       this->global_infos.Set(name, info);
     }
   
     void Remove(const GlobalVar& var) {
       auto functions_node = this->functions.CopyOnWrite();
       functions_node->erase(var);
       auto gvar_node = global_var_map_.CopyOnWrite();
       gvar_node->erase(var->name_hint);
     }
   
     bool ContainGlobalVar(const ffi::String& name) const {
       return global_var_map_.find(name) != global_var_map_.end();
     }
   
     TVM_TIRX_DLL GlobalVar GetGlobalVar(const ffi::String& str) const;
   
     TVM_TIRX_DLL ffi::Array<GlobalVar> GetGlobalVars() const;
   
     BaseFunc Lookup(const GlobalVar& var) const {
       auto it = functions.find(var);
       TVM_FFI_ICHECK(it != functions.end()) << "There is no definition of " << var;
       return (*it).second;
     }
   
     BaseFunc Lookup(const ffi::String& name) const {
       GlobalVar id = this->GetGlobalVar(name);
       return this->Lookup(id);
     }
   
     void Update(const IRModule& other);
   
     IRModule ShallowCopy();
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IRModule", IRModuleNode, ffi::Object);
   
    private:
     friend class IRModule;
   };
   
   class IRModule : public ffi::ObjectRef {
    public:
     static constexpr bool _type_is_nullable = false;
     TVM_TIRX_DLL explicit IRModule(ffi::Map<GlobalVar, BaseFunc> functions,
                                    DictAttrs attrs = DictAttrs(),
                                    ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos = {});
   
     IRModule() : IRModule(ffi::Map<GlobalVar, BaseFunc>({})) {}
     explicit IRModule(ffi::ObjectPtr<IRModuleNode> n) : ffi::ObjectRef(std::move(n)) {
       TVM_FFI_ICHECK(data_ != nullptr);
     }
     explicit IRModule(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
     IRModuleNode* operator->() const {
       auto* ptr = get_mutable();
       TVM_FFI_ICHECK(ptr != nullptr);
       return static_cast<IRModuleNode*>(ptr);
     }
   
     TVM_TIRX_DLL static IRModule FromExpr(const Expr& expr,
                                           const ffi::Map<GlobalVar, BaseFunc>& global_funcs = {});
   
     using ContainerType = IRModuleNode;
   
     // allow copy on write.
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IRModuleNode);
   };
   
   inline void IRModuleNode::Update(const IRModule& mod) {
     for (auto pair : mod->functions) {
       this->AddUnchecked(pair.first, pair.second);
     }
   }
   
   inline IRModule IRModuleNode::ShallowCopy() {
     return IRModule(this->functions, this->attrs, this->global_infos);
   }
   
   namespace attr {
   
   // Following are attributes for IRModule only.
   
   inline constexpr ffi::reflection::AttrKey<ffi::Array<ffi::Module>> kExternalMods{"external_mods"};
   
   inline constexpr ffi::reflection::AttrKey<ffi::String> kSystemLibPrefix{"system_lib_prefix"};
   
   }  // namespace attr
   }  // namespace tvm
   #endif  // TVM_IR_MODULE_H_
