
.. _program_listing_file_tvm_ir_attrs.h:

Program Listing for File attrs.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_attrs.h>` (``tvm/ir/attrs.h``)

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
   #ifndef TVM_IR_ATTRS_H_
   #define TVM_IR_ATTRS_H_
   
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/extra/structural_equal.h>
   #include <tvm/ffi/extra/structural_hash.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/reflection/accessor.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/cow.h>
   
   #include <string>
   #include <type_traits>
   #include <unordered_map>
   #include <utility>
   
   namespace tvm {
   
   class AttrsObj : public ffi::Object {
    public:
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Attrs", AttrsObj, ffi::Object);
   };
   
   class Attrs : public ffi::ObjectRef {
    public:
     explicit Attrs(ffi::ObjectPtr<AttrsObj> n) : ffi::ObjectRef(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(n != nullptr);
       data_ = std::move(n);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Attrs, ffi::ObjectRef, AttrsObj);
   };
   
   class DictAttrsObj : public AttrsObj {
    public:
     ffi::Map<ffi::String, ffi::Any> dict;
   
     static void RegisterReflection() {
       namespace rfl = ffi::reflection;
       rfl::ObjectDef<DictAttrsObj>().def_ro("dict", &DictAttrsObj::dict);
     }
   
     // type info
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.DictAttrs", DictAttrsObj, AttrsObj);
   };
   
   class DictAttrs : public Attrs {
    public:
     explicit DictAttrs(ffi::Map<ffi::String, Any> dict = {}) : Attrs(ffi::UnsafeInit{}) {
       ffi::ObjectPtr<DictAttrsObj> n = ffi::make_object<DictAttrsObj>();
       n->dict = std::move(dict);
       data_ = std::move(n);
     }
   
     DictAttrs(DictAttrs&& other) noexcept : Attrs(ffi::UnsafeInit{}) {
       data_ = std::move(other.data_);
       other.data_ = ffi::make_object<DictAttrsObj>();
     }
   
     DictAttrs& operator=(DictAttrs&& other) noexcept {
       if (this != &other) {
         data_ = std::move(other.data_);
         other.data_ = ffi::make_object<DictAttrsObj>();
       }
       return *this;
     }
   
     // Explicit copy ctor/assign defaults. Declaring the move members above
     // would otherwise suppress the implicit copy members.
     DictAttrs(const DictAttrs& other) = default;
     DictAttrs& operator=(const DictAttrs& other) = default;
   
     // Utils for accessing attributes
     template <typename TObjectRef>
     ffi::Optional<TObjectRef> GetAttr(
         const std::string& attr_key,
         ffi::Optional<TObjectRef> default_value = ffi::Optional<TObjectRef>(std::nullopt)) const {
       const DictAttrsObj* node = get();
       auto it = node->dict.find(attr_key);
       if (it != node->dict.end()) {
         return (*it).second.cast<TObjectRef>();
       } else {
         return default_value;
       }
     }
     template <typename T>
     ffi::Optional<T> GetAttr(ffi::reflection::AttrKey<T> key,
                              ffi::Optional<typename ffi::reflection::AttrKey<T>::ValueType>
                                  default_value = std::nullopt) const {
       return GetAttr<T>(key.name, std::move(default_value));
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
     bool HasNonzeroAttr(const std::string& attr_key) const {
       return GetAttr<int64_t>(attr_key, 0).value_or(0) != 0;
     }
   
     // Inline-expand TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE here, minus
     // the default copy/move it normally injects (we define our own move members
     // above so the moved-from instance stays defined-but-empty).
     explicit DictAttrs(::tvm::ffi::UnsafeInit tag) : Attrs(tag) {}
     using __PtrType =
         std::conditional_t<DictAttrsObj::_type_mutable, DictAttrsObj*, const DictAttrsObj*>;
     __PtrType operator->() const { return static_cast<__PtrType>(data_.get()); }
     __PtrType get() const { return static_cast<__PtrType>(data_.get()); }
     static constexpr bool _type_is_nullable = false;
     using ContainerType = DictAttrsObj;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(DictAttrsObj);
   };
   
   template <typename TFunc>
   inline TFunc WithAttr(TFunc input, const std::string& attr_key, Any attr_value) {
     using TNode = typename TFunc::ContainerType;
     if (!input.unique()) {
       static ffi::reflection::TypeAttrColumn shallow_copy(ffi::type_attr::kShallowCopy);
       ffi::AnyView copy_func = shallow_copy[input->type_index()];
       TVM_FFI_CHECK(copy_func.type_index() == ffi::TypeIndex::kTVMFFIFunction, TypeError)
           << "Type " << input->GetTypeKey()
           << " must register an ffi.Function for __ffi_shallow_copy__";
       Any copy = copy_func.cast<ffi::Function>()(input);
       TVM_FFI_CHECK(copy.type_index() == input->type_index() && copy.as<ffi::Object>() != input.get(),
                     TypeError)
           << "Shallow copy must return a distinct object of type " << input->GetTypeKey();
       input = std::move(copy).cast<TFunc>();
     }
     TNode* node = const_cast<TNode*>(input.operator->());
     // node->attrs is NOTNULLABLE by contract, but defend against a caller
     // that left a moved-from DictAttrs in place by re-initializing here.
     if (!node->attrs.defined()) node->attrs = DictAttrs();
     node->attrs.CopyOnWrite()->dict.Set(attr_key, std::move(attr_value));
     return input;
   }
   
   template <typename TFunc, typename T, typename Value,
             std::enable_if_t<(std::is_convertible_v<Value&&, T> &&
                               ((!std::is_same_v<std::decay_t<Value>, ffi::Any> &&
                                 !std::is_same_v<std::decay_t<Value>, ffi::AnyView>) ||
                                std::is_same_v<T, ffi::Any> || std::is_same_v<T, ffi::AnyView>)),
                              int> = 0>
   inline TFunc WithAttr(TFunc input, ffi::reflection::AttrKey<T> key, Value&& value) {
     T typed_value(std::forward<Value>(value));
     return WithAttr(std::move(input), key.name, ffi::Any(typed_value));
   }
   
   template <typename TFunc>
   inline TFunc WithAttrs(TFunc input, ffi::Map<ffi::String, Any> attrs) {
     using TNode = typename TFunc::ContainerType;
     if (attrs.empty()) return input;
     if (!input.unique()) {
       static ffi::reflection::TypeAttrColumn shallow_copy(ffi::type_attr::kShallowCopy);
       ffi::AnyView copy_func = shallow_copy[input->type_index()];
       TVM_FFI_CHECK(copy_func.type_index() == ffi::TypeIndex::kTVMFFIFunction, TypeError)
           << "Type " << input->GetTypeKey()
           << " must register an ffi.Function for __ffi_shallow_copy__";
       Any copy = copy_func.cast<ffi::Function>()(input);
       TVM_FFI_CHECK(copy.type_index() == input->type_index() && copy.as<ffi::Object>() != input.get(),
                     TypeError)
           << "Shallow copy must return a distinct object of type " << input->GetTypeKey();
       input = std::move(copy).cast<TFunc>();
     }
     TNode* node = const_cast<TNode*>(input.operator->());
     // node->attrs is NOTNULLABLE by contract, but defend against a caller
     // that left a moved-from DictAttrs in place by re-initializing here.
     if (!node->attrs.defined()) node->attrs = DictAttrs();
     auto* dict_node = node->attrs.CopyOnWrite();
     for (const auto& [k, v] : attrs) {
       dict_node->dict.Set(k, v);
     }
     return input;
   }
   
   template <typename TFunc>
   inline TFunc WithoutAttr(TFunc input, const std::string& attr_key) {
     using TNode = typename TFunc::ContainerType;
     if (!input.unique()) {
       static ffi::reflection::TypeAttrColumn shallow_copy(ffi::type_attr::kShallowCopy);
       ffi::AnyView copy_func = shallow_copy[input->type_index()];
       TVM_FFI_CHECK(copy_func.type_index() == ffi::TypeIndex::kTVMFFIFunction, TypeError)
           << "Type " << input->GetTypeKey()
           << " must register an ffi.Function for __ffi_shallow_copy__";
       Any copy = copy_func.cast<ffi::Function>()(input);
       TVM_FFI_CHECK(copy.type_index() == input->type_index() && copy.as<ffi::Object>() != input.get(),
                     TypeError)
           << "Shallow copy must return a distinct object of type " << input->GetTypeKey();
       input = std::move(copy).cast<TFunc>();
     }
     TNode* node = const_cast<TNode*>(input.operator->());
     // node->attrs is NOTNULLABLE by contract, but defend against a caller
     // that left a moved-from DictAttrs in place; nothing to erase from an
     // empty dict.
     if (!node->attrs.defined()) {
       node->attrs = DictAttrs();
       return input;
     }
     node->attrs.CopyOnWrite()->dict.erase(attr_key);
     return input;
   }
   
   template <typename TFunc, typename T>
   inline TFunc WithoutAttr(TFunc input, ffi::reflection::AttrKey<T> key) {
     return WithoutAttr(std::move(input), key.name);
   }
   
   }  // namespace tvm
   #endif  // TVM_IR_ATTRS_H_
