
.. _program_listing_file_tvm_ir_op.h:

Program Listing for File op.h
=============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_op.h>` (``tvm/ir/op.h``)

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
   
   #ifndef TVM_IR_OP_H_
   #define TVM_IR_OP_H_
   
   #include <tvm/ffi/container/list.h>
   #include <tvm/ffi/expected.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/reflection/native_function.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/attrs.h>
   #include <tvm/ir/env_func.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/type.h>
   
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   
   template <typename>
   class OpAttrMap;
   
   class ArgumentInfoObj : public ffi::Object {
    public:
     ffi::String name;
     ffi::String doc;
   
     static void RegisterReflection() {
       namespace refl = ffi::reflection;
       refl::ObjectDef<ArgumentInfoObj>()
           .def_ro("name", &ArgumentInfoObj::name)
           .def_ro("doc", &ArgumentInfoObj::doc);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.ArgumentInfo", ArgumentInfoObj, ffi::Object);
   };
   
   class ArgumentInfo : public ffi::ObjectRef {
    public:
     explicit ArgumentInfo(ffi::UnsafeInit tag, ffi::ObjectPtr<ArgumentInfoObj> n)
         : ffi::ObjectRef(tag) {
       data_ = std::move(n);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ArgumentInfo, ffi::ObjectRef, ArgumentInfoObj);
   };
   
   class OpObj : public ExprObj {
    private:
     // Dense process-local index into attribute columns; never serialized.
     uint32_t index_{0};
   
    public:
     ffi::String name;
     ffi::String doc;
     ffi::Array<ArgumentInfo> args_info;
     ffi::Array<ArgumentInfo> ty_args_info;
     ffi::Optional<ArgumentInfo> var_args_info;
     ffi::Optional<ArgumentInfo> var_ty_args_info;
     ffi::String attrs_type_key;
     uint32_t attrs_type_index{0};
     ffi::Any validator;
     bool validator_is_custom{false};
   
     TVM_TIRX_DLL static void RegisterReflection();
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Op", OpObj, ExprObj);
   
    private:
     friend class OpRegistry;
     friend class OpDef;
     friend class Op;
     template <typename>
     friend class OpAttrMap;
   };
   
   class Op : public Expr {
    public:
     explicit Op(ffi::ObjectPtr<OpObj> n) : Expr(std::move(n)) {
       TVM_FFI_CHECK(defined(), ValueError) << "Op expects a defined OpObj";
     }
   
     template <typename ValueType>
     static OpAttrMap<ValueType> GetAttrMap(const ffi::String& attr_name);
     template <typename ValueType>
     static OpAttrMap<ValueType> GetAttrMap(ffi::reflection::AttrKey<ValueType> key);
   
     TVM_TIRX_DLL static bool HasAttrMap(const ffi::String& attr_name);
     template <typename ValueType>
     static bool HasAttrMap(ffi::reflection::AttrKey<ValueType> key) {
       return HasAttrMap(key.name);
     }
   
     TVM_TIRX_DLL static Op Get(const ffi::String& op_name);
     TVM_TIRX_DLL static ffi::Array<ffi::String> ListNames();
   
     TVM_FFI_INLINE void Validate(const CallObj* call) const {
       const auto& validator = get()->validator;
       if (validator != nullptr) {
         if (TVM_FFI_PREDICT_FALSE(!call)) {
           ThrowInvalidCall(get());
         }
         using View = ffi::reflection::NativeFunctionView<void(const CallObj*)>;
         // set_validator stores only an owning NativeFunction with this signature.
         ffi::details::AnyUnsafe::CopyFromAnyViewAfterCheck<View>(validator)
             .CallExpected(call)
             .value();
       }
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Op, Expr, OpObj);
   
    private:
     TVM_TIRX_DLL static void ThrowInvalidCall(const OpObj* op);
     TVM_TIRX_DLL static ffi::List<ffi::Any> GetAttrColumn(const ffi::String& attr_name);
   };
   
   namespace op_attr {
   inline constexpr ffi::reflection::AttrKey<
       ffi::reflection::NativeFunctionView<Type(const CallObj* call)>>
       kInferType{"FInferType"};
   inline constexpr ffi::reflection::AttrKey<Type> kFixedReturnType{"TFixedReturnType"};
   }  // namespace op_attr
   
   // Compatibility names; new callers can deduce the value type from the key.
   using FInferType = decltype(op_attr::kInferType)::ValueType;
   using TFixedReturnType = decltype(op_attr::kFixedReturnType)::ValueType;
   
   namespace sig {
   
   struct SignatureTrait {
     static constexpr size_t kArgsCount = 0;
     static constexpr size_t kTyArgsCount = 0;
     static constexpr size_t kVarArgsCount = 0;
     static constexpr size_t kVarTyArgsCount = 0;
     static constexpr size_t kCallAttrsCount = 0;
   };
   
   struct ValidateState {
     const CallObj* call;
     const OpObj* op;
     size_t args_index = 0;
     size_t ty_args_index = 0;
   };
   
   namespace details {
   
   template <typename T>
   inline bool ReportTypeMismatch(ffi::Expected<void>* out, const ValidateState* state, bool type_arg,
                                  size_t index, ffi::AnyView actual) {
     const auto& fixed = type_arg ? state->op->ty_args_info : state->op->args_info;
     const auto& tail = type_arg ? state->op->var_ty_args_info : state->op->var_args_info;
     std::string name;
     if (index < fixed.size()) {
       name = fixed[index]->name;
     } else if (tail.has_value()) {
       name = tail.value()->name;
     }
     std::string name_clause = name.empty() ? "" : " (`" + name + "`)";
     TVMFFIAny raw = actual.CopyToTVMFFIAny();
     *out = TVM_FFI_UNEXPECTED(TypeError)
            << "Op `" << state->op->name << "`: `" << (type_arg ? "Call.ty_args[" : "Call.args[")
            << index << "]`" << name_clause << " expected `" << ffi::TypeTraits<T>::TypeStr()
            << "`, got `" << ffi::TypeTraits<T>::GetMismatchTypeInfo(&raw) << "`.";
     return false;
   }
   
   }  // namespace details
   
   template <typename T = Expr>
   struct arg : SignatureTrait {
     static_assert(std::is_base_of_v<Expr, T>, "arg<T> expects an expression view");
     static constexpr size_t kArgsCount = 1;
     ffi::String name;
     ffi::String doc;
   
     explicit arg(ffi::String name, ffi::String doc = "")
         : name(std::move(name)), doc(std::move(doc)) {}
   
     void CollectMetadata(OpObj* op) const {
       auto info = ffi::make_object<ArgumentInfoObj>();
       info->name = name;
       info->doc = doc;
       op->args_info.push_back(ArgumentInfo(ffi::UnsafeInit{}, std::move(info)));
     }
   
     static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
       if constexpr (!std::is_same_v<T, Expr>) {
         ffi::AnyView value(state->call->args.GetArrayObj()->begin()[state->args_index]);
         if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                   !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
           return details::ReportTypeMismatch<T>(out, state, false, state->args_index, value);
         }
       }
       ++state->args_index;
       return true;
     }
   };
   
   template <typename T = Expr>
   struct var_args : SignatureTrait {
     static_assert(std::is_base_of_v<Expr, T>, "var_args<T> expects an expression view");
     static constexpr size_t kVarArgsCount = 1;
     ffi::String name;
     ffi::String doc;
   
     explicit var_args(ffi::String name, ffi::String doc = "")
         : name(std::move(name)), doc(std::move(doc)) {}
   
     void CollectMetadata(OpObj* op) const {
       auto info = ffi::make_object<ArgumentInfoObj>();
       info->name = name;
       info->doc = doc;
       op->var_args_info = ArgumentInfo(ffi::UnsafeInit{}, std::move(info));
     }
   
     static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
       if constexpr (!std::is_same_v<T, Expr>) {
         const ffi::Any* values = state->call->args.GetArrayObj()->begin();
         for (; state->args_index != state->call->args.size(); ++state->args_index) {
           ffi::AnyView value(values[state->args_index]);
           if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                     !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
             return details::ReportTypeMismatch<T>(out, state, false, state->args_index, value);
           }
         }
       } else {
         state->args_index = state->call->args.size();
       }
       return true;
     }
   };
   
   template <typename T = Type>
   struct ty_arg : SignatureTrait {
     static_assert(std::is_base_of_v<Type, T>, "ty_arg<T> expects a type view");
     static constexpr size_t kTyArgsCount = 1;
     ffi::String name;
     ffi::String doc;
   
     explicit ty_arg(ffi::String name, ffi::String doc = "")
         : name(std::move(name)), doc(std::move(doc)) {}
   
     void CollectMetadata(OpObj* op) const {
       auto info = ffi::make_object<ArgumentInfoObj>();
       info->name = name;
       info->doc = doc;
       op->ty_args_info.push_back(ArgumentInfo(ffi::UnsafeInit{}, std::move(info)));
     }
   
     static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
       if constexpr (!std::is_same_v<T, Type>) {
         ffi::AnyView value(state->call->ty_args.GetArrayObj()->begin()[state->ty_args_index]);
         if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                   !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
           return details::ReportTypeMismatch<T>(out, state, true, state->ty_args_index, value);
         }
       }
       ++state->ty_args_index;
       return true;
     }
   };
   
   template <typename T = Type>
   struct var_ty_args : SignatureTrait {
     static_assert(std::is_base_of_v<Type, T>, "var_ty_args<T> expects a type view");
     static constexpr size_t kVarTyArgsCount = 1;
     ffi::String name;
     ffi::String doc;
   
     explicit var_ty_args(ffi::String name, ffi::String doc = "")
         : name(std::move(name)), doc(std::move(doc)) {}
   
     void CollectMetadata(OpObj* op) const {
       auto info = ffi::make_object<ArgumentInfoObj>();
       info->name = name;
       info->doc = doc;
       op->var_ty_args_info = ArgumentInfo(ffi::UnsafeInit{}, std::move(info));
     }
   
     static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
       if constexpr (!std::is_same_v<T, Type>) {
         const ffi::Any* values = state->call->ty_args.GetArrayObj()->begin();
         for (; state->ty_args_index != state->call->ty_args.size(); ++state->ty_args_index) {
           ffi::AnyView value(values[state->ty_args_index]);
           if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                     !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
             return details::ReportTypeMismatch<T>(out, state, true, state->ty_args_index, value);
           }
         }
       } else {
         state->ty_args_index = state->call->ty_args.size();
       }
       return true;
     }
   };
   
   template <typename T>
   struct call_attrs : SignatureTrait {
     static_assert(std::is_base_of_v<AttrsObj, T>, "call_attrs<T> expects an attribute node type");
     static constexpr size_t kCallAttrsCount = 1;
   
     void CollectMetadata(OpObj* op) const {
       uint32_t index = T::RuntimeTypeIndex();
       op->attrs_type_key = T::_type_key;
       op->attrs_type_index = index;
     }
   
     static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
       ffi::AnyView attrs(state->call->attrs);
       if (TVM_FFI_PREDICT_FALSE(!state->call->attrs.has_value() ||
                                 !ffi::details::AnyUnsafe::CheckAnyViewStrict<const T*>(attrs))) {
         TVMFFIAny raw = attrs.CopyToTVMFFIAny();
         std::string actual = state->call->attrs.has_value()
                                  ? "`" + ffi::TypeTraits<const T*>::GetMismatchTypeInfo(&raw) + "`"
                                  : "None";
         *out = TVM_FFI_UNEXPECTED(TypeError)
                << "Op `" << state->op->name << "`: Call.attrs expected `"
                << ffi::TypeTraits<const T*>::TypeStr() << "`, got " << actual;
         return false;
       }
       return true;
     }
   };
   
   }  // namespace sig
   
   class OpDef {
    public:
     TVM_TIRX_DLL explicit OpDef(const ffi::String& name);
     TVM_TIRX_DLL OpDef(const ffi::String& name, const ffi::String& doc);
     OpDef(const OpDef&) = delete;
     OpDef& operator=(const OpDef&) = delete;
     OpDef(OpDef&&) = delete;
     OpDef& operator=(OpDef&&) = delete;
     Op op() const { return op_; }
     TVM_TIRX_DLL OpDef& add_arg(const ffi::String& name, const ffi::String& doc);
     template <typename AttrsType>
     OpDef& attrs_type() {
       uint32_t index = AttrsType::RuntimeTypeIndex();
       get()->attrs_type_key = AttrsType::_type_key;
       get()->attrs_type_index = index;
       return *this;
     }
     TVM_TIRX_DLL OpDef& set_attrs_type_key(const ffi::String& key);
     TVM_TIRX_DLL OpDef& add_ty_arg(const ffi::String& name, const ffi::String& doc);
     template <typename... Specs>
     OpDef& signature(const Specs&... specs) {
       auto updated = ffi::make_object<OpObj>();
       (ApplySignatureTrait(updated.get(), specs), ...);
       if (op_->validator == nullptr) {
         using View = ffi::reflection::NativeFunctionView<void(const CallObj*)>;
         set_validator(View::FromNative<&ValidateSignature<Specs...>>());
         get()->validator_is_custom = false;
       }
       OpObj* target = get();
       target->args_info = std::move(updated->args_info);
       target->ty_args_info = std::move(updated->ty_args_info);
       target->var_args_info = std::move(updated->var_args_info);
       target->var_ty_args_info = std::move(updated->var_ty_args_info);
       target->attrs_type_key = std::move(updated->attrs_type_key);
       target->attrs_type_index = updated->attrs_type_index;
       return *this;
     }
     template <typename ValueType>
     OpDef& set_attr(const ffi::String& attr_name, const ValueType& value, bool override = false) {
       UpdateAttr(attr_name, ffi::Any(value), override);
       return *this;
     }
     template <typename ValueType, typename Value,
               std::enable_if_t<(std::is_convertible_v<Value&&, ValueType> &&
                                 ((!std::is_same_v<std::decay_t<Value>, ffi::Any> &&
                                   !std::is_same_v<std::decay_t<Value>, ffi::AnyView>) ||
                                  std::is_same_v<ValueType, ffi::Any> ||
                                  std::is_same_v<ValueType, ffi::AnyView>)),
                                int> = 0>
     OpDef& set_attr(ffi::reflection::AttrKey<ValueType> key, Value&& value, bool override = false) {
       ValueType typed_value(std::forward<Value>(value));
       return set_attr(key.name, typed_value, override);
     }
     template <auto Fn, typename ValueType>
     OpDef& set_native_fn_attr(ffi::reflection::AttrKey<ValueType> key, bool override = false) {
       return set_attr(key, ValueType::template FromNative<Fn>(), override);
     }
     OpDef& set_validator(ffi::reflection::NativeFunctionView<void(const CallObj*)> validator,
                          bool override = false) {
       TVM_FFI_CHECK(override || op_->validator == nullptr, ValueError)
           << "Validator of " << op_->name << " is already registered";
       get()->validator = ffi::reflection::NativeFunction<void(const CallObj*)>::From(validator);
       get()->validator_is_custom = true;
       return *this;
     }
     TVM_TIRX_DLL OpDef& reset_attr(const ffi::String& attr_name);
     template <typename ValueType>
     OpDef& reset_attr(ffi::reflection::AttrKey<ValueType> key) {
       return reset_attr(key.name);
     }
   
    private:
     template <typename Spec>
     static void ApplySignatureTrait(OpObj* op, const Spec& spec) {
       static_assert(std::is_base_of_v<sig::SignatureTrait, Spec>, "Unknown signature descriptor");
       spec.CollectMetadata(op);
     }
   
     template <typename... Specs>
     TVM_FFI_INLINE static ffi::Expected<void> ValidateSignature(const CallObj* call) noexcept {
       static_assert((std::is_base_of_v<sig::SignatureTrait, Specs> && ...),
                     "Unknown signature descriptor");
       constexpr size_t n_args = (size_t{0} + ... + Specs::kArgsCount);
       constexpr size_t n_ty_args = (size_t{0} + ... + Specs::kTyArgsCount);
       constexpr size_t n_var_args = (size_t{0} + ... + Specs::kVarArgsCount);
       constexpr size_t n_var_ty_args = (size_t{0} + ... + Specs::kVarTyArgsCount);
       constexpr size_t n_attrs = (size_t{0} + ... + Specs::kCallAttrsCount);
       static_assert(n_var_args <= 1 && n_var_ty_args <= 1 && n_attrs <= 1,
                     "Duplicate signature tail or call_attrs");
       const OpObj* op = call ? call->op.as<OpObj>() : nullptr;
       if (TVM_FFI_PREDICT_FALSE(!op)) {
         return TVM_FFI_UNEXPECTED(TypeError) << "Invalid Op Call";
       }
       if (TVM_FFI_PREDICT_FALSE(!call->args.defined() || !call->ty_args.defined())) {
         return TVM_FFI_UNEXPECTED(TypeError) << "Op `" << op->name << "`: invalid Call";
       }
       ffi::Expected<void> result;
       auto* out = &result;
       if (TVM_FFI_PREDICT_FALSE(n_var_args ? call->args.size() < n_args
                                            : call->args.size() != n_args)) {
         *out = TVM_FFI_UNEXPECTED(TypeError)
                << "Op `" << op->name << "`: Call.args expected " << (n_var_args ? "at least " : "")
                << n_args << (n_args == 1 ? " argument" : " arguments") << ", got "
                << call->args.size();
         return result;
       }
       if (TVM_FFI_PREDICT_FALSE(n_var_ty_args ? call->ty_args.size() < n_ty_args
                                               : call->ty_args.size() != n_ty_args)) {
         *out = TVM_FFI_UNEXPECTED(TypeError)
                << "Op `" << op->name << "`: Call.ty_args expected "
                << (n_var_ty_args ? "at least " : "") << n_ty_args
                << (n_ty_args == 1 ? " type argument" : " type arguments") << ", got "
                << call->ty_args.size();
         return result;
       }
       if constexpr (sizeof...(Specs) != 0) {
         sig::ValidateState state{call, op};
         if (TVM_FFI_PREDICT_FALSE(!(Specs::Validate(&state, out) && ...))) {
           return result;
         }
       }
       return result;
     }
   
     OpObj* get() { return const_cast<OpObj*>(op_.operator->()); }
     TVM_TIRX_DLL void UpdateAttr(const ffi::String& attr_name, ffi::Any value, bool override);
     Op op_;
   };
   
   template <typename ValueType>
   class OpAttrMap {
    public:
     int count(const Op& op) const {
       return op->index_ < column_.size() && column_[op->index_] != nullptr;
     }
     ValueType operator[](const Op& op) const {
       TVM_FFI_ICHECK(count(op)) << "Attribute " << attr_name_ << " has not been registered for "
                                 << op->name;
       if constexpr (std::is_same_v<ValueType, ffi::Any>) {
         return column_[op->index_];
       } else {
         return column_[op->index_].template cast<ValueType>();
       }
     }
     ValueType get(const Op& op, ValueType def_value) const {
       return count(op) ? (*this)[op] : def_value;
     }
     ValueType get(const Expr& expr, ValueType def_value) const {
       TVM_FFI_ICHECK(expr.defined());
       if (const auto* op = expr.as<OpObj>()) {
         return get(ffi::GetRef<Op>(op), def_value);
       }
       return def_value;
     }
   
    private:
     friend class Op;
     OpAttrMap(ffi::List<ffi::Any> column, ffi::String attr_name)
         : column_(std::move(column)), attr_name_(std::move(attr_name)) {}
     ffi::List<ffi::Any> column_;
     ffi::String attr_name_;
   };
   
   template <typename ValueType>
   OpAttrMap<ValueType> Op::GetAttrMap(const ffi::String& attr_name) {
     return OpAttrMap<ValueType>(GetAttrColumn(attr_name), attr_name);
   }
   
   template <typename ValueType>
   OpAttrMap<ValueType> Op::GetAttrMap(ffi::reflection::AttrKey<ValueType> key) {
     return GetAttrMap<ValueType>(key.name);
   }
   
   }  // namespace tvm
   #endif  // TVM_IR_OP_H_
