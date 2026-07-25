
.. _program_listing_file_tvm_ffi_extra_structural_visit.h:

Program Listing for File structural_visit.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_structural_visit.h>` (``tvm/ffi/extra/structural_visit.h``)

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
   #ifndef TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
   #define TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ffi/container/variant.h>
   #include <tvm/ffi/expected.h>
   #include <tvm/ffi/extra/visit_error_context.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/function_details.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/reflection/accessor.h>
   
   #include <cstddef>
   #include <exception>
   #include <optional>
   #include <string>
   #include <tuple>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   class VisitInterruptObj : public Object {
    public:
     Any value;
   
     VisitInterruptObj() = default;
     explicit VisitInterruptObj(Any value) : value(std::move(value)) {}
   
     static constexpr const int32_t _type_index = TypeIndex::kTVMFFIVisitInterrupt;
     static const constexpr bool _type_final = true;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIVisitInterrupt, VisitInterruptObj,
                                        Object);
   };
   
   class VisitInterrupt : public ObjectRef {
    public:
     VisitInterrupt() : VisitInterrupt(Any(nullptr)) {}
     explicit VisitInterrupt(Any value)
         : ObjectRef(make_object<VisitInterruptObj>(std::move(value))) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisitInterrupt, ObjectRef, VisitInterruptObj);
   };
   
   class StructuralVisitorObj;
   
   using FStructuralVisit = TVMFFIAny (*)(StructuralVisitorObj* visitor, AnyView value) noexcept;
   
   namespace details {
   
   // Visit reflected structural fields of an object-backed value.
   TVM_FFI_INLINE static Expected<Optional<VisitInterrupt>> VisitReflectedFieldsExpected(
       StructuralVisitorObj* visitor, const Object* obj) noexcept;
   
   }  // namespace details
   
   struct StructuralVisitorVTable {
     FStructuralVisit visit = nullptr;
   };
   
   class StructuralVisitorObj : public Object {
    public:
     TVM_FFI_INLINE Optional<VisitInterrupt> Visit(AnyView value) {
       return VisitExpected(value).value();
     }
   
     TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> VisitExpected(AnyView value) noexcept {
       return details::ExpectedUnsafe::MoveFromTVMFFIAny<Optional<VisitInterrupt>>(
           (*vtable_->visit)(this, value));
     }
   
     TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }
   
     template <typename Callback>
     TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback) {
       class Scope {
        public:
         Scope(StructuralVisitorObj* visitor, TVMFFIDefRegionKind kind)
             : visitor_(visitor), old_kind_(visitor->def_region_mode_) {
           visitor_->def_region_mode_ = kind;
         }
         ~Scope() { visitor_->def_region_mode_ = old_kind_; }
         Scope(const Scope&) = delete;
         Scope& operator=(const Scope&) = delete;
   
        private:
         StructuralVisitorObj* visitor_;
         TVMFFIDefRegionKind old_kind_;
       };
       Scope scope(this, kind);
       return std::forward<Callback>(callback)();
     }
   
     TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> DefaultVisitExpected(AnyView value) noexcept {
       int32_t type_index = value.type_index();
       static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
       AnyView attr = column[type_index];
   
       // case 1: Type-specific override registered as an opaque ABI visit function pointer.
       if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
         auto* visit_fn = reinterpret_cast<FStructuralVisit>(attr.cast<void*>());
         return details::ExpectedUnsafe::MoveFromTVMFFIAny<Optional<VisitInterrupt>>(
             (*visit_fn)(this, value));
       }
   
       // case 2: Type-specific override registered as an ffi::Function.
       if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
         return attr.cast<Function>().CallExpected<Optional<VisitInterrupt>>(this, value);
       }
   
       if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFINone)) {
         return Unexpected(Error("TypeError",
                                 std::string(reflection::type_attr::kStructuralVisit) +
                                     " must be an opaque function pointer or ffi.Function",
                                 ""));
       }
   
       if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
         return Optional<VisitInterrupt>(std::nullopt);
       }
   
       return details::VisitReflectedFieldsExpected(this, value.cast<const Object*>());
     }
   
     static constexpr const bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralVisitor", StructuralVisitorObj, Object);
   
    protected:
     explicit StructuralVisitorObj(const StructuralVisitorVTable* vtable) : vtable_(vtable) {}
   
     const StructuralVisitorVTable* vtable_ = nullptr;
   
     TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
   };
   
   class StructuralVisitor : public ObjectRef {
    public:
     explicit StructuralVisitor(ObjectPtr<StructuralVisitorObj> n) : ObjectRef(std::move(n)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralVisitor, ObjectRef, StructuralVisitorObj);
   };
   
   namespace details {
   
   template <typename T>
   TVM_FFI_INLINE bool StructuralVisitNeedEarlyReturn(const Expected<T>& result) noexcept {
     int32_t type_index = result.type_index();
     return type_index == TypeIndex::kTVMFFIError || type_index == TypeIndex::kTVMFFIVisitInterrupt;
   }
   
   TVM_FFI_INLINE static Expected<Optional<VisitInterrupt>> VisitReflectedFieldsExpected(
       StructuralVisitorObj* visitor, const Object* obj) noexcept {
     int32_t type_index = obj->type_index();
     const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
     // A non-recursive definition applies to a FreeVar itself, but not to its children. All other
     // inherited modes propagate until an explicit field annotation overrides them.
     TVMFFIDefRegionKind inherited_kind = visitor->def_region_kind();
     if (inherited_kind == kTVMFFIDefRegionKindNonRecursive && type_info->metadata != nullptr &&
         type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
       inherited_kind = kTVMFFIDefRegionKindNone;
     }
   
     Expected<Optional<VisitInterrupt>> result = Optional<VisitInterrupt>(std::nullopt);
     reflection::ForEachFieldInfoWithEarlyStop(
         type_info, [&](const TVMFFIFieldInfo* field_info) -> bool {
           if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
             return false;
           }
   
           Any field_value;
           const void* field_addr = reinterpret_cast<const char*>(obj) + field_info->offset;
           int ret_code = field_info->getter(const_cast<void*>(field_addr),
                                             reinterpret_cast<TVMFFIAny*>(&field_value));
           if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
             result = Unexpected(details::MoveFromSafeCallRaised());
             return true;
           }
   
           TVMFFIDefRegionKind kind = inherited_kind;
           if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
             kind = kTVMFFIDefRegionKindNonRecursive;
           } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
             kind = kTVMFFIDefRegionKindRecursive;
           }
   
           result =
               visitor->WithDefRegionKind(kind, [&]() { return visitor->VisitExpected(field_value); });
           return StructuralVisitNeedEarlyReturn(result);
         });
     return result;
   }
   
   }  // namespace details
   
   // ---------------------------------------------------------------------------
   // Structural Walk API.
   // ---------------------------------------------------------------------------
   
   class WalkResult : public Variant<VisitInterrupt, int32_t> {
    public:
     static constexpr int32_t kAdvanceTag = 0;
     static constexpr int32_t kSkipTag = 1;
   
     using Storage = Variant<VisitInterrupt, int32_t>;
   
     static WalkResult Advance() { return WalkResult(kAdvanceTag); }
   
     static WalkResult Skip() { return WalkResult(kSkipTag); }
   
     static WalkResult Interrupt(VisitInterrupt signal = VisitInterrupt()) {
       return WalkResult(Storage(std::move(signal)));
     }
   
    private:
     // Keep raw storage construction behind the named factories.
     explicit WalkResult(int32_t tag) : Storage(tag) {}
     explicit WalkResult(Storage storage) : Storage(std::move(storage)) {}
   
     friend struct TypeTraits<WalkResult>;
   };
   
   template <>
   inline constexpr bool use_default_type_traits_v<WalkResult> = false;
   
   // Allow WalkResult to round-trip through Any / Expected while reusing Variant storage.
   template <>
   struct TypeTraits<WalkResult> : public TypeTraits<WalkResult::Storage> {
     using Base = TypeTraits<WalkResult::Storage>;
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return src->type_index == TypeIndex::kTVMFFINone || Base::CheckAnyStrict(src);
     }
     // Decode from borrowed Any storage after a strict type check.
     TVM_FFI_INLINE static WalkResult CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFINone) {
         return WalkResult::Advance();
       }
       return WalkResult(Base::CopyFromAnyViewAfterCheck(src));
     }
     // Decode by moving from owned Any storage after a strict type check.
     TVM_FFI_INLINE static WalkResult MoveFromAnyAfterCheck(TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFINone) {
         return WalkResult::Advance();
       }
       return WalkResult(Base::MoveFromAnyAfterCheck(src));
     }
     // Try all conversions supported by the underlying Variant storage.
     TVM_FFI_INLINE static std::optional<WalkResult> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFINone) {
         return WalkResult::Advance();
       }
       if (auto opt = Base::TryCastFromAnyView(src)) {
         return WalkResult(*std::move(opt));
       }
       return std::nullopt;
     }
     TVM_FFI_INLINE static std::string TypeStr() { return "WalkResult"; }
   };
   
   enum class WalkOrder : int32_t {
     kPreOrder = 0,
     kPostOrder = 1,
   };
   
   namespace details {
   
   // Return from the current ABI visit function if Result stops traversal.
   // Result must evaluate to Expected whose raw storage can be moved to TVMFFIAny.
   #define TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Result)                                          \
     do {                                                                                      \
       auto&& tvm_ffi_res_ = (Result);                                                         \
       if (TVM_FFI_PREDICT_FALSE(                                                              \
               ::tvm::ffi::details::StructuralVisitNeedEarlyReturn(tvm_ffi_res_))) {           \
         return ::tvm::ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(tvm_ffi_res_)); \
       }                                                                                       \
     } while (0)
   
   // Return from the current ABI visit function if Result stops traversal.
   // If Result is an Error, append Node to the visit error context before returning.
   #define TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(Result, Node)                   \
     do {                                                                                        \
       auto&& tvm_ffi_res_ = (Result);                                                           \
       if (TVM_FFI_PREDICT_FALSE(                                                                \
               ::tvm::ffi::details::StructuralVisitNeedEarlyReturn(tvm_ffi_res_))) {             \
         if (TVM_FFI_PREDICT_FALSE(tvm_ffi_res_.type_index() ==                                  \
                                   ::tvm::ffi::TypeIndex::kTVMFFIError)) {                       \
           if ((Node).type_index() >= ::tvm::ffi::TypeIndex::kTVMFFIStaticObjectBegin) {         \
             ::tvm::ffi::Error tvm_ffi_visit_err_ = tvm_ffi_res_.error();                        \
             ::tvm::ffi::details::UpdateVisitErrorContext(tvm_ffi_visit_err_,                    \
                                                          (Node).cast<::tvm::ffi::ObjectRef>()); \
           }                                                                                     \
         }                                                                                       \
         return ::tvm::ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(tvm_ffi_res_));   \
       }                                                                                         \
     } while (0)
   
   template <WalkOrder order, typename Dispatch>
   class StructuralWalkCallbackVisitorObj : public StructuralVisitorObj {
    public:
     explicit StructuralWalkCallbackVisitorObj(Dispatch dispatch)
         : StructuralVisitorObj(VTable()), dispatch_(std::move(dispatch)) {}
   
    private:
     static const StructuralVisitorVTable* VTable() {
       static const StructuralVisitorVTable vtable{
           &StructuralWalkCallbackVisitorObj::DispatchVisit,
       };
       return &vtable;
     }
   
     static TVMFFIAny DispatchVisit(StructuralVisitorObj* self, AnyView value) noexcept {
       return static_cast<StructuralWalkCallbackVisitorObj*>(self)->VisitImpl(value);
     }
   
     TVMFFIAny VisitImpl(AnyView value) noexcept {
       if (TVM_FFI_PREDICT_FALSE(value.type_index() == TypeIndex::kTVMFFINone)) {
         return details::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<Optional<VisitInterrupt>>(std::nullopt));
       }
       if constexpr (order == WalkOrder::kPreOrder) {
         auto result = dispatch_(value, this->def_region_kind());
         TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(result, value);
         // Hoist the call out of TVM_FFI_UNSAFE_ASSUME: clang's -Wassume rejects
         // arguments that contain a call expression (its potential side effects
         // would be discarded), while [[maybe_unused]] keeps -Wunused-variable
         // quiet on configs where the assume macro compiles away.
         [[maybe_unused]] int32_t type_index = result.type_index();
         TVM_FFI_UNSAFE_ASSUME(type_index == TypeIndex::kTVMFFIInt);
         if (TVM_FFI_PREDICT_FALSE(details::ExpectedUnsafe::ValueAs<int32_t>(result) ==
                                   WalkResult::kSkipTag)) {
           return details::ExpectedUnsafe::MoveToTVMFFIAny(
               Expected<Optional<VisitInterrupt>>(std::nullopt));
         }
       }
   
       TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(DefaultVisitExpected(value), value);
   
       if constexpr (order == WalkOrder::kPostOrder) {
         TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(
             dispatch_(value, this->def_region_kind()), value);
       }
   
       return details::ExpectedUnsafe::MoveToTVMFFIAny(
           Expected<Optional<VisitInterrupt>>(std::nullopt));
     }
   
     Dispatch dispatch_;
   };
   
   struct StructuralWalkCallbackChain {
     template <typename... Callbacks>
     static auto FromChain(Callbacks... callbacks) {
       return [=](AnyView x, TVMFFIDefRegionKind kind) mutable -> Expected<WalkResult> {
         try {
           Optional<Expected<WalkResult>> result;
           // Fold expression: each TryCallLink returns empty Optional on no-match
           // (falsy) or a result on match (truthy); || short-circuits on first match.
           (... || (result = TryCallLink(callbacks, x, kind)));
           if (result.has_value()) {
             return std::move(result).value();
           }
           return WalkResult::Advance();
         } catch (const Error& err) {
           return Unexpected(err);
         }
       };
     }
   
    private:
     template <typename Callback>
     static Optional<Expected<WalkResult>> TryCallLink(Callback& callback, AnyView x,
                                                       TVMFFIDefRegionKind kind) {
       using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
       static_assert(FuncInfo::num_args == 1 || FuncInfo::num_args == 2,
                     "StructuralWalk callbacks must take one argument (value) or two arguments "
                     "(value, def-region kind)");
       using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
       using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
       if constexpr (std::is_same_v<TSub, AnyView>) {
         // callback on AnyView
         return InvokeCallback(callback, x, kind);
       } else if constexpr (std::is_same_v<TSub, Any>) {
         // callback on Any
         return InvokeCallback(callback, Any(x), kind);
       } else {
         if (auto opt = x.template as<TSub>()) {
           return InvokeCallback(callback, *std::move(opt), kind);
         }
       }
       return std::nullopt;
     }
   
     template <typename Callback, typename Value>
     static Expected<WalkResult> InvokeCallback(Callback& callback, Value&& value,
                                                TVMFFIDefRegionKind kind) {
       using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
       if constexpr (FuncInfo::num_args == 1) {
         return callback(std::forward<Value>(value));
       } else {
         return callback(std::forward<Value>(value), kind);
       }
     }
   };
   
   }  // namespace details
   
   template <WalkOrder order, typename... Callbacks>
   Expected<Optional<VisitInterrupt>> StructuralWalkExpected(AnyView root,
                                                             Callbacks&&... callbacks) noexcept {
     static_assert(sizeof...(Callbacks) != 0, "StructuralWalk requires at least one callback");
     auto dispatch =
         details::StructuralWalkCallbackChain::FromChain(std::forward<Callbacks>(callbacks)...);
     using Visitor = details::StructuralWalkCallbackVisitorObj<order, decltype(dispatch)>;
     StructuralVisitor visitor(make_object<Visitor>(std::move(dispatch)));
     return visitor->VisitExpected(root);
   }
   
   template <WalkOrder order, typename... Callbacks>
   Optional<VisitInterrupt> StructuralWalk(AnyView root, Callbacks&&... callbacks) {
     return StructuralWalkExpected<order>(root, std::forward<Callbacks>(callbacks)...).value();
   }
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
