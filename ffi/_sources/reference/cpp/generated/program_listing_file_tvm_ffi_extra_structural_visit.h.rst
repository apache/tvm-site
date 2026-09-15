
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
     using VisitorObjType = StructuralVisitorObj;
     using StateTupleType = std::tuple<>;
   
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
       // Precedence: a pattern region propagates; entering any kind inside it has no effect.
       if (def_region_mode_ == kTVMFFIDefRegionKindPattern) {
         return std::forward<Callback>(callback)();
       }
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
       return details::ExpectedUnsafe::MoveFromTVMFFIAny<Optional<VisitInterrupt>>(
           DefaultVisitRaw(value));
     }
   
     static constexpr const bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralVisitor", StructuralVisitorObj, Object);
   
    private:
     TVM_FFI_INLINE TVMFFIAny DefaultVisitRaw(AnyView value) noexcept {
       static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
       AnyView attr = column[value.type_index()];
       if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
         return (*reinterpret_cast<FStructuralVisit>(attr.cast<void*>()))(this, value);
       }
       return DefaultVisitRawTail(value, attr);
     }
   
     TVMFFIAny DefaultVisitRawTail(AnyView value, AnyView attr) noexcept {
       if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
         return details::ExpectedUnsafe::MoveToTVMFFIAny(
             attr.cast<Function>().CallExpected<Optional<VisitInterrupt>>(this, value));
       }
       if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFINone)) {
         return details::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<Optional<VisitInterrupt>>(Unexpected(Error(
                 "TypeError", "__s_visit__ must be an opaque function pointer or ffi.Function", ""))));
       }
       if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
         return details::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<Optional<VisitInterrupt>>(std::nullopt));
       }
       return details::ExpectedUnsafe::MoveToTVMFFIAny(
           details::VisitReflectedFieldsExpected(this, value.cast<const Object*>()));
     }
   
    protected:
     TVM_FFI_INLINE StateTupleType StateTuple() const noexcept { return {}; }
   
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
   TVM_FFI_INLINE auto VisitReturnHelper(T&& result) {
     if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                  Optional<VisitInterrupt>>) {
       return std::forward<T>(result);
     } else {
       return ExpectedReturnHelper(std::forward<T>(result));
     }
   }
   
   TVM_FFI_INLINE bool StructuralVisitNeedEarlyReturn(
       const Optional<VisitInterrupt>& result) noexcept {
     return result.has_value();
   }
   
   template <typename T>
   TVM_FFI_INLINE bool StructuralVisitNeedEarlyReturn(const Expected<T>& result) noexcept {
     int32_t type_index = result.type_index();
     return type_index == TypeIndex::kTVMFFIError || type_index == TypeIndex::kTVMFFIVisitInterrupt;
   }
   
   TVM_FFI_INLINE bool StructuralVisitRawNeedEarlyReturn(const TVMFFIAny& result) noexcept {
     return result.type_index != TypeIndex::kTVMFFINone;
   }
   
   // Keep the raw result in registers on success; only error decoration takes its address.
   TVM_FFI_COLD_CODE inline TVMFFIAny AttachStructuralVisitErrorContextRaw(TVMFFIAny result,
                                                                           AnyView value) noexcept {
     UpdateVisitErrorContext(result, value);
     return result;
   }
   
   TVM_FFI_INLINE static Expected<Optional<VisitInterrupt>> VisitReflectedFieldsExpected(
       StructuralVisitorObj* visitor, const Object* obj) noexcept {
     int32_t type_index = obj->type_index();
     const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
     auto visit_fields = [&]() -> Expected<Optional<VisitInterrupt>> {
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
   
             if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefSimple) {
               result = visitor->WithDefRegionKind(
                   kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(field_value); });
             } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefPattern) {
               result = visitor->WithDefRegionKind(
                   kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(field_value); });
             } else {
               result = visitor->VisitExpected(field_value);
             }
             return StructuralVisitNeedEarlyReturn(result);
           });
       return result;
     };
   
     // A simple definition applies to the FreeVar itself, but its fields are uses. The
     // complete field traversal are clamped to None, then the definition region is restored.
     if (visitor->def_region_kind() == kTVMFFIDefRegionKindSimple && type_info->metadata != nullptr &&
         type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
       return visitor->WithDefRegionKind(kTVMFFIDefRegionKindNone, visit_fields);
     }
     return visit_fields();
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
   
     TVM_FFI_INLINE ~WalkResult() = default;
     TVM_FFI_INLINE WalkResult(const WalkResult&) = default;
     TVM_FFI_INLINE WalkResult(WalkResult&&) noexcept = default;
     TVM_FFI_INLINE WalkResult& operator=(const WalkResult&) = default;
     TVM_FFI_INLINE WalkResult& operator=(WalkResult&&) noexcept = default;
   
     TVM_FFI_INLINE static WalkResult Advance() { return WalkResult(kAdvanceTag); }
   
     TVM_FFI_INLINE static WalkResult Skip() { return WalkResult(kSkipTag); }
   
     TVM_FFI_INLINE static WalkResult Interrupt(VisitInterrupt signal = VisitInterrupt()) {
       return WalkResult(Storage(std::move(signal)));
     }
   
    private:
     // Keep raw storage construction behind the named factories.
     TVM_FFI_INLINE explicit WalkResult(int32_t tag) : Storage(tag) {}
     TVM_FFI_INLINE explicit WalkResult(Storage storage) : Storage(std::move(storage)) {}
   
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
   
   #define TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Result)                                \
     do {                                                                            \
       auto&& tvm_ffi_res_ = (Result);                                               \
       if (TVM_FFI_PREDICT_FALSE(                                                    \
               ::tvm::ffi::details::StructuralVisitNeedEarlyReturn(tvm_ffi_res_))) { \
         return ::tvm::ffi::details::VisitReturnHelper(::std::move(tvm_ffi_res_));   \
       }                                                                             \
     } while (0)
   
   }  // namespace details
   
   template <typename Parent, WalkOrder order, typename... Callbacks>
   class StructuralWalkEngine : public Parent {
    public:
     static_assert(std::is_base_of_v<StructuralVisitorObj, Parent>,
                   "StructuralWalk Parent must derive from StructuralVisitorObj");
     using StateTupleType = typename Parent::StateTupleType;
   
     explicit StructuralWalkEngine(Callbacks... callbacks)
         : Parent(VTable()), callbacks_(std::move(callbacks)...) {}
   
    private:
     static const StructuralVisitorVTable* VTable() {
       static const StructuralVisitorVTable vtable{
           &StructuralWalkEngine::DispatchVisit,
       };
       return &vtable;
     }
   
     static TVMFFIAny DispatchVisit(StructuralVisitorObj* self, AnyView value) noexcept {
       return static_cast<StructuralWalkEngine*>(self)->VisitImpl(value);
     }
   
     template <typename Callback, typename Value, size_t... Is>
     TVM_FFI_INLINE Expected<WalkResult> InvokeCallbackLink(Callback& callback, Value&& value,
                                                            std::index_sequence<Is...>) noexcept {
       using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
       static_assert(
           FuncInfo::num_args == 1 + sizeof...(Is) || FuncInfo::num_args == 2 + sizeof...(Is),
           "StructuralWalk callback takes (value, state...) with an optional trailing "
           "definition-region kind");
       try {
         static_assert(std::is_same_v<decltype(Parent::StateTuple()), StateTupleType>,
                       "Parent::StateTuple() must return Parent::StateTupleType by value");
         StateTupleType states = Parent::StateTuple();
         if constexpr (FuncInfo::num_args == 1 + sizeof...(Is)) {
           return callback(std::forward<Value>(value), std::get<Is>(states)...);
         } else {
           return callback(std::forward<Value>(value), std::get<Is>(states)...,
                           Parent::def_region_kind());
         }
       } catch (const Error& err) {
         return Unexpected(err);
       }
     }
   
     template <typename Callback>
     TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value,
                                 Expected<WalkResult>* out) noexcept {
       using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
       static_assert(FuncInfo::num_args >= 1, "StructuralWalk callback requires a value argument");
       using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
       using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
       using StateIndices = std::make_index_sequence<std::tuple_size_v<StateTupleType>>;
       if constexpr (std::is_same_v<TSub, AnyView>) {
         *out = InvokeCallbackLink(callback, value, StateIndices{});
         return true;
       } else if constexpr (std::is_same_v<TSub, Any>) {
         *out = InvokeCallbackLink(callback, Any(value), StateIndices{});
         return true;
       } else if (auto matched = value.template as<TSub>()) {
         *out = InvokeCallbackLink(callback, *std::move(matched), StateIndices{});
         return true;
       }
       return false;
     }
   
     template <size_t... Is>
     TVM_FFI_INLINE bool TryLinks(AnyView value, Expected<WalkResult>* out,
                                  std::index_sequence<Is...>) noexcept {
       return (TryLink(std::get<Is>(callbacks_), value, out) || ...);
     }
   
     TVMFFIAny VisitImpl(AnyView value) noexcept {
       if (TVM_FFI_PREDICT_FALSE(value.type_index() == TypeIndex::kTVMFFINone)) {
         return details::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<Optional<VisitInterrupt>>(std::nullopt));
       }
       if constexpr (order == WalkOrder::kPreOrder) {
         Expected<WalkResult> result = WalkResult::Advance();
         TryLinks(value, &result, std::index_sequence_for<Callbacks...>{});
         if (TVM_FFI_PREDICT_FALSE(details::StructuralVisitNeedEarlyReturn(result))) {
           if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
             Error err = result.error();
             details::UpdateVisitErrorContext(err, value);
           }
           return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
         }
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
   
       {
         TVMFFIAny result =
             details::ExpectedUnsafe::MoveToTVMFFIAny(Parent::DefaultVisitExpected(value));
         if (TVM_FFI_PREDICT_FALSE(details::StructuralVisitRawNeedEarlyReturn(result))) {
           if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
             return details::AttachStructuralVisitErrorContextRaw(result, value);
           }
           return result;
         }
       }
   
       if constexpr (order == WalkOrder::kPostOrder) {
         Expected<WalkResult> result = WalkResult::Advance();
         TryLinks(value, &result, std::index_sequence_for<Callbacks...>{});
         if (TVM_FFI_PREDICT_FALSE(details::StructuralVisitNeedEarlyReturn(result))) {
           if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
             Error err = result.error();
             details::UpdateVisitErrorContext(err, value);
           }
           return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
         }
       }
   
       return details::ExpectedUnsafe::MoveToTVMFFIAny(
           Expected<Optional<VisitInterrupt>>(std::nullopt));
     }
   
     std::tuple<Callbacks...> callbacks_;
   };
   
   template <WalkOrder order, typename... Callbacks>
   Expected<Optional<VisitInterrupt>> StructuralWalkExpected(AnyView root,
                                                             Callbacks&&... callbacks) noexcept {
     static_assert(sizeof...(Callbacks) != 0, "StructuralWalk requires at least one callback");
     using Visitor = StructuralWalkEngine<StructuralVisitorObj, order, std::decay_t<Callbacks>...>;
     StructuralVisitor visitor(make_object<Visitor>(std::forward<Callbacks>(callbacks)...));
     return visitor->VisitExpected(root);
   }
   
   template <WalkOrder order, typename... Callbacks>
   Optional<VisitInterrupt> StructuralWalk(AnyView root, Callbacks&&... callbacks) {
     return StructuralWalkExpected<order>(root, std::forward<Callbacks>(callbacks)...).value();
   }
   
   // ---------------------------------------------------------------------------
   // Structural Visit API.
   // ---------------------------------------------------------------------------
   
   template <typename Parent, typename... Callbacks>
   class StructuralVisitEngine : public Parent {
    public:
     static_assert(std::is_base_of_v<StructuralVisitorObj, Parent>,
                   "StructuralVisit Parent must derive from StructuralVisitorObj");
     explicit StructuralVisitEngine(Callbacks... callbacks)
         : Parent(VTable()), callbacks_(std::move(callbacks)...) {}
   
    private:
     static const StructuralVisitorVTable* VTable() {
       static const StructuralVisitorVTable vtable{
           &StructuralVisitEngine::DispatchVisit,
       };
       return &vtable;
     }
   
     static TVMFFIAny DispatchVisit(StructuralVisitorObj* self, AnyView value) noexcept {
       return static_cast<StructuralVisitEngine*>(self)->VisitImpl(value);
     }
   
     TVMFFIAny VisitImpl(AnyView value) noexcept {
       if (TVM_FFI_PREDICT_FALSE(value.type_index() == TypeIndex::kTVMFFINone)) {
         return details::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<Optional<VisitInterrupt>>(std::nullopt));
       }
       TVMFFIAny result;
       if (!TryLinks(value, &result, std::index_sequence_for<Callbacks...>{})) {
         // Only an unmatched value uses the Parent layer's default descent. A matched
         // callback already traversed as much of the value as it wanted.
         result = details::ExpectedUnsafe::MoveToTVMFFIAny(Parent::DefaultVisitExpected(value));
       }
       if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
         return details::AttachStructuralVisitErrorContextRaw(result, value);
       }
       return result;
     }
   
     template <typename Callback>
     inline bool TryLink(Callback& callback, AnyView value, TVMFFIAny* out) noexcept {
       using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
       static_assert(FuncInfo::num_args == 2, "StructuralVisit callback takes (value, visitor)");
       using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
       using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
       using SecondArg = std::decay_t<std::tuple_element_t<1, typename FuncInfo::ArgType>>;
       using Second = std::remove_pointer_t<SecondArg>;
       static_assert(std::is_same_v<Second, typename Parent::VisitorObjType>,
                     "second StructuralVisit callback argument must be "
                     "exactly Parent::VisitorObjType*");
       auto* visitor = static_cast<typename Parent::VisitorObjType*>(this);
       try {
         if constexpr (std::is_same_v<TSub, AnyView>) {
           *out = details::ExpectedUnsafe::MoveToTVMFFIAny(
               Expected<Optional<VisitInterrupt>>(callback(value, visitor)));
           return true;
         } else if constexpr (std::is_same_v<TSub, Any>) {
           *out = details::ExpectedUnsafe::MoveToTVMFFIAny(
               Expected<Optional<VisitInterrupt>>(callback(Any(value), visitor)));
           return true;
         } else if (auto matched = value.template as<TSub>()) {
           *out = details::ExpectedUnsafe::MoveToTVMFFIAny(
               Expected<Optional<VisitInterrupt>>(callback(*std::move(matched), visitor)));
           return true;
         }
       } catch (const Error& err) {
         *out = details::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<Optional<VisitInterrupt>>(Unexpected(err)));
         return true;
       }
       return false;
     }
   
     template <size_t... Is>
     TVM_FFI_INLINE bool TryLinks(AnyView value, TVMFFIAny* out, std::index_sequence<Is...>) noexcept {
       return (TryLink(std::get<Is>(callbacks_), value, out) || ...);
     }
   
     std::tuple<Callbacks...> callbacks_;
   };
   
   template <typename... Callbacks>
   Expected<Optional<VisitInterrupt>> StructuralVisitExpected(AnyView root,
                                                              Callbacks&&... callbacks) noexcept {
     static_assert(sizeof...(Callbacks) != 0, "StructuralVisit requires at least one callback");
     using Engine = StructuralVisitEngine<StructuralVisitorObj, std::decay_t<Callbacks>...>;
     StructuralVisitor visitor(make_object<Engine>(std::forward<Callbacks>(callbacks)...));
     return visitor->VisitExpected(root);
   }
   
   template <typename... Callbacks>
   Optional<VisitInterrupt> StructuralVisit(AnyView root, Callbacks&&... callbacks) {
     return StructuralVisitExpected(root, std::forward<Callbacks>(callbacks)...).value();
   }
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
