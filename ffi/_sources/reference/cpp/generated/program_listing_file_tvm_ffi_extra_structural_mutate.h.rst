
.. _program_listing_file_tvm_ffi_extra_structural_mutate.h:

Program Listing for File structural_mutate.h
============================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_structural_mutate.h>` (``tvm/ffi/extra/structural_mutate.h``)

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
   #ifndef TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
   #define TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ffi/container/variant.h>
   #include <tvm/ffi/expected.h>
   #include <tvm/ffi/extra/structural_visit.h>
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
   #include <unordered_map>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   enum class InplaceMode : int32_t {
     kDisallow = 0,
     kAllow = 1,
   };
   
   class StructuralMutatorObj;
   template <typename T>
   class UnchangedOr;
   
   
   template <typename T, typename U>
   inline constexpr bool type_subsumes_v<UnchangedOr<T>, UnchangedOr<U>> = type_subsumes_v<T, U>;
   
   template <typename Parent, WalkOrder order, typename... Callbacks>
   class StructuralMapEngine;
   template <typename Parent, WalkOrder order>
   class StructuralMapDynEngine;
   template <typename Parent, typename... Callbacks>
   class StructuralMutateEngine;
   
   using FStructuralMutate = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView value) noexcept;
   
   using FStructuralVarRemapGet = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView var) noexcept;
   
   using FStructuralVarRemapSet = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView var,
                                                AnyView mapped_value) noexcept;
   
   namespace details {
   
   // Copy and structurally mutate the reflected fields of an object-backed value.
   TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                     AnyView value) noexcept;
   
   }  // namespace details
   
   struct StructuralMutatorVTable {
     FStructuralMutate mutate = nullptr;
     FStructuralMutate maybe_inplace_mutate = nullptr;
     FStructuralVarRemapGet var_remap_get = nullptr;
     FStructuralVarRemapSet var_remap_set = nullptr;
   };
   
   namespace details {
   template <typename Parent>
   class StructuralMutateDynEngine;
   
   struct UnchangedOrUnsafe;
   
   template <typename T>
   inline constexpr bool is_unchanged_or_v = false;
   
   template <typename T>
   inline constexpr bool is_unchanged_or_v<UnchangedOr<T>> = true;
   }  // namespace details
   
   struct Unchanged {
     TVM_FFI_INLINE TVMFFIAny CopyToTVMFFIAny() const noexcept {
       // The marker needs a reserved type index because every ordinary index is a legal mutation
       // result. In particular, kTVMFFINone is a valid replacement and cannot double as the marker.
       TVMFFIAny raw;
       raw.type_index = TypeIndex::kTVMFFIUnchanged;
       // invariance: always set the union padding part to 0
       raw.zero_padding = 0;
       raw.v_int64 = 0;
       return raw;
     }
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     TVM_FFI_INLINE operator Any() const noexcept {
       TVMFFIAny raw = CopyToTVMFFIAny();
       return details::AnyUnsafe::MoveTVMFFIAnyRawToAny(raw);
     }
   };
   
   template <typename T>
   class UnchangedOr {
    public:
     static_assert(!std::is_base_of_v<Error, std::remove_cv_t<T>>,
                   "UnchangedOr<Error> is not supported");
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     TVM_FFI_INLINE UnchangedOr(Unchanged unchanged) noexcept : data_(static_cast<Any>(unchanged)) {}
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     TVM_FFI_INLINE UnchangedOr(T value) : data_(Any(std::move(value))) {}
   
     // Preserve the dedicated tag and wrapper routes. Subsumption applies only to materialized
     // wrapper storage; a bare value must first be implicitly convertible to T.
     template <typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, Unchanged> &&
                                                       !details::is_unchanged_or_v<std::decay_t<U>> &&
                                                       !details::is_expected_v<std::decay_t<U>> &&
                                                       !details::is_unexpected_v<std::decay_t<U>> &&
                                                       !std::is_base_of_v<Error, std::decay_t<U>> &&
                                                       std::is_convertible_v<U, T>>>
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     TVM_FFI_INLINE UnchangedOr(U&& value) : data_(Any(T(std::forward<U>(value)))) {}
   
     template <typename U,
               typename = std::enable_if_t<type_subsumes_v<T, U> || std::is_convertible_v<U, T>>>
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     TVM_FFI_INLINE UnchangedOr(UnchangedOr<U> other)
         : data_([&other]() {
             if constexpr (type_subsumes_v<T, U>) {
               // Reuse materialized storage, including the unchanged marker.
               return details::AnyUnsafe::MoveTVMFFIAnyRawToAny(
                   details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(other.data_)));
             } else {
               return other.IsUnchanged() ? std::move(other.data_)
                                          : Any(T(std::move(other).ValueUnchecked()));
             }
           }()) {}
   
     TVM_FFI_INLINE UnchangedOr(const UnchangedOr&) = default;
     TVM_FFI_INLINE UnchangedOr(UnchangedOr&&) noexcept = default;
     TVM_FFI_INLINE ~UnchangedOr() = default;
     TVM_FFI_INLINE UnchangedOr& operator=(const UnchangedOr&) = default;
     TVM_FFI_INLINE UnchangedOr& operator=(UnchangedOr&&) noexcept = default;
   
     TVM_FFI_INLINE bool IsUnchanged() const& noexcept {
       return data_.type_index() == TypeIndex::kTVMFFIUnchanged;
     }
   
     TVM_FFI_INLINE bool UnchangedOrSameAs(const T& original) const& noexcept {
       return IsUnchanged() || data_.same_as(original);
     }
   
     TVM_FFI_INLINE T ValueOrUnchanged(const T& original) && {
       return IsUnchanged() ? original
                            : details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
     }
   
     TVM_FFI_INLINE T ValueOrUnchanged(T&& original) && {
       return IsUnchanged() ? std::move(original)
                            : details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
     }
   
     template <typename U = T,
               typename = std::enable_if_t<std::is_same_v<T, Any> && std::is_same_v<U, T>>>
     TVM_FFI_INLINE Any ValueOrUnchanged(AnyView original) && {
       return IsUnchanged() ? Any(original)
                            : details::AnyUnsafe::MoveFromAnyAfterCheck<Any>(std::move(data_));
     }
   
     TVM_FFI_INLINE T ValueUnchecked() && {
       return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
     }
   
     template <typename U,
               typename = std::enable_if_t<TypeTraits<U>::storage_enabled || std::is_same_v<U, Any>>>
     TVM_FFI_INLINE std::optional<U> as() && {
       return std::move(data_).template as<U>();
     }
   
     template <typename U,
               typename = std::enable_if_t<TypeTraits<U>::storage_enabled || std::is_same_v<U, Any>>>
     TVM_FFI_INLINE U as_or_throw() && {
       return std::move(data_).template as_or_throw<U>();
     }
   
    private:
     template <typename>
     friend class UnchangedOr;
     friend struct details::UnchangedOrUnsafe;
     template <typename, typename>
     friend struct TypeTraits;
     struct UnsafeInit {};
     TVM_FFI_INLINE explicit UnchangedOr(UnsafeInit, Any data) noexcept : data_(std::move(data)) {}
     Any data_;
   };
   
   namespace details {
   struct UnchangedOrUnsafe {
     template <typename T>
     TVM_FFI_INLINE static UnchangedOr<T> MoveFromTVMFFIAny(TVMFFIAny raw) {
       return UnchangedOr<T>(typename UnchangedOr<T>::UnsafeInit{},
                             AnyUnsafe::MoveTVMFFIAnyRawToAny(raw));
     }
   
     template <typename T>
     TVM_FFI_INLINE static TVMFFIAny MoveToTVMFFIAny(UnchangedOr<T>&& result) noexcept {
       return AnyUnsafe::MoveAnyToTVMFFIAny(std::move(result.data_));
     }
   };
   
   }  // namespace details
   
   namespace details {
   // Out of line so its strings and Error construction stay out of the hot path of whatever hook
   // body TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN expands into. Same reason as
   // BadStructuralMutateHookError.
   // Takes nothing on purpose. Naming the offending type in the message would keep the result live
   // across the predicted-not-taken guard in the hot path. The declared type is already present in
   // the source line to which the diagnostic points.
   TVM_FFI_COLD_CODE inline UnexpectedReturnHelper SMutateDeclaredTypeError() noexcept {
     return UnexpectedReturnHelper(Unexpected(
         Error("TypeError", "structural mutate result does not match the declared type", "")));
   }
   }  // namespace details
   
   class StructuralMutatorObj : public Object {
    public:
     using MutatorObjType = StructuralMutatorObj;
   
     TVM_FFI_INLINE UnchangedOr<Any> Mutate(AnyView value,
                                            InplaceMode inplace_mode = InplaceMode::kDisallow) {
       return std::move(MutateExpected(value, inplace_mode)).value();
     }
   
     TVM_FFI_INLINE Expected<UnchangedOr<Any>> MutateExpected(
         AnyView value, InplaceMode inplace_mode = InplaceMode::kDisallow) noexcept {
       const Object* object = value.as<Object>();
       // Check uniqueness on the borrowed view before callbacks can acquire owning references.
       if (inplace_mode == InplaceMode::kAllow && object != nullptr && object->unique()) {
         return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(
             (*vtable_->maybe_inplace_mutate)(this, value));
       }
       return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(
           (*vtable_->mutate)(this, value));
     }
   
     TVM_FFI_INLINE Expected<UnchangedOr<Any>> DefaultMutateExpected(
         AnyView value, InplaceMode inplace_mode = InplaceMode::kDisallow) noexcept {
       return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(
           inplace_mode == InplaceMode::kAllow ? DefaultMaybeInplaceMutateRaw(value)
                                               : DefaultMutateRaw(value));
     }
   
     TVM_FFI_INLINE Expected<Any> VarRemapGetExpected(AnyView var) noexcept {
       return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*vtable_->var_remap_get)(this, var));
     }
   
     TVM_FFI_INLINE Expected<void> VarRemapSetExpected(AnyView var, AnyView mapped_value) noexcept {
       return details::ExpectedUnsafe::MoveFromTVMFFIAny<void>(
           (*vtable_->var_remap_set)(this, var, mapped_value));
     }
   
     TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }
   
     template <typename Callback>
     TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
         -> decltype(std::forward<Callback>(callback)()) {
       // Precedence: a pattern region propagates; entering any kind inside it has no effect.
       if (def_region_mode_ == kTVMFFIDefRegionKindPattern) {
         return std::forward<Callback>(callback)();
       }
       class Scope {
        public:
         Scope(StructuralMutatorObj* mutator, TVMFFIDefRegionKind kind)
             : mutator_(mutator), old_kind_(mutator->def_region_mode_) {
           mutator_->def_region_mode_ = kind;
         }
         ~Scope() { mutator_->def_region_mode_ = old_kind_; }
         Scope(const Scope&) = delete;
         Scope& operator=(const Scope&) = delete;
   
        private:
         StructuralMutatorObj* mutator_;
         TVMFFIDefRegionKind old_kind_;
       };
       Scope scope(this, kind);
       return std::forward<Callback>(callback)();
     }
   
     static constexpr const bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralMutator", StructuralMutatorObj, Object);
   
    private:
     template <typename Parent>
     friend class details::StructuralMutateDynEngine;
   
     TVM_FFI_COLD_CODE static Expected<Any> BadStructuralMutateHookError() noexcept {
       return Unexpected(
           Error("TypeError", "__s_mutate__ must be an opaque function pointer or ffi.Function", ""));
     }
   
     // Convention: the ABI boundary is a raw TVMFFIAny; mutation results inside a callback or hook
     // body use Expected<Any> and move out to TVMFFIAny at that boundary. Unchanged converts to an
     // Any carrying kTVMFFIUnchanged.
     //
     // The Raw forms below exist because that boundary is also the default path. A hook is a C-ABI
     // function pointer returning TVMFFIAny, a 16-byte POD that stays in registers; wrapping the
     // result in Expected<Any> would force it to memory because the C++ wrapper is not
     // trivially destructible and is therefore classified MEMORY. Descent through an unmatched node
     // calls a hook and returns its result unchanged, so keeping that path raw removes the round trip
     // entirely. Only a matched callback pays for the Expected wrapper.
     //
     // Engine-internal: subclasses call the Expected forms above.
     TVM_FFI_INLINE TVMFFIAny DefaultMutateRaw(AnyView value) noexcept {
       static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMutate);
       AnyView attr = column[value.type_index()];
       // Exactly one frame per node: hooks propagate errors untouched, and this is the engine
       // dispatching into `value`, so both exits below name it here and nowhere else.
       TVMFFIAny result;
       if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
         result = (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(this, value);
       } else {
         result = DefaultMutateRawTail(value, attr);
       }
       if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
         return AttachVisitErrorContextRaw(result, value);
       }
       return result;
     }
   
     TVM_FFI_COLD_CODE static TVMFFIAny AttachVisitErrorContextRaw(TVMFFIAny result,
                                                                   AnyView value) noexcept {
       details::UpdateVisitErrorContext(result, value);
       return result;
     }
   
     TVMFFIAny DefaultMutateRawTail(AnyView value, AnyView attr) noexcept {
       if (attr.type_index() != TypeIndex::kTVMFFINone) {
         // Registered, but as an ffi.Function rather than an opaque pointer.
         if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
           return details::ExpectedUnsafe::MoveToTVMFFIAny(
               attr.cast<Function>().CallExpected<Any>(this, value));
         }
         // Registered as neither: a malformed hook.
         return details::ExpectedUnsafe::MoveToTVMFFIAny(BadStructuralMutateHookError());
       }
       // No hook at all. A POD carries through unchanged; an object walks its reflected fields.
       if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
         return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
       }
       const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value.type_index());
       const int32_t identity_kind = type_info->metadata == nullptr
                                         ? kTVMFFISEqHashKindUnsupported
                                         : type_info->metadata->structural_eq_hash_kind;
       const bool is_free_var = identity_kind == kTVMFFISEqHashKindFreeVar;
       const bool is_dag_node = identity_kind == kTVMFFISEqHashKindDAGNode;
       if (is_free_var || is_dag_node) {
         // Only None means no cached descent result; every other value, including the unchanged
         // marker used by a pattern definition, is returned directly.
         Expected<Any> mapped = VarRemapGetExpected(value);
         if (details::ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
           return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(mapped));
         }
       }
   
       // A FreeVar outside a definition region is a use. A miss means its definition was unchanged
       // (or it is free), so there is no field descent and no remap insertion.
       if (is_free_var && def_region_kind() == kTVMFFIDefRegionKindNone) {
         return Unchanged().CopyToTVMFFIAny();
       }
   
       Expected<Any> result = details::MutateReflectedFieldsExpected(this, value);
       if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
         return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
       }
       if (is_free_var || is_dag_node) {
         const Any& result_value = details::ExpectedUnsafe::GetData(result);
         // Bind the descent result. The one exception is an unchanged simple definition: its
         // uses resolve to the var itself on a miss, so there is nothing to record.
         if (is_dag_node || def_region_kind() == kTVMFFIDefRegionKindPattern ||
             result_value.type_index() != TypeIndex::kTVMFFIUnchanged) {
           Expected<void> set_result = VarRemapSetExpected(value, result_value);
           if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
             return details::ExpectedUnsafe::MoveToTVMFFIAny(
                 Expected<Any>(Unexpected(std::move(set_result).error())));
           }
         }
       }
       return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
     }
     TVM_FFI_INLINE TVMFFIAny DefaultMaybeInplaceMutateRaw(AnyView value) noexcept {
       static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMaybeInplaceMutate);
       AnyView attr = column[value.type_index()];
       if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
         // This is the engine dispatching into `value`; hooks propagate errors untouched, so the
         // node is named here. The fall-through re-dispatches the same node through
         // DefaultMutateRaw, which names it there instead -- exactly one frame either way.
         TVMFFIAny result = (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(this, value);
         if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
           return AttachVisitErrorContextRaw(result, value);
         }
         return result;
       }
       return DefaultMaybeInplaceMutateRawTail(value, attr);
     }
   
     TVMFFIAny DefaultMaybeInplaceMutateRawTail(AnyView value, AnyView attr) noexcept {
       if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
         TVMFFIAny result = details::ExpectedUnsafe::MoveToTVMFFIAny(
             attr.cast<Function>().CallExpected<Any>(this, value));
         if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
           return AttachVisitErrorContextRaw(result, value);
         }
         return result;
       }
       return details::ExpectedUnsafe::MoveToTVMFFIAny(
           DefaultMutateExpected(value, InplaceMode::kDisallow));
     }
   
    protected:
     explicit StructuralMutatorObj(const StructuralMutatorVTable* vtable) : vtable_(vtable) {}
   
     const StructuralMutatorVTable* vtable_ = nullptr;
   
     TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
   };
   
   class StructuralMutator : public ObjectRef {
    public:
     explicit StructuralMutator(ObjectPtr<StructuralMutatorObj> n) : ObjectRef(std::move(n)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralMutator, ObjectRef, StructuralMutatorObj);
   };
   
   namespace details {
   
   TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                     AnyView value) noexcept {
     const Object* obj = value.as<Object>();
     int32_t type_index = obj->type_index();
   
     static reflection::TypeAttrColumn column(reflection::type_attr::kShallowCopy);
     AnyView attr = column[type_index];
     if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFIFunction)) {
       return Unexpected(Error("TypeError", "__ffi_shallow_copy__ must be an ffi.Function", ""));
     }
   
     Expected<Any> result = attr.cast<Function>().CallExpected<Any>(value);
     if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
       return result;
     }
   
     const Any& result_value = details::ExpectedUnsafe::GetData(result);
     Object* new_obj = const_cast<Object*>(result_value.as<Object>());
     // Copy-on-write mutation requires a distinct target so partial updates cannot modify the source.
     if (TVM_FFI_PREDICT_FALSE(new_obj == nullptr || result.type_index() != value.type_index() ||
                               new_obj == obj)) {
       return Unexpected(Error(
           "TypeError",
           "Shallow copy callback must return a distinct object with the same type as its input", ""));
     }
   
     const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(new_obj->type_index());
     bool field_changed = false;
     auto mutate_fields = [&]() {
       reflection::ForEachFieldInfoWithEarlyStop(
           type_info, [&](const TVMFFIFieldInfo* field_info) -> bool {
             if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
               return false;
             }
   
             Any field_value;
             void* field_addr = reinterpret_cast<char*>(new_obj) + field_info->offset;
             int ret_code = field_info->getter(field_addr, reinterpret_cast<TVMFFIAny*>(&field_value));
             if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
               result = Unexpected(details::MoveFromSafeCallRaised());
               return true;
             }
   
             // Reflected fields use the same unchanged-or-value descent protocol.
             Expected<UnchangedOr<Any>> mutated_field = [&]() -> Expected<UnchangedOr<Any>> {
               if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefSimple) {
                 return mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                   return mutator->MutateExpected(field_value, InplaceMode::kDisallow);
                 });
               } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefPattern) {
                 return mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                   return mutator->MutateExpected(field_value, InplaceMode::kDisallow);
                 });
               } else {
                 return mutator->MutateExpected(field_value, InplaceMode::kDisallow);
               }
             }();
             if (TVM_FFI_PREDICT_FALSE(mutated_field.is_err())) {
               result = Unexpected(std::move(mutated_field).error());
               return true;
             }
             const Any& mutated_field_data = details::ExpectedUnsafe::GetData(mutated_field);
             // Unchanged first: it is the common case, and it is one type-index test where the
             // resolved form ran a full same_as against a value it had just been handed back.
             if (mutated_field_data.type_index() == TypeIndex::kTVMFFIUnchanged ||
                 field_value.same_as(mutated_field_data)) {
               return false;
             }
   
             if (TVM_FFI_PREDICT_FALSE(field_info->setter == nullptr)) {
               result = Unexpected(Error(
                   "TypeError",
                   "Cannot structurally mutate field `" +
                       std::string(field_info->name.data, field_info->name.size) + "` of type `" +
                       std::string(type_info->type_key.data, type_info->type_key.size) +
                       "` because it does not define a setter",
                   ""));
               return true;
             }
   
             ret_code = reflection::CallFieldSetter(
                 field_info, field_addr, reinterpret_cast<const TVMFFIAny*>(&mutated_field_data));
             if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
               result = Unexpected(details::MoveFromSafeCallRaised());
               return true;
             }
             field_changed = true;
             return false;
           });
     };
   
     // A simple definition applies to the FreeVar itself, but its fields are uses. The
     // complete field traversal are clamped to None, then the definition region is restored.
     if (mutator->def_region_kind() == kTVMFFIDefRegionKindSimple && type_info->metadata != nullptr &&
         type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
       mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_fields);
     } else {
       mutate_fields();
     }
   
     if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
       return result;
     }
     if (!field_changed) {
       return Unchanged();
     }
     return result;
   }
   
   }  // namespace details
   
   // ---------------------------------------------------------------------------
   // Structural Map API.
   // ---------------------------------------------------------------------------
   
   namespace details {
   // Return an error from the current raw or Expected mutation function.
   // The rvalue-only helper lets the enclosing return type select the representation.
   #define TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result)                   \
     do {                                                                \
       auto&& tvm_ffi_res_ = (Result);                                   \
       if (TVM_FFI_PREDICT_FALSE(tvm_ffi_res_.is_err())) {               \
         return ::tvm::ffi::details::UnexpectedReturnHelper(             \
             ::tvm::ffi::Unexpected(::std::move(tvm_ffi_res_).error())); \
       }                                                                 \
     } while (0)
   
   
   #define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr)               \
     auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                        \
     TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                                \
     if constexpr (!::tvm::ffi::type_subsumes_v<::tvm::ffi::Expected<Type>, decltype(Result)>) { \
       if (TVM_FFI_PREDICT_FALSE(!::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<Type>(          \
               ::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))) {                         \
         return ::tvm::ffi::details::SMutateDeclaredTypeError();                                 \
       }                                                                                         \
     }                                                                                           \
     Type Name = /* NOLINT(bugprone-macro-parentheses) */                                        \
         ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                            \
             ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
   
   #define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, Name, ResultExpr)                                  \
     TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), \
                                             Type, Name, ResultExpr)
   
   #define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr) \
     auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                 \
     TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                         \
     Type Name = /* NOLINT(bugprone-macro-parentheses) */                                 \
         ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                     \
             ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
   
   #define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN(Type, Name, ResultExpr) \
     TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(                        \
         TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), Type, Name, ResultExpr)
   
   }  // namespace details
   
   class StructuralMapEngineBase : public StructuralMutatorObj {
    public:
     using StateTupleType = std::tuple<>;
   
     explicit StructuralMapEngineBase(const StructuralMutatorVTable* vtable)
         : StructuralMutatorObj(vtable) {}
   
     ~StructuralMapEngineBase() {
       for (const auto& kv : var_remap_) {
         details::ObjectUnsafe::DecRefObjectHandle(
             reinterpret_cast<TVMFFIObjectHandle>(const_cast<Object*>(kv.first)));
       }
     }
   
    protected:
     TVM_FFI_INLINE StateTupleType StateTuple() const noexcept { return {}; }
   
     // Out of line so its strings stay out of the per-node dispatch function, which TryLink inlines
     // into. Shared by the typed and dynamic engines below.
     TVM_FFI_COLD_CODE static Expected<Any> SMutateDescentTypeError() noexcept {
       return Unexpected(Error("TypeError", "structural mutate: descent changed the node type", ""));
     }
   
     TVM_FFI_COLD_CODE static details::UnexpectedReturnHelper VarRemapKeyTypeError() noexcept {
       return details::UnexpectedReturnHelper(
           Unexpected(Error("TypeError", "Variable-remap key must be an object-backed value", "")));
     }
   
     static TVMFFIAny DispatchVarRemapGet(StructuralMutatorObj* mutator, AnyView var) noexcept {
       auto* self = static_cast<StructuralMapEngineBase*>(mutator);
       return details::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapGetImpl(var));
     }
   
     static TVMFFIAny DispatchVarRemapSet(StructuralMutatorObj* mutator, AnyView var,
                                          AnyView mapped_value) noexcept {
       auto* self = static_cast<StructuralMapEngineBase*>(mutator);
       return details::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapSetImpl(var, mapped_value));
     }
   
     TVM_FFI_COLD_CODE static void UpdateVisitErrorContext(const Expected<Any>& result,
                                                           AnyView node) noexcept {
       // The Error is refcounted, so annotating the local handle annotates the object the result
       // holds. A non-object node has no context to add.
       if (node.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
         Error err = result.error();
         ::tvm::ffi::details::UpdateVisitErrorContext(err, node.cast<ObjectRef>());
       }
     }
   
     Expected<Any> VarRemapGetImpl(AnyView var) noexcept {
       if (TVM_FFI_PREDICT_FALSE(var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin)) {
         return VarRemapKeyTypeError();
       }
       if (var_remap_.empty()) return Any(nullptr);
       const Object* var_ptr =
           details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const Object>(var);
       auto it = var_remap_.find(var_ptr);
       return it == var_remap_.end() ? Any(nullptr) : it->second;
     }
   
     Expected<void> VarRemapSetImpl(AnyView var, AnyView mapped_value) noexcept {
       if (TVM_FFI_PREDICT_FALSE(var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin)) {
         return VarRemapKeyTypeError();
       }
       const Object* var_ptr =
           details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const Object>(var);
       Any owned_mapped_value(mapped_value);
       auto [it, inserted] = var_remap_.try_emplace(var_ptr, std::move(owned_mapped_value));
       if (inserted) {
         details::ObjectUnsafe::IncRefObjectHandle(
             reinterpret_cast<TVMFFIObjectHandle>(const_cast<Object*>(var_ptr)));
       } else {
         it->second = std::move(owned_mapped_value);
       }
       return Expected<void>();
     }
   
     template <typename Parent, WalkOrder order, typename... Callbacks>
     friend class StructuralMapEngine;
     template <typename Parent, WalkOrder order>
     friend class StructuralMapDynEngine;
     template <typename Parent, typename... Callbacks>
     friend class StructuralMutateEngine;
     template <typename Parent>
     friend class details::StructuralMutateDynEngine;
   
     // Raw-pointer key: IncRef once on first insert, DecRef all keys in the destructor.
     std::unordered_map<const Object*, Any> var_remap_;
   };
   
   template <typename Parent, WalkOrder order, typename... Callbacks>
   class StructuralMapEngine : public Parent {
    public:
     static_assert(std::is_base_of_v<StructuralMapEngineBase, Parent>,
                   "StructuralMap Parent must derive from StructuralMapEngineBase");
     using StateTupleType = typename Parent::StateTupleType;
   
     explicit StructuralMapEngine(Callbacks... callbacks)
         : Parent(VTable()), callbacks_(std::move(callbacks)...) {}
   
    private:
     using ExpectedUnsafe = details::ExpectedUnsafe;
     using AnyUnsafe = details::AnyUnsafe;
   
     static const StructuralMutatorVTable* VTable() {
       static const StructuralMutatorVTable vtable{
           &StructuralMapEngine::DispatchMutate,
           &StructuralMapEngine::DispatchMaybeInplaceMutate,
           &StructuralMapEngine::DispatchVarRemapGet,
           &StructuralMapEngine::DispatchVarRemapSet,
       };
       return &vtable;
     }
   
     static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                 AnyView value) noexcept {
       auto* self = static_cast<StructuralMapEngine*>(mutator);
       return self->MaybeInplaceMutateImplRaw(value);
     }
   
     static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
       auto* self = static_cast<StructuralMapEngine*>(mutator);
       return self->MutateImplRaw(value);
     }
   
     template <typename Callback, typename Value, size_t... Is>
     TVM_FFI_INLINE Expected<Any> InvokeTypedCallbackLink(Callback& callback, Value&& value,
                                                          std::index_sequence<Is...>) noexcept {
       using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
       static_assert(std::is_convertible_v<typename FuncInfo::RetType, Expected<Any>>,
                     "StructuralMap callbacks must return a replacement value, Error, Unexpected, "
                     "unchanged marker, or Expected<Any>");
       static_assert(
           FuncInfo::num_args == 1 + sizeof...(Is) || FuncInfo::num_args == 2 + sizeof...(Is),
           "StructuralMap callback takes (value, state...) with an optional trailing "
           "definition-region kind");
       try {
         static_assert(std::is_same_v<decltype(this->StateTuple()), StateTupleType>,
                       "Parent::StateTuple() must return Parent::StateTupleType by value");
         StateTupleType states = this->StateTuple();
         if constexpr (FuncInfo::num_args == 1 + sizeof...(Is)) {
           return callback(std::forward<Value>(value), std::get<Is>(states)...);
         } else {
           return callback(std::forward<Value>(value), std::get<Is>(states)...,
                           this->def_region_kind());
         }
       } catch (const Error& err) {
         return Unexpected(err);
       }
     }
   
     template <InplaceMode kInplaceMode, typename Callback>
     TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value, Expected<Any>* out) noexcept {
       using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
       static_assert(FuncInfo::num_args >= 1,
                     "StructuralMap callback must take at least a value argument");
       using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
       using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
   
       using StateIndices = std::make_index_sequence<std::tuple_size_v<StateTupleType>>;
       if constexpr (order == WalkOrder::kPreOrder) {
         std::optional<TSub> matched;
         if constexpr (!std::is_same_v<TSub, AnyView> && !std::is_same_v<TSub, Any>) {
           matched = value.template as<TSub>();
           if (!matched.has_value()) return false;
         }
         // Pre-order: the callback rewrites this node first, then descent runs over whatever it
         // produced, so a replacement subtree is itself mapped.
         Expected<Any> callback_result = [&]() -> Expected<Any> {
           if constexpr (std::is_same_v<TSub, AnyView>) {
             return InvokeTypedCallbackLink(callback, value, StateIndices{});
           } else if constexpr (std::is_same_v<TSub, Any>) {
             return InvokeTypedCallbackLink(callback, Any(value), StateIndices{});
           } else {
             // Reuses the conversion the match already performed.
             return InvokeTypedCallbackLink(callback, *std::move(matched), StateIndices{});
           }
         }();
         if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
           this->UpdateVisitErrorContext(callback_result, value);
           *out = std::move(callback_result);
           return true;
         }
         Any mapped_value = std::move(ExpectedUnsafe::GetData(callback_result));
         const AnyView descent_view =
             mapped_value.type_index() == TypeIndex::kTVMFFIUnchanged ? value : AnyView(mapped_value);
         // Each descent names the node it actually ran on in the error context.
         *out = [&]() -> Expected<Any> {
           if constexpr (kInplaceMode == InplaceMode::kAllow) {
             // A pre-order result can be mutated in place if unchanged or uniquely owned.
             if (descent_view.same_as(value)) {
               return this->DefaultMutateExpected(value, InplaceMode::kAllow);
             }
             const Object* mapped_obj = descent_view.as<Object>();
             InplaceMode inplace_mode = mapped_obj != nullptr && mapped_obj->unique()
                                            ? InplaceMode::kAllow
                                            : InplaceMode::kDisallow;
             return this->DefaultMutateExpected(descent_view, inplace_mode);
           } else {
             return this->DefaultMutateExpected(descent_view, InplaceMode::kDisallow);
           }
         }();
         if (TVM_FFI_PREDICT_FALSE(out->is_err())) return true;
         if (ExpectedUnsafe::GetData(*out).type_index() == TypeIndex::kTVMFFIUnchanged) {
           *out = std::move(mapped_value);
         }
         return true;
       } else {
         // Post-order descent is performed once by the engine entry before the callback probes.
         // Descended unchanged uses the original view; otherwise the callback sees the replacement.
         const Any& descended_value = ExpectedUnsafe::GetData(*out);
         const AnyView mapped_view = descended_value.type_index() == TypeIndex::kTVMFFIUnchanged
                                         ? value
                                         : AnyView(descended_value);
         std::optional<TSub> matched;
         if constexpr (!std::is_same_v<TSub, AnyView> && !std::is_same_v<TSub, Any>) {
           matched = mapped_view.template as<TSub>();
           if (!matched.has_value()) return false;
         }
         *out = [&]() -> Expected<Any> {
           if constexpr (std::is_same_v<TSub, AnyView>) {
             return InvokeTypedCallbackLink(callback, mapped_view, StateIndices{});
           } else if constexpr (std::is_same_v<TSub, Any>) {
             return InvokeTypedCallbackLink(callback, Any(mapped_view), StateIndices{});
           } else {
             return InvokeTypedCallbackLink(callback, *std::move(matched), StateIndices{});
           }
         }();
         if (TVM_FFI_PREDICT_FALSE(out->is_err())) {
           this->UpdateVisitErrorContext(*out, mapped_view);
           return true;
         }
         return true;
       }
     }
   
     template <InplaceMode kInplaceMode, size_t... Is>
     TVM_FFI_INLINE bool TryLinks(AnyView value, Expected<Any>* out,
                                  std::index_sequence<Is...>) noexcept {
       return (TryLink<kInplaceMode>(std::get<Is>(callbacks_), value, out) || ...);
     }
   
     TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
       Expected<Any> out{Any()};
       if constexpr (order == WalkOrder::kPostOrder) {
         out = this->DefaultMutateExpected(value, InplaceMode::kDisallow);
         if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         TryLinks<InplaceMode::kDisallow>(value, &out, std::index_sequence_for<Callbacks...>{});
         return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
       } else {
         if (TryLinks<InplaceMode::kDisallow>(value, &out, std::index_sequence_for<Callbacks...>{})) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         return ExpectedUnsafe::MoveToTVMFFIAny(
             this->DefaultMutateExpected(value, InplaceMode::kDisallow));
       }
     }
   
     TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
       Expected<Any> out{Any()};
       if constexpr (order == WalkOrder::kPostOrder) {
         out = this->DefaultMutateExpected(value, InplaceMode::kAllow);
         if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         TryLinks<InplaceMode::kAllow>(value, &out, std::index_sequence_for<Callbacks...>{});
         return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
       } else {
         if (TryLinks<InplaceMode::kAllow>(value, &out, std::index_sequence_for<Callbacks...>{})) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         return ExpectedUnsafe::MoveToTVMFFIAny(
             this->DefaultMutateExpected(value, InplaceMode::kAllow));
       }
     }
   
     std::tuple<Callbacks...> callbacks_;
   };
   
   template <typename Parent, WalkOrder order>
   class StructuralMapDynEngine : public Parent {
    public:
     static_assert(std::is_base_of_v<StructuralMapEngineBase, Parent>,
                   "StructuralMap Parent must derive from StructuralMapEngineBase");
     StructuralMapDynEngine(Array<Tuple<int32_t, Function>> callbacks,
                            Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind)
         : Parent(VTable()),
           callbacks_(std::move(callbacks)),
           callbacks_with_def_region_kind_(std::move(callbacks_with_def_region_kind)) {}
   
    private:
     using ExpectedUnsafe = details::ExpectedUnsafe;
     using AnyUnsafe = details::AnyUnsafe;
   
     static const StructuralMutatorVTable* VTable() {
       static const StructuralMutatorVTable vtable{
           &StructuralMapDynEngine::DispatchMutate,
           &StructuralMapDynEngine::DispatchMaybeInplaceMutate,
           &StructuralMapDynEngine::DispatchVarRemapGet,
           &StructuralMapDynEngine::DispatchVarRemapSet,
       };
       return &vtable;
     }
   
     static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                 AnyView value) noexcept {
       return static_cast<StructuralMapDynEngine*>(mutator)->MaybeInplaceMutateImplRaw(value);
     }
   
     static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
       return static_cast<StructuralMapDynEngine*>(mutator)->MutateImplRaw(value);
     }
   
     Optional<Function> FindLink(int32_t type_index, bool* with_kind) const noexcept {
       for (const Tuple<int32_t, Function>& entry : callbacks_) {
         if (details::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
           *with_kind = false;
           return entry.get<1>();
         }
       }
       for (const Tuple<int32_t, Function>& entry : callbacks_with_def_region_kind_) {
         if (details::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
           *with_kind = true;
           return entry.get<1>();
         }
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static Expected<Any> InvokeLink(const Function& fn, bool with_kind, AnyView target,
                                                    TVMFFIDefRegionKind kind) noexcept {
       return with_kind ? fn.CallExpected<Any>(target, kind) : fn.CallExpected<Any>(target);
     }
   
     template <InplaceMode kInplaceMode>
     TVM_FFI_INLINE bool TryLink(AnyView value, Expected<Any>* out) noexcept {
       if constexpr (order == WalkOrder::kPreOrder) {
         bool with_kind = false;
         Optional<Function> matched = FindLink(value.type_index(), &with_kind);
         if (!matched.has_value()) return false;
         // Pre-order: the callback rewrites this node first, then descent runs over what it made.
         Expected<Any> callback_result =
             InvokeLink(*matched, with_kind, value, this->def_region_kind());
         if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
           this->UpdateVisitErrorContext(callback_result, value);
           *out = std::move(callback_result);
           return true;
         }
         Any mapped_value = std::move(ExpectedUnsafe::GetData(callback_result));
         const AnyView descent_view =
             mapped_value.type_index() == TypeIndex::kTVMFFIUnchanged ? value : AnyView(mapped_value);
         *out = [&]() -> Expected<Any> {
           if constexpr (kInplaceMode == InplaceMode::kAllow) {
             if (descent_view.same_as(value)) {
               return this->DefaultMutateExpected(value, InplaceMode::kAllow);
             }
             const Object* mapped_obj = descent_view.as<Object>();
             InplaceMode inplace_mode = mapped_obj != nullptr && mapped_obj->unique()
                                            ? InplaceMode::kAllow
                                            : InplaceMode::kDisallow;
             return this->DefaultMutateExpected(descent_view, inplace_mode);
           } else {
             return this->DefaultMutateExpected(descent_view, InplaceMode::kDisallow);
           }
         }();
         if (TVM_FFI_PREDICT_FALSE(out->is_err())) return true;
         if (ExpectedUnsafe::GetData(*out).type_index() == TypeIndex::kTVMFFIUnchanged) {
           *out = std::move(mapped_value);
         }
         return true;
       } else {
         // Post-order descent is performed once by the engine entry before link selection.
         // See the typed engine: a borrowed view of the original, never an owning copy of it.
         const Any& descended_value = ExpectedUnsafe::GetData(*out);
         const AnyView mapped_view = descended_value.type_index() == TypeIndex::kTVMFFIUnchanged
                                         ? value
                                         : AnyView(descended_value);
         bool with_kind = false;
         Optional<Function> matched = FindLink(mapped_view.type_index(), &with_kind);
         if (!matched.has_value()) return false;
         // WithDefRegionKind restores its state through RAII, so this late read is equivalent to
         // the typed engine's invocation-time read even after recursive descent.
         *out = InvokeLink(*matched, with_kind, mapped_view, this->def_region_kind());
         if (TVM_FFI_PREDICT_FALSE(out->is_err())) {
           this->UpdateVisitErrorContext(*out, mapped_view);
           return true;
         }
         return true;
       }
     }
   
     TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
       Expected<Any> out{Any()};
       if constexpr (order == WalkOrder::kPostOrder) {
         out = this->DefaultMutateExpected(value, InplaceMode::kDisallow);
         if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         TryLink<InplaceMode::kDisallow>(value, &out);
         return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
       } else {
         if (TryLink<InplaceMode::kDisallow>(value, &out)) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         return ExpectedUnsafe::MoveToTVMFFIAny(
             this->DefaultMutateExpected(value, InplaceMode::kDisallow));
       }
     }
   
     TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
       Expected<Any> out{Any()};
       if constexpr (order == WalkOrder::kPostOrder) {
         out = this->DefaultMutateExpected(value, InplaceMode::kAllow);
         if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         TryLink<InplaceMode::kAllow>(value, &out);
         return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
       } else {
         if (TryLink<InplaceMode::kAllow>(value, &out)) {
           return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
         }
         return ExpectedUnsafe::MoveToTVMFFIAny(
             this->DefaultMutateExpected(value, InplaceMode::kAllow));
       }
     }
   
     Array<Tuple<int32_t, Function>> callbacks_;
     Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind_;
   };
   
   template <typename Parent, typename... Callbacks>
   class StructuralMutateEngine : public Parent {
    public:
     static_assert(std::is_base_of_v<StructuralMapEngineBase, Parent>,
                   "StructuralMutate Parent must derive from StructuralMapEngineBase");
   
     explicit StructuralMutateEngine(Callbacks... callbacks)
         : Parent(VTable()), callbacks_(std::move(callbacks)...) {}
   
    private:
     static const StructuralMutatorVTable* VTable() {
       static const StructuralMutatorVTable vtable{
           &StructuralMutateEngine::DispatchMutate,
           &StructuralMutateEngine::DispatchMaybeInplaceMutate,
           &StructuralMutateEngine::DispatchVarRemapGet,
           &StructuralMutateEngine::DispatchVarRemapSet,
       };
       return &vtable;
     }
   
     static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
       auto* self = static_cast<StructuralMutateEngine*>(mutator);
       if constexpr (sizeof...(Callbacks) == 1) {
         return self->template MutateSingleCallbackRaw<InplaceMode::kDisallow>(value);
       } else {
         return self->MutateImplRaw(value);
       }
     }
   
     static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                 AnyView value) noexcept {
       auto* self = static_cast<StructuralMutateEngine*>(mutator);
       if constexpr (sizeof...(Callbacks) == 1) {
         return self->template MutateSingleCallbackRaw<InplaceMode::kAllow>(value);
       } else {
         return self->MaybeInplaceMutateImplRaw(value);
       }
     }
   
     TVMFFIAny MutateImplRaw(AnyView value) noexcept {
       Expected<Any> result{Any()};
       if (DispatchCallbacks(value, InplaceMode::kDisallow, &result)) {
         if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
           // Keep callback-boundary context in addition to the default-descent
           // context: a callback may return a rebuilt value, so the two nodes can differ.
           Parent::UpdateVisitErrorContext(result, value);
         }
         return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
       }
       return details::ExpectedUnsafe::MoveToTVMFFIAny(
           Parent::DefaultMutateExpected(value, InplaceMode::kDisallow));
     }
   
     TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
       Expected<Any> result{Any()};
       if (DispatchCallbacks(value, InplaceMode::kAllow, &result)) {
         if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
           // Keep callback-boundary context in addition to the default-descent
           // context: a callback may return a rebuilt value, so the two nodes can differ.
           Parent::UpdateVisitErrorContext(result, value);
         }
         return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
       }
       return details::ExpectedUnsafe::MoveToTVMFFIAny(
           Parent::DefaultMutateExpected(value, InplaceMode::kAllow));
     }
   
     template <typename Callback, typename Matched>
     TVM_FFI_INLINE Expected<Any> InvokeCallback(Callback& callback, Matched&& matched,
                                                 InplaceMode inplace_mode) noexcept {
       using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
       static_assert(FuncInfo::num_args == 2 || FuncInfo::num_args == 3,
                     "StructuralMutate callback must take (value, mutator) or "
                     "(value, mutator, inplace_mode)");
       using SecondArg = std::decay_t<std::tuple_element_t<1, typename FuncInfo::ArgType>>;
       using Second = std::remove_pointer_t<SecondArg>;
       static_assert(std::is_same_v<Second, typename Parent::MutatorObjType>,
                     "second StructuralMutate callback argument must be exactly "
                     "Parent::MutatorObjType*");
       if constexpr (FuncInfo::num_args == 3) {
         using ThirdArg = std::decay_t<std::tuple_element_t<2, typename FuncInfo::ArgType>>;
         static_assert(std::is_same_v<ThirdArg, InplaceMode>,
                       "third StructuralMutate callback argument must be InplaceMode");
       }
       auto* mutator = static_cast<typename Parent::MutatorObjType*>(this);
       try {
         if constexpr (FuncInfo::num_args == 3) {
           return callback(std::forward<Matched>(matched), mutator, inplace_mode);
         } else {
           return callback(std::forward<Matched>(matched), mutator);
         }
       } catch (Error& err) {
         return Unexpected(std::move(err));
       }
     }
   
     TVM_FFI_COLD_CODE TVMFFIAny AnnotateCallbackErrorRaw(TVMFFIAny raw, AnyView value) noexcept {
       Expected<Any> result = details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(raw);
       Parent::UpdateVisitErrorContext(result, value);
       return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
     }
   
     // One callback forwards its result directly, without a callback-chain envelope.
     // AnyView/Any always match; only a typed miss needs the Parent's default descent.
     template <InplaceMode kInplaceMode>
     TVM_FFI_INLINE TVMFFIAny MutateSingleCallbackRaw(AnyView value) noexcept {
       auto& callback = std::get<0>(callbacks_);
       using FuncInfo = details::FunctionInfo<std::decay_t<decltype(callback)>>;
       using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
       using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
       TVMFFIAny result;
       if constexpr (std::is_same_v<TSub, AnyView>) {
         result =
             details::ExpectedUnsafe::MoveToTVMFFIAny(InvokeCallback(callback, value, kInplaceMode));
       } else if constexpr (std::is_same_v<TSub, Any>) {
         result = details::ExpectedUnsafe::MoveToTVMFFIAny(
             InvokeCallback(callback, Any(value), kInplaceMode));
       } else if (auto matched = value.template as<TSub>()) {
         result = details::ExpectedUnsafe::MoveToTVMFFIAny(
             InvokeCallback(callback, *std::move(matched), kInplaceMode));
       } else {
         if constexpr (kInplaceMode == InplaceMode::kAllow) {
           return details::ExpectedUnsafe::MoveToTVMFFIAny(
               Parent::DefaultMutateExpected(value, InplaceMode::kAllow));
         } else {
           return details::ExpectedUnsafe::MoveToTVMFFIAny(
               Parent::DefaultMutateExpected(value, InplaceMode::kDisallow));
         }
       }
       // Release any owning match before naming the callback boundary, as in the general path.
       if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
         return AnnotateCallbackErrorRaw(result, value);
       }
       return result;
     }
   
     template <typename Callback>
     TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value, InplaceMode inplace_mode,
                                 Expected<Any>* out) noexcept {
       using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
       using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
       using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
       if constexpr (std::is_same_v<TSub, AnyView>) {
         *out = InvokeCallback(callback, value, inplace_mode);
         return true;
       } else if constexpr (std::is_same_v<TSub, Any>) {
         *out = InvokeCallback(callback, Any(value), inplace_mode);
         return true;
       } else if (auto matched = value.template as<TSub>()) {
         *out = InvokeCallback(callback, *std::move(matched), inplace_mode);
         return true;
       }
       return false;
     }
   
     template <size_t... Is>
     TVM_FFI_INLINE bool TryLinks(AnyView value, InplaceMode inplace_mode, Expected<Any>* out,
                                  std::index_sequence<Is...>) noexcept {
       return (TryLink(std::get<Is>(callbacks_), value, inplace_mode, out) || ...);
     }
   
     TVM_FFI_INLINE bool DispatchCallbacks(AnyView value, InplaceMode inplace_mode,
                                           Expected<Any>* out) noexcept {
       return TryLinks(value, inplace_mode, out, std::index_sequence_for<Callbacks...>{});
     }
   
     std::tuple<Callbacks...> callbacks_;
   };
   
   template <WalkOrder order, typename... Callbacks>
   // The owning parameter makes caller ownership visible to the uniqueness check.
   Expected<Any> StructuralMapExpected(
       Any root, Callbacks&&... callbacks) noexcept {  // NOLINT(performance-unnecessary-value-param)
     static_assert(sizeof...(Callbacks) != 0, "StructuralMap requires at least one callback");
     using Mutator = StructuralMapEngine<StructuralMapEngineBase, order, std::decay_t<Callbacks>...>;
     StructuralMutator mutator(make_object<Mutator>(std::forward<Callbacks>(callbacks)...));
     auto result = mutator->MutateExpected(root, InplaceMode::kAllow);
     if (TVM_FFI_PREDICT_FALSE(result.is_err())) return Unexpected(std::move(result).error());
     UnchangedOr<Any> mapped = details::AnyUnsafe::MoveFromAnyAfterCheck<UnchangedOr<Any>>(
         std::move(details::ExpectedUnsafe::GetData(result)));
     return std::move(mapped).ValueOrUnchanged(std::move(root));
   }
   
   template <WalkOrder order, typename... Callbacks>
   // The owning parameter makes caller ownership visible to the uniqueness check.
   Any StructuralMap(Any root,
                     Callbacks&&... callbacks) {  // NOLINT(performance-unnecessary-value-param)
     return StructuralMapExpected<order>(std::move(root), std::forward<Callbacks>(callbacks)...)
         .value();
   }
   
   template <typename... Callbacks>
   // The owning parameter makes caller ownership visible to the uniqueness check.
   Expected<Any> StructuralMutateExpected(
       Any root, Callbacks&&... callbacks) noexcept {  // NOLINT(performance-unnecessary-value-param)
     static_assert(sizeof...(Callbacks) != 0, "StructuralMutate requires at least one callback");
     using Mutator = StructuralMutateEngine<StructuralMapEngineBase, std::decay_t<Callbacks>...>;
     StructuralMutator mutator(make_object<Mutator>(std::forward<Callbacks>(callbacks)...));
     auto result = mutator->MutateExpected(root, InplaceMode::kAllow);
     if (TVM_FFI_PREDICT_FALSE(result.is_err())) return Unexpected(std::move(result).error());
     UnchangedOr<Any> mapped = details::AnyUnsafe::MoveFromAnyAfterCheck<UnchangedOr<Any>>(
         std::move(details::ExpectedUnsafe::GetData(result)));
     return std::move(mapped).ValueOrUnchanged(std::move(root));
   }
   
   template <typename... Callbacks>
   // The owning parameter makes caller ownership visible to the uniqueness check.
   Any StructuralMutate(Any root,
                        Callbacks&&... callbacks) {  // NOLINT(performance-unnecessary-value-param)
     return StructuralMutateExpected(std::move(root), std::forward<Callbacks>(callbacks)...).value();
   }
   
   template <typename T>
   inline constexpr bool use_default_type_traits_v<UnchangedOr<T>> = false;
   
   template <typename T>
   struct TypeTraits<UnchangedOr<T>> : public TypeTraitsBase {
     TVM_FFI_INLINE static void CopyToAnyView(const UnchangedOr<T>& src, TVMFFIAny* result) {
       *result = AnyView(src.data_).CopyToTVMFFIAny();
     }
   
     TVM_FFI_INLINE static void MoveToAny(UnchangedOr<T> src, TVMFFIAny* result) {
       *result = details::UnchangedOrUnsafe::MoveToTVMFFIAny(std::move(src));
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if constexpr (std::is_same_v<T, Any>) {
         return src->type_index != TypeIndex::kTVMFFIError;
       } else {
         return src->type_index == TypeIndex::kTVMFFIUnchanged || TypeTraits<T>::CheckAnyStrict(src);
       }
     }
   
     TVM_FFI_INLINE static UnchangedOr<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFIUnchanged) return Unchanged();
       if constexpr (std::is_same_v<T, Any>) {
         return UnchangedOr<T>(Any(AnyView::CopyFromTVMFFIAny(*src)));
       } else {
         return UnchangedOr<T>(TypeTraits<T>::CopyFromAnyViewAfterCheck(src));
       }
     }
   
     TVM_FFI_INLINE static UnchangedOr<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
       return UnchangedOr<T>(typename UnchangedOr<T>::UnsafeInit{},
                             details::AnyUnsafe::MoveTVMFFIAnyToAny(src));
     }
     TVM_FFI_INLINE static std::string TypeStr() {
       return "UnchangedOr<" + details::Type2Str<T>::v() + ">";
     }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":"UnchangedOr","args":[)" + details::TypeSchema<T>::v() + "]}";
     }
   };
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
