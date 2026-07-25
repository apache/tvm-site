
.. _program_listing_file_tvm_ffi_expected.h:

Program Listing for File expected.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_expected.h>` (``tvm/ffi/expected.h``)

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
   
   #ifndef TVM_FFI_EXPECTED_H_
   #define TVM_FFI_EXPECTED_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/error.h>
   
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   template <typename E = Error>
   class Unexpected {
     static_assert(std::is_base_of_v<Error, std::remove_cv_t<E>>,
                   "Unexpected<E> requires E to be Error or a subclass of Error.");
   
    public:
     explicit Unexpected(E error) : error_(std::move(error)) {}
   
     const E& error() const& noexcept { return error_; }
     E& error() & noexcept { return error_; }
     const E&& error() const&& noexcept { return std::move(error_); }
     E&& error() && noexcept { return std::move(error_); }
   
    private:
     E error_;
   };
   
   #ifndef TVM_FFI_DOXYGEN_MODE
   template <typename E>
   Unexpected(E) -> Unexpected<E>;
   #endif
   
   namespace details {
   
   struct ExpectedUnsafe;
   
   }  // namespace details
   
   template <typename T>
   class Expected {
    public:
     static_assert(
         !std::is_void_v<T>,
         "Expected with a cv-qualified void success type is not allowed. Use Expected<void>.");
     static_assert(!std::is_same_v<T, Error>, "Expected<Error> is not allowed. Use Error directly.");
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(T value) : data_(Any(std::move(value))) {}
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(Error error) : data_(Any(std::move(error))) {}
   
     template <typename E, typename = std::enable_if_t<std::is_base_of_v<Error, std::remove_cv_t<E>>>>
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(Unexpected<E> unexpected) : data_(Any(std::move(unexpected).error())) {}
   
     TVM_FFI_INLINE int32_t type_index() const noexcept { return data_.type_index(); }
   
     TVM_FFI_INLINE bool is_ok() const noexcept {
       return data_.type_index() != TypeIndex::kTVMFFIError;
     }
   
     TVM_FFI_INLINE bool is_err() const noexcept {
       return data_.type_index() == TypeIndex::kTVMFFIError;
     }
   
     TVM_FFI_INLINE bool has_value() const noexcept { return is_ok(); }
   
     TVM_FFI_INLINE T value() const& {
       if (TVM_FFI_PREDICT_TRUE(is_ok())) {
         return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data_);
       }
       throw details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
     }
   
     TVM_FFI_INLINE T value() && {
       if (TVM_FFI_PREDICT_TRUE(is_ok())) {
         return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
       }
       throw details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
     }
   
     TVM_FFI_INLINE Error error() const& {
       // No branch hint: error() is itself a cold path — callers only invoke it
       // after observing !is_ok(), so the branch direction here doesn't matter.
       if (is_ok()) {
         TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
     }
   
     TVM_FFI_INLINE Error error() && {
       // No branch hint: error() is itself a cold path — callers only invoke it
       // after observing !is_ok(), so the branch direction here doesn't matter.
       if (is_ok()) {
         TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
       }
       return details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
     }
   
     template <typename U = std::remove_cv_t<T>>
     TVM_FFI_INLINE T value_or(U&& default_value) const& {
       if (TVM_FFI_PREDICT_TRUE(is_ok())) {
         return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data_);
       }
       return T(std::forward<U>(default_value));
     }
   
     template <typename U = std::remove_cv_t<T>>
     TVM_FFI_INLINE T value_or(U&& default_value) && {
       if (TVM_FFI_PREDICT_TRUE(is_ok())) {
         return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
       }
       return T(std::forward<U>(default_value));
     }
   
    private:
     Expected() = default;
   
     friend struct details::ExpectedUnsafe;
   
     Any data_;  // Invariant: holds a T (type_index != kTVMFFIError) or an Error.
   };
   
   template <>
   class Expected<void> {
    public:
     Expected() = default;
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(Error error) : data_(Any(std::move(error))) {}
   
     template <typename E, typename = std::enable_if_t<std::is_base_of_v<Error, std::remove_cv_t<E>>>>
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(Unexpected<E> unexpected) : data_(Any(std::move(unexpected).error())) {}
   
     TVM_FFI_INLINE int32_t type_index() const noexcept { return data_.type_index(); }
   
     TVM_FFI_INLINE bool is_ok() const noexcept {
       return data_.type_index() != TypeIndex::kTVMFFIError;
     }
   
     TVM_FFI_INLINE bool is_err() const noexcept {
       return data_.type_index() == TypeIndex::kTVMFFIError;
     }
   
     TVM_FFI_INLINE bool has_value() const noexcept { return is_ok(); }
   
     TVM_FFI_INLINE void value() const& {
       if (TVM_FFI_PREDICT_FALSE(is_err())) {
         throw details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
       }
     }
   
     TVM_FFI_INLINE void value() && {
       if (TVM_FFI_PREDICT_FALSE(is_err())) {
         throw details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
       }
     }
   
     TVM_FFI_INLINE Error error() const& {
       if (is_ok()) {
         TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
     }
   
     TVM_FFI_INLINE Error error() && {
       if (is_ok()) {
         TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
       }
       return details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
     }
   
    private:
     friend struct details::ExpectedUnsafe;
   
     Any data_;  // Invariant: holds FFI None on success or an Error.
   };
   
   namespace details {
   
   struct ExpectedUnsafe {
     template <typename T>
     TVM_FFI_INLINE static Expected<T> MoveFromTVMFFIAny(TVMFFIAny raw) {
       Expected<T> result;
       result.data_ = AnyUnsafe::MoveTVMFFIAnyToAny(&raw);
       return result;
     }
   
     template <typename T>
     TVM_FFI_INLINE static TVMFFIAny MoveToTVMFFIAny(Expected<T>&& result) {
       return AnyUnsafe::MoveAnyToTVMFFIAny(std::move(result.data_));
     }
   
     template <typename T>
     TVM_FFI_INLINE static const Any& GetData(const Expected<T>& result) noexcept {
       return result.data_;
     }
   
     template <typename T, typename U>
     TVM_FFI_INLINE static T ValueAs(const Expected<U>& result) {
       if constexpr (std::is_void_v<T>) {
         static_assert(std::is_void_v<U>, "ExpectedUnsafe::ValueAs<void> requires an Expected<void>");
         result.value();
       } else {
         const Any& data = result.data_;
         if (TVM_FFI_PREDICT_TRUE(data.type_index() != TypeIndex::kTVMFFIError)) {
           return AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data);
         }
         throw AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data);
       }
     }
   };
   
   }  // namespace details
   
   // TypeTraits specialization for Expected<T>
   template <typename T>
   inline constexpr bool use_default_type_traits_v<Expected<T>> = false;
   
   template <typename T>
   struct TypeTraits<Expected<T>> : public TypeTraitsBase {
     TVM_FFI_INLINE static void CopyToAnyView(const Expected<T>& src, TVMFFIAny* result) {
       if (src.is_err()) {
         TypeTraits<Error>::CopyToAnyView(src.error(), result);
       } else {
         TypeTraits<T>::CopyToAnyView(src.value(), result);
       }
     }
   
     TVM_FFI_INLINE static void MoveToAny(Expected<T> src, TVMFFIAny* result) {
       if (src.is_err()) {
         TypeTraits<Error>::MoveToAny(std::move(src).error(), result);
       } else {
         TypeTraits<T>::MoveToAny(std::move(src).value(), result);
       }
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return TypeTraits<T>::CheckAnyStrict(src) || TypeTraits<Error>::CheckAnyStrict(src);
     }
   
     TVM_FFI_INLINE static Expected<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       if (TypeTraits<T>::CheckAnyStrict(src)) {
         return TypeTraits<T>::CopyFromAnyViewAfterCheck(src);
       }
       return TypeTraits<Error>::CopyFromAnyViewAfterCheck(src);
     }
   
     TVM_FFI_INLINE static Expected<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
       if (TypeTraits<T>::CheckAnyStrict(src)) {
         return TypeTraits<T>::MoveFromAnyAfterCheck(src);
       }
       return TypeTraits<Error>::MoveFromAnyAfterCheck(src);
     }
   
     TVM_FFI_INLINE static std::optional<Expected<T>> TryCastFromAnyView(const TVMFFIAny* src) {
       if (auto opt = TypeTraits<T>::TryCastFromAnyView(src)) {
         return Expected<T>(*std::move(opt));
       }
       if (auto opt_err = TypeTraits<Error>::TryCastFromAnyView(src)) {
         return Expected<T>(*std::move(opt_err));
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() {
       return "Expected<" + TypeTraits<T>::TypeStr() + ">";
     }
   
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":"Expected","args":[)" + details::TypeSchema<T>::v() +
              R"(,{"type":"ffi.Error"}]})";
     }
   };
   
   template <>
   struct TypeTraits<Expected<void>> : public TypeTraitsBase {
     TVM_FFI_INLINE static void CopyToAnyView(const Expected<void>& src, TVMFFIAny* result) {
       if (src.is_err()) {
         TypeTraits<Error>::CopyToAnyView(src.error(), result);
       } else {
         TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
       }
     }
   
     TVM_FFI_INLINE static void MoveToAny(Expected<void> src, TVMFFIAny* result) {
       if (src.is_err()) {
         TypeTraits<Error>::MoveToAny(std::move(src).error(), result);
       } else {
         TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
       }
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return TypeTraits<std::nullptr_t>::CheckAnyStrict(src) ||
              TypeTraits<Error>::CheckAnyStrict(src);
     }
   
     TVM_FFI_INLINE static Expected<void> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       if (TypeTraits<std::nullptr_t>::CheckAnyStrict(src)) {
         return Expected<void>();
       }
       return TypeTraits<Error>::CopyFromAnyViewAfterCheck(src);
     }
   
     TVM_FFI_INLINE static Expected<void> MoveFromAnyAfterCheck(TVMFFIAny* src) {
       if (TypeTraits<std::nullptr_t>::CheckAnyStrict(src)) {
         return Expected<void>();
       }
       return TypeTraits<Error>::MoveFromAnyAfterCheck(src);
     }
   
     TVM_FFI_INLINE static std::optional<Expected<void>> TryCastFromAnyView(const TVMFFIAny* src) {
       if (TypeTraits<std::nullptr_t>::CheckAnyStrict(src)) {
         return Expected<void>();
       }
       if (auto opt_err = TypeTraits<Error>::TryCastFromAnyView(src)) {
         return Expected<void>(*std::move(opt_err));
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return "Expected<void>"; }
   
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":"Expected","args":[)" + TypeTraits<std::nullptr_t>::TypeSchema() +
              R"(,{"type":"ffi.Error"}]})";
     }
   };
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXPECTED_H_
