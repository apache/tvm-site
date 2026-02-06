
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
   
   template <typename T>
   class Expected {
    public:
     static_assert(!std::is_same_v<T, Error>, "Expected<Error> is not allowed. Use Error directly.");
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(T value) : data_(Any(std::move(value))) {}
   
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(Error error) : data_(Any(std::move(error))) {}
   
     template <typename E, typename = std::enable_if_t<std::is_base_of_v<Error, std::remove_cv_t<E>>>>
     // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
     Expected(Unexpected<E> unexpected) : data_(Any(std::move(unexpected).error())) {}
   
     TVM_FFI_INLINE bool is_ok() const { return !data_.as<Error>().has_value(); }
   
     TVM_FFI_INLINE bool is_err() const { return !is_ok(); }
   
     TVM_FFI_INLINE bool has_value() const { return is_ok(); }
   
     TVM_FFI_INLINE T value() const& {
       if (is_err()) throw data_.cast<Error>();
       return data_.cast<T>();
     }
     TVM_FFI_INLINE T value() && {
       if (is_err()) throw std::move(data_).template cast<Error>();
       return std::move(data_).template cast<T>();
     }
   
     TVM_FFI_INLINE Error error() const& {
       if (!is_err()) TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
       return data_.cast<Error>();
     }
     TVM_FFI_INLINE Error error() && {
       if (!is_err()) TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
       return std::move(data_).template cast<Error>();
     }
   
     template <typename U = std::remove_cv_t<T>>
     TVM_FFI_INLINE T value_or(U&& default_value) const {
       if (is_ok()) {
         return data_.cast<T>();
       }
       return T(std::forward<U>(default_value));
     }
   
    private:
     Any data_;  // Holds either T or Error
   };
   
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
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXPECTED_H_
