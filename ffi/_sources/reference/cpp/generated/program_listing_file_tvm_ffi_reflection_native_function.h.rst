
.. _program_listing_file_tvm_ffi_reflection_native_function.h:

Program Listing for File native_function.h
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_reflection_native_function.h>` (``tvm/ffi/reflection/native_function.h``)

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
   #ifndef TVM_FFI_REFLECTION_NATIVE_FUNCTION_H_
   #define TVM_FFI_REFLECTION_NATIVE_FUNCTION_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/expected.h>
   #include <tvm/ffi/function.h>
   
   #include <exception>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   namespace reflection {
   
   template <typename Signature>
   class NativeFunctionView;
   
   template <typename R, typename... Args>
   class NativeFunctionView<R(Args...)> {
    public:
     static_assert(((std::is_lvalue_reference_v<Args> ||
                     (std::is_standard_layout_v<Args> && std::is_trivially_copyable_v<Args>)) &&
                    ...),
                   "NativeFunctionView parameters require lvalue references or standard-layout, "
                   "trivially-copyable values");
     static_assert((ffi::detail::ArgSupported<Args> && ...) && ffi::detail::RetSupported<R>,
                   "NativeFunctionView signature must be supported by TypedFunction");
     using ABIType = TVMFFIAny (*)(Args...) noexcept;
   
     NativeFunctionView() = delete;
     NativeFunctionView(std::nullptr_t) = delete;
     template <typename PackedR,
               std::enable_if_t<std::is_same_v<PackedR, R> || std::is_same_v<PackedR, Expected<R>>,
                                int> = 0>
     NativeFunctionView(const TypedFunction<PackedR(Args...)>& packed)  // NOLINT(*)
         : data_(AnyView(packed.packed()).CopyToTVMFFIAny()) {
       TVM_FFI_CHECK(packed != nullptr, ValueError)
           << "NativeFunctionView requires a non-null typed function";
     }
     template <
         typename PackedR,
         std::enable_if_t<std::is_same_v<PackedR, R> || std::is_same_v<PackedR, Expected<R>>, int> = 0>
     NativeFunctionView(const TypedFunction<PackedR(Args...)>&&) = delete;
   
     template <auto Fn>
     static NativeFunctionView FromNative() {
       static_assert(Fn != nullptr, "NativeFunctionView requires a non-null native function");
       TVMFFIAny data;
       data.type_index = TypeIndex::kTVMFFIOpaquePtr;
       data.zero_padding = 0;
       TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&data);
       data.v_ptr = reinterpret_cast<void*>(&NativeABIAdapter<Fn>);
       return NativeFunctionView(UnsafeInit{}, data);
     }
   
     TVM_FFI_INLINE Expected<R> CallExpected(Args... args) const noexcept {
       return ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<R>(CallABI(std::forward<Args>(args)...));
     }
   
     TVM_FFI_INLINE R operator()(Args... args) const {
       return CallExpected(std::forward<Args>(args)...).value();
     }
   
    private:
     explicit NativeFunctionView(UnsafeInit, TVMFFIAny data) : data_(data) {}
   
     template <auto Fn>
     static TVMFFIAny NativeABIAdapter(Args... args) noexcept {
       return NativeABIFuncPtr<Fn>(std::forward<Args>(args)...);
     }
   
     template <Expected<R> (*Fn)(Args...) noexcept>
     TVM_FFI_INLINE static TVMFFIAny NativeABIFuncPtr(Args... args) noexcept {
       return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Fn(std::forward<Args>(args)...));
     }
   
     template <R (*Fn)(Args...)>
     TVM_FFI_INLINE static TVMFFIAny NativeABIFuncPtr(Args... args) noexcept {
       try {
         if constexpr (std::is_void_v<R>) {
           Fn(std::forward<Args>(args)...);
           return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<void>());
         } else {
           return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
               Expected<R>(Fn(std::forward<Args>(args)...)));
         }
       } catch (const ffi::Error& error) {
         return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(Unexpected(error)));
       } catch (const std::exception& error) {
         return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<R>(Unexpected(ffi::Error("InternalError", error.what(), ""))));
       }
     }
   
     TVM_FFI_INLINE TVMFFIAny CallABI(Args... args) const noexcept {
       if (TVM_FFI_PREDICT_TRUE(data_.type_index == TypeIndex::kTVMFFIOpaquePtr)) {
         return reinterpret_cast<ABIType>(data_.v_obj)(std::forward<Args>(args)...);
       }
       return CallABITail(std::forward<Args>(args)...);
     }
   
     TVM_FFI_COLD_CODE TVMFFIAny CallABITail(Args... args) const noexcept {
       auto* cell = TVMFFIFunctionGetCellPtr(data_.v_obj);
       AnyView packed_args[sizeof...(Args) > 0 ? sizeof...(Args) : 1];
       PackedArgs::Fill(packed_args, std::forward<Args>(args)...);
       Any result;
       // The function cell follows the object header in the C ABI allocation.
       // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
       int code = cell->safe_call(data_.v_obj, reinterpret_cast<const TVMFFIAny*>(packed_args),
                                  sizeof...(Args), reinterpret_cast<TVMFFIAny*>(&result));
       if (TVM_FFI_PREDICT_FALSE(code != 0)) {
         return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
             Expected<R>(Unexpected(ffi::detail::MoveFromSafeCallRaised())));
       }
       if (result.type_index() == TypeIndex::kTVMFFIError) {
         return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(
             Unexpected(ffi::detail::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(result)))));
       }
       if constexpr (std::is_same_v<R, Any>) {
         return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(std::move(result)));
       } else if constexpr (std::is_void_v<R>) {
         if (result.type_index() == TypeIndex::kTVMFFINone) {
           return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<void>());
         }
       } else if (auto value = result.template try_cast<R>()) {
         return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(*std::move(value)));
       }
       return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
           Expected<R>(Unexpected(Error("TypeError", "Packed hook result type mismatch", ""))));
     }
   
     template <typename, typename>
     friend struct ::tvm::ffi::TypeTraits;
     template <typename>
     friend class NativeFunction;
   
     TVMFFIAny data_;
   };
   
   template <typename Signature>
   class NativeFunction;
   
   template <typename R, typename... Args>
   class NativeFunction<R(Args...)> {
    public:
     using View = NativeFunctionView<R(Args...)>;
   
     explicit NativeFunction(View view) : data_(AnyView(view)) {}
   
     template <typename PackedR,
               std::enable_if_t<std::is_same_v<PackedR, R> || std::is_same_v<PackedR, Expected<R>>,
                                int> = 0>
     NativeFunction(const TypedFunction<PackedR(Args...)>& packed)  // NOLINT(*)
         : NativeFunction(View(packed)) {}
   
     static NativeFunction From(View view) { return NativeFunction(view); }
   
     template <auto Fn>
     static NativeFunction FromNative() {
       return From(View::template FromNative<Fn>());
     }
   
     TVM_FFI_INLINE Expected<R> CallExpected(Args... args) const noexcept {
       return View(UnsafeInit{}, AnyView(data_).CopyToTVMFFIAny())
           .CallExpected(std::forward<Args>(args)...);
     }
   
     TVM_FFI_INLINE R operator()(Args... args) const {
       return CallExpected(std::forward<Args>(args)...).value();
     }
   
    private:
     explicit NativeFunction(UnsafeInit, Any data) : data_(std::move(data)) {}
   
     template <typename, typename>
     friend struct ::tvm::ffi::TypeTraits;
   
     Any data_;
   };
   
   }  // namespace reflection
   
   template <typename Signature>
   inline constexpr bool use_default_type_traits_v<reflection::NativeFunctionView<Signature>> = false;
   
   template <typename Signature>
   inline constexpr bool use_default_type_traits_v<reflection::NativeFunction<Signature>> = false;
   
   template <typename Signature>
   struct TypeTraits<reflection::NativeFunctionView<Signature>> : public TypeTraitsBase {
     using View = reflection::NativeFunctionView<Signature>;
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;
   
     static void CopyToAnyView(const View& src, TVMFFIAny* result) { *result = src.data_; }
     static void MoveToAny(View src, TVMFFIAny* result) {
       *result = detail::AnyUnsafe::MoveAnyToTVMFFIAny(Any(AnyView::CopyFromTVMFFIAny(src.data_)));
     }
     static bool CheckAnyStrict(const TVMFFIAny* src) {
       return (src->type_index == TypeIndex::kTVMFFIOpaquePtr && src->v_ptr != nullptr) ||
              src->type_index == TypeIndex::kTVMFFIFunction;
     }
     static View CopyFromAnyViewAfterCheck(const TVMFFIAny* src) { return View(UnsafeInit{}, *src); }
     static View MoveFromAnyAfterCheck(TVMFFIAny* src) { return CopyFromAnyViewAfterCheck(src); }
     static std::optional<View> TryCastFromAnyView(const TVMFFIAny* src) {
       if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
       return std::nullopt;
     }
     static std::string TypeStr() {
       return "Variant<" + TypeTraits<void*>::TypeStr() + ", " +
              TypeTraits<TypedFunction<Signature>>::TypeStr() + ">";
     }
     static std::string TypeSchema() {
       return R"({"type":"Variant","args":[)" + TypeTraits<void*>::TypeSchema() + "," +
              TypeTraits<TypedFunction<Signature>>::TypeSchema() + "]}";
     }
   };
   
   template <typename Signature>
   struct TypeTraits<reflection::NativeFunction<Signature>> : public TypeTraitsBase {
     using Owner = reflection::NativeFunction<Signature>;
     using View = typename Owner::View;
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;
   
     static void CopyToAnyView(const Owner& src, TVMFFIAny* result) {
       *result = AnyView(src.data_).CopyToTVMFFIAny();
     }
     static void MoveToAny(Owner src, TVMFFIAny* result) {
       *result = detail::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src.data_));
     }
     static bool CheckAnyStrict(const TVMFFIAny* src) { return TypeTraits<View>::CheckAnyStrict(src); }
     static Owner CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       return Owner(TypeTraits<View>::CopyFromAnyViewAfterCheck(src));
     }
     static Owner MoveFromAnyAfterCheck(TVMFFIAny* src) {
       return Owner(UnsafeInit{}, detail::AnyUnsafe::MoveTVMFFIAnyToAny(src));
     }
     static std::optional<Owner> TryCastFromAnyView(const TVMFFIAny* src) {
       if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
       return std::nullopt;
     }
     static std::string TypeStr() { return TypeTraits<View>::TypeStr(); }
     static std::string TypeSchema() { return TypeTraits<View>::TypeSchema(); }
   };
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_REFLECTION_NATIVE_FUNCTION_H_
