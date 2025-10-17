
.. _program_listing_file_tvm_ffi_function_details.h:

Program Listing for File function_details.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_function_details.h>` (``tvm/ffi/function_details.h``)

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
   #ifndef TVM_FFI_FUNCTION_DETAILS_H_
   #define TVM_FFI_FUNCTION_DETAILS_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/base_details.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/error.h>
   
   #include <string>
   #include <tuple>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   namespace details {
   
   template <typename ArgType>
   struct Arg2Str {
     template <size_t i>
     TVM_FFI_INLINE static void Apply(std::ostream& os) {
       using Arg = std::tuple_element_t<i, ArgType>;
       if constexpr (i != 0) {
         os << ", ";
       }
       os << i << ": " << Type2Str<Arg>::v();
     }
     template <size_t... I>
     TVM_FFI_INLINE static void Run(std::ostream& os, std::index_sequence<I...>) {
       using TExpander = int[];
       (void)TExpander{0, (Apply<I>(os), 0)...};
     }
   };
   
   template <typename T>
   static constexpr bool ArgSupported =
       (std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, Any> ||
        std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, AnyView> ||
        TypeTraitsNoCR<T>::convert_enabled);
   
   // NOTE: return type can only support non-reference managed returns
   template <typename T>
   static constexpr bool RetSupported =
       (std::is_same_v<T, Any> || std::is_void_v<T> || TypeTraits<T>::convert_enabled);
   
   template <typename R, typename... Args>
   struct FuncFunctorImpl {
     using FType = R(Args...);
     using ArgType = std::tuple<Args...>;
     using RetType = R;
     static constexpr size_t num_args = sizeof...(Args);
     // MSVC is not that friendly to in-template nested bool evaluation
   #ifndef _MSC_VER
     static constexpr bool unpacked_supported = (ArgSupported<Args> && ...) && (RetSupported<R>);
   #endif
     TVM_FFI_INLINE static std::string Sig() {
       using IdxSeq = std::make_index_sequence<sizeof...(Args)>;
       std::ostringstream ss;
       ss << "(";
       Arg2Str<std::tuple<Args...>>::Run(ss, IdxSeq{});
       ss << ") -> " << Type2Str<R>::v();
       return ss.str();
     }
     TVM_FFI_INLINE static std::string TypeSchema() {
       std::ostringstream oss;
       oss << R"({"type":")" << StaticTypeKey::kTVMFFIFunction << R"(","args":[)";
       oss << details::TypeSchema<R>::v();
       ((oss << "," << details::TypeSchema<Args>::v()), ...);
       oss << "]}";
       return oss.str();
     }
   };
   
   template <typename T>
   struct FunctionInfoHelper;
   
   template <typename T, typename R, typename... Args>
   struct FunctionInfoHelper<R (T::*)(Args...)> : FuncFunctorImpl<R, Args...> {};
   template <typename T, typename R, typename... Args>
   struct FunctionInfoHelper<R (T::*)(Args...) const> : FuncFunctorImpl<R, Args...> {};
   
   template <typename T, typename = void>
   struct FunctionInfo : FunctionInfoHelper<decltype(&T::operator())> {};
   template <typename R, typename... Args>
   struct FunctionInfo<R(Args...), void> : FuncFunctorImpl<R, Args...> {};
   template <typename R, typename... Args>
   struct FunctionInfo<R (*)(Args...), void> : FuncFunctorImpl<R, Args...> {};
   // Support pointer-to-member functions used in reflection (e.g. &Class::method)
   template <typename Class, typename R, typename... Args>
   struct FunctionInfo<R (Class::*)(Args...), std::enable_if_t<std::is_base_of_v<Object, Class>>>
       : FuncFunctorImpl<R, Class*, Args...> {};
   template <typename Class, typename R, typename... Args>
   struct FunctionInfo<R (Class::*)(Args...) const, std::enable_if_t<std::is_base_of_v<Object, Class>>>
       : FuncFunctorImpl<R, const Class*, Args...> {};
   
   template <typename Class, typename R, typename... Args>
   struct FunctionInfo<R (Class::*)(Args...), std::enable_if_t<std::is_base_of_v<ObjectRef, Class>>>
       : FuncFunctorImpl<R, Class, Args...> {};
   template <typename Class, typename R, typename... Args>
   struct FunctionInfo<R (Class::*)(Args...) const,
                       std::enable_if_t<std::is_base_of_v<ObjectRef, Class>>>
       : FuncFunctorImpl<R, const Class, Args...> {};
   
   using FGetFuncSignature = std::string (*)();
   
   class ArgValueWithContext {
    public:
     TVM_FFI_INLINE ArgValueWithContext(const AnyView* args, int32_t arg_index,
                                        const std::string* optional_name, FGetFuncSignature f_sig)
         : args_(args), arg_index_(arg_index), optional_name_(optional_name), f_sig_(f_sig) {}
   
     template <typename Type>
     TVM_FFI_INLINE operator Type() {  // NOLINT(google-explicit-constructor)
       using TypeWithoutCR = std::remove_const_t<std::remove_reference_t<Type>>;
   
       if constexpr (std::is_same_v<TypeWithoutCR, AnyView>) {
         return args_[arg_index_];
       } else if constexpr (std::is_same_v<TypeWithoutCR, Any>) {
         return Any(args_[arg_index_]);
       } else {
         std::optional<TypeWithoutCR> opt = args_[arg_index_].try_cast<TypeWithoutCR>();
         if (!opt.has_value()) {
           TVMFFIAny any_data = args_[arg_index_].CopyToTVMFFIAny();
           TVM_FFI_THROW(TypeError) << "Mismatched type on argument #" << arg_index_
                                    << " when calling: `"
                                    << (optional_name_ == nullptr ? "" : *optional_name_)
                                    << (f_sig_ == nullptr ? "" : (*f_sig_)()) << "`. Expected `"
                                    << Type2Str<TypeWithoutCR>::v() << "` but got `"
                                    << TypeTraits<TypeWithoutCR>::GetMismatchTypeInfo(&any_data)
                                    << '`';
         }
         return *std::move(opt);
       }
     }
   
    private:
     const AnyView* args_;
     int32_t arg_index_;
     const std::string* optional_name_;
     FGetFuncSignature f_sig_;
   };
   
   template <typename R, std::size_t... Is, typename F>
   TVM_FFI_INLINE void unpack_call(std::index_sequence<Is...>, const std::string* optional_name,
                                   const F& f, [[maybe_unused]] const AnyView* args,
                                   [[maybe_unused]] int32_t num_args, [[maybe_unused]] Any* rv) {
     using FuncInfo = FunctionInfo<F>;
     FGetFuncSignature f_sig = FuncInfo::Sig;
   
     // somehow MSVC does not support the static constexpr member in this case, function is fine
   #ifndef _MSC_VER
     static_assert(FuncInfo::unpacked_supported, "The function signature do not support unpacked");
   #endif
     constexpr size_t nargs = sizeof...(Is);
     if (nargs != num_args) {
       TVM_FFI_THROW(TypeError) << "Mismatched number of arguments when calling: `"
                                << (optional_name == nullptr ? "" : *optional_name)
                                << (f_sig == nullptr ? "" : (*f_sig)()) << "`. Expected " << nargs
                                << " but got " << num_args << " arguments";
     }
     // use index sequence to do recursive-less unpacking
     if constexpr (std::is_same_v<R, void>) {
       f(ArgValueWithContext(args, Is, optional_name, f_sig)...);
     } else {
       *rv = R(f(ArgValueWithContext(args, Is, optional_name, f_sig)...));
     }
   }
   
   TVM_FFI_INLINE static Error MoveFromSafeCallRaised() {
     TVMFFIObjectHandle handle;
     TVMFFIErrorMoveFromRaised(&handle);
     // handle is owned by caller
     return details::ObjectUnsafe::ObjectRefFromObjectPtr<Error>(
         details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<TVMFFIObject*>(handle)));
   }
   
   TVM_FFI_INLINE static void SetSafeCallRaised(const Error& error) {
     TVMFFIErrorSetRaised(details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(error));
   }
   
   template <typename T>
   struct TypeSchemaImpl {
     static std::string v() {
       using U = std::remove_const_t<std::remove_reference_t<T>>;
       return TypeTraits<U>::TypeSchema();
     }
   };
   
   template <>
   struct TypeSchemaImpl<void> {
     static std::string v() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFINone) + R"("})";
     }
   };
   
   template <>
   struct TypeSchemaImpl<Any> {
     static std::string v() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIAny) + R"("})";
     }
   };
   
   template <>
   struct TypeSchemaImpl<AnyView> {
     static std::string v() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIAny) + R"("})";
     }
   };
   
   }  // namespace details
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_FUNCTION_DETAILS_H_
