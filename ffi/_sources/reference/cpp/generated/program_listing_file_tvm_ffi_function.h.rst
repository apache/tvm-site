
.. _program_listing_file_tvm_ffi_function.h:

Program Listing for File function.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_function.h>` (``tvm/ffi/function.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     TVM_FFI_SAFE_CALL_BEGIN();
     // c++ code region here
     TVM_FFI_SAFE_CALL_END();
   }
   TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_arr, &type_index));
     return x + 1;
   }
   
   // Expose the function as "AddOne"
   TVM_FFI_DLL_EXPORT_TYPED_FUNC(AddOne, AddOne_);
   
   // Expose the function as "SubOne"
   TVM_FFI_DLL_EXPORT_TYPED_FUNC(SubOne, [](int x) {
     return x - 1;
   });
   
   // The following code will cause compilation error.
   // Because the same Function and ExportName
   // TVM_FFI_DLL_EXPORT_TYPED_FUNC(AddOne_, AddOne_);
   
   // The following code is OK, assuming the macro
   // is in a different namespace from xyz
   // TVM_FFI_DLL_EXPORT_TYPED_FUNC(AddOne_, xyz::AddOne_);
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
   #ifndef TVM_FFI_FUNCTION_H_
   #define TVM_FFI_FUNCTION_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/base_details.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/function_details.h>
   
   #include <functional>
   #include <string>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   namespace ffi {
   
   #define TVM_FFI_SAFE_CALL_BEGIN() \
     try {                           \
     (void)0
   
   #define TVM_FFI_SAFE_CALL_END()                                                                \
     return 0;                                                                                    \
     }                                                                                            \
     catch (const ::tvm::ffi::Error& err) {                                                       \
       ::tvm::ffi::details::SetSafeCallRaised(err);                                               \
       return -1;                                                                                 \
     }                                                                                            \
     catch (const ::tvm::ffi::EnvErrorAlreadySet&) {                                              \
       return -2;                                                                                 \
     }                                                                                            \
     catch (const std::exception& ex) {                                                           \
       ::tvm::ffi::details::SetSafeCallRaised(::tvm::ffi::Error("InternalError", ex.what(), "")); \
       return -1;                                                                                 \
     }                                                                                            \
     TVM_FFI_UNREACHABLE()
   
   #define TVM_FFI_CHECK_SAFE_CALL(func)                      \
     {                                                        \
       int ret_code = (func);                                 \
       if (ret_code != 0) {                                   \
         if (ret_code == -2) {                                \
           throw ::tvm::ffi::EnvErrorAlreadySet();            \
         }                                                    \
         throw ::tvm::ffi::details::MoveFromSafeCallRaised(); \
       }                                                      \
     }
   
   class FunctionObj : public Object, public TVMFFIFunctionCell {
    public:
     typedef void (*FCall)(const FunctionObj*, const AnyView*, int32_t, Any*);
     using TVMFFIFunctionCell::safe_call;
     FCall call;
     TVM_FFI_INLINE void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
       this->call(this, args, num_args, result);
     }
     static constexpr const uint32_t _type_index = TypeIndex::kTVMFFIFunction;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIFunction, FunctionObj, Object);
   
    protected:
     FunctionObj() {}
     // Implementing safe call style
     static int SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
       TVM_FFI_SAFE_CALL_BEGIN();
       TVM_FFI_ICHECK_LT(result->type_index, TypeIndex::kTVMFFIStaticObjectBegin);
       FunctionObj* self = static_cast<FunctionObj*>(func);
       self->call(self, reinterpret_cast<const AnyView*>(args), num_args,
                  reinterpret_cast<Any*>(result));
       TVM_FFI_SAFE_CALL_END();
     }
     friend class Function;
   };
   
   namespace details {
   template <typename TCallable>
   class FunctionObjImpl : public FunctionObj {
    public:
     using TStorage = typename std::remove_cv<typename std::remove_reference<TCallable>::type>::type;
     using TSelf = FunctionObjImpl<TCallable>;
     explicit FunctionObjImpl(TCallable callable) : callable_(callable) {
       this->safe_call = SafeCall;
       this->call = Call;
     }
   
    private:
     // implementation of call
     static void Call(const FunctionObj* func, const AnyView* args, int32_t num_args, Any* result) {
       (static_cast<const TSelf*>(func))->callable_(args, num_args, result);
     }
   
     mutable TStorage callable_;
   };
   
   template <typename Derived>
   struct RedirectCallToSafeCall {
     static void Call(const FunctionObj* func, const AnyView* args, int32_t num_args, Any* rv) {
       Derived* self = static_cast<Derived*>(const_cast<FunctionObj*>(func));
       TVM_FFI_CHECK_SAFE_CALL(self->RedirectSafeCall(reinterpret_cast<const TVMFFIAny*>(args),
                                                      num_args, reinterpret_cast<TVMFFIAny*>(rv)));
     }
   
     static int32_t SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
       Derived* self = reinterpret_cast<Derived*>(func);
       return self->RedirectSafeCall(args, num_args, rv);
     }
   };
   
   class ExternCFunctionObjImpl : public FunctionObj,
                                  public RedirectCallToSafeCall<ExternCFunctionObjImpl> {
    public:
     using RedirectCallToSafeCall<ExternCFunctionObjImpl>::SafeCall;
   
     ExternCFunctionObjImpl(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self))
         : self_(self), safe_call_(safe_call), deleter_(deleter) {
       this->call = RedirectCallToSafeCall<ExternCFunctionObjImpl>::Call;
       this->safe_call = RedirectCallToSafeCall<ExternCFunctionObjImpl>::SafeCall;
     }
   
     ~ExternCFunctionObjImpl() { deleter_(self_); }
   
     TVM_FFI_INLINE int32_t RedirectSafeCall(const TVMFFIAny* args, int32_t num_args,
                                             TVMFFIAny* rv) const {
       return safe_call_(self_, args, num_args, rv);
     }
   
    private:
     void* self_;
     TVMFFISafeCallType safe_call_;
     void (*deleter_)(void* self);
   };
   
   class ImportedFunctionObjImpl : public FunctionObj,
                                   public RedirectCallToSafeCall<ImportedFunctionObjImpl> {
    public:
     using RedirectCallToSafeCall<ImportedFunctionObjImpl>::SafeCall;
   
     explicit ImportedFunctionObjImpl(ObjectPtr<Object> data) : data_(data) {
       this->call = RedirectCallToSafeCall<ImportedFunctionObjImpl>::Call;
       this->safe_call = RedirectCallToSafeCall<ImportedFunctionObjImpl>::SafeCall;
     }
   
     TVM_FFI_INLINE int32_t RedirectSafeCall(const TVMFFIAny* args, int32_t num_args,
                                             TVMFFIAny* rv) const {
       FunctionObj* func = const_cast<FunctionObj*>(static_cast<const FunctionObj*>(data_.get()));
       return func->safe_call(func, args, num_args, rv);
     }
   
    private:
     ObjectPtr<Object> data_;
   };
   
   // Helper class to set packed arguments
   class PackedArgsSetter {
    public:
     explicit PackedArgsSetter(AnyView* args) : args_(args) {}
   
     // NOTE: setter needs to be very carefully designed
     // such that we do not have temp variable conversion(eg. convert from lvalue to rvalue)
     // that is why we need T&& and std::forward here
     template <typename T>
     TVM_FFI_INLINE void operator()(size_t i, T&& value) const {
       args_[i].operator=(std::forward<T>(value));
     }
   
    private:
     AnyView* args_;
   };
   }  // namespace details
   
   class PackedArgs {
    public:
     PackedArgs(const AnyView* data, int32_t size) : data_(data), size_(size) {}
   
     int size() const { return size_; }
   
     const AnyView* data() const { return data_; }
   
     PackedArgs Slice(int begin, int end = -1) const {
       if (end == -1) {
         end = size_;
       }
       return PackedArgs(data_ + begin, end - begin);
     }
   
     AnyView operator[](int i) const { return data_[i]; }
   
     template <typename... Args>
     TVM_FFI_INLINE static void Fill(AnyView* data, Args&&... args) {
       details::for_each(details::PackedArgsSetter(data), std::forward<Args>(args)...);
     }
   
    private:
     const AnyView* data_;
     int32_t size_;
   };
   
   class Function : public ObjectRef {
    public:
     Function(std::nullptr_t) : ObjectRef(nullptr) {}  // NOLINT(*)
     template <typename TCallable>
     explicit Function(TCallable packed_call) {
       *this = FromPacked(packed_call);
     }
     template <typename TCallable>
     static Function FromPacked(TCallable packed_call) {
       static_assert(
           std::is_convertible_v<TCallable, std::function<void(const AnyView*, int32_t, Any*)>> ||
               std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>,
           "tvm::ffi::Function::FromPacked requires input function signature to match packed func "
           "format");
       if constexpr (std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>) {
         auto wrapped_call = [packed_call](const AnyView* args, int32_t num_args,
                                           Any* rv) mutable -> void {
           PackedArgs args_pack(args, num_args);
           packed_call(args_pack, rv);
         };
         return FromPackedInternal(wrapped_call);
       } else {
         return FromPackedInternal(packed_call);
       }
     }
     static Function ImportFromExternDLL(Function other) {
       const FunctionObj* other_func = static_cast<const FunctionObj*>(other.get());
       // the other function comes from the same dll, no action needed
       if (other_func->safe_call == &(FunctionObj::SafeCall) ||
           other_func->safe_call == &(details::ImportedFunctionObjImpl::SafeCall) ||
           other_func->safe_call == &(details::ExternCFunctionObjImpl::SafeCall)) {
         return other;
       }
       // the other function coems from a different library
       Function func;
       func.data_ = make_object<details::ImportedFunctionObjImpl>(std::move(other.data_));
       return func;
     }
     static Function FromExternC(void* self, TVMFFISafeCallType safe_call,
                                 void (*deleter)(void* self)) {
       // the other function coems from a different library
       Function func;
       func.data_ = make_object<details::ExternCFunctionObjImpl>(self, safe_call, deleter);
       return func;
     }
     static std::optional<Function> GetGlobal(std::string_view name) {
       TVMFFIObjectHandle handle;
       TVMFFIByteArray name_arr{name.data(), name.size()};
       TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionGetGlobal(&name_arr, &handle));
       if (handle != nullptr) {
         return Function(
             details::ObjectUnsafe::ObjectPtrFromOwned<FunctionObj>(static_cast<Object*>(handle)));
       } else {
         return std::nullopt;
       }
     }
   
     static std::optional<Function> GetGlobal(const std::string& name) {
       return GetGlobal(std::string_view(name.data(), name.length()));
     }
   
     static std::optional<Function> GetGlobal(const String& name) {
       return GetGlobal(std::string_view(name.data(), name.length()));
     }
   
     static std::optional<Function> GetGlobal(const char* name) {
       return GetGlobal(std::string_view(name));
     }
     static Function GetGlobalRequired(std::string_view name) {
       std::optional<Function> res = GetGlobal(name);
       if (!res.has_value()) {
         TVM_FFI_THROW(ValueError) << "Function " << name << " not found";
       }
       return *res;
     }
   
     static Function GetGlobalRequired(const std::string& name) {
       return GetGlobalRequired(std::string_view(name.data(), name.length()));
     }
   
     static Function GetGlobalRequired(const String& name) {
       return GetGlobalRequired(std::string_view(name.data(), name.length()));
     }
   
     static Function GetGlobalRequired(const char* name) {
       return GetGlobalRequired(std::string_view(name));
     }
     static void SetGlobal(std::string_view name, Function func, bool override = false) {
       TVMFFIByteArray name_arr{name.data(), name.size()};
       TVM_FFI_CHECK_SAFE_CALL(
           TVMFFIFunctionSetGlobal(&name_arr, details::ObjectUnsafe::GetHeader(func.get()), override));
     }
     static std::vector<String> ListGlobalNames() {
       Function fname_functor =
           GetGlobalRequired("ffi.FunctionListGlobalNamesFunctor")().cast<Function>();
       std::vector<String> names;
       int len = fname_functor(-1).cast<int>();
       for (int i = 0; i < len; ++i) {
         names.push_back(fname_functor(i).cast<String>());
       }
       return names;
     }
     static void RemoveGlobal(const String& name) {
       static Function fremove = GetGlobalRequired("ffi.FunctionRemoveGlobal");
       fremove(name);
     }
     template <typename TCallable>
     static Function FromTyped(TCallable callable) {
       using FuncInfo = details::FunctionInfo<TCallable>;
       auto call_packed = [callable](const AnyView* args, int32_t num_args, Any* rv) mutable -> void {
         details::unpack_call<typename FuncInfo::RetType>(
             std::make_index_sequence<FuncInfo::num_args>{}, nullptr, callable, args, num_args, rv);
       };
       return FromPackedInternal(call_packed);
     }
     template <typename TCallable>
     static Function FromTyped(TCallable callable, std::string name) {
       using FuncInfo = details::FunctionInfo<TCallable>;
       auto call_packed = [callable, name](const AnyView* args, int32_t num_args,
                                           Any* rv) mutable -> void {
         details::unpack_call<typename FuncInfo::RetType>(
             std::make_index_sequence<FuncInfo::num_args>{}, &name, callable, args, num_args, rv);
       };
       return FromPackedInternal(call_packed);
     }
     template <typename... Args>
     TVM_FFI_INLINE Any operator()(Args&&... args) const {
       const int kNumArgs = sizeof...(Args);
       const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
       AnyView args_pack[kArraySize];
       PackedArgs::Fill(args_pack, std::forward<Args>(args)...);
       Any result;
       static_cast<FunctionObj*>(data_.get())->CallPacked(args_pack, kNumArgs, &result);
       return result;
     }
     TVM_FFI_INLINE void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
       static_cast<FunctionObj*>(data_.get())->CallPacked(args, num_args, result);
     }
     TVM_FFI_INLINE void CallPacked(PackedArgs args, Any* result) const {
       static_cast<FunctionObj*>(data_.get())->CallPacked(args.data(), args.size(), result);
     }
   
     TVM_FFI_INLINE bool operator==(std::nullptr_t) const { return data_ == nullptr; }
     TVM_FFI_INLINE bool operator!=(std::nullptr_t) const { return data_ != nullptr; }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Function, ObjectRef, FunctionObj);
   
     class Registry;
   
    private:
     template <typename TCallable>
     static Function FromPackedInternal(TCallable packed_call) {
       using ObjType = typename details::FunctionObjImpl<TCallable>;
       Function func;
       func.data_ = make_object<ObjType>(std::forward<TCallable>(packed_call));
       return func;
     }
   };
   
   template <typename FType>
   class TypedFunction;
   
   template <typename R, typename... Args>
   class TypedFunction<R(Args...)> {
    public:
     using TSelf = TypedFunction<R(Args...)>;
     TypedFunction() {}
     TypedFunction(std::nullptr_t null) {}  // NOLINT(*)
     TypedFunction(Function packed) : packed_(packed) {}  // NOLINT(*)
     template <typename FLambda, typename = typename std::enable_if<std::is_convertible<
                                     FLambda, std::function<R(Args...)>>::value>::type>
     TypedFunction(FLambda typed_lambda, std::string name) {  // NOLINT(*)
       packed_ = Function::FromTyped(typed_lambda, name);
     }
     template <typename FLambda, typename = typename std::enable_if<std::is_convertible<
                                     FLambda, std::function<R(Args...)>>::value>::type>
     TypedFunction(const FLambda& typed_lambda) {  // NOLINT(*)
       packed_ = Function::FromTyped(typed_lambda);
     }
     template <typename FLambda, typename = typename std::enable_if<
                                     std::is_convertible<FLambda,
                                                         std::function<R(Args...)>>::value>::type>
     TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
       packed_ = Function::FromTyped(typed_lambda);
       return *this;
     }
     TSelf& operator=(Function packed) {
       packed_ = std::move(packed);
       return *this;
     }
     TVM_FFI_INLINE R operator()(Args... args) const {
       if constexpr (std::is_same_v<R, void>) {
         packed_(std::forward<Args>(args)...);
       } else {
         Any res = packed_(std::forward<Args>(args)...);
         if constexpr (std::is_same_v<R, Any>) {
           return res;
         } else {
           return std::move(res).cast<R>();
         }
       }
     }
     operator Function() const { return packed(); }
     const Function& packed() const& { return packed_; }
     constexpr Function&& packed() && { return std::move(packed_); }
     bool operator==(std::nullptr_t null) const { return packed_ == nullptr; }
     bool operator!=(std::nullptr_t null) const { return packed_ != nullptr; }
   
    private:
     Function packed_;
   };
   
   template <typename FType>
   inline constexpr bool use_default_type_traits_v<TypedFunction<FType>> = false;
   
   template <typename FType>
   struct TypeTraits<TypedFunction<FType>> : public TypeTraitsBase {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFunction;
   
     TVM_FFI_INLINE static void CopyToAnyView(const TypedFunction<FType>& src, TVMFFIAny* result) {
       TypeTraits<Function>::CopyToAnyView(src.packed(), result);
     }
   
     TVM_FFI_INLINE static void MoveToAny(TypedFunction<FType> src, TVMFFIAny* result) {
       TypeTraits<Function>::MoveToAny(std::move(src.packed()), result);
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return src->type_index == TypeIndex::kTVMFFIFunction;
     }
   
     TVM_FFI_INLINE static TypedFunction<FType> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       return TypedFunction<FType>(TypeTraits<Function>::CopyFromAnyViewAfterCheck(src));
     }
   
     TVM_FFI_INLINE static std::optional<TypedFunction<FType>> TryCastFromAnyView(
         const TVMFFIAny* src) {
       std::optional<Function> opt = TypeTraits<Function>::TryCastFromAnyView(src);
       if (opt.has_value()) {
         return TypedFunction<FType>(*std::move(opt));
       } else {
         return std::nullopt;
       }
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return details::FunctionInfo<FType>::Sig(); }
   };
   
   inline int32_t TypeKeyToIndex(std::string_view type_key) {
     int32_t type_index;
     TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
     TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
     return type_index;
   }
   
   #define TVM_FFI_DLL_EXPORT_TYPED_FUNC(ExportName, Function)                                    \
     extern "C" {                                                                                 \
     TVM_FFI_DLL_EXPORT int __tvm_ffi_##ExportName(void* self, TVMFFIAny* args, int32_t num_args, \
                                                   TVMFFIAny* result) {                           \
       TVM_FFI_SAFE_CALL_BEGIN();                                                                 \
       using FuncInfo = ::tvm::ffi::details::FunctionInfo<decltype(Function)>;                    \
       static std::string name = #ExportName;                                                     \
       ::tvm::ffi::details::unpack_call<typename FuncInfo::RetType>(                              \
           std::make_index_sequence<FuncInfo::num_args>{}, &name, Function,                       \
           reinterpret_cast<const ::tvm::ffi::AnyView*>(args), num_args,                          \
           reinterpret_cast<::tvm::ffi::Any*>(result));                                           \
       TVM_FFI_SAFE_CALL_END();                                                                   \
     }                                                                                            \
     }
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_FUNCTION_H_
