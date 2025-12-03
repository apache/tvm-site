
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
     return a + b;
   }
   
   TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, Add);
   TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC(
       add,
       R"(Add two integers and return the sum.
   
   Parameters
   ----------
   a : int
       First integer
   b : int
       Second integer
   
   Returns
   -------
   result : int
       Sum of a and b)");
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
   
   #ifndef TVM_FFI_DLL_EXPORT_INCLUDE_METADATA
   #define TVM_FFI_DLL_EXPORT_INCLUDE_METADATA 0
   #endif
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/base_details.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/function_details.h>
   
   #include <functional>
   #include <sstream>
   #include <string>
   #include <type_traits>
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
     using FCall = void (*)(const FunctionObj*, const AnyView*, int32_t, Any*);
     using TVMFFIFunctionCell::cpp_call;
     using TVMFFIFunctionCell::safe_call;
     TVM_FFI_INLINE void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
       // if cpp_call is set, use it to call the function, otherwise, redirect to safe_call
       // use conditional expression here so the select is branchless
       FCall call_ptr =
           this->cpp_call ? reinterpret_cast<FCall>(this->cpp_call) : CppCallDedirectToSafeCall;
       (*call_ptr)(this, args, num_args, result);
     }
     static constexpr const uint32_t _type_index = TypeIndex::kTVMFFIFunction;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIFunction, FunctionObj, Object);
   
    protected:
     FunctionObj() {}
     friend class Function;
   
    private:
     static void CppCallDedirectToSafeCall(const FunctionObj* func, const AnyView* args,
                                           int32_t num_args, Any* rv) {
       FunctionObj* self = static_cast<FunctionObj*>(const_cast<FunctionObj*>(func));
       TVM_FFI_CHECK_SAFE_CALL(self->safe_call(self, reinterpret_cast<const TVMFFIAny*>(args),
                                               num_args, reinterpret_cast<TVMFFIAny*>(rv)));
     }
   };
   
   namespace details {
   template <typename TCallable>
   class FunctionObjImpl : public FunctionObj {
    public:
     static_assert(std::is_same_v<TCallable, std::remove_cv_t<std::remove_reference_t<TCallable>>>,
                   "TCallable of FunctionObjImpl cannot be const or reference type");
   
     using TSelf = FunctionObjImpl<TCallable>;
   
     explicit FunctionObjImpl(TCallable&& callable) : callable_(std::move(callable)) {
       this->safe_call = SafeCall;
       this->cpp_call = reinterpret_cast<void*>(CppCall);
     }
     explicit FunctionObjImpl(const TCallable& callable) : callable_(callable) {
       this->safe_call = SafeCall;
       this->cpp_call = reinterpret_cast<void*>(CppCall);
     }
   
    private:
     // implementation of call
     static void CppCall(const FunctionObj* func, const AnyView* args, int32_t num_args, Any* result) {
       (static_cast<const TSelf*>(func))->callable_(args, num_args, result);
     }
     // Implementing safe call style
     static int SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
       TVM_FFI_SAFE_CALL_BEGIN();
       TVM_FFI_ICHECK_LT(result->type_index, TypeIndex::kTVMFFIStaticObjectBegin);
       FunctionObj* self = static_cast<FunctionObj*>(func);
       reinterpret_cast<FCall>(self->cpp_call)(self, reinterpret_cast<const AnyView*>(args), num_args,
                                               reinterpret_cast<Any*>(result));
       TVM_FFI_SAFE_CALL_END();
     }
   
     mutable TCallable callable_;
   };
   
   class ExternCFunctionObjNullHandleImpl : public FunctionObj {
    public:
     explicit ExternCFunctionObjNullHandleImpl(TVMFFISafeCallType safe_call) {
       this->safe_call = safe_call;
       this->cpp_call = nullptr;
     }
   };
   
   class ExternCFunctionObjImpl : public FunctionObj {
    public:
     ExternCFunctionObjImpl(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self))
         : self_(self), safe_call_(safe_call), deleter_(deleter) {
       this->safe_call = SafeCall;
       this->cpp_call = nullptr;
     }
   
     ~ExternCFunctionObjImpl() {
       if (deleter_) deleter_(self_);
     }
   
    private:
     static int32_t SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
       ExternCFunctionObjImpl* self = reinterpret_cast<ExternCFunctionObjImpl*>(func);
       return self->safe_call_(self->self_, args, num_args, rv);
     }
   
     void* self_;
     TVMFFISafeCallType safe_call_;
     void (*deleter_)(void* self);
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
     template <typename TCallable,
               typename = std::enable_if_t<!std::is_same_v<std::decay_t<TCallable>, Function>>>
     explicit Function(TCallable&& packed_call) {
       *this = FromPacked(std::forward<TCallable>(packed_call));
     }
     template <typename TCallable>
     static Function FromPacked(TCallable&& packed_call) {
       static_assert(
           std::is_convertible_v<TCallable, std::function<void(const AnyView*, int32_t, Any*)>> ||
               std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>,
           "tvm::ffi::Function::FromPacked requires input function signature to match packed func "
           "format");
       if constexpr (std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>) {
         return FromPackedInternal(
             [packed_call = std::forward<TCallable>(packed_call)](
                 const AnyView* args, int32_t num_args, Any* rv) mutable -> void {
               packed_call(PackedArgs{args, num_args}, rv);
             });
       } else {
         return FromPackedInternal(std::forward<TCallable>(packed_call));
       }
     }
   
     static Function FromExternC(void* self, TVMFFISafeCallType safe_call,
                                 void (*deleter)(void* self)) {
       // the other function coems from a different library
       Function func;
       if (self == nullptr && deleter == nullptr) {
         func.data_ = make_object<details::ExternCFunctionObjNullHandleImpl>(safe_call);
       } else {
         func.data_ = make_object<details::ExternCFunctionObjImpl>(self, safe_call, deleter);
       }
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
     static void SetGlobal(std::string_view name,
                           Function func,  // NOLINT(performance-unnecessary-value-param)
                           bool override = false) {
       TVMFFIByteArray name_arr{name.data(), name.size()};
       TVM_FFI_CHECK_SAFE_CALL(
           TVMFFIFunctionSetGlobal(&name_arr, details::ObjectUnsafe::GetHeader(func.get()), override));
     }
     static std::vector<String> ListGlobalNames() {
       Function fname_functor =
           GetGlobalRequired("ffi.FunctionListGlobalNamesFunctor")().cast<Function>();
       std::vector<String> names;
       int len = fname_functor(-1).cast<int>();
       names.reserve(len);
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
     static Function FromTyped(TCallable&& callable) {
       using FuncInfo = details::FunctionInfo<std::decay_t<TCallable>>;
       // Callable is always captured by value here to avoid possible dangling reference
       auto call_packed = [callable = std::forward<TCallable>(callable)](
                              const AnyView* args, int32_t num_args, Any* rv) mutable -> void {
         details::unpack_call<typename FuncInfo::RetType>(
             std::make_index_sequence<FuncInfo::num_args>{}, nullptr, callable, args, num_args, rv);
       };
       return FromPackedInternal(std::move(call_packed));
     }
     template <typename TCallable>
     static Function FromTyped(TCallable&& callable, std::string name) {
       using FuncInfo = details::FunctionInfo<std::decay_t<TCallable>>;
       // Callable is always captured by value here to avoid possible dangling reference
       auto call_packed = [callable = std::forward<TCallable>(callable), name = std::move(name)](
                              const AnyView* args, int32_t num_args, Any* rv) mutable -> void {
         details::unpack_call<typename FuncInfo::RetType>(
             std::make_index_sequence<FuncInfo::num_args>{}, &name, callable, args, num_args, rv);
       };
       return FromPackedInternal(std::move(call_packed));
     }
   
     template <typename... Args>
     TVM_FFI_INLINE static Any InvokeExternC(void* handle, TVMFFISafeCallType safe_call,
                                             Args&&... args) {
       const int kNumArgs = sizeof...(Args);
       const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
       AnyView args_pack[kArraySize];
       PackedArgs::Fill(args_pack, std::forward<Args>(args)...);
       Any result;
       TVM_FFI_CHECK_SAFE_CALL(safe_call(handle, reinterpret_cast<const TVMFFIAny*>(args_pack),
                                         kNumArgs, reinterpret_cast<TVMFFIAny*>(&result)));
       return result;
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
     static Function FromPackedInternal(TCallable&& packed_call) {
       // We must make TCallable a value type (decay_t) that can hold the callable object
       using ObjType = typename details::FunctionObjImpl<std::decay_t<TCallable>>;
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
     TypedFunction() = default;
     TypedFunction(std::nullptr_t null) {}  // NOLINT(*)
     TypedFunction(Function packed) : packed_(std::move(packed)) {}  // NOLINT(*)
     template <typename FLambda,
               typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>>>>
     TypedFunction(FLambda&& typed_lambda, std::string name) {
       packed_ = Function::FromTyped(std::forward<FLambda>(typed_lambda), std::move(name));
     }
     template <typename FLambda,
               typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>> &&
                                           !std::is_same_v<std::decay_t<FLambda>, TSelf>>>
     TypedFunction(FLambda&& typed_lambda) {  // NOLINT(google-explicit-constructor)
       packed_ = Function::FromTyped(std::forward<FLambda>(typed_lambda));
     }
     template <typename FLambda,
               typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>> &&
                                           !std::is_same_v<std::decay_t<FLambda>, TSelf>>>
     TSelf& operator=(FLambda&& typed_lambda) {
       packed_ = Function::FromTyped(std::forward<FLambda>(typed_lambda));
       return *this;
     }
     TSelf& operator=(Function packed) {
       packed_ = std::move(packed);
       return *this;
     }
     TVM_FFI_INLINE R operator()(Args... args) const {  // NOLINT(performance-unnecessary-value-param)
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
     operator Function() const { return packed(); }  // NOLINT(google-explicit-constructor)
     const Function& packed() const& { return packed_; }
     constexpr Function&& packed() && { return std::move(packed_); }
     bool operator==(std::nullptr_t null) const { return packed_ == nullptr; }
     bool operator!=(std::nullptr_t null) const { return packed_ != nullptr; }
     static std::string TypeSchema() { return details::FuncFunctorImpl<R, Args...>::TypeSchema(); }
   
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
       // Move from rvalue to trigger TypedFunction rvalue packed() overload
       TypeTraits<Function>::MoveToAny(std::move(src).packed(), result);
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
     TVM_FFI_INLINE static std::string TypeSchema() { return TypedFunction<FType>::TypeSchema(); }
   };
   
   inline int32_t TypeKeyToIndex(std::string_view type_key) {
     int32_t type_index;
     TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
     TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
     return type_index;
   }
   
   // Internal implementation macros used by TVM_FFI_DLL_EXPORT_TYPED_FUNC and related macros.
   // These should not be used directly; use the public macros instead.
   
   // Internal implementation macro that generates the C ABI wrapper function
   #define TVM_FFI_DLL_EXPORT_TYPED_FUNC_IMPL_(ExportName, Function)                      \
     extern "C" {                                                                         \
     TVM_FFI_DLL_EXPORT int __tvm_ffi_##ExportName(void* self, const TVMFFIAny* args,     \
                                                   int32_t num_args, TVMFFIAny* result) { \
       TVM_FFI_SAFE_CALL_BEGIN();                                                         \
       using FuncInfo = ::tvm::ffi::details::FunctionInfo<decltype(Function)>;            \
       static std::string name = #ExportName;                                             \
       ::tvm::ffi::details::unpack_call<typename FuncInfo::RetType>(                      \
           std::make_index_sequence<FuncInfo::num_args>{}, &name, Function,               \
           reinterpret_cast<const ::tvm::ffi::AnyView*>(args), num_args,                  \
           reinterpret_cast<::tvm::ffi::Any*>(result));                                   \
       TVM_FFI_SAFE_CALL_END();                                                           \
     }                                                                                    \
     }
   
   #if TVM_FFI_DLL_EXPORT_INCLUDE_METADATA
   // Implementation note: we specifically use TVMFFIStringFromByteArray
   // so the returned string metadata is allocated in the libtvm_ffi and long lived.
   #define TVM_FFI_DLL_EXPORT_TYPED_FUNC(ExportName, Function)                                      \
     TVM_FFI_DLL_EXPORT_TYPED_FUNC_IMPL_(ExportName, Function)                                      \
     extern "C" {                                                                                   \
     TVM_FFI_DLL_EXPORT int __tvm_ffi__metadata_##ExportName(void* self, const TVMFFIAny* args,     \
                                                             int32_t num_args, TVMFFIAny* result) { \
       TVM_FFI_SAFE_CALL_BEGIN();                                                                   \
       using FuncInfo = ::tvm::ffi::details::FunctionInfo<decltype(Function)>;                      \
       std::ostringstream os;                                                                       \
       os << R"({"type_schema":)"                                                                   \
          << ::tvm::ffi::EscapeString(::tvm::ffi::String(FuncInfo::TypeSchema())) << R"(})";        \
       std::string data = os.str();                                                                 \
       TVMFFIByteArray data_array{data.data(), data.size()};                                        \
       return TVMFFIStringFromByteArray(&data_array, result);                                       \
       TVM_FFI_SAFE_CALL_END();                                                                     \
     }                                                                                              \
     }
   #else
   #define TVM_FFI_DLL_EXPORT_TYPED_FUNC(ExportName, Function) \
     TVM_FFI_DLL_EXPORT_TYPED_FUNC_IMPL_(ExportName, Function)
   #endif
   
   #if TVM_FFI_DLL_EXPORT_INCLUDE_METADATA
   // Implementation note: we specifically use TVMFFIStringFromByteArray
   // so the returned string metadata is allocated in the libtvm_ffi and long lived.
   #define TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC(ExportName, DocString)                            \
     extern "C" {                                                                              \
     TVM_FFI_DLL_EXPORT int __tvm_ffi__doc_##ExportName(void* self, const TVMFFIAny* args,     \
                                                        int32_t num_args, TVMFFIAny* result) { \
       TVM_FFI_SAFE_CALL_BEGIN();                                                              \
       std::string_view data(DocString);                                                       \
       TVMFFIByteArray data_array{data.data(), data.size()};                                   \
       return TVMFFIStringFromByteArray(&data_array, result);                                  \
       TVM_FFI_SAFE_CALL_END();                                                                \
     }                                                                                         \
     }
   #else
   #define TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC(ExportName, DocString)
   #endif
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_FUNCTION_H_
