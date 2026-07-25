
.. _program_listing_file_tvm_ffi_type_traits.h:

Program Listing for File type_traits.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_type_traits.h>` (``tvm/ffi/type_traits.h``)

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
   #ifndef TVM_FFI_TYPE_TRAITS_H_
   #define TVM_FFI_TYPE_TRAITS_H_
   
   #include <tvm/ffi/base_details.h>
   #include <tvm/ffi/c_api.h>
   
   #include <optional>
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   class Any;
   
   using TypeIndex = TVMFFITypeIndex;
   using TypeInfo = TVMFFITypeInfo;
   
   struct StaticTypeKey {
     static constexpr const char* kTVMFFIAny = "Any";
     static constexpr const char* kTVMFFINone = "None";
     static constexpr const char* kTVMFFIBool = "bool";
     static constexpr const char* kTVMFFIInt = "int";
     static constexpr const char* kTVMFFIFloat = "float";
     static constexpr const char* kTVMFFIOpaquePtr = "void*";
     static constexpr const char* kTVMFFIDataType = "DataType";
     static constexpr const char* kTVMFFIDevice = "Device";
     static constexpr const char* kTVMFFIDLTensorPtr = "DLTensor*";
     static constexpr const char* kTVMFFIRawStr = "const char*";
     static constexpr const char* kTVMFFIByteArrayPtr = "TVMFFIByteArray*";
     static constexpr const char* kTVMFFIObjectRValueRef = "ObjectRValueRef";
     static constexpr const char* kTVMFFISmallStr = "ffi.SmallStr";
     static constexpr const char* kTVMFFISmallBytes = "ffi.SmallBytes";
     static constexpr const char* kTVMFFIError = "ffi.Error";
     static constexpr const char* kTVMFFIBytes = "ffi.Bytes";
     static constexpr const char* kTVMFFIStr = "ffi.String";
     static constexpr const char* kTVMFFIShape = "ffi.Shape";
     static constexpr const char* kTVMFFITensor = "ffi.Tensor";
     static constexpr const char* kTVMFFIObject = "ffi.Object";
     static constexpr const char* kTVMFFIFunction = "ffi.Function";
     static constexpr const char* kTVMFFIArray = "ffi.Array";
     static constexpr const char* kTVMFFIList = "ffi.List";
     static constexpr const char* kTVMFFIMap = "ffi.Map";
     static constexpr const char* kTVMFFIModule = "ffi.Module";
     static constexpr const char* kTVMFFIDict = "ffi.Dict";
     static constexpr const char* kTVMFFIVisitInterrupt = "ffi.VisitInterrupt";
     static constexpr const char* kTVMFFIOpaquePyObject = "ffi.OpaquePyObject";
   };
   
   inline std::string TypeIndexToTypeKey(int32_t type_index) {
     const TypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
     return std::string(type_info->type_key.data, type_info->type_key.size);
   }
   
   template <typename TargetType, typename SourceType, typename = void>
   inline constexpr bool type_subsumes_v =
       std::is_base_of_v<TargetType, SourceType> || std::is_same_v<TargetType, SourceType>;
   
   
   template <typename SourceType>
   inline constexpr bool type_subsumes_v<Any, SourceType> = true;
   
   template <typename, typename = void>
   struct TypeTraits {
     static constexpr bool convert_enabled = false;
     static constexpr bool storage_enabled = false;
   };
   
   template <typename T>
   using TypeTraitsNoCR = TypeTraits<std::remove_const_t<std::remove_reference_t<T>>>;
   
   template <typename T>
   inline constexpr bool use_default_type_traits_v = true;
   
   struct TypeTraitsBase {
     static constexpr bool convert_enabled = true;
     static constexpr bool storage_enabled = true;
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;
     // get mismatched type when result mismatches the trait.
     // this function is called after TryCastFromAnyView fails
     // to get more detailed type information in runtime
     // especially when the error involves nested container type
     TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* source) {
       return TypeIndexToTypeKey(source->type_index);
     }
   };
   
   template <typename T, typename = void>
   struct TypeToFieldStaticTypeIndex {
     static constexpr int32_t value = TypeIndex::kTVMFFIAny;
   };
   
   template <typename T>
   struct TypeToFieldStaticTypeIndex<T, std::enable_if_t<TypeTraits<T>::convert_enabled>> {
     static constexpr int32_t value = TypeTraits<T>::field_static_type_index;
   };
   
   template <typename T, typename = void>
   struct TypeToRuntimeTypeIndex {
     static int32_t v() { return TypeToFieldStaticTypeIndex<T>::value; }
   };
   
   // None
   template <>
   struct TypeTraits<std::nullptr_t> : public TypeTraitsBase {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFINone;
     TVM_FFI_INLINE static void CopyToAnyView(const std::nullptr_t&, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFINone;
       result->zero_padding = 0;
       // invariant: the pointer field also equals nullptr
       // this will simplify same_as comparisons and hash
       result->v_int64 = 0;
     }
   
     TVM_FFI_INLINE static void MoveToAny(std::nullptr_t, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFINone;
       result->zero_padding = 0;
       // invariant: the pointer field also equals nullptr
       // this will simplify same_as comparisons and hash
       result->v_int64 = 0;
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return src->type_index == TypeIndex::kTVMFFINone;
     }
   
     TVM_FFI_INLINE static std::nullptr_t CopyFromAnyViewAfterCheck(const TVMFFIAny*) {
       return nullptr;
     }
   
     TVM_FFI_INLINE static std::nullptr_t MoveFromAnyAfterCheck(TVMFFIAny*) { return nullptr; }
   
     TVM_FFI_INLINE static std::optional<std::nullptr_t> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFINone) {
         return nullptr;
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFINone; }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFINone) + R"("})";
     }
   };
   
   class StrictBool {
    public:
     StrictBool(bool value) : value_(value) {}  // NOLINT(google-explicit-constructor)
     operator bool() const { return value_; }  // NOLINT(google-explicit-constructor)
   
    private:
     bool value_;
   };
   
   template <>
   struct TypeTraits<StrictBool> : public TypeTraitsBase {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;
   
     TVM_FFI_INLINE static void CopyToAnyView(const StrictBool& src, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFIBool;
       result->zero_padding = 0;
       result->v_int64 = static_cast<bool>(src);
     }
   
     TVM_FFI_INLINE static void MoveToAny(StrictBool src, TVMFFIAny* result) {
       CopyToAnyView(src, result);
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return src->type_index == TypeIndex::kTVMFFIBool;
     }
   
     TVM_FFI_INLINE static StrictBool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIBool);
       return static_cast<bool>(src->v_int64);
     }
   
     TVM_FFI_INLINE static StrictBool MoveFromAnyAfterCheck(TVMFFIAny* src) {
       // POD type, we can just copy the value
       return CopyFromAnyViewAfterCheck(src);
     }
   
     TVM_FFI_INLINE static std::optional<StrictBool> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFIBool) {
         return StrictBool(static_cast<bool>(src->v_int64));
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIBool) + R"("})";
     }
   };
   
   // Bool type, allow implicit casting from int
   template <>
   struct TypeTraits<bool> : public TypeTraitsBase {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;
   
     TVM_FFI_INLINE static void CopyToAnyView(const bool& src, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFIBool;
       result->zero_padding = 0;
       result->v_int64 = static_cast<int64_t>(src);
     }
   
     TVM_FFI_INLINE static void MoveToAny(bool src, TVMFFIAny* result) { CopyToAnyView(src, result); }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return src->type_index == TypeIndex::kTVMFFIBool;
     }
   
     TVM_FFI_INLINE static bool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIBool);
       return static_cast<bool>(src->v_int64);
     }
   
     TVM_FFI_INLINE static bool MoveFromAnyAfterCheck(TVMFFIAny* src) {
       // POD type, we can just copy the value
       return CopyFromAnyViewAfterCheck(src);
     }
   
     TVM_FFI_INLINE static std::optional<bool> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
         return static_cast<bool>(src->v_int64);
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIBool) + R"("})";
     }
   };
   
   template <typename Int>
   struct TypeTraitsIntBase : public TypeTraitsBase {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIInt;
   
     TVM_FFI_INLINE static void CopyInt64ToAnyView(int64_t src, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFIInt;
       result->zero_padding = 0;
       result->v_int64 = src;
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
       return src->type_index == TypeIndex::kTVMFFIInt;
     }
   
     TVM_FFI_INLINE static Int CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIInt);
       return static_cast<Int>(src->v_int64);
     }
   
     TVM_FFI_INLINE static Int MoveFromAnyAfterCheck(TVMFFIAny* src) {
       // POD type, we can just copy the value
       return CopyFromAnyViewAfterCheck(src);
     }
   
     TVM_FFI_INLINE static std::optional<Int> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
         return Int(src->v_int64);
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIInt; }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIInt) + R"("})";
     }
   };
   
   // Integer POD values
   template <typename Int>
   struct TypeTraits<Int, std::enable_if_t<std::is_integral_v<Int>>> : public TypeTraitsIntBase<Int> {
     TVM_FFI_INLINE static void CopyToAnyView(const Int& src, TVMFFIAny* result) {
       TypeTraitsIntBase<Int>::CopyInt64ToAnyView(static_cast<int64_t>(src), result);
     }
   
     TVM_FFI_INLINE static void MoveToAny(Int src, TVMFFIAny* result) { CopyToAnyView(src, result); }
   };
   
   
   // trait to check if a type is an integeral enum
   // note that we need this trait so we can confirm underlying_type_t is an integral type
   // to avoid potential undefined behavior
   template <typename T, bool = std::is_enum_v<T>>
   constexpr bool is_integeral_enum_v = false;
   
   template <typename T>
   constexpr bool is_integeral_enum_v<T, true> = std::is_integral_v<std::underlying_type_t<T>>;
   
   
   // Enum Integer POD values
   template <typename IntEnum>
   struct TypeTraits<IntEnum, std::enable_if_t<is_integeral_enum_v<IntEnum>>>
       : public TypeTraitsIntBase<IntEnum> {
     TVM_FFI_INLINE static void CopyToAnyView(const IntEnum& src, TVMFFIAny* result) {
       TypeTraitsIntBase<IntEnum>::CopyInt64ToAnyView(static_cast<int64_t>(src), result);
     }
   
     TVM_FFI_INLINE static void MoveToAny(IntEnum src, TVMFFIAny* result) {
       CopyToAnyView(src, result);
     }
   };
   
   // Float POD values
   template <typename Float>
   struct TypeTraits<Float, std::enable_if_t<std::is_floating_point_v<Float>>>
       : public TypeTraitsBase {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFloat;
   
     TVM_FFI_INLINE static void CopyToAnyView(const Float& src, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFIFloat;
       result->zero_padding = 0;
       result->v_float64 = static_cast<double>(src);
     }
   
     TVM_FFI_INLINE static void MoveToAny(Float src, TVMFFIAny* result) { CopyToAnyView(src, result); }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
       return src->type_index == TypeIndex::kTVMFFIFloat;
     }
   
     TVM_FFI_INLINE static Float CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIFloat);
       return static_cast<Float>(src->v_float64);
     }
   
     TVM_FFI_INLINE static Float MoveFromAnyAfterCheck(TVMFFIAny* src) {
       // POD type, we can just copy the value
       return CopyFromAnyViewAfterCheck(src);
     }
   
     TVM_FFI_INLINE static std::optional<Float> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFIFloat) {
         return Float(src->v_float64);
       } else if (src->type_index == TypeIndex::kTVMFFIInt ||
                  src->type_index == TypeIndex::kTVMFFIBool) {
         return Float(src->v_int64);
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIFloat; }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIFloat) + R"("})";
     }
   };
   
   // void*
   template <>
   struct TypeTraits<void*> : public TypeTraitsBase {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIOpaquePtr;
   
     TVM_FFI_INLINE static void CopyToAnyView(void* src, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFIOpaquePtr;
       result->zero_padding = 0;
       TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
       result->v_ptr = src;
     }
   
     TVM_FFI_INLINE static void MoveToAny(void* src, TVMFFIAny* result) { CopyToAnyView(src, result); }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
       return src->type_index == TypeIndex::kTVMFFIOpaquePtr;
     }
   
     TVM_FFI_INLINE static void* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIOpaquePtr);
       return src->v_ptr;
     }
   
     TVM_FFI_INLINE static void* MoveFromAnyAfterCheck(TVMFFIAny* src) {
       // POD type, we can just copy the value
       return CopyFromAnyViewAfterCheck(src);
     }
   
     TVM_FFI_INLINE static std::optional<void*> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFIOpaquePtr) {
         return static_cast<void*>(src->v_ptr);
       }
       if (src->type_index == TypeIndex::kTVMFFINone) {
         return static_cast<void*>(nullptr);
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIOpaquePtr; }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIOpaquePtr) + R"("})";
     }
   };
   
   template <typename T, typename... FallbackTypes>
   struct FallbackOnlyTraitsBase : public TypeTraitsBase {
     // disable container for FallbackOnlyTraitsBase
     static constexpr bool storage_enabled = false;
   
     TVM_FFI_INLINE static std::optional<T> TryCastFromAnyView(const TVMFFIAny* src) {
       return TryFallbackTypes<FallbackTypes...>(src);
     }
   
     template <typename FallbackType, typename... Rest>
     TVM_FFI_INLINE static std::optional<T> TryFallbackTypes(const TVMFFIAny* src) {
       static_assert(!std::is_same_v<bool, FallbackType>,
                     "Using bool as FallbackType can cause bug because int will be detected as bool, "
                     "use tvm::ffi::StrictBool instead");
       if (auto opt_fallback = TypeTraits<FallbackType>::TryCastFromAnyView(src)) {
         return TypeTraits<T>::ConvertFallbackValue(*std::move(opt_fallback));
       }
       if constexpr (sizeof...(Rest) > 0) {
         return TryFallbackTypes<Rest...>(src);
       }
       return std::nullopt;
     }
   };
   
   }  // namespace ffi
   }  // namespace tvm
   
   #define TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP(RegisteredKey, TypeDepth)           \
     static constexpr const char* _type_key = RegisteredKey;                      \
     static constexpr int32_t _type_depth = TypeDepth;                            \
     static constexpr bool _type_mutable = true;                                  \
     static constexpr bool _type_final = false;                                   \
     static constexpr uint32_t _type_child_slots = 0;                             \
     static constexpr bool _type_child_slots_can_overflow = true;                 \
     static int32_t RuntimeTypeIndex() {                                          \
       static const int32_t type_index = []() {                                   \
         constexpr TVMFFIByteArray key{RegisteredKey, sizeof(RegisteredKey) - 1}; \
         int32_t result = -1;                                                     \
         TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&key, &result));            \
         return result;                                                           \
       }();                                                                       \
       return type_index;                                                         \
     }
   
   #endif  // TVM_FFI_TYPE_TRAITS_H_
