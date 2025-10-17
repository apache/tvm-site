
.. _program_listing_file_tvm_ffi_reflection_registry.h:

Program Listing for File registry.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_reflection_registry.h>` (``tvm/ffi/reflection/registry.h``)

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
   #ifndef TVM_FFI_REFLECTION_REGISTRY_H_
   #define TVM_FFI_REFLECTION_REGISTRY_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/container/variant.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ffi/type_traits.h>
   
   #include <iterator>
   #include <optional>
   #include <sstream>
   #include <string>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   namespace ffi {
   namespace reflection {
   using _MetadataType = std::vector<std::pair<String, Any>>;  // NOLINT(bugprone-reserved-identifier)
   struct FieldInfoBuilder : public TVMFFIFieldInfo {
     _MetadataType metadata_;
   };
   struct MethodInfoBuilder : public TVMFFIMethodInfo {
     _MetadataType metadata_;
   };
   
   struct InfoTrait {};
   
   class Metadata : public InfoTrait {
    public:
     Metadata(std::initializer_list<std::pair<String, Any>> dict) : dict_(dict) {}
     inline void Apply(FieldInfoBuilder* info) const { this->Apply(&info->metadata_); }
     inline void Apply(MethodInfoBuilder* info) const { this->Apply(&info->metadata_); }
   
    private:
     friend class GlobalDef;
     template <typename T>
     friend class ObjectDef;
     inline void Apply(_MetadataType* out) const {
       std::copy(std::make_move_iterator(dict_.begin()), std::make_move_iterator(dict_.end()),
                 std::back_inserter(*out));
     }
     static std::string ToJSON(const _MetadataType& metadata) {
       using ::tvm::ffi::details::StringObj;
       std::ostringstream os;
       os << "{";
       bool first = true;
       for (const auto& [key, value] : metadata) {
         if (!first) {
           os << ",";
         }
         os << "\"" << key << "\":";
         if (std::optional<int> v = value.as<int>()) {
           os << *v;
         } else if (std::optional<bool> v = value.as<bool>()) {
           os << (*v ? "true" : "false");
         } else if (std::optional<String> v = value.as<String>()) {
           String escaped = EscapeString(*v);
           os << escaped.c_str();
         } else {
           TVM_FFI_LOG_AND_THROW(TypeError) << "Metadata can be only int, bool or string, but on key `"
                                            << key << "`, the type is " << value.GetTypeKey();
         }
         first = false;
       }
       os << "}";
       return os.str();
     }
   
     std::vector<std::pair<String, Any>> dict_;
   };
   class DefaultValue : public InfoTrait {
    public:
     explicit DefaultValue(Any value) : value_(std::move(value)) {}
   
     TVM_FFI_INLINE void Apply(TVMFFIFieldInfo* info) const {
       info->default_value = AnyView(value_).CopyToTVMFFIAny();
       info->flags |= kTVMFFIFieldFlagBitMaskHasDefault;
     }
   
    private:
     Any value_;
   };
   
   class AttachFieldFlag : public InfoTrait {
    public:
     explicit AttachFieldFlag(int32_t flag) : flag_(flag) {}
   
     TVM_FFI_INLINE static AttachFieldFlag SEqHashDef() {
       return AttachFieldFlag(kTVMFFIFieldFlagBitMaskSEqHashDef);
     }
     TVM_FFI_INLINE static AttachFieldFlag SEqHashIgnore() {
       return AttachFieldFlag(kTVMFFIFieldFlagBitMaskSEqHashIgnore);
     }
   
     TVM_FFI_INLINE void Apply(TVMFFIFieldInfo* info) const { info->flags |= flag_; }
   
    private:
     int32_t flag_;
   };
   
   template <typename Class, typename T>
   TVM_FFI_INLINE int64_t GetFieldByteOffsetToObject(T Class::* field_ptr) {
     int64_t field_offset_to_class =
         reinterpret_cast<int64_t>(&(static_cast<Class*>(nullptr)->*field_ptr));
     return field_offset_to_class - details::ObjectUnsafe::GetObjectOffsetToSubclass<Class>();
   }
   
   class ReflectionDefBase {
    protected:
     template <typename T>
     static int FieldGetter(void* field, TVMFFIAny* result) {
       TVM_FFI_SAFE_CALL_BEGIN();
       *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(*reinterpret_cast<T*>(field)));
       TVM_FFI_SAFE_CALL_END();
     }
   
     template <typename T>
     static int FieldSetter(void* field, const TVMFFIAny* value) {
       TVM_FFI_SAFE_CALL_BEGIN();
       if constexpr (std::is_same_v<T, Any>) {
         *reinterpret_cast<T*>(field) = AnyView::CopyFromTVMFFIAny(*value);
       } else {
         *reinterpret_cast<T*>(field) = AnyView::CopyFromTVMFFIAny(*value).cast<T>();
       }
       TVM_FFI_SAFE_CALL_END();
     }
   
     template <typename T>
     static int ObjectCreatorDefault(TVMFFIObjectHandle* result) {
       TVM_FFI_SAFE_CALL_BEGIN();
       ObjectPtr<T> obj = make_object<T>();
       *result = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(obj));
       TVM_FFI_SAFE_CALL_END();
     }
   
     template <typename T>
     static int ObjectCreatorUnsafeInit(TVMFFIObjectHandle* result) {
       TVM_FFI_SAFE_CALL_BEGIN();
       ObjectPtr<T> obj = make_object<T>(UnsafeInit{});
       *result = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(obj));
       TVM_FFI_SAFE_CALL_END();
     }
   
     template <typename T>
     TVM_FFI_INLINE static void ApplyFieldInfoTrait(FieldInfoBuilder* info, const T& value) {
       if constexpr (std::is_base_of_v<InfoTrait, std::decay_t<T>>) {
         value.Apply(info);
       }
       if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
         info->doc = TVMFFIByteArray{value, std::char_traits<char>::length(value)};
       }
     }
   
     template <typename T>
     TVM_FFI_INLINE static void ApplyMethodInfoTrait(MethodInfoBuilder* info, const T& value) {
       if constexpr (std::is_base_of_v<InfoTrait, std::decay_t<T>>) {
         value.Apply(info);
       }
       if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
         info->doc = TVMFFIByteArray{value, std::char_traits<char>::length(value)};
       }
     }
   
     template <typename T>
     TVM_FFI_INLINE static void ApplyExtraInfoTrait(TVMFFITypeMetadata* info, const T& value) {
       if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
         info->doc = TVMFFIByteArray{value, std::char_traits<char>::length(value)};
       }
     }
   
     template <typename Class, typename R, typename... Args>
     TVM_FFI_INLINE static Function GetMethod(std::string name, R (Class::*func)(Args...)) {
       static_assert(std::is_base_of_v<ObjectRef, Class> || std::is_base_of_v<Object, Class>,
                     "Class must be derived from ObjectRef or Object");
       if constexpr (std::is_base_of_v<ObjectRef, Class>) {
         auto fwrap = [func](Class target, Args... params) -> R {
           // call method pointer
           return (target.*func)(std::forward<Args>(params)...);
         };
         return ffi::Function::FromTyped(fwrap, std::move(name));
       }
   
       if constexpr (std::is_base_of_v<Object, Class>) {
         auto fwrap = [func](const Class* target, Args... params) -> R {
           // call method pointer
           return (const_cast<Class*>(target)->*func)(std::forward<Args>(params)...);
         };
         return ffi::Function::FromTyped(fwrap, std::move(name));
       }
     }
   
     template <typename Class, typename R, typename... Args>
     TVM_FFI_INLINE static Function GetMethod(std::string name, R (Class::*func)(Args...) const) {
       static_assert(std::is_base_of_v<ObjectRef, Class> || std::is_base_of_v<Object, Class>,
                     "Class must be derived from ObjectRef or Object");
       if constexpr (std::is_base_of_v<ObjectRef, Class>) {
         auto fwrap = [func](const Class target, Args... params) -> R {
           // call method pointer
           return (target.*func)(std::forward<Args>(params)...);
         };
         return ffi::Function::FromTyped(fwrap, std::move(name));
       }
   
       if constexpr (std::is_base_of_v<Object, Class>) {
         auto fwrap = [func](const Class* target, Args... params) -> R {
           // call method pointer
           return (target->*func)(std::forward<Args>(params)...);
         };
         return ffi::Function::FromTyped(fwrap, std::move(name));
       }
     }
   
     template <typename Func>
     TVM_FFI_INLINE static Function GetMethod(std::string name, Func&& func) {
       return ffi::Function::FromTyped(std::forward<Func>(func), std::move(name));
     }
   };
   
   class GlobalDef : public ReflectionDefBase {
    public:
     template <typename Func, typename... Extra>
     GlobalDef& def(const char* name, Func&& func, Extra&&... extra) {
       using FuncInfo = details::FunctionInfo<std::decay_t<Func>>;
       RegisterFunc(name, ffi::Function::FromTyped(std::forward<Func>(func), std::string(name)),
                    FuncInfo::TypeSchema(), std::forward<Extra>(extra)...);
       return *this;
     }
   
     template <typename Func, typename... Extra>
     GlobalDef& def_packed(const char* name, Func func, Extra&&... extra) {
       RegisterFunc(name, ffi::Function::FromPacked(func), details::TypeSchemaImpl<Function>::v(),
                    std::forward<Extra>(extra)...);
       return *this;
     }
   
     template <typename Func, typename... Extra>
     GlobalDef& def_method(const char* name, Func&& func, Extra&&... extra) {
       using FuncInfo = details::FunctionInfo<std::decay_t<Func>>;
       RegisterFunc(name, GetMethod(std::string(name), std::forward<Func>(func)),
                    FuncInfo::TypeSchema(), std::forward<Extra>(extra)...);
       return *this;
     }
   
    private:
     template <typename... Extra>  // NOLINTNEXTLINE(performance-unnecessary-value-param)
     void RegisterFunc(const char* name, ffi::Function func, String type_schema, Extra&&... extra) {
       MethodInfoBuilder info;
       info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
       info.doc = TVMFFIByteArray{nullptr, 0};
       info.flags = 0;
       info.method = AnyView(func).CopyToTVMFFIAny();
       info.metadata_.emplace_back("type_schema", type_schema);
       ((ApplyMethodInfoTrait(&info, std::forward<Extra>(extra)), ...));
       std::string metadata_str = Metadata::ToJSON(info.metadata_);
       info.metadata = TVMFFIByteArray{metadata_str.c_str(), metadata_str.size()};
       TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionSetGlobalFromMethodInfo(&info, 0));
     }
   };
   
   template <typename... Args>
   struct init {
     // Allow ObjectDef to access the execute function
     template <typename Class>
     friend class ObjectDef;
   
     constexpr init() noexcept = default;
   
    private:
     template <typename Class>
     static inline ObjectRef execute(Args&&... args) {
       return ObjectRef(ffi::make_object<Class>(std::forward<Args>(args)...));
     }
   };
   
   template <typename Class>
   class ObjectDef : public ReflectionDefBase {
    public:
     template <typename... ExtraArgs>
     explicit ObjectDef(ExtraArgs&&... extra_args)
         : type_index_(Class::_GetOrAllocRuntimeTypeIndex()), type_key_(Class::_type_key) {
       RegisterExtraInfo(std::forward<ExtraArgs>(extra_args)...);
     }
   
     template <typename T, typename BaseClass, typename... Extra>
     TVM_FFI_INLINE ObjectDef& def_ro(const char* name, T BaseClass::* field_ptr, Extra&&... extra) {
       RegisterField(name, field_ptr, false, std::forward<Extra>(extra)...);
       return *this;
     }
   
     template <typename T, typename BaseClass, typename... Extra>
     TVM_FFI_INLINE ObjectDef& def_rw(const char* name, T BaseClass::* field_ptr, Extra&&... extra) {
       static_assert(Class::_type_mutable, "Only mutable classes are supported for writable fields");
       RegisterField(name, field_ptr, true, std::forward<Extra>(extra)...);
       return *this;
     }
   
     template <typename Func, typename... Extra>
     TVM_FFI_INLINE ObjectDef& def(const char* name, Func&& func, Extra&&... extra) {
       RegisterMethod(name, false, std::forward<Func>(func), std::forward<Extra>(extra)...);
       return *this;
     }
   
     template <typename Func, typename... Extra>
     TVM_FFI_INLINE ObjectDef& def_static(const char* name, Func&& func, Extra&&... extra) {
       RegisterMethod(name, true, std::forward<Func>(func), std::forward<Extra>(extra)...);
       return *this;
     }
   
     template <typename... Args, typename... Extra>
     TVM_FFI_INLINE ObjectDef& def([[maybe_unused]] init<Args...> init_func, Extra&&... extra) {
       RegisterMethod(kInitMethodName, true, &init<Args...>::template execute<Class>,
                      std::forward<Extra>(extra)...);
       return *this;
     }
   
    private:
     template <typename... ExtraArgs>
     void RegisterExtraInfo(ExtraArgs&&... extra_args) {
       TVMFFITypeMetadata info;
       info.total_size = sizeof(Class);
       info.structural_eq_hash_kind = Class::_type_s_eq_hash_kind;
       info.creator = nullptr;
       info.doc = TVMFFIByteArray{nullptr, 0};
       if constexpr (std::is_default_constructible_v<Class>) {
         info.creator = ObjectCreatorDefault<Class>;
       } else if constexpr (std::is_constructible_v<Class, UnsafeInit>) {
         info.creator = ObjectCreatorUnsafeInit<Class>;
       }
       // apply extra info traits
       ((ApplyExtraInfoTrait(&info, std::forward<ExtraArgs>(extra_args)), ...));
       TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMetadata(type_index_, &info));
     }
   
     template <typename T, typename BaseClass, typename... ExtraArgs>
     void RegisterField(const char* name, T BaseClass::* field_ptr, bool writable,
                        ExtraArgs&&... extra_args) {
       static_assert(std::is_base_of_v<BaseClass, Class>, "BaseClass must be a base class of Class");
       FieldInfoBuilder info;
       info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
       info.field_static_type_index = TypeToFieldStaticTypeIndex<T>::value;
       // store byte offset and setter, getter
       // so the same setter can be reused for all the same type
       info.offset = GetFieldByteOffsetToObject<Class, T>(field_ptr);
       info.size = sizeof(T);
       info.alignment = alignof(T);
       info.flags = 0;
       if (writable) {
         info.flags |= kTVMFFIFieldFlagBitMaskWritable;
       }
       info.getter = FieldGetter<T>;
       info.setter = FieldSetter<T>;
       // initialize default value to nullptr
       info.default_value = AnyView(nullptr).CopyToTVMFFIAny();
       info.doc = TVMFFIByteArray{nullptr, 0};
       info.metadata_.emplace_back("type_schema", details::TypeSchema<T>::v());
       // apply field info traits
       ((ApplyFieldInfoTrait(&info, std::forward<ExtraArgs>(extra_args)), ...));
       // call register
       std::string metadata_str = Metadata::ToJSON(info.metadata_);
       info.metadata = TVMFFIByteArray{metadata_str.c_str(), metadata_str.size()};
       TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterField(type_index_, &info));
     }
   
     // register a method
     template <typename Func, typename... Extra>
     void RegisterMethod(const char* name, bool is_static, Func&& func, Extra&&... extra) {
       using FuncInfo = details::FunctionInfo<std::decay_t<Func>>;
       MethodInfoBuilder info;
       info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
       info.doc = TVMFFIByteArray{nullptr, 0};
       info.flags = 0;
       if (is_static) {
         info.flags |= kTVMFFIFieldFlagBitMaskIsStaticMethod;
       }
       // obtain the method function
       Function method = GetMethod(std::string(type_key_) + "." + name, std::forward<Func>(func));
       info.method = AnyView(method).CopyToTVMFFIAny();
       info.metadata_.emplace_back("type_schema", FuncInfo::TypeSchema());
       // apply method info traits
       ((ApplyMethodInfoTrait(&info, std::forward<Extra>(extra)), ...));
       std::string metadata_str = Metadata::ToJSON(info.metadata_);
       info.metadata = TVMFFIByteArray{metadata_str.c_str(), metadata_str.size()};
       TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMethod(type_index_, &info));
     }
   
     int32_t type_index_;
     const char* type_key_;
     static constexpr const char* kInitMethodName = "__ffi_init__";
   };
   
   template <typename Class, typename = std::enable_if_t<std::is_base_of_v<Object, Class>>>
   class TypeAttrDef : public ReflectionDefBase {
    public:
     template <typename... ExtraArgs>
     explicit TypeAttrDef(ExtraArgs&&... extra_args)
         : type_index_(Class::RuntimeTypeIndex()), type_key_(Class::_type_key) {}
   
     template <typename Func>
     TypeAttrDef& def(const char* name, Func&& func) {
       TVMFFIByteArray name_array = {name, std::char_traits<char>::length(name)};
       ffi::Function ffi_func =
           GetMethod(std::string(type_key_) + "." + name, std::forward<Func>(func));
       TVMFFIAny value_any = AnyView(ffi_func).CopyToTVMFFIAny();
       TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name_array, &value_any));
       return *this;
     }
   
     template <typename T>
     TypeAttrDef& attr(const char* name, T value) {
       TVMFFIByteArray name_array = {name, std::char_traits<char>::length(name)};
       TVMFFIAny value_any = AnyView(value).CopyToTVMFFIAny();
       TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name_array, &value_any));
       return *this;
     }
   
    private:
     int32_t type_index_;
     const char* type_key_;
   };
   
   inline void EnsureTypeAttrColumn(std::string_view name) {
     TVMFFIByteArray name_array = {name.data(), name.size()};
     AnyView any_view(nullptr);
     TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(kTVMFFINone, &name_array,
                                                    reinterpret_cast<const TVMFFIAny*>(&any_view)));
   }
   
   }  // namespace reflection
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_REFLECTION_REGISTRY_H_
