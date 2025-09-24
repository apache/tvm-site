
.. _program_listing_file_tvm_ffi_object.h:

Program Listing for File object.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_object.h>` (``tvm/ffi/object.h``)

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
   #ifndef TVM_FFI_OBJECT_H_
   #define TVM_FFI_OBJECT_H_
   
   #include <tvm/ffi/base_details.h>
   #include <tvm/ffi/c_api.h>
   
   #include <optional>
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   using TypeIndex = TVMFFITypeIndex;
   
   using TypeInfo = TVMFFITypeInfo;
   
   struct UnsafeInit {};
   
   struct StaticTypeKey {
     static constexpr const char* kTVMFFIAny = "Any";
     static constexpr const char* kTVMFFINone = "None";
     static constexpr const char* kTVMFFIBool = "bool";
     static constexpr const char* kTVMFFIInt = "int";
     static constexpr const char* kTVMFFIFloat = "float";
     static constexpr const char* kTVMFFIOpaquePtr = "void*";
     static constexpr const char* kTVMFFIDataType = "DataType";
     static constexpr const char* kTVMFFIDevice = "Device";
     static constexpr const char* kTVMFFIRawStr = "const char*";
     static constexpr const char* kTVMFFIByteArrayPtr = "TVMFFIByteArray*";
     static constexpr const char* kTVMFFIObjectRValueRef = "ObjectRValueRef";
     static constexpr const char* kTVMFFISmallStr = "ffi.SmallStr";
     static constexpr const char* kTVMFFISmallBytes = "ffi.SmallBytes";
     static constexpr const char* kTVMFFIBytes = "ffi.Bytes";
     static constexpr const char* kTVMFFIStr = "ffi.String";
     static constexpr const char* kTVMFFIShape = "ffi.Shape";
     static constexpr const char* kTVMFFITensor = "ffi.Tensor";
     static constexpr const char* kTVMFFIObject = "ffi.Object";
     static constexpr const char* kTVMFFIFunction = "ffi.Function";
     static constexpr const char* kTVMFFIArray = "ffi.Array";
     static constexpr const char* kTVMFFIMap = "ffi.Map";
     static constexpr const char* kTVMFFIModule = "ffi.Module";
   };
   
   inline std::string TypeIndexToTypeKey(int32_t type_index) {
     const TypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
     return std::string(type_info->type_key.data, type_info->type_key.size);
   }
   
   namespace details {
   // Helper to perform
   // unsafe operations related to object
   struct ObjectUnsafe;
   
   template <typename TargetType>
   TVM_FFI_INLINE bool IsObjectInstance(int32_t object_type_index);
   }  // namespace details
   
   class Object {
    protected:
     TVMFFIObject header_;
   
    public:
     Object() {
       header_.strong_ref_count = 0;
       header_.weak_ref_count = 0;
       header_.deleter = nullptr;
     }
     template <typename TargetType>
     bool IsInstance() const {
       return details::IsObjectInstance<TargetType>(header_.type_index);
     }
   
     int32_t type_index() const { return header_.type_index; }
   
     std::string GetTypeKey() const {
       // the function checks that the info exists
       const TypeInfo* type_info = TVMFFIGetTypeInfo(header_.type_index);
       return std::string(type_info->type_key.data, type_info->type_key.size);
     }
   
     uint64_t GetTypeKeyHash() const {
       // the function checks that the info exists
       const TypeInfo* type_info = TVMFFIGetTypeInfo(header_.type_index);
       return type_info->type_key_hash;
     }
   
     static std::string TypeIndex2Key(int32_t tindex) {
       const TypeInfo* type_info = TVMFFIGetTypeInfo(tindex);
       return std::string(type_info->type_key.data, type_info->type_key.size);
     }
   
     bool unique() const { return use_count() == 1; }
   
     int32_t use_count() const {
       // only need relaxed load of counters
   #ifdef _MSC_VER
       return (reinterpret_cast<const volatile __int64*>(&header_.strong_ref_count))[0];  // NOLINT(*)
   #else
       return __atomic_load_n(&(header_.strong_ref_count), __ATOMIC_RELAXED);
   #endif
     }
   
     //----------------------------------------------------------------------------
     //  The following fields are configuration flags for subclasses of object
     //----------------------------------------------------------------------------
     static constexpr const char* _type_key = StaticTypeKey::kTVMFFIObject;
     static constexpr bool _type_final = false;
     static constexpr bool _type_mutable = false;
     static constexpr uint32_t _type_child_slots = 0;
     static constexpr bool _type_child_slots_can_overflow = true;
     static constexpr int32_t _type_index = TypeIndex::kTVMFFIObject;
     static constexpr int32_t _type_depth = 0;
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUnsupported;
     // The following functions are provided by macro
     // TVM_FFI_DECLARE_OBJECT_INFO and TVM_FFI_DECLARE_OBJECT_INFO_FINAL
     static int32_t RuntimeTypeIndex() { return TypeIndex::kTVMFFIObject; }
     static int32_t _GetOrAllocRuntimeTypeIndex() { return TypeIndex::kTVMFFIObject; }
   
    private:
     void IncRef() {
   #ifdef _MSC_VER
       _InterlockedIncrement64(
           reinterpret_cast<volatile __int64*>(&header_.strong_ref_count));  // NOLINT(*)
   #else
       __atomic_fetch_add(&(header_.strong_ref_count), 1, __ATOMIC_RELAXED);
   #endif
     }
     bool TryPromoteWeakPtr() {
   #ifdef _MSC_VER
       uint64_t old_count =
           (reinterpret_cast<const volatile __int64*>(&header_.strong_ref_count))[0];  // NOLINT(*)
       while (old_count > 0) {
         uint64_t new_count = old_count + 1;
         uint64_t old_count_loaded = _InterlockedCompareExchange64(
             reinterpret_cast<volatile __int64*>(&header_.strong_ref_count), new_count, old_count);
         if (old_count == old_count_loaded) {
           return true;
         }
         old_count = old_count_loaded;
       }
       return false;
   #else
       uint64_t old_count = __atomic_load_n(&(header_.strong_ref_count), __ATOMIC_RELAXED);
       while (old_count > 0) {
         // must do CAS to ensure that we are the only one that increases the reference count
         // avoid condition when two threads tries to promote weak to strong at same time
         // or when strong deletion happens between the load and the CAS
         uint64_t new_count = old_count + 1;
         if (__atomic_compare_exchange_n(&(header_.strong_ref_count), &old_count, new_count, true,
                                         __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
           return true;
         }
       }
       return false;
   #endif
     }
   
     void IncWeakRef() {
   #ifdef _MSC_VER
       _InterlockedIncrement(reinterpret_cast<volatile long*>(&header_.weak_ref_count));  // NOLINT(*)
   #else
       __atomic_fetch_add(&(header_.weak_ref_count), 1, __ATOMIC_RELAXED);
   #endif
     }
   
     void DecRef() {
   #ifdef _MSC_VER
       // use simpler impl in windows to ensure correctness
       if (_InterlockedDecrement64(                                                     //
               reinterpret_cast<volatile __int64*>(&header_.strong_ref_count)) == 0) {  // NOLINT(*)
         // full barrrier is implicit in InterlockedDecrement
         if (header_.deleter != nullptr) {
           header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskStrong);
         }
         if (_InterlockedDecrement(                                                  //
                 reinterpret_cast<volatile long*>(&header_.weak_ref_count)) == 0) {  // NOLINT(*)
           if (header_.deleter != nullptr) {
             header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
           }
         }
       }
   #else
       // first do a release, note we only need to acquire for deleter
       if (__atomic_fetch_sub(&(header_.strong_ref_count), 1, __ATOMIC_RELEASE) == 1) {
         if (__atomic_load_n(&(header_.weak_ref_count), __ATOMIC_RELAXED) == 1) {
           // common case, we need to delete both the object and the memory block
           // only acquire when we need to call deleter
           __atomic_thread_fence(__ATOMIC_ACQUIRE);
           if (header_.deleter != nullptr) {
             // call deleter once
             header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskBoth);
           }
         } else {
           // Slower path: there is still a weak reference left
           __atomic_thread_fence(__ATOMIC_ACQUIRE);
           // call destructor first, then decrease weak reference count
           if (header_.deleter != nullptr) {
             header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskStrong);
           }
           // now decrease weak reference count
           if (__atomic_fetch_sub(&(header_.weak_ref_count), 1, __ATOMIC_RELEASE) == 1) {
             __atomic_thread_fence(__ATOMIC_ACQUIRE);
             if (header_.deleter != nullptr) {
               header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
             }
           }
         }
       }
   #endif
     }
   
     void DecWeakRef() {
   #ifdef _MSC_VER
       if (_InterlockedDecrement(                                                  //
               reinterpret_cast<volatile long*>(&header_.weak_ref_count)) == 0) {  // NOLINT(*)
         if (header_.deleter != nullptr) {
           header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
         }
       }
   #else
       // now decrease weak reference count
       if (__atomic_fetch_sub(&(header_.weak_ref_count), 1, __ATOMIC_RELEASE) == 1) {
         __atomic_thread_fence(__ATOMIC_ACQUIRE);
         if (header_.deleter != nullptr) {
           header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
         }
       }
   #endif
     }
   
     // friend classes
     template <typename>
     friend class ObjectPtr;
     template <typename>
     friend class WeakObjectPtr;
     friend struct tvm::ffi::details::ObjectUnsafe;
   };
   
   template <typename T>
   class ObjectPtr {
    public:
     ObjectPtr() {}
     ObjectPtr(std::nullptr_t) {}  // NOLINT(*)
     ObjectPtr(const ObjectPtr<T>& other)  // NOLINT(*)
         : ObjectPtr(other.data_) {}
     template <typename U>
     ObjectPtr(const ObjectPtr<U>& other)  // NOLINT(*)
         : ObjectPtr(other.data_) {
       static_assert(std::is_base_of<T, U>::value,
                     "can only assign of child class ObjectPtr to parent");
     }
     ObjectPtr(ObjectPtr<T>&& other)  // NOLINT(*)
         : data_(other.data_) {
       other.data_ = nullptr;
     }
     template <typename Y>
     ObjectPtr(ObjectPtr<Y>&& other)  // NOLINT(*)
         : data_(other.data_) {
       static_assert(std::is_base_of<T, Y>::value,
                     "can only assign of child class ObjectPtr to parent");
       other.data_ = nullptr;
     }
     ~ObjectPtr() { this->reset(); }
     void swap(ObjectPtr<T>& other) {  // NOLINT(*)
       std::swap(data_, other.data_);
     }
     T* get() const { return static_cast<T*>(data_); }
     T* operator->() const { return get(); }
     T& operator*() const {  // NOLINT(*)
       return *get();
     }
     ObjectPtr<T>& operator=(const ObjectPtr<T>& other) {  // NOLINT(*)
       // takes in plane operator to enable copy elison.
       // copy-and-swap idiom
       ObjectPtr(other).swap(*this);  // NOLINT(*)
       return *this;
     }
     ObjectPtr<T>& operator=(ObjectPtr<T>&& other) {  // NOLINT(*)
       // copy-and-swap idiom
       ObjectPtr(std::move(other)).swap(*this);  // NOLINT(*)
       return *this;
     }
     explicit operator bool() const { return get() != nullptr; }
     void reset() {
       if (data_ != nullptr) {
         data_->DecRef();
         data_ = nullptr;
       }
     }
     int use_count() const { return data_ != nullptr ? data_->use_count() : 0; }
     bool unique() const { return data_ != nullptr && data_->use_count() == 1; }
     bool operator==(const ObjectPtr<T>& other) const { return data_ == other.data_; }
     bool operator!=(const ObjectPtr<T>& other) const { return data_ != other.data_; }
     bool operator==(std::nullptr_t) const { return data_ == nullptr; }
     bool operator!=(std::nullptr_t) const { return data_ != nullptr; }
   
    private:
     Object* data_{nullptr};
     explicit ObjectPtr(Object* data) : data_(data) {
       if (data_ != nullptr) {
         data_->IncRef();
       }
     }
     // friend classes
     friend class Object;
     friend class ObjectRef;
     friend struct ObjectPtrHash;
     template <typename>
     friend class ObjectPtr;
     template <typename>
     friend class WeakObjectPtr;
     friend struct tvm::ffi::details::ObjectUnsafe;
   };
   
   template <typename T>
   class WeakObjectPtr {
    public:
     WeakObjectPtr() {}
     WeakObjectPtr(std::nullptr_t) {}  // NOLINT(*)
     WeakObjectPtr(const WeakObjectPtr<T>& other)  // NOLINT(*)
         : WeakObjectPtr(other.data_) {}
   
     WeakObjectPtr(const ObjectPtr<T>& other)  // NOLINT(*)
         : WeakObjectPtr(other.get()) {}
     template <typename U>
     WeakObjectPtr(const WeakObjectPtr<U>& other)  // NOLINT(*)
         : WeakObjectPtr(other.data_) {
       static_assert(std::is_base_of<T, U>::value,
                     "can only assign of child class ObjectPtr to parent");
     }
     template <typename U>
     WeakObjectPtr(const ObjectPtr<U>& other)  // NOLINT(*)
         : WeakObjectPtr(other.data_) {
       static_assert(std::is_base_of<T, U>::value,
                     "can only assign of child class ObjectPtr to parent");
     }
     WeakObjectPtr(WeakObjectPtr<T>&& other)  // NOLINT(*)
         : data_(other.data_) {
       other.data_ = nullptr;
     }
     template <typename Y>
     WeakObjectPtr(WeakObjectPtr<Y>&& other)  // NOLINT(*)
         : data_(other.data_) {
       static_assert(std::is_base_of<T, Y>::value,
                     "can only assign of child class ObjectPtr to parent");
       other.data_ = nullptr;
     }
     ~WeakObjectPtr() { this->reset(); }
     void swap(WeakObjectPtr<T>& other) {  // NOLINT(*)
       std::swap(data_, other.data_);
     }
   
     WeakObjectPtr<T>& operator=(const WeakObjectPtr<T>& other) {  // NOLINT(*)
       // takes in plane operator to enable copy elison.
       // copy-and-swap idiom
       WeakObjectPtr(other).swap(*this);  // NOLINT(*)
       return *this;
     }
     WeakObjectPtr<T>& operator=(WeakObjectPtr<T>&& other) {  // NOLINT(*)
       // copy-and-swap idiom
       WeakObjectPtr(std::move(other)).swap(*this);  // NOLINT(*)
       return *this;
     }
   
     ObjectPtr<T> lock() const {
       if (data_ != nullptr && data_->TryPromoteWeakPtr()) {
         ObjectPtr<T> ret;
         // we already increase the reference count, so we don't need to do it again
         ret.data_ = data_;
         return ret;
       }
       return nullptr;
     }
   
     void reset() {
       if (data_ != nullptr) {
         data_->DecWeakRef();
         data_ = nullptr;
       }
     }
   
     int use_count() const { return data_ != nullptr ? data_->use_count() : 0; }
   
     bool expired() const { return data_ == nullptr || data_->use_count() == 0; }
   
    private:
     Object* data_{nullptr};
   
     explicit WeakObjectPtr(Object* data) : data_(data) {
       if (data_ != nullptr) {
         data_->IncWeakRef();
       }
     }
   
     template <typename>
     friend class WeakObjectPtr;
     friend struct tvm::ffi::details::ObjectUnsafe;
   };
   
   template <typename T, typename = void>
   class Optional;
   
   class ObjectRef {
    public:
     ObjectRef() = default;
     ObjectRef(const ObjectRef& other) = default;
     ObjectRef(ObjectRef&& other) = default;
     ObjectRef& operator=(const ObjectRef& other) = default;
     ObjectRef& operator=(ObjectRef&& other) = default;
     explicit ObjectRef(ObjectPtr<Object> data) : data_(data) {}
     explicit ObjectRef(UnsafeInit) : data_(nullptr) {}
     bool same_as(const ObjectRef& other) const { return data_ == other.data_; }
     bool operator==(const ObjectRef& other) const { return data_ == other.data_; }
     bool operator!=(const ObjectRef& other) const { return data_ != other.data_; }
     bool operator<(const ObjectRef& other) const { return data_.get() < other.data_.get(); }
     bool defined() const { return data_ != nullptr; }
     const Object* get() const { return data_.get(); }
     const Object* operator->() const { return get(); }
     bool unique() const { return data_.unique(); }
     int use_count() const { return data_.use_count(); }
   
     template <typename ObjectType, typename = std::enable_if_t<std::is_base_of_v<Object, ObjectType>>>
     const ObjectType* as() const {
       if (data_ != nullptr && data_->IsInstance<ObjectType>()) {
         return static_cast<ObjectType*>(data_.get());
       } else {
         return nullptr;
       }
     }
   
     template <typename ObjectRefType,
               typename = std::enable_if_t<std::is_base_of_v<ObjectRef, ObjectRefType>>>
     TVM_FFI_INLINE std::optional<ObjectRefType> as() const {
       if (data_ != nullptr) {
         if (data_->IsInstance<typename ObjectRefType::ContainerType>()) {
           ObjectRefType ref(UnsafeInit{});
           ref.data_ = data_;
           return ref;
         } else {
           return std::nullopt;
         }
       } else {
         return std::nullopt;
       }
     }
   
     int32_t type_index() const {
       return data_ != nullptr ? data_->type_index() : TypeIndex::kTVMFFINone;
     }
   
     std::string GetTypeKey() const {
       return data_ != nullptr ? data_->GetTypeKey() : StaticTypeKey::kTVMFFINone;
     }
   
     using ContainerType = Object;
     static constexpr bool _type_is_nullable = true;
   
    protected:
     ObjectPtr<Object> data_;
     Object* get_mutable() const { return data_.get(); }
     // friend classes.
     friend struct ObjectPtrHash;
     friend struct tvm::ffi::details::ObjectUnsafe;
   };
   
   // forward delcare variant
   template <typename... V>
   class Variant;
   
   struct ObjectPtrHash {
     size_t operator()(const ObjectRef& a) const { return operator()(a.data_); }
   
     template <typename T>
     size_t operator()(const ObjectPtr<T>& a) const {
       return std::hash<Object*>()(a.get());
     }
   
     template <typename... V>
     TVM_FFI_INLINE size_t operator()(const Variant<V...>& a) const;
   };
   
   struct ObjectPtrEqual {
     bool operator()(const ObjectRef& a, const ObjectRef& b) const { return a.same_as(b); }
   
     template <typename T>
     bool operator()(const ObjectPtr<T>& a, const ObjectPtr<T>& b) const {
       return a == b;
     }
   
     template <typename... V>
     TVM_FFI_INLINE bool operator()(const Variant<V...>& a, const Variant<V...>& b) const;
   };
   
   #define TVM_FFI_REGISTER_STATIC_TYPE_INFO(TypeName, ParentType)                               \
     static constexpr int32_t _type_depth = ParentType::_type_depth + 1;                         \
     static int32_t _GetOrAllocRuntimeTypeIndex() {                                              \
       static_assert(!ParentType::_type_final, "ParentType marked as final");                    \
       static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 ||   \
                         TypeName::_type_child_slots < ParentType::_type_child_slots,            \
                     "Need to set _type_child_slots when parent specifies it.");                 \
       TVMFFIByteArray type_key{TypeName::_type_key,                                             \
                                std::char_traits<char>::length(TypeName::_type_key)};            \
       static int32_t tindex = TVMFFITypeGetOrAllocIndex(                                        \
           &type_key, TypeName::_type_index, TypeName::_type_depth, TypeName::_type_child_slots, \
           TypeName::_type_child_slots_can_overflow, ParentType::_GetOrAllocRuntimeTypeIndex()); \
       return tindex;                                                                            \
     }                                                                                           \
     static inline int32_t _register_type_index = _GetOrAllocRuntimeTypeIndex()
   
   #define TVM_FFI_DECLARE_OBJECT_INFO_STATIC(TypeKey, TypeName, ParentType) \
     static constexpr const char* _type_key = TypeKey;                       \
     static int32_t RuntimeTypeIndex() { return TypeName::_type_index; }     \
     TVM_FFI_REGISTER_STATIC_TYPE_INFO(TypeName, ParentType)
   
   #define TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(TypeName, ParentType)                 \
     static constexpr int32_t _type_depth = ParentType::_type_depth + 1;                         \
     static int32_t _GetOrAllocRuntimeTypeIndex() {                                              \
       static_assert(!ParentType::_type_final, "ParentType marked as final");                    \
       static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 ||   \
                         TypeName::_type_child_slots < ParentType::_type_child_slots,            \
                     "Need to set _type_child_slots when parent specifies it.");                 \
       TVMFFIByteArray type_key{TypeName::_type_key,                                             \
                                std::char_traits<char>::length(TypeName::_type_key)};            \
       static int32_t tindex = TVMFFITypeGetOrAllocIndex(                                        \
           &type_key, -1, TypeName::_type_depth, TypeName::_type_child_slots,                    \
           TypeName::_type_child_slots_can_overflow, ParentType::_GetOrAllocRuntimeTypeIndex()); \
       return tindex;                                                                            \
     }                                                                                           \
     static int32_t RuntimeTypeIndex() { return _GetOrAllocRuntimeTypeIndex(); }                 \
     static inline int32_t _type_index = _GetOrAllocRuntimeTypeIndex()
   
   #define TVM_FFI_DECLARE_OBJECT_INFO(TypeKey, TypeName, ParentType) \
     static constexpr const char* _type_key = TypeKey;                \
     TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(TypeName, ParentType)
   
   #define TVM_FFI_DECLARE_OBJECT_INFO_FINAL(TypeKey, TypeName, ParentType) \
     static const constexpr int _type_child_slots [[maybe_unused]] = 0;     \
     static const constexpr bool _type_final [[maybe_unused]] = true;       \
     TVM_FFI_DECLARE_OBJECT_INFO(TypeKey, TypeName, ParentType)
   
   #define TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TypeName, ParentType, ObjectName)               \
     TypeName() = default;                                                                            \
     explicit TypeName(::tvm::ffi::ObjectPtr<ObjectName> n) : ParentType(n) {}                        \
     explicit TypeName(::tvm::ffi::UnsafeInit tag) : ParentType(tag) {}                               \
     TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName)                                            \
     using __PtrType = std::conditional_t<ObjectName::_type_mutable, ObjectName*, const ObjectName*>; \
     __PtrType operator->() const { return static_cast<__PtrType>(data_.get()); }                     \
     __PtrType get() const { return static_cast<__PtrType>(data_.get()); }                            \
     [[maybe_unused]] static constexpr bool _type_is_nullable = true;                                 \
     using ContainerType = ObjectName
   
   #define TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TypeName, ParentType, ObjectName)            \
     explicit TypeName(::tvm::ffi::UnsafeInit tag) : ParentType(tag) {}                               \
     TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName)                                            \
     using __PtrType = std::conditional_t<ObjectName::_type_mutable, ObjectName*, const ObjectName*>; \
     __PtrType operator->() const { return static_cast<__PtrType>(data_.get()); }                     \
     __PtrType get() const { return static_cast<__PtrType>(data_.get()); }                            \
     [[maybe_unused]] static constexpr bool _type_is_nullable = false;                                \
     using ContainerType = ObjectName
   
   namespace details {
   
   template <typename TargetType>
   TVM_FFI_INLINE bool IsObjectInstance(int32_t object_type_index) {
     static_assert(std::is_base_of_v<Object, TargetType>);
     // Everything is a subclass of object.
     if constexpr (std::is_same<TargetType, Object>::value) {
       return true;
     } else if constexpr (TargetType::_type_final) {
       // if the target type is a final type
       // then we only need to check the equivalence.
       return object_type_index == TargetType::RuntimeTypeIndex();
     } else {
       // Explicitly enclose in else to eliminate this branch early in compilation.
       // if target type is a non-leaf type
       // Check if type index falls into the range of reserved slots.
       int32_t target_type_index = TargetType::RuntimeTypeIndex();
       int32_t begin = target_type_index;
       // The condition will be optimized by constant-folding.
       if constexpr (TargetType::_type_child_slots != 0) {
         // total_slots = child_slots + 1 (including self)
         int32_t end = begin + TargetType::_type_child_slots + 1;
         if (object_type_index >= begin && object_type_index < end) return true;
       } else {
         if (object_type_index == begin) return true;
       }
       if constexpr (TargetType::_type_child_slots_can_overflow) {
         // Invariance: parent index is always smaller than the child.
         if (object_type_index < target_type_index) return false;
         // Do a runtime lookup of type information
         // the function checks that the info exists
         const TypeInfo* type_info = TVMFFIGetTypeInfo(object_type_index);
         return (type_info->type_depth > TargetType::_type_depth &&
                 type_info->type_acenstors[TargetType::_type_depth]->type_index == target_type_index);
       } else {
         return false;
       }
     }
   }
   
   struct ObjectUnsafe {
     // NOTE: get ffi header from an object
     TVM_FFI_INLINE static TVMFFIObject* GetHeader(const Object* src) {
       return const_cast<TVMFFIObject*>(&(src->header_));
     }
   
     template <typename Class>
     TVM_FFI_INLINE static int64_t GetObjectOffsetToSubclass() {
       return (reinterpret_cast<int64_t>(&(static_cast<Class*>(nullptr)->header_)) -
               reinterpret_cast<int64_t>(&(static_cast<Object*>(nullptr)->header_)));
     }
   
     template <typename T>
     TVM_FFI_INLINE static T ObjectRefFromObjectPtr(const ObjectPtr<Object>& ptr) {
       T ref(UnsafeInit{});
       ref.data_ = ptr;
       return ref;
     }
   
     template <typename T>
     TVM_FFI_INLINE static T ObjectRefFromObjectPtr(ObjectPtr<Object>&& ptr) {
       T ref(UnsafeInit{});
       ref.data_ = std::move(ptr);
       return ref;
     }
   
     template <typename T>
     TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromObjectRef(const ObjectRef& ref) {
       if constexpr (std::is_same_v<T, Object>) {
         return ref.data_;
       } else {
         return tvm::ffi::ObjectPtr<T>(ref.data_.data_);
       }
     }
   
     template <typename T>
     TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromObjectRef(ObjectRef&& ref) {
       if constexpr (std::is_same_v<T, Object>) {
         return std::move(ref.data_);
       } else {
         ObjectPtr<T> result;
         result.data_ = std::move(ref.data_.data_);
         ref.data_.data_ = nullptr;
         return result;
       }
     }
   
     template <typename T>
     TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromOwned(Object* raw_ptr) {
       tvm::ffi::ObjectPtr<T> ptr;
       ptr.data_ = raw_ptr;
       return ptr;
     }
   
     template <typename T>
     TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromOwned(TVMFFIObject* obj_ptr) {
       return ObjectPtrFromOwned<T>(reinterpret_cast<Object*>(obj_ptr));
     }
   
     template <typename T>
     TVM_FFI_INLINE static T* RawObjectPtrFromUnowned(TVMFFIObject* obj_ptr) {
       // NOTE: this is important to first cast to Object*
       // then cast back to T* because objptr and tptr may not be the same
       // depending on how sub-class allocates the space.
       return static_cast<T*>(reinterpret_cast<Object*>(obj_ptr));
     }
   
     // Create ObjectPtr from unowned ptr
     template <typename T>
     TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromUnowned(Object* raw_ptr) {
       return tvm::ffi::ObjectPtr<T>(raw_ptr);
     }
   
     template <typename T>
     TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromUnowned(TVMFFIObject* obj_ptr) {
       return tvm::ffi::ObjectPtr<T>(reinterpret_cast<Object*>(obj_ptr));
     }
   
     TVM_FFI_INLINE static void DecRefObjectHandle(TVMFFIObjectHandle handle) {
       reinterpret_cast<Object*>(handle)->DecRef();
     }
   
     TVM_FFI_INLINE static void IncRefObjectHandle(TVMFFIObjectHandle handle) {
       reinterpret_cast<Object*>(handle)->IncRef();
     }
   
     TVM_FFI_INLINE static Object* RawObjectPtrFromObjectRef(const ObjectRef& src) {
       return src.data_.data_;
     }
   
     TVM_FFI_INLINE static TVMFFIObject* TVMFFIObjectPtrFromObjectRef(const ObjectRef& src) {
       return GetHeader(src.data_.data_);
     }
   
     template <typename T>
     TVM_FFI_INLINE static TVMFFIObject* TVMFFIObjectPtrFromObjectPtr(const ObjectPtr<T>& src) {
       return GetHeader(src.data_);
     }
   
     template <typename T>
     TVM_FFI_INLINE static TVMFFIObject* MoveObjectPtrToTVMFFIObjectPtr(ObjectPtr<T>&& src) {
       Object* obj_ptr = src.data_;
       src.data_ = nullptr;
       return GetHeader(obj_ptr);
     }
   
     TVM_FFI_INLINE static TVMFFIObject* MoveObjectRefToTVMFFIObjectPtr(ObjectRef&& src) {
       Object* obj_ptr = src.data_.data_;
       src.data_.data_ = nullptr;
       return GetHeader(obj_ptr);
     }
   };
   }  // namespace details
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_OBJECT_H_
