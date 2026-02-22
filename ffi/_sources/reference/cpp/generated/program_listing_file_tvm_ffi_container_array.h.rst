
.. _program_listing_file_tvm_ffi_container_array.h:

Program Listing for File array.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_container_array.h>` (``tvm/ffi/container/array.h``)

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
   
   #ifndef TVM_FFI_CONTAINER_ARRAY_H_
   #define TVM_FFI_CONTAINER_ARRAY_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/container_details.h>
   #include <tvm/ffi/container/seq_base.h>
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/optional.h>
   
   #include <algorithm>
   #include <string>
   #include <type_traits>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   namespace ffi {
   
   class ArrayObj : public SeqBaseObj {
    public:
     static ObjectPtr<ArrayObj> CopyFrom(int64_t cap, ArrayObj* from) {
       int64_t size = from->TVMFFISeqCell::size;
       if (size > cap) {
         TVM_FFI_THROW(ValueError) << "Not enough capacity";
       }
       ObjectPtr<ArrayObj> p = ArrayObj::Empty(cap);
       Any* write = p->MutableBegin();
       Any* read = from->MutableBegin();
       // To ensure exception safety, size is only incremented after the initialization succeeds
       for (int64_t& i = p->TVMFFISeqCell::size = 0; i < size; ++i) {
         new (write++) Any(*read++);
       }
       return p;
     }
   
     static ObjectPtr<ArrayObj> MoveFrom(int64_t cap, ArrayObj* from) {
       int64_t size = from->TVMFFISeqCell::size;
       if (size > cap) {
         TVM_FFI_THROW(RuntimeError) << "Not enough capacity";
       }
       ObjectPtr<ArrayObj> p = ArrayObj::Empty(cap);
       Any* write = p->MutableBegin();
       Any* read = from->MutableBegin();
       // To ensure exception safety, size is only incremented after the initialization succeeds
       for (int64_t& i = p->TVMFFISeqCell::size = 0; i < size; ++i) {
         new (write++) Any(std::move(*read++));
       }
       from->TVMFFISeqCell::size = 0;
       return p;
     }
   
     static ObjectPtr<ArrayObj> CreateRepeated(int64_t n, const Any& val) {
       ObjectPtr<ArrayObj> p = ArrayObj::Empty(n);
       Any* itr = p->MutableBegin();
       for (int64_t& i = p->TVMFFISeqCell::size = 0; i < n; ++i) {
         new (itr++) Any(val);
       }
       return p;
     }
   
     static constexpr const int32_t _type_index = TypeIndex::kTVMFFIArray;
     static const constexpr bool _type_final = true;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIArray, ArrayObj, Object);
   
    private:
     size_t GetSize() const { return TVMFFISeqCell::size; }
   
     static ObjectPtr<ArrayObj> Empty(int64_t n = kInitSize) {
       ObjectPtr<ArrayObj> p = make_inplace_array_object<ArrayObj, Any>(n);
       p->TVMFFISeqCell::capacity = n;
       p->TVMFFISeqCell::size = 0;
       p->data = reinterpret_cast<char*>(p.get()) + sizeof(ArrayObj);
       p->data_deleter = nullptr;
       return p;
     }
   
     template <typename IterType>
     ArrayObj* InitRange(int64_t idx, IterType first, IterType last) {
       Any* itr = MutableBegin() + idx;
       for (; first != last; ++first) {
         Any ref = *first;
         new (itr++) Any(std::move(ref));
       }
       return this;
     }
   
     static constexpr int64_t kInitSize = 4;
   
     static constexpr int64_t kIncFactor = 2;
   
     // Reference class
     template <typename, typename>
     friend class Array;
   
     template <typename... Types>
     friend class Tuple;
   
     template <typename, typename>
     friend struct TypeTraits;
   
     // To specialize make_object<ArrayObj>
     friend ObjectPtr<ArrayObj> make_object<>();
   };
   
   template <typename T, typename IterType>
   struct is_valid_iterator
       : std::bool_constant<
             std::is_same_v<
                 T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<IterType>())>>> ||
             std::is_base_of_v<
                 T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<IterType>())>>>> {
   };
   
   template <typename T, typename IterType>
   struct is_valid_iterator<Optional<T>, IterType> : is_valid_iterator<T, IterType> {};
   
   template <typename IterType>
   struct is_valid_iterator<Any, IterType> : std::true_type {};
   
   template <typename T, typename IterType>
   inline constexpr bool is_valid_iterator_v = is_valid_iterator<T, IterType>::value;
   
   template <typename T, typename = typename std::enable_if_t<details::storage_enabled_v<T>>>
   class Array : public ObjectRef {
    public:
     using value_type = T;
     // constructors
     explicit Array(UnsafeInit tag) : ObjectRef(tag) {}
     Array() { data_ = ArrayObj::Empty(); }  // NOLINT(modernize-use-equals-default)
     Array(Array<T>&& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(std::move(other.data_)) {}
     Array(const Array<T>& other) : ObjectRef(other.data_) {}  // NOLINT(google-explicit-constructor)
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     Array(Array<U>&& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(std::move(other.data_)) {}
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     Array(const Array<U>& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(other.data_) {}
   
     TVM_FFI_INLINE Array<T>& operator=(Array<T>&& other) {
       data_ = std::move(other.data_);
       return *this;
     }
     TVM_FFI_INLINE Array<T>& operator=(const Array<T>& other) {
       data_ = other.data_;
       return *this;
     }
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     TVM_FFI_INLINE Array<T>& operator=(Array<U>&& other) {
       data_ = std::move(other.data_);
       return *this;
     }
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     TVM_FFI_INLINE Array<T>& operator=(const Array<U>& other) {
       data_ = other.data_;
       return *this;
     }
   
     explicit Array(ObjectPtr<Object> n) : ObjectRef(std::move(n)) {}
   
     template <typename IterType>
     Array(IterType first, IterType last) {  // NOLINT(performance-unnecessary-value-param)
       static_assert(is_valid_iterator_v<T, IterType>,
                     "IterType cannot be inserted into a tvm::Array<T>");
       Assign(first, last);
     }
   
     Array(std::initializer_list<T> init) {  // NOLINT(*)
       Assign(init.begin(), init.end());
     }
   
     Array(const std::vector<T>& init) {  // NOLINT(*)
       Assign(init.begin(), init.end());
     }
   
     explicit Array(const size_t n, const T& val) { data_ = ArrayObj::CreateRepeated(n, val); }
   
    public:
     // iterators
     struct ValueConverter {
       using ResultType = T;
       static T convert(const Any& n) { return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(n); }
     };
   
     using iterator = details::IterAdapter<ValueConverter, const Any*>;
     using reverse_iterator = details::ReverseIterAdapter<ValueConverter, const Any*>;
   
     iterator begin() const { return iterator(GetArrayObj()->begin()); }
   
     iterator end() const { return iterator(GetArrayObj()->end()); }
   
     reverse_iterator rbegin() const {
       // ArrayObj::end() is never nullptr
       return reverse_iterator(GetArrayObj()->end() - 1);
     }
   
     reverse_iterator rend() const {
       // ArrayObj::begin() is never nullptr
       return reverse_iterator(GetArrayObj()->begin() - 1);
     }
   
    public:
     // const methods in std::vector
     const T operator[](int64_t i) const {
       ArrayObj* p = GetArrayObj();
       if (p == nullptr) {
         TVM_FFI_THROW(IndexError) << "cannot index a null array";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(p->at(i));
     }
   
     size_t size() const {
       ArrayObj* p = GetArrayObj();
       return p == nullptr ? 0 : p->size();
     }
   
     size_t capacity() const {
       ArrayObj* p = GetArrayObj();
       return p == nullptr ? 0 : p->SeqBaseObj::capacity();
     }
   
     bool empty() const { return size() == 0; }
   
     T front() const {
       ArrayObj* p = GetArrayObj();
       if (p == nullptr) {
         TVM_FFI_THROW(IndexError) << "cannot index a null array";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(p->front());
     }
   
     T back() const {
       ArrayObj* p = GetArrayObj();
       if (p == nullptr) {
         TVM_FFI_THROW(IndexError) << "cannot index a null array";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(p->back());
     }
   
    public:
     // mutation in std::vector, implements copy-on-write
     void push_back(const T& item) {
       ArrayObj* p = CopyOnWrite(1);
       p->EmplaceInit(p->TVMFFISeqCell::size++, item);
     }
   
     template <typename... Args>
     void emplace_back(Args&&... args) {
       ArrayObj* p = CopyOnWrite(1);
       p->EmplaceInit(p->TVMFFISeqCell::size++, std::forward<Args>(args)...);
     }
   
     void insert(iterator position, const T& val) {
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot insert a null array";
       }
       int64_t idx = std::distance(begin(), position);
       CopyOnWrite(1)->insert(idx, Any(val));
     }
   
     template <typename IterType>
     void insert(iterator position, IterType first, IterType last) {
       static_assert(is_valid_iterator_v<T, IterType>,
                     "IterType cannot be inserted into a tvm::Array<T>");
       if (first == last) return;
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot insert a null array";
       }
       int64_t idx = std::distance(begin(), position);
       int64_t numel = std::distance(first, last);
       CopyOnWrite(numel)->insert(idx, first, last);
     }
   
     void pop_back() {
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot pop_back a null array";
       }
       CopyOnWrite()->pop_back();
     }
   
     void erase(iterator position) {
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot erase a null array";
       }
       int64_t idx = std::distance(begin(), position);
       CopyOnWrite()->erase(idx);
     }
   
     void erase(iterator first, iterator last) {
       if (first == last) return;
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot erase a null array";
       }
       int64_t st = std::distance(begin(), first);
       int64_t ed = std::distance(begin(), last);
       CopyOnWrite()->erase(st, ed);
     }
   
     void resize(int64_t n) {
       if (n < 0) {
         TVM_FFI_THROW(ValueError) << "cannot resize an Array to negative size";
       }
       if (data_ == nullptr) {
         SwitchContainer(n);
         return;
       }
       int64_t cur_size = GetArrayObj()->TVMFFISeqCell::size;
       if (cur_size < n) {
         CopyOnWrite(n - cur_size)->resize(n);
       } else if (cur_size > n) {
         CopyOnWrite()->resize(n);
       }
     }
   
     void reserve(int64_t n) {
       if (data_ == nullptr || n > static_cast<int64_t>(GetArrayObj()->SeqBaseObj::capacity())) {
         SwitchContainer(n);
       }
     }
   
     void clear() {
       if (data_ != nullptr) {
         ArrayObj* p = CopyOnWrite();
         p->clear();
       }
     }
     template <typename... Args>
     static size_t CalcCapacityImpl() {
       return 0;
     }
   
     template <typename... Args>
     static size_t CalcCapacityImpl(Array<T> value, Args... args) {
       return value.size() + CalcCapacityImpl(args...);
     }
   
     template <typename... Args>
     static size_t CalcCapacityImpl(T value, Args... args) {
       return 1 + CalcCapacityImpl(args...);
     }
   
     template <typename... Args>
     static void AgregateImpl(Array<T>& dest) {}  // NOLINT(*)
   
     template <typename... Args>
     static void AgregateImpl(Array<T>& dest, Array<T> value, Args... args) {  // NOLINT(*)
       dest.insert(dest.end(), value.begin(), value.end());
       AgregateImpl(dest, args...);
     }
   
     template <typename... Args>
     static void AgregateImpl(Array<T>& dest, T value, Args... args) {  // NOLINT(*)
       dest.push_back(value);
       AgregateImpl(dest, args...);
     }
   
    public:
     // Array's own methods
   
     void Set(int64_t i, T value) { CopyOnWrite()->SetItem(i, std::move(value)); }
   
     ArrayObj* GetArrayObj() const { return static_cast<ArrayObj*>(data_.get()); }
   
     template <typename F, typename U = std::invoke_result_t<F, T>>
     Array<U> Map(F fmap) const {
       return Array<U>(MapHelper(data_, fmap));
     }
   
     template <typename F, typename = std::enable_if_t<std::is_same_v<T, std::invoke_result_t<F, T>>>>
     void MutateByApply(F fmutate) {
       data_ = MapHelper(std::move(data_), fmutate);
     }
   
     template <typename IterType>
     void Assign(IterType first, IterType last) {  // NOLINT(performance-unnecessary-value-param)
       int64_t cap = std::distance(first, last);
       if (cap < 0) {
         TVM_FFI_THROW(ValueError) << "cannot construct an Array of negative size";
       }
       ArrayObj* p = GetArrayObj();
       if (p != nullptr && data_.unique() && p->TVMFFISeqCell::capacity >= cap) {
         // do not have to make new space
         p->clear();
       } else {
         // create new space
         data_ = ArrayObj::Empty(cap);
         p = GetArrayObj();
       }
       // To ensure exception safety, size is only incremented after the initialization succeeds
       Any* itr = p->MutableBegin();
       for (int64_t& i = p->TVMFFISeqCell::size = 0; i < cap; ++i, ++first, ++itr) {
         new (itr) Any(*first);
       }
     }
   
     ArrayObj* CopyOnWrite() {
       if (data_ == nullptr) {
         return SwitchContainer(ArrayObj::kInitSize);
       }
       if (!data_.unique()) {
         return SwitchContainer(capacity());
       }
       return static_cast<ArrayObj*>(data_.get());
     }
   
     using ContainerType = ArrayObj;
   
     template <typename... Args>
     static Array<T> Agregate(Args... args) {
       Array<T> result;
       result.reserve(CalcCapacityImpl(args...));
       AgregateImpl(result, args...);
       return result;
     }
   
    private:
     ArrayObj* CopyOnWrite(int64_t reserve_extra) {
       ArrayObj* p = GetArrayObj();
       if (p == nullptr) {
         // necessary to get around the constexpr address issue before c++17
         const int64_t kInitSize = ArrayObj::kInitSize;
         return SwitchContainer(std::max(kInitSize, reserve_extra));
       }
       if (p->TVMFFISeqCell::capacity >= p->TVMFFISeqCell::size + reserve_extra) {
         return CopyOnWrite();
       }
       int64_t cap = p->TVMFFISeqCell::capacity * ArrayObj::kIncFactor;
       cap = std::max(cap, p->TVMFFISeqCell::size + reserve_extra);
       return SwitchContainer(cap);
     }
   
     ArrayObj* SwitchContainer(int64_t capacity) {
       if (data_ == nullptr) {
         data_ = ArrayObj::Empty(capacity);
       } else if (data_.unique()) {
         data_ = ArrayObj::MoveFrom(capacity, GetArrayObj());
       } else {
         data_ = ArrayObj::CopyFrom(capacity, GetArrayObj());
       }
       return static_cast<ArrayObj*>(data_.get());
     }
   
     template <typename F, typename U = std::invoke_result_t<F, T>>
     static ObjectPtr<Object> MapHelper(ObjectPtr<Object> data, F fmap) {
       if (data == nullptr) {
         return nullptr;
       }
   
       TVM_FFI_ICHECK(data->IsInstance<ArrayObj>());
   
       constexpr bool is_same_output_type = std::is_same_v<T, U>;
   
       if constexpr (is_same_output_type) {
         if (data.unique()) {
           // Mutate-in-place path.  Only allowed if the output type U is
           // the same as type T, we have a mutable this*, and there are
           // no other shared copies of the array.
           auto arr = static_cast<ArrayObj*>(data.get());
           for (auto it = arr->MutableBegin(); it != arr->MutableEnd(); it++) {
             T value = details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(*it);
             // reset the original value to nullptr, to ensure unique ownership
             it->reset();
             T mapped = fmap(std::move(value));
             *it = std::move(mapped);
           }
           return data;
         }
       }
   
       constexpr bool compatible_types = is_valid_iterator_v<T, U*> || is_valid_iterator_v<U, T*>;
   
       ObjectPtr<ArrayObj> output = nullptr;
       auto arr = static_cast<ArrayObj*>(data.get());
   
       auto it = arr->begin();
       if constexpr (compatible_types) {
         // Copy-on-write path, if the output Array<U> might be
         // represented by the same underlying array as the existing
         // Array<T>.  Typically, this is for functions that map `T` to
         // `T`, but can also apply to functions that map `T` to
         // `Optional<T>`, or that map `T` to a subclass or superclass of
         // `T`.
         bool all_identical = true;
         for (; it != arr->end(); it++) {
           U mapped = fmap(details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(*it));
           if (!(*it).same_as(mapped)) {
             // At least one mapped element is different than the
             // original.  Therefore, prepare the output array,
             // consisting of any previous elements that had mapped to
             // themselves (if any), and the element that didn't map to
             // itself.
             //
             // We cannot use `U()` as the default object, as `U` may be
             // a non-nullable type.  Since the default `Any()`
             // will be overwritten before returning, all objects will be
             // of type `U` for the calling scope.
             all_identical = false;
             output = ArrayObj::CreateRepeated(static_cast<int64_t>(arr->size()), Any());
             output->InitRange(0, arr->begin(), it);
             output->SetItem(it - arr->begin(), std::move(mapped));
             it++;
             break;
           }
         }
         if (all_identical) {
           return data;
         }
       } else {
         // Path for incompatible types.  The constexpr check for
         // compatible types isn't strictly necessary, as the first
         // (*it).same_as(mapped) would return false, but we might as well
         // avoid it altogether.
         //
         // We cannot use `U()` as the default object, as `U` may be a
         // non-nullable type.  Since the default `Any()` will be
         // overwritten before returning, all objects will be of type `U`
         // for the calling scope.
         output = ArrayObj::CreateRepeated(static_cast<int64_t>(arr->size()), Any());
       }
   
       // Normal path for incompatible types, or post-copy path for
       // copy-on-write instances.
       //
       // If the types are incompatible, then at this point `output` is
       // empty, and `it` points to the first element of the input.
       //
       // If the types were compatible, then at this point `output`
       // contains zero or more elements that mapped to themselves
       // followed by the first element that does not map to itself, and
       // `it` points to the element just after the first element that
       // does not map to itself.  Because at least one element has been
       // changed, we no longer have the opportunity to avoid a copy, so
       // we don't need to check the result.
       //
       // In both cases, `it` points to the next element to be processed,
       // so we can either start or resume the iteration from that point,
       // with no further checks on the result.
       for (; it != arr->end(); it++) {
         U mapped = fmap(details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(*it));
         output->SetItem(it - arr->begin(), std::move(mapped));
       }
   
       return output;
     }
     template <typename, typename>
     friend class Array;
   };
   
   template <typename T, typename = typename std::enable_if_t<std::is_same_v<T, Any> ||
                                                              TypeTraits<T>::convert_enabled>>
   inline Array<T> Concat(Array<T> lhs, const Array<T>& rhs) {
     for (const auto& x : rhs) {
       lhs.push_back(x);
     }
     return std::move(lhs);
   }
   
   template <>
   inline ObjectPtr<ArrayObj> make_object() {
     return ArrayObj::Empty();
   }
   
   // Traits for Array
   template <typename T>
   inline constexpr bool use_default_type_traits_v<Array<T>> = false;
   
   template <typename T>
   struct TypeTraits<Array<T>> : public SeqTypeTraitsBase<TypeTraits<Array<T>>, Array<T>, T> {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;
     static constexpr int32_t kPrimaryTypeIndex = TypeIndex::kTVMFFIArray;
     static constexpr int32_t kOtherTypeIndex = TypeIndex::kTVMFFIList;
     static constexpr const char* kTypeName = "Array";
     static constexpr const char* kStaticTypeKey = StaticTypeKey::kTVMFFIArray;
   
     TVM_FFI_INLINE static std::string TypeSchema() {
       std::ostringstream oss;
       oss << R"({"type":")" << kStaticTypeKey << R"(","args":[)";
       oss << details::TypeSchema<T>::v();
       oss << "]}";
       return oss.str();
     }
   };
   
   namespace details {
   template <typename T, typename U>
   inline constexpr bool type_contains_v<Array<T>, Array<U>> = type_contains_v<T, U>;
   }  // namespace details
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_CONTAINER_ARRAY_H_
