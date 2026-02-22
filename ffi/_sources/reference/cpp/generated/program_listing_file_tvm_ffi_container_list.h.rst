
.. _program_listing_file_tvm_ffi_container_list.h:

Program Listing for File list.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_container_list.h>` (``tvm/ffi/container/list.h``)

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
   
   #ifndef TVM_FFI_CONTAINER_LIST_H_
   #define TVM_FFI_CONTAINER_LIST_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/seq_base.h>
   #include <tvm/ffi/object.h>
   
   #include <algorithm>
   #include <initializer_list>
   #include <sstream>
   #include <string>
   #include <type_traits>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   namespace ffi {
   
   class ListObj : public SeqBaseObj {
    public:
     static ObjectPtr<ListObj> CreateRepeated(int64_t n, const Any& val) {
       ObjectPtr<ListObj> p = ListObj::Empty(n);
       Any* itr = p->MutableBegin();
       for (int64_t& i = p->TVMFFISeqCell::size = 0; i < n; ++i) {
         new (itr++) Any(val);
       }
       return p;
     }
   
     static constexpr const int32_t _type_index = TypeIndex::kTVMFFIList;
     static const constexpr bool _type_final = true;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIList, ListObj, Object);
   
    private:
     void Reserve(int64_t n) {
       if (n <= TVMFFISeqCell::capacity) {
         return;
       }
       Any* old_data = MutableBegin();
       Any* new_data = static_cast<Any*>(::operator new(sizeof(Any) * static_cast<size_t>(n)));
       for (int64_t i = 0; i < TVMFFISeqCell::size; ++i) {
         new (new_data + i) Any(std::move(old_data[i]));
       }
       for (int64_t j = 0; j < TVMFFISeqCell::size; ++j) {
         (old_data + j)->Any::~Any();
       }
       data_deleter(data);
       data = new_data;
       TVMFFISeqCell::capacity = n;
     }
   
     static ObjectPtr<ListObj> Empty(int64_t n = kInitSize) {
       if (n < 0) {
         TVM_FFI_THROW(ValueError) << "cannot construct a List of negative size";
       }
       ObjectPtr<ListObj> p = make_object<ListObj>();
       p->TVMFFISeqCell::capacity = n;
       p->TVMFFISeqCell::size = 0;
       p->data = n == 0 ? nullptr : static_cast<void*>(::operator new(sizeof(Any) * n));
       p->data_deleter = RawDataDeleter;
       return p;
     }
   
     static void RawDataDeleter(void* data) { ::operator delete(data); }
   
     static constexpr int64_t kInitSize = 4;
     static constexpr int64_t kIncFactor = 2;
   
     template <typename, typename>
     friend class List;
   
     template <typename, typename>
     friend struct TypeTraits;
   };
   
   template <typename T, typename = typename std::enable_if_t<details::storage_enabled_v<T>>>
   class List : public ObjectRef {
    public:
     using value_type = T;
   
     explicit List(UnsafeInit tag) : ObjectRef(tag) {}
     List() { data_ = ListObj::Empty(0); }  // NOLINT(modernize-use-equals-default)
     List(List<T>&& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(std::move(other.data_)) {}
     List(const List<T>& other) : ObjectRef(other.data_) {}  // NOLINT(google-explicit-constructor)
   
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     List(List<U>&& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(std::move(other.data_)) {}
   
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     List(const List<U>& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(other.data_) {}
   
     TVM_FFI_INLINE List<T>& operator=(List<T>&& other) {
       data_ = std::move(other.data_);
       return *this;
     }
   
     TVM_FFI_INLINE List<T>& operator=(const List<T>& other) {
       data_ = other.data_;
       return *this;
     }
   
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     TVM_FFI_INLINE List<T>& operator=(List<U>&& other) {
       data_ = std::move(other.data_);
       return *this;
     }
   
     template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
     TVM_FFI_INLINE List<T>& operator=(const List<U>& other) {
       data_ = other.data_;
       return *this;
     }
   
     explicit List(ObjectPtr<Object> n) : ObjectRef(std::move(n)) {}
   
     template <typename IterType>
     List(IterType first, IterType last) {  // NOLINT(performance-unnecessary-value-param)
       static_assert(is_valid_iterator_v<T, IterType>,
                     "IterType cannot be inserted into a tvm::List<T>");
       Assign(first, last);
     }
   
     List(std::initializer_list<T> init) {  // NOLINT(*)
       Assign(init.begin(), init.end());
     }
   
     List(const std::vector<T>& init) {  // NOLINT(*)
       Assign(init.begin(), init.end());
     }
   
     explicit List(const size_t n, const T& val) { data_ = ListObj::CreateRepeated(n, val); }
   
    public:
     // iterators
     struct ValueConverter {
       using ResultType = T;
       static T convert(const Any& n) { return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(n); }
     };
   
     using iterator = details::IterAdapter<ValueConverter, const Any*>;
     using reverse_iterator = details::ReverseIterAdapter<ValueConverter, const Any*>;
   
     iterator begin() const { return iterator(GetListObj()->begin()); }
     iterator end() const { return iterator(GetListObj()->end()); }
     reverse_iterator rbegin() const { return reverse_iterator(GetListObj()->end() - 1); }
     reverse_iterator rend() const { return reverse_iterator(GetListObj()->begin() - 1); }
   
    public:
     // const methods in std::vector
     T operator[](int64_t i) const {
       ListObj* p = GetListObj();
       if (p == nullptr) {
         TVM_FFI_THROW(IndexError) << "cannot index a null list";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(p->at(i));
     }
   
     size_t size() const {
       ListObj* p = GetListObj();
       return p == nullptr ? 0 : p->size();
     }
   
     size_t capacity() const {
       ListObj* p = GetListObj();
       return p == nullptr ? 0 : p->SeqBaseObj::capacity();
     }
   
     bool empty() const { return size() == 0; }
   
     T front() const {
       ListObj* p = GetListObj();
       if (p == nullptr) {
         TVM_FFI_THROW(IndexError) << "cannot index a null list";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(p->front());
     }
   
     T back() const {
       ListObj* p = GetListObj();
       if (p == nullptr) {
         TVM_FFI_THROW(IndexError) << "cannot index a null list";
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(p->back());
     }
   
    public:
     // mutation in std::vector
     void push_back(const T& item) {
       ListObj* p = EnsureCapacity(1);
       p->EmplaceInit(p->TVMFFISeqCell::size++, item);
     }
   
     template <typename... Args>
     void emplace_back(Args&&... args) {
       ListObj* p = EnsureCapacity(1);
       p->EmplaceInit(p->TVMFFISeqCell::size++, std::forward<Args>(args)...);
     }
   
     void insert(iterator position, const T& val) {
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot insert to a null list";
       }
       int64_t idx = std::distance(begin(), position);
       EnsureCapacity(1)->insert(idx, Any(val));
     }
   
     template <typename IterType>
     void insert(iterator position, IterType first, IterType last) {
       static_assert(is_valid_iterator_v<T, IterType>,
                     "IterType cannot be inserted into a tvm::List<T>");
       if (first == last) return;
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot insert to a null list";
       }
       int64_t idx = std::distance(begin(), position);
       int64_t numel = std::distance(first, last);
       EnsureCapacity(numel)->insert(idx, first, last);
     }
   
     void pop_back() {
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot pop_back a null list";
       }
       GetListObj()->pop_back();
     }
   
     void erase(iterator position) {
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot erase a null list";
       }
       int64_t idx = std::distance(begin(), position);
       GetListObj()->erase(idx);
     }
   
     void erase(iterator first, iterator last) {
       if (first == last) return;
       if (data_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "cannot erase a null list";
       }
       int64_t st = std::distance(begin(), first);
       int64_t ed = std::distance(begin(), last);
       GetListObj()->erase(st, ed);
     }
   
     void resize(int64_t n) {
       if (n < 0) {
         TVM_FFI_THROW(ValueError) << "cannot resize a List to negative size";
       }
       EnsureCapacity(std::max<int64_t>(0, n - static_cast<int64_t>(size())))->resize(n);
     }
   
     void reserve(int64_t n) { EnsureListObj()->Reserve(n); }
   
     void clear() {
       if (data_ != nullptr) {
         GetListObj()->clear();
       }
     }
   
    public:
     // List's own methods
     void Set(int64_t i, T value) { EnsureListObj()->SetItem(i, std::move(value)); }
   
     ListObj* GetListObj() const { return static_cast<ListObj*>(data_.get()); }
   
     template <typename IterType>
     void Assign(IterType first, IterType last) {  // NOLINT(performance-unnecessary-value-param)
       int64_t cap = std::distance(first, last);
       if (cap < 0) {
         TVM_FFI_THROW(ValueError) << "cannot construct a List of negative size";
       }
       ListObj* p = EnsureListObj();
       p->Reserve(cap);
       p->clear();
       Any* itr = p->MutableBegin();
       for (int64_t& i = p->TVMFFISeqCell::size = 0; i < cap; ++i, ++first, ++itr) {
         new (itr) Any(*first);
       }
     }
   
     using ContainerType = ListObj;
   
    private:
     ListObj* EnsureCapacity(int64_t reserve_extra) {
       ListObj* p = EnsureListObj();
       if (p->TVMFFISeqCell::capacity >= p->TVMFFISeqCell::size + reserve_extra) {
         return p;
       }
       int64_t cap = p->TVMFFISeqCell::capacity * ListObj::kIncFactor;
       cap = std::max(cap, p->TVMFFISeqCell::size + reserve_extra);
       p->Reserve(cap);
       return p;
     }
   
     ListObj* EnsureListObj() {
       if (data_ == nullptr) {
         data_ = ListObj::Empty();
       }
       return static_cast<ListObj*>(data_.get());
     }
   
     template <typename, typename>
     friend class List;
   };
   
   // Traits for List
   template <typename T>
   inline constexpr bool use_default_type_traits_v<List<T>> = false;
   
   template <typename T>
   struct TypeTraits<List<T>> : public SeqTypeTraitsBase<TypeTraits<List<T>>, List<T>, T> {
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIList;
     static constexpr int32_t kPrimaryTypeIndex = TypeIndex::kTVMFFIList;
     static constexpr int32_t kOtherTypeIndex = TypeIndex::kTVMFFIArray;
     static constexpr const char* kTypeName = "List";
     static constexpr const char* kStaticTypeKey = StaticTypeKey::kTVMFFIList;
   
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
   inline constexpr bool type_contains_v<List<T>, List<U>> = type_contains_v<T, U>;
   }  // namespace details
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_CONTAINER_LIST_H_
