
.. _program_listing_file_tvm_ffi_container_dict.h:

Program Listing for File dict.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_container_dict.h>` (``tvm/ffi/container/dict.h``)

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
   
   #ifndef TVM_FFI_CONTAINER_DICT_H_
   #define TVM_FFI_CONTAINER_DICT_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/container_details.h>
   #include <tvm/ffi/container/map_base.h>
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/optional.h>
   
   #include <unordered_map>
   
   namespace tvm {
   namespace ffi {
   
   class DictObj : public MapBaseObj {
    public:
     static constexpr const int32_t _type_index = TypeIndex::kTVMFFIDict;
     static const constexpr bool _type_final = true;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIDict, DictObj, Object);
   
    protected:
     template <typename, typename, typename>
     friend class Dict;
   };
   
   static_assert(sizeof(DictObj) == sizeof(MapBaseObj), "DictObj must match MapBaseObj layout");
   
   template <typename K, typename V,
             typename = typename std::enable_if_t<details::storage_enabled_v<K> &&
                                                  details::storage_enabled_v<V>>>
   class Dict : public ObjectRef {
    public:
     using key_type = K;
     using mapped_type = V;
     class iterator;
     explicit Dict(UnsafeInit tag) : ObjectRef(tag) {}
     Dict() { data_ = DictObj::Empty<DictObj>(); }
     Dict(Dict<K, V>&& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(std::move(other.data_)) {}
     Dict(const Dict<K, V>& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(other.data_) {}
   
     template <typename KU, typename VU,
               typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                           details::type_contains_v<V, VU>>>
     Dict(Dict<KU, VU>&& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(std::move(other.data_)) {}
   
     template <typename KU, typename VU,
               typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                           details::type_contains_v<V, VU>>>
     // NOLINTNEXTLINE(google-explicit-constructor)
     Dict(const Dict<KU, VU>& other) : ObjectRef(other.data_) {}
   
     Dict<K, V>& operator=(Dict<K, V>&& other) {
       data_ = std::move(other.data_);
       return *this;
     }
   
     Dict<K, V>& operator=(const Dict<K, V>& other) {
       data_ = other.data_;
       return *this;
     }
   
     template <typename KU, typename VU,
               typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                           details::type_contains_v<V, VU>>>
     Dict<K, V>& operator=(Dict<KU, VU>&& other) {
       data_ = std::move(other.data_);
       return *this;
     }
   
     template <typename KU, typename VU,
               typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                           details::type_contains_v<V, VU>>>
     Dict<K, V>& operator=(const Dict<KU, VU>& other) {
       data_ = other.data_;
       return *this;
     }
     explicit Dict(ObjectPtr<Object> n) : ObjectRef(n) {}
     template <typename IterType>
     Dict(IterType begin, IterType end) {
       data_ = DictObj::CreateFromRange<DictObj>(begin, end);
     }
     Dict(std::initializer_list<std::pair<K, V>> init) {
       data_ = DictObj::CreateFromRange<DictObj>(init.begin(), init.end());
     }
     template <typename Hash, typename Equal>
     Dict(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
       data_ = DictObj::CreateFromRange<DictObj>(init.begin(), init.end());
     }
     V at(const K& key) const {
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(GetDictObj()->at(key));
     }
     V operator[](const K& key) const { return this->at(key); }
     size_t size() const {
       DictObj* n = GetDictObj();
       return n == nullptr ? 0 : n->size();
     }
     size_t count(const K& key) const {
       DictObj* n = GetDictObj();
       return n == nullptr ? 0 : n->count(key);
     }
     bool empty() const { return size() == 0; }
     void clear() {
       DictObj* n = GetDictObj();
       if (n != nullptr) {
         n->clear();
       }
     }
     void Set(const K& key, const V& value) {
       EnsureDictObj();
       ObjectPtr<Object> new_container =
           MapBaseObj::InsertMaybeReHash<DictObj>(DictObj::KVType(key, value), data_);
       if (new_container != nullptr) {
         static_cast<MapBaseObj*>(data_.get())->InplaceSwitchTo(std::move(new_container));
       }
     }
     iterator begin() const { return iterator(GetDictObj()->begin()); }
     iterator end() const { return iterator(GetDictObj()->end()); }
     iterator find(const K& key) const { return iterator(GetDictObj()->find(key)); }
     std::optional<V> Get(const K& key) const {
       DictObj::iterator iter = GetDictObj()->find(key);
       if (iter == GetDictObj()->end()) {
         return std::nullopt;
       }
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(iter->second);
     }
   
     void erase(const K& key) {
       DictObj* n = GetDictObj();
       if (n != nullptr) {
         n->erase(key);
       }
     }
   
     using ContainerType = DictObj;
   
   
     class iterator {
      public:
       using iterator_category = std::bidirectional_iterator_tag;
       using difference_type = int64_t;
       using value_type = const std::pair<K, V>;
       using pointer = value_type*;
       using reference = value_type;
   
       iterator() : itr() {}
   
       bool operator==(const iterator& other) const { return itr == other.itr; }
       bool operator!=(const iterator& other) const { return itr != other.itr; }
       pointer operator->() const = delete;
       reference operator*() const {
         auto& kv = *itr;
         return std::make_pair(details::AnyUnsafe::CopyFromAnyViewAfterCheck<K>(kv.first),
                               details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(kv.second));
       }
       iterator& operator++() {
         ++itr;
         return *this;
       }
       iterator operator++(int) {
         iterator copy = *this;
         ++(*this);
         return copy;
       }
   
       iterator& operator--() {
         --itr;
         return *this;
       }
       iterator operator--(int) {
         iterator copy = *this;
         --(*this);
         return copy;
       }
   
      private:
       iterator(const DictObj::iterator& itr)  // NOLINT(*)
           : itr(itr) {}
   
       template <typename, typename, typename>
       friend class Dict;
   
       DictObj::iterator itr;
     };
   
    private:
     DictObj* GetDictObj() const { return static_cast<DictObj*>(data_.get()); }
   
     void EnsureDictObj() {
       if (data_ == nullptr) {
         data_ = DictObj::Empty<DictObj>();
       }
     }
   
     template <typename, typename, typename>
     friend class Dict;
   };
   
   // Traits for Dict
   template <typename K, typename V>
   inline constexpr bool use_default_type_traits_v<Dict<K, V>> = false;
   
   template <typename K, typename V>
   struct TypeTraits<Dict<K, V>> : public MapTypeTraitsBase<TypeTraits<Dict<K, V>>, Dict<K, V>, K, V> {
     static constexpr int32_t kPrimaryTypeIndex = TypeIndex::kTVMFFIDict;
     static constexpr int32_t kOtherTypeIndex = TypeIndex::kTVMFFIMap;
     static constexpr const char* kTypeName = "Dict";
   
     TVM_FFI_INLINE static std::string TypeSchema() {
       std::ostringstream oss;
       oss << R"({"type":")" << StaticTypeKey::kTVMFFIDict << R"(","args":[)";
       oss << details::TypeSchema<K>::v() << ",";
       oss << details::TypeSchema<V>::v();
       oss << "]}";
       return oss.str();
     }
   };
   
   namespace details {
   template <typename K, typename V, typename KU, typename VU>
   inline constexpr bool type_contains_v<Dict<K, V>, Dict<KU, VU>> =
       type_contains_v<K, KU> && type_contains_v<V, VU>;
   }  // namespace details
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_CONTAINER_DICT_H_
