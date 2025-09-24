
.. _program_listing_file_tvm_ffi_container_container_details.h:

Program Listing for File container_details.h
============================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_container_container_details.h>` (``tvm/ffi/container/container_details.h``)

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
   
   #ifndef TVM_FFI_CONTAINER_CONTAINER_DETAILS_H_
   #define TVM_FFI_CONTAINER_CONTAINER_DETAILS_H_
   
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   
   #include <sstream>
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   namespace details {
   template <typename ArrayType, typename ElemType>
   class InplaceArrayBase {
    public:
     const ElemType& operator[](size_t idx) const {
       size_t size = Self()->GetSize();
       if (idx > size) {
         TVM_FFI_THROW(IndexError) << "Index " << idx << " out of bounds " << size;
       }
       return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
     }
   
     ElemType& operator[](size_t idx) {
       size_t size = Self()->GetSize();
       if (idx > size) {
         TVM_FFI_THROW(IndexError) << "Index " << idx << " out of bounds " << size;
       }
       return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
     }
   
     ~InplaceArrayBase() {
       if constexpr (!(std::is_standard_layout<ElemType>::value && std::is_trivial<ElemType>::value)) {
         size_t size = Self()->GetSize();
         for (size_t i = 0; i < size; ++i) {
           ElemType* fp = reinterpret_cast<ElemType*>(AddressOf(i));
           fp->ElemType::~ElemType();
         }
       }
     }
   
    protected:
     template <typename... Args>
     void EmplaceInit(size_t idx, Args&&... args) {
       void* field_ptr = AddressOf(idx);
       new (field_ptr) ElemType(std::forward<Args>(args)...);
     }
   
     inline ArrayType* Self() const {
       return static_cast<ArrayType*>(const_cast<InplaceArrayBase*>(this));
     }
   
     void* AddressOf(size_t idx) const {
       static_assert(
           alignof(ArrayType) % alignof(ElemType) == 0 && sizeof(ArrayType) % alignof(ElemType) == 0,
           "The size and alignment of ArrayType should respect "
           "ElemType's alignment.");
   
       size_t kDataStart = sizeof(ArrayType);
       ArrayType* self = Self();
       char* data_start = reinterpret_cast<char*>(self) + kDataStart;
       return data_start + idx * sizeof(ElemType);
     }
   };
   
   template <typename Converter, typename TIter>
   class IterAdapter {
    public:
     using difference_type = typename std::iterator_traits<TIter>::difference_type;
     using value_type = typename Converter::ResultType;
     using pointer = typename Converter::ResultType*;
     using reference = typename Converter::ResultType&;
     using iterator_category = typename std::iterator_traits<TIter>::iterator_category;
   
     explicit IterAdapter(TIter iter) : iter_(iter) {}
     IterAdapter& operator++() {
       ++iter_;
       return *this;
     }
     IterAdapter& operator--() {
       --iter_;
       return *this;
     }
     IterAdapter operator++(int) {
       IterAdapter copy = *this;
       ++iter_;
       return copy;
     }
     IterAdapter operator--(int) {
       IterAdapter copy = *this;
       --iter_;
       return copy;
     }
   
     IterAdapter operator+(difference_type offset) const { return IterAdapter(iter_ + offset); }
   
     IterAdapter operator-(difference_type offset) const { return IterAdapter(iter_ - offset); }
   
     IterAdapter& operator+=(difference_type offset) {
       iter_ += offset;
       return *this;
     }
   
     IterAdapter& operator-=(difference_type offset) {
       iter_ -= offset;
       return *this;
     }
   
     template <typename T = IterAdapter>
     typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                             typename T::difference_type>::type inline
     operator-(const IterAdapter& rhs) const {
       return iter_ - rhs.iter_;
     }
   
     bool operator==(IterAdapter other) const { return iter_ == other.iter_; }
     bool operator!=(IterAdapter other) const { return !(*this == other); }
     const value_type operator*() const { return Converter::convert(*iter_); }
   
    private:
     TIter iter_;
   };
   
   template <typename Converter, typename TIter>
   class ReverseIterAdapter {
    public:
     using difference_type = typename std::iterator_traits<TIter>::difference_type;
     using value_type = typename Converter::ResultType;
     using pointer = typename Converter::ResultType*;
     using reference = typename Converter::ResultType&;  // NOLINT(*)
     using iterator_category = typename std::iterator_traits<TIter>::iterator_category;
   
     explicit ReverseIterAdapter(TIter iter) : iter_(iter) {}
     ReverseIterAdapter& operator++() {
       --iter_;
       return *this;
     }
     ReverseIterAdapter& operator--() {
       ++iter_;
       return *this;
     }
     ReverseIterAdapter operator++(int) {
       ReverseIterAdapter copy = *this;
       --iter_;
       return copy;
     }
     ReverseIterAdapter operator--(int) {
       ReverseIterAdapter copy = *this;
       ++iter_;
       return copy;
     }
     ReverseIterAdapter operator+(difference_type offset) const {
       return ReverseIterAdapter(iter_ - offset);
     }
   
     template <typename T = ReverseIterAdapter>
     typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                             typename T::difference_type>::type inline
     operator-(const ReverseIterAdapter& rhs) const {
       return rhs.iter_ - iter_;
     }
   
     bool operator==(ReverseIterAdapter other) const { return iter_ == other.iter_; }
     bool operator!=(ReverseIterAdapter other) const { return !(*this == other); }
     const value_type operator*() const { return Converter::convert(*iter_); }
   
    private:
     TIter iter_;
   };
   
   template <typename T>
   inline constexpr bool storage_enabled_v = std::is_same_v<T, Any> || TypeTraits<T>::storage_enabled;
   
   template <typename... T>
   inline constexpr bool all_storage_enabled_v = (storage_enabled_v<T> && ...);
   
   template <typename... T>
   inline constexpr bool all_object_ref_v = (std::is_base_of_v<ObjectRef, T> && ...);
   template <typename Base, typename Derived>
   inline constexpr bool type_contains_v =
       std::is_base_of_v<Base, Derived> || std::is_same_v<Base, Derived>;
   // special case for Any
   template <typename Derived>
   inline constexpr bool type_contains_v<Any, Derived> = true;
   
   template <typename... V>
   std::string ContainerTypeStr(const char* name) {
     std::stringstream ss;
     // helper to construct concated string of TypeStr
     class TypeStrHelper {
      public:
       TypeStrHelper(std::stringstream& stream) : stream_(stream) {}  // NOLINT(*)
   
       TypeStrHelper& operator<<(const std::string& str) {
         if (counter_ > 0) {
           stream_ << ", ";
         }
         stream_ << str;
         counter_++;
         return *this;
       }
   
      private:
       std::stringstream& stream_;  // NOLINT(*)
       int counter_ = 0;
     };
     TypeStrHelper helper(ss);
     ss << name << '<';
     (helper << ... << Type2Str<V>::v());
     ss << '>';
     return ss.str();
   }
   
   }  // namespace details
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_CONTAINER_CONTAINER_DETAILS_H_
