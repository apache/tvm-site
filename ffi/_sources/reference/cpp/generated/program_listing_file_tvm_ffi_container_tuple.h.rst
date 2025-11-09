
.. _program_listing_file_tvm_ffi_container_tuple.h:

Program Listing for File tuple.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_container_tuple.h>` (``tvm/ffi/container/tuple.h``)

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
   
   #ifndef TVM_FFI_CONTAINER_TUPLE_H_
   #define TVM_FFI_CONTAINER_TUPLE_H_
   
   #include <tvm/ffi/container/array.h>
   
   #include <cstddef>
   #include <string>
   #include <tuple>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   template <typename... Types>
   class Tuple : public ObjectRef {
    public:
     static_assert(details::all_storage_enabled_v<Types...>,
                   "All types used in Tuple<...> must be compatible with Any");
     Tuple() : ObjectRef(MakeDefaultTupleNode()) {}
     explicit Tuple(UnsafeInit tag) : ObjectRef(tag) {}
     Tuple(const Tuple<Types...>& other) : ObjectRef(other) {}
     Tuple(Tuple<Types...>&& other) noexcept : ObjectRef(std::move(other)) {}
     template <typename... UTypes,
               typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...), int>>
     Tuple(const Tuple<UTypes...>& other) : ObjectRef(other) {}  // NOLINT(google-explicit-constructor)
   
     template <typename... UTypes,
               typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...), int>>
     Tuple(Tuple<UTypes...>&& other)  // NOLINT(google-explicit-constructor)
         : ObjectRef(std::move(other)) {}
   
     template <typename... UTypes, typename = std::enable_if_t<
                                       sizeof...(Types) == sizeof...(UTypes) &&
                                       !(sizeof...(Types) == 1 &&
                                         (std::is_same_v<std::decay_t<UTypes>, Tuple<Types>> && ...))>>
     explicit Tuple(UTypes&&... args) : ObjectRef(MakeTupleNode(std::forward<UTypes>(args)...)) {}
   
     TVM_FFI_INLINE Tuple& operator=(const Tuple<Types...>& other) {
       data_ = other.data_;
       return *this;
     }
   
     TVM_FFI_INLINE Tuple& operator=(Tuple<Types...>&& other) noexcept {
       data_ = std::move(other.data_);
       return *this;
     }
   
     template <typename... UTypes,
               typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...)>>
     TVM_FFI_INLINE Tuple& operator=(const Tuple<UTypes...>& other) {
       data_ = other.data_;
       return *this;
     }
   
     template <typename... UTypes,
               typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...)>>
     TVM_FFI_INLINE Tuple& operator=(Tuple<UTypes...>&& other) {
       data_ = std::move(other.data_);
       return *this;
     }
   
     template <size_t I>
     auto get() const& {
       static_assert(I < sizeof...(Types), "Tuple index out of bounds");
       using ReturnType = std::tuple_element_t<I, std::tuple<Types...>>;
       const Any* ptr = GetArrayObj()->begin() + I;
       return details::AnyUnsafe::CopyFromAnyViewAfterCheck<ReturnType>(*ptr);
     }
   
     template <size_t I>
     auto get() && {
       if (!this->unique()) {
         // fallback to const& version if not unique
         return std::as_const(*this).template get<I>();
       }
       static_assert(I < sizeof...(Types), "Tuple index out of bounds");
       using ReturnType = std::tuple_element_t<I, std::tuple<Types...>>;
       Any* ptr = GetArrayObj()->MutableBegin() + I;
       return details::AnyUnsafe::MoveFromAnyAfterCheck<ReturnType>(std::move(*ptr));
     }
   
     template <size_t I, typename U>
     void Set(U&& item) {
       static_assert(I < sizeof...(Types), "Tuple index out of bounds");
       using T = std::tuple_element_t<I, std::tuple<Types...>>;
       this->CopyIfNotUnique();
       Any* ptr = GetArrayObj()->MutableBegin() + I;
       *ptr = T(std::forward<U>(item));
     }
   
     using ContainerType = ArrayObj;
   
    private:
     static ObjectPtr<ArrayObj> MakeDefaultTupleNode() {
       ObjectPtr<ArrayObj> p = ArrayObj::Empty(sizeof...(Types));
       Any* itr = p->MutableBegin();
       // increase size after each new to ensure exception safety
       ((new (itr++) Any(Types()), p->size_++), ...);
       return p;
     }
   
     template <typename... UTypes>
     static ObjectPtr<ArrayObj> MakeTupleNode(UTypes&&... args) {
       ObjectPtr<ArrayObj> p = ArrayObj::Empty(sizeof...(Types));
       Any* itr = p->MutableBegin();
       // increase size after each new to ensure exception safety
       ((new (itr++) Any(Types(std::forward<UTypes>(args))), p->size_++), ...);
       return p;
     }
   
     void CopyIfNotUnique() {
       if (!data_.unique()) {
         ObjectPtr<ArrayObj> p = ArrayObj::Empty(sizeof...(Types));
         Any* itr = p->MutableBegin();
         const Any* read = GetArrayObj()->begin();
         // increase size after each new to ensure exception safety
         for (size_t i = 0; i < sizeof...(Types); ++i) {
           new (itr++) Any(*read++);
           p->size_++;
         }
         data_ = std::move(p);
       }
     }
   
     ArrayObj* GetArrayObj() const { return static_cast<ArrayObj*>(data_.get()); }
   
     template <typename... UTypes>
     friend class Tuple;
   };
   
   template <typename... Types>
   inline constexpr bool use_default_type_traits_v<Tuple<Types...>> = false;
   
   template <typename... Types>
   struct TypeTraits<Tuple<Types...>> : public ObjectRefTypeTraitsBase<Tuple<Types...>> {
     using ObjectRefTypeTraitsBase<Tuple<Types...>>::CopyFromAnyViewAfterCheck;
   
     TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
       if (src->type_index != TypeIndex::kTVMFFIArray) {
         return TypeTraitsBase::GetMismatchTypeInfo(src);
       }
       const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
       if (n->size() != sizeof...(Types)) {
         return "Array[size=" + std::to_string(n->size()) + "]";
       }
       return GetMismatchTypeInfoHelper<0, Types...>(n->begin());
     }
   
     template <size_t I, typename T, typename... Rest>
     TVM_FFI_INLINE static std::string GetMismatchTypeInfoHelper(const Any* arr) {
       if constexpr (!std::is_same_v<T, Any>) {
         const Any& any_v = arr[I];
         if (!details::AnyUnsafe::CheckAnyStrict<T>(any_v) && !(any_v.try_cast<T>().has_value())) {
           // now report the accurate mismatch information
           return "Array[index " + std::to_string(I) + ": " +
                  details::AnyUnsafe::GetMismatchTypeInfo<T>(any_v) + "]";
         }
       }
       if constexpr (sizeof...(Rest) > 0) {
         return GetMismatchTypeInfoHelper<I + 1, Rest...>(arr);
       }
       TVM_FFI_THROW(InternalError) << "Cannot reach here";
       TVM_FFI_UNREACHABLE();
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if (src->type_index != TypeIndex::kTVMFFIArray) return false;
       const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
       if (n->size() != sizeof...(Types)) return false;
       const TVMFFIAny* ffi_any_arr = reinterpret_cast<const TVMFFIAny*>(n->begin());
       return CheckAnyStrictHelper<0, Types...>(ffi_any_arr);
     }
   
     template <size_t I, typename T, typename... Rest>
     TVM_FFI_INLINE static bool CheckAnyStrictHelper(const TVMFFIAny* src_arr) {
       if constexpr (!std::is_same_v<T, Any>) {
         if (!TypeTraits<T>::CheckAnyStrict(src_arr + I)) {
           return false;
         }
       }
       if constexpr (sizeof...(Rest) > 0) {
         return CheckAnyStrictHelper<I + 1, Rest...>(src_arr);
       }
       return true;
     }
   
     TVM_FFI_INLINE static std::optional<Tuple<Types...>> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index != TypeIndex::kTVMFFIArray) return std::nullopt;
       const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
       if (n->size() != sizeof...(Types)) return std::nullopt;
       // fast path, storage is already in the right type
       if (CheckAnyStrict(src)) {
         return CopyFromAnyViewAfterCheck(src);
       }
       // slow path, try to convert to each type to match the tuple storage need.
       Array<Any> arr = TypeTraits<Array<Any>>::CopyFromAnyViewAfterCheck(src);
       Any* ptr = arr.CopyOnWrite()->MutableBegin();
       if (TryConvertElements<0, Types...>(ptr)) {
         return details::ObjectUnsafe::ObjectRefFromObjectPtr<Tuple<Types...>>(
             details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(arr));
       }
       return std::nullopt;
     }
   
     template <size_t I, typename T, typename... Rest>
     TVM_FFI_INLINE static bool TryConvertElements(Any* arr) {
       if constexpr (!std::is_same_v<T, Any>) {
         if (auto opt_convert = arr[I].try_cast<T>()) {
           arr[I] = *std::move(opt_convert);
         } else {
           return false;
         }
       }
       if constexpr (sizeof...(Rest) > 0) {
         return TryConvertElements<I + 1, Rest...>(std::move(arr));
       } else {
         return true;
       }
     }
   
     TVM_FFI_INLINE static std::string TypeStr() {
       return details::ContainerTypeStr<Types...>("Tuple");
     }
     TVM_FFI_INLINE static std::string TypeSchema() {
       std::ostringstream oss;
       oss << R"({"type":"Tuple","args":[)";
       const char* sep = "";
       ((oss << sep << details::TypeSchema<Types>::v(), sep = ","), ...);
       oss << "]}";
       return oss.str();
     }
   };
   
   namespace details {
   template <typename... T, typename... U>
   inline constexpr bool type_contains_v<Tuple<T...>, Tuple<U...>> = (type_contains_v<T, U> && ...);
   }  // namespace details
   
   
   
   template <std::size_t I, typename... Types>
   inline constexpr auto get(const Tuple<Types...>& t)
       -> std::tuple_element_t<I, std::tuple<Types...>> {
     return t.template get<I>();
   }
   
   template <std::size_t I, typename... Types>
   inline constexpr auto get(Tuple<Types...>&& t) -> std::tuple_element_t<I, std::tuple<Types...>> {
     return std::move(t).template get<I>();
   }
   
   template <typename... UTypes>
   Tuple(UTypes&&...) -> Tuple<std::remove_cv_t<std::remove_reference_t<UTypes>>...>;
   
   
   }  // namespace ffi
   }  // namespace tvm
   
   namespace std {
   
   template <typename... Types>
   struct tuple_size<::tvm::ffi::Tuple<Types...>>
       : public std::integral_constant<size_t, sizeof...(Types)> {};
   
   template <size_t I, typename... Types>
   struct tuple_element<I, ::tvm::ffi::Tuple<Types...>> {
     using type = std::tuple_element_t<I, std::tuple<Types...>>;
   };
   
   }  // namespace std
   
   #endif  // TVM_FFI_CONTAINER_TUPLE_H_
