
.. _program_listing_file_tvm_ffi_base_details.h:

Program Listing for File base_details.h
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_base_details.h>` (``tvm/ffi/base_details.h``)

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
   #ifndef TVM_FFI_BASE_DETAILS_H_
   #define TVM_FFI_BASE_DETAILS_H_
   
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/endian.h>
   
   #include <cstddef>
   #include <utility>
   
   #if defined(_MSC_VER)
   #ifndef WIN32_LEAN_AND_MEAN
   #define WIN32_LEAN_AND_MEAN
   #endif
   
   #ifndef NOMINMAX
   #define NOMINMAX
   #endif
   
   #include <windows.h>
   
   #ifdef ERROR
   #undef ERROR
   #endif
   
   #endif
   
   #if defined(_MSC_VER)
   #define TVM_FFI_INLINE [[msvc::forceinline]] inline
   #else
   #define TVM_FFI_INLINE [[gnu::always_inline]] inline
   #endif
   
   #if defined(_MSC_VER)
   #define TVM_FFI_NO_INLINE [[msvc::noinline]]
   #else
   #define TVM_FFI_NO_INLINE [[gnu::noinline]]
   #endif
   
   #if defined(_MSC_VER)
   #define TVM_FFI_UNREACHABLE() __assume(false)
   #else
   #define TVM_FFI_UNREACHABLE() __builtin_unreachable()
   #endif
   
   #define TVM_FFI_STR_CONCAT_(__x, __y) __x##__y
   #define TVM_FFI_STR_CONCAT(__x, __y) TVM_FFI_STR_CONCAT_(__x, __y)
   
   #if defined(__GNUC__) || defined(__clang__)
   #define TVM_FFI_FUNC_SIG __PRETTY_FUNCTION__
   #elif defined(_MSC_VER)
   #define TVM_FFI_FUNC_SIG __FUNCSIG__
   #else
   #define TVM_FFI_FUNC_SIG __func__
   #endif
   
   #if defined(__GNUC__)
   // gcc and clang and attribute constructor
   #define TVM_FFI_STATIC_INIT_BLOCK_DEF_(FnName) __attribute__((constructor)) static void FnName()
   /*
    * \brief Macro that defines a block that will be called during static initialization.
    *
    * \code
    * TVM_FFI_STATIC_INIT_BLOCK() {
    *   RegisterFunctions();
    * }
    * \endcode
    */
   #define TVM_FFI_STATIC_INIT_BLOCK() \
     TVM_FFI_STATIC_INIT_BLOCK_DEF_(TVM_FFI_STR_CONCAT(__TVMFFIStaticInitFunc, __COUNTER__))
   
   #else
   // for other compilers, use the variable trick
   #define TVM_FFI_STATIC_INIT_BLOCK_DEF_(FnName, RegVar) \
     static void FnName();                                \
     [[maybe_unused]] static inline int RegVar = []() {   \
       FnName();                                          \
       return 0;                                          \
     }();                                                 \
     static void FnName()
   
   #define TVM_FFI_STATIC_INIT_BLOCK()                                                       \
     TVM_FFI_STATIC_INIT_BLOCK_DEF_(TVM_FFI_STR_CONCAT(__TVMFFIStaticInitFunc, __COUNTER__), \
                                    TVM_FFI_STR_CONCAT(__TVMFFIStaticInitReg, __COUNTER__))
   #endif
   
   /*
    * \brief Define the default copy/move constructor and assign operator
    * \param TypeName The class typename.
    */
   #define TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName) \
     TypeName(const TypeName& other) = default;                  \
     TypeName(TypeName&& other) = default;                       \
     TypeName& operator=(const TypeName& other) = default;       \
     TypeName& operator=(TypeName&& other) = default;
   
   #define TVM_FFI_LOG_EXCEPTION_CALL_BEGIN() \
     try {                                    \
     (void)0
   
   #define TVM_FFI_LOG_EXCEPTION_CALL_END(Name)                                              \
     }                                                                                       \
     catch (const std::exception& err) {                                                     \
       std::cerr << "Exception caught during " << #Name << ":\n" << err.what() << std::endl; \
       exit(-1);                                                                             \
     }
   
   #define TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result)                    \
     if constexpr (sizeof((result)->v_obj) != sizeof((result)->v_int64)) { \
       (result)->v_int64 = 0;                                              \
     }
   
   namespace tvm {
   namespace ffi {
   namespace details {
   
   // for each iterator
   struct for_each_dispatcher {
     template <typename F, typename... Args, size_t... I>
     static void run(std::index_sequence<I...>, const F& f, Args&&... args) {  // NOLINT(*)
       (f(I, std::forward<Args>(args)), ...);
     }
   };
   
   template <typename F, typename... Args>
   void for_each(const F& f, Args&&... args) {  // NOLINT(*)
     for_each_dispatcher::run(std::index_sequence_for<Args...>{}, f, std::forward<Args>(args)...);
   }
   
   template <typename T, std::enable_if_t<std::is_convertible<T, uint64_t>::value, bool> = true>
   TVM_FFI_INLINE uint64_t StableHashCombine(uint64_t key, const T& value) {
     // XXX: do not use std::hash in this function. This hash must be stable
     // across different platforms and std::hash is implementation dependent.
     return key ^ (uint64_t(value) + 0x9e3779b9 + (key << 6) + (key >> 2));
   }
   
   TVM_FFI_INLINE uint64_t StableHashBytes(const void* data_ptr, size_t size) {
     const char* data = reinterpret_cast<const char*>(data_ptr);
     const constexpr uint64_t kMultiplier = 1099511628211ULL;
     const constexpr uint64_t kMod = 2147483647ULL;
     union Union {
       uint8_t a[8];
       uint64_t b;
     } u;
     static_assert(sizeof(Union) == sizeof(uint64_t), "sizeof(Union) != sizeof(uint64_t)");
     const char* it = data;
     const char* end = it + size;
     uint64_t result = 0;
     if constexpr (TVM_FFI_IO_NO_ENDIAN_SWAP) {
       // if alignment requirement is met, directly use load
       if (reinterpret_cast<uintptr_t>(it) % 8 == 0) {
         for (; it + 8 <= end; it += 8) {
           u.b = *reinterpret_cast<const uint64_t*>(it);
           result = (result * kMultiplier + u.b) % kMod;
         }
       } else {
         // unaligned version
         for (; it + 8 <= end; it += 8) {
           u.a[0] = it[0];
           u.a[1] = it[1];
           u.a[2] = it[2];
           u.a[3] = it[3];
           u.a[4] = it[4];
           u.a[5] = it[5];
           u.a[6] = it[6];
           u.a[7] = it[7];
           result = (result * kMultiplier + u.b) % kMod;
         }
       }
     } else {
       // need endian swap
       for (; it + 8 <= end; it += 8) {
         u.a[0] = it[7];
         u.a[1] = it[6];
         u.a[2] = it[5];
         u.a[3] = it[4];
         u.a[4] = it[3];
         u.a[5] = it[2];
         u.a[6] = it[1];
         u.a[7] = it[0];
         result = (result * kMultiplier + u.b) % kMod;
       }
     }
   
     if (it < end) {
       u.b = 0;
       uint8_t* a = u.a;
       if (it + 4 <= end) {
         a[0] = it[0];
         a[1] = it[1];
         a[2] = it[2];
         a[3] = it[3];
         it += 4;
         a += 4;
       }
       if (it + 2 <= end) {
         a[0] = it[0];
         a[1] = it[1];
         it += 2;
         a += 2;
       }
       if (it + 1 <= end) {
         a[0] = it[0];
         it += 1;
         a += 1;
       }
       if constexpr (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
         std::swap(u.a[0], u.a[7]);
         std::swap(u.a[1], u.a[6]);
         std::swap(u.a[2], u.a[5]);
         std::swap(u.a[3], u.a[4]);
       }
       result = (result * kMultiplier + u.b) % kMod;
     }
     return result;
   }
   
   TVM_FFI_INLINE uint64_t StableHashSmallStrBytes(const TVMFFIAny* data) {
     if constexpr (TVM_FFI_IO_NO_ENDIAN_SWAP) {
       // fast path, no endian swap, simply hash as uint64_t
       const constexpr uint64_t kMod = 2147483647ULL;
       return data->v_uint64 % kMod;
     }
     return StableHashBytes(reinterpret_cast<const void*>(data), sizeof(data->v_uint64));
   }
   
   }  // namespace details
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_BASE_DETAILS_H_
