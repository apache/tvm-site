
.. _program_listing_file_tvm_ffi_error.h:

Program Listing for File error.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_error.h>` (``tvm/ffi/error.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     TVM_FFI_THROW(RuntimeError) << "error message";
   }
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
   
   /*
    * \file tvm/ffi/error.h
    * \brief Error handling component.
    */
   #ifndef TVM_FFI_ERROR_H_
   #define TVM_FFI_ERROR_H_
   
   #include <tvm/ffi/base_details.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   
   #include <iostream>
   #include <memory>
   #include <sstream>
   #include <string>
   #include <string_view>
   #include <utility>
   
   #ifndef TVM_FFI_USE_LIBBACKTRACE
   #define TVM_FFI_USE_LIBBACKTRACE 1
   #endif
   
   #ifndef TVM_FFI_BACKTRACE_ON_SEGFAULT
   #define TVM_FFI_BACKTRACE_ON_SEGFAULT 1
   #endif
   
   #ifndef TVM_FFI_ALWAYS_LOG_BEFORE_THROW
   #define TVM_FFI_ALWAYS_LOG_BEFORE_THROW 0
   #endif
   
   namespace tvm {
   namespace ffi {
   
   struct EnvErrorAlreadySet : public std::exception {};
   
   class ErrorObj : public Object, public TVMFFIErrorCell {
    public:
     static constexpr const int32_t _type_index = TypeIndex::kTVMFFIError;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC("ffi.Error", ErrorObj, Object);
   };
   
   namespace details {
   class ErrorObjFromStd : public ErrorObj {
    public:
     ErrorObjFromStd(std::string kind, std::string message, std::string traceback)
         : kind_data_(kind), message_data_(message), traceback_data_(traceback) {
       this->kind = TVMFFIByteArray{kind_data_.data(), kind_data_.length()};
       this->message = TVMFFIByteArray{message_data_.data(), message_data_.length()};
       this->traceback = TVMFFIByteArray{traceback_data_.data(), traceback_data_.length()};
       this->update_traceback = UpdateTraceback;
     }
   
    private:
     static void UpdateTraceback(TVMFFIObjectHandle self, const TVMFFIByteArray* traceback_str) {
       ErrorObjFromStd* obj = static_cast<ErrorObjFromStd*>(self);
       obj->traceback_data_ = std::string(traceback_str->data, traceback_str->size);
       obj->traceback = TVMFFIByteArray{obj->traceback_data_.data(), obj->traceback_data_.length()};
     }
   
     std::string kind_data_;
     std::string message_data_;
     std::string traceback_data_;
   };
   }  // namespace details
   
   class Error : public ObjectRef, public std::exception {
    public:
     Error(std::string kind, std::string message, std::string traceback) {
       data_ = make_object<details::ErrorObjFromStd>(kind, message, traceback);
     }
   
     Error(std::string kind, std::string message, const TVMFFIByteArray* traceback)
         : Error(kind, message, std::string(traceback->data, traceback->size)) {}
   
     std::string kind() const {
       ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
       return std::string(obj->kind.data, obj->kind.size);
     }
   
     std::string message() const {
       ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
       return std::string(obj->message.data, obj->message.size);
     }
   
     std::string traceback() const {
       ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
       return std::string(obj->traceback.data, obj->traceback.size);
     }
   
     void UpdateTraceback(const TVMFFIByteArray* traceback_str) {
       ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
       obj->update_traceback(obj, traceback_str);
     }
   
     const char* what() const noexcept(true) override {
       thread_local std::string what_data;
       ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
       what_data = (std::string("Traceback (most recent call last):\n") +
                    std::string(obj->traceback.data, obj->traceback.size) +
                    std::string(obj->kind.data, obj->kind.size) + std::string(": ") +
                    std::string(obj->message.data, obj->message.size) + '\n');
       return what_data.c_str();
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Error, ObjectRef, ErrorObj);
   };
   
   namespace details {
   
   class ErrorBuilder {
    public:
     explicit ErrorBuilder(std::string kind, std::string traceback, bool log_before_throw)
         : kind_(kind), traceback_(traceback), log_before_throw_(log_before_throw) {}
   
     explicit ErrorBuilder(std::string kind, const TVMFFIByteArray* traceback, bool log_before_throw)
         : ErrorBuilder(kind, std::string(traceback->data, traceback->size), log_before_throw) {}
   
   // MSVC disable warning in error builder as it is exepected
   #ifdef _MSC_VER
   #pragma warning(push)
   #pragma warning(disable : 4722)
   #endif
     // avoid inline to reduce binary size, error throw path do not need to be fast
     [[noreturn]] ~ErrorBuilder() noexcept(false) {
       ::tvm::ffi::Error error(std::move(kind_), stream_.str(), std::move(traceback_));
       if (log_before_throw_) {
         std::cerr << error.what();
       }
       throw error;
     }
   #ifdef _MSC_VER
   #pragma warning(pop)
   #endif
   
     std::ostringstream& stream() { return stream_; }
   
    protected:
     std::string kind_;
     std::ostringstream stream_;
     std::string traceback_;
     bool log_before_throw_;
   };
   
   }  // namespace details
   
   #define TVM_FFI_THROW(ErrorKind)                                                              \
     ::tvm::ffi::details::ErrorBuilder(#ErrorKind,                                               \
                                       TVMFFITraceback(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0), \
                                       TVM_FFI_ALWAYS_LOG_BEFORE_THROW)                          \
         .stream()
   
   #define TVM_FFI_LOG_AND_THROW(ErrorKind)                                          \
     ::tvm::ffi::details::ErrorBuilder(                                              \
         #ErrorKind, TVMFFITraceback(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0), true) \
         .stream()
   
   // Glog style checks with TVM_FFI prefix
   // NOTE: we explicitly avoid glog style generic macros (LOG/CHECK) in tvm ffi
   // to avoid potential conflict of downstream users who might have their own GLOG style macros
   namespace details {
   
   template <typename X, typename Y>
   TVM_FFI_INLINE std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
     std::ostringstream os;
     os << " (" << x << " vs. " << y << ") ";  // CHECK_XX(x, y) requires x and y can be serialized to
                                               // string. Use CHECK(x OP y) otherwise.
     return std::make_unique<std::string>(os.str());
   }
   
   #define TVM_FFI_CHECK_FUNC(name, op)                                                   \
     template <typename X, typename Y>                                                    \
     TVM_FFI_INLINE std::unique_ptr<std::string> LogCheck##name(const X& x, const Y& y) { \
       if (x op y) return nullptr;                                                        \
       return LogCheckFormat(x, y);                                                       \
     }                                                                                    \
     TVM_FFI_INLINE std::unique_ptr<std::string> LogCheck##name(int x, int y) {           \
       return LogCheck##name<int, int>(x, y);                                             \
     }
   
   // Inline _Pragma in macros does not work reliably on old version of MSVC and
   // GCC. We wrap all comparisons in a function so that we can use #pragma to
   // silence bad comparison warnings.
   #if defined(__GNUC__) || defined(__clang__)  // GCC and Clang
   #pragma GCC diagnostic push
   #pragma GCC diagnostic ignored "-Wsign-compare"
   #elif defined(_MSC_VER)  // MSVC
   #pragma warning(push)
   #pragma warning(disable : 4389)  // '==' : signed/unsigned mismatch
   #endif
   
   TVM_FFI_CHECK_FUNC(_LT, <)
   TVM_FFI_CHECK_FUNC(_GT, >)
   TVM_FFI_CHECK_FUNC(_LE, <=)
   TVM_FFI_CHECK_FUNC(_GE, >=)
   TVM_FFI_CHECK_FUNC(_EQ, ==)
   TVM_FFI_CHECK_FUNC(_NE, !=)
   
   #if defined(__GNUC__) || defined(__clang__)  // GCC and Clang
   #pragma GCC diagnostic pop
   #elif defined(_MSC_VER)  // MSVC
   #pragma warning(pop)
   #endif
   }  // namespace details
   
   #define TVM_FFI_ICHECK_BINARY_OP(name, op, x, y)                        \
     if (auto __tvm__log__err = ::tvm::ffi::details::LogCheck##name(x, y)) \
     TVM_FFI_THROW(InternalError) << "Check failed: " << #x " " #op " " #y << *__tvm__log__err << ": "
   
   #define TVM_FFI_ICHECK(x) \
     if (!(x)) TVM_FFI_THROW(InternalError) << "Check failed: (" #x << ") is false: "
   
   #define TVM_FFI_ICHECK_LT(x, y) TVM_FFI_ICHECK_BINARY_OP(_LT, <, x, y)
   #define TVM_FFI_ICHECK_GT(x, y) TVM_FFI_ICHECK_BINARY_OP(_GT, >, x, y)
   #define TVM_FFI_ICHECK_LE(x, y) TVM_FFI_ICHECK_BINARY_OP(_LE, <=, x, y)
   #define TVM_FFI_ICHECK_GE(x, y) TVM_FFI_ICHECK_BINARY_OP(_GE, >=, x, y)
   #define TVM_FFI_ICHECK_EQ(x, y) TVM_FFI_ICHECK_BINARY_OP(_EQ, ==, x, y)
   #define TVM_FFI_ICHECK_NE(x, y) TVM_FFI_ICHECK_BINARY_OP(_NE, !=, x, y)
   #define TVM_FFI_ICHECK_NOTNULL(x)                                                 \
     ((x) == nullptr ? TVM_FFI_THROW(InternalError) << "Check not null: " #x << ' ', \
      (x)            : (x))  // NOLINT(*)
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_ERROR_H_
