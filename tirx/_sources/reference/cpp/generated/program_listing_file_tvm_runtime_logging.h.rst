
.. _program_listing_file_tvm_runtime_logging.h:

Program Listing for File logging.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_runtime_logging.h>` (``tvm/runtime/logging.h``)

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
   
   #ifndef TVM_RUNTIME_LOGGING_H_
   #define TVM_RUNTIME_LOGGING_H_
   
   #include <tvm/ffi/error.h>
   #include <tvm/runtime/base.h>
   
   #include <ctime>
   #include <iomanip>
   #include <iostream>
   #include <sstream>
   #include <string>
   #include <unordered_map>
   #include <vector>
   
   #ifndef TVM_LOG_CUSTOMIZE
   #define TVM_LOG_CUSTOMIZE 0
   #endif
   
   namespace tvm {
   namespace runtime {
   
   namespace detail {
   // Provide support for customized logging.
   #if TVM_LOG_CUSTOMIZE
   [[noreturn]] TVM_RUNTIME_DLL void LogFatalImpl(const std::string& file, int lineno,
                                                  const std::string& message);
   
   TVM_RUNTIME_DLL void LogMessageImpl(const std::string& file, int lineno, int level,
                                       const std::string& message);
   
   class LogFatal {
    public:
     LogFatal(const std::string& file, int lineno) : file_(file), lineno_(lineno) {}
   #ifdef _MSC_VER
   #pragma warning(push)
   #pragma warning(disable : 4722)
   #endif
     [[noreturn]] ~LogFatal() noexcept(false) { LogFatalImpl(file_, lineno_, stream_.str()); }
   #ifdef _MSC_VER
   #pragma warning(pop)
   #endif
     std::ostringstream& stream() { return stream_; }
   
    private:
     std::ostringstream stream_;
     std::string file_;
     int lineno_;
   };
   
   class LogMessage {
    public:
     LogMessage(const std::string& file, int lineno, int level)
         : file_(file), lineno_(lineno), level_(level) {}
     ~LogMessage() { LogMessageImpl(file_, lineno_, level_, stream_.str()); }
     std::ostringstream& stream() { return stream_; }
   
    private:
     std::string file_;
     int lineno_;
     int level_;
     std::ostringstream stream_;
   };
   
   #else
   
   class LogFatal {
    public:
     TVM_FFI_NO_INLINE LogFatal(const char* file, int lineno) { GetEntry().Init(file, lineno); }
   #ifdef _MSC_VER
   #pragma warning(push)
   #pragma warning(disable : 4722)
   #endif
     [[noreturn]] ~LogFatal() noexcept(false) {
       GetEntry().Finalize();
       throw;
     }
   #ifdef _MSC_VER
   #pragma warning(pop)
   #endif
     std::ostringstream& stream() { return GetEntry().stream_; }
   
    private:
     struct Entry {
       void Init(const char* file, int lineno) {
         this->stream_.str("");
         this->file_ = file;
         this->lineno_ = lineno;
       }
       [[noreturn]] TVM_FFI_NO_INLINE ffi::Error Finalize() noexcept(false) {
         ffi::Error error("InternalError", stream_.str(),
                          TVMFFIBacktrace(file_.c_str(), lineno_, "", 0));
         throw error;
       }
       std::ostringstream stream_;
       std::string file_;
       int lineno_;
     };
   
     TVM_FFI_NO_INLINE TVM_RUNTIME_DLL static Entry& GetEntry();
   };
   
   class LogMessage {
    public:
     LogMessage(const std::string& file, int lineno, int level) {
       // Use inline constexpr to avoid ODR-issues with hidden static members
       // when libtvm_runtime.so is built with -fvisibility=hidden.
       static constexpr const char* kLevelStrings[] = {
           ": Debug: ",    // TVM_LOG_LEVEL_DEBUG
           ": ",           // TVM_LOG_LEVEL_INFO
           ": Warning: ",  // TVM_LOG_LEVEL_WARNING
           ": Error: ",    // TVM_LOG_LEVEL_ERROR
       };
       std::time_t t = std::time(nullptr);
       stream_ << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] " << file << ":" << lineno
               << kLevelStrings[level];
     }
     TVM_FFI_NO_INLINE ~LogMessage() { std::cerr << stream_.str() << std::endl; }
     std::ostringstream& stream() { return stream_; }
   
    private:
     std::ostringstream stream_;
   };
   
   #endif
   
   // This class is used to explicitly ignore values in the conditional
   // logging macros.  This avoids compiler warnings like "value computed
   // is not used" and "statement has no effect".
   class LogMessageVoidify {
    public:
     LogMessageVoidify() {}
     // This has to be an operator with a precedence lower than << but
     // higher than "?:". See its usage.
     void operator&(std::ostream&) {}
   };
   
   class TVM_RUNTIME_DLL TvmLogDebugSettings {
    public:
     inline static const TvmLogDebugSettings& FromFlag() {
       // Parse and cache the verbosity level map.
       static const auto* settings =
           new TvmLogDebugSettings(TvmLogDebugSettings::ParseSpec(std::getenv("TVM_LOG_DEBUG")));
       return *settings;
     }
   
     static TvmLogDebugSettings ParseSpec(const char* opt_spec);
   
     inline bool VerboseEnabled(const char* opt_filename, int level) const {
       if (opt_filename == nullptr || level < 0 || vlog_level_map_.empty()) {
         return false;
       }
       return VerboseEnabledImpl(opt_filename, level);
     }
   
     bool dlog_enabled() const { return dlog_enabled_; }
   
    private:
     // Slow path for VerboseEnabled.
     bool VerboseEnabledImpl(const std::string& filename, int level) const;
   
     bool dlog_enabled_ = false;
     std::unordered_map<std::string, int> vlog_level_map_;
   };
   
   inline bool DebugLoggingEnabled() {
     static int state = 0;
     if (state == 0) {
       state = TvmLogDebugSettings::FromFlag().dlog_enabled() ? 1 : -1;
     }
     return state == 1;
   }
   
   inline bool VerboseLoggingEnabled(const char* opt_filename, int level) {
     return TvmLogDebugSettings::FromFlag().VerboseEnabled(opt_filename, level);
   }
   
   class VLogContext {
    public:
     void Push(std::stringstream* stream) { context_stack_.push_back(stream); }
     void Pop() {
       if (!context_stack_.empty()) {
         context_stack_.pop_back();
       }
     }
   
     std::string str() const {
       std::stringstream result;
       for (const auto* entry : context_stack_) {
         TVM_FFI_ICHECK_NOTNULL(entry);
         result << entry->str();
         result << ": ";
       }
       return result.str();
     }
   
    private:
     std::vector<std::stringstream*> context_stack_;
   };
   
   inline VLogContext* ThreadLocalVLogContext() {
     static thread_local VLogContext inst;
     return &inst;
   }
   
   class VLogContextEntry {
    public:
     VLogContextEntry() { ThreadLocalVLogContext()->Push(&sstream_); }
     ~VLogContextEntry() { ThreadLocalVLogContext()->Pop(); }
     std::ostream& stream() { return sstream_; }
   
    private:
     std::stringstream sstream_;
   };
   
   }  // namespace detail
   
   #define TVM_LOG_LEVEL_DEBUG 0
   #define TVM_LOG_LEVEL_INFO 1
   #define TVM_LOG_LEVEL_WARNING 2
   #define TVM_LOG_LEVEL_ERROR 3
   #define TVM_LOG_LEVEL_FATAL 4
   #define LOG(level) LOG_##level
   #define LOG_DEBUG \
     ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_DEBUG).stream()
   #define LOG_FATAL ::tvm::runtime::detail::LogFatal(__FILE__, __LINE__).stream()
   #define LOG_INFO ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_INFO).stream()
   #define LOG_ERROR \
     ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_ERROR).stream()
   #define LOG_WARNING \
     ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_WARNING).stream()
   
   #define LOG_IF(severity, condition) \
     !(condition) ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(severity)
   
   #if TVM_LOG_DEBUG
   
   #define LOG_DFATAL LOG_FATAL
   #define DFATAL FATAL
   #define DLOG(severity) LOG_IF(severity, ::tvm::runtime::detail::DebugLoggingEnabled())
   #define DLOG_IF(severity, condition) \
     LOG_IF(severity, ::tvm::runtime::detail::DebugLoggingEnabled() && (condition))
   
   #define VLOG_CONTEXT                                    \
     ::tvm::runtime::detail::VLogContextEntry vlog_entry_; \
     vlog_entry_.stream()
   
   #else
   
   #define LOG_DFATAL LOG_ERROR
   #define DFATAL ERROR
   #define DLOG(severity) true ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(severity)
   #define DLOG_IF(severity, condition) \
     (true || !(condition)) ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(severity)
   #define VLOG_CONTEXT true ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(INFO)
   
   #endif
   
   #define VLOG(level)                                                               \
     DLOG_IF(INFO, ::tvm::runtime::detail::VerboseLoggingEnabled(__FILE__, (level))) \
         << ::tvm::runtime::detail::ThreadLocalVLogContext()->str()
   
   }  // namespace runtime
   }  // namespace tvm
   #endif  // TVM_RUNTIME_LOGGING_H_
