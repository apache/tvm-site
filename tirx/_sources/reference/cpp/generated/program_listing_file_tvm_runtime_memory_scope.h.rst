
.. _program_listing_file_tvm_runtime_memory_scope.h:

Program Listing for File memory_scope.h
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_runtime_memory_scope.h>` (``tvm/runtime/memory_scope.h``)

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
   #ifndef TVM_RUNTIME_MEMORY_SCOPE_H_
   #define TVM_RUNTIME_MEMORY_SCOPE_H_
   
   #include <tvm/ffi/enum.h>
   #include <tvm/ffi/reflection/enum_def.h>
   #include <tvm/runtime/base.h>
   
   #include <ostream>
   #include <string_view>
   
   namespace tvm {
   namespace runtime {
   namespace detail {
   TVM_RUNTIME_DLL void RegisterBuiltinMemoryScopes();
   }  // namespace detail
   
   enum class MemoryAllocProperty : int {
     kNormal = 0,
     kSingleAllocation = 1,
     kSeparateAllocations = 2,
   };
   
   namespace attr {
   inline constexpr ffi::reflection::AttrKey<MemoryAllocProperty> kMemoryAllocProperty{
       "runtime.memory_alloc_property"};
   }  // namespace attr
   
   class MemoryScopeObj : public ffi::EnumObj {
    public:
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.MemoryScope", MemoryScopeObj, ffi::EnumObj);
   };
   
   class MemoryScope : public ffi::Enum {
    public:
     MemoryScope() : ffi::Enum(Global()) {}
     explicit MemoryScope(const ffi::String& name) : ffi::Enum(Get(name)) {}
     const ffi::String& name() const { return get()->_str_index; }
     static bool IsDefined(const ffi::String& name) {
       return ffi::reflection::EnumDef<MemoryScopeObj>::IsDefined(name);
     }
     TVM_RUNTIME_DLL static MemoryScope Get(const ffi::String& name);
     static MemoryScope Register(const ffi::String& name) {
       namespace refl = ffi::reflection;
       detail::RegisterBuiltinMemoryScopes();
       TVM_FFI_CHECK(!name.empty(), ValueError) << "Memory scope name must not be empty";
       TVM_FFI_CHECK(std::string_view(name.data(), name.size()).find('\0') == std::string_view::npos,
                     ValueError)
           << "Memory scope name must not contain a null character";
       refl::EnumDef<MemoryScopeObj>(name.c_str());
       return Get(name);
     }
     TVM_RUNTIME_DLL static const MemoryScope& Global();
     bool operator==(const MemoryScope& other) const { return same_as(other); }
     bool operator!=(const MemoryScope& other) const { return !same_as(other); }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MemoryScope, ffi::Enum, MemoryScopeObj);
   };
   
   inline std::ostream& operator<<(std::ostream& os, const MemoryScope& scope) {
     return os << scope.name();
   }
   
   }  // namespace runtime
   namespace ffi {
   template <>
   inline constexpr bool use_default_type_traits_v<runtime::MemoryScope> = false;
   template <>
   struct TypeTraits<runtime::MemoryScope>
       : public ObjectRefWithFallbackTraitsBase<runtime::MemoryScope, ffi::String> {
     static runtime::MemoryScope ConvertFallbackValue(ffi::String name) {
       return runtime::MemoryScope::Get(name);
     }
   };
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_RUNTIME_MEMORY_SCOPE_H_
