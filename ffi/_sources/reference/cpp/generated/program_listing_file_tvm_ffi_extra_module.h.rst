
.. _program_listing_file_tvm_ffi_extra_module.h:

Program Listing for File module.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_module.h>` (``tvm/ffi/extra/module.h``)

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
   #ifndef TVM_FFI_EXTRA_MODULE_H_
   #define TVM_FFI_EXTRA_MODULE_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/extra/base.h>
   #include <tvm/ffi/function.h>
   
   namespace tvm {
   namespace ffi {
   
   // forward declare Module
   class Module;
   
   class TVM_FFI_EXTRA_CXX_API ModuleObj : public Object {
    public:
     virtual const char* kind() const = 0;
     virtual int GetPropertyMask() const { return 0b000; }
     virtual Optional<Function> GetFunction(const String& name) = 0;
     virtual bool ImplementsFunction(const String& name) { return GetFunction(name).defined(); }
     virtual Optional<String> GetFunctionMetadata(const String& name) { return std::nullopt; }
     virtual void WriteToFile(const String& file_name, const String& format) const {
       TVM_FFI_THROW(RuntimeError) << "Module[" << kind() << "] does not support WriteToFile";
     }
     virtual Array<String> GetWriteFormats() const { return Array<String>(); }
     virtual Bytes SaveToBytes() const {
       TVM_FFI_THROW(RuntimeError) << "Module[" << kind() << "] does not support SaveToBytes";
       TVM_FFI_UNREACHABLE();
     }
     virtual String InspectSource(const String& format = "") const { return String(); }
     virtual void ImportModule(const Module& other);
     virtual void ClearImports();
     Optional<Function> GetFunction(const String& name, bool query_imports);
     bool ImplementsFunction(const String& name, bool query_imports);
     Optional<String> GetFunctionMetadata(const String& name, bool query_imports);
     const Array<Any>& imports() const { return this->imports_; }
   
     struct InternalUnsafe;
   
     static constexpr const int32_t _type_index = TypeIndex::kTVMFFIModule;
     static constexpr const bool _type_mutable = true;
     static const constexpr bool _type_final = true;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIModule, ModuleObj, Object);
   
    protected:
     friend struct InternalUnsafe;
   
     Array<Any> imports_;
   
    private:
     Map<String, ffi::Function> import_lookup_cache_;
   };
   
   class Module : public ObjectRef {
    public:
     enum ModulePropertyMask : int {
       kBinarySerializable = 0b001,
       kRunnable = 0b010,
       kCompilationExportable = 0b100
     };
     explicit Module(ObjectPtr<ModuleObj> ptr) : ObjectRef(ptr) { TVM_FFI_ICHECK(ptr != nullptr); }
     TVM_FFI_EXTRA_CXX_API static Module LoadFromFile(const String& file_name);
     TVM_FFI_EXTRA_CXX_API static void VisitContextSymbols(
         const ffi::TypedFunction<void(String, void*)>& callback);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Module, ObjectRef, ModuleObj);
   };
   
   /*
    * \brief Symbols for library module.
    */
   namespace symbol {
   constexpr const char* tvm_ffi_symbol_prefix = "__tvm_ffi_";
   // Special symbols have one extra _ prefix to avoid conflict with user symbols
   constexpr const char* tvm_ffi_main = "__tvm_ffi_main";
   constexpr const char* tvm_ffi_library_ctx = "__tvm_ffi__library_ctx";
   constexpr const char* tvm_ffi_library_bin = "__tvm_ffi__library_bin";
   constexpr const char* tvm_ffi_metadata_prefix = "__tvm_ffi__metadata_";
   }  // namespace symbol
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_EXTRA_MODULE_H_
