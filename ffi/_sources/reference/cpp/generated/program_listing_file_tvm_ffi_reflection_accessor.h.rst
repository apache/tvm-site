
.. _program_listing_file_tvm_ffi_reflection_accessor.h:

Program Listing for File accessor.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_reflection_accessor.h>` (``tvm/ffi/reflection/accessor.h``)

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
   #ifndef TVM_FFI_REFLECTION_ACCESSOR_H_
   #define TVM_FFI_REFLECTION_ACCESSOR_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/type_traits.h>
   
   #include <string>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   namespace reflection {
   
   inline const TVMFFIFieldInfo* GetFieldInfo(std::string_view type_key, const char* field_name) {
     int32_t type_index;
     TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
     TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
     const TypeInfo* info = TVMFFIGetTypeInfo(type_index);
     for (int32_t i = 0; i < info->num_fields; ++i) {
       if (std::strncmp(info->fields[i].name.data, field_name, info->fields[i].name.size) == 0) {
         return &(info->fields[i]);
       }
     }
     TVM_FFI_THROW(RuntimeError) << "Cannot find field  `" << field_name << "` in " << type_key;
     TVM_FFI_UNREACHABLE();
   }
   
   class FieldGetter {
    public:
     explicit FieldGetter(const TVMFFIFieldInfo* field_info) : field_info_(field_info) {}
   
     explicit FieldGetter(std::string_view type_key, const char* field_name)
         : FieldGetter(GetFieldInfo(type_key, field_name)) {}
   
     Any operator()(const Object* obj_ptr) const {
       Any result;
       const void* addr = reinterpret_cast<const char*>(obj_ptr) + field_info_->offset;
       TVM_FFI_CHECK_SAFE_CALL(
           field_info_->getter(const_cast<void*>(addr), reinterpret_cast<TVMFFIAny*>(&result)));
       return result;
     }
   
     Any operator()(const ObjectPtr<Object>& obj_ptr) const { return operator()(obj_ptr.get()); }
   
     Any operator()(const ObjectRef& obj) const { return operator()(obj.get()); }
   
    private:
     const TVMFFIFieldInfo* field_info_;
   };
   
   class FieldSetter {
    public:
     explicit FieldSetter(const TVMFFIFieldInfo* field_info) : field_info_(field_info) {}
   
     explicit FieldSetter(std::string_view type_key, const char* field_name)
         : FieldSetter(GetFieldInfo(type_key, field_name)) {}
   
     void operator()(const Object* obj_ptr, AnyView value) const {
       const void* addr = reinterpret_cast<const char*>(obj_ptr) + field_info_->offset;
       TVM_FFI_CHECK_SAFE_CALL(
           field_info_->setter(const_cast<void*>(addr), reinterpret_cast<const TVMFFIAny*>(&value)));
     }
   
     void operator()(const ObjectPtr<Object>& obj_ptr, AnyView value) const {
       operator()(obj_ptr.get(), value);
     }
   
     void operator()(const ObjectRef& obj, AnyView value) const { operator()(obj.get(), value); }
   
    private:
     const TVMFFIFieldInfo* field_info_;
   };
   
   class TypeAttrColumn {
    public:
     explicit TypeAttrColumn(std::string_view attr_name) {
       TVMFFIByteArray attr_name_array = {attr_name.data(), attr_name.size()};
       column_ = TVMFFIGetTypeAttrColumn(&attr_name_array);
       if (column_ == nullptr) {
         TVM_FFI_THROW(RuntimeError) << "Cannot find type attribute " << attr_name;
       }
     }
     AnyView operator[](int32_t type_index) const {
       size_t tindex = static_cast<size_t>(type_index);
       if (tindex >= column_->size) {
         return AnyView();
       }
       const AnyView* any_view_data = reinterpret_cast<const AnyView*>(column_->data);
       return any_view_data[tindex];
     }
   
    private:
     const TVMFFITypeAttrColumn* column_;
   };
   
   inline const TVMFFIMethodInfo* GetMethodInfo(std::string_view type_key, const char* method_name) {
     int32_t type_index;
     TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
     TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
     const TypeInfo* info = TVMFFIGetTypeInfo(type_index);
     for (int32_t i = 0; i < info->num_methods; ++i) {
       if (std::strncmp(info->methods[i].name.data, method_name, info->methods[i].name.size) == 0) {
         return &(info->methods[i]);
       }
     }
     TVM_FFI_THROW(RuntimeError) << "Cannot find method " << method_name << " in " << type_key;
     TVM_FFI_UNREACHABLE();
   }
   
   inline Function GetMethod(std::string_view type_key, const char* method_name) {
     const TVMFFIMethodInfo* info = GetMethodInfo(type_key, method_name);
     return AnyView::CopyFromTVMFFIAny(info->method).cast<Function>();
   }
   
   template <typename Callback>
   inline void ForEachFieldInfo(const TypeInfo* type_info, Callback callback) {
     using ResultType = decltype(callback(type_info->fields));
     static_assert(std::is_same_v<ResultType, void>, "Callback must return void");
     // iterate through acenstors in parent to child order
     // skip the first one since it is always the root object
     for (int i = 1; i < type_info->type_depth; ++i) {
       const TVMFFITypeInfo* parent_info = type_info->type_ancestors[i];
       for (int j = 0; j < parent_info->num_fields; ++j) {
         callback(parent_info->fields + j);
       }
     }
     for (int i = 0; i < type_info->num_fields; ++i) {
       callback(type_info->fields + i);
     }
   }
   
   template <typename Callback>
   inline bool ForEachFieldInfoWithEarlyStop(const TypeInfo* type_info,
                                             Callback callback_with_early_stop) {
     // iterate through acenstors in parent to child order
     // skip the first one since it is always the root object
     for (int i = 1; i < type_info->type_depth; ++i) {
       const TVMFFITypeInfo* parent_info = type_info->type_ancestors[i];
       for (int j = 0; j < parent_info->num_fields; ++j) {
         if (callback_with_early_stop(parent_info->fields + j)) return true;
       }
     }
     for (int i = 0; i < type_info->num_fields; ++i) {
       if (callback_with_early_stop(type_info->fields + i)) return true;
     }
     return false;
   }
   
   }  // namespace reflection
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_REFLECTION_ACCESSOR_H_
