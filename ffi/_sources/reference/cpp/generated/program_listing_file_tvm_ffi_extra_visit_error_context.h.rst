
.. _program_listing_file_tvm_ffi_extra_visit_error_context.h:

Program Listing for File visit_error_context.h
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_extra_visit_error_context.h>` (``tvm/ffi/extra/visit_error_context.h``)

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
   #ifndef TVM_FFI_EXTRA_VISIT_ERROR_CONTEXT_H_
   #define TVM_FFI_EXTRA_VISIT_ERROR_CONTEXT_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/list.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/extra/base.h>
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/reflection/access_path.h>
   
   namespace tvm {
   namespace ffi {
   
   class VisitErrorContextObj : public Object {
    public:
     List<ObjectRef> reverse_visit_pattern;
   
     Optional<ObjectRef> prev_error_context;
   
     static constexpr bool _type_mutable = true;
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUnsupported;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.VisitErrorContext", VisitErrorContextObj, Object);
   };
   
   class VisitErrorContext : public ObjectRef {
    public:
     TVM_FFI_COLD_CODE
     static Optional<VisitErrorContext> TryGetFromError(const Error& err) {
       std::optional<ObjectRef> extra_context = err.extra_context();
       if (extra_context) {
         return extra_context->as<VisitErrorContext>();
       }
       return std::nullopt;
     }
   
     TVM_FFI_COLD_CODE
     TVM_FFI_EXTRA_CXX_API static Array<reflection::AccessPath> FindAccessPaths(
         const ObjectRef& root, const VisitErrorContext& visit_context,
         bool allow_prefix_match = false);
   
     explicit VisitErrorContext(ObjectPtr<VisitErrorContextObj> n) : ObjectRef(std::move(n)) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisitErrorContext, ObjectRef, VisitErrorContextObj);
   };
   
   #define TVM_FFI_VISIT_BEGIN() try {
   #define TVM_FFI_VISIT_END(node)                                                \
     }                                                                            \
     catch (::tvm::ffi::Error & _tvm_ffi_visit_err_) {                            \
       ::tvm::ffi::details::UpdateVisitErrorContext(_tvm_ffi_visit_err_, (node)); \
       throw;                                                                     \
     }
   
   #define TVM_FFI_VISIT_END_RETURN_EXPECTED(node)                                \
     }                                                                            \
     catch (::tvm::ffi::Error & _tvm_ffi_visit_err_) {                            \
       ::tvm::ffi::details::UpdateVisitErrorContext(_tvm_ffi_visit_err_, (node)); \
       return ::tvm::ffi::Unexpected(_tvm_ffi_visit_err_);                        \
     }
   
   #define TVM_FFI_VISIT_THROW(ErrorKind, node)                                                    \
     ::tvm::ffi::details::ErrorBuilder(                                                            \
         #ErrorKind, TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0),                     \
         TVM_FFI_ALWAYS_LOG_BEFORE_THROW, ::std::nullopt,                                          \
         ::std::optional<::tvm::ffi::ObjectRef>(::tvm::ffi::details::MakeVisitErrorContext(node))) \
         .stream()
   
   namespace details {
   TVM_FFI_COLD_CODE
   inline VisitErrorContext MakeVisitErrorContext(const ObjectRef& node) {
     ObjectPtr<VisitErrorContextObj> obj = make_object<VisitErrorContextObj>();
     obj->reverse_visit_pattern = List<ObjectRef>{node};
     return VisitErrorContext(std::move(obj));
   }
   
   TVM_FFI_COLD_CODE
   inline void UpdateVisitErrorContext(Error& err, const ObjectRef& node) {  // NOLINT(*)
     // NOTE: This function mutates the ErrorObj in place via ObjectUnsafe.
     // Expected to run only inside the exception throw chain, where the Error
     // is single-owned by this thread. The tradeoff avoids reallocating a
     // fresh Error per catch frame; the immutability invariant returns once
     // the unwind window closes.
     std::optional<ObjectRef> extra_context = err.extra_context();
     if (extra_context) {
       Optional<VisitErrorContext> visit_context = extra_context->as<VisitErrorContext>();
       if (visit_context) {
         visit_context.value()->reverse_visit_pattern.push_back(node);
         return;
       }
     }
     // Build a fresh VisitErrorContext, preserving any pre-existing payload.
     ObjectPtr<VisitErrorContextObj> new_context = make_object<VisitErrorContextObj>();
     new_context->reverse_visit_pattern = List<ObjectRef>{node};
     if (extra_context) new_context->prev_error_context = *extra_context;
   
     ErrorObj* error_obj =
         static_cast<ErrorObj*>(details::ObjectUnsafe::RawObjectPtrFromObjectRef(err));
     if (error_obj->extra_context != nullptr) {
       details::ObjectUnsafe::DecRefObjectHandle(error_obj->extra_context);
     }
     error_obj->extra_context =
         details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(new_context));
   }
   
   TVM_FFI_COLD_CODE inline void UpdateVisitErrorContext(const TVMFFIAny& result,
                                                         AnyView node) noexcept {
     if (node.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
       Error err = AnyView::CopyFromTVMFFIAny(result).cast<Error>();
       UpdateVisitErrorContext(err, node.cast<ObjectRef>());
     }
   }
   
   TVM_FFI_COLD_CODE inline void UpdateVisitErrorContext(Error& err,
                                                         AnyView node) noexcept {  // NOLINT(*)
     if (node.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
       UpdateVisitErrorContext(err, node.cast<ObjectRef>());
     }
   }
   }  // namespace details
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_VISIT_ERROR_CONTEXT_H_
