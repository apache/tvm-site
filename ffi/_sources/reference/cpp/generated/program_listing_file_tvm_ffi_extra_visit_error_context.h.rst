
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
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/extra/base.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/reflection/access_path.h>
   #include <tvm/ffi/reflection/accessor.h>
   
   namespace tvm {
   namespace ffi {
   namespace type_attr {
   
   inline constexpr reflection::AttrKey<TypedFunction<Error(Error, ObjectRef)>> kEnrichVisitError{
       "__ffi_enrich_visit_error__"};
   
   }  // namespace type_attr
   
   class VisitErrorContextObj : public Object {
    public:
     Array<ObjectRef> reverse_visit_pattern;
   
     Optional<ObjectRef> prev_error_context;
   
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
     static Error EnrichVisitError(Error error, const ObjectRef& root) noexcept {
       if (!root.defined() || !VisitErrorContext::TryGetFromError(error)) {
         return error;
       }
       // Diagnostics must not replace the original failure if the hook itself fails.
       try {
         static reflection::TypeAttrColumn column(type_attr::kEnrichVisitError);
         if (auto hook = type_attr::kEnrichVisitError.ValueFrom(column[root->type_index()])) {
           return (*hook)(error, root);
         }
       } catch (...) {
         return error;
       }
       return error;
     }
   
     TVM_FFI_COLD_CODE
     TVM_FFI_EXTRA_CXX_API static Array<reflection::AccessPath> FindAccessPaths(
         const ObjectRef& root, const VisitErrorContext& visit_context,
         bool allow_prefix_match = false);
   
     explicit VisitErrorContext(ObjectPtr<VisitErrorContextObj> n) : ObjectRef(std::move(n)) {
       TVM_FFI_ICHECK_NOTNULL(data_.get());
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisitErrorContext, ObjectRef, VisitErrorContextObj);
   };
   
   #define TVM_FFI_VISIT_BEGIN() try {
   #define TVM_FFI_VISIT_END(node)                                                \
     }                                                                            \
     catch (::tvm::ffi::Error & _tvm_ffi_visit_err_) {                            \
       ::tvm::ffi::detail::UpdateVisitErrorContext(&_tvm_ffi_visit_err_, (node)); \
       throw;                                                                     \
     }
   
   #define TVM_FFI_VISIT_END_RETURN_EXPECTED(node)                                \
     }                                                                            \
     catch (::tvm::ffi::Error & _tvm_ffi_visit_err_) {                            \
       ::tvm::ffi::detail::UpdateVisitErrorContext(&_tvm_ffi_visit_err_, (node)); \
       return ::tvm::ffi::Unexpected(_tvm_ffi_visit_err_);                        \
     }
   
   #define TVM_FFI_VISIT_THROW(ErrorKind, node)                                                   \
     ::tvm::ffi::detail::ErrorBuilder(                                                            \
         #ErrorKind, TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0),                    \
         TVM_FFI_ALWAYS_LOG_BEFORE_THROW, ::std::nullopt,                                         \
         ::std::optional<::tvm::ffi::ObjectRef>(::tvm::ffi::detail::MakeVisitErrorContext(node))) \
         .stream()
   
   namespace detail {
   TVM_FFI_COLD_CODE
   inline VisitErrorContext MakeVisitErrorContext(const ObjectRef& node) {
     ObjectPtr<VisitErrorContextObj> obj = make_object<VisitErrorContextObj>();
     obj->reverse_visit_pattern = Array<ObjectRef>{node};
     return VisitErrorContext(std::move(obj));
   }
   
   TVM_FFI_COLD_CODE
   inline void UpdateVisitErrorContext(Error* err, const ObjectRef& node) {  // NOLINT(*)
     // Callback, policy and default descent can report the same frame.
     const Object* current = static_cast<const Object*>(
         static_cast<ErrorObj*>(detail::ObjectUnsafe::RawObjectPtrFromObjectRef(*err))->extra_context);
     if (current != nullptr && current->IsInstance<VisitErrorContextObj>()) {
       const Array<ObjectRef>& pattern =
           static_cast<const VisitErrorContextObj*>(current)->reverse_visit_pattern;
       if (!pattern.empty() && pattern.back().same_as(node)) return;
     }
     // Copy-on-write at each level: the Error is replaced by a shallow copy when shared, and its
     // VisitErrorContext likewise. Each level is borrowed through its parent's field, so an Error
     // owned only by the unwinding frame is updated in place.
     if (!err->unique()) {
       *err = Error(err->kind(), err->message(), err->backtrace(), err->cause_chain(),
                    err->extra_context());
     }
     ErrorObj* error_obj =
         static_cast<ErrorObj*>(detail::ObjectUnsafe::RawObjectPtrFromObjectRef(*err));
     Object* extra_context = static_cast<Object*>(error_obj->extra_context);
     if (extra_context != nullptr && extra_context->IsInstance<VisitErrorContextObj>()) {
       VisitErrorContextObj* visit_context = static_cast<VisitErrorContextObj*>(extra_context);
       if (!visit_context->unique()) {
         ObjectPtr<VisitErrorContextObj> copy = make_object<VisitErrorContextObj>(*visit_context);
         visit_context = copy.get();
         detail::ObjectUnsafe::DecRefObjectHandle(error_obj->extra_context);
         error_obj->extra_context =
             detail::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(copy));
       }
       visit_context->reverse_visit_pattern.push_back(node);
       return;
     }
     // Build a fresh VisitErrorContext, preserving any pre-existing payload.
     ObjectPtr<VisitErrorContextObj> new_context = make_object<VisitErrorContextObj>();
     new_context->reverse_visit_pattern = Array<ObjectRef>{node};
     if (extra_context != nullptr) {
       new_context->prev_error_context = detail::ObjectUnsafe::ObjectRefFromObjectPtr<ObjectRef>(
           detail::ObjectUnsafe::ObjectPtrFromUnowned<Object>(extra_context));
       detail::ObjectUnsafe::DecRefObjectHandle(error_obj->extra_context);
     }
     error_obj->extra_context =
         detail::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(new_context));
   }
   
   TVM_FFI_COLD_CODE inline void UpdateVisitErrorContext(TVMFFIAny* result, AnyView node) noexcept {
     if (node.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
       // Take the Error out of the result and store it back, so one owned only by the result is
       // updated in place.
       Error err = detail::ObjectUnsafe::ObjectRefFromObjectPtr<Error>(
           detail::ObjectUnsafe::ObjectPtrFromOwned<Object>(result->v_obj));
       UpdateVisitErrorContext(&err, node.cast<ObjectRef>());
       result->v_obj = detail::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(err));
     }
   }
   
   TVM_FFI_COLD_CODE inline void UpdateVisitErrorContext(Error* err,
                                                         AnyView node) noexcept {  // NOLINT(*)
     if (node.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
       UpdateVisitErrorContext(err, node.cast<ObjectRef>());
     }
   }
   
   template <typename T>
   TVM_FFI_COLD_CODE inline void UpdateVisitErrorContext(Expected<T>* result, AnyView node) noexcept {
     // Move the Error out and back, so one owned only by the result is updated in place.
     Error err = std::move(*result).error();
     UpdateVisitErrorContext(&err, node);
     *result = Unexpected<Error>(std::move(err));
   }
   }  // namespace detail
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_EXTRA_VISIT_ERROR_CONTEXT_H_
