
.. _program_listing_file_tvm_tirx_structural_mutate_with_constraint_context.h:

Program Listing for File structural_mutate_with_constraint_context.h
====================================================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_structural_mutate_with_constraint_context.h>` (``tvm/tirx/structural_mutate_with_constraint_context.h``)

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
   
   #ifndef TVM_TIRX_STRUCTURAL_MUTATE_WITH_CONSTRAINT_CONTEXT_H_
   #define TVM_TIRX_STRUCTURAL_MUTATE_WITH_CONSTRAINT_CONTEXT_H_
   
   #include <tvm/ffi/extra/structural_mutate.h>
   #include <tvm/tirx/structural_visit_with_constraint_context.h>
   
   namespace tvm {
   namespace tirx {
   
   class StructuralMutatorWithConstraintContextObj : public ffi::StructuralMapEngineBase {
    public:
     using MutatorObjType = StructuralMutatorWithConstraintContextObj;
     using StateTupleType = std::tuple<const sym::ConstraintContext&>;
     StructuralMutatorWithConstraintContextObj();
     sym::ConstraintContext constraints;
   
     void AppendConstraint(const sym::Constraint& constraint) {
       if (needs_child_context_) {
         constraints = sym::ConstraintContext(constraints, constraints->LocalSize());
         needs_child_context_ = false;
       }
       constraints->Append(constraint);
     }
   
     template <typename F>
     auto WithConstraintScope(F&& f) -> decltype(f()) {
       sym::ConstraintContext saved_context = constraints;
       int64_t saved_prefix = constraints->LocalSize();
       bool saved_needs_child = needs_child_context_;
       needs_child_context_ = true;
       auto result = std::forward<F>(f)();
       if (!result.is_err()) {
         constraints = saved_context->LocalSize() == saved_prefix
                           ? saved_context
                           : sym::ConstraintContext(saved_context, saved_prefix);
         needs_child_context_ = saved_needs_child;
       }
       return result;
     }
   
     ffi::Expected<ffi::UnchangedOr<ffi::Any>> DefaultMutateExpected(
         ffi::AnyView value, ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) noexcept;
     TVM_FFI_DECLARE_OBJECT_INFO("tirx.StructuralMutatorWithConstraintContext",
                                 StructuralMutatorWithConstraintContextObj, ffi::StructuralMutatorObj);
   
    protected:
     explicit StructuralMutatorWithConstraintContextObj(const ffi::StructuralMutatorVTable* vtable)
         : ffi::StructuralMapEngineBase(vtable) {}
   
     StateTupleType StateTuple() const noexcept { return {constraints}; }
   
    private:
     bool needs_child_context_{false};
   };
   class StructuralMutatorWithConstraintContext : public ffi::StructuralMutator {
    public:
     explicit StructuralMutatorWithConstraintContext(
         ffi::ObjectPtr<StructuralMutatorWithConstraintContextObj> node)
         : ffi::StructuralMutator(std::move(node)) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralMutatorWithConstraintContext,
                                                   ffi::StructuralMutator,
                                                   StructuralMutatorWithConstraintContextObj);
   };
   
   template <ffi::WalkOrder order, typename... Callbacks>
   ffi::Expected<ffi::Any> StructuralMapWithConstraintContextExpected(
       ffi::Any root, Callbacks&&... callbacks) noexcept {
     static_assert(sizeof...(Callbacks) != 0, "At least one callback is required");
     using Engine = ffi::StructuralMapEngine<StructuralMutatorWithConstraintContextObj, order,
                                             std::decay_t<Callbacks>...>;
     auto mutator = ffi::make_object<Engine>(std::forward<Callbacks>(callbacks)...);
     auto result = mutator->MutateExpected(root, ffi::InplaceMode::kAllow);
     if (result.is_err()) return ffi::Unexpected(std::move(result).error());
     return std::move(result).value().ValueOrUnchanged(std::move(root));
   }
   
   template <ffi::WalkOrder order, typename... Callbacks>
   ffi::Any StructuralMapWithConstraintContext(ffi::Any root, Callbacks&&... callbacks) {
     return StructuralMapWithConstraintContextExpected<order>(std::move(root),
                                                              std::forward<Callbacks>(callbacks)...)
         .value();
   }
   
   template <typename... Callbacks>
   ffi::Expected<ffi::Any> StructuralMutateWithConstraintContextExpected(
       ffi::Any root, Callbacks&&... callbacks) noexcept {
     static_assert(sizeof...(Callbacks) != 0, "At least one callback is required");
     using Engine = ffi::StructuralMutateEngine<StructuralMutatorWithConstraintContextObj,
                                                std::decay_t<Callbacks>...>;
     auto mutator = ffi::make_object<Engine>(std::forward<Callbacks>(callbacks)...);
     auto result = mutator->MutateExpected(root, ffi::InplaceMode::kAllow);
     if (result.is_err()) return ffi::Unexpected(std::move(result).error());
     return std::move(result).value().ValueOrUnchanged(std::move(root));
   }
   
   template <typename... Callbacks>
   ffi::Any StructuralMutateWithConstraintContext(ffi::Any root, Callbacks&&... callbacks) {
     return StructuralMutateWithConstraintContextExpected(std::move(root),
                                                          std::forward<Callbacks>(callbacks)...)
         .value();
   }
   
   using FStructuralMutateWithConstraintContext =
       ffi::reflection::NativeFunctionView<ffi::UnchangedOr<ffi::Any>(
           StructuralMutatorWithConstraintContextObj*, ffi::AnyView)>;
   
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<FStructuralMutateWithConstraintContext>
       kStructuralMutateWithConstraintContext{"__s_mutate_with_constraint_context__"};
   inline constexpr ffi::reflection::AttrKey<FStructuralMutateWithConstraintContext>
       kStructuralMaybeInplaceMutateWithConstraintContext{
           "__s_maybe_inplace_mutate_with_constraint_context__"};
   }  // namespace type_attr
   
   inline StructuralMutatorWithConstraintContextObj::StructuralMutatorWithConstraintContextObj()
       : StructuralMutatorWithConstraintContextObj([] {
           static const ffi::StructuralMutatorVTable table{
               [](ffi::StructuralMutatorObj* self, ffi::AnyView value) noexcept {
                 return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
                     static_cast<StructuralMutatorWithConstraintContextObj*>(self)
                         ->DefaultMutateExpected(value, ffi::InplaceMode::kDisallow));
               },
               [](ffi::StructuralMutatorObj* self, ffi::AnyView value) noexcept {
                 return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
                     static_cast<StructuralMutatorWithConstraintContextObj*>(self)
                         ->DefaultMutateExpected(value, ffi::InplaceMode::kAllow));
               },
               DispatchVarRemapGet, DispatchVarRemapSet};
           return &table;
         }()) {}
   
   inline ffi::Expected<ffi::UnchangedOr<ffi::Any>>
   StructuralMutatorWithConstraintContextObj::DefaultMutateExpected(ffi::AnyView value,
                                                                    ffi::InplaceMode mode) noexcept {
     static ffi::reflection::TypeAttrColumn copying(type_attr::kStructuralMutateWithConstraintContext);
     static ffi::reflection::TypeAttrColumn inplace(
         type_attr::kStructuralMaybeInplaceMutateWithConstraintContext);
     auto attr = mode == ffi::InplaceMode::kAllow ? inplace[value.type_index()] : ffi::AnyView();
     if (attr == nullptr) attr = copying[value.type_index()];
     if (attr == nullptr) return ffi::StructuralMutatorObj::DefaultMutateExpected(value, mode);
     if (!attr.try_cast<FStructuralMutateWithConstraintContext>()) {
       return ffi::Unexpected(ffi::Error(
           "TypeError",
           "Constraint-aware mutation hook must be an opaque function pointer or ffi.Function", ""));
     }
     using ABI = TVMFFIAny (*)(StructuralMutatorWithConstraintContextObj*, ffi::AnyView) noexcept;
     TVMFFIAny raw;
     if (attr.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
       raw = reinterpret_cast<ABI>(attr.cast<void*>())(this, value);
     } else {
       raw = ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
           attr.cast<ffi::Function>().CallExpected<ffi::Any>(this, value));
     }
     auto result = ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::UnchangedOr<ffi::Any>>(raw);
     if (result.is_err()) {
       auto error = result.error();
       ffi::details::UpdateVisitErrorContext(error, value);
     }
     return result;
   }
   
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_STRUCTURAL_MUTATE_WITH_CONSTRAINT_CONTEXT_H_
