
.. _program_listing_file_tvm_tirx_structural_visit_with_constraint_context.h:

Program Listing for File structural_visit_with_constraint_context.h
===================================================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_structural_visit_with_constraint_context.h>` (``tvm/tirx/structural_visit_with_constraint_context.h``)

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
   
   #ifndef TVM_TIRX_STRUCTURAL_VISIT_WITH_CONSTRAINT_CONTEXT_H_
   #define TVM_TIRX_STRUCTURAL_VISIT_WITH_CONSTRAINT_CONTEXT_H_
   
   #include <tvm/ffi/extra/structural_visit.h>
   #include <tvm/sym/constraint.h>
   
   namespace tvm {
   namespace tirx {
   
   class StructuralVisitorWithConstraintContextObj : public ffi::StructuralVisitorObj {
    public:
     using VisitorObjType = StructuralVisitorWithConstraintContextObj;
     using StateTupleType = std::tuple<const sym::ConstraintContext&>;
     StructuralVisitorWithConstraintContextObj();
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
   
     ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> DefaultVisitExpected(
         ffi::AnyView value) noexcept;
     TVM_FFI_DECLARE_OBJECT_INFO("tirx.StructuralVisitorWithConstraintContext",
                                 StructuralVisitorWithConstraintContextObj, ffi::StructuralVisitorObj);
   
    protected:
     explicit StructuralVisitorWithConstraintContextObj(const ffi::StructuralVisitorVTable* vtable)
         : ffi::StructuralVisitorObj(vtable) {}
   
     StateTupleType StateTuple() const noexcept { return {constraints}; }
   
    private:
     bool needs_child_context_{false};
   };
   
   class StructuralVisitorWithConstraintContext : public ffi::StructuralVisitor {
    public:
     explicit StructuralVisitorWithConstraintContext(
         ffi::ObjectPtr<StructuralVisitorWithConstraintContextObj> node)
         : ffi::StructuralVisitor(std::move(node)) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralVisitorWithConstraintContext,
                                                   ffi::StructuralVisitor,
                                                   StructuralVisitorWithConstraintContextObj);
   };
   
   template <ffi::WalkOrder order, typename... Callbacks>
   ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> StructuralWalkWithConstraintContextExpected(
       ffi::AnyView root, Callbacks&&... callbacks) noexcept {
     static_assert(sizeof...(Callbacks) != 0, "At least one callback is required");
     using Engine = ffi::StructuralWalkEngine<StructuralVisitorWithConstraintContextObj, order,
                                              std::decay_t<Callbacks>...>;
     auto visitor = ffi::make_object<Engine>(std::forward<Callbacks>(callbacks)...);
     return visitor->VisitExpected(root);
   }
   
   template <ffi::WalkOrder order, typename... Callbacks>
   ffi::Optional<ffi::VisitInterrupt> StructuralWalkWithConstraintContext(ffi::AnyView root,
                                                                          Callbacks&&... callbacks) {
     return StructuralWalkWithConstraintContextExpected<order>(root,
                                                               std::forward<Callbacks>(callbacks)...)
         .value();
   }
   
   template <typename... Callbacks>
   ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> StructuralVisitWithConstraintContextExpected(
       ffi::AnyView root, Callbacks&&... callbacks) noexcept {
     static_assert(sizeof...(Callbacks) != 0, "At least one callback is required");
     using Engine = ffi::StructuralVisitEngine<StructuralVisitorWithConstraintContextObj,
                                               std::decay_t<Callbacks>...>;
     auto visitor = ffi::make_object<Engine>(std::forward<Callbacks>(callbacks)...);
     return visitor->VisitExpected(root);
   }
   
   template <typename... Callbacks>
   ffi::Optional<ffi::VisitInterrupt> StructuralVisitWithConstraintContext(ffi::AnyView root,
                                                                           Callbacks&&... callbacks) {
     return StructuralVisitWithConstraintContextExpected(root, std::forward<Callbacks>(callbacks)...)
         .value();
   }
   
   using FStructuralVisitWithConstraintContext =
       ffi::reflection::NativeFunctionView<ffi::Optional<ffi::VisitInterrupt>(
           StructuralVisitorWithConstraintContextObj*, ffi::AnyView)>;
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<FStructuralVisitWithConstraintContext>
       kStructuralVisitWithConstraintContext{"__s_visit_with_constraint_context__"};
   }  // namespace type_attr
   
   inline StructuralVisitorWithConstraintContextObj::StructuralVisitorWithConstraintContextObj()
       : StructuralVisitorWithConstraintContextObj([] {
           static const ffi::StructuralVisitorVTable table{[](ffi::StructuralVisitorObj* self,
                                                              ffi::AnyView value) noexcept {
             return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
                 static_cast<StructuralVisitorWithConstraintContextObj*>(self)->DefaultVisitExpected(
                     value));
           }};
           return &table;
         }()) {}
   
   inline ffi::Expected<ffi::Optional<ffi::VisitInterrupt>>
   StructuralVisitorWithConstraintContextObj::DefaultVisitExpected(ffi::AnyView value) noexcept {
     static ffi::reflection::TypeAttrColumn column(type_attr::kStructuralVisitWithConstraintContext);
     auto attr = column[value.type_index()];
     if (attr == nullptr) return ffi::StructuralVisitorObj::DefaultVisitExpected(value);
     auto hook = attr.try_cast<FStructuralVisitWithConstraintContext>();
     if (!hook) {
       return ffi::Unexpected(ffi::Error(
           "TypeError",
           "__s_visit_with_constraint_context__ must be an opaque function pointer or ffi.Function",
           ""));
     }
     return hook->CallExpected(this, value);
   }
   
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_STRUCTURAL_VISIT_WITH_CONSTRAINT_CONTEXT_H_
