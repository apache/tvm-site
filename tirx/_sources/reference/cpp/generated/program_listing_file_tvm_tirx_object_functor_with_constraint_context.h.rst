
.. _program_listing_file_tvm_tirx_object_functor_with_constraint_context.h:

Program Listing for File object_functor_with_constraint_context.h
=================================================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_object_functor_with_constraint_context.h>` (``tvm/tirx/object_functor_with_constraint_context.h``)

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
   
   #ifndef TVM_TIRX_OBJECT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
   #define TVM_TIRX_OBJECT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
   
   #include <tvm/ir/object_functor.h>
   #include <tvm/tirx/structural_mutate_with_constraint_context.h>
   
   namespace tvm {
   namespace tirx {
   class TVM_TIRX_DLL ObjectVisitorWithConstraintContext
       : public StructuralVisitorWithConstraintContextObj {
    public:
     ObjectVisitorWithConstraintContext() : ObjectVisitorWithConstraintContext(GlobalVTable()) {}
     ObjectVisitorWithConstraintContext(const ObjectVisitorWithConstraintContext&) = delete;
     ObjectVisitorWithConstraintContext& operator=(const ObjectVisitorWithConstraintContext&) = delete;
   
     virtual ffi::Optional<ffi::VisitInterrupt> Visit(ffi::AnyView value);
     ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> VisitExpected(ffi::AnyView value) noexcept {
       try {
         return Visit(value);
       } catch (ffi::Error& error) {
         return ffi::Unexpected(std::move(error));
       } catch (const std::exception& error) {
         return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
       }
     }
   
    protected:
     using VTable = ObjectFunctor<ffi::Optional<ffi::VisitInterrupt>(
         const ffi::Object*, ObjectVisitorWithConstraintContext*)>;
   
     explicit ObjectVisitorWithConstraintContext(const VTable* table)
         : StructuralVisitorWithConstraintContextObj(StructuralVTable()), native_vtable_(table) {}
   
     static void InitVTable(VTable* table) {}
   
     template <typename Self, typename Node>
     static void SetDispatch(VTable* table) {
       table->SetDispatch<Node>([](const ffi::Object* node, ObjectVisitorWithConstraintContext* self) {
         return static_cast<Self*>(self)->Visit_(static_cast<const Node*>(node));
       });
     }
   
    private:
     static const VTable* GlobalVTable();
     static const ffi::StructuralVisitorVTable* StructuralVTable();
     const VTable* const native_vtable_;
   };
   
   class TVM_TIRX_DLL ObjectMutatorWithConstraintContext
       : public StructuralMutatorWithConstraintContextObj {
    public:
     ObjectMutatorWithConstraintContext() : ObjectMutatorWithConstraintContext(GlobalVTable()) {}
     ObjectMutatorWithConstraintContext(const ObjectMutatorWithConstraintContext&) = delete;
     ObjectMutatorWithConstraintContext& operator=(const ObjectMutatorWithConstraintContext&) = delete;
   
     virtual ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                                               ffi::InplaceMode mode = ffi::InplaceMode::kDisallow);
     ffi::Expected<ffi::UnchangedOr<ffi::Any>> MutateExpected(
         ffi::AnyView value, ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) noexcept {
       try {
         return Mutate(value, mode);
       } catch (ffi::Error& error) {
         return ffi::Unexpected(std::move(error));
       } catch (const std::exception& error) {
         return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
       }
     }
   
    protected:
     using VTable = ObjectFunctor<ffi::UnchangedOr<ffi::Any>(
         const ffi::Object*, ObjectMutatorWithConstraintContext*, ffi::InplaceMode)>;
   
     explicit ObjectMutatorWithConstraintContext(const VTable* table)
         : StructuralMutatorWithConstraintContextObj(StructuralVTable()), native_vtable_(table) {}
   
     static void InitVTable(VTable* table) {}
   
     template <typename Self, typename Node>
     static void SetDispatch(VTable* table) {
       table->SetDispatch<Node>([](const ffi::Object* node, ObjectMutatorWithConstraintContext* self,
                                   ffi::InplaceMode mode) -> ffi::UnchangedOr<ffi::Any> {
         return static_cast<Self*>(self)->Mutate_(static_cast<const Node*>(node), mode);
       });
     }
   
    private:
     static const VTable* GlobalVTable();
     static const ffi::StructuralMutatorVTable* StructuralVTable();
     const VTable* const native_vtable_;
   };
   
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_OBJECT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
