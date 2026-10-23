
.. _program_listing_file_tvm_ir_expr_functor.h:

Program Listing for File expr_functor.h
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_expr_functor.h>` (``tvm/ir/expr_functor.h``)

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
   #ifndef TVM_IR_EXPR_FUNCTOR_H_
   #define TVM_IR_EXPR_FUNCTOR_H_
   #include <tvm/ir/expr.h>
   #include <tvm/ir/object_functor.h>
   #include <tvm/ir/op.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/ir/prim/vector_expr.h>
   
   namespace tvm {
   
   template <typename FType>
   class ExprFunctor;
   
   template <typename R, typename... Args>
   class ExprFunctor<R(const Expr&, Args...)> {
    private:
     using TSelf = ExprFunctor<R(const Expr&, Args...)>;
   
    public:
     using result_type = R;
     ExprFunctor() : ExprFunctor(GlobalVTable()) {}
     TVM_FFI_INLINE R operator()(const Expr& node, Args... args) {
       return Dispatch(node, std::forward<Args>(args)...);
     }
     TVM_FFI_INLINE virtual R Dispatch(const Expr& node, Args... args) {
       TVM_FFI_ICHECK(node.defined()) << "Cannot dispatch a null expression";
       return (*vtable_)(node, this, std::forward<Args>(args)...);
     }
   
     virtual R Dispatch_(const LambdaExprObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const OpaqueExprObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TupleObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TupleGetItemObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TensorRegionObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TensorLoadObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const VarObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const GlobalVarObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const CallObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const GenericConstObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const DataTypeImmObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const IntImmObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const FloatImmObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const OpObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const StringImmObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::CastObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::AddObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LShiftObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::RShiftObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseAndObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseOrObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseXorObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseNotObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::SubObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::MulObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::DivObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::ModObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::FloorDivObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::FloorModObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::MinObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::MaxObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::EQObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::NEObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LTObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LEObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::GTObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::GEObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::AndObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::OrObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::NotObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::SelectObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LetObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::RampObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BroadcastObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::ShuffleObj* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
   
     virtual R DispatchDefault_(const ffi::Object* node, Args... args) {
       TVM_FFI_THROW(InternalError) << "Do not have a default for " << node->GetTypeKey();
       TVM_FFI_UNREACHABLE();
     }
   
    protected:
     using VTable = ObjectFunctor<R(const ffi::ObjectRef&, TSelf*, Args...)>;
     explicit ExprFunctor(const VTable* vtable) : vtable_(vtable) {}
     static void InitVTable(VTable* vtable) {
       SetDispatch<TSelf, LambdaExprObj>(vtable);
       SetDispatch<TSelf, OpaqueExprObj>(vtable);
       SetDispatch<TSelf, TupleObj>(vtable);
       SetDispatch<TSelf, TupleGetItemObj>(vtable);
       SetDispatch<TSelf, TensorLoadObj>(vtable);
       SetDispatch<TSelf, TensorRegionObj>(vtable);
       SetDispatch<TSelf, VarObj>(vtable);
       SetDispatch<TSelf, GlobalVarObj>(vtable);
       SetDispatch<TSelf, CallObj>(vtable);
       SetDispatch<TSelf, GenericConstObj>(vtable);
       SetDispatch<TSelf, DataTypeImmObj>(vtable);
       SetDispatch<TSelf, IntImmObj>(vtable);
       SetDispatch<TSelf, FloatImmObj>(vtable);
       SetDispatch<TSelf, OpObj>(vtable);
       SetDispatch<TSelf, StringImmObj>(vtable);
       SetDispatch<TSelf, prim::CastObj>(vtable);
       SetDispatch<TSelf, prim::AddObj>(vtable);
       SetDispatch<TSelf, prim::LShiftObj>(vtable);
       SetDispatch<TSelf, prim::RShiftObj>(vtable);
       SetDispatch<TSelf, prim::BitwiseAndObj>(vtable);
       SetDispatch<TSelf, prim::BitwiseOrObj>(vtable);
       SetDispatch<TSelf, prim::BitwiseXorObj>(vtable);
       SetDispatch<TSelf, prim::BitwiseNotObj>(vtable);
       SetDispatch<TSelf, prim::SubObj>(vtable);
       SetDispatch<TSelf, prim::MulObj>(vtable);
       SetDispatch<TSelf, prim::DivObj>(vtable);
       SetDispatch<TSelf, prim::ModObj>(vtable);
       SetDispatch<TSelf, prim::FloorDivObj>(vtable);
       SetDispatch<TSelf, prim::FloorModObj>(vtable);
       SetDispatch<TSelf, prim::MinObj>(vtable);
       SetDispatch<TSelf, prim::MaxObj>(vtable);
       SetDispatch<TSelf, prim::EQObj>(vtable);
       SetDispatch<TSelf, prim::NEObj>(vtable);
       SetDispatch<TSelf, prim::LTObj>(vtable);
       SetDispatch<TSelf, prim::LEObj>(vtable);
       SetDispatch<TSelf, prim::GTObj>(vtable);
       SetDispatch<TSelf, prim::GEObj>(vtable);
       SetDispatch<TSelf, prim::AndObj>(vtable);
       SetDispatch<TSelf, prim::OrObj>(vtable);
       SetDispatch<TSelf, prim::NotObj>(vtable);
       SetDispatch<TSelf, prim::SelectObj>(vtable);
       SetDispatch<TSelf, prim::LetObj>(vtable);
       SetDispatch<TSelf, prim::RampObj>(vtable);
       SetDispatch<TSelf, prim::BroadcastObj>(vtable);
       SetDispatch<TSelf, prim::ShuffleObj>(vtable);
     }
     template <typename Self, typename Node>
     static void SetDispatch(VTable* vtable) {
       vtable->template SetDispatch<Node>(
           [](const ffi::ObjectRef& node, TSelf* self, Args... args) -> R {
             return static_cast<Self*>(self)->Dispatch_(static_cast<const Node*>(node.get()),
                                                        std::forward<Args>(args)...);
           });
     }
   
    private:
     static const VTable* GlobalVTable() {
       static const VTable table = [] {
         VTable table;
         InitVTable(&table);
         table.Finalize();
         return table;
       }();
       return &table;
     }
     const VTable* vtable_;
   };
   
   class TVM_TIRX_DLL ExprVisitor : public ObjectVisitor {
    public:
     TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(ExprVisitor, ObjectVisitor)
   
     using ObjectVisitor::Visit;
   
     // Override existing hooks directly. Extra types need a fresh inherited table.
     // Hooks borrow the node and return None or an owning interrupt, and throw on failure.
     virtual ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TupleObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TensorRegionObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const VarObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const GlobalVarObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const CallObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const GenericConstObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const DataTypeImmObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const IntImmObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const FloatImmObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const OpObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const StringImmObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::CastObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AddObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LShiftObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::RShiftObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseAndObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseOrObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseXorObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseNotObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SubObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MulObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::DivObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ModObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorDivObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorModObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MinObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MaxObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::EQObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NEObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LTObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LEObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GTObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GEObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AndObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::OrObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NotObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SelectObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LetObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::RampObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastObj* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleObj* node);
   
    protected:
     explicit ExprVisitor(const VTable* vtable) : ObjectVisitor(vtable) {}
     static void InitVTable(VTable* vtable);
   };
   
   class TVM_TIRX_DLL ExprMutator : public ObjectMutator {
    public:
     TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(ExprMutator, ObjectMutator)
   
     using ObjectMutator::Mutate;
   
     TVM_FFI_INLINE UnchangedOr<PrimExpr> Mutate(const PrimExpr& expr,
                                                 InplaceMode inplace_mode = InplaceMode::kDisallow) {
       return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<PrimExpr>(
           ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(expr), inplace_mode)));
     }
   
     TVM_FFI_INLINE UnchangedOr<Expr> Mutate(const Expr& expr,
                                             InplaceMode inplace_mode = InplaceMode::kDisallow) {
       return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<Expr>(
           ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(expr), inplace_mode)));
     }
   
     // A downstream class overrides any existing hook without rebuilding the table.
     // Extra node types use a fresh inherited table and SetDispatch<Self, ExtraObj>.
     // Hooks borrow the node and return Unchanged or an owning replacement in its expression
     // category, throwing on failure. Narrower field types are checked separately. Forward
     // inplace_mode on every child edge.
     virtual UnchangedOr<Expr> Mutate_(const OpaqueExprObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const TupleObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const TupleGetItemObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const TensorRegionObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const VarObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const GlobalVarObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const CallObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const GenericConstObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const DataTypeImmObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const IntImmObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const FloatImmObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const OpObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const StringImmObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::CastObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LShiftObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::RShiftObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::SubObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::MulObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::DivObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::ModObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorModObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::MinObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::MaxObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::AndObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::OrObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::NotObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LetObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::RampObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastObj* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleObj* node, InplaceMode inplace_mode);
   
    protected:
     static UnchangedOr<Expr> ReinferMutatedCallType(UnchangedOr<Expr> mutated,
                                                     const CallObj* original,
                                                     InplaceMode inplace_mode);
   
     explicit ExprMutator(const VTable* vtable) : ObjectMutator(vtable) {}
     static void InitVTable(VTable* vtable);
   };
   
   }  // namespace tvm
   #endif  // TVM_IR_EXPR_FUNCTOR_H_
