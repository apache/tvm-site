
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
   
     virtual R Dispatch_(const LambdaExprNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const OpaqueExprNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TupleNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TupleGetItemNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TensorRegionNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TensorLoadNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const VarNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const GlobalVarNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const CallNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const GenericConstNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const DataTypeImmNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const IntImmNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const FloatImmNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const OpNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const StringImmNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::CastNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::AddNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LShiftNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::RShiftNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseAndNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseOrNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseXorNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BitwiseNotNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::SubNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::MulNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::DivNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::ModNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::FloorDivNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::FloorModNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::MinNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::MaxNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::EQNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::NENode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LTNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LENode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::GTNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::GENode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::AndNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::OrNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::NotNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::SelectNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::LetNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::RampNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::BroadcastNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const prim::ShuffleNode* node, Args... args) {
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
       SetDispatch<TSelf, LambdaExprNode>(vtable);
       SetDispatch<TSelf, OpaqueExprNode>(vtable);
       SetDispatch<TSelf, TupleNode>(vtable);
       SetDispatch<TSelf, TupleGetItemNode>(vtable);
       SetDispatch<TSelf, TensorLoadNode>(vtable);
       SetDispatch<TSelf, TensorRegionNode>(vtable);
       SetDispatch<TSelf, VarNode>(vtable);
       SetDispatch<TSelf, GlobalVarNode>(vtable);
       SetDispatch<TSelf, CallNode>(vtable);
       SetDispatch<TSelf, GenericConstNode>(vtable);
       SetDispatch<TSelf, DataTypeImmNode>(vtable);
       SetDispatch<TSelf, IntImmNode>(vtable);
       SetDispatch<TSelf, FloatImmNode>(vtable);
       SetDispatch<TSelf, OpNode>(vtable);
       SetDispatch<TSelf, StringImmNode>(vtable);
       SetDispatch<TSelf, prim::CastNode>(vtable);
       SetDispatch<TSelf, prim::AddNode>(vtable);
       SetDispatch<TSelf, prim::LShiftNode>(vtable);
       SetDispatch<TSelf, prim::RShiftNode>(vtable);
       SetDispatch<TSelf, prim::BitwiseAndNode>(vtable);
       SetDispatch<TSelf, prim::BitwiseOrNode>(vtable);
       SetDispatch<TSelf, prim::BitwiseXorNode>(vtable);
       SetDispatch<TSelf, prim::BitwiseNotNode>(vtable);
       SetDispatch<TSelf, prim::SubNode>(vtable);
       SetDispatch<TSelf, prim::MulNode>(vtable);
       SetDispatch<TSelf, prim::DivNode>(vtable);
       SetDispatch<TSelf, prim::ModNode>(vtable);
       SetDispatch<TSelf, prim::FloorDivNode>(vtable);
       SetDispatch<TSelf, prim::FloorModNode>(vtable);
       SetDispatch<TSelf, prim::MinNode>(vtable);
       SetDispatch<TSelf, prim::MaxNode>(vtable);
       SetDispatch<TSelf, prim::EQNode>(vtable);
       SetDispatch<TSelf, prim::NENode>(vtable);
       SetDispatch<TSelf, prim::LTNode>(vtable);
       SetDispatch<TSelf, prim::LENode>(vtable);
       SetDispatch<TSelf, prim::GTNode>(vtable);
       SetDispatch<TSelf, prim::GENode>(vtable);
       SetDispatch<TSelf, prim::AndNode>(vtable);
       SetDispatch<TSelf, prim::OrNode>(vtable);
       SetDispatch<TSelf, prim::NotNode>(vtable);
       SetDispatch<TSelf, prim::SelectNode>(vtable);
       SetDispatch<TSelf, prim::LetNode>(vtable);
       SetDispatch<TSelf, prim::RampNode>(vtable);
       SetDispatch<TSelf, prim::BroadcastNode>(vtable);
       SetDispatch<TSelf, prim::ShuffleNode>(vtable);
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
     virtual ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TupleNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TensorRegionNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const VarNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const GlobalVarNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const CallNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const GenericConstNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const DataTypeImmNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const IntImmNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const FloatImmNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const OpNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const StringImmNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::CastNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AddNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LShiftNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::RShiftNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseAndNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseOrNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseXorNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseNotNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SubNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MulNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::DivNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ModNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorDivNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorModNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MinNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MaxNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::EQNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NENode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LTNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LENode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GTNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GENode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AndNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::OrNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NotNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SelectNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::RampNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastNode* node);
     virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleNode* node);
   
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
     // Extra node types use a fresh inherited table and SetDispatch<Self, ExtraNode>.
     // Hooks borrow the node and return Unchanged or an owning replacement in its expression
     // category, throwing on failure. Narrower field types are checked separately. Forward
     // inplace_mode on every child edge.
     virtual UnchangedOr<Expr> Mutate_(const OpaqueExprNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const TupleNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const TupleGetItemNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const TensorRegionNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const VarNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const GlobalVarNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const CallNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const GenericConstNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const DataTypeImmNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const IntImmNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const FloatImmNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const OpNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<Expr> Mutate_(const StringImmNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LShiftNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::RShiftNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::SubNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::MulNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::DivNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::ModNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::MinNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::NENode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LTNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LENode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::GTNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::GENode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::AndNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::OrNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::NotNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::RampNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* node, InplaceMode inplace_mode);
     virtual UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* node, InplaceMode inplace_mode);
   
    protected:
     static UnchangedOr<Expr> ReinferMutatedCallType(UnchangedOr<Expr> mutated,
                                                     const CallNode* original,
                                                     InplaceMode inplace_mode);
   
     explicit ExprMutator(const VTable* vtable) : ObjectMutator(vtable) {}
     static void InitVTable(VTable* vtable);
   };
   
   }  // namespace tvm
   #endif  // TVM_IR_EXPR_FUNCTOR_H_
