
.. _program_listing_file_tvm_tirx_stmt_functor.h:

Program Listing for File stmt_functor.h
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_stmt_functor.h>` (``tvm/tirx/stmt_functor.h``)

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
   
   #ifndef TVM_TIRX_STMT_FUNCTOR_H_
   #define TVM_TIRX_STMT_FUNCTOR_H_
   
   #include <tvm/ir/expr_functor.h>
   #include <tvm/ir/object_functor.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/tirx/function.h>
   #include <tvm/tirx/stmt.h>
   
   #include <unordered_map>
   #include <utility>
   
   namespace tvm {
   namespace tirx {
   template <typename FType>
   class StmtFunctor;
   
   template <typename R, typename... Args>
   class StmtFunctor<R(const Stmt&, Args...)> {
    private:
     using TSelf = StmtFunctor<R(const Stmt&, Args...)>;
   
    public:
     using result_type = R;
     StmtFunctor() : StmtFunctor(GlobalVTable()) {}
     virtual ~StmtFunctor() = default;
     TVM_FFI_INLINE R operator()(const Stmt& node, Args... args) {
       return Dispatch(node, std::forward<Args>(args)...);
     }
     TVM_FFI_INLINE virtual R Dispatch(const Stmt& node, Args... args) {
       TVM_FFI_ICHECK(node.defined()) << "Cannot dispatch a null statement";
       return (*vtable_)(node, this, std::forward<Args>(args)...);
     }
   
     virtual R Dispatch_(const BindNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const RegionStmtNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const IfNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const ForNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const WhileNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const ReturnNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const BreakNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const ContinueNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const TensorStoreNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const AssertStmtNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const SeqStmtNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R Dispatch_(const EvaluateNode* node, Args... args) {
       return DispatchDefault_(node, std::forward<Args>(args)...);
     }
     virtual R DispatchDefault_(const ffi::Object* node, Args...) {
       TVM_FFI_THROW(InternalError) << "Do not have a default for " << node->GetTypeKey();
       TVM_FFI_UNREACHABLE();
     }
   
    protected:
     using VTable = ObjectFunctor<R(const ffi::ObjectRef&, TSelf*, Args...)>;
     explicit StmtFunctor(const VTable* vtable) : vtable_(vtable) {}
     static void InitVTable(VTable* vtable) {
       vtable->template SetDispatch<StmtNode>(
           [](const ffi::ObjectRef& node, TSelf* self, Args... args) -> R {
             return self->DispatchDefault_(node.get(), std::forward<Args>(args)...);
           });
       SetDispatch<TSelf, BindNode>(vtable);
       SetDispatch<TSelf, RegionStmtNode>(vtable);
       SetDispatch<TSelf, IfNode>(vtable);
       SetDispatch<TSelf, ForNode>(vtable);
       SetDispatch<TSelf, WhileNode>(vtable);
       SetDispatch<TSelf, ReturnNode>(vtable);
       SetDispatch<TSelf, BreakNode>(vtable);
       SetDispatch<TSelf, ContinueNode>(vtable);
       SetDispatch<TSelf, TensorStoreNode>(vtable);
       SetDispatch<TSelf, AssertStmtNode>(vtable);
       SetDispatch<TSelf, SeqStmtNode>(vtable);
       SetDispatch<TSelf, EvaluateNode>(vtable);
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
   
   class TVM_TIRX_DLL StmtExprVisitor : public tvm::ExprVisitor {
    public:
     TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitor, tvm::ExprVisitor)
   
     using tvm::ExprVisitor::Visit;
     using tvm::ExprVisitor::Visit_;
   
     virtual ffi::Optional<VisitInterrupt> Visit_(const BindNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const IfNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const ForNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const WhileNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const ReturnNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const BreakNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const ContinueNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const TensorStoreNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const AssertStmtNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const SeqStmtNode* op);
     virtual ffi::Optional<VisitInterrupt> Visit_(const EvaluateNode* op);
   
     // Preserve TIRx operand traversal where it differs from the shared defaults.
     ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const TupleNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const prim::RampNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastNode* op) override;
     ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleNode* op) override;
   
    protected:
     explicit StmtExprVisitor(const VTable* vtable) : tvm::ExprVisitor(vtable) {}
     static void InitVTable(VTable* vtable);
   };
   
   class TVM_TIRX_DLL StmtExprMutator : public tvm::ExprMutator {
    public:
     TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprMutator, tvm::ExprMutator)
     using tvm::ExprMutator::Mutate;
     using tvm::ExprMutator::Mutate_;
   
     TVM_FFI_INLINE UnchangedOr<Stmt> Mutate(const Stmt& stmt,
                                             InplaceMode inplace_mode = InplaceMode::kDisallow) {
       return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<Stmt>(
           ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(stmt), inplace_mode)));
     }
   
     TVM_FFI_INLINE UnchangedOr<SeqStmt> Mutate(const SeqStmt& stmt,
                                                InplaceMode inplace_mode = InplaceMode::kDisallow) {
       auto result = Mutate(static_cast<const Stmt&>(stmt), inplace_mode);
       if (result.UnchangedOrSameAs(stmt)) return ffi::Unchanged();
       return SeqStmt(std::move(result).ValueUnchecked());
     }
   
     TVM_FFI_INLINE UnchangedOr<ffi::Optional<SeqStmt>> Mutate(
         const ffi::Optional<SeqStmt>& stmt, InplaceMode inplace_mode = InplaceMode::kDisallow) {
       if (!stmt.has_value()) return ffi::Unchanged();
       auto result = Mutate(stmt.value(), inplace_mode);
       if (result.IsUnchanged()) return ffi::Unchanged();
       return ffi::Optional<SeqStmt>(std::move(result).ValueUnchecked());
     }
   
     TVM_FFI_INLINE UnchangedOr<ffi::Optional<Stmt>> Mutate(
         const ffi::Optional<Stmt>& stmt, InplaceMode inplace_mode = InplaceMode::kDisallow) {
       if (!stmt.has_value()) return ffi::Unchanged();
       auto result = Mutate(stmt.value(), inplace_mode);
       if (result.IsUnchanged()) return ffi::Unchanged();
       return ffi::Optional<Stmt>(std::move(result).ValueUnchecked());
     }
   
     virtual UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const WhileNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const ReturnNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const BreakNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const ContinueNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const AssertStmtNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode);
     virtual UnchangedOr<Stmt> Mutate_(const EvaluateNode* op, InplaceMode inplace_mode);
   
    protected:
     explicit StmtExprMutator(const VTable* vtable) : tvm::ExprMutator(vtable) {}
     static void InitVTable(VTable* vtable);
   };
   
   TVM_TIRX_DLL Stmt SubstituteWithDataTypeLegalization(
       Stmt stmt, ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var&)> vmap);
   
   TVM_TIRX_DLL PrimExpr SubstituteWithDataTypeLegalization(
       PrimExpr expr, ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var&)> vmap);
   
   template <typename Node, typename = std::enable_if_t<std::is_base_of_v<StmtNode, Node>>>
   bool ContainsNode(const Stmt& stmt) {
     struct Visitor : StmtExprVisitor {
       // Early bail-out, if we already found the node. Skip expression operands.
       ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
         if (contains_node || value.as<ExprNode>()) {
           return std::nullopt;
         }
         if (value.as<Node>()) {
           contains_node = true;
           return std::nullopt;
         }
         return StmtExprVisitor::Visit(value);
       }
   
       bool contains_node{false};
     };
   
     auto visitor = ffi::make_object<Visitor>();
     visitor->Visit(stmt);
     return visitor->contains_node;
   }
   
   }  // namespace tirx
   }  // namespace tvm
   
   #endif  // TVM_TIRX_STMT_FUNCTOR_H_
