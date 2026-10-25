
.. _program_listing_file_tvm_tirx_stmt_functor_with_constraint_context.h:

Program Listing for File stmt_functor_with_constraint_context.h
===============================================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_stmt_functor_with_constraint_context.h>` (``tvm/tirx/stmt_functor_with_constraint_context.h``)

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
   
   #ifndef TVM_TIRX_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
   #define TVM_TIRX_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
   
   #include <tvm/sym/analyzer.h>
   #include <tvm/tirx/function.h>
   #include <tvm/tirx/object_functor_with_constraint_context.h>
   #include <tvm/tirx/stmt.h>
   
   namespace tvm {
   namespace tirx {
   class TVM_TIRX_DLL StmtExprVisitorWithConstraintContext
       : public ObjectVisitorWithConstraintContext {
    public:
     StmtExprVisitorWithConstraintContext() : ObjectVisitorWithConstraintContext(GlobalVTable()) {}
   
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const LambdaExprObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const OpaqueExprObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TupleObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TupleGetItemObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorRegionObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const VarObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const GlobalVarObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const GenericConstObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const DataTypeImmObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const IntImmObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const FloatImmObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const OpObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const StringImmObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::CastObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::AddObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LShiftObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::RShiftObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseAndObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseOrObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseXorObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseNotObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::SubObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::MulObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::DivObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::ModObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::FloorDivObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::FloorModObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::MinObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::MaxObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::EQObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::NEObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LTObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LEObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::GTObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::GEObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::AndObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::OrObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::NotObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::SelectObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LetObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::RampObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BroadcastObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::ShuffleObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const BindObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const IfObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const ForObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const WhileObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const ReturnObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const BreakObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const ContinueObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorStoreObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const AssertStmtObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const SeqStmtObj* node);
     virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const EvaluateObj* node);
   
    protected:
     explicit StmtExprVisitorWithConstraintContext(const VTable* table)
         : ObjectVisitorWithConstraintContext(table) {}
   
     static void InitVTable(VTable* table);
   
    private:
     static const VTable* GlobalVTable();
   };
   
   class TVM_TIRX_DLL StmtExprMutatorWithConstraintContext
       : public ObjectMutatorWithConstraintContext {
    public:
     StmtExprMutatorWithConstraintContext() : ObjectMutatorWithConstraintContext(GlobalVTable()) {}
     explicit StmtExprMutatorWithConstraintContext(const sym::ConstraintContext& context)
         : StmtExprMutatorWithConstraintContext() {
       constraints = context->Snapshot();
     }
     using ObjectMutatorWithConstraintContext::Mutate;
     ffi::UnchangedOr<Expr> Mutate(const Expr& value,
                                   ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
       return Mutate(ffi::AnyView(value), mode).as_or_throw<ffi::UnchangedOr<Expr>>();
     }
     ffi::UnchangedOr<PrimExpr> Mutate(const PrimExpr& value,
                                       ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
       return Mutate(ffi::AnyView(value), mode).as_or_throw<ffi::UnchangedOr<PrimExpr>>();
     }
     ffi::UnchangedOr<Stmt> Mutate(const Stmt& value,
                                   ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
       return Mutate(ffi::AnyView(value), mode).as_or_throw<ffi::UnchangedOr<Stmt>>();
     }
     ffi::UnchangedOr<SeqStmt> Mutate(const SeqStmt& value,
                                      ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
       auto result = Mutate(ffi::AnyView(value), mode);
       if (result.IsUnchanged()) return ffi::Unchanged();
       return SeqStmt(std::move(result).ValueUnchecked().cast<Stmt>());
     }
     virtual ffi::UnchangedOr<Expr> Mutate_(const LambdaExprObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const OpaqueExprObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const TupleObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const TupleGetItemObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const TensorRegionObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const VarObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const GlobalVarObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const CallObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const GenericConstObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const DataTypeImmObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const IntImmObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const FloatImmObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const OpObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Expr> Mutate_(const StringImmObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::CastObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LShiftObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::RShiftObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndObj* node,
                                                ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorObj* node,
                                                ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotObj* node,
                                                ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::SubObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::MulObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::DivObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::ModObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::FloorModObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::MinObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::MaxObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::AndObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::OrObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::NotObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LetObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::RampObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const BindObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const IfObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const ForObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const WhileObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const ReturnObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const BreakObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const ContinueObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const AssertStmtObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj* node, ffi::InplaceMode mode);
     virtual ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj* node, ffi::InplaceMode mode);
   
    protected:
     sym::AnalyzerObj* GetSyncedAnalyzer() {
       analyzer_->SyncConstraintContext(constraints);
       return analyzer_.get();
     }
   
     void AddParamConstraints(const Function& func);
   
     ffi::Array<PrimExpr> IterMapSimplifyWithContext(const ffi::Array<PrimExpr>& indices,
                                                     bool non_trivial_only);
   
     explicit StmtExprMutatorWithConstraintContext(const VTable* table)
         : ObjectMutatorWithConstraintContext(table) {}
   
     static void InitVTable(VTable* table);
   
    private:
     static const VTable* GlobalVTable();
     sym::Analyzer analyzer_;
   };
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
