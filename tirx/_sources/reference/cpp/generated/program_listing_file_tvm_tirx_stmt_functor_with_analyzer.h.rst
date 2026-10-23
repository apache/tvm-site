
.. _program_listing_file_tvm_tirx_stmt_functor_with_analyzer.h:

Program Listing for File stmt_functor_with_analyzer.h
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_stmt_functor_with_analyzer.h>` (``tvm/tirx/stmt_functor_with_analyzer.h``)

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
   
   #ifndef TVM_TIRX_STMT_FUNCTOR_WITH_ANALYZER_H_
   #define TVM_TIRX_STMT_FUNCTOR_WITH_ANALYZER_H_
   
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/extra/structural_visit.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/ir/scope_stack.h>
   #include <tvm/ir/with_context.h>
   #include <tvm/sym/analyzer.h>
   #include <tvm/tirx/analysis.h>
   #include <tvm/tirx/stmt_functor.h>
   
   #include <unordered_set>
   #include <utility>
   
   namespace tvm {
   namespace tirx {
   
   class TVM_TIRX_DLL StmtExprVisitorWithAnalyzer : public StmtExprVisitor {
    public:
     TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitorWithAnalyzer, StmtExprVisitor)
   
     PrimExpr Simplify(const PrimExpr& expr) { return analyzer_->Simplify(expr); }
   
     using StmtExprVisitor::Visit_;
   
     ffi::Optional<VisitInterrupt> Visit_(const ForNode* op);
     ffi::Optional<VisitInterrupt> Visit_(const BindNode* op);
     ffi::Optional<VisitInterrupt> Visit_(const IfNode* op);
     ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op);
     ffi::Optional<VisitInterrupt> Visit_(const AssertStmtNode* op);
     ffi::Optional<VisitInterrupt> Visit_(const CallNode* op);
     ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* op);
   
     // StmtExprVisitorWithAnalyzer deliberately does not handle Select nodes,
     // because both sides of a Select node are visited regardless of the
     // condition.
   
    protected:
     static void InitVTable(VTable* vtable);
     explicit StmtExprVisitorWithAnalyzer(const VTable* vtable) : StmtExprVisitor(vtable) {}
     sym::Analyzer analyzer_;
   
     ScopeStack<WithGroup<sym::ConstraintContext>> constraint_scope_;
   
     PrimExpr ExtractRealCondition(PrimExpr condition) const;
   };
   
   class TVM_TIRX_DLL StmtExprMutatorWithAnalyzer : public StmtExprMutator {
    public:
     using StmtExprMutator::Mutate;
     using StmtExprMutator::Mutate_;
     explicit StmtExprMutatorWithAnalyzer(const sym::Analyzer& analyzer)
         : StmtExprMutatorWithAnalyzer(analyzer.get()) {}
     explicit StmtExprMutatorWithAnalyzer(sym::AnalyzerObj* analyzer)
         : StmtExprMutatorWithAnalyzer(analyzer, GlobalVTable()) {}
   
     // override functions that need to populate the context information.
     UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) override;
     UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override;
     UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) override;
     UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) override;
     UnchangedOr<Stmt> Mutate_(const AssertStmtNode* op, InplaceMode inplace_mode) override;
     UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
     UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) override;
     UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;
   
    protected:
     static void InitVTable(VTable* vtable);
     StmtExprMutatorWithAnalyzer(sym::AnalyzerObj* analyzer, const VTable* vtable)
         : StmtExprMutator(vtable), analyzer_(analyzer) {}
     static const VTable* GlobalVTable();
     void MarkTensorParamShapes(const Function& func);
   
     ffi::Array<PrimExpr> IterMapSimplifyWithContext(const ffi::Array<PrimExpr>& indices,
                                                     bool non_trivial_only);
   
     sym::AnalyzerObj* analyzer_;
     ScopeStack<WithGroup<sym::ConstraintContext>> constraint_scope_;
     // the following two fields are useful in case we want
     // note however that iter map analysis are usually more
     // expensive and we only encourage doing them during
     // necessary cases like layout remapping
     ffi::Map<PrimVar, Range> iter_vars_;
     ffi::Array<PrimExpr> iter_predicates_;
     template <typename FLambda>
     void WithRecordIterPredicate(PrimExpr condition, FLambda callback) {
       std::unordered_set<const VarNode*> iter_var_nodes;
       for (const auto& [var, _] : iter_vars_) {
         iter_var_nodes.insert(var.get());
       }
       auto f_use_itervar = [&iter_var_nodes](const tvm::VarNode* v) {
         return iter_var_nodes.count(v);
       };
       auto walkfn = [&](const tvm::Var& var) -> ffi::Expected<ffi::WalkResult> {
         return f_use_itervar(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                         : ffi::WalkResult::Advance();
       };
       // simple heuristics for detecting predicate
       if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(condition, walkfn).has_value()) {
         iter_predicates_.push_back(condition);
         callback();
         iter_predicates_.pop_back();
       } else {
         callback();
       }
     }
   };
   
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_STMT_FUNCTOR_WITH_ANALYZER_H_
