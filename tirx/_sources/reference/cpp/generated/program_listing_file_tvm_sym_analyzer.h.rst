
.. _program_listing_file_tvm_sym_analyzer.h:

Program Listing for File analyzer.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_sym_analyzer.h>` (``tvm/sym/analyzer.h``)

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
   
   #ifndef TVM_SYM_ANALYZER_H_
   #define TVM_SYM_ANALYZER_H_
   
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/with_context.h>
   #include <tvm/sym/constraint.h>
   #include <tvm/sym/int_set.h>
   
   #include <functional>
   #include <limits>
   #include <memory>
   #include <unordered_map>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   namespace sym {
   //-------------------------------------------------------
   // Base integer analysis API.
   //
   // We have multiple type of analyzers to do relaxed
   // integer set analysis(bound analysis, modulo) and
   // equivalence checking and simplification.
   //
   // Importantly, each analyzer may need result from
   // another analyzer.
   //-------------------------------------------------------
   
   // Forward declare the analyzer object and its reference handle.
   class AnalyzerObj;
   class Analyzer;
   namespace detail {
   class TempConstraintScope;
   }  // namespace detail
   
   enum DivMode {
     kTruncDiv,
     kFloorDiv
   };
   
   enum class ProofStrength : int {
     kDefault = 0,
     kSymbolicBound = 1,
   };
   
   class ConstIntBoundObj : public ffi::Object {
    public:
     int64_t min_value;
     int64_t max_value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ConstIntBoundObj>()
           .def_ro("min_value", &ConstIntBoundObj::min_value)
           .def_ro("max_value", &ConstIntBoundObj::max_value);
     }
   
     static const constexpr int64_t kPosInf = std::numeric_limits<int64_t>::max();
     static const constexpr int64_t kNegInf = -kPosInf;
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ConstIntBound", ConstIntBoundObj, ffi::Object);
   };
   
   class ConstIntBound : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL ConstIntBound(int64_t min_value, int64_t max_value);
   
     static const constexpr int64_t kPosInf = ConstIntBoundObj::kPosInf;
     static const constexpr int64_t kNegInf = ConstIntBoundObj::kNegInf;
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConstIntBound, ffi::ObjectRef, ConstIntBoundObj);
   };
   
   class ConstIntBoundAnalyzer {
    public:
     using BoundMapType =
         std::unordered_map<PrimExpr, ConstIntBound, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
     TVM_TIRX_DLL ConstIntBound operator()(const PrimExpr& expr) const;
   
     TVM_TIRX_DLL ConstIntBound operator()(const PrimExpr& expr, BoundMapType* bound);
   
     TVM_TIRX_DLL bool IsBound(const Var& var) const;
   
    private:
     friend class AnalyzerObj;
     // Clear context-derived facts before replay; retain query settings and diagnostics.
     void Reset();
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Record a half-open variable range from the accepted premise prefix.
     void IngestBinding(const Var& var, const Range& range);
     // Consume a predicate prepared against the preceding premise prefix.
     void IngestPredicate(const PrimExpr& constraint);
     // Borrow the owning analyzer for cross-backend queries.
     explicit ConstIntBoundAnalyzer(AnalyzerObj* parent);
     // Exported for destruction through the inline Analyzer object allocator.
     TVM_TIRX_DLL ~ConstIntBoundAnalyzer();
     // Copy durable facts/settings into this backend without sharing mutable query state.
     void CopyFrom(const ConstIntBoundAnalyzer& other);
     struct Entry;
     class Impl;
     // Owned backend state; its definition stays inside the compiler library.
     std::unique_ptr<Impl> impl_;
   };
   
   class ModularSetObj : public ffi::Object {
    public:
     int64_t coeff;
     int64_t base;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ModularSetObj>()
           .def_ro("coeff", &ModularSetObj::coeff)
           .def_ro("base", &ModularSetObj::base);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ModularSet", ModularSetObj, ffi::Object);
   };
   
   class ModularSet : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL ModularSet(int64_t coeff, int64_t base);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ModularSet, ffi::ObjectRef, ModularSetObj);
   };
   
   class ModularSetAnalyzer {
    public:
     TVM_TIRX_DLL ModularSet operator()(const PrimExpr& expr);
   
    private:
     friend class AnalyzerObj;
     // Clear context-derived facts before replay; retain query settings and diagnostics.
     void Reset();
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Consume a predicate prepared against the preceding premise prefix.
     void IngestPredicate(const PrimExpr& constraint);
     // Borrow the owning analyzer for cross-backend queries.
     explicit ModularSetAnalyzer(AnalyzerObj* parent);
     // Exported for destruction through the inline Analyzer object allocator.
     TVM_TIRX_DLL ~ModularSetAnalyzer();
     // Copy durable facts/settings into this backend without sharing mutable query state.
     void CopyFrom(const ModularSetAnalyzer& other);
     struct Entry;
     class Impl;
     // Owned backend state; its definition stays inside the compiler library.
     std::unique_ptr<Impl> impl_;
   };
   
   class RewriteSimplifier {
    public:
     TVM_TIRX_DLL PrimExpr operator()(const PrimExpr& expr);
   
     TVM_TIRX_DLL ffi::ObjectRef GetStatsCounters() const;
   
     TVM_TIRX_DLL void ResetStatsCounters();
   
     TVM_TIRX_DLL void SetMaximumRewriteSteps(int64_t maximum);
   
    private:
     friend class AnalyzerObj;
     // Clear context-derived facts before replay; retain query settings and diagnostics.
     void Reset();
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Consume a predicate prepared against the preceding premise prefix.
     void IngestPredicate(const PrimExpr& constraint);
     friend class CanonicalSimplifier;
     // Borrow the owning analyzer for cross-backend queries.
     explicit RewriteSimplifier(AnalyzerObj* parent);
     // Exported for destruction through the inline Analyzer object allocator.
     TVM_TIRX_DLL ~RewriteSimplifier();
     // Copy durable facts/settings into this backend without sharing mutable query state.
     void CopyFrom(const RewriteSimplifier& other);
     class Impl;
     // Owned backend state; its definition stays inside the compiler library.
     ffi::ObjectPtr<Impl> impl_;
   };
   
   class CanonicalSimplifier {
    public:
     TVM_TIRX_DLL PrimExpr operator()(const PrimExpr& expr);
   
    private:
     friend class AnalyzerObj;
     // Clear context-derived facts before replay; retain query settings and diagnostics.
     void Reset();
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Borrow the owning analyzer for cross-backend queries.
     explicit CanonicalSimplifier(AnalyzerObj* parent);
     // Exported for destruction through the inline Analyzer object allocator.
     TVM_TIRX_DLL ~CanonicalSimplifier();
     // Copy durable facts/settings into this backend without sharing mutable query state.
     void CopyFrom(const CanonicalSimplifier& other);
     class Impl;
     // Owned backend state; its definition stays inside the compiler library.
     ffi::ObjectPtr<Impl> impl_;
   };
   
   enum class CompareResult : int {
     kInconsistent = 0,
     kEQ = 1,
     kLT = 2,
     kLE = 3,
     kGT = 4,
     kGE = 5,
     kNE = 6,
     kUnknown = 7
   };
   
   inline constexpr CompareResult operator&(CompareResult lhs, CompareResult rhs) {
     return CompareResult(static_cast<int>(lhs) & static_cast<int>(rhs));
   }
   inline constexpr CompareResult operator|(CompareResult lhs, CompareResult rhs) {
     return CompareResult(static_cast<int>(lhs) | static_cast<int>(rhs));
   }
   
   class KnownComparisonAnalyzer {
    public:
     TVM_TIRX_DLL CompareResult TryCompare(const PrimExpr& lhs, const PrimExpr& rhs);
   
    private:
     friend class AnalyzerObj;
     // Clear context-derived facts before replay; retain query settings and diagnostics.
     void Reset();
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Record a half-open variable range from the accepted premise prefix.
     void IngestBinding(const Var& var, const Range& range);
     // Consume a predicate prepared against the preceding premise prefix.
     void IngestPredicate(const PrimExpr& constraint);
     // Borrow the owning analyzer for cross-backend queries.
     explicit KnownComparisonAnalyzer(AnalyzerObj* parent);
     // Exported for destruction through the inline Analyzer object allocator.
     TVM_TIRX_DLL ~KnownComparisonAnalyzer();
     // Copy durable facts/settings into this backend without sharing mutable query state.
     void CopyFrom(const KnownComparisonAnalyzer& other);
     class Impl;
     // Owned backend state; its definition stays inside the compiler library.
     std::unique_ptr<Impl> impl_;
   };
   
   class IntSetAnalyzer {
    public:
     TVM_TIRX_DLL IntSet operator()(const PrimExpr& expr, const ffi::Map<Var, IntSet>& dom_map);
   
     TVM_TIRX_DLL IntSet operator()(const PrimExpr& expr);
   
    private:
     friend class AnalyzerObj;
     // Clear context-derived facts before replay; retain query settings and diagnostics.
     void Reset();
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Record a half-open variable range from the accepted premise prefix.
     void IngestBinding(const Var& var, const Range& range);
     // Consume a predicate prepared against the preceding premise prefix.
     void IngestPredicate(const PrimExpr& constraint);
     // Borrow the owning analyzer for cross-backend queries.
     explicit IntSetAnalyzer(AnalyzerObj* parent);
     // Exported for destruction through the inline Analyzer object allocator.
     TVM_TIRX_DLL ~IntSetAnalyzer();
     // Copy durable facts/settings into this backend without sharing mutable query state.
     void CopyFrom(const IntSetAnalyzer& other);
     class Impl;
     // Owned backend state; its definition stays inside the compiler library.
     Impl* impl_;
   };
   
   class Z3Prover {
    public:
     TVM_TIRX_DLL bool IsEnabled() const;
   
     TVM_TIRX_DLL bool CanProve(const PrimExpr& expr);
   
     TVM_TIRX_DLL ffi::String GetSMTLIB2(const ffi::Optional<PrimExpr> expr);
   
     TVM_TIRX_DLL ffi::String GetStats();
   
     TVM_TIRX_DLL void SetTimeoutMs(unsigned timeout_ms);
   
     TVM_TIRX_DLL void SetRLimit(unsigned rlimit);
   
     TVM_TIRX_DLL ffi::String GetModel(const PrimExpr& expr);
   
     TVM_TIRX_DLL int64_t CountSatisfyingValues(const Var& var, int64_t max_count = 2048,
                                                int64_t min_consecutive = 1);
   
    private:
     friend class AnalyzerObj;
     // Clear facts, translations, and solver state; retain resource limits.
     void Reset();
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Record a half-open variable range from the accepted premise prefix.
     void IngestBinding(const Var& var, const Range& range);
     // Consume a predicate prepared against the preceding premise prefix.
     void IngestPredicate(const PrimExpr& constraint);
     friend class Analyzer;
     // Borrow the owning analyzer for cross-backend queries.
     explicit Z3Prover(AnalyzerObj* parent);
     // Exported for destruction through the inline Analyzer object allocator.
     TVM_TIRX_DLL ~Z3Prover();
     // Copy durable facts/settings into this backend without sharing mutable query state.
     void CopyFrom(const Z3Prover& other);
     class Impl;
     std::unique_ptr<Impl> impl_;
   };
   
   class TVM_TIRX_DLL AnalyzerObj : public ffi::Object {
    public:
     ConstIntBoundAnalyzer const_int_bound;
     ModularSetAnalyzer modular_set;
     RewriteSimplifier rewrite_simplify;
     CanonicalSimplifier canonical_simplify;
     IntSetAnalyzer int_set;
     KnownComparisonAnalyzer known_comparisons;
     Z3Prover z3_prover;
     AnalyzerObj();
     void SyncConstraintContext(const ConstraintContext& context);
     ConstraintContext GetConstraintContext() const;
     bool CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound);
     bool CanProveLess(const PrimExpr& expr, int64_t upper_bound);
     bool CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs);
     bool CanProveLessEqualThanSymbolicShapeValue(const PrimExpr& lhs, const PrimExpr& shape);
     bool CanProve(const PrimExpr& cond, ProofStrength strength = ProofStrength::kDefault);
   
     PrimExpr Simplify(const PrimExpr& expr, int steps = 2);
   
     Analyzer Clone() const;
   
     static constexpr bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.Analyzer", AnalyzerObj, ffi::Object);
   
    private:
     friend class detail::TempConstraintScope;
     // Record a value binding after AnalyzerObj validates and orders the premises.
     void IngestBinding(const Var& var, const PrimExpr& expr);
     // Record a half-open variable range from the accepted premise prefix.
     void IngestBinding(const Var& var, const Range& range);
     // Dispatch a supported pure premise to the participating backends.
     void IngestConstraint(const Constraint& constraint);
     // Normalize and derive predicate facts using only the already valid prefix.
     std::vector<PrimExpr> PreparePredicate(const PrimExpr& condition);
     // Fixed position of the last successfully accepted premise environment.
     ConstraintContext last_snapshot_;
     // Query-local premises, managed in LIFO order by TempConstraintScope.
     std::vector<Constraint> temp_constraints_;
     // Prevent reentry and reuse after an incomplete synchronization.
     bool synchronizing_{false};
     // Nested query count; synchronization requires zero active queries.
     int active_queries_{0};
   };
   
   class Analyzer : public ffi::ObjectRef {
    public:
     Analyzer() : Analyzer(ffi::make_object<AnalyzerObj>()) {}
     explicit Analyzer(ffi::ObjectPtr<AnalyzerObj> n) : ffi::ObjectRef(std::move(n)) {
       TVM_FFI_ICHECK(this->get() != nullptr);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Analyzer, ffi::ObjectRef, AnalyzerObj);
   };
   
   class AllowUIntAsIndex {
    public:
     TVM_TIRX_DLL static bool Enabled();
   
    private:
     friend class With<AllowUIntAsIndex>;
     // Called across the DLL boundary by the inline With<> constructor.
     TVM_TIRX_DLL void EnterWithScope();
     // Called across the DLL boundary by the inline With<> destructor.
     TVM_TIRX_DLL void ExitWithScope();
     // Number of active scopes in this thread.
     static thread_local int depth_;
   };
   
   }  // namespace sym
   }  // namespace tvm
   #endif  // TVM_SYM_ANALYZER_H_
