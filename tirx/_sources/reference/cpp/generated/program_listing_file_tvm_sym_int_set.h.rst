
.. _program_listing_file_tvm_sym_int_set.h:

Program Listing for File int_set.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_sym_int_set.h>` (``tvm/sym/int_set.h``)

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
   
   #ifndef TVM_SYM_INT_SET_H_
   #define TVM_SYM_INT_SET_H_
   
   #include <tvm/ir/expr.h>
   #include <tvm/ir/prim/expr.h>
   
   #include <unordered_map>
   
   namespace tvm {
   namespace sym {
   
   class AnalyzerObj;
   class Analyzer;
   
   //-----------------------------------------------
   // Integer set data structure.
   //
   // This is a API build on top of the base
   // integer analysis API to provide set analysis.
   //------------------------------------------------
   enum SignType { kPositive, kNegative, kZero, kUnknown };
   
   class IntSetObj : public ffi::Object {
    public:
     TVM_FFI_DECLARE_OBJECT_INFO("sym.IntSet", IntSetObj, ffi::Object);
   };
   
   class IntSet : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL ffi::Optional<Range> CoverRange(ffi::Optional<Range> max_range) const;
     TVM_TIRX_DLL PrimExpr min() const;
     TVM_TIRX_DLL PrimExpr max() const;
     TVM_TIRX_DLL SignType GetSignType() const;
     TVM_TIRX_DLL bool IsNothing() const;
     TVM_TIRX_DLL bool IsEverything() const;
     TVM_TIRX_DLL bool IsSinglePoint() const;
     TVM_TIRX_DLL bool CanProveSinglePoint(const Analyzer& ana) const;
     // TODO(tvm-team): update all CanProve to explicitly take
     // analyzer to encourage more analyzer reuse
     TVM_TIRX_DLL bool CanProvePositive() const;
     TVM_TIRX_DLL bool CanProveNegative() const;
     TVM_TIRX_DLL bool CanProveNonPositive() const;
     TVM_TIRX_DLL bool CanProveNonNegative() const;
     TVM_TIRX_DLL bool HasUpperBound() const;
     TVM_TIRX_DLL bool HasLowerBound() const;
   
     TVM_TIRX_DLL PrimExpr PointValue() const;
     TVM_TIRX_DLL bool MatchRange(const tvm::Range& r) const;
     TVM_TIRX_DLL static IntSet Nothing();
     TVM_TIRX_DLL static IntSet Everything();
     TVM_TIRX_DLL static IntSet SinglePoint(PrimExpr point);
     TVM_TIRX_DLL static IntSet Vector(PrimExpr vec);
     TVM_TIRX_DLL static IntSet FromMinExtent(PrimExpr min, PrimExpr extent);
     TVM_TIRX_DLL static IntSet FromRange(tvm::Range r);
     TVM_TIRX_DLL static IntSet Interval(PrimExpr min, PrimExpr max);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntSet, ffi::ObjectRef, IntSetObj);
   };
   
   TVM_TIRX_DLL IntSet EvalSet(PrimExpr e, const ffi::Map<Var, IntSet>& dom_map);
   TVM_TIRX_DLL IntSet EvalSet(IntSet s, const ffi::Map<Var, IntSet>& dom_map);
   TVM_TIRX_DLL IntSet EvalSet(Range r, const ffi::Map<Var, IntSet>& dom_map);
   TVM_TIRX_DLL ffi::Array<IntSet> EvalSet(const ffi::Array<Range>& region,
                                           const ffi::Map<Var, IntSet>& dom_map);
   using ExprIntSetMap = std::unordered_map<PrimExpr, IntSet, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
   TVM_TIRX_DLL ExprIntSetMap EvalSetForEachSubExpr(PrimExpr e, const ffi::Map<Var, IntSet>& dom_map);
   
   TVM_TIRX_DLL IntSet Union(const ffi::Array<IntSet>& sets);
   
   TVM_TIRX_DLL ffi::Array<IntSet> UnionRegion(const ffi::Array<ffi::Array<IntSet>>& nd_int_sets);
   
   TVM_TIRX_DLL IntSet UnionLowerBound(const ffi::Array<IntSet>& sets);
   
   TVM_TIRX_DLL ffi::Array<IntSet> UnionRegionLowerBound(
       const ffi::Array<ffi::Array<IntSet>>& nd_int_sets);
   
   TVM_TIRX_DLL IntSet Intersect(const ffi::Array<IntSet>& sets);
   
   TVM_TIRX_DLL IntSet Intersect(const IntSet& lhs, const IntSet& rhs, const Analyzer& analyzer);
   
   TVM_TIRX_DLL ffi::Map<Var, sym::IntSet> AsIntSet(const ffi::Map<Var, Range>& var_dom);
   
   TVM_TIRX_DLL ffi::Optional<ffi::Array<IntSet>> EstimateRegionStrictBound(
       const ffi::Array<Range>& region, const ffi::Map<Var, Range>& var_dom, const PrimExpr& predicate,
       const sym::Analyzer& analyzer);
   
   TVM_TIRX_DLL ffi::Optional<ffi::Array<IntSet>> EstimateRegionLowerBound(
       const ffi::Array<Range>& region, const ffi::Map<Var, Range>& var_dom, const PrimExpr& predicate,
       const sym::Analyzer& analyzer);
   
   TVM_TIRX_DLL ffi::Array<IntSet> EstimateRegionUpperBound(const ffi::Array<Range>& region,
                                                            const ffi::Map<Var, Range>& var_dom,
                                                            const PrimExpr& predicate,
                                                            const sym::Analyzer& analyzer);
   
   TVM_TIRX_DLL IntSet DeduceBound(PrimExpr v, PrimExpr cond, const ffi::Map<Var, IntSet>& hint_map,
                                   const ffi::Map<Var, IntSet>& relax_map);
   
   }  // namespace sym
   }  // namespace tvm
   #endif  // TVM_SYM_INT_SET_H_
