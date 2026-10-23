
.. _program_listing_file_tvm_sym_iter_affine_map.h:

Program Listing for File iter_affine_map.h
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_sym_iter_affine_map.h>` (``tvm/sym/iter_affine_map.h``)

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
   
   #ifndef TVM_SYM_ITER_AFFINE_MAP_H_
   #define TVM_SYM_ITER_AFFINE_MAP_H_
   
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/cow.h>
   #include <tvm/ir/expr.h>
   #include <tvm/sym/analyzer.h>
   
   namespace tvm {
   namespace sym {
   
   class IterMapExprObj : public ExprObj {
    public:
     static constexpr const uint32_t _type_child_slots = 2;
     TVM_FFI_DECLARE_OBJECT_INFO("sym.IterMapExpr", IterMapExprObj, ExprObj);
   };
   
   class IterMapExpr : public PrimExpr {
    public:
     explicit IterMapExpr(ffi::ObjectPtr<IterMapExprObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterMapExpr, PrimExpr, IterMapExprObj);
     static constexpr bool _type_container_is_exact = true;
   };
   
   class IterMarkObj : public ffi::Object {
    public:
     explicit IterMarkObj(PrimExpr source, PrimExpr extent)
         : source(std::move(source)), extent(std::move(extent)) {}
     explicit IterMarkObj(ffi::UnsafeInit) : source(ffi::UnsafeInit{}), extent(ffi::UnsafeInit{}) {}
   
     PrimExpr source;
     PrimExpr extent;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IterMarkObj>()
           .def_ro("source", &IterMarkObj::source)
           .def_ro("extent", &IterMarkObj::extent);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.IterMark", IterMarkObj, ffi::Object);
   };
   
   class IterMark : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL IterMark(PrimExpr source, PrimExpr extent);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterMark, ffi::ObjectRef, IterMarkObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IterMarkObj);
   };
   
   class IterSplitExprObj : public IterMapExprObj {
    public:
     explicit IterSplitExprObj(IterMark source, PrimExpr lower_factor, PrimExpr extent, PrimExpr scale)
         : source(std::move(source)),
           lower_factor(std::move(lower_factor)),
           extent(std::move(extent)),
           scale(std::move(scale)) {}
     explicit IterSplitExprObj(ffi::UnsafeInit)
         : source(ffi::UnsafeInit{}),
           lower_factor(ffi::UnsafeInit{}),
           extent(ffi::UnsafeInit{}),
           scale(ffi::UnsafeInit{}) {}
   
     IterMark source;
     PrimExpr lower_factor;
     PrimExpr extent;
     PrimExpr scale;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IterSplitExprObj>()
           .def_ro("source", &IterSplitExprObj::source)
           .def_ro("lower_factor", &IterSplitExprObj::lower_factor)
           .def_ro("extent", &IterSplitExprObj::extent)
           .def_ro("scale", &IterSplitExprObj::scale);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.IterSplitExpr", IterSplitExprObj, IterMapExprObj);
   };
   
   class IterSplitExpr : public IterMapExpr {
    public:
     TVM_TIRX_DLL explicit IterSplitExpr(IterMark source);
     TVM_TIRX_DLL explicit IterSplitExpr(IterMark source, PrimExpr scale);
     TVM_TIRX_DLL explicit IterSplitExpr(IterMark source, PrimExpr lower_factor, PrimExpr extent,
                                         PrimExpr scale);
   
     explicit IterSplitExpr(ffi::ObjectPtr<IterSplitExprObj> node) : IterMapExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterSplitExpr, IterMapExpr, IterSplitExprObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSplitExprObj);
   };
   
   class IterSumExprObj : public IterMapExprObj {
    public:
     explicit IterSumExprObj(PrimExpr base) : base(std::move(base)) {}
     explicit IterSumExprObj(ffi::UnsafeInit) : base(ffi::UnsafeInit{}) {}
   
     ffi::Array<IterSplitExpr> args;
     PrimExpr base;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IterSumExprObj>()
           .def_ro("args", &IterSumExprObj::args)
           .def_ro("base", &IterSumExprObj::base);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.IterSumExpr", IterSumExprObj, IterMapExprObj);
   };
   
   class IterSumExpr : public IterMapExpr {
    public:
     TVM_TIRX_DLL IterSumExpr(ffi::Array<IterSplitExpr> args, PrimExpr base);
   
     explicit IterSumExpr(ffi::ObjectPtr<IterSumExprObj> node) : IterMapExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterSumExpr, IterMapExpr, IterSumExprObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSumExprObj);
   };
   
   }  // namespace sym
   
   namespace ffi {
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, sym::IterSplitExprObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, sym::IterSumExprObj> = true;
   }  // namespace ffi
   
   namespace sym {
   
   enum IterMapLevel {
     // Require the mapping to be bijective.
     Bijective = 0,
     // Require the mapping to be surjective.
     Surjective = 1,
     // No mapping safety check.
     NoCheck = 3
   };
   
   class IterMapResultObj : public ffi::Object {
    public:
     // The detected pattern if a match exists.
     ffi::Array<IterSumExpr> indices;
   
     // Any errors that occurred while converting the input indices.  If
     // the array is empty, the conversion was successful.
     ffi::Array<ffi::String> errors;
   
     ffi::Optional<PrimExpr> padding_predicate;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IterMapResultObj>()
           .def_ro("indices", &IterMapResultObj::indices)
           .def_ro("errors", &IterMapResultObj::errors)
           .def_ro("padding_predicate", &IterMapResultObj::padding_predicate);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.IterMapResult", IterMapResultObj, ffi::Object);
   };
   
   class IterMapResult : public ffi::ObjectRef {
    public:
     static constexpr bool _type_is_nullable = false;
     // constructor
     IterMapResult() { data_ = ffi::make_object<IterMapResultObj>(); }
   
     IterMapResultObj* operator->() const { return static_cast<IterMapResultObj*>(get_mutable()); }
   };
   
   TVM_TIRX_DLL IterMapResult DetectIterMap(const ffi::Array<PrimExpr>& indices,
                                            const ffi::Map<PrimVar, Range>& input_iters,
                                            const PrimExpr& predicate, IterMapLevel check_level,
                                            const sym::Analyzer& analyzer,
                                            bool simplify_trivial_iterators = true);
   
   TVM_TIRX_DLL ffi::Array<PrimExpr> IterMapSimplify(const ffi::Array<PrimExpr>& indices,
                                                     const ffi::Map<PrimVar, Range>& input_iters,
                                                     const PrimExpr& input_pred,
                                                     IterMapLevel check_level,
                                                     const sym::Analyzer& analyzer,
                                                     bool simplify_trivial_iterators = true);
   
   TVM_TIRX_DLL ffi::Map<Var, PrimExpr> InverseAffineIterMap(const ffi::Array<IterSumExpr>& iter_map,
                                                             const ffi::Array<PrimExpr> outputs);
   
   TVM_TIRX_DLL ffi::Array<ffi::Array<IterMark>> SubspaceDivide(
       const ffi::Array<PrimExpr>& bindings, const ffi::Map<PrimVar, Range>& input_iters,
       const ffi::Array<PrimVar>& sub_iters, const PrimExpr& predicate, IterMapLevel check_level,
       const sym::Analyzer& analyzer, bool simplify_trivial_iterators = true);
   
   TVM_TIRX_DLL PrimExpr NormalizeIterMapToExpr(const PrimExpr& expr);
   
   TVM_TIRX_DLL IterSumExpr NormalizeToIterSum(PrimExpr index,
                                               const ffi::Map<PrimVar, Range>& input_iters,
                                               const sym::Analyzer& analyzer);
   
   }  // namespace sym
   }  // namespace tvm
   #endif  // TVM_SYM_ITER_AFFINE_MAP_H_
