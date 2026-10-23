
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
   
   class IterMapExprNode : public ExprNode {
    public:
     static constexpr const uint32_t _type_child_slots = 2;
     TVM_FFI_DECLARE_OBJECT_INFO("sym.IterMapExpr", IterMapExprNode, ExprNode);
   };
   
   class IterMapExpr : public PrimExpr {
    public:
     explicit IterMapExpr(ffi::ObjectPtr<IterMapExprNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterMapExpr, PrimExpr, IterMapExprNode);
     static constexpr bool _type_container_is_exact = true;
   };
   
   class IterMarkNode : public ffi::Object {
    public:
     explicit IterMarkNode(PrimExpr source, PrimExpr extent)
         : source(std::move(source)), extent(std::move(extent)) {}
     explicit IterMarkNode(ffi::UnsafeInit) : source(ffi::UnsafeInit{}), extent(ffi::UnsafeInit{}) {}
   
     PrimExpr source;
     PrimExpr extent;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IterMarkNode>()
           .def_ro("source", &IterMarkNode::source)
           .def_ro("extent", &IterMarkNode::extent);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.IterMark", IterMarkNode, ffi::Object);
   };
   
   class IterMark : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL IterMark(PrimExpr source, PrimExpr extent);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterMark, ffi::ObjectRef, IterMarkNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IterMarkNode);
   };
   
   class IterSplitExprNode : public IterMapExprNode {
    public:
     explicit IterSplitExprNode(IterMark source, PrimExpr lower_factor, PrimExpr extent,
                                PrimExpr scale)
         : source(std::move(source)),
           lower_factor(std::move(lower_factor)),
           extent(std::move(extent)),
           scale(std::move(scale)) {}
     explicit IterSplitExprNode(ffi::UnsafeInit)
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
       refl::ObjectDef<IterSplitExprNode>()
           .def_ro("source", &IterSplitExprNode::source)
           .def_ro("lower_factor", &IterSplitExprNode::lower_factor)
           .def_ro("extent", &IterSplitExprNode::extent)
           .def_ro("scale", &IterSplitExprNode::scale);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.IterSplitExpr", IterSplitExprNode, IterMapExprNode);
   };
   
   class IterSplitExpr : public IterMapExpr {
    public:
     TVM_TIRX_DLL explicit IterSplitExpr(IterMark source);
     TVM_TIRX_DLL explicit IterSplitExpr(IterMark source, PrimExpr scale);
     TVM_TIRX_DLL explicit IterSplitExpr(IterMark source, PrimExpr lower_factor, PrimExpr extent,
                                         PrimExpr scale);
   
     explicit IterSplitExpr(ffi::ObjectPtr<IterSplitExprNode> node) : IterMapExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterSplitExpr, IterMapExpr, IterSplitExprNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSplitExprNode);
   };
   
   class IterSumExprNode : public IterMapExprNode {
    public:
     explicit IterSumExprNode(PrimExpr base) : base(std::move(base)) {}
     explicit IterSumExprNode(ffi::UnsafeInit) : base(ffi::UnsafeInit{}) {}
   
     ffi::Array<IterSplitExpr> args;
     PrimExpr base;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IterSumExprNode>()
           .def_ro("args", &IterSumExprNode::args)
           .def_ro("base", &IterSumExprNode::base);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.IterSumExpr", IterSumExprNode, IterMapExprNode);
   };
   
   class IterSumExpr : public IterMapExpr {
    public:
     TVM_TIRX_DLL IterSumExpr(ffi::Array<IterSplitExpr> args, PrimExpr base);
   
     explicit IterSumExpr(ffi::ObjectPtr<IterSumExprNode> node) : IterMapExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IterSumExpr, IterMapExpr, IterSumExprNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSumExprNode);
   };
   
   }  // namespace sym
   
   namespace ffi {
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, sym::IterSplitExprNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, sym::IterSumExprNode> = true;
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
