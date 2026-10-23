
.. _program_listing_file_tvm_tirx_index_map.h:

Program Listing for File index_map.h
====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_index_map.h>` (``tvm/tirx/index_map.h``)

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
   
   #ifndef TVM_TIRX_INDEX_MAP_H_
   #define TVM_TIRX_INDEX_MAP_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ir/expr.h>
   #include <tvm/runtime/tensor.h>
   
   #include <utility>
   
   namespace tvm {
   namespace sym {
   class Analyzer;
   }  // namespace sym
   }  // namespace tvm
   
   namespace tvm {
   namespace tirx {
   
   class IndexMapObj : public ffi::Object {
    public:
     ffi::Array<PrimVar> initial_indices;
   
     ffi::Array<PrimExpr> final_indices;
   
     ffi::Optional<ffi::ObjectRef> inverse_index_map;
   
     IndexMapObj() {}
   
     TVM_TIRX_DLL ffi::Array<PrimExpr> MapIndices(const ffi::Array<PrimExpr>& indices) const;
     TVM_TIRX_DLL ffi::Array<PrimExpr> MapIndices(const ffi::Array<PrimExpr>& indices,
                                                  const sym::Analyzer& analyzer) const;
   
     TVM_TIRX_DLL ffi::Array<Range> MapRanges(const ffi::Array<Range>& ranges) const;
     TVM_TIRX_DLL ffi::Array<Range> MapRanges(const ffi::Array<Range>& ranges,
                                              const sym::Analyzer& analyzer) const;
   
     TVM_TIRX_DLL ffi::Array<PrimExpr> MapShape(const ffi::Array<PrimExpr>& shape) const;
     TVM_TIRX_DLL ffi::Array<PrimExpr> MapShape(const ffi::Array<PrimExpr>& shape,
                                                const sym::Analyzer& analyzer) const;
   
     /* \brief Map an Tensor according to this index map
      *
      * \param arr_src The Tensor whose layout is transformed by this index map.
      *
      * \returns The transformed Tensor.
      */
     TVM_TIRX_DLL ffi::Tensor MapTensor(ffi::Tensor arr_src) const;
   
     TVM_TIRX_DLL ffi::String ToPythonString(
         const ffi::TypedFunction<ffi::Optional<ffi::String>(const Var& var)>& f_name_map =
             nullptr) const;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IndexMapObj>()
           .def_ro("initial_indices", &IndexMapObj::initial_indices,
                   refl::AttachFieldFlag::SEqHashDefPattern())
           .def_ro("final_indices", &IndexMapObj::final_indices)
           .def_ro("inverse_index_map", &IndexMapObj::inverse_index_map,
                   refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.IndexMap", IndexMapObj, ffi::Object);
   };
   
   class IndexMap : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL IndexMap(ffi::Array<PrimVar> initial_indices, ffi::Array<PrimExpr> final_indices,
                           ffi::Optional<IndexMap> inverse_index_map = std::nullopt);
   
     TVM_TIRX_DLL static IndexMap FromFunc(
         int ndim, ffi::TypedFunction<ffi::Array<PrimExpr>(ffi::Array<Var>)> func,
         ffi::Optional<IndexMap> inverse_index_map = std::nullopt);
   
     TVM_TIRX_DLL IndexMap Inverse(ffi::Array<Range> initial_ranges) const;
     TVM_TIRX_DLL IndexMap Inverse(ffi::Array<Range> initial_ranges,
                                   const sym::Analyzer& analyzer) const;
   
     TVM_TIRX_DLL IndexMap
     RenameVariables(const ffi::TypedFunction<ffi::Optional<ffi::String>(const Var& var)>& f_name_map =
                         nullptr) const;
   
     TVM_TIRX_DLL ffi::Tuple<IndexMap, PrimExpr> NonSurjectiveInverse(
         ffi::Array<Range> initial_ranges) const;
     TVM_TIRX_DLL ffi::Tuple<IndexMap, PrimExpr> NonSurjectiveInverse(
         ffi::Array<Range> initial_ranges, const sym::Analyzer& analyzer) const;
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IndexMap, ffi::ObjectRef, IndexMapObj);
   };
   
   }  // namespace tirx
   }  // namespace tvm
   
   #endif  // TVM_TIRX_INDEX_MAP_H_
