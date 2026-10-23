
.. _program_listing_file_tvm_tirx_type.h:

Program Listing for File type.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_type.h>` (``tvm/tirx/type.h``)

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
   
   #ifndef TVM_TIRX_TYPE_H_
   #define TVM_TIRX_TYPE_H_
   
   #include <tvm/ir/expr.h>
   #include <tvm/ir/prim/op.h>
   #include <tvm/ir/type.h>
   #include <tvm/runtime/device_api.h>
   #include <tvm/sym/analyzer.h>
   #include <tvm/tirx/layout.h>
   
   #include <optional>
   
   namespace tvm::tirx {
   
   #ifndef TVM_INDEX_DEFAULT_I64
   #define TVM_INDEX_DEFAULT_I64 1
   #endif
   inline PrimType DefaultIndexType() {
   #if TVM_INDEX_DEFAULT_I64
     static const PrimType default_index_ty = PrimType::Int(64);
   #else
     static const PrimType default_index_ty = PrimType::Int(32);
   #endif
     return default_index_ty;
   }
   
   class TensorTypeObj : public TypeObj {
    public:
     explicit TensorTypeObj(ffi::UnsafeInit) {}
     TensorTypeObj() = default;
   
     PrimType dtype = PrimType::Void();
     runtime::MemoryScope scope;
     ffi::Array<PrimExpr> shape;
     int data_alignment;
     ffi::Optional<Layout> layout;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorTypeObj>()
           .def_ro("shape", &TensorTypeObj::shape, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("dtype", &TensorTypeObj::dtype)
           .def_ro("scope", &TensorTypeObj::scope, refl::DefaultValue(runtime::MemoryScope::Global()))
           .def_ro("data_alignment", &TensorTypeObj::data_alignment,
                   refl::DefaultValue(static_cast<int>(runtime::kAllocAlignment)))
           .def_ro("layout", &TensorTypeObj::layout, refl::DefaultValue(nullptr));
     }
   
     TVM_TIRX_DLL bool HasDefaultLayout(ffi::Optional<sym::Analyzer> analyzer = std::nullopt) const;
   
     PrimType DefaultIndexType() const {
       return !shape.empty() ? shape[0].ty() : tvm::tirx::DefaultIndexType();
     }
   
     PtrType DataPtrType() const { return PtrType(dtype, scope); }
   
     TVM_TIRX_DLL bool IsScalar(bool alloc_or_decl = true) const;
   
     TVM_TIRX_DLL ffi::Optional<int64_t> ConstantAllocationSize() const;
   
     TVM_TIRX_DLL ffi::Array<PrimExpr> ElemOffset(ffi::Array<PrimExpr> index) const;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TensorType", TensorTypeObj, TypeObj);
   };
   
   class TensorType : public Type {
    public:
     TVM_TIRX_DLL TensorType(ffi::Array<PrimExpr> shape, PrimType dtype = PrimType::Float(32),
                             runtime::MemoryScope scope = {}, int data_alignment = 0,
                             ffi::Optional<Layout> layout = std::nullopt, Location loc = UnknownLoc());
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorType, Type, TensorTypeObj);
   
     explicit TensorType(ffi::ObjectPtr<TensorTypeObj> n) : Type(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(n != nullptr);
       data_ = std::move(n);
     }
   };
   
   inline ffi::Optional<ffi::Tuple<ffi::Array<PrimExpr>, ffi::Array<PrimExpr>, PrimExpr>>
   MatchShapeStridePattern(const TensorType& tensor) {
     ffi::Array<PrimExpr> strides;
     PrimExpr offset = IntImm(tensor->DefaultIndexType(), 0);
     if (!tensor->layout.has_value()) {
       PrimExpr stride = IntImm(tensor->DefaultIndexType(), 1);
       strides = ffi::Array<PrimExpr>(tensor->shape.size(), stride);
       for (size_t i = tensor->shape.size(); i > 0; --i) {
         strides.Set(i - 1, stride);
         stride = stride * tensor->shape[i - 1];
       }
     } else {
       const auto* tile = tensor->layout.as<TileLayoutObj>();
       if (!tile || !tile->replica.empty() || tile->shard.size() != tensor->shape.size()) {
         return std::nullopt;
       }
       prim::ExprDeepEqual equal;
       static const Axis memory_axis = Axis::Get("m");
       for (size_t i = 0; i < tile->shard.size(); ++i) {
         if (tile->shard[i]->axis != memory_axis || !equal(tile->shard[i]->extent, tensor->shape[i]))
           return std::nullopt;
         strides.push_back(tile->shard[i]->stride);
       }
       for (const auto& [axis, value] : tile->offset) {
         if (axis != memory_axis) return std::nullopt;
         offset = value;
       }
     }
     return ffi::Tuple(tensor->shape, std::move(strides), std::move(offset));
   }
   
   class TensorMapTypeObj : public TypeObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorMapTypeObj>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TensorMapType", TensorMapTypeObj, TypeObj);
   };
   
   class TensorMapType : public Type {
    public:
     TVM_TIRX_DLL TensorMapType(Location loc = UnknownLoc());
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorMapType, Type, TensorMapTypeObj);
   };
   
   }  // namespace tvm::tirx
   #endif  // TVM_TIRX_TYPE_H_
