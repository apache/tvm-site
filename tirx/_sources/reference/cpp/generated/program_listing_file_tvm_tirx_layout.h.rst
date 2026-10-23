
.. _program_listing_file_tvm_tirx_layout.h:

Program Listing for File layout.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_layout.h>` (``tvm/tirx/layout.h``)

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
   #ifndef TVM_TIRX_LAYOUT_H_
   #define TVM_TIRX_LAYOUT_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ffi/enum.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/module.h>
   #include <tvm/ir/prim/op.h>
   #include <tvm/sym/analyzer.h>
   #include <tvm/target/target.h>
   #include <tvm/tirx/tile/thread_scope.h>
   
   #include <algorithm>
   
   namespace tvm {
   
   namespace tirx {
   
   class Layout;
   class TileLayout;
   class Iter;
   using ffi::Array;
   using ffi::Tuple;
   
   class LayoutObj : public ffi::Object {
    public:
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO("tirx.Layout", LayoutObj, ffi::Object);
   };
   
   class Layout : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL void CheckWellFormed() const;
   
     TVM_TIRX_DLL PrimExpr GetSize(ffi::Optional<ffi::String> axis_name = std::nullopt) const;
   
     TVM_TIRX_DLL PrimExpr GetSpan(ffi::Optional<ffi::String> axis_name = std::nullopt) const;
   
     TVM_TIRX_DLL ffi::Map<ffi::String, PrimExpr> Apply(ffi::Array<PrimExpr> coord) const;
     TVM_TIRX_DLL ffi::Map<ffi::String, PrimExpr> Apply(PrimExpr coord) const;
     TVM_TIRX_DLL ffi::Map<ffi::String, PrimExpr> Apply(const ffi::Array<PrimExpr>& coord,
                                                        const ffi::Array<PrimExpr>& shape) const;
   
     TVM_TIRX_DLL Layout Canonicalize() const;
   
     TVM_TIRX_DLL Layout Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                              const ffi::Array<PrimExpr>& inner_shape) const;
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Layout, ffi::ObjectRef, LayoutObj);
   };
   
   // target, subscope, scope, iter -> fused_iter
   using FAxisFuser = ffi::TypedFunction<ffi::Optional<Iter>(Target, ThreadScope, ThreadScope, Iter)>;
   
   // Axis
   class AxisObj : public ffi::EnumObj {
    public:
     bool IsThreadAxis() const {
       static ffi::reflection::TypeAttrColumn state_column(ffi::type_attr::kEnumState);
       static ffi::EnumState state =
           ffi::Any(state_column[_GetOrAllocRuntimeTypeIndex()]).as_or_throw<ffi::EnumState>();
       auto column = state->attrs.Get("thread");
       auto thread = column ? column->Get(ffi::GetRef<ffi::ObjectRef>(this)) : std::nullopt;
       TVM_FFI_ICHECK(thread.has_value())
           << "Axis '" << _str_index << "' has no thread classification";
       return thread->cast<bool>();
     }
   
     bool IsMemoryAxis() const { return !IsThreadAxis(); }
   
     TVM_TIRX_DLL ffi::Optional<ffi::String> GetBackend() const;
   
     TVM_TIRX_DLL ffi::Optional<ThreadScope> GetScope() const;
   
     TVM_TIRX_DLL ffi::Optional<ThreadScope> GetSubscope() const;
   
     TVM_TIRX_DLL ffi::Optional<FAxisFuser> GetFuser() const;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Axis", AxisObj, ffi::EnumObj);
   };
   
   class Axis : public ffi::Enum {
    public:
     const ffi::String& name() const { return get()->_str_index; }
   
     TVM_TIRX_DLL static Axis Get(const ffi::String& name);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Axis, ffi::Enum, AxisObj);
   };
   
   class IterObj : public ffi::Object {
    public:
     explicit IterObj(ffi::UnsafeInit tag) : extent(tag), stride(tag), axis(tag) {}
   
     IterObj(PrimExpr extent, PrimExpr stride, Axis axis)
         : extent(std::move(extent)), stride(std::move(stride)), axis(std::move(axis)) {}
   
     PrimExpr extent;
     PrimExpr stride;
     Axis axis;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IterObj>()
           .def_ro("extent", &IterObj::extent)
           .def_ro("stride", &IterObj::stride)
           .def_ro("axis", &IterObj::axis);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Iter", IterObj, ffi::Object);
   };
   
   class Iter : public ffi::ObjectRef {
    public:
     explicit Iter(PrimExpr extent, PrimExpr stride, Axis axis) {
       auto n = ffi::make_object<IterObj>(std::move(extent), std::move(stride), std::move(axis));
       data_ = std::move(n);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Iter, ffi::ObjectRef, IterObj);
   };
   
   class TileLayoutObj : public LayoutObj {
    public:
     ffi::Array<Iter> shard;
     ffi::Array<Iter> replica;
     ffi::Map<Axis, PrimExpr> offset;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TileLayoutObj>()
           .def_ro("shard", &TileLayoutObj::shard)
           .def_ro("replica", &TileLayoutObj::replica)
           .def_ro("offset", &TileLayoutObj::offset);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TileLayout", TileLayoutObj, LayoutObj);
   };
   
   class TileLayout : public Layout {
    public:
     explicit TileLayout(ffi::Array<Iter> shard, ffi::Array<Iter> replica,
                         ffi::Map<Axis, PrimExpr> offset)
         : Layout(ffi::UnsafeInit{}) {
       auto n = ffi::make_object<TileLayoutObj>();
       n->shard = std::move(shard);
       n->replica = std::move(replica);
       n->offset = std::move(offset);
       data_ = std::move(n);
     }
   
     ffi::Array<PrimExpr> GetShardShape() const {
       return get()->shard.Map([](const Iter& iter) { return iter->extent; });
     }
   
     TVM_TIRX_DLL bool IsContiguous(ffi::Optional<sym::Analyzer> analyzer = std::nullopt) const;
   
     bool IsTrivial() const {
       if (get()->shard.size() > 1) return false;
       if (get()->shard.size() == 1) {
         if (!get()->shard[0]->axis->IsMemoryAxis() || !prim::IsOne(get()->shard[0]->stride))
           return false;
       }
       return get()->replica.size() == 0 && get()->offset.size() == 0;
     }
   
     bool HasThreadAxis() const {
       return std::any_of(get()->shard.begin(), get()->shard.end(),
                          [](const Iter& iter) { return iter->axis->IsThreadAxis(); });
     }
   
     TVM_TIRX_DLL ffi::Optional<Tuple<ThreadScope, ThreadScope>> GetScope() const;
   
     TVM_TIRX_DLL static TileLayout DefaultLayout(ffi::Array<PrimExpr> shape);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TileLayout, Layout, TileLayoutObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TileLayoutObj);
   };
   
   // ComposeLayout
   class ComposeLayoutObj : public LayoutObj {
    public:
     explicit ComposeLayoutObj(TileLayout tile_layout) : tile_layout(std::move(tile_layout)) {}
     explicit ComposeLayoutObj(ffi::UnsafeInit) : tile_layout(ffi::UnsafeInit{}) {}
   
     int per_element;
     int swizzle_len;
     int atom_len;
     bool swizzle_inner;
     TileLayout tile_layout;
     // Cached swizzle masks: inner_mask = (1 << swizzle_len) - 1;
     // outer_mask = inner_mask << atom_len. Set by the ComposeLayout ctor.
     int inner_mask;
     int outer_mask;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ComposeLayoutObj>()
           .def_ro("per_element", &ComposeLayoutObj::per_element)
           .def_ro("swizzle_len", &ComposeLayoutObj::swizzle_len)
           .def_ro("atom_len", &ComposeLayoutObj::atom_len)
           .def_ro("swizzle_inner", &ComposeLayoutObj::swizzle_inner)
           .def_ro("inner_mask", &ComposeLayoutObj::inner_mask)
           .def_ro("outer_mask", &ComposeLayoutObj::outer_mask)
           .def_ro("tile_layout", &ComposeLayoutObj::tile_layout);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.ComposeLayout", ComposeLayoutObj, LayoutObj);
   };
   
   class ComposeLayout : public Layout {
    public:
     TVM_TIRX_DLL explicit ComposeLayout(int per_element, int swizzle_len, int atom_len,
                                         TileLayout tile_layout, bool swizzle_inner = true);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ComposeLayout, Layout, ComposeLayoutObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ComposeLayoutObj);
   };
   
   }  // namespace tirx
   }  // namespace tvm
   
   #endif  // TVM_TIRX_LAYOUT_H_
