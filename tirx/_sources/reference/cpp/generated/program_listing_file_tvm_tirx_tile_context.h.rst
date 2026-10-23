
.. _program_listing_file_tvm_tirx_tile_context.h:

Program Listing for File context.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_tile_context.h>` (``tvm/tirx/tile/context.h``)

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
   #ifndef TVM_TIRX_TILE_CONTEXT_H_
   #define TVM_TIRX_TILE_CONTEXT_H_
   
   #include <tvm/target/target.h>
   #include <tvm/tirx/stmt.h>
   #include <tvm/tirx/tile/thread_scope.h>
   
   namespace tvm {
   namespace tirx {
   namespace tile {
   
   namespace callback {
   inline constexpr ffi::reflection::AttrKey<ffi::Array<Bind>> kPrivateAlloc{"private_alloc"};
   inline constexpr ffi::reflection::AttrKey<ffi::Array<Stmt>> kDeviceInitStmt{"device_init_stmt"};
   inline constexpr ffi::reflection::AttrKey<ffi::Array<Stmt>> kHostInitStmt{"host_init_stmt"};
   inline constexpr ffi::reflection::AttrKey<ffi::Map<TensorVar, ffi::Array<Stmt>>> kPostTensorDefStmt{
       "post_tensor_def_stmt"};
   }  // namespace callback
   
   class DispatchContextObj : public ffi::Object {
    public:
     DispatchContextObj(Target target, ThreadScope thread_scope)
         : target(std::move(target)), thread_scope(std::move(thread_scope)) {}
     explicit DispatchContextObj(ffi::UnsafeInit)
         : target(ffi::UnsafeInit{}), thread_scope(ffi::UnsafeInit{}) {}
   
     Target target;
     ThreadScope thread_scope;
     ffi::Map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params;
     ffi::Map<Var, Range> var_range_map;
     bool alloc_only;
     ffi::Map<ffi::String, ffi::ObjectRef> callbacks;
     ffi::Map<ffi::String, ffi::ObjectRef> shared_state;
     ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter;
     ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra;
     ffi::String scope_kind;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<DispatchContextObj>()
           .def_ro("target", &DispatchContextObj::target)
           .def_ro("thread_scope", &DispatchContextObj::thread_scope)
           .def_ro("launch_params", &DispatchContextObj::launch_params)
           .def_ro("var_range_map", &DispatchContextObj::var_range_map)
           .def_ro("alloc_only", &DispatchContextObj::alloc_only)
           .def_ro("callbacks", &DispatchContextObj::callbacks)
           .def_ro("shared_state", &DispatchContextObj::shared_state)
           .def_ro("inter", &DispatchContextObj::inter)
           .def_ro("intra", &DispatchContextObj::intra)
           .def_ro("scope_kind", &DispatchContextObj::scope_kind);
     }
   
     TVM_TIRX_DLL void AddAllocTensor(TensorVar tensor, ffi::Array<PrimExpr> allocated_addr = {},
                                      ffi::Map<ffi::String, ffi::Any> annotations = {});
   
     TVM_TIRX_DLL void AddInitStmt(Stmt stmt, bool host = false);
   
     TVM_TIRX_DLL void AddPostTensorDefStmt(TensorVar tensor, Stmt stmt);
   
     void SharedStateSet(ffi::String key, ffi::ObjectRef value) { shared_state.Set(key, value); }
   
     ffi::Optional<ffi::ObjectRef> SharedStateGet(ffi::String key) {
       auto it = shared_state.find(key);
       if (it != shared_state.end()) {
         return (*it).second;
       }
       return ffi::Optional<ffi::ObjectRef>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.DispatchContext", DispatchContextObj, ffi::Object);
   };
   
   class DispatchContext : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL DispatchContext(
         Target target, ThreadScope thread_scope,
         ffi::Map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params = {},
         ffi::Map<Var, Range> var_range_map = {}, bool alloc_only = false,
         ffi::Map<ffi::String, ffi::ObjectRef> callbacks = {},
         ffi::Map<ffi::String, ffi::ObjectRef> shared_state = {},
         ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter = {},
         ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra = {}, ffi::String scope_kind = "");
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DispatchContext, ffi::ObjectRef,
                                                   DispatchContextObj);
   };
   
   }  // namespace tile
   }  // namespace tirx
   }  // namespace tvm
   
   #endif  // TVM_TIRX_TILE_CONTEXT_H_
