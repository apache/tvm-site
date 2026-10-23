
.. _program_listing_file_tvm_ir_prim_vector_expr.h:

Program Listing for File vector_expr.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_prim_vector_expr.h>` (``tvm/ir/prim/vector_expr.h``)

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
   #ifndef TVM_IR_PRIM_VECTOR_EXPR_H_
   #define TVM_IR_PRIM_VECTOR_EXPR_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ir/cow.h>
   #include <tvm/ir/expr.h>
   
   namespace tvm {
   namespace prim {
   
   class RampNode : public ExprNode {
    public:
     RampNode(PrimExpr base, PrimExpr stride, PrimExpr lanes)
         : base(std::move(base)), stride(std::move(stride)), lanes(std::move(lanes)) {}
     explicit RampNode(ffi::UnsafeInit tag) : base(tag), stride(tag), lanes(tag) {}
   
     PrimExpr base;
     PrimExpr stride;
     PrimExpr lanes;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<RampNode>()
           .def_ro("base", &RampNode::base)
           .def_ro("stride", &RampNode::stride)
           .def_ro("lanes", &RampNode::lanes);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Ramp", RampNode, ExprNode);
   };
   
   class Ramp : public PrimExpr {
    public:
     TVM_TIRX_DLL Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Location loc = UnknownLoc());
     explicit Ramp(ffi::ObjectPtr<RampNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Ramp, PrimExpr, RampNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(RampNode);
   };
   
   class BroadcastNode : public ExprNode {
    public:
     BroadcastNode(PrimExpr value, PrimExpr lanes)
         : value(std::move(value)), lanes(std::move(lanes)) {}
     explicit BroadcastNode(ffi::UnsafeInit tag) : value(tag), lanes(tag) {}
   
     PrimExpr value;
     PrimExpr lanes;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BroadcastNode>()
           .def_ro("value", &BroadcastNode::value)
           .def_ro("lanes", &BroadcastNode::lanes);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Broadcast", BroadcastNode, ExprNode);
   };
   
   class Broadcast : public PrimExpr {
    public:
     TVM_TIRX_DLL Broadcast(PrimExpr value, PrimExpr lanes, Location loc = UnknownLoc());
     explicit Broadcast(ffi::ObjectPtr<BroadcastNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Broadcast, PrimExpr, BroadcastNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BroadcastNode);
   };
   
   class ShuffleNode : public ExprNode {
    public:
     ffi::Array<PrimExpr> vectors;
     ffi::Array<PrimExpr> indices;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ShuffleNode>()
           .def_ro("vectors", &ShuffleNode::vectors)
           .def_ro("indices", &ShuffleNode::indices);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Shuffle", ShuffleNode, ExprNode);
   };
   
   class Shuffle : public PrimExpr {
    public:
     TVM_TIRX_DLL Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices,
                          Location loc = UnknownLoc());
     TVM_TIRX_DLL static PrimExpr Concat(ffi::Array<PrimExpr> vectors, Location loc = UnknownLoc());
     TVM_TIRX_DLL static PrimExpr ExtractElement(PrimExpr vector, int index,
                                                 Location loc = UnknownLoc());
     explicit Shuffle(ffi::ObjectPtr<ShuffleNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Shuffle, PrimExpr, ShuffleNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ShuffleNode);
   };
   
   }  // namespace prim
   }  // namespace tvm
   #endif  // TVM_IR_PRIM_VECTOR_EXPR_H_
