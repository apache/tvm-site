
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
   
   class RampObj : public ExprObj {
    public:
     RampObj(PrimExpr base, PrimExpr stride, PrimExpr lanes)
         : base(std::move(base)), stride(std::move(stride)), lanes(std::move(lanes)) {}
     explicit RampObj(ffi::UnsafeInit tag) : base(tag), stride(tag), lanes(tag) {}
   
     PrimExpr base;
     PrimExpr stride;
     PrimExpr lanes;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<RampObj>()
           .def_ro("base", &RampObj::base)
           .def_ro("stride", &RampObj::stride)
           .def_ro("lanes", &RampObj::lanes);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Ramp", RampObj, ExprObj);
   };
   
   class Ramp : public PrimExpr {
    public:
     TVM_TIRX_DLL Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Location loc = UnknownLoc());
     explicit Ramp(ffi::ObjectPtr<RampObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Ramp, PrimExpr, RampObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(RampObj);
   };
   
   class BroadcastObj : public ExprObj {
    public:
     BroadcastObj(PrimExpr value, PrimExpr lanes) : value(std::move(value)), lanes(std::move(lanes)) {}
     explicit BroadcastObj(ffi::UnsafeInit tag) : value(tag), lanes(tag) {}
   
     PrimExpr value;
     PrimExpr lanes;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BroadcastObj>()
           .def_ro("value", &BroadcastObj::value)
           .def_ro("lanes", &BroadcastObj::lanes);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Broadcast", BroadcastObj, ExprObj);
   };
   
   class Broadcast : public PrimExpr {
    public:
     TVM_TIRX_DLL Broadcast(PrimExpr value, PrimExpr lanes, Location loc = UnknownLoc());
     explicit Broadcast(ffi::ObjectPtr<BroadcastObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Broadcast, PrimExpr, BroadcastObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BroadcastObj);
   };
   
   class ShuffleObj : public ExprObj {
    public:
     ffi::Array<PrimExpr> vectors;
     ffi::Array<PrimExpr> indices;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ShuffleObj>()
           .def_ro("vectors", &ShuffleObj::vectors)
           .def_ro("indices", &ShuffleObj::indices);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Shuffle", ShuffleObj, ExprObj);
   };
   
   class Shuffle : public PrimExpr {
    public:
     TVM_TIRX_DLL Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices,
                          Location loc = UnknownLoc());
     TVM_TIRX_DLL static PrimExpr Concat(ffi::Array<PrimExpr> vectors, Location loc = UnknownLoc());
     TVM_TIRX_DLL static PrimExpr ExtractElement(PrimExpr vector, int index,
                                                 Location loc = UnknownLoc());
     explicit Shuffle(ffi::ObjectPtr<ShuffleObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Shuffle, PrimExpr, ShuffleObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ShuffleObj);
   };
   
   }  // namespace prim
   }  // namespace tvm
   #endif  // TVM_IR_PRIM_VECTOR_EXPR_H_
