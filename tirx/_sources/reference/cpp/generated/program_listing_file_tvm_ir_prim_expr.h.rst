
.. _program_listing_file_tvm_ir_prim_expr.h:

Program Listing for File expr.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_prim_expr.h>` (``tvm/ir/prim/expr.h``)

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
   
   // Acknowledgement: Many low-level IR nodes originate from Halide.
   #ifndef TVM_IR_PRIM_EXPR_H_
   #define TVM_IR_PRIM_EXPR_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/dtype.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/attrs.h>
   #include <tvm/ir/cow.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/object_functor.h>
   #include <tvm/ir/prim/vector_expr.h>
   #include <tvm/runtime/base.h>
   
   #include <algorithm>
   #include <iostream>
   #include <limits>
   #include <string>
   #include <unordered_map>
   #include <utility>
   
   namespace tvm {
   namespace prim {
   
   using IntImmNode = tvm::IntImmNode;
   using FloatImmNode = tvm::FloatImmNode;
   
   class CastNode : public ExprNode {
    public:
     explicit CastNode(PrimExpr value) : value(std::move(value)) {}
     explicit CastNode(ffi::UnsafeInit tag) : value(tag) {}
   
     PrimExpr value;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CastNode>().def_ro("value", &CastNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Cast", CastNode, ExprNode);
   };
   
   class Cast : public PrimExpr {
    public:
     TVM_TIRX_DLL Cast(PrimType ty, PrimExpr value, Location loc = UnknownLoc());
     explicit Cast(ffi::ObjectPtr<CastNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Cast, PrimExpr, CastNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(CastNode);
   };
   
   template <typename T>
   class BinaryOpNode : public ExprNode {
    public:
     BinaryOpNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit BinaryOpNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
     }
     static const constexpr int _type_child_slots [[maybe_unused]] = 0;
     static const constexpr bool _type_final [[maybe_unused]] = true;
     TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprNode);
   };
   
   class AddNode : public BinaryOpNode<AddNode> {
    public:
     using BinaryOpNode<AddNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.Add";
   };
   
   class Add : public PrimExpr {
    public:
     TVM_TIRX_DLL Add(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Add(ffi::ObjectPtr<AddNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Add, PrimExpr, AddNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(AddNode);
   };
   
   class LShiftNode : public BinaryOpNode<LShiftNode> {
    public:
     using BinaryOpNode<LShiftNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.LShift";
   };
   
   class LShift : public PrimExpr {
    public:
     TVM_TIRX_DLL LShift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit LShift(ffi::ObjectPtr<LShiftNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LShift, PrimExpr, LShiftNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LShiftNode);
   };
   
   class RShiftNode : public BinaryOpNode<RShiftNode> {
    public:
     using BinaryOpNode<RShiftNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.RShift";
   };
   
   class RShift : public PrimExpr {
    public:
     TVM_TIRX_DLL RShift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit RShift(ffi::ObjectPtr<RShiftNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RShift, PrimExpr, RShiftNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(RShiftNode);
   };
   
   class BitwiseAndNode : public BinaryOpNode<BitwiseAndNode> {
    public:
     using BinaryOpNode<BitwiseAndNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.BitwiseAnd";
   };
   
   class BitwiseAnd : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseAnd(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit BitwiseAnd(ffi::ObjectPtr<BitwiseAndNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseAnd, PrimExpr, BitwiseAndNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseAndNode);
   };
   
   class BitwiseOrNode : public BinaryOpNode<BitwiseOrNode> {
    public:
     using BinaryOpNode<BitwiseOrNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.BitwiseOr";
   };
   
   class BitwiseOr : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseOr(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit BitwiseOr(ffi::ObjectPtr<BitwiseOrNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseOr, PrimExpr, BitwiseOrNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseOrNode);
   };
   
   class BitwiseXorNode : public BinaryOpNode<BitwiseXorNode> {
    public:
     using BinaryOpNode<BitwiseXorNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.BitwiseXor";
   };
   
   class BitwiseXor : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseXor(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit BitwiseXor(ffi::ObjectPtr<BitwiseXorNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseXor, PrimExpr, BitwiseXorNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseXorNode);
   };
   
   class SubNode : public BinaryOpNode<SubNode> {
    public:
     using BinaryOpNode<SubNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.Sub";
   };
   
   class Sub : public PrimExpr {
    public:
     TVM_TIRX_DLL Sub(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
     explicit Sub(ffi::ObjectPtr<SubNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Sub, PrimExpr, SubNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(SubNode);
   };
   
   class MulNode : public BinaryOpNode<MulNode> {
    public:
     using BinaryOpNode<MulNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.Mul";
   };
   
   class Mul : public PrimExpr {
    public:
     TVM_TIRX_DLL Mul(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Mul(ffi::ObjectPtr<MulNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Mul, PrimExpr, MulNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(MulNode);
   };
   
   class DivNode : public BinaryOpNode<DivNode> {
    public:
     using BinaryOpNode<DivNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.Div";
   };
   
   class Div : public PrimExpr {
    public:
     TVM_TIRX_DLL Div(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Div(ffi::ObjectPtr<DivNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Div, PrimExpr, DivNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(DivNode);
   };
   
   class ModNode : public BinaryOpNode<ModNode> {
    public:
     using BinaryOpNode<ModNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.Mod";
   };
   
   class Mod : public PrimExpr {
    public:
     TVM_TIRX_DLL Mod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Mod(ffi::ObjectPtr<ModNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Mod, PrimExpr, ModNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ModNode);
   };
   
   class FloorDivNode : public BinaryOpNode<FloorDivNode> {
    public:
     using BinaryOpNode<FloorDivNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.FloorDiv";
   };
   
   class FloorDiv : public PrimExpr {
    public:
     TVM_TIRX_DLL FloorDiv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit FloorDiv(ffi::ObjectPtr<FloorDivNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloorDiv, PrimExpr, FloorDivNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorDivNode);
   };
   
   class FloorModNode : public BinaryOpNode<FloorModNode> {
    public:
     using BinaryOpNode<FloorModNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.FloorMod";
   };
   
   class FloorMod : public PrimExpr {
    public:
     TVM_TIRX_DLL FloorMod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit FloorMod(ffi::ObjectPtr<FloorModNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloorMod, PrimExpr, FloorModNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorModNode);
   };
   
   class MinNode : public BinaryOpNode<MinNode> {
    public:
     using BinaryOpNode<MinNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.Min";
   };
   
   class Min : public PrimExpr {
    public:
     TVM_TIRX_DLL Min(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Min(ffi::ObjectPtr<MinNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Min, PrimExpr, MinNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(MinNode);
   };
   
   class MaxNode : public BinaryOpNode<MaxNode> {
    public:
     using BinaryOpNode<MaxNode>::BinaryOpNode;
     static constexpr const char* _type_key = "prim.Max";
   };
   
   class Max : public PrimExpr {
    public:
     TVM_TIRX_DLL Max(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Max(ffi::ObjectPtr<MaxNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Max, PrimExpr, MaxNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(MaxNode);
   };
   
   template <typename T>
   class CmpOpNode : public ExprNode {
    public:
     CmpOpNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit CmpOpNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
     }
     static const constexpr int _type_child_slots [[maybe_unused]] = 0;
     static const constexpr bool _type_final [[maybe_unused]] = true;
     TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprNode);
   };
   
   class EQNode : public CmpOpNode<EQNode> {
    public:
     using CmpOpNode<EQNode>::CmpOpNode;
     static constexpr const char* _type_key = "prim.EQ";
   };
   
   class EQ : public PrimExpr {
    public:
     TVM_TIRX_DLL EQ(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit EQ(ffi::ObjectPtr<EQNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EQ, PrimExpr, EQNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(EQNode);
   };
   
   class NENode : public CmpOpNode<NENode> {
    public:
     using CmpOpNode<NENode>::CmpOpNode;
     static constexpr const char* _type_key = "prim.NE";
   };
   
   class NE : public PrimExpr {
    public:
     TVM_TIRX_DLL NE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit NE(ffi::ObjectPtr<NENode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NE, PrimExpr, NENode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(NENode);
   };
   
   class LTNode : public CmpOpNode<LTNode> {
    public:
     using CmpOpNode<LTNode>::CmpOpNode;
     static constexpr const char* _type_key = "prim.LT";
   };
   
   class LT : public PrimExpr {
    public:
     TVM_TIRX_DLL LT(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit LT(ffi::ObjectPtr<LTNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LT, PrimExpr, LTNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LTNode);
   };
   
   struct LENode : public CmpOpNode<LENode> {
    public:
     using CmpOpNode<LENode>::CmpOpNode;
     static constexpr const char* _type_key = "prim.LE";
   };
   
   class LE : public PrimExpr {
    public:
     TVM_TIRX_DLL LE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit LE(ffi::ObjectPtr<LENode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LE, PrimExpr, LENode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LENode);
   };
   
   class GTNode : public CmpOpNode<GTNode> {
    public:
     using CmpOpNode<GTNode>::CmpOpNode;
     static constexpr const char* _type_key = "prim.GT";
   };
   
   class GT : public PrimExpr {
    public:
     TVM_TIRX_DLL GT(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit GT(ffi::ObjectPtr<GTNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GT, PrimExpr, GTNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GTNode);
   };
   
   class GENode : public CmpOpNode<GENode> {
    public:
     using CmpOpNode<GENode>::CmpOpNode;
     static constexpr const char* _type_key = "prim.GE";
   };
   
   class GE : public PrimExpr {
    public:
     TVM_TIRX_DLL GE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit GE(ffi::ObjectPtr<GENode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GE, PrimExpr, GENode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GENode);
   };
   
   class AndNode : public ExprNode {
    public:
     AndNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit AndNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AndNode>().def_ro("a", &AndNode::a).def_ro("b", &AndNode::b);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.And", AndNode, ExprNode);
   };
   
   class And : public PrimExpr {
    public:
     TVM_TIRX_DLL And(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit And(ffi::ObjectPtr<AndNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(And, PrimExpr, AndNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(AndNode);
   };
   
   class OrNode : public ExprNode {
    public:
     OrNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit OrNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<OrNode>().def_ro("a", &OrNode::a).def_ro("b", &OrNode::b);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Or", OrNode, ExprNode);
   };
   
   class Or : public PrimExpr {
    public:
     TVM_TIRX_DLL Or(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Or(ffi::ObjectPtr<OrNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Or, PrimExpr, OrNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(OrNode);
   };
   
   class NotNode : public ExprNode {
    public:
     explicit NotNode(PrimExpr a) : a(std::move(a)) {}
     explicit NotNode(ffi::UnsafeInit tag) : a(tag) {}
   
     PrimExpr a;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<NotNode>().def_ro("a", &NotNode::a);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Not", NotNode, ExprNode);
   };
   
   class Not : public PrimExpr {
    public:
     TVM_TIRX_DLL Not(PrimExpr a, Location loc = UnknownLoc());
     explicit Not(ffi::ObjectPtr<NotNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Not, PrimExpr, NotNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(NotNode);
   };
   
   class BitwiseNotNode : public ExprNode {
    public:
     explicit BitwiseNotNode(PrimExpr a) : a(std::move(a)) {}
     explicit BitwiseNotNode(ffi::UnsafeInit tag) : a(tag) {}
   
     PrimExpr a;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BitwiseNotNode>().def_ro("a", &BitwiseNotNode::a);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.BitwiseNot", BitwiseNotNode, ExprNode);
   };
   
   class BitwiseNot : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseNot(PrimExpr a, Location loc = UnknownLoc());
     explicit BitwiseNot(ffi::ObjectPtr<BitwiseNotNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseNot, PrimExpr, BitwiseNotNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseNotNode);
   };
   
   class SelectNode : public ExprNode {
    public:
     SelectNode(PrimExpr condition, PrimExpr true_value, PrimExpr false_value)
         : condition(std::move(condition)),
           true_value(std::move(true_value)),
           false_value(std::move(false_value)) {}
     explicit SelectNode(ffi::UnsafeInit tag) : condition(tag), true_value(tag), false_value(tag) {}
   
     PrimExpr condition;
     PrimExpr true_value;
     PrimExpr false_value;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<SelectNode>()
           .def_ro("condition", &SelectNode::condition)
           .def_ro("true_value", &SelectNode::true_value)
           .def_ro("false_value", &SelectNode::false_value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Select", SelectNode, ExprNode);
   };
   
   class Select : public PrimExpr {
    public:
     TVM_TIRX_DLL Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value,
                         Location loc = UnknownLoc());
   
     explicit Select(ffi::ObjectPtr<SelectNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Select, PrimExpr, SelectNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(SelectNode);
   };
   
   class LetNode : public ExprNode {
    public:
     LetNode(Var var, PrimExpr value, PrimExpr body)
         : var(std::move(var)), value(std::move(value)), body(std::move(body)) {}
     explicit LetNode(ffi::UnsafeInit tag) : var(tag), value(tag), body(tag) {}
   
     Var var;
     PrimExpr value;
     PrimExpr body;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LetNode>()
           .def_ro("var", &LetNode::var, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("value", &LetNode::value)
           .def_ro("body", &LetNode::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Let", LetNode, ExprNode);
   };
   
   class Let : public PrimExpr {
    public:
     TVM_TIRX_DLL Let(Var var, PrimExpr value, PrimExpr body, Location loc = UnknownLoc());
     explicit Let(ffi::ObjectPtr<LetNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Let, PrimExpr, LetNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LetNode);
   };
   
   /*
    * \brief Template function to convert Map to unordered_map
    *  Sometimes useful for API gluing when internal uses unordered_map
    * \param dmap The container map
    * \return The corresponding unordered_map.
    * \tparam K the key of the Map.
    * \tparam V the value of the Map.
    */
   template <typename K, typename V>
   inline std::unordered_map<K, V> as_unordered_map(const ffi::Map<K, V>& dmap) {
     std::unordered_map<K, V> ret;
     for (auto kv : dmap) {
       ret[kv.first] = kv.second;
     }
     return ret;
   }
   
   struct ExprDeepEqual {
    public:
     TVM_TIRX_DLL bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const;
   };
   
   }  // namespace prim
   
   namespace ffi {
   
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::CastNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::AddNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseNotNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseXorNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseOrNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseAndNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::RShiftNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LShiftNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::SubNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::MulNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::DivNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::ModNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::FloorDivNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::FloorModNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::MinNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::MaxNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::EQNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::NENode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LTNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LENode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::GTNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::GENode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::AndNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::OrNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::NotNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::SelectNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::RampNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BroadcastNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LetNode> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::ShuffleNode> = true;
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_IR_PRIM_EXPR_H_
