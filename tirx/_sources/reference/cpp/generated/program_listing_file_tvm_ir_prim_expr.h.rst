
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
   
   using IntImmObj = tvm::IntImmObj;
   using FloatImmObj = tvm::FloatImmObj;
   
   class CastObj : public ExprObj {
    public:
     explicit CastObj(PrimExpr value) : value(std::move(value)) {}
     explicit CastObj(ffi::UnsafeInit tag) : value(tag) {}
   
     PrimExpr value;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CastObj>().def_ro("value", &CastObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Cast", CastObj, ExprObj);
   };
   
   class Cast : public PrimExpr {
    public:
     TVM_TIRX_DLL Cast(PrimType ty, PrimExpr value, Location loc = UnknownLoc());
     explicit Cast(ffi::ObjectPtr<CastObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Cast, PrimExpr, CastObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(CastObj);
   };
   
   template <typename T>
   class BinaryOpObj : public ExprObj {
    public:
     BinaryOpObj(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit BinaryOpObj(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
     }
     static const constexpr int _type_child_slots [[maybe_unused]] = 0;
     static const constexpr bool _type_final [[maybe_unused]] = true;
     TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprObj);
   };
   
   class AddObj : public BinaryOpObj<AddObj> {
    public:
     using BinaryOpObj<AddObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.Add";
   };
   
   class Add : public PrimExpr {
    public:
     TVM_TIRX_DLL Add(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Add(ffi::ObjectPtr<AddObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Add, PrimExpr, AddObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(AddObj);
   };
   
   class LShiftObj : public BinaryOpObj<LShiftObj> {
    public:
     using BinaryOpObj<LShiftObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.LShift";
   };
   
   class LShift : public PrimExpr {
    public:
     TVM_TIRX_DLL LShift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit LShift(ffi::ObjectPtr<LShiftObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LShift, PrimExpr, LShiftObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LShiftObj);
   };
   
   class RShiftObj : public BinaryOpObj<RShiftObj> {
    public:
     using BinaryOpObj<RShiftObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.RShift";
   };
   
   class RShift : public PrimExpr {
    public:
     TVM_TIRX_DLL RShift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit RShift(ffi::ObjectPtr<RShiftObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RShift, PrimExpr, RShiftObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(RShiftObj);
   };
   
   class BitwiseAndObj : public BinaryOpObj<BitwiseAndObj> {
    public:
     using BinaryOpObj<BitwiseAndObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.BitwiseAnd";
   };
   
   class BitwiseAnd : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseAnd(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit BitwiseAnd(ffi::ObjectPtr<BitwiseAndObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseAnd, PrimExpr, BitwiseAndObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseAndObj);
   };
   
   class BitwiseOrObj : public BinaryOpObj<BitwiseOrObj> {
    public:
     using BinaryOpObj<BitwiseOrObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.BitwiseOr";
   };
   
   class BitwiseOr : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseOr(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit BitwiseOr(ffi::ObjectPtr<BitwiseOrObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseOr, PrimExpr, BitwiseOrObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseOrObj);
   };
   
   class BitwiseXorObj : public BinaryOpObj<BitwiseXorObj> {
    public:
     using BinaryOpObj<BitwiseXorObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.BitwiseXor";
   };
   
   class BitwiseXor : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseXor(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit BitwiseXor(ffi::ObjectPtr<BitwiseXorObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseXor, PrimExpr, BitwiseXorObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseXorObj);
   };
   
   class SubObj : public BinaryOpObj<SubObj> {
    public:
     using BinaryOpObj<SubObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.Sub";
   };
   
   class Sub : public PrimExpr {
    public:
     TVM_TIRX_DLL Sub(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
     explicit Sub(ffi::ObjectPtr<SubObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Sub, PrimExpr, SubObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(SubObj);
   };
   
   class MulObj : public BinaryOpObj<MulObj> {
    public:
     using BinaryOpObj<MulObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.Mul";
   };
   
   class Mul : public PrimExpr {
    public:
     TVM_TIRX_DLL Mul(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Mul(ffi::ObjectPtr<MulObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Mul, PrimExpr, MulObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(MulObj);
   };
   
   class DivObj : public BinaryOpObj<DivObj> {
    public:
     using BinaryOpObj<DivObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.Div";
   };
   
   class Div : public PrimExpr {
    public:
     TVM_TIRX_DLL Div(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Div(ffi::ObjectPtr<DivObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Div, PrimExpr, DivObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(DivObj);
   };
   
   class ModObj : public BinaryOpObj<ModObj> {
    public:
     using BinaryOpObj<ModObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.Mod";
   };
   
   class Mod : public PrimExpr {
    public:
     TVM_TIRX_DLL Mod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Mod(ffi::ObjectPtr<ModObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Mod, PrimExpr, ModObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ModObj);
   };
   
   class FloorDivObj : public BinaryOpObj<FloorDivObj> {
    public:
     using BinaryOpObj<FloorDivObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.FloorDiv";
   };
   
   class FloorDiv : public PrimExpr {
    public:
     TVM_TIRX_DLL FloorDiv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit FloorDiv(ffi::ObjectPtr<FloorDivObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloorDiv, PrimExpr, FloorDivObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorDivObj);
   };
   
   class FloorModObj : public BinaryOpObj<FloorModObj> {
    public:
     using BinaryOpObj<FloorModObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.FloorMod";
   };
   
   class FloorMod : public PrimExpr {
    public:
     TVM_TIRX_DLL FloorMod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit FloorMod(ffi::ObjectPtr<FloorModObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloorMod, PrimExpr, FloorModObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorModObj);
   };
   
   class MinObj : public BinaryOpObj<MinObj> {
    public:
     using BinaryOpObj<MinObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.Min";
   };
   
   class Min : public PrimExpr {
    public:
     TVM_TIRX_DLL Min(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Min(ffi::ObjectPtr<MinObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Min, PrimExpr, MinObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(MinObj);
   };
   
   class MaxObj : public BinaryOpObj<MaxObj> {
    public:
     using BinaryOpObj<MaxObj>::BinaryOpObj;
     static constexpr const char* _type_key = "prim.Max";
   };
   
   class Max : public PrimExpr {
    public:
     TVM_TIRX_DLL Max(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Max(ffi::ObjectPtr<MaxObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Max, PrimExpr, MaxObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(MaxObj);
   };
   
   template <typename T>
   class CmpOpObj : public ExprObj {
    public:
     CmpOpObj(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit CmpOpObj(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
     }
     static const constexpr int _type_child_slots [[maybe_unused]] = 0;
     static const constexpr bool _type_final [[maybe_unused]] = true;
     TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprObj);
   };
   
   class EQObj : public CmpOpObj<EQObj> {
    public:
     using CmpOpObj<EQObj>::CmpOpObj;
     static constexpr const char* _type_key = "prim.EQ";
   };
   
   class EQ : public PrimExpr {
    public:
     TVM_TIRX_DLL EQ(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit EQ(ffi::ObjectPtr<EQObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EQ, PrimExpr, EQObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(EQObj);
   };
   
   class NEObj : public CmpOpObj<NEObj> {
    public:
     using CmpOpObj<NEObj>::CmpOpObj;
     static constexpr const char* _type_key = "prim.NE";
   };
   
   class NE : public PrimExpr {
    public:
     TVM_TIRX_DLL NE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit NE(ffi::ObjectPtr<NEObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NE, PrimExpr, NEObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(NEObj);
   };
   
   class LTObj : public CmpOpObj<LTObj> {
    public:
     using CmpOpObj<LTObj>::CmpOpObj;
     static constexpr const char* _type_key = "prim.LT";
   };
   
   class LT : public PrimExpr {
    public:
     TVM_TIRX_DLL LT(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit LT(ffi::ObjectPtr<LTObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LT, PrimExpr, LTObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LTObj);
   };
   
   struct LEObj : public CmpOpObj<LEObj> {
    public:
     using CmpOpObj<LEObj>::CmpOpObj;
     static constexpr const char* _type_key = "prim.LE";
   };
   
   class LE : public PrimExpr {
    public:
     TVM_TIRX_DLL LE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit LE(ffi::ObjectPtr<LEObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LE, PrimExpr, LEObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LEObj);
   };
   
   class GTObj : public CmpOpObj<GTObj> {
    public:
     using CmpOpObj<GTObj>::CmpOpObj;
     static constexpr const char* _type_key = "prim.GT";
   };
   
   class GT : public PrimExpr {
    public:
     TVM_TIRX_DLL GT(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit GT(ffi::ObjectPtr<GTObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GT, PrimExpr, GTObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GTObj);
   };
   
   class GEObj : public CmpOpObj<GEObj> {
    public:
     using CmpOpObj<GEObj>::CmpOpObj;
     static constexpr const char* _type_key = "prim.GE";
   };
   
   class GE : public PrimExpr {
    public:
     TVM_TIRX_DLL GE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit GE(ffi::ObjectPtr<GEObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GE, PrimExpr, GEObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GEObj);
   };
   
   class AndObj : public ExprObj {
    public:
     AndObj(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit AndObj(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AndObj>().def_ro("a", &AndObj::a).def_ro("b", &AndObj::b);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.And", AndObj, ExprObj);
   };
   
   class And : public PrimExpr {
    public:
     TVM_TIRX_DLL And(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit And(ffi::ObjectPtr<AndObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(And, PrimExpr, AndObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(AndObj);
   };
   
   class OrObj : public ExprObj {
    public:
     OrObj(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
     explicit OrObj(ffi::UnsafeInit tag) : a(tag), b(tag) {}
   
     PrimExpr a;
     PrimExpr b;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<OrObj>().def_ro("a", &OrObj::a).def_ro("b", &OrObj::b);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Or", OrObj, ExprObj);
   };
   
   class Or : public PrimExpr {
    public:
     TVM_TIRX_DLL Or(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
     explicit Or(ffi::ObjectPtr<OrObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Or, PrimExpr, OrObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(OrObj);
   };
   
   class NotObj : public ExprObj {
    public:
     explicit NotObj(PrimExpr a) : a(std::move(a)) {}
     explicit NotObj(ffi::UnsafeInit tag) : a(tag) {}
   
     PrimExpr a;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<NotObj>().def_ro("a", &NotObj::a);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Not", NotObj, ExprObj);
   };
   
   class Not : public PrimExpr {
    public:
     TVM_TIRX_DLL Not(PrimExpr a, Location loc = UnknownLoc());
     explicit Not(ffi::ObjectPtr<NotObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Not, PrimExpr, NotObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(NotObj);
   };
   
   class BitwiseNotObj : public ExprObj {
    public:
     explicit BitwiseNotObj(PrimExpr a) : a(std::move(a)) {}
     explicit BitwiseNotObj(ffi::UnsafeInit tag) : a(tag) {}
   
     PrimExpr a;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BitwiseNotObj>().def_ro("a", &BitwiseNotObj::a);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.BitwiseNot", BitwiseNotObj, ExprObj);
   };
   
   class BitwiseNot : public PrimExpr {
    public:
     TVM_TIRX_DLL BitwiseNot(PrimExpr a, Location loc = UnknownLoc());
     explicit BitwiseNot(ffi::ObjectPtr<BitwiseNotObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseNot, PrimExpr, BitwiseNotObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseNotObj);
   };
   
   class SelectObj : public ExprObj {
    public:
     SelectObj(PrimExpr condition, PrimExpr true_value, PrimExpr false_value)
         : condition(std::move(condition)),
           true_value(std::move(true_value)),
           false_value(std::move(false_value)) {}
     explicit SelectObj(ffi::UnsafeInit tag) : condition(tag), true_value(tag), false_value(tag) {}
   
     PrimExpr condition;
     PrimExpr true_value;
     PrimExpr false_value;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<SelectObj>()
           .def_ro("condition", &SelectObj::condition)
           .def_ro("true_value", &SelectObj::true_value)
           .def_ro("false_value", &SelectObj::false_value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Select", SelectObj, ExprObj);
   };
   
   class Select : public PrimExpr {
    public:
     TVM_TIRX_DLL Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value,
                         Location loc = UnknownLoc());
   
     explicit Select(ffi::ObjectPtr<SelectObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Select, PrimExpr, SelectObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(SelectObj);
   };
   
   class LetObj : public ExprObj {
    public:
     LetObj(Var var, PrimExpr value, PrimExpr body)
         : var(std::move(var)), value(std::move(value)), body(std::move(body)) {}
     explicit LetObj(ffi::UnsafeInit tag) : var(tag), value(tag), body(tag) {}
   
     Var var;
     PrimExpr value;
     PrimExpr body;
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LetObj>()
           .def_ro("var", &LetObj::var, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("value", &LetObj::value)
           .def_ro("body", &LetObj::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Let", LetObj, ExprObj);
   };
   
   class Let : public PrimExpr {
    public:
     TVM_TIRX_DLL Let(Var var, PrimExpr value, PrimExpr body, Location loc = UnknownLoc());
     explicit Let(ffi::ObjectPtr<LetObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Let, PrimExpr, LetObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(LetObj);
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
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::CastObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::AddObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseNotObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseXorObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseOrObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseAndObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::RShiftObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LShiftObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::SubObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::MulObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::DivObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::ModObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::FloorDivObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::FloorModObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::MinObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::MaxObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::EQObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::NEObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LTObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LEObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::GTObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::GEObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::AndObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::OrObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::NotObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::SelectObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::RampObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::BroadcastObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::LetObj> = true;
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, prim::ShuffleObj> = true;
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_IR_PRIM_EXPR_H_
