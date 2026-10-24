
.. _program_listing_file_tvm_ir_expr.h:

Program Listing for File expr.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_expr.h>` (``tvm/ir/expr.h``)

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
   
   #ifndef TVM_IR_EXPR_H_
   #define TVM_IR_EXPR_H_
   
   #include <tvm/ffi/big_int.h>
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ffi/container/variant.h>
   #include <tvm/ffi/dtype.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/extra/dataclass.h>
   #include <tvm/ffi/extra/structural_equal.h>
   #include <tvm/ffi/extra/structural_hash.h>
   #include <tvm/ffi/extra/structural_mutate.h>
   #include <tvm/ffi/extra/structural_visit.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/reflection/native_function.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/attrs.h>
   #include <tvm/ir/cow.h>
   #include <tvm/ir/location.h>
   #include <tvm/ir/type.h>
   
   #include <algorithm>
   #include <cstddef>
   #include <cstdint>
   #include <functional>
   #include <limits>
   #include <optional>
   #include <ostream>
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   
   class ExprObj : public ffi::Object {
    public:
     mutable Location loc = UnknownLoc();
   
     mutable Type ty = MissingType();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       // loc does not participate in structural equal and hash.
       refl::ObjectDef<ExprObj>()
           .def_ro("loc", &ExprObj::loc, refl::DefaultValue(UnknownLoc()),
                   refl::AttachFieldFlag::SEqHashIgnore())
           .def_ro("ty", &ExprObj::ty, refl::DefaultValue(MissingType()));
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
   
     static constexpr const uint32_t _type_child_slots = 64;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Expr", ExprObj, ffi::Object);
   };
   
   class Expr : public ffi::ObjectRef {
    public:
     // Expressions do not implicitly compare by object identity or address.  Callers must name
     // whether they intend object identity, structural equality, or primitive symbolic comparison.
     bool operator==(const Expr& other) const = delete;
     bool operator!=(const Expr& other) const = delete;
     bool operator<(const Expr& other) const = delete;
   
     explicit Expr(ffi::ObjectPtr<ExprObj> node) : ffi::ObjectRef(std::move(node)) {
       TVM_FFI_CHECK(defined(), ValueError) << "Expr expects a defined node";
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Expr, ffi::ObjectRef, ExprObj);
   };
   
   class StagingExprObj : public ExprObj {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<StagingExprObj>(); }
     static constexpr const uint32_t _type_child_slots = 4;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.StagingExpr", StagingExprObj, ExprObj);
   };
   
   class StagingExpr : public Expr {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StagingExpr, Expr, StagingExprObj);
   };
   
   class OpaqueExprObj : public ExprObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<OpaqueExprObj>();
     }
   
     static constexpr const uint32_t _type_child_slots = 2;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.OpaqueExpr", OpaqueExprObj, ExprObj);
   };
   
   class OpaqueExpr : public Expr {
    public:
     explicit OpaqueExpr(ffi::ObjectPtr<OpaqueExprObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OpaqueExpr, Expr, OpaqueExprObj);
   };
   
   template <typename ExpectedType>
   class TypedExpr : public Expr {
    public:
     ExpectedType ty() const {
       const auto* node = get();
       TVM_FFI_DCHECK(node != nullptr);
       const auto* ty_node = node->ExprObj::ty.template as<typename ExpectedType::ContainerType>();
       TVM_FFI_DCHECK(ty_node != nullptr);
       return ffi::GetRef<ExpectedType>(ty_node);
     }
   
     explicit TypedExpr(ffi::ObjectPtr<ExprObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TypedExpr, Expr, ExprObj);
     static constexpr bool _type_container_is_exact = false;
   };
   
   class PrimExpr : public TypedExpr<PrimType> {
    public:
     using TypedExpr<PrimType>::ty;
   
     TVM_TIRX_DLL PrimExpr(int32_t value);  // NOLINT(*)
     TVM_TIRX_DLL PrimExpr(float value);  // NOLINT(*)
   
     explicit PrimExpr(ffi::ObjectPtr<ExprObj> node) : TypedExpr<PrimType>(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimExpr, TypedExpr<PrimType>, ExprObj);
     static constexpr bool _type_container_is_exact = false;
   };
   
   class IntExpr : public PrimExpr {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntExpr, PrimExpr, ExprObj);
     static constexpr bool _type_container_is_exact = false;
   };
   
   namespace ffi::detail {
   // Array members require storage availability before the expression traits below are complete.
   template <>
   inline constexpr bool storage_enabled_v<Expr> = true;
   template <>
   inline constexpr bool storage_enabled_v<PrimExpr> = true;
   }  // namespace ffi::detail
   
   class ConstantObj : public ExprObj {
    public:
     static constexpr uint32_t _type_child_slots = 5;
     static void RegisterReflection() { ffi::reflection::ObjectDef<ConstantObj>(); }
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Constant", ConstantObj, ExprObj);
   };
   
   class Constant : public Expr {
    public:
     explicit Constant(ffi::ObjectPtr<ConstantObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Constant, Expr, ConstantObj);
   };
   
   class GenericConstObj : public ConstantObj {
    public:
     ffi::Any value;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<GenericConstObj>().def_ro("value", &GenericConstObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GenericConst", GenericConstObj, ConstantObj);
   };
   
   class GenericConst : public Constant {
    public:
     TVM_TIRX_DLL GenericConst(Type ty, ffi::Any value, Location loc = UnknownLoc());
   
     explicit GenericConst(ffi::ObjectPtr<GenericConstObj> node) : Constant(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GenericConst, Constant, GenericConstObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GenericConstObj);
   };
   
   class StringImmObj : public ConstantObj {
    public:
     ffi::String value;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<StringImmObj>().def_ro("value", &StringImmObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.StringImm", StringImmObj, ConstantObj);
   };
   
   class StringImm : public Constant {
    public:
     TVM_TIRX_DLL explicit StringImm(ffi::String value, Location loc = UnknownLoc());
   
     explicit StringImm(ffi::ObjectPtr<StringImmObj> node) : Constant(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StringImm, Constant, StringImmObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmObj);
   };
   
   class DataTypeImmObj : public ConstantObj {
    public:
     DLDataType value;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<DataTypeImmObj>().def_ro("value", &DataTypeImmObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.DataTypeImm", DataTypeImmObj, ConstantObj);
   };
   
   class DataTypeImm : public Constant {
    public:
     TVM_TIRX_DLL explicit DataTypeImm(DLDataType value, Location loc = UnknownLoc());
   
     explicit DataTypeImm(ffi::ObjectPtr<DataTypeImmObj> node) : Constant(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataTypeImm, Constant, DataTypeImmObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(DataTypeImmObj);
   };
   
   class IntImmObj : public ConstantObj {
    public:
     ffi::BigInt value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IntImmObj>().def_ro("value", &IntImmObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IntImm", IntImmObj, ConstantObj);
   };
   
   class IntImm : public PrimExpr {
    public:
     TVM_TIRX_DLL IntImm(PrimType value_ty, ffi::BigInt value, Location loc = UnknownLoc());
   
     template <typename Enum, std::enable_if_t<std::is_enum_v<Enum>, int> = 0>
     IntImm(PrimType value_ty, Enum value, Location loc = UnknownLoc())
         : IntImm(std::move(value_ty), ffi::BigInt(static_cast<std::underlying_type_t<Enum>>(value)),
                  std::move(loc)) {}
   
     static IntImm Bool(bool value, Location loc = UnknownLoc()) {
       return IntImm(PrimType::Bool(), value, loc);
     }
   
     static IntImm Int32(ffi::BigInt value, Location loc = UnknownLoc()) {
       return IntImm(PrimType::Int(32), std::move(value), loc);
     }
   
     template <typename Enum, std::enable_if_t<std::is_enum_v<Enum>, int> = 0>
     static IntImm Int32(Enum value, Location loc = UnknownLoc()) {
       return IntImm(PrimType::Int(32), value, std::move(loc));
     }
   
     static IntImm Int64(ffi::BigInt value, Location loc = UnknownLoc()) {
       return IntImm(PrimType::Int(64), std::move(value), loc);
     }
   
     template <typename Enum, std::enable_if_t<std::is_enum_v<Enum>, int> = 0>
     static IntImm Int64(Enum value, Location loc = UnknownLoc()) {
       return IntImm(PrimType::Int(64), value, std::move(loc));
     }
   
     explicit IntImm(ffi::ObjectPtr<IntImmObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntImm, PrimExpr, IntImmObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IntImmObj);
   };
   
   class FloatImmObj : public ConstantObj {
    public:
     double value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FloatImmObj>().def_ro("value", &FloatImmObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FloatImm", FloatImmObj, ConstantObj);
   };
   
   class FloatImm : public PrimExpr {
    public:
     TVM_TIRX_DLL FloatImm(PrimType value_ty, double value, Location loc = UnknownLoc());
   
     explicit FloatImm(ffi::ObjectPtr<FloatImmObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloatImm, PrimExpr, FloatImmObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(FloatImmObj);
   };
   
   class TupleObj : public ExprObj {
    public:
     ffi::Array<Expr> fields;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleObj>().def_ro("fields", &TupleObj::fields);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Tuple", TupleObj, ExprObj);
   };
   
   class Tuple : public Expr {
    public:
     TVM_TIRX_DLL explicit Tuple(ffi::Array<Expr> fields, Location loc = UnknownLoc());
   
     explicit Tuple(ffi::ObjectPtr<TupleObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Tuple, Expr, TupleObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleObj);
   };
   
   class TupleGetItemObj : public ExprObj {
    public:
     explicit TupleGetItemObj(Expr tuple) : tuple(std::move(tuple)) {}
     explicit TupleGetItemObj(ffi::UnsafeInit tag) : tuple(tag) {}
   
     Expr tuple;
     int index;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleGetItemObj>()
           .def_ro("tuple_value", &TupleGetItemObj::tuple)
           .def_ro("index", &TupleGetItemObj::index);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleGetItem", TupleGetItemObj, ExprObj);
   };
   
   class TupleGetItem : public Expr {
    public:
     TVM_TIRX_DLL TupleGetItem(Expr tuple, int index, Location loc = UnknownLoc());
   
     explicit TupleGetItem(ffi::ObjectPtr<TupleGetItemObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleGetItem, Expr, TupleGetItemObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemObj);
   };
   
   class TensorLoadObj : public ExprObj {
    public:
     explicit TensorLoadObj(Expr source) : source(std::move(source)) {}
     explicit TensorLoadObj(ffi::UnsafeInit tag) : source(tag) {}
   
     Expr source;
     ffi::Array<PrimExpr> indices;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorLoadObj>()
           .def_ro("source", &TensorLoadObj::source)
           .def_ro("indices", &TensorLoadObj::indices);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorLoad", TensorLoadObj, ExprObj);
   };
   
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<
       ffi::reflection::NativeFunctionView<PrimType(const TensorLoadObj*)>>
       kTensorLoadInferType{"__tvm_tensor_load_infer_type__"};
   }  // namespace type_attr
   
   class TensorLoad : public PrimExpr {
    public:
     TVM_TIRX_DLL TensorLoad(Expr source, ffi::Array<PrimExpr> indices, Location loc = UnknownLoc());
   
     TVM_TIRX_DLL static TensorLoad Unchecked(PrimType ty, Expr source, ffi::Array<PrimExpr> indices,
                                              Location loc = UnknownLoc());
   
     TVM_TIRX_DLL explicit TensorLoad(ffi::ObjectPtr<TensorLoadObj> node);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorLoad, PrimExpr, TensorLoadObj);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorLoadObj);
   };
   
   class VarObj : public ExprObj {
    public:
     ffi::String name;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<VarObj>().def_ro("name", &VarObj::name, refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
     static constexpr const uint32_t _type_child_slots = 1;
     // VarObj reserves its sole child slot for the final relax::DataflowVarNode subtype, so its
     // descendant type-index range cannot overflow.
     static constexpr bool _type_child_slots_can_overflow = false;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Var", VarObj, ExprObj);
   };
   
   class Var : public Expr {
    public:
     TVM_TIRX_DLL explicit Var(ffi::String name, Type ty, Location loc = UnknownLoc());
   
     TVM_TIRX_DLL Var CopyWithName(const ffi::String& name) const;
   
     TVM_TIRX_DLL Var CopyWithSuffix(const ffi::String& suffix) const;
   
     TVM_TIRX_DLL Var CopyWithDType(PrimType dtype) const;
   
     TVM_TIRX_DLL Var CopyWithType(Type type) const;
   
     explicit Var(ffi::ObjectPtr<VarObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Var, Expr, VarObj);
   };
   
   class PrimVar : public PrimExpr {
    public:
     explicit PrimVar(ffi::String name, PrimType dtype = PrimType::Int(32),
                      Location loc = UnknownLoc())
         : PrimExpr(Var(std::move(name), std::move(dtype), std::move(loc)).as_or_throw<PrimExpr>()) {}
   
     explicit PrimVar(ffi::String name, Type type_annotation, Location loc = UnknownLoc())
         : PrimExpr(Var(std::move(name), std::move(type_annotation), std::move(loc))
                        .as_or_throw<PrimExpr>()) {}
   
     operator Var() const { return this->as_or_throw<Var>(); }
   
     PrimVar CopyWithSuffix(const ffi::String& suffix) const {
       return this->as_or_throw<Var>().CopyWithSuffix(suffix).as_or_throw<PrimVar>();
     }
     PrimVar CopyWithDType(PrimType dtype) const {
       return this->as_or_throw<Var>().CopyWithDType(dtype).as_or_throw<PrimVar>();
     }
   
     explicit PrimVar(ffi::ObjectPtr<VarObj> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimVar, PrimExpr, VarObj);
     static constexpr bool _type_container_is_exact = false;
   };
   
   class LambdaExprObj : public StagingExprObj {
    public:
     ffi::Array<Var> vars;
     Expr body;
   
     explicit LambdaExprObj(ffi::UnsafeInit tag) : body(tag) {}
     explicit LambdaExprObj(Expr body) : body(std::move(body)) {}
     TVM_TIRX_DLL Expr Apply(const ffi::Array<Expr>& arguments) const;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LambdaExprObj>()
           .def_ro("vars", &LambdaExprObj::vars, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("body", &LambdaExprObj::body);
     }
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.LambdaExpr", LambdaExprObj, StagingExprObj);
   };
   
   class LambdaExpr : public StagingExpr {
    public:
     TVM_TIRX_DLL explicit LambdaExpr(ffi::Array<Var> vars, Expr body, Location loc = UnknownLoc());
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LambdaExpr, StagingExpr, LambdaExprObj);
   };
   
   class GlobalVar;
   class GlobalVarObj : public ExprObj {
    public:
     ffi::String name_hint;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<GlobalVarObj>().def_ro("name_hint", &GlobalVarObj::name_hint);
       // A GlobalVar identifies a module-level symbol.  Its type is derived from the
       // corresponding function definition and is not part of the symbol identity.
       refl::TypeAttrDef<GlobalVarObj>()
           .def(tvm::ffi::type_attr::kSEqual.name, &GlobalVarObj::SEqual)
           .def(tvm::ffi::type_attr::kSHash.name, &GlobalVarObj::SHash);
     }
   
     bool SEqual(const GlobalVarObj* other,
                 ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
       return equal(name_hint, other->name_hint, false, "name_hint");
     }
   
     int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
       return hash(name_hint, init_hash, false);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GlobalVar", GlobalVarObj, ExprObj);
   };
   
   class GlobalVar : public Expr {
    public:
     TVM_TIRX_DLL explicit GlobalVar(ffi::String name_hint, Location loc = UnknownLoc());
   
     explicit GlobalVar(ffi::ObjectPtr<GlobalVarObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GlobalVar, Expr, GlobalVarObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalVarObj);
   };
   
   enum class CallEffectKind : int {
     kExprAnnotation = 0,
     kPure = 1,
     kReadState = 2,
     kUpdateState = 3,
     kOpaque = kUpdateState,
     kSpecialCallArg = 4,
     kEmbedInfo = 5,
     kControlJump = 6,
   };
   
   inline std::ostream& operator<<(std::ostream& os, CallEffectKind side_effect) {
     switch (side_effect) {
       case CallEffectKind::kExprAnnotation:
         return os << "kExprAnnotation";
   
       case CallEffectKind::kPure:
         return os << "kPure";
   
       case CallEffectKind::kReadState:
         return os << "kReadState";
   
       case CallEffectKind::kUpdateState:
         return os << "kUpdateState";
   
       case CallEffectKind::kSpecialCallArg:
         return os << "kSpecialCallArg";
   
       case CallEffectKind::kEmbedInfo:
         return os << "kEmbedInfo";
   
       case CallEffectKind::kControlJump:
         return os << "kControlJump";
   
       default:
         TVM_FFI_THROW(InternalError) << "Unknown CallEffectKind: " << static_cast<int>(side_effect);
     }
   }
   
   class CallObj : public ExprObj {
    public:
     explicit CallObj(Expr op) : op(std::move(op)) {}
     explicit CallObj(ffi::UnsafeInit tag) : op(tag) {}
   
     Expr op;
   
     ffi::Array<Expr> args;
   
     ffi::Optional<Attrs> attrs;
   
     ffi::Array<Type> ty_args;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CallObj>()
           .def_ro("op", &CallObj::op)
           .def_ro("args", &CallObj::args)
           .def_ro("attrs", &CallObj::attrs)
           .def_ro("ty_args", &CallObj::ty_args);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Call", CallObj, ExprObj);
   };
   
   class Call : public Expr {
    public:
     TVM_TIRX_DLL Call(ffi::Optional<Type> ret_ty, Expr op, ffi::Array<Expr> args,
                       ffi::Optional<Attrs> attrs = std::nullopt,
                       ffi::Array<Type> ty_args = ffi::Array<Type>(), Location loc = UnknownLoc());
   
     TVM_TIRX_DLL void Validate() const;
     TVM_TIRX_DLL static Type ReinferType(const CallObj* call);
   
     explicit Call(ffi::ObjectPtr<CallObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Call, Expr, CallObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(CallObj);
   };
   
   namespace op_attr {
   inline constexpr ffi::reflection::AttrKey<CallEffectKind> kCallEffectKind{"TCallEffectKind"};
   }  // namespace op_attr
   
   class RangeObj : public ffi::Object {
    public:
     PrimExpr begin;
     PrimExpr extent;
     mutable Location loc = UnknownLoc();
     explicit RangeObj(ffi::UnsafeInit tag) : begin(tag), extent(tag) {}
     RangeObj(PrimExpr begin, PrimExpr extent, Location loc = UnknownLoc())
         : begin(begin), extent(extent), loc(loc) {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<RangeObj>()
           .def_ro("begin", &RangeObj::begin)
           .def_ro("extent", &RangeObj::extent)
           .def_ro("loc", &RangeObj::loc, refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Range", RangeObj, ffi::Object);
   };
   
   class Range : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL Range(PrimExpr begin, PrimExpr end, Location loc = UnknownLoc());
     TVM_TIRX_DLL static Range FromMinExtent(PrimExpr min, PrimExpr extent,
                                             Location loc = UnknownLoc());
     TVM_TIRX_DLL static PrimExpr GetExtentFromBeginEnd(const PrimExpr& begin, const PrimExpr& end);
     // declare range.
     explicit Range(ffi::UnsafeInit tag, ffi::ObjectPtr<RangeObj> n) : ffi::ObjectRef(tag) {
       data_ = std::move(n);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Range, ffi::ObjectRef, RangeObj);
   };
   
   class TensorRegionObj : public ExprObj {
    public:
     explicit TensorRegionObj(Expr source) : source(std::move(source)) {}
     explicit TensorRegionObj(ffi::UnsafeInit tag) : source(tag) {}
   
     Expr source;
     ffi::Array<Range> region;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       // A region's source and symbolic source type form a definition pattern;
       // ranges then refer to those same identities.
       refl::ObjectDef<TensorRegionObj>()
           .def_ro("source", &TensorRegionObj::source, refl::AttachFieldFlag::SEqHashDefPattern())
           .def_ro("region", &TensorRegionObj::region);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorRegion", TensorRegionObj, ExprObj);
   };
   
   class TensorRegion : public Expr {
    public:
     TVM_TIRX_DLL TensorRegion(Expr source, ffi::Array<Range> region, Type ty,
                               Location loc = UnknownLoc());
   
     TVM_TIRX_DLL static TensorRegion FromTensorLoad(TensorLoad value);
   
     explicit TensorRegion(ffi::ObjectPtr<TensorRegionObj> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRegion, Expr, TensorRegionObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorRegionObj);
   };
   
   TVM_TIRX_DLL CallEffectKind SideEffect(const Expr& expr);
   
   namespace ffi {
   
   template <>
   inline constexpr bool use_default_type_traits_v<tvm::StringImm> = false;
   
   template <>
   struct TypeTraits<tvm::StringImm>
       : public ObjectRefWithFallbackTraitsBase<tvm::StringImm, ffi::String> {
     TVM_FFI_INLINE static tvm::StringImm ConvertFallbackValue(ffi::String value) {
       return tvm::StringImm(value);
     }
   };
   
   template <>
   inline constexpr bool use_default_type_traits_v<IntImm> = false;
   
   template <>
   struct TypeTraits<IntImm> : public ObjectRefWithFallbackTraitsBase<IntImm, int64_t> {
     TVM_FFI_INLINE static IntImm ConvertFallbackValue(int64_t value) {
       auto value_ty =
           (value > std::numeric_limits<int>::max() || value < std::numeric_limits<int>::min())
               ? PrimType::Int(64)
               : PrimType::Int(32);
       return IntImm(value_ty, value);
     }
   };
   
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, IntImmObj> = true;
   
   template <>
   inline constexpr bool use_default_type_traits_v<FloatImm> = false;
   
   template <>
   struct TypeTraits<FloatImm> : public ObjectRefWithFallbackTraitsBase<FloatImm, double> {
     TVM_FFI_INLINE static FloatImm ConvertFallbackValue(double value) {
       return FloatImm(PrimType::Float(32), value);
     }
   };
   
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, FloatImmObj> = true;
   
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, TensorLoadObj> = true;
   
   template <typename ExpectedType>
   inline constexpr bool use_default_type_traits_v<TypedExpr<ExpectedType>> = false;
   
   template <typename ExpectedType>
   struct TypeTraits<TypedExpr<ExpectedType>>
       : public ObjectRefTypeTraitsBase<TypedExpr<ExpectedType>> {
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
           !detail::IsObjectInstance<ExprObj>(src->type_index)) {
         return false;
       }
       // Non-owning: this only reads `ty`, and the owning form's incref/decref pair costs two
       // atomics per check on a path every typed field assignment takes.
       const auto* expr = detail::ObjectUnsafe::RawObjectPtrFromUnowned<ExprObj>(src->v_obj);
       return detail::AnyUnsafe::CheckAnyViewStrict<ExpectedType>(expr->ty);
     }
   
     TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
       if (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin && src->v_obj != nullptr &&
           detail::IsObjectInstance<ExprObj>(src->type_index)) {
         const auto* expr = detail::ObjectUnsafe::RawObjectPtrFromUnowned<ExprObj>(src->v_obj);
         if (expr->ty.defined()) {
           const auto* prim_ty = expr->ty.as<PrimTypeObj>();
           std::string ty =
               prim_ty ? std::string(ffi::DLDataTypeToString(prim_ty->dtype)) : expr->ty.GetTypeKey();
           return TypeIndexToTypeKey(src->type_index) + "[ty=" + ty + "]";
         }
       }
       return TypeTraitsBase::GetMismatchTypeInfo(src);
     }
   };
   
   // Primitive expression arguments also accept bool, integer, and floating-point literals.
   template <>
   inline constexpr bool use_default_type_traits_v<PrimExpr> = false;
   
   template <>
   struct TypeTraits<PrimExpr>
       : public ObjectRefWithFallbackTraitsBase<PrimExpr, StrictBool, int64_t, double> {
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return TypeTraits<TypedExpr<PrimType>>::CheckAnyStrict(src);
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return "ir.PrimExpr"; }
   
     TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
       return TypeTraits<TypedExpr<PrimType>>::GetMismatchTypeInfo(src);
     }
   
     TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(StrictBool value) {
       return IntImm::Bool(value);
     }
     TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(int64_t value) {
       return TypeTraits<IntImm>::ConvertFallbackValue(value);
     }
     TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(double value) {
       return TypeTraits<FloatImm>::ConvertFallbackValue(value);
     }
   };
   
   // Generic expression arguments accept primitive/string literals and arrays of expressions.
   template <>
   inline constexpr bool use_default_type_traits_v<Expr> = false;
   
   template <>
   struct TypeTraits<Expr> : public ObjectRefWithFallbackTraitsBase<Expr, PrimExpr, ffi::String> {
     TVM_FFI_INLINE static Expr ConvertFallbackValue(PrimExpr value) { return value; }
     TVM_FFI_INLINE static Expr ConvertFallbackValue(ffi::String value) {
       return StringImm(std::move(value));
     }
     static std::optional<Expr> TryCastFromAnyView(const TVMFFIAny* src) {
       if (auto value =
               ObjectRefWithFallbackTraitsBase<Expr, PrimExpr, ffi::String>::TryCastFromAnyView(src)) {
         return value;
       }
       if (auto fields = TypeTraits<ffi::Array<Expr>>::TryCastFromAnyView(src)) {
         return tvm::Tuple(std::move(*fields));
       }
       return std::nullopt;
     }
   };
   
   template <>
   inline constexpr bool use_default_type_traits_v<IntExpr> = false;
   
   template <>
   struct TypeTraits<IntExpr> : public ObjectRefWithFallbackTraitsBase<IntExpr, int64_t> {
     using Base = ObjectRefWithFallbackTraitsBase<IntExpr, int64_t>;
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if (!Base::CheckAnyStrict(src)) return false;
       const auto* expr = detail::ObjectUnsafe::RawObjectPtrFromUnowned<ExprObj>(src->v_obj);
       const auto* ty = expr->ty.as<PrimTypeObj>();
       return ty != nullptr && ty->dtype.lanes == 1 &&
              (ty->dtype.code == kDLInt || ty->dtype.code == kDLUInt);
     }
   
     TVM_FFI_INLINE static IntExpr ConvertFallbackValue(int64_t value) {
       return TypeTraits<PrimExpr>::ConvertFallbackValue(value).as_or_throw<IntExpr>();
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return "ir.IntExpr"; }
   
     TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
       return TypeTraits<TypedExpr<PrimType>>::GetMismatchTypeInfo(src);
     }
   };
   
   template <>
   inline constexpr bool use_default_type_traits_v<PrimVar> = false;
   
   template <>
   struct TypeTraits<PrimVar> : public ObjectRefTypeTraitsBase<PrimVar> {
     using Base = ObjectRefTypeTraitsBase<PrimVar>;
     using Base::CopyFromAnyViewAfterCheck;
     using Base::CopyToAnyView;
     using Base::GetMismatchTypeInfo;
     using Base::MoveFromAnyAfterCheck;
     using Base::MoveToAny;
     using Base::TypeSchema;
     using Base::TypeStr;
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if (src->type_index != VarObj::RuntimeTypeIndex()) {
         return false;
       }
       const auto* var = static_cast<const VarObj*>(
           detail::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
       return detail::AnyUnsafe::CheckAnyStrict<PrimType>(var->ExprObj::ty);
     }
   
     TVM_FFI_INLINE static std::optional<PrimVar> TryCastFromAnyView(const TVMFFIAny* src) {
       if (CheckAnyStrict(src)) {
         return detail::ObjectUnsafe::ObjectRefFromObjectPtr<PrimVar>(
             detail::ObjectUnsafe::ObjectPtrFromUnowned<VarObj>(src->v_obj));
       }
       return std::nullopt;
     }
   };
   
   template <>
   inline constexpr bool use_default_type_traits_v<TensorRegion> = false;
   
   template <>
   struct TypeTraits<TensorRegion> : public ObjectRefWithFallbackTraitsBase<TensorRegion, TensorLoad> {
     TVM_FFI_INLINE static TensorRegion ConvertFallbackValue(TensorLoad value) {
       return TensorRegion::FromTensorLoad(std::move(value));
     }
   };
   
   }  // namespace ffi
   
   using SubscriptSlice = ffi::Array<ffi::Variant<
       ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>,
       PrimExpr>>;
   
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<
       ffi::TypedFunction<ffi::ObjectRef(Expr, SubscriptSlice, Location)>>
       kExprSubscriptRealizeByTy{"__tvm_expr_subscript_realize__"};
   }  // namespace type_attr
   
   }  // namespace tvm
   
   /* \brief Allow tvm.Var and tvm.GlobalVar as keys in STL tables
    *
    * For most IR expressions, it would be ambiguous whether the
    * expression should follow reference equality or structural equality.
    * This is not the case for variables, which do not contain nested
    * internal structure, and are frequently used as keys in lookup
    * tables.
    *
    * Providing `std::hash` and `std::equal_to` specializations for
    * `tvm::Var` and `tvm::GlobalVar` allows them to be used as keys in STL tables.  For
    * other IR expressions, the user must specify the type of equality
    * used (e.g. `std::unordered_set<T, StructuralHash, StructuralEqual>`
    * or `std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>`).
    */
   template <>
   struct std::hash<tvm::Var> {
     std::size_t operator()(const tvm::Var& var) const { return tvm::ffi::ObjectPtrHash()(var); }
   };
   
   template <>
   struct std::equal_to<tvm::Var> {
     bool operator()(const tvm::Var& var_a, const tvm::Var& var_b) const {
       return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
     }
   };
   
   template <>
   struct std::hash<tvm::GlobalVar> {
     std::size_t operator()(const tvm::GlobalVar& var) const { return tvm::ffi::ObjectPtrHash()(var); }
   };
   
   template <>
   struct std::equal_to<tvm::GlobalVar> {
     bool operator()(const tvm::GlobalVar& var_a, const tvm::GlobalVar& var_b) const {
       return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
     }
   };
   #endif  // TVM_IR_EXPR_H_
