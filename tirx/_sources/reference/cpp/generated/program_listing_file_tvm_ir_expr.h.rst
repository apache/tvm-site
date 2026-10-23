
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
   
   class ExprNode : public ffi::Object {
    public:
     mutable Location loc = UnknownLoc();
   
     mutable Type ty = MissingType();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       // loc does not participate in structural equal and hash.
       refl::ObjectDef<ExprNode>()
           .def_ro("loc", &ExprNode::loc, refl::DefaultValue(UnknownLoc()),
                   refl::AttachFieldFlag::SEqHashIgnore())
           .def_ro("ty", &ExprNode::ty, refl::DefaultValue(MissingType()));
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
   
     static constexpr const uint32_t _type_child_slots = 64;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Expr", ExprNode, ffi::Object);
   };
   
   class Expr : public ffi::ObjectRef {
    public:
     // Expressions do not implicitly compare by object identity or address.  Callers must name
     // whether they intend object identity, structural equality, or primitive symbolic comparison.
     bool operator==(const Expr& other) const = delete;
     bool operator!=(const Expr& other) const = delete;
     bool operator<(const Expr& other) const = delete;
   
     explicit Expr(ffi::ObjectPtr<ExprNode> node) : ffi::ObjectRef(std::move(node)) {
       TVM_FFI_CHECK(defined(), ValueError) << "Expr expects a defined node";
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Expr, ffi::ObjectRef, ExprNode);
   };
   
   class StagingExprNode : public ExprNode {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<StagingExprNode>(); }
     static constexpr const uint32_t _type_child_slots = 4;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.StagingExpr", StagingExprNode, ExprNode);
   };
   
   class StagingExpr : public Expr {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StagingExpr, Expr, StagingExprNode);
   };
   
   class OpaqueExprNode : public ExprNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<OpaqueExprNode>();
     }
   
     static constexpr const uint32_t _type_child_slots = 2;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.OpaqueExpr", OpaqueExprNode, ExprNode);
   };
   
   class OpaqueExpr : public Expr {
    public:
     explicit OpaqueExpr(ffi::ObjectPtr<OpaqueExprNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OpaqueExpr, Expr, OpaqueExprNode);
   };
   
   template <typename ExpectedType>
   class TypedExpr : public Expr {
    public:
     ExpectedType ty() const {
       const auto* node = get();
       TVM_FFI_DCHECK(node != nullptr);
       const auto* ty_node = node->ExprNode::ty.template as<typename ExpectedType::ContainerType>();
       TVM_FFI_DCHECK(ty_node != nullptr);
       return ffi::GetRef<ExpectedType>(ty_node);
     }
   
     explicit TypedExpr(ffi::ObjectPtr<ExprNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TypedExpr, Expr, ExprNode);
     static constexpr bool _type_container_is_exact = false;
   };
   
   class PrimExpr : public TypedExpr<PrimType> {
    public:
     using TypedExpr<PrimType>::ty;
   
     TVM_TIRX_DLL PrimExpr(int32_t value);  // NOLINT(*)
     TVM_TIRX_DLL PrimExpr(float value);  // NOLINT(*)
   
     explicit PrimExpr(ffi::ObjectPtr<ExprNode> node) : TypedExpr<PrimType>(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimExpr, TypedExpr<PrimType>, ExprNode);
     static constexpr bool _type_container_is_exact = false;
   };
   
   class IntExpr : public PrimExpr {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntExpr, PrimExpr, ExprNode);
     static constexpr bool _type_container_is_exact = false;
   };
   
   namespace ffi::details {
   // Array members require storage availability before the expression traits below are complete.
   template <>
   inline constexpr bool storage_enabled_v<Expr> = true;
   template <>
   inline constexpr bool storage_enabled_v<PrimExpr> = true;
   }  // namespace ffi::details
   
   class ConstantNode : public ExprNode {
    public:
     static constexpr uint32_t _type_child_slots = 5;
     static void RegisterReflection() { ffi::reflection::ObjectDef<ConstantNode>(); }
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Constant", ConstantNode, ExprNode);
   };
   
   class Constant : public Expr {
    public:
     explicit Constant(ffi::ObjectPtr<ConstantNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Constant, Expr, ConstantNode);
   };
   
   class GenericConstNode : public ConstantNode {
    public:
     ffi::Any value;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<GenericConstNode>().def_ro("value", &GenericConstNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GenericConst", GenericConstNode, ConstantNode);
   };
   
   class GenericConst : public Constant {
    public:
     TVM_TIRX_DLL GenericConst(Type ty, ffi::Any value, Location loc = UnknownLoc());
   
     explicit GenericConst(ffi::ObjectPtr<GenericConstNode> node) : Constant(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GenericConst, Constant, GenericConstNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GenericConstNode);
   };
   
   class StringImmNode : public ConstantNode {
    public:
     ffi::String value;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<StringImmNode>().def_ro("value", &StringImmNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.StringImm", StringImmNode, ConstantNode);
   };
   
   class StringImm : public Constant {
    public:
     TVM_TIRX_DLL explicit StringImm(ffi::String value, Location loc = UnknownLoc());
   
     explicit StringImm(ffi::ObjectPtr<StringImmNode> node) : Constant(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StringImm, Constant, StringImmNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmNode);
   };
   
   class DataTypeImmNode : public ConstantNode {
    public:
     DLDataType value;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<DataTypeImmNode>().def_ro("value", &DataTypeImmNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.DataTypeImm", DataTypeImmNode, ConstantNode);
   };
   
   class DataTypeImm : public Constant {
    public:
     TVM_TIRX_DLL explicit DataTypeImm(DLDataType value, Location loc = UnknownLoc());
   
     explicit DataTypeImm(ffi::ObjectPtr<DataTypeImmNode> node) : Constant(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataTypeImm, Constant, DataTypeImmNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(DataTypeImmNode);
   };
   
   class IntImmNode : public ConstantNode {
    public:
     ffi::BigInt value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IntImmNode>().def_ro("value", &IntImmNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IntImm", IntImmNode, ConstantNode);
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
   
     explicit IntImm(ffi::ObjectPtr<IntImmNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntImm, PrimExpr, IntImmNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IntImmNode);
   };
   
   class FloatImmNode : public ConstantNode {
    public:
     double value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FloatImmNode>().def_ro("value", &FloatImmNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FloatImm", FloatImmNode, ConstantNode);
   };
   
   class FloatImm : public PrimExpr {
    public:
     TVM_TIRX_DLL FloatImm(PrimType value_ty, double value, Location loc = UnknownLoc());
   
     explicit FloatImm(ffi::ObjectPtr<FloatImmNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloatImm, PrimExpr, FloatImmNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(FloatImmNode);
   };
   
   class TupleNode : public ExprNode {
    public:
     ffi::Array<Expr> fields;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleNode>().def_ro("fields", &TupleNode::fields);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Tuple", TupleNode, ExprNode);
   };
   
   class Tuple : public Expr {
    public:
     TVM_TIRX_DLL explicit Tuple(ffi::Array<Expr> fields, Location loc = UnknownLoc());
   
     explicit Tuple(ffi::ObjectPtr<TupleNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Tuple, Expr, TupleNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleNode);
   };
   
   class TupleGetItemNode : public ExprNode {
    public:
     explicit TupleGetItemNode(Expr tuple) : tuple(std::move(tuple)) {}
     explicit TupleGetItemNode(ffi::UnsafeInit tag) : tuple(tag) {}
   
     Expr tuple;
     int index;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleGetItemNode>()
           .def_ro("tuple_value", &TupleGetItemNode::tuple)
           .def_ro("index", &TupleGetItemNode::index);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleGetItem", TupleGetItemNode, ExprNode);
   };
   
   class TupleGetItem : public Expr {
    public:
     TVM_TIRX_DLL TupleGetItem(Expr tuple, int index, Location loc = UnknownLoc());
   
     explicit TupleGetItem(ffi::ObjectPtr<TupleGetItemNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleGetItem, Expr, TupleGetItemNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemNode);
   };
   
   class TensorLoadNode : public ExprNode {
    public:
     explicit TensorLoadNode(Expr source) : source(std::move(source)) {}
     explicit TensorLoadNode(ffi::UnsafeInit tag) : source(tag) {}
   
     Expr source;
     ffi::Array<PrimExpr> indices;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorLoadNode>()
           .def_ro("source", &TensorLoadNode::source)
           .def_ro("indices", &TensorLoadNode::indices);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorLoad", TensorLoadNode, ExprNode);
   };
   
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<
       ffi::reflection::NativeFunctionView<PrimType(const TensorLoadNode*)>>
       kTensorLoadInferType{"__tvm_tensor_load_infer_type__"};
   }  // namespace type_attr
   
   class TensorLoad : public PrimExpr {
    public:
     TVM_TIRX_DLL TensorLoad(Expr source, ffi::Array<PrimExpr> indices, Location loc = UnknownLoc());
   
     TVM_TIRX_DLL static TensorLoad Unchecked(PrimType ty, Expr source, ffi::Array<PrimExpr> indices,
                                              Location loc = UnknownLoc());
   
     TVM_TIRX_DLL explicit TensorLoad(ffi::ObjectPtr<TensorLoadNode> node);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorLoad, PrimExpr, TensorLoadNode);
     static constexpr bool _type_container_is_exact = true;
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorLoadNode);
   };
   
   class VarNode : public ExprNode {
    public:
     ffi::String name;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<VarNode>().def_ro("name", &VarNode::name,
                                         refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
     static constexpr const uint32_t _type_child_slots = 1;
     // VarNode reserves its sole child slot for the final relax::DataflowVarNode subtype, so its
     // descendant type-index range cannot overflow.
     static constexpr bool _type_child_slots_can_overflow = false;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Var", VarNode, ExprNode);
   };
   
   class Var : public Expr {
    public:
     TVM_TIRX_DLL explicit Var(ffi::String name, Type ty, Location loc = UnknownLoc());
   
     TVM_TIRX_DLL Var CopyWithName(const ffi::String& name) const;
   
     TVM_TIRX_DLL Var CopyWithSuffix(const ffi::String& suffix) const;
   
     TVM_TIRX_DLL Var CopyWithDType(PrimType dtype) const;
   
     TVM_TIRX_DLL Var CopyWithType(Type type) const;
   
     explicit Var(ffi::ObjectPtr<VarNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Var, Expr, VarNode);
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
   
     explicit PrimVar(ffi::ObjectPtr<VarNode> node) : PrimExpr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimVar, PrimExpr, VarNode);
     static constexpr bool _type_container_is_exact = false;
   };
   
   class LambdaExprNode : public StagingExprNode {
    public:
     ffi::Array<Var> vars;
     Expr body;
   
     explicit LambdaExprNode(ffi::UnsafeInit tag) : body(tag) {}
     explicit LambdaExprNode(Expr body) : body(std::move(body)) {}
     TVM_TIRX_DLL Expr Apply(const ffi::Array<Expr>& arguments) const;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LambdaExprNode>()
           .def_ro("vars", &LambdaExprNode::vars, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("body", &LambdaExprNode::body);
     }
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.LambdaExpr", LambdaExprNode, StagingExprNode);
   };
   
   class LambdaExpr : public StagingExpr {
    public:
     TVM_TIRX_DLL explicit LambdaExpr(ffi::Array<Var> vars, Expr body, Location loc = UnknownLoc());
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LambdaExpr, StagingExpr, LambdaExprNode);
   };
   
   class GlobalVar;
   class GlobalVarNode : public ExprNode {
    public:
     ffi::String name_hint;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<GlobalVarNode>().def_ro("name_hint", &GlobalVarNode::name_hint);
       // A GlobalVar identifies a module-level symbol.  Its type is derived from the
       // corresponding function definition and is not part of the symbol identity.
       refl::TypeAttrDef<GlobalVarNode>()
           .def(tvm::ffi::type_attr::kSEqual.name, &GlobalVarNode::SEqual)
           .def(tvm::ffi::type_attr::kSHash.name, &GlobalVarNode::SHash);
     }
   
     bool SEqual(const GlobalVarNode* other,
                 ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
       return equal(name_hint, other->name_hint, false, "name_hint");
     }
   
     int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
       return hash(name_hint, init_hash, false);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GlobalVar", GlobalVarNode, ExprNode);
   };
   
   class GlobalVar : public Expr {
    public:
     TVM_TIRX_DLL explicit GlobalVar(ffi::String name_hint, Location loc = UnknownLoc());
   
     explicit GlobalVar(ffi::ObjectPtr<GlobalVarNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GlobalVar, Expr, GlobalVarNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalVarNode);
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
   
   class CallNode : public ExprNode {
    public:
     explicit CallNode(Expr op) : op(std::move(op)) {}
     explicit CallNode(ffi::UnsafeInit tag) : op(tag) {}
   
     Expr op;
   
     ffi::Array<Expr> args;
   
     ffi::Optional<Attrs> attrs;
   
     ffi::Array<Type> ty_args;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CallNode>()
           .def_ro("op", &CallNode::op)
           .def_ro("args", &CallNode::args)
           .def_ro("attrs", &CallNode::attrs)
           .def_ro("ty_args", &CallNode::ty_args);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Call", CallNode, ExprNode);
   };
   
   class Call : public Expr {
    public:
     TVM_TIRX_DLL Call(ffi::Optional<Type> ret_ty, Expr op, ffi::Array<Expr> args,
                       ffi::Optional<Attrs> attrs = std::nullopt,
                       ffi::Array<Type> ty_args = ffi::Array<Type>(), Location loc = UnknownLoc());
   
     TVM_TIRX_DLL void Validate() const;
     TVM_TIRX_DLL static Type ReinferType(const CallNode* call);
   
     explicit Call(ffi::ObjectPtr<CallNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Call, Expr, CallNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
   };
   
   namespace op_attr {
   inline constexpr ffi::reflection::AttrKey<CallEffectKind> kCallEffectKind{"TCallEffectKind"};
   }  // namespace op_attr
   
   class RangeNode : public ffi::Object {
    public:
     PrimExpr begin;
     PrimExpr extent;
     mutable Location loc = UnknownLoc();
     explicit RangeNode(ffi::UnsafeInit tag) : begin(tag), extent(tag) {}
     RangeNode(PrimExpr begin, PrimExpr extent, Location loc = UnknownLoc())
         : begin(begin), extent(extent), loc(loc) {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<RangeNode>()
           .def_ro("begin", &RangeNode::begin)
           .def_ro("extent", &RangeNode::extent)
           .def_ro("loc", &RangeNode::loc, refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Range", RangeNode, ffi::Object);
   };
   
   class Range : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL Range(PrimExpr begin, PrimExpr end, Location loc = UnknownLoc());
     TVM_TIRX_DLL static Range FromMinExtent(PrimExpr min, PrimExpr extent,
                                             Location loc = UnknownLoc());
     TVM_TIRX_DLL static PrimExpr GetExtentFromBeginEnd(const PrimExpr& begin, const PrimExpr& end);
     // declare range.
     explicit Range(ffi::UnsafeInit tag, ffi::ObjectPtr<RangeNode> n) : ffi::ObjectRef(tag) {
       data_ = std::move(n);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Range, ffi::ObjectRef, RangeNode);
   };
   
   class TensorRegionNode : public ExprNode {
    public:
     explicit TensorRegionNode(Expr source) : source(std::move(source)) {}
     explicit TensorRegionNode(ffi::UnsafeInit tag) : source(tag) {}
   
     Expr source;
     ffi::Array<Range> region;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       // A region's source and symbolic source type form a definition pattern;
       // ranges then refer to those same identities.
       refl::ObjectDef<TensorRegionNode>()
           .def_ro("source", &TensorRegionNode::source, refl::AttachFieldFlag::SEqHashDefPattern())
           .def_ro("region", &TensorRegionNode::region);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorRegion", TensorRegionNode, ExprNode);
   };
   
   class TensorRegion : public Expr {
    public:
     TVM_TIRX_DLL TensorRegion(Expr source, ffi::Array<Range> region, Type ty,
                               Location loc = UnknownLoc());
   
     TVM_TIRX_DLL static TensorRegion FromTensorLoad(TensorLoad value);
   
     explicit TensorRegion(ffi::ObjectPtr<TensorRegionNode> node) : Expr(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRegion, Expr, TensorRegionNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorRegionNode);
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
   inline constexpr bool object_ref_contains_v<PrimExpr, IntImmNode> = true;
   
   template <>
   inline constexpr bool use_default_type_traits_v<FloatImm> = false;
   
   template <>
   struct TypeTraits<FloatImm> : public ObjectRefWithFallbackTraitsBase<FloatImm, double> {
     TVM_FFI_INLINE static FloatImm ConvertFallbackValue(double value) {
       return FloatImm(PrimType::Float(32), value);
     }
   };
   
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, FloatImmNode> = true;
   
   template <>
   inline constexpr bool object_ref_contains_v<PrimExpr, TensorLoadNode> = true;
   
   template <typename ExpectedType>
   inline constexpr bool use_default_type_traits_v<TypedExpr<ExpectedType>> = false;
   
   template <typename ExpectedType>
   struct TypeTraits<TypedExpr<ExpectedType>>
       : public ObjectRefTypeTraitsBase<TypedExpr<ExpectedType>> {
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
           !details::IsObjectInstance<ExprNode>(src->type_index)) {
         return false;
       }
       // Non-owning: this only reads `ty`, and the owning form's incref/decref pair costs two
       // atomics per check on a path every typed field assignment takes.
       const auto* expr = details::ObjectUnsafe::RawObjectPtrFromUnowned<ExprNode>(src->v_obj);
       return details::AnyUnsafe::CheckAnyViewStrict<ExpectedType>(expr->ty);
     }
   
     TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
       if (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin && src->v_obj != nullptr &&
           details::IsObjectInstance<ExprNode>(src->type_index)) {
         const auto* expr = details::ObjectUnsafe::RawObjectPtrFromUnowned<ExprNode>(src->v_obj);
         if (expr->ty.defined()) {
           const auto* prim_ty = expr->ty.as<PrimTypeNode>();
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
       const auto* expr = details::ObjectUnsafe::RawObjectPtrFromUnowned<ExprNode>(src->v_obj);
       const auto* ty = expr->ty.as<PrimTypeNode>();
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
       if (src->type_index != VarNode::RuntimeTypeIndex()) {
         return false;
       }
       const auto* var = static_cast<const VarNode*>(
           details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
       return details::AnyUnsafe::CheckAnyStrict<PrimType>(var->ExprNode::ty);
     }
   
     TVM_FFI_INLINE static std::optional<PrimVar> TryCastFromAnyView(const TVMFFIAny* src) {
       if (CheckAnyStrict(src)) {
         return details::ObjectUnsafe::ObjectRefFromObjectPtr<PrimVar>(
             details::ObjectUnsafe::ObjectPtrFromUnowned<VarNode>(src->v_obj));
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
