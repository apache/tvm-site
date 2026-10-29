
.. _program_listing_file_tvm_ir_stmt.h:

Program Listing for File stmt.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_stmt.h>` (``tvm/ir/stmt.h``)

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
   #ifndef TVM_IR_STMT_H_
   #define TVM_IR_STMT_H_
   
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/op.h>
   #include <tvm/ir/prim/expr.h>
   
   #include <initializer_list>
   #include <optional>
   #include <string>
   #include <utility>
   
   namespace tvm {
   
   class StmtObj : public ffi::Object {
    public:
     mutable Location loc = UnknownLoc();
   
     StmtObj() = default;
     explicit StmtObj(Location loc) : loc(loc) {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<StmtObj>().def_ro("loc", &StmtObj::loc, refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;
   
     static constexpr const uint32_t _type_child_slots = 15;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Stmt", StmtObj, ffi::Object);
   };
   
   class Stmt : public ffi::ObjectRef {
    public:
     explicit Stmt(ffi::ObjectPtr<StmtObj> node) : ffi::ObjectRef(std::move(node)) {
       TVM_FFI_CHECK(defined(), ValueError) << "Stmt expects a defined node";
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Stmt, ffi::ObjectRef, StmtObj);
   };
   
   class SeqStmtObj : public StmtObj {
    public:
     ffi::Array<Stmt> seq;
   
     size_t size() const { return seq.size(); }
     Stmt operator[](size_t index) const { return seq[index]; }
   
     TVM_TIRX_DLL static void RegisterReflection();
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SeqStmt", SeqStmtObj, StmtObj);
   };
   
   class SeqStmt : public Stmt {
    public:
     TVM_TIRX_DLL explicit SeqStmt(ffi::Array<Stmt> seq, Location loc = UnknownLoc());
     TVM_TIRX_DLL SeqStmt(Stmt stmt, ffi::Optional<Location> loc = std::nullopt);
     SeqStmt(std::initializer_list<Stmt> seq, Location loc = UnknownLoc())
         : SeqStmt(ffi::Array<Stmt>(seq), std::move(loc)) {}
   
     size_t size() const { return operator->()->size(); }
     Stmt operator[](size_t index) const { return (*(operator->()))[index]; }
     explicit SeqStmt(ffi::ObjectPtr<SeqStmtObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SeqStmt, Stmt, SeqStmtObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqStmtObj);
   };
   
   namespace ffi {
   template <>
   inline constexpr bool use_default_type_traits_v<SeqStmt> = false;
   
   template <>
   struct TypeTraits<SeqStmt> : public ObjectRefWithFallbackTraitsBase<SeqStmt, Stmt, Array<Stmt>> {
     static SeqStmt ConvertFallbackValue(Stmt stmt) { return SeqStmt(std::move(stmt)); }
     static SeqStmt ConvertFallbackValue(Array<Stmt> seq) { return SeqStmt(std::move(seq)); }
   };
   }  // namespace ffi
   
   class BindObj : public StmtObj {
    public:
     explicit BindObj(ffi::UnsafeInit tag) : var(tag), value(tag) {}
   
     BindObj(Var var, Expr value) : var(std::move(var)), value(std::move(value)) {}
   
     Var var;
     Expr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BindObj>()
           .def_ro("var", &BindObj::var, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("value", &BindObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Bind", BindObj, StmtObj);
   };
   
   class Bind : public Stmt {
    public:
     TVM_TIRX_DLL Bind(Var var, Expr value, Location loc = UnknownLoc());
   
     explicit Bind(ffi::ObjectPtr<BindObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Bind, Stmt, BindObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BindObj);
   };
   
   TVM_TIRX_DLL bool IsRegionOp(const Op& op);
   
   TVM_TIRX_DLL ffi::Array<Var> GetRegionBodyParams(Op op, ffi::Array<Expr> args, DictAttrs attrs);
   
   class RegionStmtObj : public StmtObj {
    public:
     Op op;
     ffi::Array<Expr> args;
     ffi::Array<Var> body_params;
     DictAttrs attrs;
     SeqStmt body;
     ffi::Array<Var> result_vars;
   
     explicit RegionStmtObj(ffi::UnsafeInit tag) : op(tag), body(tag) {}
   
     RegionStmtObj(Op op, SeqStmt body) : op(std::move(op)), body(std::move(body)) {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<RegionStmtObj>()
           .def_ro("op", &RegionStmtObj::op)
           .def_ro("attrs", &RegionStmtObj::attrs)
           .def_ro("args", &RegionStmtObj::args)
           .def_ro("body_params", &RegionStmtObj::body_params,
                   refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("body", &RegionStmtObj::body)
           .def_ro("result_vars", &RegionStmtObj::result_vars,
                   refl::AttachFieldFlag::SEqHashDefSimple());
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.RegionStmt", RegionStmtObj, StmtObj);
   };
   
   class RegionStmt : public Stmt {
    public:
     TVM_TIRX_DLL RegionStmt(Op op, ffi::Array<Expr> args, ffi::Array<Var> body_params,
                             DictAttrs attrs, SeqStmt body, ffi::Array<Var> result_vars = {},
                             Location loc = UnknownLoc());
   
     explicit RegionStmt(ffi::ObjectPtr<RegionStmtObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionStmt, Stmt, RegionStmtObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(RegionStmtObj);
   };
   
   namespace op_attr {
   inline constexpr ffi::reflection::AttrKey<
       ffi::reflection::NativeFunctionView<ffi::Array<Var>(const CallObj* call)>>
       kRegionGetBodyParams{"FRegionGetBodyParams"};
   inline constexpr ffi::reflection::AttrKey<
       ffi::reflection::NativeFunctionView<void(const RegionStmtObj*)>>
       kRegionValidate{"FRegionValidate"};
   }  // namespace op_attr
   
   // Compatibility names; new callers can deduce the value type from the key.
   using FRegionGetBodyParams = decltype(op_attr::kRegionGetBodyParams)::ValueType;
   using FRegionValidate = decltype(op_attr::kRegionValidate)::ValueType;
   
   class AssertStmtObj : public StmtObj {
    public:
     explicit AssertStmtObj(ffi::UnsafeInit tag) : condition(tag), error_kind(tag) {}
   
     AssertStmtObj(PrimExpr condition, StringImm error_kind)
         : condition(std::move(condition)), error_kind(std::move(error_kind)) {}
   
     PrimExpr condition;
     StringImm error_kind;
     ffi::Array<StringImm> message_parts;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AssertStmtObj>()
           .def_ro("condition", &AssertStmtObj::condition)
           .def_ro("error_kind", &AssertStmtObj::error_kind)
           .def_ro("message_parts", &AssertStmtObj::message_parts);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.AssertStmt", AssertStmtObj, StmtObj);
   };
   
   class AssertStmt : public Stmt {
    public:
     TVM_TIRX_DLL AssertStmt(PrimExpr condition, StringImm error_kind,
                             ffi::Array<StringImm> message_parts, Location loc = UnknownLoc());
   
     explicit AssertStmt(ffi::ObjectPtr<AssertStmtObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertStmt, Stmt, AssertStmtObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(AssertStmtObj);
   };
   
   class TensorStoreObj : public StmtObj {
    public:
     explicit TensorStoreObj(ffi::UnsafeInit tag) : dest(tag), value(tag) {}
   
     TensorStoreObj(Expr dest, PrimExpr value) : dest(std::move(dest)), value(std::move(value)) {}
   
     Expr dest;
     ffi::Array<PrimExpr> indices;
     PrimExpr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorStoreObj>()
           .def_ro("dest", &TensorStoreObj::dest)
           .def_ro("indices", &TensorStoreObj::indices)
           .def_ro("value", &TensorStoreObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorStore", TensorStoreObj, StmtObj);
   };
   
   class TensorStore : public Stmt {
    public:
     TVM_TIRX_DLL explicit TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value,
                                       Location loc = UnknownLoc());
   
     explicit TensorStore(ffi::ObjectPtr<TensorStoreObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorStore, Stmt, TensorStoreObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorStoreObj);
   };
   
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<void(
       const Expr&, const ffi::Array<PrimExpr>&, const PrimExpr&)>>
       kTensorStoreValidateByTy{"__tvm_tensor_store_validate__"};
   }  // namespace type_attr
   
   class EvaluateObj : public StmtObj {
    public:
     explicit EvaluateObj(ffi::UnsafeInit tag) : value(tag) {}
   
     explicit EvaluateObj(Expr value) : value(std::move(value)) {}
   
     Expr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<EvaluateObj>().def_ro("value", &EvaluateObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Evaluate", EvaluateObj, StmtObj);
   };
   
   class Evaluate : public Stmt {
    public:
     TVM_TIRX_DLL explicit Evaluate(Expr value, Location loc = UnknownLoc());
   
     explicit Evaluate(int value, Location loc = UnknownLoc()) : Evaluate(PrimExpr(value), loc) {}
   
     explicit Evaluate(ffi::ObjectPtr<EvaluateObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Evaluate, Stmt, EvaluateObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluateObj);
   };
   
   class IfObj : public StmtObj {
    public:
     explicit IfObj(ffi::UnsafeInit tag) : condition(tag), then_case(tag) {}
   
     IfObj(PrimExpr condition, SeqStmt then_case)
         : condition(std::move(condition)), then_case(std::move(then_case)) {}
   
     PrimExpr condition;
     SeqStmt then_case;
     ffi::Optional<SeqStmt> else_case;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IfObj>()
           .def_ro("condition", &IfObj::condition)
           .def_ro("then_case", &IfObj::then_case)
           .def_ro("else_case", &IfObj::else_case);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.If", IfObj, StmtObj);
   };
   
   class If : public Stmt {
    public:
     TVM_TIRX_DLL If(PrimExpr condition, SeqStmt then_case,
                     ffi::Optional<SeqStmt> else_case = std::nullopt, Location loc = UnknownLoc());
   
     explicit If(ffi::ObjectPtr<IfObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(If, Stmt, IfObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IfObj);
   };
   
   enum class ForKind : int {
     kDefault = 0,
     kParallel = 1,
     kVectorized = 2,
     kUnrolled = 3,
   };
   
   class ForObj : public StmtObj {
    public:
     explicit ForObj(ffi::UnsafeInit tag) : var(tag), begin(tag), end(tag), step(tag), body(tag) {}
   
     ForObj(PrimVar var, PrimExpr begin, PrimExpr end, PrimExpr step, ForKind kind, SeqStmt body)
         : var(std::move(var)),
           begin(std::move(begin)),
           end(std::move(end)),
           step(std::move(step)),
           kind(kind),
           body(std::move(body)) {}
   
     PrimVar var;
     PrimExpr begin;
     PrimExpr end;
     PrimExpr step;
     ForKind kind;
     SeqStmt body;
     ffi::Map<ffi::String, ffi::Any> annotations;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ForObj>()
           .def_ro("var", &ForObj::var, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("begin", &ForObj::begin)
           .def_ro("end", &ForObj::end)
           // Reflection stores one typed default. Wider unit steps remain distinct.
           .def_ro("step", &ForObj::step, refl::DefaultValue(IntImm::Int32(1)))
           .def_ro("kind", &ForObj::kind, refl::DefaultValue(ForKind::kDefault))
           .def_ro("body", &ForObj::body)
           .def_ro("annotations", &ForObj::annotations,
                   refl::DefaultValue(ffi::Map<ffi::String, ffi::Any>()));
     }
   
     TVM_TIRX_DLL bool HasTrivialStep() const;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.For", ForObj, StmtObj);
   };
   
   class For : public Stmt {
    public:
     TVM_TIRX_DLL For(PrimVar var, PrimExpr begin, PrimExpr end, PrimExpr step, ForKind kind,
                      SeqStmt body, ffi::Map<ffi::String, ffi::Any> annotations = {},
                      Location loc = UnknownLoc());
   
     explicit For(ffi::ObjectPtr<ForObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(For, Stmt, ForObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ForObj);
   };
   
   class WhileObj : public StmtObj {
    public:
     explicit WhileObj(ffi::UnsafeInit tag) : condition(tag), body(tag) {}
   
     WhileObj(PrimExpr condition, SeqStmt body)
         : condition(std::move(condition)), body(std::move(body)) {}
   
     PrimExpr condition;
     SeqStmt body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<WhileObj>()
           .def_ro("condition", &WhileObj::condition)
           .def_ro("body", &WhileObj::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.While", WhileObj, StmtObj);
   };
   
   class While : public Stmt {
    public:
     TVM_TIRX_DLL While(PrimExpr condition, SeqStmt body, Location loc = UnknownLoc());
   
     explicit While(ffi::ObjectPtr<WhileObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(While, Stmt, WhileObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(WhileObj);
   };
   
   class ReturnObj : public StmtObj {
    public:
     explicit ReturnObj(ffi::UnsafeInit tag) : value(tag) {}
   
     explicit ReturnObj(Expr value) : value(std::move(value)) {}
   
     Expr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ReturnObj>().def_ro("value", &ReturnObj::value);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Return", ReturnObj, StmtObj);
   };
   
   class Return : public Stmt {
    public:
     TVM_TIRX_DLL explicit Return(Expr value, Location loc = UnknownLoc());
   
     explicit Return(ffi::ObjectPtr<ReturnObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Return, Stmt, ReturnObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ReturnObj);
   };
   
   class BreakObj : public StmtObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BreakObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Break", BreakObj, StmtObj);
   };
   
   class Break : public Stmt {
    public:
     TVM_TIRX_DLL explicit Break(Location loc);
   
     explicit Break(ffi::ObjectPtr<BreakObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Break, Stmt, BreakObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BreakObj);
   };
   
   class ContinueObj : public StmtObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ContinueObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Continue", ContinueObj, StmtObj);
   };
   
   class Continue : public Stmt {
    public:
     TVM_TIRX_DLL explicit Continue(Location loc);
   
     explicit Continue(ffi::ObjectPtr<ContinueObj> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Continue, Stmt, ContinueObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ContinueObj);
   };
   
   // overload printing of for type.
   TVM_TIRX_DLL std::ostream& operator<<(std::ostream& os, ForKind kind);
   
   // inline implementations
   inline const char* ForKind2String(ForKind t) {
     switch (t) {
       case ForKind::kDefault:
         return "serial";
       case ForKind::kParallel:
         return "parallel";
       case ForKind::kVectorized:
         return "vectorized";
       case ForKind::kUnrolled:
         return "unroll";
     }
     TVM_FFI_THROW(InternalError) << "Unknown ForKind" << t;
     TVM_FFI_UNREACHABLE();
   }
   
   }  // namespace tvm
   #endif  // TVM_IR_STMT_H_
