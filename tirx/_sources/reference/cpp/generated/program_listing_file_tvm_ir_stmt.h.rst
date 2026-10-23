
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
   
   class StmtNode : public ffi::Object {
    public:
     mutable Location loc = UnknownLoc();
   
     StmtNode() = default;
     explicit StmtNode(Location loc) : loc(loc) {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<StmtNode>().def_ro("loc", &StmtNode::loc,
                                          refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;
   
     static constexpr const uint32_t _type_child_slots = 15;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Stmt", StmtNode, ffi::Object);
   };
   
   class Stmt : public ffi::ObjectRef {
    public:
     explicit Stmt(ffi::ObjectPtr<StmtNode> node) : ffi::ObjectRef(std::move(node)) {
       TVM_FFI_CHECK(defined(), ValueError) << "Stmt expects a defined node";
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Stmt, ffi::ObjectRef, StmtNode);
   };
   
   class SeqStmtNode : public StmtNode {
    public:
     ffi::Array<Stmt> seq;
   
     size_t size() const { return seq.size(); }
     Stmt operator[](size_t index) const { return seq[index]; }
   
     TVM_TIRX_DLL static void RegisterReflection();
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SeqStmt", SeqStmtNode, StmtNode);
   };
   
   class SeqStmt : public Stmt {
    public:
     TVM_TIRX_DLL explicit SeqStmt(ffi::Array<Stmt> seq, Location loc = UnknownLoc());
     TVM_TIRX_DLL SeqStmt(Stmt stmt, ffi::Optional<Location> loc = std::nullopt);
     SeqStmt(std::initializer_list<Stmt> seq, Location loc = UnknownLoc())
         : SeqStmt(ffi::Array<Stmt>(seq), std::move(loc)) {}
   
     size_t size() const { return operator->()->size(); }
     Stmt operator[](size_t index) const { return (*(operator->()))[index]; }
     explicit SeqStmt(ffi::ObjectPtr<SeqStmtNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SeqStmt, Stmt, SeqStmtNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqStmtNode);
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
   
   class BindNode : public StmtNode {
    public:
     explicit BindNode(ffi::UnsafeInit tag) : var(tag), value(tag) {}
   
     BindNode(Var var, Expr value) : var(std::move(var)), value(std::move(value)) {}
   
     Var var;
     Expr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BindNode>()
           .def_ro("var", &BindNode::var, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("value", &BindNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Bind", BindNode, StmtNode);
   };
   
   class Bind : public Stmt {
    public:
     TVM_TIRX_DLL Bind(Var var, Expr value, Location loc = UnknownLoc());
   
     explicit Bind(ffi::ObjectPtr<BindNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Bind, Stmt, BindNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BindNode);
   };
   
   TVM_TIRX_DLL bool IsRegionOp(const Op& op);
   
   TVM_TIRX_DLL ffi::Array<Var> GetRegionBodyParams(Op op, ffi::Array<Expr> args, DictAttrs attrs);
   
   class RegionStmtNode : public StmtNode {
    public:
     Op op;
     ffi::Array<Expr> args;
     ffi::Array<Var> body_params;
     DictAttrs attrs;
     SeqStmt body;
     ffi::Array<Var> result_vars;
   
     explicit RegionStmtNode(ffi::UnsafeInit tag) : op(tag), body(tag) {}
   
     RegionStmtNode(Op op, SeqStmt body) : op(std::move(op)), body(std::move(body)) {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<RegionStmtNode>()
           .def_ro("op", &RegionStmtNode::op)
           .def_ro("attrs", &RegionStmtNode::attrs)
           .def_ro("args", &RegionStmtNode::args)
           .def_ro("body_params", &RegionStmtNode::body_params,
                   refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("body", &RegionStmtNode::body)
           .def_ro("result_vars", &RegionStmtNode::result_vars,
                   refl::AttachFieldFlag::SEqHashDefSimple());
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.RegionStmt", RegionStmtNode, StmtNode);
   };
   
   class RegionStmt : public Stmt {
    public:
     TVM_TIRX_DLL RegionStmt(Op op, ffi::Array<Expr> args, ffi::Array<Var> body_params,
                             DictAttrs attrs, SeqStmt body, ffi::Array<Var> result_vars = {},
                             Location loc = UnknownLoc());
   
     explicit RegionStmt(ffi::ObjectPtr<RegionStmtNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionStmt, Stmt, RegionStmtNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(RegionStmtNode);
   };
   
   namespace op_attr {
   inline constexpr ffi::reflection::AttrKey<
       ffi::reflection::NativeFunctionView<ffi::Array<Var>(const CallNode* call)>>
       kRegionGetBodyParams{"FRegionGetBodyParams"};
   inline constexpr ffi::reflection::AttrKey<
       ffi::reflection::NativeFunctionView<void(const RegionStmtNode*)>>
       kRegionValidate{"FRegionValidate"};
   }  // namespace op_attr
   
   // Compatibility names; new callers can deduce the value type from the key.
   using FRegionGetBodyParams = decltype(op_attr::kRegionGetBodyParams)::ValueType;
   using FRegionValidate = decltype(op_attr::kRegionValidate)::ValueType;
   
   class AssertStmtNode : public StmtNode {
    public:
     explicit AssertStmtNode(ffi::UnsafeInit tag) : condition(tag), error_kind(tag) {}
   
     AssertStmtNode(PrimExpr condition, StringImm error_kind)
         : condition(std::move(condition)), error_kind(std::move(error_kind)) {}
   
     PrimExpr condition;
     StringImm error_kind;
     ffi::Array<StringImm> message_parts;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AssertStmtNode>()
           .def_ro("condition", &AssertStmtNode::condition)
           .def_ro("error_kind", &AssertStmtNode::error_kind)
           .def_ro("message_parts", &AssertStmtNode::message_parts);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.AssertStmt", AssertStmtNode, StmtNode);
   };
   
   class AssertStmt : public Stmt {
    public:
     TVM_TIRX_DLL AssertStmt(PrimExpr condition, StringImm error_kind,
                             ffi::Array<StringImm> message_parts, Location loc = UnknownLoc());
   
     explicit AssertStmt(ffi::ObjectPtr<AssertStmtNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertStmt, Stmt, AssertStmtNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(AssertStmtNode);
   };
   
   class TensorStoreNode : public StmtNode {
    public:
     explicit TensorStoreNode(ffi::UnsafeInit tag) : dest(tag), value(tag) {}
   
     TensorStoreNode(Expr dest, PrimExpr value) : dest(std::move(dest)), value(std::move(value)) {}
   
     Expr dest;
     ffi::Array<PrimExpr> indices;
     PrimExpr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorStoreNode>()
           .def_ro("dest", &TensorStoreNode::dest)
           .def_ro("indices", &TensorStoreNode::indices)
           .def_ro("value", &TensorStoreNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorStore", TensorStoreNode, StmtNode);
   };
   
   class TensorStore : public Stmt {
    public:
     TVM_TIRX_DLL explicit TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value,
                                       Location loc = UnknownLoc());
   
     explicit TensorStore(ffi::ObjectPtr<TensorStoreNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorStore, Stmt, TensorStoreNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorStoreNode);
   };
   
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<void(
       const Expr&, const ffi::Array<PrimExpr>&, const PrimExpr&)>>
       kTensorStoreValidateByTy{"__tvm_tensor_store_validate__"};
   }  // namespace type_attr
   
   class EvaluateNode : public StmtNode {
    public:
     explicit EvaluateNode(ffi::UnsafeInit tag) : value(tag) {}
   
     explicit EvaluateNode(Expr value) : value(std::move(value)) {}
   
     Expr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<EvaluateNode>().def_ro("value", &EvaluateNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Evaluate", EvaluateNode, StmtNode);
   };
   
   class Evaluate : public Stmt {
    public:
     TVM_TIRX_DLL explicit Evaluate(Expr value, Location loc = UnknownLoc());
   
     explicit Evaluate(int value, Location loc = UnknownLoc()) : Evaluate(PrimExpr(value), loc) {}
   
     explicit Evaluate(ffi::ObjectPtr<EvaluateNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Evaluate, Stmt, EvaluateNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluateNode);
   };
   
   class IfNode : public StmtNode {
    public:
     explicit IfNode(ffi::UnsafeInit tag) : condition(tag), then_case(tag) {}
   
     IfNode(PrimExpr condition, SeqStmt then_case)
         : condition(std::move(condition)), then_case(std::move(then_case)) {}
   
     PrimExpr condition;
     SeqStmt then_case;
     ffi::Optional<SeqStmt> else_case;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IfNode>()
           .def_ro("condition", &IfNode::condition)
           .def_ro("then_case", &IfNode::then_case)
           .def_ro("else_case", &IfNode::else_case);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.If", IfNode, StmtNode);
   };
   
   class If : public Stmt {
    public:
     TVM_TIRX_DLL If(PrimExpr condition, SeqStmt then_case,
                     ffi::Optional<SeqStmt> else_case = std::nullopt, Location loc = UnknownLoc());
   
     explicit If(ffi::ObjectPtr<IfNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(If, Stmt, IfNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(IfNode);
   };
   
   enum class ForKind : int {
     kDefault = 0,
     kParallel = 1,
     kVectorized = 2,
     kUnrolled = 3,
   };
   
   class ForNode : public StmtNode {
    public:
     explicit ForNode(ffi::UnsafeInit tag) : var(tag), begin(tag), end(tag), step(tag), body(tag) {}
   
     ForNode(PrimVar var, PrimExpr begin, PrimExpr end, PrimExpr step, ForKind kind, SeqStmt body)
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
       refl::ObjectDef<ForNode>()
           .def_ro("var", &ForNode::var, refl::AttachFieldFlag::SEqHashDefSimple())
           .def_ro("begin", &ForNode::begin)
           .def_ro("end", &ForNode::end)
           .def_ro("step", &ForNode::step)
           .def_ro("kind", &ForNode::kind)
           .def_ro("body", &ForNode::body)
           .def_ro("annotations", &ForNode::annotations);
     }
   
     TVM_TIRX_DLL bool HasTrivialStep() const;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.For", ForNode, StmtNode);
   };
   
   class For : public Stmt {
    public:
     TVM_TIRX_DLL For(PrimVar var, PrimExpr begin, PrimExpr end, PrimExpr step, ForKind kind,
                      SeqStmt body, ffi::Map<ffi::String, ffi::Any> annotations = {},
                      Location loc = UnknownLoc());
   
     explicit For(ffi::ObjectPtr<ForNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(For, Stmt, ForNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ForNode);
   };
   
   class WhileNode : public StmtNode {
    public:
     explicit WhileNode(ffi::UnsafeInit tag) : condition(tag), body(tag) {}
   
     WhileNode(PrimExpr condition, SeqStmt body)
         : condition(std::move(condition)), body(std::move(body)) {}
   
     PrimExpr condition;
     SeqStmt body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<WhileNode>()
           .def_ro("condition", &WhileNode::condition)
           .def_ro("body", &WhileNode::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.While", WhileNode, StmtNode);
   };
   
   class While : public Stmt {
    public:
     TVM_TIRX_DLL While(PrimExpr condition, SeqStmt body, Location loc = UnknownLoc());
   
     explicit While(ffi::ObjectPtr<WhileNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(While, Stmt, WhileNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(WhileNode);
   };
   
   class ReturnNode : public StmtNode {
    public:
     explicit ReturnNode(ffi::UnsafeInit tag) : value(tag) {}
   
     explicit ReturnNode(Expr value) : value(std::move(value)) {}
   
     Expr value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ReturnNode>().def_ro("value", &ReturnNode::value);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Return", ReturnNode, StmtNode);
   };
   
   class Return : public Stmt {
    public:
     TVM_TIRX_DLL explicit Return(Expr value, Location loc = UnknownLoc());
   
     explicit Return(ffi::ObjectPtr<ReturnNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Return, Stmt, ReturnNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ReturnNode);
   };
   
   class BreakNode : public StmtNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BreakNode>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Break", BreakNode, StmtNode);
   };
   
   class Break : public Stmt {
    public:
     TVM_TIRX_DLL explicit Break(Location loc);
   
     explicit Break(ffi::ObjectPtr<BreakNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Break, Stmt, BreakNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(BreakNode);
   };
   
   class ContinueNode : public StmtNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ContinueNode>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Continue", ContinueNode, StmtNode);
   };
   
   class Continue : public Stmt {
    public:
     TVM_TIRX_DLL explicit Continue(Location loc);
   
     explicit Continue(ffi::ObjectPtr<ContinueNode> node) : Stmt(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Continue, Stmt, ContinueNode);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(ContinueNode);
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
