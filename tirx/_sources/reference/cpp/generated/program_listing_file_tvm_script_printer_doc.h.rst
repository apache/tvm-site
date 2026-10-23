
.. _program_listing_file_tvm_script_printer_doc.h:

Program Listing for File doc.h
==============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_printer_doc.h>` (``tvm/script/printer/doc.h``)

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
   #ifndef TVM_SCRIPT_PRINTER_DOC_H_
   #define TVM_SCRIPT_PRINTER_DOC_H_
   
   #include <tvm/ffi/dtype.h>
   #include <tvm/ffi/reflection/access_path.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/type.h>
   #include <tvm/runtime/device_api.h>
   
   #include <string>
   
   namespace tvm {
   namespace script {
   namespace printer {
   
   using AccessPath = ffi::reflection::AccessPath;
   
   class DocNode : public ffi::Object {
    public:
     mutable ffi::Array<AccessPath> source_paths;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<DocNode>().def_rw("source_paths", &DocNode::source_paths);
     }
   
     static constexpr bool _type_mutable = true;
   
     TVM_FFI_DECLARE_OBJECT_INFO("script.printer.Doc", DocNode, ffi::Object);
   };
   
   class Doc : public ffi::ObjectRef {
    protected:
     Doc() = default;
   
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Doc, ffi::ObjectRef, DocNode);
   };
   
   class ExprDoc;
   
   class ExprDocNode : public DocNode {
    public:
     TVM_TIRX_DLL ExprDoc Attr(ffi::String attr) const;
   
     TVM_TIRX_DLL ExprDoc operator[](ffi::Array<Doc> indices) const;
   
     TVM_TIRX_DLL ExprDoc Call(ffi::Array<ExprDoc, void> args) const;
   
     TVM_TIRX_DLL ExprDoc Call(ffi::Array<ExprDoc, void> args,       //
                               ffi::Array<ffi::String> kwargs_keys,  //
                               ffi::Array<ExprDoc, void> kwargs_values) const;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ExprDocNode>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO("script.printer.ExprDoc", ExprDocNode, DocNode);
   };
   
   class ExprDoc : public Doc {
    protected:
     ExprDoc() = default;
   
    public:
     TVM_TIRX_DLL ExprDoc operator[](ffi::Array<Doc> indices) const;
   
     explicit ExprDoc(ffi::ObjectPtr<ExprDocNode> data) : Doc(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(data != nullptr);
       data_ = std::move(data);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprDoc, Doc, ExprDocNode);
   };
   
   class StmtDocNode : public DocNode {
    public:
     mutable ffi::Optional<ffi::String> comment{std::nullopt};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<StmtDocNode>().def_rw("comment", &StmtDocNode::comment);
     }
     TVM_FFI_DECLARE_OBJECT_INFO("script.printer.StmtDoc", StmtDocNode, DocNode);
   };
   
   class StmtDoc : public Doc {
    protected:
     StmtDoc() = default;
   
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtDoc, Doc, StmtDocNode);
   };
   
   class StmtBlockDocNode : public DocNode {
    public:
     ffi::Array<StmtDoc> stmts;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<StmtBlockDocNode>().def_ro("stmts", &StmtBlockDocNode::stmts);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.StmtBlockDoc", StmtBlockDocNode, DocNode);
   };
   
   class StmtBlockDoc : public Doc {
    public:
     TVM_TIRX_DLL explicit StmtBlockDoc(ffi::Array<StmtDoc> stmts);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtBlockDoc, Doc, StmtBlockDocNode);
   };
   
   class LiteralDocNode : public ExprDocNode {
    public:
     ffi::Any value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LiteralDocNode>().def_ro("value", &LiteralDocNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.LiteralDoc", LiteralDocNode, ExprDocNode);
   };
   
   class LiteralDoc : public ExprDoc {
    protected:
     TVM_TIRX_DLL explicit LiteralDoc(ffi::Any value, const ffi::Optional<AccessPath>& object_path);
   
    public:
     static LiteralDoc None(const ffi::Optional<AccessPath>& p) {
       return LiteralDoc(ffi::Any(nullptr), p);
     }
     static LiteralDoc Int(int64_t v, const ffi::Optional<AccessPath>& p) {
       return LiteralDoc(IntImm::Int64(v), p);
     }
   
     static LiteralDoc Int(IntImm v, const ffi::Optional<AccessPath>& p) {
       return LiteralDoc(std::move(v), p);
     }
     static LiteralDoc Boolean(bool v, const ffi::Optional<AccessPath>& p) {
       return LiteralDoc(IntImm::Bool(v), p);
     }
     static LiteralDoc Float(double v, const ffi::Optional<AccessPath>& p) {
       return LiteralDoc(FloatImm(PrimType::Float(64), v), p);
     }
     static LiteralDoc Str(const ffi::String& v, const ffi::Optional<AccessPath>& p) {
       return LiteralDoc(v, p);
     }
     static LiteralDoc DataType(DLDataType v, const ffi::Optional<AccessPath>& p) {
       std::string dtype =
           v == DLDataType{kDLOpaqueHandle, 0, 0} ? "void" : ffi::DLDataTypeToString(v);
       return LiteralDoc::Str(dtype, p);
     }
     static LiteralDoc Device(const DLDevice& v, const ffi::Optional<AccessPath>& p) {
       std::ostringstream os;
       runtime::operator<<(os, v);
       return LiteralDoc::Str(os.str(), p);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LiteralDoc, ExprDoc, LiteralDocNode);
   };
   
   class ExprStringDocNode : public ExprDocNode {
    public:
     ExprDoc value{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ExprStringDocNode>().def_ro("value", &ExprStringDocNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ExprStringDoc", ExprStringDocNode, ExprDocNode);
   };
   
   class ExprStringDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit ExprStringDoc(ExprDoc value, const ffi::Optional<AccessPath>& object_path);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprStringDoc, ExprDoc, ExprStringDocNode);
   };
   
   class IdDocNode : public ExprDocNode {
    public:
     ffi::String name;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IdDocNode>().def_ro("name", &IdDocNode::name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IdDoc", IdDocNode, ExprDocNode);
   };
   
   class IdDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit IdDoc(ffi::String name);
     explicit IdDoc(std::nullptr_t) : ExprDoc(nullptr) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IdDoc, ExprDoc, IdDocNode);
   };
   
   class AttrAccessDocNode : public ExprDocNode {
    public:
     ExprDoc value{ffi::UnsafeInit()};
     ffi::String name;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AttrAccessDocNode>()
           .def_ro("value", &AttrAccessDocNode::value)
           .def_ro("name", &AttrAccessDocNode::name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AttrAccessDoc", AttrAccessDocNode, ExprDocNode);
   };
   
   class AttrAccessDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit AttrAccessDoc(ExprDoc value, ffi::String name);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AttrAccessDoc, ExprDoc, AttrAccessDocNode);
   };
   
   class IndexDocNode : public ExprDocNode {
    public:
     ExprDoc value{ffi::UnsafeInit()};
     ffi::Array<Doc> indices;  // Each element is union of: Slice / ExprDoc
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IndexDocNode>()
           .def_ro("value", &IndexDocNode::value)
           .def_ro("indices", &IndexDocNode::indices);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IndexDoc", IndexDocNode, ExprDocNode);
   };
   
   class IndexDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit IndexDoc(ExprDoc value, ffi::Array<Doc> indices);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IndexDoc, ExprDoc, IndexDocNode);
   };
   
   class CallDocNode : public ExprDocNode {
    public:
     ExprDoc callee{ffi::UnsafeInit()};
     ffi::Array<ExprDoc> args;
     ffi::Array<ffi::String> kwargs_keys;
     ffi::Array<ExprDoc> kwargs_values;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CallDocNode>()
           .def_ro("callee", &CallDocNode::callee)
           .def_ro("args", &CallDocNode::args)
           .def_ro("kwargs_keys", &CallDocNode::kwargs_keys)
           .def_ro("kwargs_values", &CallDocNode::kwargs_values);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.CallDoc", CallDocNode, ExprDocNode);
   };
   
   class CallDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL CallDoc(ExprDoc callee, ffi::Array<ExprDoc> args,
                          ffi::Array<ffi::String> kwargs_keys, ffi::Array<ExprDoc> kwargs_values);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CallDoc, ExprDoc, CallDocNode);
   };
   
   class OperationDocNode : public ExprDocNode {
    public:
     enum class Kind : int32_t {
       // Unary operators
       kUnaryStart = 0,
       kUSub = 1,    // -x
       kInvert = 2,  // ~x
       kNot = 3,     // not x
       kUnaryEnd = 4,
   
       // Binary operators
       kBinaryStart = 5,
       kAdd = 6,        // +
       kSub = 7,        // -
       kMult = 8,       // *
       kDiv = 9,        // /
       kFloorDiv = 10,  // // in Python
       kMod = 11,       // % in Python
       kPow = 12,       // ** in Python
       kLShift = 13,    // <<
       kRShift = 14,    // >>
       kBitAnd = 15,    // &
       kBitOr = 16,     // |
       kBitXor = 17,    // ^
       kLt = 18,        // <
       kLtE = 19,       // <=
       kEq = 20,        // ==
       kNotEq = 21,     // !=
       kGt = 22,        // >
       kGtE = 23,       // >=
       kAnd = 24,       // and
       kOr = 25,        // or
       kMatMul = 26,    // @
       kBinaryEnd = 27,
   
       // Special
       kSpecialStart = 28,
       kIfThenElse = 29,  // <operands[1]> if <operands[0]> else <operands[2]>
       kSpecialEnd = 30
     };
   
     Kind kind;
     ffi::Array<ExprDoc> operands;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<OperationDocNode>()
           .def_ro("kind", &OperationDocNode::kind)
           .def_ro("operands", &OperationDocNode::operands);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.OperationDoc", OperationDocNode, ExprDocNode);
   };
   
   class OperationDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit OperationDoc(OperationDocNode::Kind kind, ffi::Array<ExprDoc> operands);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OperationDoc, ExprDoc, OperationDocNode);
   };
   
   class LambdaDocNode : public ExprDocNode {
    public:
     ffi::Array<IdDoc> args;
     ExprDoc body{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LambdaDocNode>()
           .def_ro("args", &LambdaDocNode::args)
           .def_ro("body", &LambdaDocNode::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.LambdaDoc", LambdaDocNode, ExprDocNode);
   };
   
   class LambdaDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit LambdaDoc(ffi::Array<IdDoc> args, ExprDoc body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LambdaDoc, ExprDoc, LambdaDocNode);
   };
   
   class TupleDocNode : public ExprDocNode {
    public:
     ffi::Array<ExprDoc> elements;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleDocNode>().def_ro("elements", &TupleDocNode::elements);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.TupleDoc", TupleDocNode, ExprDocNode);
   };
   
   class TupleDoc : public ExprDoc {
    public:
     TupleDoc() : ExprDoc(ffi::make_object<TupleDocNode>()) {}
     TVM_TIRX_DLL explicit TupleDoc(ffi::Array<ExprDoc> elements);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleDoc, ExprDoc, TupleDocNode);
   };
   
   class ListDocNode : public ExprDocNode {
    public:
     ffi::Array<ExprDoc> elements;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ListDocNode>().def_ro("elements", &ListDocNode::elements);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ListDoc", ListDocNode, ExprDocNode);
   };
   
   class ListDoc : public ExprDoc {
    public:
     ListDoc() : ExprDoc(ffi::make_object<ListDocNode>()) {}
     TVM_TIRX_DLL explicit ListDoc(ffi::Array<ExprDoc> elements);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ListDoc, ExprDoc, ListDocNode);
   };
   
   class DictDocNode : public ExprDocNode {
    public:
     ffi::Array<ExprDoc> keys;
     ffi::Array<ExprDoc> values;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<DictDocNode>()
           .def_ro("keys", &DictDocNode::keys)
           .def_ro("values", &DictDocNode::values);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.DictDoc", DictDocNode, ExprDocNode);
   };
   
   class DictDoc : public ExprDoc {
    public:
     DictDoc() : ExprDoc(ffi::make_object<DictDocNode>()) {}
     TVM_TIRX_DLL explicit DictDoc(ffi::Array<ExprDoc> keys, ffi::Array<ExprDoc> values);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DictDoc, ExprDoc, DictDocNode);
   };
   
   class SliceDocNode : public DocNode {
    public:
     ffi::Optional<ExprDoc> start;
     ffi::Optional<ExprDoc> stop;
     ffi::Optional<ExprDoc> step;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<SliceDocNode>()
           .def_ro("start", &SliceDocNode::start)
           .def_ro("stop", &SliceDocNode::stop)
           .def_ro("step", &SliceDocNode::step);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.SliceDoc", SliceDocNode, DocNode);
   };
   
   class SliceDoc : public Doc {
    public:
     TVM_TIRX_DLL explicit SliceDoc(ffi::Optional<ExprDoc> start, ffi::Optional<ExprDoc> stop,
                                    ffi::Optional<ExprDoc> step);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SliceDoc, Doc, SliceDocNode);
   };
   
   class AssignDocNode : public StmtDocNode {
    public:
     ExprDoc lhs{ffi::UnsafeInit()};
     ffi::Optional<ExprDoc> rhs;
     ffi::Optional<ExprDoc> annotation;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AssignDocNode>()
           .def_ro("lhs", &AssignDocNode::lhs)
           .def_ro("rhs", &AssignDocNode::rhs)
           .def_ro("annotation", &AssignDocNode::annotation);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AssignDoc", AssignDocNode, StmtDocNode);
   };
   
   class AssignDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit AssignDoc(ExprDoc lhs, ffi::Optional<ExprDoc> rhs,
                                     ffi::Optional<ExprDoc> annotation);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssignDoc, StmtDoc, AssignDocNode);
   };
   
   class IfDocNode : public StmtDocNode {
    public:
     ExprDoc predicate{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> then_branch;
     ffi::Optional<ffi::Array<StmtDoc>> else_branch;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IfDocNode>()
           .def_ro("predicate", &IfDocNode::predicate)
           .def_ro("then_branch", &IfDocNode::then_branch)
           .def_ro("else_branch", &IfDocNode::else_branch);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IfDoc", IfDocNode, StmtDocNode);
   };
   
   class IfDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit IfDoc(ExprDoc predicate, ffi::Array<StmtDoc> then_branch,
                                 ffi::Optional<ffi::Array<StmtDoc>> else_branch);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfDoc, StmtDoc, IfDocNode);
   };
   
   class WhileDocNode : public StmtDocNode {
    public:
     ExprDoc predicate{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<WhileDocNode>()
           .def_ro("predicate", &WhileDocNode::predicate)
           .def_ro("body", &WhileDocNode::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.WhileDoc", WhileDocNode, StmtDocNode);
   };
   
   class WhileDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit WhileDoc(ExprDoc predicate, ffi::Array<StmtDoc> body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WhileDoc, StmtDoc, WhileDocNode);
   };
   
   class BreakDocNode : public StmtDocNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BreakDocNode>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.BreakDoc", BreakDocNode, StmtDocNode);
   };
   
   class BreakDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit BreakDoc();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BreakDoc, StmtDoc, BreakDocNode);
   };
   
   class ContinueDocNode : public StmtDocNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ContinueDocNode>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ContinueDoc", ContinueDocNode, StmtDocNode);
   };
   
   class ContinueDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ContinueDoc();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ContinueDoc, StmtDoc, ContinueDocNode);
   };
   
   class ForDocNode : public StmtDocNode {
    public:
     ExprDoc lhs{ffi::UnsafeInit()};
     ExprDoc rhs{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ForDocNode>()
           .def_ro("lhs", &ForDocNode::lhs)
           .def_ro("rhs", &ForDocNode::rhs)
           .def_ro("body", &ForDocNode::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ForDoc", ForDocNode, StmtDocNode);
   };
   
   class ForDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ForDoc(ExprDoc lhs, ExprDoc rhs, ffi::Array<StmtDoc> body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ForDoc, StmtDoc, ForDocNode);
   };
   
   class ScopeDocNode : public StmtDocNode {
    public:
     ffi::Optional<ExprDoc> lhs{std::nullopt};
     ExprDoc rhs{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> body;
     bool allow_concise_scoping{false};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ScopeDocNode>()
           .def_ro("lhs", &ScopeDocNode::lhs)
           .def_ro("rhs", &ScopeDocNode::rhs)
           .def_ro("body", &ScopeDocNode::body)
           .def_ro("allow_concise_scoping", &ScopeDocNode::allow_concise_scoping);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ScopeDoc", ScopeDocNode, StmtDocNode);
   };
   
   class ScopeDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ScopeDoc(ffi::Optional<ExprDoc> lhs, ExprDoc rhs, ffi::Array<StmtDoc> body,
                                    bool allow_concise_scoping = false);
   
     TVM_TIRX_DLL explicit ScopeDoc(ExprDoc rhs, ffi::Array<StmtDoc> body);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScopeDoc, StmtDoc, ScopeDocNode);
   };
   
   class ExprStmtDocNode : public StmtDocNode {
    public:
     ExprDoc expr{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ExprStmtDocNode>().def_ro("expr", &ExprStmtDocNode::expr);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ExprStmtDoc", ExprStmtDocNode, StmtDocNode);
   };
   
   class ExprStmtDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ExprStmtDoc(ExprDoc expr);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprStmtDoc, StmtDoc, ExprStmtDocNode);
   };
   
   class AssertDocNode : public StmtDocNode {
    public:
     ExprDoc test{ffi::UnsafeInit()};
     ffi::Optional<ExprDoc> msg{std::nullopt};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AssertDocNode>()
           .def_ro("test", &AssertDocNode::test)
           .def_ro("msg", &AssertDocNode::msg);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AssertDoc", AssertDocNode, StmtDocNode);
   };
   
   class AssertDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit AssertDoc(ExprDoc test, ffi::Optional<ExprDoc> msg = std::nullopt);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertDoc, StmtDoc, AssertDocNode);
   };
   
   class ReturnDocNode : public StmtDocNode {
    public:
     ExprDoc value{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ReturnDocNode>().def_ro("value", &ReturnDocNode::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ReturnDoc", ReturnDocNode, StmtDocNode);
   };
   
   class ReturnDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ReturnDoc(ExprDoc value);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ReturnDoc, StmtDoc, ReturnDocNode);
   };
   
   class FunctionDocNode : public StmtDocNode {
    public:
     IdDoc name{ffi::UnsafeInit{}};
     ffi::Array<AssignDoc> args;
     ffi::Array<ExprDoc> decorators;
     ffi::Optional<ExprDoc> return_type{std::nullopt};
     ffi::Array<StmtDoc> body;
     ffi::Array<Doc> type_params;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FunctionDocNode>()
           .def_ro("name", &FunctionDocNode::name)
           .def_ro("args", &FunctionDocNode::args)
           .def_ro("decorators", &FunctionDocNode::decorators)
           .def_ro("return_type", &FunctionDocNode::return_type)
           .def_ro("body", &FunctionDocNode::body)
           .def_ro("type_params", &FunctionDocNode::type_params);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.FunctionDoc", FunctionDocNode, StmtDocNode);
   };
   
   class FunctionDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit FunctionDoc(IdDoc name, ffi::Array<AssignDoc> args,
                                       ffi::Array<ExprDoc> decorators,
                                       ffi::Optional<ExprDoc> return_type, ffi::Array<StmtDoc> body,
                                       ffi::Array<Doc> type_params = {});
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionDoc, StmtDoc, FunctionDocNode);
   };
   
   class ClassDocNode : public StmtDocNode {
    public:
     IdDoc name{ffi::UnsafeInit{}};
     ffi::Array<ExprDoc> decorators;
     ffi::Array<StmtDoc> body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ClassDocNode>()
           .def_ro("name", &ClassDocNode::name)
           .def_ro("decorators", &ClassDocNode::decorators)
           .def_ro("body", &ClassDocNode::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ClassDoc", ClassDocNode, StmtDocNode);
   };
   
   class ClassDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ClassDoc(IdDoc name, ffi::Array<ExprDoc> decorators,
                                    ffi::Array<StmtDoc> body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ClassDoc, StmtDoc, ClassDocNode);
   };
   
   class CommentDocNode : public StmtDocNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CommentDocNode>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.CommentDoc", CommentDocNode, StmtDocNode);
   };
   
   class CommentDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit CommentDoc(ffi::String comment);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CommentDoc, StmtDoc, CommentDocNode);
   };
   
   class DocStringDocNode : public StmtDocNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<DocStringDocNode>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.DocStringDoc", DocStringDocNode, StmtDocNode);
   };
   
   class DocStringDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit DocStringDoc(ffi::String docs);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DocStringDoc, StmtDoc, DocStringDocNode);
   };
   
   class NamespaceDocNode : public ExprDocNode {
    public:
     ffi::String canonical_name;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<NamespaceDocNode>().def_ro("canonical_name",
                                                             &NamespaceDocNode::canonical_name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.NamespaceDoc", NamespaceDocNode, ExprDocNode);
   };
   
   class NamespaceDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit NamespaceDoc(ffi::String canonical_name);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NamespaceDoc, ExprDoc, NamespaceDocNode);
   };
   
   }  // namespace printer
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_SCRIPT_PRINTER_DOC_H_
