
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
   
   class DocObj : public ffi::Object {
    public:
     mutable ffi::Array<AccessPath> source_paths;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<DocObj>().def_rw("source_paths", &DocObj::source_paths);
     }
   
     static constexpr bool _type_mutable = true;
   
     TVM_FFI_DECLARE_OBJECT_INFO("script.printer.Doc", DocObj, ffi::Object);
   };
   
   class Doc : public ffi::ObjectRef {
    protected:
     Doc() = default;
   
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Doc, ffi::ObjectRef, DocObj);
   };
   
   class ExprDoc;
   
   class ExprDocObj : public DocObj {
    public:
     TVM_TIRX_DLL ExprDoc Attr(ffi::String attr) const;
   
     TVM_TIRX_DLL ExprDoc operator[](ffi::Array<Doc> indices) const;
   
     TVM_TIRX_DLL ExprDoc Call(ffi::Array<ExprDoc, void> args) const;
   
     TVM_TIRX_DLL ExprDoc Call(ffi::Array<ExprDoc, void> args,       //
                               ffi::Array<ffi::String> kwargs_keys,  //
                               ffi::Array<ExprDoc, void> kwargs_values) const;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ExprDocObj>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO("script.printer.ExprDoc", ExprDocObj, DocObj);
   };
   
   class ExprDoc : public Doc {
    protected:
     ExprDoc() = default;
   
    public:
     TVM_TIRX_DLL ExprDoc operator[](ffi::Array<Doc> indices) const;
   
     explicit ExprDoc(ffi::ObjectPtr<ExprDocObj> data) : Doc(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(data != nullptr);
       data_ = std::move(data);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprDoc, Doc, ExprDocObj);
   };
   
   class StmtDocObj : public DocObj {
    public:
     mutable ffi::Optional<ffi::String> comment{std::nullopt};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<StmtDocObj>().def_rw("comment", &StmtDocObj::comment);
     }
     TVM_FFI_DECLARE_OBJECT_INFO("script.printer.StmtDoc", StmtDocObj, DocObj);
   };
   
   class StmtDoc : public Doc {
    protected:
     StmtDoc() = default;
   
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtDoc, Doc, StmtDocObj);
   };
   
   class StmtBlockDocObj : public DocObj {
    public:
     ffi::Array<StmtDoc> stmts;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<StmtBlockDocObj>().def_ro("stmts", &StmtBlockDocObj::stmts);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.StmtBlockDoc", StmtBlockDocObj, DocObj);
   };
   
   class StmtBlockDoc : public Doc {
    public:
     TVM_TIRX_DLL explicit StmtBlockDoc(ffi::Array<StmtDoc> stmts);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtBlockDoc, Doc, StmtBlockDocObj);
   };
   
   class LiteralDocObj : public ExprDocObj {
    public:
     ffi::Any value;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LiteralDocObj>().def_ro("value", &LiteralDocObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.LiteralDoc", LiteralDocObj, ExprDocObj);
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
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LiteralDoc, ExprDoc, LiteralDocObj);
   };
   
   class ExprStringDocObj : public ExprDocObj {
    public:
     ExprDoc value{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ExprStringDocObj>().def_ro("value", &ExprStringDocObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ExprStringDoc", ExprStringDocObj, ExprDocObj);
   };
   
   class ExprStringDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit ExprStringDoc(ExprDoc value, const ffi::Optional<AccessPath>& object_path);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprStringDoc, ExprDoc, ExprStringDocObj);
   };
   
   class IdDocObj : public ExprDocObj {
    public:
     ffi::String name;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IdDocObj>().def_ro("name", &IdDocObj::name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IdDoc", IdDocObj, ExprDocObj);
   };
   
   class IdDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit IdDoc(ffi::String name);
     explicit IdDoc(std::nullptr_t) : ExprDoc(nullptr) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IdDoc, ExprDoc, IdDocObj);
   };
   
   class AttrAccessDocObj : public ExprDocObj {
    public:
     ExprDoc value{ffi::UnsafeInit()};
     ffi::String name;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AttrAccessDocObj>()
           .def_ro("value", &AttrAccessDocObj::value)
           .def_ro("name", &AttrAccessDocObj::name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AttrAccessDoc", AttrAccessDocObj, ExprDocObj);
   };
   
   class AttrAccessDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit AttrAccessDoc(ExprDoc value, ffi::String name);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AttrAccessDoc, ExprDoc, AttrAccessDocObj);
   };
   
   class IndexDocObj : public ExprDocObj {
    public:
     ExprDoc value{ffi::UnsafeInit()};
     ffi::Array<Doc> indices;  // Each element is union of: Slice / ExprDoc
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IndexDocObj>()
           .def_ro("value", &IndexDocObj::value)
           .def_ro("indices", &IndexDocObj::indices);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IndexDoc", IndexDocObj, ExprDocObj);
   };
   
   class IndexDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit IndexDoc(ExprDoc value, ffi::Array<Doc> indices);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IndexDoc, ExprDoc, IndexDocObj);
   };
   
   class CallDocObj : public ExprDocObj {
    public:
     ExprDoc callee{ffi::UnsafeInit()};
     ffi::Array<ExprDoc> args;
     ffi::Array<ffi::String> kwargs_keys;
     ffi::Array<ExprDoc> kwargs_values;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CallDocObj>()
           .def_ro("callee", &CallDocObj::callee)
           .def_ro("args", &CallDocObj::args)
           .def_ro("kwargs_keys", &CallDocObj::kwargs_keys)
           .def_ro("kwargs_values", &CallDocObj::kwargs_values);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.CallDoc", CallDocObj, ExprDocObj);
   };
   
   class CallDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL CallDoc(ExprDoc callee, ffi::Array<ExprDoc> args,
                          ffi::Array<ffi::String> kwargs_keys, ffi::Array<ExprDoc> kwargs_values);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CallDoc, ExprDoc, CallDocObj);
   };
   
   class OperationDocObj : public ExprDocObj {
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
       refl::ObjectDef<OperationDocObj>()
           .def_ro("kind", &OperationDocObj::kind)
           .def_ro("operands", &OperationDocObj::operands);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.OperationDoc", OperationDocObj, ExprDocObj);
   };
   
   class OperationDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit OperationDoc(OperationDocObj::Kind kind, ffi::Array<ExprDoc> operands);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OperationDoc, ExprDoc, OperationDocObj);
   };
   
   class LambdaDocObj : public ExprDocObj {
    public:
     ffi::Array<IdDoc> args;
     ExprDoc body{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<LambdaDocObj>()
           .def_ro("args", &LambdaDocObj::args)
           .def_ro("body", &LambdaDocObj::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.LambdaDoc", LambdaDocObj, ExprDocObj);
   };
   
   class LambdaDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit LambdaDoc(ffi::Array<IdDoc> args, ExprDoc body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LambdaDoc, ExprDoc, LambdaDocObj);
   };
   
   class TupleDocObj : public ExprDocObj {
    public:
     ffi::Array<ExprDoc> elements;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleDocObj>().def_ro("elements", &TupleDocObj::elements);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.TupleDoc", TupleDocObj, ExprDocObj);
   };
   
   class TupleDoc : public ExprDoc {
    public:
     TupleDoc() : ExprDoc(ffi::make_object<TupleDocObj>()) {}
     TVM_TIRX_DLL explicit TupleDoc(ffi::Array<ExprDoc> elements);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleDoc, ExprDoc, TupleDocObj);
   };
   
   class ListDocObj : public ExprDocObj {
    public:
     ffi::Array<ExprDoc> elements;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ListDocObj>().def_ro("elements", &ListDocObj::elements);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ListDoc", ListDocObj, ExprDocObj);
   };
   
   class ListDoc : public ExprDoc {
    public:
     ListDoc() : ExprDoc(ffi::make_object<ListDocObj>()) {}
     TVM_TIRX_DLL explicit ListDoc(ffi::Array<ExprDoc> elements);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ListDoc, ExprDoc, ListDocObj);
   };
   
   class DictDocObj : public ExprDocObj {
    public:
     ffi::Array<ExprDoc> keys;
     ffi::Array<ExprDoc> values;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<DictDocObj>()
           .def_ro("keys", &DictDocObj::keys)
           .def_ro("values", &DictDocObj::values);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.DictDoc", DictDocObj, ExprDocObj);
   };
   
   class DictDoc : public ExprDoc {
    public:
     DictDoc() : ExprDoc(ffi::make_object<DictDocObj>()) {}
     TVM_TIRX_DLL explicit DictDoc(ffi::Array<ExprDoc> keys, ffi::Array<ExprDoc> values);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DictDoc, ExprDoc, DictDocObj);
   };
   
   class SliceDocObj : public DocObj {
    public:
     ffi::Optional<ExprDoc> start;
     ffi::Optional<ExprDoc> stop;
     ffi::Optional<ExprDoc> step;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<SliceDocObj>()
           .def_ro("start", &SliceDocObj::start)
           .def_ro("stop", &SliceDocObj::stop)
           .def_ro("step", &SliceDocObj::step);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.SliceDoc", SliceDocObj, DocObj);
   };
   
   class SliceDoc : public Doc {
    public:
     TVM_TIRX_DLL explicit SliceDoc(ffi::Optional<ExprDoc> start, ffi::Optional<ExprDoc> stop,
                                    ffi::Optional<ExprDoc> step);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SliceDoc, Doc, SliceDocObj);
   };
   
   class AssignDocObj : public StmtDocObj {
    public:
     ExprDoc lhs{ffi::UnsafeInit()};
     ffi::Optional<ExprDoc> rhs;
     ffi::Optional<ExprDoc> annotation;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AssignDocObj>()
           .def_ro("lhs", &AssignDocObj::lhs)
           .def_ro("rhs", &AssignDocObj::rhs)
           .def_ro("annotation", &AssignDocObj::annotation);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AssignDoc", AssignDocObj, StmtDocObj);
   };
   
   class AssignDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit AssignDoc(ExprDoc lhs, ffi::Optional<ExprDoc> rhs,
                                     ffi::Optional<ExprDoc> annotation);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssignDoc, StmtDoc, AssignDocObj);
   };
   
   class IfDocObj : public StmtDocObj {
    public:
     ExprDoc predicate{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> then_branch;
     ffi::Optional<ffi::Array<StmtDoc>> else_branch;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IfDocObj>()
           .def_ro("predicate", &IfDocObj::predicate)
           .def_ro("then_branch", &IfDocObj::then_branch)
           .def_ro("else_branch", &IfDocObj::else_branch);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IfDoc", IfDocObj, StmtDocObj);
   };
   
   class IfDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit IfDoc(ExprDoc predicate, ffi::Array<StmtDoc> then_branch,
                                 ffi::Optional<ffi::Array<StmtDoc>> else_branch);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfDoc, StmtDoc, IfDocObj);
   };
   
   class WhileDocObj : public StmtDocObj {
    public:
     ExprDoc predicate{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<WhileDocObj>()
           .def_ro("predicate", &WhileDocObj::predicate)
           .def_ro("body", &WhileDocObj::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.WhileDoc", WhileDocObj, StmtDocObj);
   };
   
   class WhileDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit WhileDoc(ExprDoc predicate, ffi::Array<StmtDoc> body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WhileDoc, StmtDoc, WhileDocObj);
   };
   
   class BreakDocObj : public StmtDocObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<BreakDocObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.BreakDoc", BreakDocObj, StmtDocObj);
   };
   
   class BreakDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit BreakDoc();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BreakDoc, StmtDoc, BreakDocObj);
   };
   
   class ContinueDocObj : public StmtDocObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ContinueDocObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ContinueDoc", ContinueDocObj, StmtDocObj);
   };
   
   class ContinueDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ContinueDoc();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ContinueDoc, StmtDoc, ContinueDocObj);
   };
   
   class ForDocObj : public StmtDocObj {
    public:
     ExprDoc lhs{ffi::UnsafeInit()};
     ExprDoc rhs{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ForDocObj>()
           .def_ro("lhs", &ForDocObj::lhs)
           .def_ro("rhs", &ForDocObj::rhs)
           .def_ro("body", &ForDocObj::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ForDoc", ForDocObj, StmtDocObj);
   };
   
   class ForDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ForDoc(ExprDoc lhs, ExprDoc rhs, ffi::Array<StmtDoc> body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ForDoc, StmtDoc, ForDocObj);
   };
   
   class ScopeDocObj : public StmtDocObj {
    public:
     ffi::Optional<ExprDoc> lhs{std::nullopt};
     ExprDoc rhs{ffi::UnsafeInit()};
     ffi::Array<StmtDoc> body;
     bool allow_concise_scoping{false};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ScopeDocObj>()
           .def_ro("lhs", &ScopeDocObj::lhs)
           .def_ro("rhs", &ScopeDocObj::rhs)
           .def_ro("body", &ScopeDocObj::body)
           .def_ro("allow_concise_scoping", &ScopeDocObj::allow_concise_scoping);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ScopeDoc", ScopeDocObj, StmtDocObj);
   };
   
   class ScopeDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ScopeDoc(ffi::Optional<ExprDoc> lhs, ExprDoc rhs, ffi::Array<StmtDoc> body,
                                    bool allow_concise_scoping = false);
   
     TVM_TIRX_DLL explicit ScopeDoc(ExprDoc rhs, ffi::Array<StmtDoc> body);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScopeDoc, StmtDoc, ScopeDocObj);
   };
   
   class ExprStmtDocObj : public StmtDocObj {
    public:
     ExprDoc expr{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ExprStmtDocObj>().def_ro("expr", &ExprStmtDocObj::expr);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ExprStmtDoc", ExprStmtDocObj, StmtDocObj);
   };
   
   class ExprStmtDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ExprStmtDoc(ExprDoc expr);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprStmtDoc, StmtDoc, ExprStmtDocObj);
   };
   
   class AssertDocObj : public StmtDocObj {
    public:
     ExprDoc test{ffi::UnsafeInit()};
     ffi::Optional<ExprDoc> msg{std::nullopt};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AssertDocObj>()
           .def_ro("test", &AssertDocObj::test)
           .def_ro("msg", &AssertDocObj::msg);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AssertDoc", AssertDocObj, StmtDocObj);
   };
   
   class AssertDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit AssertDoc(ExprDoc test, ffi::Optional<ExprDoc> msg = std::nullopt);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertDoc, StmtDoc, AssertDocObj);
   };
   
   class ReturnDocObj : public StmtDocObj {
    public:
     ExprDoc value{ffi::UnsafeInit()};
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ReturnDocObj>().def_ro("value", &ReturnDocObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ReturnDoc", ReturnDocObj, StmtDocObj);
   };
   
   class ReturnDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ReturnDoc(ExprDoc value);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ReturnDoc, StmtDoc, ReturnDocObj);
   };
   
   class FunctionDocObj : public StmtDocObj {
    public:
     IdDoc name{ffi::UnsafeInit{}};
     ffi::Array<AssignDoc> args;
     ffi::Array<ExprDoc> decorators;
     ffi::Optional<ExprDoc> return_type{std::nullopt};
     ffi::Array<StmtDoc> body;
     ffi::Array<Doc> type_params;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FunctionDocObj>()
           .def_ro("name", &FunctionDocObj::name)
           .def_ro("args", &FunctionDocObj::args)
           .def_ro("decorators", &FunctionDocObj::decorators)
           .def_ro("return_type", &FunctionDocObj::return_type)
           .def_ro("body", &FunctionDocObj::body)
           .def_ro("type_params", &FunctionDocObj::type_params);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.FunctionDoc", FunctionDocObj, StmtDocObj);
   };
   
   class FunctionDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit FunctionDoc(IdDoc name, ffi::Array<AssignDoc> args,
                                       ffi::Array<ExprDoc> decorators,
                                       ffi::Optional<ExprDoc> return_type, ffi::Array<StmtDoc> body,
                                       ffi::Array<Doc> type_params = {});
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionDoc, StmtDoc, FunctionDocObj);
   };
   
   class ClassDocObj : public StmtDocObj {
    public:
     IdDoc name{ffi::UnsafeInit{}};
     ffi::Array<ExprDoc> decorators;
     ffi::Array<StmtDoc> body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ClassDocObj>()
           .def_ro("name", &ClassDocObj::name)
           .def_ro("decorators", &ClassDocObj::decorators)
           .def_ro("body", &ClassDocObj::body);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ClassDoc", ClassDocObj, StmtDocObj);
   };
   
   class ClassDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit ClassDoc(IdDoc name, ffi::Array<ExprDoc> decorators,
                                    ffi::Array<StmtDoc> body);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ClassDoc, StmtDoc, ClassDocObj);
   };
   
   class CommentDocObj : public StmtDocObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CommentDocObj>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.CommentDoc", CommentDocObj, StmtDocObj);
   };
   
   class CommentDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit CommentDoc(ffi::String comment);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CommentDoc, StmtDoc, CommentDocObj);
   };
   
   class DocStringDocObj : public StmtDocObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<DocStringDocObj>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.DocStringDoc", DocStringDocObj, StmtDocObj);
   };
   
   class DocStringDoc : public StmtDoc {
    public:
     TVM_TIRX_DLL explicit DocStringDoc(ffi::String docs);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DocStringDoc, StmtDoc, DocStringDocObj);
   };
   
   class NamespaceDocObj : public ExprDocObj {
    public:
     ffi::String canonical_name;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<NamespaceDocObj>().def_ro("canonical_name",
                                                            &NamespaceDocObj::canonical_name);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.NamespaceDoc", NamespaceDocObj, ExprDocObj);
   };
   
   class NamespaceDoc : public ExprDoc {
    public:
     TVM_TIRX_DLL explicit NamespaceDoc(ffi::String canonical_name);
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NamespaceDoc, ExprDoc, NamespaceDocObj);
   };
   
   }  // namespace printer
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_SCRIPT_PRINTER_DOC_H_
