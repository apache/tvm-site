
.. _program_listing_file_tvm_script_ir_builder_frame.h:

Program Listing for File frame.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_ir_builder_frame.h>` (``tvm/script/ir_builder/frame.h``)

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
   #ifndef TVM_SCRIPT_IR_BUILDER_FRAME_H_
   #define TVM_SCRIPT_IR_BUILDER_FRAME_H_
   
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/function.h>
   #include <tvm/ir/module.h>
   #include <tvm/ir/stmt.h>
   #include <tvm/script/ir_builder/base.h>
   
   #include <vector>
   
   namespace tvm {
   namespace script {
   namespace ir_builder {
   namespace ir {
   
   class IRModuleFrameObj : public IRBuilderFrameObj {
    public:
     ffi::Map<ffi::String, GlobalVar> global_var_map;
     ffi::Map<GlobalVar, BaseFunc> functions;
     ffi::Map<ffi::String, Any> attrs;
     ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IRModuleFrameObj>()
           .def_ro("global_vars", &IRModuleFrameObj::global_var_map)
           .def_ro("functions", &IRModuleFrameObj::functions)
           .def_ro("attrs", &IRModuleFrameObj::attrs)
           .def_ro("global_infos", &IRModuleFrameObj::global_infos);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.IRModuleFrame", IRModuleFrameObj,
                                       IRBuilderFrameObj);
   
    public:
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class IRModuleFrame : public IRBuilderFrame {
    public:
     explicit IRModuleFrame(ffi::ObjectPtr<IRModuleFrameObj> data) : IRBuilderFrame(data) {
       TVM_FFI_ICHECK(data != nullptr);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IRModuleFrame, IRBuilderFrame, IRModuleFrameObj);
   };
   
   TVM_TIRX_DLL IRModuleFrame FindModuleFrame();
   
   class StmtFrameObj : public IRBuilderFrameObj {
    public:
     ffi::Array<tvm::Stmt> stmts;
   
     static void RegisterReflection() {
       ffi::reflection::ObjectDef<StmtFrameObj>().def_ro("stmts", &StmtFrameObj::stmts);
     }
     TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.StmtFrame", StmtFrameObj, IRBuilderFrameObj);
   };
   
   class StmtFrame : public IRBuilderFrame {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtFrame, IRBuilderFrame, StmtFrameObj);
   
    protected:
     StmtFrame() = default;
     explicit StmtFrame(ffi::ObjectPtr<StmtFrameObj> data) : IRBuilderFrame(data) {}
   };
   
   TVM_TIRX_DLL void AddToParent(tvm::Stmt stmt, Location loc);
   
   class ForFrameObj : public StmtFrameObj {
    public:
     using FMakeForLoop = ffi::TypedFunction<tvm::Stmt(
         ffi::Array<tvm::Var> loop_vars, ffi::Array<Range> loop_extents,
         ffi::Array<ffi::Optional<PrimExpr>> loop_steps, tvm::SeqStmt loop_body, Location loc)>;
     ffi::Array<tvm::Var> vars;
     ffi::Array<Range> doms;
     ffi::Array<ffi::Optional<PrimExpr>> steps;
     FMakeForLoop f_make_for_loop;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ForFrameObj>()
           .def_ro("vars", &ForFrameObj::vars)
           .def_ro("doms", &ForFrameObj::doms);
       // `f_make_for_loop` is not registered as it's not visited.
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.ForFrame", ForFrameObj, StmtFrameObj);
   
    public:
     TVM_TIRX_DLL void SetNames(
         ffi::Optional<ffi::Variant<ffi::String, ffi::Array<ffi::String>>> names);
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class ForFrame : public StmtFrame {
    public:
     explicit ForFrame(ffi::ObjectPtr<ForFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(data != nullptr);
       data_ = std::move(data);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ForFrame, StmtFrame, ForFrameObj);
   };
   
   class AssertFrameObj : public StmtFrameObj {
    public:
     explicit AssertFrameObj(ffi::UnsafeInit tag) : condition(tag), error_kind(tag) {}
   
     AssertFrameObj(PrimExpr condition, tvm::StringImm error_kind)
         : condition(std::move(condition)), error_kind(std::move(error_kind)) {}
   
     PrimExpr condition;
     tvm::StringImm error_kind;
     ffi::Array<tvm::StringImm> message_parts;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<AssertFrameObj>()
           .def_ro("condition", &AssertFrameObj::condition)
           .def_ro("error_kind", &AssertFrameObj::error_kind)
           .def_ro("message_parts", &AssertFrameObj::message_parts);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.AssertFrame", AssertFrameObj, StmtFrameObj);
   
    public:
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class AssertFrame : public StmtFrame {
    public:
     explicit AssertFrame(ffi::ObjectPtr<AssertFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(data != nullptr);
       data_ = std::move(data);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertFrame, StmtFrame, AssertFrameObj);
   };
   
   class RegionFrameObj : public StmtFrameObj {
    public:
     explicit RegionFrameObj(ffi::UnsafeInit tag) : op(tag) {}
     explicit RegionFrameObj(Op op) : op(std::move(op)) {}
   
     Op op;
     ffi::Array<Expr> args;
     ffi::Array<Var> body_params;
     DictAttrs attrs;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<RegionFrameObj>()
           .def_ro("op", &RegionFrameObj::op)
           .def_ro("args", &RegionFrameObj::args)
           .def_ro("body_params", &RegionFrameObj::body_params)
           .def_ro("attrs", &RegionFrameObj::attrs);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.RegionFrame", RegionFrameObj, StmtFrameObj);
   
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class RegionFrame : public StmtFrame {
    public:
     explicit RegionFrame(ffi::ObjectPtr<RegionFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(data != nullptr);
       data_ = std::move(data);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionFrame, StmtFrame, RegionFrameObj);
   };
   
   class WhileFrameObj : public StmtFrameObj {
    public:
     explicit WhileFrameObj(ffi::UnsafeInit tag) : condition(tag) {}
   
     explicit WhileFrameObj(PrimExpr condition) : condition(std::move(condition)) {}
   
     PrimExpr condition;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<WhileFrameObj>().def_ro("condition", &WhileFrameObj::condition);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.WhileFrame", WhileFrameObj, StmtFrameObj);
   
    public:
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class WhileFrame : public StmtFrame {
    public:
     explicit WhileFrame(ffi::ObjectPtr<WhileFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(data != nullptr);
       data_ = std::move(data);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WhileFrame, StmtFrame, WhileFrameObj);
   };
   
   class IfFrameObj : public StmtFrameObj {
    public:
     explicit IfFrameObj(ffi::UnsafeInit tag) : condition(tag) {}
   
     explicit IfFrameObj(PrimExpr condition) : condition(std::move(condition)) {}
   
     PrimExpr condition;
     ffi::Optional<ffi::Array<tvm::Stmt>> then_stmts;
     ffi::Optional<ffi::Array<tvm::Stmt>> else_stmts;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IfFrameObj>()
           .def_ro("condition", &IfFrameObj::condition)
           .def_ro("then_stmts", &IfFrameObj::then_stmts)
           .def_ro("else_stmts", &IfFrameObj::else_stmts);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.IfFrame", IfFrameObj, StmtFrameObj);
   
    public:
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class IfFrame : public StmtFrame {
    public:
     explicit IfFrame(ffi::ObjectPtr<IfFrameObj> data) : StmtFrame(data) {
       TVM_FFI_ICHECK(data != nullptr);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfFrame, StmtFrame, IfFrameObj);
   };
   
   class ThenFrameObj : public StmtFrameObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ThenFrameObj>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.ThenFrame", ThenFrameObj, StmtFrameObj);
   
    public:
     TVM_TIRX_DLL void EnterWithScope() final;
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class ThenFrame : public StmtFrame {
    public:
     explicit ThenFrame(ffi::ObjectPtr<ThenFrameObj> data) : StmtFrame(data) {
       TVM_FFI_ICHECK(data != nullptr);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ThenFrame, StmtFrame, ThenFrameObj);
   };
   
   class ElseFrameObj : public StmtFrameObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<ElseFrameObj>();
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.ElseFrame", ElseFrameObj, StmtFrameObj);
   
    public:
     TVM_TIRX_DLL void EnterWithScope() final;
     TVM_TIRX_DLL void ExitWithScope() final;
   };
   
   class ElseFrame : public StmtFrame {
    public:
     explicit ElseFrame(ffi::ObjectPtr<ElseFrameObj> data) : StmtFrame(data) {
       TVM_FFI_ICHECK(data != nullptr);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ElseFrame, StmtFrame, ElseFrameObj);
   };
   
   }  // namespace ir
   }  // namespace ir_builder
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_SCRIPT_IR_BUILDER_FRAME_H_
