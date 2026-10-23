
.. _program_listing_file_tvm_tirx_script_ir_builder_frame.h:

Program Listing for File frame.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_script_ir_builder_frame.h>` (``tvm/tirx/script/ir_builder/frame.h``)

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
   #ifndef TVM_TIRX_SCRIPT_IR_BUILDER_FRAME_H_
   #define TVM_TIRX_SCRIPT_IR_BUILDER_FRAME_H_
   
   #include <tvm/script/ir_builder/base.h>
   #include <tvm/script/ir_builder/frame.h>
   #include <tvm/tirx/function.h>
   #include <tvm/tirx/stmt.h>
   #include <tvm/tirx/tile/thread_scope.h>
   
   #include <utility>
   
   namespace tvm {
   namespace script {
   namespace ir_builder {
   namespace tirx {
   
   class TVM_TIRX_DLL FunctionFrameObj : public ir::StmtFrameObj {
    public:
     ffi::Optional<ffi::String> name;
     ffi::Array<tvm::Var> args;
     bool is_private;
     ffi::Optional<Type> ret_type;
     ffi::Map<ffi::String, Any> attrs;
     bool persistent;
     bool is_declaration{false};
     ffi::Optional<tvm::tirx::Function> function;
     ffi::Optional<GlobalVar> global_var;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FunctionFrameObj>()
           .def_ro("name", &FunctionFrameObj::name)
           .def_ro("args", &FunctionFrameObj::args)
           .def_ro("is_private", &FunctionFrameObj::is_private)
           .def_ro("ret_type", &FunctionFrameObj::ret_type)
           .def_ro("attrs", &FunctionFrameObj::attrs)
           .def_ro("persistent", &FunctionFrameObj::persistent)
           .def_ro("is_declaration", &FunctionFrameObj::is_declaration)
           .def_ro("function", &FunctionFrameObj::function)
           .def_ro("global_var", &FunctionFrameObj::global_var);
     }
     TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.tirx.FunctionFrame", FunctionFrameObj,
                                 ir::StmtFrameObj);
   
    public:
     void ExitWithScope() final;
   
     using AttrValidator = ffi::TypedFunction<void(const FunctionFrameObj*, const ffi::Any&)>;
     static void RegisterAttrValidator(ffi::String key, AttrValidator validator);
     void ValidateAttrs() const;
   
     virtual tvm::tirx::Function FinalizeFunction(tvm::tirx::Function func);
   };
   
   class FunctionFrame : public ir::StmtFrame {
    public:
     explicit FunctionFrame(ffi::ObjectPtr<FunctionFrameObj> data) : ir::StmtFrame(ffi::UnsafeInit{}) {
       TVM_FFI_ICHECK(data != nullptr);
       data_ = std::move(data);
     }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionFrame, ir::StmtFrame, FunctionFrameObj);
   };
   
   }  // namespace tirx
   }  // namespace ir_builder
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_TIRX_SCRIPT_IR_BUILDER_FRAME_H_
