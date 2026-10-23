
.. _program_listing_file_tvm_script_ir_builder_ir.h:

Program Listing for File ir.h
=============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_ir_builder_ir.h>` (``tvm/script/ir_builder/ir.h``)

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
   #ifndef TVM_SCRIPT_IR_BUILDER_IR_H_
   #define TVM_SCRIPT_IR_BUILDER_IR_H_
   
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ffi/container/variant.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/function.h>
   #include <tvm/script/ir_builder/frame.h>
   
   #include <vector>
   
   namespace tvm {
   namespace script {
   namespace ir_builder {
   namespace ir {
   
   TVM_TIRX_DLL IRModuleFrame IRModule();
   
   TVM_TIRX_DLL GlobalVar DeclFunction(const ffi::String& func_name, const BaseFunc& func_signature);
   
   TVM_TIRX_DLL void DefFunction(const ffi::String& func_name, const BaseFunc& func);
   
   TVM_TIRX_DLL ForFrame For(ForKind kind, PrimExpr start, PrimExpr stop,
                             ffi::Optional<PrimExpr> step = std::nullopt,
                             ffi::Optional<PrimType> dtype = std::nullopt,
                             ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt);
   
   TVM_TIRX_DLL ForFrame
   Grid(ffi::Array<ffi::Variant<PrimExpr, ffi::Tuple<PrimExpr, PrimExpr>>> extents,
        ffi::Optional<PrimType> dtype = std::nullopt);
   
   TVM_TIRX_DLL AssertFrame Assert(PrimExpr condition, ffi::String error_kind,
                                   ffi::Array<ffi::String> message_parts);
   
   TVM_TIRX_DLL WhileFrame While(PrimExpr condition);
   
   TVM_TIRX_DLL tvm::Stmt Return(Expr value);
   
   TVM_TIRX_DLL tvm::Stmt Break();
   
   TVM_TIRX_DLL tvm::Stmt Continue();
   
   TVM_TIRX_DLL IfFrame If(PrimExpr condition);
   
   TVM_TIRX_DLL ThenFrame Then();
   
   TVM_TIRX_DLL ElseFrame Else();
   
   TVM_TIRX_DLL RegionFrame Region(Op op, ffi::Array<Expr> args,
                                   ffi::Optional<ffi::Array<Var>> body_params = std::nullopt,
                                   DictAttrs attrs = DictAttrs());
   
   TVM_TIRX_DLL tvm::Stmt Evaluate(Expr value);
   
   }  // namespace ir
   }  // namespace ir_builder
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_SCRIPT_IR_BUILDER_IR_H_
