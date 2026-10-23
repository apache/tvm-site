
.. _program_listing_file_tvm_tirx_script_ir_builder_ir.h:

Program Listing for File ir.h
=============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_script_ir_builder_ir.h>` (``tvm/tirx/script/ir_builder/ir.h``)

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
   #ifndef TVM_TIRX_SCRIPT_IR_BUILDER_IR_H_
   #define TVM_TIRX_SCRIPT_IR_BUILDER_IR_H_
   
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ffi/container/variant.h>
   #include <tvm/ir/prim/op.h>
   #include <tvm/runtime/tensor.h>
   #include <tvm/script/ir_builder/ir.h>
   #include <tvm/tirx/script/ir_builder/frame.h>
   #include <tvm/tirx/stmt.h>
   #include <tvm/tirx/tile/thread_scope.h>
   #include <tvm/tirx/type.h>
   
   namespace tvm {
   namespace script {
   namespace ir_builder {
   namespace tirx {
   
   using tvm::Var;
   using tvm::ffi::Tensor;
   using tvm::ffi::Tuple;
   using tvm::ffi::Variant;
   using tvm::tirx::TensorVar;
   using tvm::tirx::ThreadScope;
   
   TVM_TIRX_DLL FunctionFrame Function(bool is_private = false, bool persistent = false,
                                       bool is_declaration = false);
   
   TVM_TIRX_DLL Var Arg(ffi::String name, Var var);
   
   TVM_TIRX_DLL TensorVar Arg(ffi::String name, TensorVar tensor);
   
   TVM_TIRX_DLL void FuncName(ffi::String name);
   
   TVM_TIRX_DLL void FuncAttrs(ffi::Map<ffi::String, ffi::Any> attrs);
   
   TVM_TIRX_DLL Type FuncRet(Type ret_type);
   
   TVM_TIRX_DLL tvm::Stmt TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value);
   
   inline Var Handle(ffi::Optional<PrimType> dtype = std::nullopt,
                     runtime::MemoryScope storage_scope = {}) {
     Type type_annotation = dtype.has_value() ? Type(PtrType(dtype.value(), storage_scope))
                                              : Type(PtrType::VoidPtrType(storage_scope));
     return tvm::Var("", type_annotation);
   }
   
   inline Var TensorMap() { return tvm::Var("", PtrType(tvm::tirx::TensorMapType())); }
   
   }  // namespace tirx
   }  // namespace ir_builder
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_TIRX_SCRIPT_IR_BUILDER_IR_H_
