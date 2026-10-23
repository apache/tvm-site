
.. _program_listing_file_tvm_tirx_analysis.h:

Program Listing for File analysis.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_analysis.h>` (``tvm/tirx/analysis.h``)

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
   
   #ifndef TVM_TIRX_ANALYSIS_H_
   #define TVM_TIRX_ANALYSIS_H_
   
   #include <tvm/ir/module.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/ir/transform.h>
   #include <tvm/target/target.h>
   #include <tvm/tirx/function.h>
   #include <tvm/tirx/op/attr.h>
   #include <tvm/tirx/stmt.h>
   
   #include <string>
   
   namespace tvm {
   
   namespace tirx {
   
   TVM_TIRX_DLL ffi::Array<PrimExpr> DetectLinearEquation(const PrimExpr& e,
                                                          const ffi::Array<PrimVar>& vars);
   
   TVM_TIRX_DLL ffi::Array<ffi::Optional<PrimExpr>> DetectClipBound(const PrimExpr& e,
                                                                    const ffi::Array<PrimVar>& vars);
   
   TVM_TIRX_DLL bool VerifyMemory(const Function& func);
   
   TVM_TIRX_DLL size_t CalculateExprComplexity(const PrimExpr& expr);
   
   TVM_TIRX_DLL bool CheckWellFormed(const Function& func, bool assert_mode = true);
   
   TVM_TIRX_DLL bool CheckWellFormed(const IRModule& mod, bool assert_mode = true);
   
   TVM_TIRX_DLL const FunctionNode* FindEntryFunc(const IRModule& mod, GlobalVar* result_g_var);
   
   // Pass variants of verification analysis
   // directly throws RuntimeError when verification fails.
   namespace transform {
   
   using tvm::transform::Pass;
   using tvm::transform::PassContext;
   
   TVM_TIRX_DLL Pass VerifyMemory();
   
   }  // namespace transform
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_ANALYSIS_H_
