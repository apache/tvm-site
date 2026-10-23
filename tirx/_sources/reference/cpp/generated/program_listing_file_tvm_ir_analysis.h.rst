
.. _program_listing_file_tvm_ir_analysis.h:

Program Listing for File analysis.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_analysis.h>` (``tvm/ir/analysis.h``)

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
   #ifndef TVM_IR_ANALYSIS_H_
   #define TVM_IR_ANALYSIS_H_
   
   #include <tvm/ffi/extra/structural_visit.h>
   #include <tvm/ir/expr.h>
   
   #include <unordered_set>
   
   namespace tvm {
   
   inline ffi::Array<Var> UndefinedVars(ffi::AnyView object, const ffi::Array<Var>& defs = {}) {
     std::unordered_set<const VarNode*> visited_vars;
     for (const Var& var : defs) visited_vars.insert(var.get());
     ffi::Array<Var> undefined;
     ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
         object, [&](const VarNode* var, TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
           if (kind != kTVMFFIDefRegionKindNone) {
             visited_vars.insert(var);
             return ffi::WalkResult::Advance();
           }
           if (visited_vars.insert(var).second) {
             undefined.push_back(ffi::GetRef<Var>(var));
           }
           return ffi::WalkResult::Skip();
         });
     return undefined;
   }
   
   }  // namespace tvm
   #endif  // TVM_IR_ANALYSIS_H_
