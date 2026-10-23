
.. _program_listing_file_tvm_tirx_stmt.h:

Program Listing for File stmt.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_stmt.h>` (``tvm/tirx/stmt.h``)

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
   #ifndef TVM_TIRX_STMT_H_
   #define TVM_TIRX_STMT_H_
   
   #include <tvm/ir/stmt.h>
   #include <tvm/tirx/expr.h>
   #include <tvm/tirx/tile/thread_scope.h>
   
   namespace tvm {
   namespace tirx {
   
   inline bool IsNoOp(const Stmt& stmt) {
     if (!stmt.defined()) return true;
     if (const auto* op = stmt.as<EvaluateNode>()) return op->value.as<IntImmNode>() != nullptr;
     if (const auto* op = stmt.as<SeqStmtNode>()) return op->seq.empty();
     return false;
   }
   
   namespace attr {
   constexpr const char* kPragmaUnroll = "tirx.pragma_unroll";
   
   inline constexpr ffi::reflection::AttrKey<bool> kDisableUnroll{"tirx.disable_unroll"};
   
   inline constexpr ffi::reflection::AttrKey<int64_t> kAutoUnrollMaxStep{"tirx.auto_unroll_max_step"};
   inline constexpr ffi::reflection::AttrKey<bool> kUnrollExplicit{"tirx.unroll_explicit"};
   inline constexpr ffi::reflection::AttrKey<bool> kVolatile{"tirx.volatile"};
   
   }  // namespace attr
   
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_STMT_H_
