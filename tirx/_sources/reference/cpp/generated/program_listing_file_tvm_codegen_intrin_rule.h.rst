
.. _program_listing_file_tvm_codegen_intrin_rule.h:

Program Listing for File intrin_rule.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_codegen_intrin_rule.h>` (``tvm/codegen/intrin_rule.h``)

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
   
   #ifndef TVM_CODEGEN_INTRIN_RULE_H_
   #define TVM_CODEGEN_INTRIN_RULE_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/dtype.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/op.h>
   #include <tvm/runtime/base.h>
   #include <tvm/tirx/op/abi.h>
   
   #include <string>
   
   namespace tvm {
   namespace codegen {
   namespace intrin {
   
   struct FloatSuffix {
     std::string operator()(const PrimType& ty, std::string name) const {
       if (ty->dtype == DLDataType{kDLFloat, 32, 1}) {
         return name + 'f';
       } else if (ty->dtype == DLDataType{kDLFloat, 64, 1}) {
         return name;
       } else {
         return "";
       }
     }
   };
   
   struct Direct {
     std::string operator()(const PrimType& ty, std::string name) const { return name; }
   };
   
   template <typename T, bool dtype_from_arg = false>
   inline PrimExpr DispatchPureExtern(const PrimExpr& e) {
     const CallNode* call = e.as<CallNode>();
     TVM_FFI_ICHECK(call != nullptr);
     // Primitive intrinsic names use the prim. namespace prefix.
     const OpNode* op = call->op.as<OpNode>();
     TVM_FFI_ICHECK(op != nullptr);
     std::string name = op->name;
     TVM_FFI_ICHECK(name.substr(0, 5) == "prim.") << "Unexpected intrinsic name: " << name;
     if (dtype_from_arg) {
       TVM_FFI_ICHECK_EQ(call->args.size(), 1U);
     }
     PrimType type = dtype_from_arg ? call->args[0].as_or_throw<PrimExpr>().ty()
                                    : call->ty.as_or_throw<PrimType>();
     name = T()(type, name.substr(5));
   
     if (!name.empty()) {
       ffi::Array<Expr> new_args = {StringImm(name)};
       for (const PrimExpr& arg : call->args.as_or_throw<ffi::Array<PrimExpr>>()) {
         new_args.push_back(arg);
       }
       return Call(e.ty(), tirx::call_pure_extern_op(), new_args).as_or_throw<PrimExpr>();
     } else {
       return e;
     }
   }
   
   TVM_TIRX_DLL PrimExpr DispatchFastErf(const PrimExpr& e);
   
   TVM_TIRX_DLL PrimExpr DispatchNumericallyStableTanh(const PrimExpr& e);
   
   }  // namespace intrin
   }  // namespace codegen
   }  // namespace tvm
   #endif  // TVM_CODEGEN_INTRIN_RULE_H_
