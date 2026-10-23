
.. _program_listing_file_tvm_tirx_transform.h:

Program Listing for File transform.h
====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_transform.h>` (``tvm/tirx/transform.h``)

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
   
   #ifndef TVM_TIRX_TRANSFORM_H_
   #define TVM_TIRX_TRANSFORM_H_
   
   #include <tvm/ir/function.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/ir/transform.h>
   #include <tvm/target/target.h>
   #include <tvm/tirx/function.h>
   
   #include <string>
   #include <vector>
   
   namespace tvm {
   namespace tirx {
   namespace transform {
   
   using tvm::transform::CreateModulePass;
   using tvm::transform::Pass;
   using tvm::transform::PassContext;
   using tvm::transform::PassContextObj;
   using tvm::transform::PassInfo;
   using tvm::transform::PassInfoObj;
   using tvm::transform::PassObj;
   using tvm::transform::Sequential;
   
   TVM_TIRX_DLL Pass DecorateDeviceScope();
   
   TVM_TIRX_DLL Pass CreateFunctionPass(
       ffi::TypedFunction<ffi::Optional<Function>(ffi::RValueRef<Function>, IRModule, PassContext)>
           pass_func,
       int opt_level, ffi::String name);
   
   inline Pass CreateFunctionPass(
       std::function<ffi::Optional<Function>(Function, IRModule, PassContext)> pass_func,
       int opt_level, ffi::String name) {
     ffi::TypedFunction<ffi::Optional<Function>(ffi::RValueRef<Function>, IRModule, PassContext)>
         callback = [pass_func = std::move(pass_func)](ffi::RValueRef<Function> func, IRModule mod,
                                                       PassContext ctx) -> ffi::Optional<Function> {
       return pass_func(*std::move(func), std::move(mod), std::move(ctx));
     };
     return CreateFunctionPass(std::move(callback), opt_level, std::move(name));
   }
   
   TVM_TIRX_DLL Pass VectorizeLoop(bool enable_vectorize = true);
   
   TVM_TIRX_DLL Pass StorageRewrite();
   
   TVM_TIRX_DLL Pass UnrollLoop();
   
   TVM_TIRX_DLL Pass RemoveNoOp();
   
   TVM_TIRX_DLL Pass HoistIf();
   
   TVM_TIRX_DLL Pass RenormalizeSplitPattern();
   
   TVM_TIRX_DLL Pass StmtSimplify();
   
   TVM_TIRX_DLL Pass ConvertSSA();
   
   TVM_TIRX_DLL Pass LowerThreadAllreduce();
   
   TVM_TIRX_DLL Pass MakePackedAPI();
   
   TVM_TIRX_DLL Pass SplitHostDevice();
   
   TVM_TIRX_DLL Pass SkipAssert();
   
   TVM_TIRX_DLL Pass LowerTVMBuiltin();
   
   TVM_TIRX_DLL Pass LowerIntrin();
   
   TVM_TIRX_DLL Pass NarrowDataType(int target_bits);
   
   TVM_TIRX_DLL Pass ForceNarrowIndexToInt32();
   
   TVM_TIRX_DLL Pass BF16ComputeLegalize();
   
   TVM_TIRX_DLL Pass FP8ComputeLegalize(ffi::String promote_dtype = "float16");
   
   TVM_TIRX_DLL Pass BF16StorageLegalize();
   
   TVM_TIRX_DLL Pass FP8StorageLegalize();
   
   TVM_TIRX_DLL Pass PointerValueTypeRewrite();
   
   TVM_TIRX_DLL Pass FlattenTensor();
   
   TVM_TIRX_DLL Pass CommonSubexprElim();
   
   TVM_TIRX_DLL Pass BindTarget(Target target);
   
   TVM_TIRX_DLL Pass AnnotateEntryFunc();
   
   TVM_TIRX_DLL Pass Filter(ffi::TypedFunction<bool(Function)> fcond);
   
   TVM_TIRX_DLL Pass TileDispatch();
   
   TVM_TIRX_DLL Pass LowerTensorLayouts();
   
   }  // namespace transform
   }  // namespace tirx
   }  // namespace tvm
   
   #endif  // TVM_TIRX_TRANSFORM_H_
