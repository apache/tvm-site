
.. _program_listing_file_tvm_codegen_codegen.h:

Program Listing for File codegen.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_codegen_codegen.h>` (``tvm/codegen/codegen.h``)

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
   
   #ifndef TVM_CODEGEN_CODEGEN_H_
   #define TVM_CODEGEN_CODEGEN_H_
   
   #include <tvm/ffi/extra/module.h>
   #include <tvm/ir/module.h>
   #include <tvm/runtime/base.h>
   #include <tvm/target/target.h>
   
   #include <string>
   
   namespace tvm {
   namespace codegen {
   
   TVM_TIRX_DLL ffi::Module Build(IRModule mod, Target target);
   
   TVM_TIRX_DLL ffi::String PackImportsToC(const ffi::Module& m, bool system_lib,
                                           const ffi::String& c_symbol_prefix = "");
   
   TVM_TIRX_DLL void PackImportsToObjectFile(const ffi::Module& mod, bool system_lib,
                                             const ffi::String& llvm_target_string,
                                             const ffi::String& file_name,
                                             const ffi::String& c_symbol_prefix = "");
   
   namespace attr {
   inline constexpr ffi::reflection::AttrKey<bool> kParallelStridePattern{
       "codegen.parallel_stride_pattern"};
   }  // namespace attr
   
   }  // namespace codegen
   }  // namespace tvm
   #endif  // TVM_CODEGEN_CODEGEN_H_
