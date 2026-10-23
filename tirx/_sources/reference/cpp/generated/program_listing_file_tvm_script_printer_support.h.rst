
.. _program_listing_file_tvm_script_printer_support.h:

Program Listing for File support.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_printer_support.h>` (``tvm/script/printer/support.h``)

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
   #ifndef TVM_SCRIPT_PRINTER_SUPPORT_H_
   #define TVM_SCRIPT_PRINTER_SUPPORT_H_
   
   #include <tvm/ffi/extra/dataclass.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/expr.h>
   #include <tvm/script/printer/config.h>
   #include <tvm/script/printer/doc_translator.h>
   
   namespace tvm {
   namespace script {
   namespace printer {
   
   template <typename ObjectType>
   void RegisterScriptRepr() {
     namespace refl = ffi::reflection;
     refl::TypeAttrDef<ObjectType>().def(ffi::type_attr::kRepr.name,
                                         [](ffi::ObjectRef obj, ffi::Function) -> ffi::String {
                                           return RedirectedReprPrinterMethod(obj);
                                         });
   }
   
   TVM_TIRX_DLL IdDoc VarDoc(DocTranslatorObj* d, const Var& var, bool explicit_def = true);
   
   TVM_TIRX_DLL ffi::Optional<ExprDoc> VarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                       const ffi::Object* destination);
   
   TVM_TIRX_DLL ExprDoc AddMetadata(DocTranslatorObj* d, ffi::Any value);
   
   TVM_TIRX_DLL ExprDoc
   RawCallDocTranslate(DocTranslatorObj* d, const CallNode* call,
                       ffi::Optional<ffi::Array<ExprDoc>> translated_args = std::nullopt);
   
   TVM_TIRX_DLL ffi::Optional<ExprDoc> StandardCallDocTranslate(DocTranslatorObj* d,
                                                                const CallNode* call);
   
   }  // namespace printer
   }  // namespace script
   }  // namespace tvm
   #endif  // TVM_SCRIPT_PRINTER_SUPPORT_H_
