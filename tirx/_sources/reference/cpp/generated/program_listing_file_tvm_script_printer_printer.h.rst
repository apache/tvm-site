
.. _program_listing_file_tvm_script_printer_printer.h:

Program Listing for File printer.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_printer_printer.h>` (``tvm/script/printer/printer.h``)

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
   #ifndef TVM_SCRIPT_PRINTER_PRINTER_H_
   #define TVM_SCRIPT_PRINTER_PRINTER_H_
   
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/script/printer/config.h>
   
   #include <optional>
   #include <string>
   
   namespace tvm {
   
   TVM_TIRX_DLL ffi::String Script(const ffi::ObjectRef& node,
                                   const ffi::Optional<PrinterConfig>& config = std::nullopt);
   
   namespace script {
   namespace printer {
   
   TVM_TIRX_DLL void RegisterNamespaceAlias(const ffi::String& key, const ffi::String& default_alias);
   
   TVM_TIRX_DLL const ffi::Map<ffi::String, ffi::String>& GetNamespaceAliases();
   
   }  // namespace printer
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_SCRIPT_PRINTER_PRINTER_H_
