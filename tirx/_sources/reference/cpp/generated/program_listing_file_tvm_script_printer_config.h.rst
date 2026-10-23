
.. _program_listing_file_tvm_script_printer_config.h:

Program Listing for File config.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_printer_config.h>` (``tvm/script/printer/config.h``)

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
   #ifndef TVM_SCRIPT_PRINTER_CONFIG_H_
   #define TVM_SCRIPT_PRINTER_CONFIG_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/dtype.h>
   #include <tvm/ffi/reflection/access_path.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ffi/string.h>
   #include <tvm/runtime/base.h>
   
   #include <string>
   
   namespace tvm {
   
   class PrinterConfigObj : public ffi::Object {
    public:
     ffi::Array<ffi::String> binding_names = {};
     bool show_meta = false;
     bool verbose_expr = false;
     int indent_spaces = 4;
     bool print_line_numbers = false;
     int num_context_lines = -1;
   
     bool render_invisible_path_info = true;
   
     /* \brief ffi::Object path to be underlined */
     ffi::Array<ffi::reflection::AccessPath> path_to_underline;
     ffi::Map<ffi::reflection::AccessPath, ffi::String> path_to_annotate;
     ffi::Array<ffi::ObjectRef> obj_to_underline = ffi::Array<ffi::ObjectRef>();
     ffi::Map<ffi::ObjectRef, ffi::String> obj_to_annotate = ffi::Map<ffi::ObjectRef, ffi::String>();
   
     ffi::Map<ffi::String, ffi::Any> extra_config;
   
     template <typename T>
     T GetExtraConfig(const ffi::String& key, T fallback) const {
       auto it = extra_config.find(key);
       if (it == extra_config.end()) return fallback;
       return (*it).second.as_or_throw<T>();
     }
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PrinterConfigObj>()
           .def_ro("binding_names", &PrinterConfigObj::binding_names)
           .def_ro("show_meta", &PrinterConfigObj::show_meta)
           .def_ro("verbose_expr", &PrinterConfigObj::verbose_expr)
           .def_ro("indent_spaces", &PrinterConfigObj::indent_spaces)
           .def_ro("print_line_numbers", &PrinterConfigObj::print_line_numbers)
           .def_ro("num_context_lines", &PrinterConfigObj::num_context_lines)
           .def_ro("render_invisible_path_info", &PrinterConfigObj::render_invisible_path_info)
           .def_ro("path_to_underline", &PrinterConfigObj::path_to_underline)
           .def_ro("path_to_annotate", &PrinterConfigObj::path_to_annotate)
           .def_ro("obj_to_underline", &PrinterConfigObj::obj_to_underline)
           .def_ro("obj_to_annotate", &PrinterConfigObj::obj_to_annotate)
           .def_ro("extra_config", &PrinterConfigObj::extra_config);
     }
   
     TVM_TIRX_DLL ffi::Array<ffi::String> GetBuiltinKeywords();
   
     static constexpr const bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.PrinterConfig", PrinterConfigObj, ffi::Object);
   };
   
   class TVM_TIRX_DLL PrinterConfig : public ffi::ObjectRef {
    public:
     explicit PrinterConfig(
         ffi::Map<ffi::String, ffi::Any> config_dict = ffi::Map<ffi::String, ffi::Any>());
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrinterConfig, ffi::ObjectRef, PrinterConfigObj);
   };
   
   TVM_TIRX_DLL ffi::String RedirectedReprPrinterMethod(const ffi::ObjectRef& obj);
   
   }  // namespace tvm
   #endif  // TVM_SCRIPT_PRINTER_CONFIG_H_
