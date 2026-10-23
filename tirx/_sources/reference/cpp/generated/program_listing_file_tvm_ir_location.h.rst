
.. _program_listing_file_tvm_ir_location.h:

Program Listing for File location.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_location.h>` (``tvm/ir/location.h``)

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
   #ifndef TVM_IR_LOCATION_H_
   #define TVM_IR_LOCATION_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/runtime/base.h>
   
   #include <fstream>
   #include <string>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   
   class SourceName;
   class SourceNameNode : public ffi::Object {
    public:
     ffi::String name;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<SourceNameNode>().def_ro("name", &SourceNameNode::name);
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceName", SourceNameNode, ffi::Object);
   };
   
   class SourceName : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL static SourceName Get(const ffi::String& name);
   
     explicit SourceName(ffi::UnsafeInit tag, ffi::ObjectPtr<SourceNameNode> n) : ffi::ObjectRef(tag) {
       data_ = std::move(n);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceName, ffi::ObjectRef, SourceNameNode);
   };
   
   class LocationNode : public ffi::Object {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<LocationNode>(); }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Location", LocationNode, ffi::Object);
   };
   
   class Location : public ffi::ObjectRef {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Location, ffi::ObjectRef, LocationNode);
   };
   
   class UnknownLocNode : public LocationNode {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<UnknownLocNode>(); }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.UnknownLoc", UnknownLocNode, LocationNode);
   };
   
   class UnknownLoc : public Location {
    public:
     TVM_TIRX_DLL UnknownLoc();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(UnknownLoc, Location, UnknownLocNode);
   };
   
   class SourceLocNode : public LocationNode {
    public:
     explicit SourceLocNode(SourceName source_name) : source_name(std::move(source_name)) {}
     explicit SourceLocNode(ffi::UnsafeInit tag) : source_name(tag) {}
     SourceName source_name;
     int start_line;
     int start_column;
     int end_line;
     int end_column;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<SourceLocNode>()
           .def_ro("source_name", &SourceLocNode::source_name)
           .def_ro("start_line", &SourceLocNode::start_line)
           .def_ro("start_column", &SourceLocNode::start_column)
           .def_ro("end_line", &SourceLocNode::end_line)
           .def_ro("end_column", &SourceLocNode::end_column);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceLoc", SourceLocNode, LocationNode);
   };
   
   class SourceLoc : public Location {
    public:
     TVM_TIRX_DLL SourceLoc(SourceName source_name, int start_line, int start_column, int end_line,
                            int end_column);
   
     TVM_TIRX_DLL SourceLoc Merge(const SourceLoc& other) const;
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceLoc, Location, SourceLocNode);
   };
   
   class CallSiteLocNode : public LocationNode {
    public:
     Location callee = UnknownLoc();
     Location caller = UnknownLoc();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<CallSiteLocNode>()
           .def_ro("callee", &CallSiteLocNode::callee)
           .def_ro("caller", &CallSiteLocNode::caller);
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.CallSiteLoc", CallSiteLocNode, LocationNode);
   };
   
   class CallSiteLoc : public Location {
    public:
     TVM_TIRX_DLL CallSiteLoc(Location callee, Location caller);
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CallSiteLoc, Location, CallSiteLocNode);
   };
   
   }  // namespace tvm
   
   #endif  // TVM_IR_LOCATION_H_
