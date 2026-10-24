
.. _program_listing_file_tvm_script_ir_builder_base.h:

Program Listing for File base.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_ir_builder_base.h>` (``tvm/script/ir_builder/base.h``)

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
   #ifndef TVM_SCRIPT_IR_BUILDER_BASE_H_
   #define TVM_SCRIPT_IR_BUILDER_BASE_H_
   
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/function.h>
   #include <tvm/ir/object_functor.h>
   
   #include <cstdint>
   #include <vector>
   
   namespace tvm {
   namespace script {
   namespace ir_builder {
   
   
   class TVM_TIRX_DLL IRBuilderFrameObj : public ffi::Object {
    public:
     IRBuilderFrameObj();
     std::vector<ffi::TypedFunction<void()>> callbacks;
     mutable Location loc = UnknownLoc();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IRBuilderFrameObj>().def_ro("loc", &IRBuilderFrameObj::loc);
       // `callbacks` is not registered as it's not visited.
     }
   
     static constexpr const bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.IRBuilderFrame", IRBuilderFrameObj, ffi::Object);
   
    public:
     virtual ~IRBuilderFrameObj() = default;
     virtual void EnterWithScope();
     virtual void ExitWithScope();
     void AddCallback(ffi::TypedFunction<void()> callback);
   };
   
   class IRBuilderFrame : public ffi::ObjectRef {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IRBuilderFrame, ffi::ObjectRef, IRBuilderFrameObj);
   
    protected:
     IRBuilderFrame() = default;
     explicit IRBuilderFrame(ffi::ObjectPtr<IRBuilderFrameObj> data) : ffi::ObjectRef(data) {}
   
    public:
     inline void EnterWithScope() {
       TVM_FFI_ICHECK(data_ != nullptr);
       static_cast<IRBuilderFrameObj*>(data_.get())->EnterWithScope();
     }
     inline void ExitWithScope() {
       TVM_FFI_ICHECK(data_ != nullptr);
       static_cast<IRBuilderFrameObj*>(data_.get())->ExitWithScope();
       data_.reset();
     }
   };
   
   
   class IRBuilderObj : public ffi::Object {
    public:
     ffi::Array<IRBuilderFrame> frames;
     ffi::Optional<ffi::ObjectRef> result;
     std::vector<Location> locs;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<IRBuilderObj>()
           .def_ro("frames", &IRBuilderObj::frames)
           .def_ro("result", &IRBuilderObj::result);
     }
   
     static constexpr const bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.IRBuilder", IRBuilderObj, ffi::Object);
   
    public:
     template <typename TFrame>
     inline ffi::Optional<TFrame> FindFrame() const;
     template <typename TFrame>
     inline ffi::Optional<TFrame> GetLastFrame() const;
     template <typename TObjectRef>
     inline TObjectRef Get() const;
     int64_t PushLoc(Location loc) {
       locs.push_back(std::move(loc));
       return static_cast<int64_t>(locs.size()) - 1;
     }
     void UpdateLoc(int64_t index, Location loc) {
       TVM_FFI_CHECK(index >= 0 && static_cast<size_t>(index) < locs.size(), ValueError)
           << "Invalid source location scope";
       locs[index] = std::move(loc);
     }
     void PopLoc() {
       TVM_FFI_CHECK(!locs.empty(), ValueError)
           << "ValueError: No source location exists in the builder scope";
       locs.pop_back();
     }
     TVM_TIRX_DLL Location ResolveLoc(Location location = UnknownLoc()) const;
     TVM_TIRX_DLL ffi::ObjectRef AttachLoc(ffi::ObjectRef obj, Location loc = UnknownLoc()) const;
   };
   
   class IRBuilder : public ffi::ObjectRef {
    public:
     IRBuilder() { data_ = ffi::make_object<IRBuilderObj>(); }
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IRBuilder, ffi::ObjectRef, IRBuilderObj);
   
    public:
     TVM_TIRX_DLL void EnterWithScope();
     TVM_TIRX_DLL void ExitWithScope();
     TVM_TIRX_DLL static IRBuilder Current();
     TVM_TIRX_DLL static bool IsInScope();
     template <class TObjectRef>
     inline static TObjectRef Name(ffi::String name, TObjectRef obj);
   };
   
   
   namespace detail {
   
   class LocationAccessor {
    public:
     using FType = ObjectFunctor<Location*(const ffi::ObjectRef&)>;
     TVM_TIRX_DLL static FType& vtable();
   };
   
   class Namer {
    public:
     using FType = ObjectFunctor<void(const ffi::ObjectRef&, ffi::String)>;
     TVM_TIRX_DLL static FType& vtable();
     TVM_TIRX_DLL static void Name(ffi::ObjectRef node, ffi::String name);
   };
   
   }  // namespace detail
   
   template <class TObjectRef>
   inline TObjectRef IRBuilder::Name(ffi::String name, TObjectRef obj) {
     detail::Namer::Name(obj, name);
     return obj.template as_or_throw<TObjectRef>();
   }
   
   template <typename TFrame>
   inline ffi::Optional<TFrame> IRBuilderObj::FindFrame() const {
     using TFrameNode = typename TFrame::ContainerType;
     for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
       if (const TFrameNode* p = (*it).template as<TFrameNode>()) {
         return ffi::GetRef<TFrame>(p);
       }
     }
     return std::nullopt;
   }
   
   template <typename TFrame>
   inline ffi::Optional<TFrame> IRBuilderObj::GetLastFrame() const {
     using TFrameNode = typename TFrame::ContainerType;
     if (!frames.empty() && frames.back()->IsInstance<TFrameNode>()) {
       return frames.back().template as_or_throw<TFrame>();
     }
     return std::nullopt;
   }
   
   template <typename TObjectRef>
   inline TObjectRef IRBuilderObj::Get() const {
     using TObject = typename TObjectRef::ContainerType;
     TVM_FFI_CHECK(result.has_value(), IndexError) << "No result exists in IRBuilder yet";
     const auto* n = result.as<TObject>();
     TVM_FFI_CHECK(n != nullptr, TypeError)
         << "IRBuilder result is not of type: " << TObject::_type_key;
     return ffi::GetRef<TObjectRef>(n);
   }
   
   }  // namespace ir_builder
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_SCRIPT_IR_BUILDER_BASE_H_
