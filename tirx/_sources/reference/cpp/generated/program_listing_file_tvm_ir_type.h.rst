
.. _program_listing_file_tvm_ir_type.h:

Program Listing for File type.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_type.h>` (``tvm/ir/type.h``)

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
   
   #ifndef TVM_IR_TYPE_H_
   #define TVM_IR_TYPE_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/dtype.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ir/location.h>
   #include <tvm/runtime/memory_scope.h>
   
   #include <string>
   
   namespace tvm {
   
   class TypeNode : public ffi::Object {
    public:
     mutable Location loc = UnknownLoc();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       // loc do not participate in structural equal and hash.
       refl::ObjectDef<TypeNode>().def_ro("loc", &TypeNode::loc, refl::DefaultValue(UnknownLoc()),
                                          refl::init(false), refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;
   
     static constexpr const uint32_t _type_child_slots = 14;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Type", TypeNode, ffi::Object);
   };
   
   class Type : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL static Type Missing();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Type, ffi::ObjectRef, TypeNode);
   };
   
   class MissingTypeNode final : public TypeNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<MissingTypeNode>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.MissingType", MissingTypeNode, TypeNode);
   };
   
   class MissingType final : public Type {
    public:
     TVM_TIRX_DLL MissingType();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MissingType, Type, MissingTypeNode);
   };
   
   class OpaqueTypeNode final : public TypeNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<OpaqueTypeNode>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.OpaqueType", OpaqueTypeNode, TypeNode);
   };
   
   class OpaqueType final : public Type {
    public:
     TVM_TIRX_DLL OpaqueType();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OpaqueType, Type, OpaqueTypeNode);
   };
   
   class PrimTypeNode final : public TypeNode {
    public:
     DLDataType dtype;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PrimTypeNode>().def_ro("dtype", &PrimTypeNode::dtype);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PrimType", PrimTypeNode, TypeNode);
   };
   
   /*
    * \brief Managed reference to PrimTypeNode.
    * \sa PrimTypeNode
    */
   class PrimType final : public Type {
    public:
     TVM_TIRX_DLL explicit PrimType(DLDataType dtype);
   
     TVM_TIRX_DLL PrimType(DLDataTypeCode code, int bits, int lanes = 1);
   
     TVM_TIRX_DLL static PrimType Int(int bits, int lanes = 1);
     TVM_TIRX_DLL static PrimType UInt(int bits, int lanes = 1);
     TVM_TIRX_DLL static PrimType Float(int bits, int lanes = 1);
     TVM_TIRX_DLL static PrimType BFloat(int bits, int lanes = 1);
     TVM_TIRX_DLL static PrimType Bool(int lanes = 1);
     TVM_TIRX_DLL static PrimType Void();
     TVM_TIRX_DLL static PrimType ScalableVector(DLDataTypeCode code, int bits, int lanes);
   
     TVM_FFI_INLINE DLDataTypeCode code() const {
       return static_cast<DLDataTypeCode>(static_cast<int>(get()->dtype.code));
     }
   
     TVM_FFI_INLINE int32_t bits() const { return get()->dtype.bits; }
   
     TVM_FFI_INLINE int32_t lanes() const {
       int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
       if (TVM_FFI_PREDICT_FALSE(encoded_lanes < 0)) {
         TVM_FFI_THROW(InternalError)
             << "Can't fetch the lanes of a scalable vector at a compile time.";
       }
       return encoded_lanes;
     }
   
     TVM_FFI_INLINE bool MatchesElementType(DLDataTypeCode code, int bits) const {
       DLDataType dtype = get()->dtype;
       return dtype.code == static_cast<uint8_t>(code) && dtype.bits == bits;
     }
   
     template <typename... Codes>
     TVM_FFI_INLINE bool MatchesCode(Codes... codes) const {
       uint8_t dtype_code = get()->dtype.code;
       return ((dtype_code == static_cast<uint8_t>(codes)) || ...);
     }
   
     TVM_FFI_INLINE bool IsScalar() const {
       int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
       return encoded_lanes == 1;
     }
   
     TVM_FFI_INLINE bool IsVoid() const {
       DLDataType dtype = get()->dtype;
       return dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLOpaqueHandle) && dtype.bits == 0 &&
              static_cast<int16_t>(dtype.lanes) == 0;
     }
   
     TVM_FFI_INLINE bool IsScalableVector() const {
       return static_cast<int16_t>(get()->dtype.lanes) < -1;
     }
   
     TVM_FFI_INLINE bool IsFixedLengthVector() const {
       return static_cast<int16_t>(get()->dtype.lanes) > 1;
     }
   
     TVM_FFI_INLINE size_t StorageBytes() const {
       DLDataType dtype = get()->dtype;
       int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
       if (TVM_FFI_PREDICT_FALSE(encoded_lanes < 0)) {
         TVM_FFI_THROW(InternalError)
             << "Cannot compute compile-time storage bytes for non-fixed vector type " << dtype;
       }
       return static_cast<size_t>(
           (static_cast<uint64_t>(dtype.bits) * static_cast<uint64_t>(dtype.lanes) + 7) / 8);
     }
   
     TVM_FFI_INLINE PrimType WithCode(DLDataTypeCode code) const {
       DLDataType dtype = get()->dtype;
       int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
       if (encoded_lanes < -1) {
         return ScalableVector(code, dtype.bits, -encoded_lanes);
       }
       return PrimType(code, dtype.bits, encoded_lanes);
     }
   
     TVM_FFI_INLINE PrimType WithBits(int bits) const {
       DLDataType dtype = get()->dtype;
       int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
       if (encoded_lanes < -1) {
         return ScalableVector(this->code(), bits, -encoded_lanes);
       }
       return PrimType(this->code(), bits, encoded_lanes);
     }
   
     TVM_FFI_INLINE PrimType WithLanes(int lanes) const {
       return PrimType(this->code(), this->bits(), lanes);
     }
   
     TVM_FFI_INLINE int32_t VScaleFactor() const {
       int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
       if (encoded_lanes >= -1) {
         TVM_FFI_THROW(InternalError) << "A fixed length vector doesn't have a vscale factor.";
       }
       return -encoded_lanes;
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimType, Type, PrimTypeNode);
   };
   
   inline bool operator==(const PrimType& lhs, const PrimType& rhs) {
     return lhs->dtype == rhs->dtype;
   }
   
   inline bool operator!=(const PrimType& lhs, const PrimType& rhs) { return !(lhs == rhs); }
   
   namespace ffi {
   template <>
   inline constexpr bool use_default_type_traits_v<PrimType> = false;
   
   template <>
   struct TypeTraits<PrimType> : public ObjectRefWithFallbackTraitsBase<PrimType, DLDataType> {
     TVM_FFI_INLINE static PrimType ConvertFallbackValue(DLDataType dtype) { return PrimType(dtype); }
   };
   
   }  // namespace ffi
   
   class NoneTypeNode : public TypeNode {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<NoneTypeNode>(); }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.NoneType", NoneTypeNode, TypeNode);
   };
   
   class NoneType : public Type {
    public:
     TVM_TIRX_DLL NoneType();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NoneType, Type, NoneTypeNode);
   };
   
   class AnyTypeNode : public TypeNode {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<AnyTypeNode>(); }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.AnyType", AnyTypeNode, TypeNode);
   };
   
   class AnyType : public Type {
    public:
     TVM_TIRX_DLL AnyType(Location loc = UnknownLoc());
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AnyType, Type, AnyTypeNode);
   };
   
   class TensorRegionTypeNode : public TypeNode {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorRegionTypeNode>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorRegionType", TensorRegionTypeNode, TypeNode);
   };
   
   class TensorRegionType : public Type {
    public:
     TVM_TIRX_DLL TensorRegionType();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRegionType, Type, TensorRegionTypeNode);
   };
   
   class StringTypeNode : public TypeNode {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<StringTypeNode>(); }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.StringType", StringTypeNode, TypeNode);
   };
   
   class StringType : public Type {
    public:
     TVM_TIRX_DLL StringType();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StringType, Type, StringTypeNode);
   };
   
   class PtrTypeNode : public TypeNode {
    public:
     Type element_type = PrimType::Void();
     runtime::MemoryScope scope;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PtrTypeNode>()
           .def_ro("element_type", &PtrTypeNode::element_type)
           .def_ro("scope", &PtrTypeNode::scope, refl::DefaultValue(runtime::MemoryScope::Global()));
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PtrType", PtrTypeNode, TypeNode);
   };
   
   /*
    * \brief Managed reference to PtrTypeNode.
    * \sa PtrTypeNode
    */
   class PtrType : public Type {
    public:
     TVM_TIRX_DLL explicit PtrType(Type element_type, runtime::MemoryScope scope = {});
   
     TVM_TIRX_DLL static PtrType VoidPtrType(runtime::MemoryScope scope = {});
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PtrType, Type, PtrTypeNode);
   };
   
   class TupleTypeNode : public TypeNode {
    public:
     ffi::Array<Type> fields;
   
     TupleTypeNode() {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleTypeNode>().def_ro("fields", &TupleTypeNode::fields);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleType", TupleTypeNode, TypeNode);
   };
   
   class TupleType : public Type {
    public:
     TVM_TIRX_DLL explicit TupleType(ffi::Array<Type> fields, Location loc = UnknownLoc());
   
     TVM_TIRX_DLL TupleType static Empty();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleType, Type, TupleTypeNode);
   };
   
   inline Type VoidType() { return TupleType::Empty(); }
   
   inline bool IsVoidType(const Type& type) {
     auto* n = type.as<TupleTypeNode>();
     return n && n->fields.size() == 0;
   }
   
   class FuncTypeNode : public TypeNode {
    public:
     ffi::Array<Type> arg_types;
     Type ret_type = VoidType();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FuncTypeNode>()
           .def_ro("arg_types", &FuncTypeNode::arg_types)
           .def_ro("ret_type", &FuncTypeNode::ret_type);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FuncType", FuncTypeNode, TypeNode);
   };
   
   class FuncType : public Type {
    public:
     TVM_TIRX_DLL FuncType(ffi::Array<Type> arg_types, Type ret_type, Location loc = UnknownLoc());
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FuncType, Type, FuncTypeNode);
   };
   
   }  // namespace tvm
   #endif  // TVM_IR_TYPE_H_
