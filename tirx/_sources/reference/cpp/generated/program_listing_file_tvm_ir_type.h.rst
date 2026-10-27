
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
   
   class TypeObj : public ffi::Object {
    public:
     mutable Location loc = UnknownLoc();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       // loc do not participate in structural equal and hash.
       refl::ObjectDef<TypeObj>().def_ro("loc", &TypeObj::loc, refl::DefaultValue(UnknownLoc()),
                                         refl::init(false), refl::AttachFieldFlag::SEqHashIgnore());
     }
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;
   
     static constexpr const uint32_t _type_child_slots = 14;
     TVM_FFI_DECLARE_OBJECT_INFO("ir.Type", TypeObj, ffi::Object);
   };
   
   class Type : public ffi::ObjectRef {
    public:
     TVM_TIRX_DLL static Type Missing();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Type, ffi::ObjectRef, TypeObj);
   };
   
   class MissingTypeObj final : public TypeObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<MissingTypeObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.MissingType", MissingTypeObj, TypeObj);
   };
   
   class MissingType final : public Type {
    public:
     TVM_TIRX_DLL MissingType();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MissingType, Type, MissingTypeObj);
   };
   
   class OpaqueTypeObj final : public TypeObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<OpaqueTypeObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.OpaqueType", OpaqueTypeObj, TypeObj);
   };
   
   class OpaqueType final : public Type {
    public:
     TVM_TIRX_DLL OpaqueType();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OpaqueType, Type, OpaqueTypeObj);
   };
   
   class PrimTypeObj final : public TypeObj {
    public:
     DLDataType dtype;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PrimTypeObj>().def_ro("dtype", &PrimTypeObj::dtype);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PrimType", PrimTypeObj, TypeObj);
   };
   
   /*
    * \brief Managed reference to PrimTypeObj.
    * \sa PrimTypeObj
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
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimType, Type, PrimTypeObj);
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
   
   class NoneTypeObj : public TypeObj {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<NoneTypeObj>(); }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.NoneType", NoneTypeObj, TypeObj);
   };
   
   class NoneType : public Type {
    public:
     TVM_TIRX_DLL NoneType();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NoneType, Type, NoneTypeObj);
   };
   
   class AnyTypeObj : public TypeObj {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<AnyTypeObj>(); }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.AnyType", AnyTypeObj, TypeObj);
   };
   
   class AnyType : public Type {
    public:
     TVM_TIRX_DLL AnyType(Location loc = UnknownLoc());
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AnyType, Type, AnyTypeObj);
   };
   
   class TensorRegionTypeObj : public TypeObj {
    public:
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TensorRegionTypeObj>();
     }
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorRegionType", TensorRegionTypeObj, TypeObj);
   };
   
   class TensorRegionType : public Type {
    public:
     TVM_TIRX_DLL TensorRegionType();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRegionType, Type, TensorRegionTypeObj);
   };
   
   class StringTypeObj : public TypeObj {
    public:
     static void RegisterReflection() { ffi::reflection::ObjectDef<StringTypeObj>(); }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.StringType", StringTypeObj, TypeObj);
   };
   
   class StringType : public Type {
    public:
     TVM_TIRX_DLL StringType();
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StringType, Type, StringTypeObj);
   };
   
   class PtrTypeObj : public TypeObj {
    public:
     Type element_type = PrimType::Void();
     runtime::MemoryScope scope;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<PtrTypeObj>()
           .def_ro("element_type", &PtrTypeObj::element_type)
           .def_ro("scope", &PtrTypeObj::scope, refl::DefaultValue(runtime::MemoryScope::Global()));
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PtrType", PtrTypeObj, TypeObj);
   };
   
   /*
    * \brief Managed reference to PtrTypeObj.
    * \sa PtrTypeObj
    */
   class PtrType : public Type {
    public:
     TVM_TIRX_DLL explicit PtrType(Type element_type, runtime::MemoryScope scope = {});
   
     TVM_TIRX_DLL static PtrType VoidPtrType(runtime::MemoryScope scope = {});
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PtrType, Type, PtrTypeObj);
   };
   
   class TupleTypeObj : public TypeObj {
    public:
     ffi::Array<Type> fields;
   
     TupleTypeObj() {}
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<TupleTypeObj>().def_ro("fields", &TupleTypeObj::fields);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleType", TupleTypeObj, TypeObj);
   };
   
   class TupleType : public Type {
    public:
     TVM_TIRX_DLL explicit TupleType(ffi::Array<Type> fields, Location loc = UnknownLoc());
   
     TVM_TIRX_DLL TupleType static Empty();
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleType, Type, TupleTypeObj);
   };
   
   inline Type VoidType() { return TupleType::Empty(); }
   
   inline bool IsVoidType(const Type& type) {
     auto* n = type.as<TupleTypeObj>();
     return n && n->fields.size() == 0;
   }
   
   class FuncTypeObj : public TypeObj {
    public:
     ffi::Array<Type> arg_types;
     Type ret_type = VoidType();
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FuncTypeObj>()
           .def_ro("arg_types", &FuncTypeObj::arg_types)
           .def_ro("ret_type", &FuncTypeObj::ret_type);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FuncType", FuncTypeObj, TypeObj);
   };
   
   class FuncType : public Type {
    public:
     TVM_TIRX_DLL FuncType(ffi::Array<Type> arg_types, Type ret_type, Location loc = UnknownLoc());
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FuncType, Type, FuncTypeObj);
   };
   
   }  // namespace tvm
   #endif  // TVM_IR_TYPE_H_
