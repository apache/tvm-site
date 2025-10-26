
.. _program_listing_file_tvm_ffi_c_api.h:

Program Listing for File c_api.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_c_api.h>` (``tvm/ffi/c_api.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   v0 = AddNode(x, 1)
   v1 = AddNode(x, 1)
   v2 = AddNode(v0, v0)
   v3 = AddNode(v1, v0)
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
   // NOLINTBEGIN(modernize-use-using,bugprone-reserved-identifier,modernize-deprecated-headers)
   /*
    * \file tvm/ffi/c_api.h
    * \brief This file defines the C convention of the FFI convention
    */
   #ifndef TVM_FFI_C_API_H_
   #define TVM_FFI_C_API_H_
   
   #include <dlpack/dlpack.h>
   #include <stdint.h>
   
   // Macros to do weak linking
   #ifdef _MSC_VER
   #define TVM_FFI_WEAK __declspec(selectany)
   #else
   #define TVM_FFI_WEAK __attribute__((weak))
   #endif
   
   // Defines two macros
   // TVM_FFI_DLL: marks the function as a DLL export/import
   //              depending on whether TVM_FFI_EXPORTS is defined
   // TVM_FFI_DLL_EXPORT: always marks the function as a DLL export
   #if !defined(TVM_FFI_DLL) && defined(__EMSCRIPTEN__)
   #include <emscripten/emscripten.h>
   #define TVM_FFI_DLL EMSCRIPTEN_KEEPALIVE
   #define TVM_FFI_DLL_EXPORT EMSCRIPTEN_KEEPALIVE
   #endif
   #if !defined(TVM_FFI_DLL) && defined(_MSC_VER)
   #ifdef TVM_FFI_EXPORTS
   #define TVM_FFI_DLL __declspec(dllexport)
   #else
   #define TVM_FFI_DLL __declspec(dllimport)
   #endif
   #define TVM_FFI_DLL_EXPORT __declspec(dllexport)
   #endif
   #ifndef TVM_FFI_DLL
   #define TVM_FFI_DLL __attribute__((visibility("default")))
   #define TVM_FFI_DLL_EXPORT __attribute__((visibility("default")))
   #endif
   
   #define TVM_FFI_VERSION_MAJOR 0
   #define TVM_FFI_VERSION_MINOR 1
   #define TVM_FFI_VERSION_PATCH 1
   
   #ifdef __cplusplus
   extern "C" {
   #endif
   
   typedef struct {
     uint32_t major;
     uint32_t minor;
     uint32_t patch;
   } TVMFFIVersion;
   
   #ifdef __cplusplus
   enum TVMFFITypeIndex : int32_t {
   #else
   typedef enum {
   #endif
     /*
      * \brief The root type of all FFI objects.
      *
      * We include it so TypeIndex captures all possible runtime values.
      * `kTVMFFIAny` code will never appear in Any::type_index.
      * However, it may appear in field annotations during reflection.
      */
     kTVMFFIAny = -1,
     // [Section] On-stack POD and special types: [0, kTVMFFIStaticObjectBegin)
     // N.B. `kTVMFFIRawStr` is a string backed by a `\0`-terminated char array,
     // which is not owned by TVMFFIAny. It is required that the following
     // invariant holds:
     // - `Any::type_index` is never `kTVMFFIRawStr`
     // - `AnyView::type_index` can be `kTVMFFIRawStr`
     //
     kTVMFFINone = 0,
     kTVMFFIInt = 1,
     kTVMFFIBool = 2,
     kTVMFFIFloat = 3,
     kTVMFFIOpaquePtr = 4,
     kTVMFFIDataType = 5,
     kTVMFFIDevice = 6,
     kTVMFFIDLTensorPtr = 7,
     kTVMFFIRawStr = 8,
     kTVMFFIByteArrayPtr = 9,
     kTVMFFIObjectRValueRef = 10,
     kTVMFFISmallStr = 11,
     kTVMFFISmallBytes = 12,
     kTVMFFIStaticObjectBegin = 64,
     kTVMFFIObject = 64,
     kTVMFFIStr = 65,
     kTVMFFIBytes = 66,
     kTVMFFIError = 67,
     kTVMFFIFunction = 68,
     kTVMFFIShape = 69,
     kTVMFFITensor = 70,
     kTVMFFIArray = 71,
     //----------------------------------------------------------------
     // more complex objects
     //----------------------------------------------------------------
     kTVMFFIMap = 72,
     kTVMFFIModule = 73,
     kTVMFFIOpaquePyObject = 74,
     kTVMFFIStaticObjectEnd,
     // [Section] Dynamic Boxed: [kTVMFFIDynObjectBegin, +oo)
     kTVMFFIDynObjectBegin = 128
   #ifdef __cplusplus
   };
   #else
   } TVMFFITypeIndex;
   #endif
   
   typedef void* TVMFFIObjectHandle;
   
   #ifdef __cplusplus
   enum TVMFFIObjectDeleterFlagBitMask : int32_t {
   #else
   typedef enum {
   #endif
     kTVMFFIObjectDeleterFlagBitMaskStrong = 1 << 0,
     kTVMFFIObjectDeleterFlagBitMaskWeak = 1 << 1,
     kTVMFFIObjectDeleterFlagBitMaskBoth =
         (kTVMFFIObjectDeleterFlagBitMaskStrong | kTVMFFIObjectDeleterFlagBitMaskWeak),
   #ifdef __cplusplus
   };
   #else
   } TVMFFIObjectDeleterFlagBitMask;
   #endif
   
   typedef struct {
     uint64_t combined_ref_count;
     int32_t type_index;
     uint32_t __padding;
   #if !defined(TVM_FFI_DOXYGEN_MODE)
     union {
   #endif
       void (*deleter)(void* self, int flags);
       int64_t __ensure_align;
   #if !defined(TVM_FFI_DOXYGEN_MODE)
     };
   #endif
   } TVMFFIObject;
   
   typedef struct {
     int32_t type_index;
   #if !defined(TVM_FFI_DOXYGEN_MODE)
     union {  // 4 bytes
   #endif
       uint32_t zero_padding;
       uint32_t small_str_len;
   #if !defined(TVM_FFI_DOXYGEN_MODE)
     };
   #endif
   #if !defined(TVM_FFI_DOXYGEN_MODE)
     union {  // 8 bytes
   #endif
       int64_t v_int64;
       double v_float64;
       void* v_ptr;
       const char* v_c_str;
       TVMFFIObject* v_obj;
       DLDataType v_dtype;
       DLDevice v_device;
       char v_bytes[8];
       uint64_t v_uint64;
   #if !defined(TVM_FFI_DOXYGEN_MODE)
     };
   #endif
   } TVMFFIAny;
   
   typedef struct {
     const char* data;
     size_t size;
   } TVMFFIByteArray;
   
   typedef struct {
     const int64_t* data;
     size_t size;
   } TVMFFIShapeCell;
   
   #ifdef __cplusplus
   enum TVMFFIBacktraceUpdateMode : int32_t {
   #else
   typedef enum {
   #endif
     kTVMFFIBacktraceUpdateModeReplace = 0,
     kTVMFFIBacktraceUpdateModeAppend = 1,
   #ifdef __cplusplus
   };
   #else
   } TVMFFIBacktraceUpdateMode;
   #endif
   
   typedef struct {
     TVMFFIByteArray kind;
     TVMFFIByteArray message;
     TVMFFIByteArray backtrace;
     void (*update_backtrace)(TVMFFIObjectHandle self, const TVMFFIByteArray* backtrace,
                              int32_t update_mode);
   } TVMFFIErrorCell;
   
   typedef int (*TVMFFISafeCallType)(void* handle, const TVMFFIAny* args, int32_t num_args,
                                     TVMFFIAny* result);
   
   typedef struct {
     TVMFFISafeCallType safe_call;
     void* cpp_call;
   } TVMFFIFunctionCell;
   
   typedef struct {
     void* handle;
   } TVMFFIOpaqueObjectCell;
   
   //-----------------------------------------------------------------------
   // Section: Version API
   //-----------------------------------------------------------------------
   TVM_FFI_DLL void TVMFFIGetVersion(TVMFFIVersion* out_version);
   
   //------------------------------------------------------------
   // Section: Basic object API
   //------------------------------------------------------------
   TVM_FFI_DLL int TVMFFIObjectIncRef(TVMFFIObjectHandle obj);
   
   TVM_FFI_DLL int TVMFFIObjectDecRef(TVMFFIObjectHandle obj);
   
   TVM_FFI_DLL int TVMFFIObjectCreateOpaque(void* handle, int32_t type_index,
                                            void (*deleter)(void* handle), TVMFFIObjectHandle* out);
   
   TVM_FFI_DLL int TVMFFITypeKeyToIndex(const TVMFFIByteArray* type_key, int32_t* out_tindex);
   
   //-----------------------------------------------------------------------
   // Section: Basic function calling API for function implementation
   //-----------------------------------------------------------------------
   TVM_FFI_DLL int TVMFFIFunctionCreate(void* self, TVMFFISafeCallType safe_call,
                                        void (*deleter)(void* self), TVMFFIObjectHandle* out);
   
   TVM_FFI_DLL int TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out);
   
   TVM_FFI_DLL int TVMFFIAnyViewToOwnedAny(const TVMFFIAny* any_view, TVMFFIAny* out);
   
   TVM_FFI_DLL int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                                      TVMFFIAny* result);
   
   TVM_FFI_DLL void TVMFFIErrorMoveFromRaised(TVMFFIObjectHandle* result);
   
   TVM_FFI_DLL void TVMFFIErrorSetRaised(TVMFFIObjectHandle error);
   
   TVM_FFI_DLL void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message);
   
   TVM_FFI_DLL void TVMFFIErrorSetRaisedFromCStrParts(const char* kind, const char** message_parts,
                                                      int32_t num_parts);
   
   TVM_FFI_DLL int TVMFFIErrorCreate(const TVMFFIByteArray* kind, const TVMFFIByteArray* message,
                                     const TVMFFIByteArray* backtrace, TVMFFIObjectHandle* out);
   
   //------------------------------------------------------------
   // Section: DLPack support APIs
   //------------------------------------------------------------
   TVM_FFI_DLL int TVMFFITensorFromDLPack(DLManagedTensor* from, int32_t require_alignment,
                                          int32_t require_contiguous, TVMFFIObjectHandle* out);
   
   TVM_FFI_DLL int TVMFFITensorToDLPack(TVMFFIObjectHandle from, DLManagedTensor** out);
   
   TVM_FFI_DLL int TVMFFITensorFromDLPackVersioned(DLManagedTensorVersioned* from,
                                                   int32_t require_alignment,
                                                   int32_t require_contiguous,
                                                   TVMFFIObjectHandle* out);
   
   TVM_FFI_DLL int TVMFFITensorToDLPackVersioned(TVMFFIObjectHandle from,
                                                 DLManagedTensorVersioned** out);
   //---------------------------------------------------------------
   // Section: string/bytes support APIs.
   // These APIs are used to simplify the string/bytes construction
   //---------------------------------------------------------------
   TVM_FFI_DLL int TVMFFIStringFromByteArray(const TVMFFIByteArray* input, TVMFFIAny* out);
   
   TVM_FFI_DLL int TVMFFIBytesFromByteArray(const TVMFFIByteArray* input, TVMFFIAny* out);
   
   //---------------------------------------------------------------
   // Section: dtype string support APIs.
   // These APIs are used to simplify the dtype printings during FFI
   //---------------------------------------------------------------
   
   TVM_FFI_DLL int TVMFFIDataTypeFromString(const TVMFFIByteArray* str, DLDataType* out);
   
   TVM_FFI_DLL int TVMFFIDataTypeToString(const DLDataType* dtype, TVMFFIAny* out);
   
   //------------------------------------------------------------
   // Section: Type reflection support APIs
   //
   // The reflec
   //------------------------------------------------------------
   typedef int (*TVMFFIFieldGetter)(void* field, TVMFFIAny* result);
   
   typedef int (*TVMFFIFieldSetter)(void* field, const TVMFFIAny* value);
   
   typedef int (*TVMFFIObjectCreator)(TVMFFIObjectHandle* result);
   
   #ifdef __cplusplus
   enum TVMFFIFieldFlagBitMask : int32_t {
   #else
   typedef enum {
   #endif
     kTVMFFIFieldFlagBitMaskWritable = 1 << 0,
     kTVMFFIFieldFlagBitMaskHasDefault = 1 << 1,
     kTVMFFIFieldFlagBitMaskIsStaticMethod = 1 << 2,
     kTVMFFIFieldFlagBitMaskSEqHashIgnore = 1 << 3,
     kTVMFFIFieldFlagBitMaskSEqHashDef = 1 << 4,
   #ifdef __cplusplus
   };
   #else
   } TVMFFIFieldFlagBitMask;
   #endif
   
   #ifdef __cplusplus
   enum TVMFFISEqHashKind : int32_t {
   #else
   typedef enum {
   #endif
     kTVMFFISEqHashKindUnsupported = 0,
     kTVMFFISEqHashKindTreeNode = 1,
     kTVMFFISEqHashKindFreeVar = 2,
     kTVMFFISEqHashKindDAGNode = 3,
     kTVMFFISEqHashKindConstTreeNode = 4,
     kTVMFFISEqHashKindUniqueInstance = 5,
   #ifdef __cplusplus
   };
   #else
   } TVMFFISEqHashKind;
   #endif
   
   typedef struct {
     TVMFFIByteArray name;
     TVMFFIByteArray doc;
     TVMFFIByteArray metadata;
     int64_t flags;
     int64_t size;
     int64_t alignment;
     int64_t offset;
     TVMFFIFieldGetter getter;
     TVMFFIFieldSetter setter;
     TVMFFIAny default_value;
     int32_t field_static_type_index;
   } TVMFFIFieldInfo;
   
   typedef struct {
     TVMFFIByteArray name;
     TVMFFIByteArray doc;
     // Rationale: We separate the docstring from the metadata since docstrings
     // can be unstructured and sometimes large, while metadata can be focused
     // on storing structured information.
     TVMFFIByteArray metadata;
     int64_t flags;
     TVMFFIAny method;
   } TVMFFIMethodInfo;
   
   typedef struct {
     TVMFFIByteArray doc;
     TVMFFIObjectCreator creator;
     int32_t total_size;
     TVMFFISEqHashKind structural_eq_hash_kind;
   } TVMFFITypeMetadata;
   
   typedef struct {
     const TVMFFIAny* data;
     size_t size;
   } TVMFFITypeAttrColumn;
   
   #ifdef __cplusplus
   struct TVMFFITypeInfo {
   #else
   typedef struct TVMFFITypeInfo {
   #endif
     int32_t type_index;
     int32_t type_depth;
     TVMFFIByteArray type_key;
     const struct TVMFFITypeInfo** type_ancestors;
     // The following fields are used for reflection
     uint64_t type_key_hash;
     int32_t num_fields;
     int32_t num_methods;
     const TVMFFIFieldInfo* fields;
     const TVMFFIMethodInfo* methods;
     const TVMFFITypeMetadata* metadata;
   #ifdef __cplusplus
   };
   #else
   } TVMFFITypeInfo;
   #endif
   
   TVM_FFI_DLL int TVMFFIFunctionSetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle f,
                                           int allow_override);
   
   TVM_FFI_DLL int TVMFFIFunctionSetGlobalFromMethodInfo(const TVMFFIMethodInfo* method_info,
                                                         int allow_override);
   
   TVM_FFI_DLL int TVMFFITypeRegisterField(int32_t type_index, const TVMFFIFieldInfo* info);
   
   TVM_FFI_DLL int TVMFFITypeRegisterMethod(int32_t type_index, const TVMFFIMethodInfo* info);
   
   TVM_FFI_DLL int TVMFFITypeRegisterMetadata(int32_t type_index, const TVMFFITypeMetadata* metadata);
   
   TVM_FFI_DLL int TVMFFITypeRegisterAttr(int32_t type_index, const TVMFFIByteArray* attr_name,
                                          const TVMFFIAny* attr_value);
   
   TVM_FFI_DLL const TVMFFITypeAttrColumn* TVMFFIGetTypeAttrColumn(const TVMFFIByteArray* attr_name);
   
   //------------------------------------------------------------
   // Section: Backend noexcept functions for internal use
   //
   // These functions are used internally and do not throw error
   // instead the error will be logged and abort the process
   // These are function are being called in startup or exit time
   // so exception handling do not apply
   //------------------------------------------------------------
   TVM_FFI_DLL const TVMFFIByteArray* TVMFFIBacktrace(const char* filename, int lineno,
                                                      const char* func, int cross_ffi_boundary);
   
   TVM_FFI_DLL int32_t TVMFFITypeGetOrAllocIndex(const TVMFFIByteArray* type_key,
                                                 int32_t static_type_index, int32_t type_depth,
                                                 int32_t num_child_slots,
                                                 int32_t child_slots_can_overflow,
                                                 int32_t parent_type_index);
   
   TVM_FFI_DLL const TVMFFITypeInfo* TVMFFIGetTypeInfo(int32_t type_index);
   
   #ifdef __cplusplus
   }  // TVM_FFI_EXTERN_C
   #endif
   
   //---------------------------------------------------------------
   // The following API defines static object attribute accessors
   // for language bindings.
   //
   // They are defined in C++ inline functions for cleaner code.
   // Note that they only have to do with address offset computation.
   // So they can always be reimplemented in bindings when c++ is
   // not available or when binding only wants to refer to the dll.
   //----------------------------------------------------------------
   #ifdef __cplusplus
   inline int32_t TVMFFIObjectGetTypeIndex(TVMFFIObjectHandle obj) {
     return static_cast<TVMFFIObject*>(obj)->type_index;
   }
   
   inline TVMFFIByteArray TVMFFISmallBytesGetContentByteArray(const TVMFFIAny* value) {
     return TVMFFIByteArray{value->v_bytes, static_cast<size_t>(value->small_str_len)};
   }
   
   inline TVMFFIByteArray* TVMFFIBytesGetByteArrayPtr(TVMFFIObjectHandle obj) {
     return reinterpret_cast<TVMFFIByteArray*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
   }
   
   inline TVMFFIErrorCell* TVMFFIErrorGetCellPtr(TVMFFIObjectHandle obj) {
     return reinterpret_cast<TVMFFIErrorCell*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
   }
   
   inline TVMFFIFunctionCell* TVMFFIFunctionGetCellPtr(TVMFFIObjectHandle obj) {
     return reinterpret_cast<TVMFFIFunctionCell*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
   }
   
   inline TVMFFIOpaqueObjectCell* TVMFFIOpaqueObjectGetCellPtr(TVMFFIObjectHandle obj) {
     return reinterpret_cast<TVMFFIOpaqueObjectCell*>(reinterpret_cast<char*>(obj) +
                                                      sizeof(TVMFFIObject));
   }
   
   inline TVMFFIShapeCell* TVMFFIShapeGetCellPtr(TVMFFIObjectHandle obj) {
     return reinterpret_cast<TVMFFIShapeCell*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
   }
   
   inline DLTensor* TVMFFITensorGetDLTensorPtr(TVMFFIObjectHandle obj) {
     return reinterpret_cast<DLTensor*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
   }
   
   inline DLDevice TVMFFIDLDeviceFromIntPair(int32_t device_type, int32_t device_id) {
     return DLDevice{static_cast<DLDeviceType>(device_type), device_id};
   }
   #endif  // __cplusplus
   #endif  // TVM_FFI_C_API_H_
   // NOLINTEND(modernize-use-using,bugprone-reserved-identifier,modernize-deprecated-headers)
