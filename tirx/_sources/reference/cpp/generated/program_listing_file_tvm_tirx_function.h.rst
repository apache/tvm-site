
.. _program_listing_file_tvm_tirx_function.h:

Program Listing for File function.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_function.h>` (``tvm/tirx/function.h``)

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
   
   #ifndef TVM_TIRX_FUNCTION_H_
   #define TVM_TIRX_FUNCTION_H_
   
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/extra/structural_equal.h>
   #include <tvm/ffi/extra/structural_hash.h>
   #include <tvm/ir/cow.h>
   #include <tvm/ir/function.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/runtime/tensor.h>
   #include <tvm/tirx/expr.h>
   #include <tvm/tirx/stmt.h>
   
   #include <string>
   
   namespace tvm {
   namespace tirx {
   
   class FunctionObj : public BaseFuncObj {
    public:
     ffi::Array<tvm::Var> params;
     Type ret_type = Type::Missing();
     ffi::Optional<SeqStmt> body;
   
     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<FunctionObj>()
           .def_ro("params", &FunctionObj::params, refl::AttachFieldFlag::SEqHashDefPattern())
           .def_ro("ret_type", &FunctionObj::ret_type)
           .def_ro("body", &FunctionObj::body);
       refl::TypeAttrDef<FunctionObj>()
           .def(tvm::ffi::type_attr::kSEqual.name, &FunctionObj::SEqual)
           .def(tvm::ffi::type_attr::kSHash.name, &FunctionObj::SHash);
     }
   
     bool SEqual(const FunctionObj* other,
                 ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
       // `ty` is derived from the signature; compare its source fields directly.
       return equal(attrs, other->attrs, false, "attrs") &&
              equal(params, other->params, true, "params") &&
              equal(ret_type, other->ret_type, false, "ret_type") &&
              equal(body, other->body, false, "body");
     }
   
     int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
       int64_t hash_value = init_hash;
       hash_value = hash(attrs, hash_value, false);
       hash_value = hash(params, hash_value, true);
       hash_value = hash(ret_type, hash_value, false);
       hash_value = hash(body, hash_value, false);
       return hash_value;
     }
   
     TVM_TIRX_DLL FuncType func_type_annotation() const;
   
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Function", FunctionObj, BaseFuncObj);
   };
   
   class Function : public BaseFunc {
    public:
     TVM_TIRX_DLL Function(ffi::Array<tvm::Var> params, ffi::Optional<SeqStmt> body,
                           Type ret_type = VoidType(), DictAttrs attrs = DictAttrs(),
                           Location loc = UnknownLoc());
   
     explicit Function(ffi::ObjectPtr<FunctionObj> node) : BaseFunc(std::move(node)) {}
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Function, BaseFunc, FunctionObj);
     TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionObj);
   };
   
   TVM_TIRX_DLL Function RenewDef(Function func);
   
   namespace attr {
   
   inline constexpr ffi::reflection::AttrKey<ffi::Array<ffi::String>> kKernelLaunchParams{
       "tirx.kernel_launch_params"};
   
   inline constexpr ffi::reflection::AttrKey<bool> kNoAlias{"tirx.noalias"};
   
   inline constexpr ffi::reflection::AttrKey<bool> kIsEntryFunc{"tirx.is_entry_func"};
   
   inline constexpr ffi::reflection::AttrKey<bool> kIsGlobalFunc{"tirx.is_global_func"};
   
   inline constexpr ffi::reflection::AttrKey<bool> kIsHostFunc{"tirx.is_host_func"};
   
   inline constexpr ffi::reflection::AttrKey<bool> kPersistentKernel{"tirx.persistent_kernel"};
   
   }  // namespace attr
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_FUNCTION_H_
