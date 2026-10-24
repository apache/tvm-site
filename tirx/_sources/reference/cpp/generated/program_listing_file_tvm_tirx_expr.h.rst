
.. _program_listing_file_tvm_tirx_expr.h:

Program Listing for File expr.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_expr.h>` (``tvm/tirx/expr.h``)

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
   
   #ifndef TVM_TIRX_EXPR_H_
   #define TVM_TIRX_EXPR_H_
   
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/reflection/registry.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/expr.h>
   #include <tvm/tirx/op/memory.h>
   #include <tvm/tirx/type.h>
   
   namespace tvm {
   // forward declare shared statement.
   class Stmt;
   
   namespace tirx {
   
   class TensorVar : public Var {
    public:
     TVM_TIRX_DLL explicit TensorVar(ffi::String name, TensorType type, Location loc = UnknownLoc());
   
     explicit TensorVar(Var var) : Var(std::move(var)) {
       TVM_FFI_ICHECK(get() != nullptr && get()->ty.as<TensorTypeObj>())
           << "Expected a non-null Var with TensorType";
     }
   
     TensorType ty() const { return get()->ty.as_or_throw<TensorType>(); }
   
     const ffi::Array<PrimExpr>& shape() const { return ty()->shape; }
   
     PrimType dtype() const { return ty()->dtype; }
   
     Expr data() const { return Call(ty()->DataPtrType(), tensor_data_ptr_op(), {*this}); }
   
     runtime::MemoryScope scope() const { return ty()->scope; }
   
     TVM_TIRX_DLL bool IsScalar(bool alloc_or_decl = true) const;
   
     explicit TensorVar(ffi::ObjectPtr<VarObj> node) : TensorVar(Var(std::move(node))) {}
     explicit TensorVar(ffi::UnsafeInit tag) : Var(tag) {}
     TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TensorVar);
   
     [[maybe_unused]] static constexpr bool _type_is_nullable = false;
     static constexpr bool _type_container_is_exact = false;
     using ContainerType = VarObj;
   };
   
   // Preserve ObjectRef-style identity comparison for exact TensorVar operands.
   // Comparisons widened to Var or Expr remain unavailable; use same_as for identity.
   inline bool operator==(const TensorVar& lhs, const TensorVar& rhs) { return lhs.same_as(rhs); }
   
   inline bool operator!=(const TensorVar& lhs, const TensorVar& rhs) { return !lhs.same_as(rhs); }
   
   }  // namespace tirx
   }  // namespace tvm
   
   namespace tvm::ffi {
   
   template <>
   inline constexpr bool use_default_type_traits_v<tirx::TensorVar> = false;
   
   template <>
   struct TypeTraits<tirx::TensorVar> : public ObjectRefTypeTraitsBase<tirx::TensorVar> {
     using Base = ObjectRefTypeTraitsBase<tirx::TensorVar>;
     using Base::CopyFromAnyViewAfterCheck;
     using Base::CopyToAnyView;
     using Base::GetMismatchTypeInfo;
     using Base::MoveFromAnyAfterCheck;
     using Base::MoveToAny;
     using Base::TypeSchema;
     using Base::TypeStr;
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFINone) {
         return false;
       }
       if (src->type_index != tvm::VarObj::RuntimeTypeIndex()) {
         return false;
       }
       const auto* var = static_cast<const tvm::VarObj*>(
           detail::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
       return detail::AnyUnsafe::CheckAnyStrict<tirx::TensorType>(var->ExprObj::ty);
     }
   
     TVM_FFI_INLINE static std::optional<tirx::TensorVar> TryCastFromAnyView(const TVMFFIAny* src) {
       if (CheckAnyStrict(src)) {
         return detail::ObjectUnsafe::ObjectRefFromObjectPtr<tirx::TensorVar>(
             detail::ObjectUnsafe::ObjectPtrFromUnowned<tvm::VarObj>(src->v_obj));
       }
       return std::nullopt;
     }
   };
   
   }  // namespace tvm::ffi
   
   #endif  // TVM_TIRX_EXPR_H_
