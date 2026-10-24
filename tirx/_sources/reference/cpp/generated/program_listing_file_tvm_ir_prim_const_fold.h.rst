
.. _program_listing_file_tvm_ir_prim_const_fold.h:

Program Listing for File const_fold.h
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_prim_const_fold.h>` (``tvm/ir/prim/const_fold.h``)

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
   
   #ifndef TVM_IR_PRIM_CONST_FOLD_H_
   #define TVM_IR_PRIM_CONST_FOLD_H_
   
   #include <tvm/ffi/optional.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/runtime/logging.h>
   
   #include <algorithm>
   #include <cmath>
   #include <limits>
   
   namespace tvm {
   namespace prim {
   
   template <typename Op>
   inline ffi::Optional<PrimExpr> TryConstFold(PrimExpr a, PrimExpr b);
   
   template <typename Op>
   inline ffi::Optional<PrimExpr> TryConstFold(PrimExpr a);
   
   namespace detail {
   
   inline bool IsIndexType(DLDataType type) {
     return type.code == static_cast<uint8_t>(DLDataTypeCode::kDLInt) &&
            (type.bits == 32 || type.bits == 64) && type.lanes == 1;
   }
   
   inline bool IsIndexTypedExpr(const ExprObj* expr) {
     TVM_FFI_DCHECK(expr != nullptr);
     TVM_FFI_DCHECK(!expr->ExprObj::ty.as<MissingType>().has_value());
     const auto* prim_ty = expr->ExprObj::ty.as<PrimTypeObj>();
     TVM_FFI_DCHECK(prim_ty != nullptr);
     return IsIndexType(prim_ty->dtype);
   }
   
   inline bool IsIndexTypedExpr(const PrimExpr& expr) { return IsIndexTypedExpr(expr.get()); }
   
   inline bool IsRepresentable(const ffi::BigInt& value, const PrimType& dtype) {
     if (!dtype.IsScalar() || dtype.bits() <= 0) return false;
     if (auto small = value.as<int64_t>()) {
       if (dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
         return *small >= 0 &&
                (dtype.bits() >= 63 || static_cast<uint64_t>(*small) < (uint64_t{1} << dtype.bits()));
       }
       if (dtype.MatchesCode(DLDataTypeCode::kDLBool) ||
           (dtype.MatchesCode(DLDataTypeCode::kDLInt) && dtype.bits() == 1)) {
         return *small == 0 || *small == 1;
       }
       if (dtype.MatchesCode(DLDataTypeCode::kDLInt)) {
         if (dtype.bits() >= 64) return true;
         int64_t limit = int64_t{1} << (dtype.bits() - 1);
         return *small >= -limit && *small < limit;
       }
       return false;
     }
     if (dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
       return value >= 0 && value < (ffi::BigInt(1) << dtype.bits());
     }
     if (dtype.MatchesCode(DLDataTypeCode::kDLInt) && dtype.bits() > 1) {
       ffi::BigInt limit = ffi::BigInt(1) << (dtype.bits() - 1);
       return value >= -limit && value < limit;
     }
     return false;
   }
   
   inline bool IsInFloat32Range(double value) {
     return !std::isfinite(value) || (value >= std::numeric_limits<float>::lowest() &&
                                      value <= std::numeric_limits<float>::max());
   }
   
   }  // namespace detail
   
   #define TVM_PRIM_CONST_PROPAGATION(BODY)       \
     using tvm::FloatImmObj;                      \
     const IntImmObj* pa = a.as<IntImmObj>();     \
     const IntImmObj* pb = b.as<IntImmObj>();     \
     const FloatImmObj* fa = a.as<FloatImmObj>(); \
     const FloatImmObj* fb = b.as<FloatImmObj>(); \
     BODY;
   
   #define TVM_PRIM_INDEX_CONST_PROPAGATION(BODY)                                                \
     const IntImmObj* pa = a.as<IntImmObj>();                                                    \
     const IntImmObj* pb = b.as<IntImmObj>();                                                    \
     if (::tvm::prim::detail::IsIndexTypedExpr(a) && ::tvm::prim::detail::IsIndexTypedExpr(b)) { \
       BODY;                                                                                     \
     }
   
   // specialization of constant folders.
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Add>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) {
         ffi::BigInt res = pa->value + pb->value;
         if (!detail::IsRepresentable(res, result_ty)) return std::nullopt;
         return IntImm(result_ty, std::move(res));
       }
       if (pa && pa->value == 0) return b;
       if (pb && pb->value == 0) return a;
       if (fa && fb) {
         if (result_ty.bits() == 32) {
           float lhs = static_cast<float>(fa->value);
           float rhs = static_cast<float>(fb->value);
           // Check before rounding can turn a finite overflow into infinity.
           if (!detail::IsInFloat32Range(static_cast<double>(lhs) + rhs)) return std::nullopt;
           return FloatImm(result_ty, static_cast<float>(lhs + rhs));
         } else if (result_ty.bits() == 64) {
           return FloatImm(result_ty, fa->value + fb->value);
         }
       }
       if (fa && fa->value == 0) return b;
       if (fb && fb->value == 0) return a;
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Sub>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) {
         ffi::BigInt res = pa->value - pb->value;
         if (!detail::IsRepresentable(res, result_ty)) return std::nullopt;
         return IntImm(result_ty, std::move(res));
       }
       if (pb && pb->value == 0) return a;
       if (fa && fb) {
         if (result_ty.bits() == 32) {
           float lhs = static_cast<float>(fa->value);
           float rhs = static_cast<float>(fb->value);
           // Check before rounding can turn a finite overflow into infinity.
           if (!detail::IsInFloat32Range(static_cast<double>(lhs) - rhs)) return std::nullopt;
           return FloatImm(result_ty, static_cast<float>(lhs - rhs));
         } else if (result_ty.bits() == 64) {
           return FloatImm(result_ty, fa->value - fb->value);
         }
       }
       if (fb && fb->value == 0) return a;
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Mul>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) {
         ffi::BigInt res = pa->value * pb->value;
         if (!detail::IsRepresentable(res, result_ty)) return std::nullopt;
         return IntImm(result_ty, std::move(res));
       }
       if (pa) {
         if (pa->value == 1) return b;
         if (pa->value == 0) return a;
       }
       if (pb) {
         if (pb->value == 1) return a;
         if (pb->value == 0) return b;
       }
       if (fa && fb) {
         if (result_ty.bits() == 32) {
           float lhs = static_cast<float>(fa->value);
           float rhs = static_cast<float>(fb->value);
           // Check before rounding can turn a finite overflow into infinity.
           if (!detail::IsInFloat32Range(static_cast<double>(lhs) * rhs)) return std::nullopt;
           return FloatImm(result_ty, static_cast<float>(lhs * rhs));
         } else if (result_ty.bits() == 64) {
           return FloatImm(result_ty, fa->value * fb->value);
         }
       }
       if (fa) {
         if (fa->value == 1) return b;
         if (fa->value == 0) return a;
       }
       if (fb) {
         if (fb->value == 1) return a;
         if (fb->value == 0) return b;
       }
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Div>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) {
         // due to division and mod can have different modes
         // NOTE: this will assumes truc div.
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
         ffi::BigInt res = pa->value / pb->value;
         if (!detail::IsRepresentable(res, result_ty)) return std::nullopt;
         return IntImm(result_ty, std::move(res));
       }
       if (pa) {
         if (pa->value == 0) return a;
       }
       if (pb) {
         if (pb->value == 1) return a;
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
       }
       if (fa && fb) {
         TVM_FFI_ICHECK_NE(fb->value, 0) << "Divide by zero";
         if (result_ty.bits() == 32) {
           float lhs = static_cast<float>(fa->value);
           float rhs = static_cast<float>(fb->value);
           // Check before rounding can turn a finite overflow into infinity.
           if (!detail::IsInFloat32Range(static_cast<double>(lhs) / rhs)) return std::nullopt;
           return FloatImm(result_ty, static_cast<float>(lhs / rhs));
         } else if (result_ty.bits() == 64) {
           return FloatImm(result_ty, fa->value / fb->value);
         }
       }
       if (fa && fa->value == 0) return a;
       if (fb) {
         if (fb->value == 1) return a;
         TVM_FFI_ICHECK_NE(fb->value, 0) << "Divide by zero";
       }
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Mod>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_INDEX_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) {
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
         ffi::BigInt res = pa->value % pb->value;
         if (!detail::IsRepresentable(res, result_ty)) return std::nullopt;
         return IntImm(result_ty, std::move(res));
       }
       if (pa) {
         if (pa->value == 0) return a;
       }
       if (pb) {
         if (pb->value == 1) return IntImm(result_ty, 0);
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
       }
     });
     // Fold wide literals without enabling symbolic index rewrites for their type.
     if (pa && pb && a.ty().bits() >= 64) {
       TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
       ffi::BigInt res = pa->value % pb->value;
       if (!detail::IsRepresentable(res, a.ty())) return std::nullopt;
       return IntImm(a.ty(), std::move(res));
     }
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::FloorDiv>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) {
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
         ffi::BigInt res = ffi::floordiv(pa->value, pb->value);
         if (!detail::IsRepresentable(res, result_ty)) return std::nullopt;
         return IntImm(result_ty, std::move(res));
       }
       if (pa) {
         if (pa->value == 0) return a;
       }
       if (pb) {
         if (pb->value == 1) return a;
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
       }
       if (fa && fb && fb->value != 0) {
         if (result_ty.bits() == 32) {
           float lhs = static_cast<float>(fa->value);
           float rhs = static_cast<float>(fb->value);
           if (!detail::IsInFloat32Range(static_cast<double>(lhs) / rhs)) return std::nullopt;
           float quotient = lhs / rhs;
           return FloatImm(result_ty, std::floor(quotient));
         } else if (result_ty.bits() == 64) {
           return FloatImm(result_ty, std::floor(fa->value / fb->value));
         } else {
           return std::nullopt;
         }
       }
       if (fa && fa->value == 0) return a;
       if (fb) {
         if (fb->value == 1) return a;
         TVM_FFI_ICHECK_NE(fb->value, 0) << "Divide by zero";
       }
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::FloorMod>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_INDEX_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) {
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
         ffi::BigInt res = ffi::floormod(pa->value, pb->value);
         if (!detail::IsRepresentable(res, result_ty)) return std::nullopt;
         return IntImm(result_ty, std::move(res));
       }
       if (pa) {
         if (pa->value == 0) return a;
       }
       if (pb) {
         if (pb->value == 1) return IntImm(result_ty, 0);
         TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
       }
     });
     // Fold wide literals without enabling symbolic index rewrites for their type.
     if (pa && pb && (a.ty().bits() >= 64 || a.ty().MatchesCode(DLDataTypeCode::kDLUInt))) {
       TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero";
       ffi::BigInt res = ffi::floormod(pa->value, pb->value);
       if (!detail::IsRepresentable(res, a.ty())) return std::nullopt;
       return IntImm(a.ty(), std::move(res));
     }
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Min>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) return IntImm(result_ty, std::min(pa->value, pb->value));
       if (fa && fb) return std::isnan(fa->value) || fa->value < fb->value ? a : b;
     });
     if (a.same_as(b)) return a;
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Max>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       PrimType result_ty = a.ty();
       if (pa && pb) return IntImm(result_ty, std::max(pa->value, pb->value));
       if (fa && fb) return std::isnan(fa->value) || fa->value > fb->value ? a : b;
     });
     if (a.same_as(b)) return a;
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::GT>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       if (pa && pb) return IntImm::Bool(pa->value > pb->value);
       if (fa && fb) return IntImm::Bool(fa->value > fb->value);
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::GE>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       if (pa && pb) return IntImm::Bool(pa->value >= pb->value);
       if (fa && fb) return IntImm::Bool(fa->value >= fb->value);
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::LT>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       if (pa && pb) return IntImm::Bool(pa->value < pb->value);
       if (fa && fb) return IntImm::Bool(fa->value < fb->value);
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::LE>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       if (pa && pb) return IntImm::Bool(pa->value <= pb->value);
       if (fa && fb) return IntImm::Bool(fa->value <= fb->value);
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::EQ>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       if (pa && pb) return IntImm::Bool(pa->value == pb->value);
       if (fa && fb) return IntImm::Bool(fa->value == fb->value);
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::NE>(PrimExpr a, PrimExpr b) {
     TVM_PRIM_CONST_PROPAGATION({
       if (pa && pb) return IntImm::Bool(pa->value != pb->value);
       if (fa && fb) return IntImm::Bool(fa->value != fb->value);
     });
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::And>(PrimExpr a, PrimExpr b) {
     const IntImmObj* pa = a.as<IntImmObj>();
     const IntImmObj* pb = b.as<IntImmObj>();
     if (pa && pa->value) return b;
     if (pa && !pa->value) return a;
     if (pb && pb->value) return a;
     if (pb && !pb->value) return b;
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Or>(PrimExpr a, PrimExpr b) {
     const IntImmObj* pa = a.as<IntImmObj>();
     const IntImmObj* pb = b.as<IntImmObj>();
     if (pa && pa->value) return a;
     if (pa && !pa->value) return b;
     if (pb && pb->value) return b;
     if (pb && !pb->value) return a;
     return std::nullopt;
   }
   
   template <>
   inline ffi::Optional<PrimExpr> TryConstFold<prim::Not>(PrimExpr a) {
     const IntImmObj* pa = a.as<IntImmObj>();
     if (pa) {
       return IntImm::Bool(!(pa->value));
     }
     return std::nullopt;
   }
   
   }  // namespace prim
   }  // namespace tvm
   #endif  // TVM_IR_PRIM_CONST_FOLD_H_
