
.. _program_listing_file_tvm_ir_prim_op.h:

Program Listing for File op.h
=============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_prim_op.h>` (``tvm/ir/prim/op.h``)

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
   
   #ifndef TVM_IR_PRIM_OP_H_
   #define TVM_IR_PRIM_OP_H_
   #include <tvm/ir/op.h>
   #include <tvm/ir/prim/expr.h>
   
   #include <algorithm>
   #include <limits>
   #include <type_traits>
   #include <utility>
   namespace tvm {
   
   TVM_TIRX_DLL PrimExpr operator+(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator-(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator-(PrimExpr a);
   
   TVM_TIRX_DLL PrimExpr operator*(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator/(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator<<(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator>>(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator>(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator>=(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator<(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator<=(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator==(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator!=(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator&&(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator||(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator!(PrimExpr a);
   
   TVM_TIRX_DLL PrimExpr operator&(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator|(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator^(PrimExpr a, PrimExpr b);
   
   TVM_TIRX_DLL PrimExpr operator~(PrimExpr a);
   
   inline PrimExpr operator+=(PrimExpr& a, PrimExpr b) {
     a = a + b;
     return a;
   }
   
   inline PrimExpr operator-=(PrimExpr& a, PrimExpr b) {
     a = a - b;
     return a;
   }
   
   inline PrimExpr operator*=(PrimExpr& a, PrimExpr b) {
     a = a * b;
     return a;
   }
   
   namespace prim {
   
   TVM_TIRX_DLL const PrimExpr& pos_inf();
   
   TVM_TIRX_DLL const PrimExpr& neg_inf();
   
   TVM_TIRX_DLL bool is_pos_inf(const PrimExpr& value);
   
   TVM_TIRX_DLL bool is_neg_inf(const PrimExpr& value);
   
   TVM_TIRX_DLL const Op& vscale_op();
   
   TVM_TIRX_DLL const Op& ceil_op();
   
   TVM_TIRX_DLL const Op& log2_op();
   
   TVM_TIRX_DLL const Op& clz_op();
   
   TVM_TIRX_DLL const Op& if_then_else_op();
   
   TVM_TIRX_DLL const Op& likely_op();
   
   }  // namespace prim
   
   TVM_TIRX_DLL PrimExpr add(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr sub(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr neg(PrimExpr a, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr mul(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr left_shift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr right_shift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr greater(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr greater_equal(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr less(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr less_equal(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr equal(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr not_equal(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr logical_and(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr logical_or(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr logical_not(PrimExpr a, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr div(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr truncdiv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr truncmod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr indexdiv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr shapediv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr indexmod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr floordiv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr ceildiv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr floormod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr max(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr min(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr bitwise_and(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr bitwise_or(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr bitwise_neg(PrimExpr a, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value,
                                      Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr likely(PrimExpr cond, Location loc = UnknownLoc());
   namespace prim {
   TVM_TIRX_DLL PrimExpr ceil(PrimExpr x, Location loc = UnknownLoc());
   TVM_TIRX_DLL PrimExpr log2(PrimExpr x, Location loc = UnknownLoc());
   TVM_TIRX_DLL PrimExpr clz(PrimExpr x, Location loc = UnknownLoc());
   TVM_TIRX_DLL PrimExpr min_value(PrimType dtype, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr max_value(PrimType dtype, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr cast(PrimExpr value, PrimType t, Location loc = UnknownLoc());
   TVM_TIRX_DLL PrimExpr IntegerAbs(PrimExpr x, Location loc = UnknownLoc());
   template <typename ValueType,
             typename = typename std::enable_if<(std::is_standard_layout<ValueType>::value &&
                                                 std::is_trivial<ValueType>::value) ||
                                                std::is_same<ValueType, ffi::BigInt>::value>::type>
   inline PrimExpr MakeConst(PrimType dtype, ValueType value, Location loc = UnknownLoc());
   
   template <typename ValueType>
   inline PrimExpr MakeConstScalar(PrimType dtype, ValueType value, Location loc = UnknownLoc()) {
     if constexpr (std::is_enum_v<ValueType>) {
       return MakeConstScalar(dtype, static_cast<std::underlying_type_t<ValueType>>(value), loc);
     } else {
       DLDataTypeCode code = dtype.code();
       if (code == DLDataTypeCode::kDLInt || code == DLDataTypeCode::kDLBool) {
         return IntImm(dtype, ffi::BigInt(value), loc);
       }
       if (code == DLDataTypeCode::kDLUInt) {
         TVM_FFI_ICHECK(value >= 0) << "cannot make uint from negative value " << value;
         return IntImm(dtype, ffi::BigInt(value), loc);
       }
       if (dtype.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLFloat8_e3m4,
                             DLDataTypeCode::kDLFloat8_e4m3, DLDataTypeCode::kDLFloat8_e4m3b11fnuz,
                             DLDataTypeCode::kDLFloat8_e4m3fn, DLDataTypeCode::kDLFloat8_e4m3fnuz,
                             DLDataTypeCode::kDLFloat8_e5m2, DLDataTypeCode::kDLFloat8_e5m2fnuz,
                             DLDataTypeCode::kDLFloat8_e8m0fnu, DLDataTypeCode::kDLFloat6_e2m3fn,
                             DLDataTypeCode::kDLFloat6_e3m2fn, DLDataTypeCode::kDLFloat4_e2m1fn) ||
           dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
         return FloatImm(dtype, static_cast<double>(value), loc);
       }
       TVM_FFI_THROW(InternalError) << "cannot make const for type " << dtype;
       throw;
     }
   }
   
   template <>
   inline PrimExpr MakeConstScalar(PrimType dtype, bool value, Location loc) {
     return MakeConstScalar(dtype, static_cast<int>(value), loc);
   }
   
   template <typename ValueType, typename>
   inline PrimExpr MakeConst(PrimType dtype, ValueType value, Location loc) {
     if (!dtype.IsScalableVector() && !dtype.IsFixedLengthVector()) {
       return MakeConstScalar(dtype, value, loc);
     }
     PrimType elem_ty = dtype.WithLanes(1);
     if (dtype.IsFixedLengthVector()) {
       return prim::Broadcast(MakeConstScalar(elem_ty, value, loc), dtype.lanes(), loc);
     }
     PrimExpr lanes = prim::Mul(Call(PrimType::Int(32), prim::vscale_op(), {}).as_or_throw<PrimExpr>(),
                                dtype.VScaleFactor());
     return prim::Broadcast(MakeConstScalar(elem_ty, value, loc), lanes, loc);
   }
   
   }  // namespace prim
   
   // additional const expression overloading
   #define TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(Name)                                                \
     inline PrimExpr Name(const PrimExpr& a, float b) { return Name(a, PrimExpr(b)); }              \
     inline PrimExpr Name(float a, const PrimExpr& b) { return Name(PrimExpr(a), b); }              \
     inline PrimExpr Name(int a, const PrimExpr& b) { return Name(prim::MakeConst(b.ty(), a), b); } \
     inline PrimExpr Name(const PrimExpr& a, int b) { return Name(a, prim::MakeConst(a.ty(), b)); } \
     inline PrimExpr Name(const PrimExpr& a, double b) {                                            \
       return Name(a, FloatImm(PrimType::Float(64), b));                                            \
     }
   
   #define TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(Name)                          \
     inline PrimExpr Name(const PrimExpr& a, float b, Location loc = UnknownLoc()) {  \
       return Name(a, PrimExpr(b), loc);                                              \
     }                                                                                \
     inline PrimExpr Name(float a, const PrimExpr& b, Location loc = UnknownLoc()) {  \
       return Name(PrimExpr(a), b, loc);                                              \
     }                                                                                \
     inline PrimExpr Name(int a, const PrimExpr& b, Location loc = UnknownLoc()) {    \
       return Name(prim::MakeConst(b.ty(), a), b, loc);                               \
     }                                                                                \
     inline PrimExpr Name(const PrimExpr& a, int b, Location loc = UnknownLoc()) {    \
       return Name(a, prim::MakeConst(a.ty(), b), loc);                               \
     }                                                                                \
     inline PrimExpr Name(const PrimExpr& a, double b, Location loc = UnknownLoc()) { \
       return Name(a, FloatImm(PrimType::Float(64), b), loc);                         \
     }
   
   #define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(Name)                             \
     inline PrimExpr Name(const PrimExpr& a, bool b) { return Name(a, PrimExpr(b)); } \
     inline PrimExpr Name(bool a, const PrimExpr& b) { return Name(PrimExpr(a), b); }
   
   #define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD_SPANNED(Name)                   \
     inline PrimExpr Name(const PrimExpr& a, bool b, Location loc = UnknownLoc()) { \
       return Name(a, PrimExpr(b), loc);                                            \
     }                                                                              \
     inline PrimExpr Name(bool a, const PrimExpr& b, Location loc = UnknownLoc()) { \
       return Name(PrimExpr(a), b, loc);                                            \
     }
   
   #define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(Name)                                               \
     inline PrimExpr Name(const PrimExpr& a, int b) { return Name(a, prim::MakeConst(a.ty(), b)); } \
     inline PrimExpr Name(int a, const PrimExpr& b) { return Name(prim::MakeConst(b.ty(), a), b); }
   
   #define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(Name)                      \
     inline PrimExpr Name(const PrimExpr& a, int b, Location loc = UnknownLoc()) { \
       return Name(a, prim::MakeConst(a.ty(), b), loc);                            \
     }                                                                             \
     inline PrimExpr Name(int a, const PrimExpr& b, Location loc = UnknownLoc()) { \
       return Name(prim::MakeConst(b.ty(), a), b, loc);                            \
     }
   
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator+);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator-);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator*);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>);  // NOLINT(*)
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>=);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<);  // NOLINT(*)
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<=);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(max);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(min);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(div);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(add);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(sub);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(mul);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(greater);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(greater_equal);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(less);
   TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(less_equal);
   // integer related ops
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(indexdiv);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(indexmod);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(truncdiv);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(truncmod);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(floordiv);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(floormod);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(right_shift);  // NOLINT(*)
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(left_shift);   // NOLINT(*)
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(bitwise_and);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(bitwise_or);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(bitwise_xor);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator>>);  // NOLINT(*)
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator<<);  // NOLINT(*)
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator&);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator|);
   TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator^);
   // logical ops
   TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator&&);
   TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator||);
   TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD_SPANNED(logical_and);
   TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD_SPANNED(logical_or);
   
   template <typename TA>
   inline void DivAmbiguityError(const TA& a) {
     constexpr bool div_ambiguity = !std::is_class<TA>::value;
     static_assert(div_ambiguity,
                   "TVM supports multiple types of integer divisions, "
                   "please call div, indexdiv/indexmod, "
                   "floordiv/floormod or truncdiv/truncmod directly "
                   "to avoid ambiguity in the code. "
                   "Checkout these functions in ir/prim/op.h.");
   }
   
   // The following code are not intended to be used in the codebase.
   // Instead, they generate clear compiler errors that ask developers
   // to use the specific division function.
   // The second template argument is necessary to make sure the
   // code compiles lazily by the compiler during invocation.
   template <typename TB>
   inline PrimExpr operator/(const PrimExpr& a, const TB& b) {
     DivAmbiguityError(a);
     return a;
   }
   
   template <typename TB>
   inline PrimExpr operator/=(const PrimExpr& a, const TB& b) {
     DivAmbiguityError(a);
     return a;
   }
   
   template <typename TB>
   inline PrimExpr operator%(const PrimExpr& a, const TB& b) {
     DivAmbiguityError(a);
     return a;
   }
   
   // Constant predicates
   namespace prim {
   
   inline bool IsConstInt(const PrimExpr& x, int64_t value);
   
   inline bool IsOne(const PrimExpr& x);
   
   inline bool IsZero(const PrimExpr& x);
   
   inline bool IsConstInt(const PrimExpr& x);
   
   inline bool IsConstNumber(const PrimExpr& x);
   
   TVM_TIRX_DLL bool IsPowerOfTwoInt(const PrimExpr& x, int* shift);
   
   inline bool IsPositiveConst(const PrimExpr& a);
   
   inline bool IsNegativeConst(const PrimExpr& a);
   
   // Inline constant predicate implementations.
   inline bool IsConstInt(const PrimExpr& x) { return x.as<IntImmNode>() != nullptr; }
   
   inline bool IsConstNumber(const PrimExpr& x) {
     if (x.as<IntImmNode>()) {
       return true;
     } else if (x.as<FloatImmNode>()) {
       return true;
     } else if (const auto* op = x.as<prim::BroadcastNode>()) {
       return (op->value->IsInstance<IntImmNode>() || op->value->IsInstance<FloatImmNode>());
     }
     return false;
   }
   
   inline bool IsPositiveConst(const PrimExpr& a) {
     const auto* as_int = a.as<IntImmNode>();
     return as_int && as_int->value > 0;
   }
   
   inline bool IsNegativeConst(const PrimExpr& a) {
     const auto* as_int = a.as<IntImmNode>();
     return as_int && as_int->value < 0;
   }
   
   inline bool IsConstInt(const PrimExpr& x, int64_t value) {
     const auto* as_int = x.as<IntImmNode>();
     return as_int && as_int->value == value;
   }
   
   inline bool IsOne(const PrimExpr& x) { return IsConstInt(x, 1); }
   
   inline bool IsZero(const PrimExpr& x) { return IsConstInt(x, 0); }
   
   TVM_TIRX_DLL PrimExpr infinity(PrimType dtype, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr pow(PrimExpr x, PrimExpr y, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr abs(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr isnan(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr isfinite(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr isinf(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr floor(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr round(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr nearbyint(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr trunc(PrimExpr x, Location loc = UnknownLoc());
   
   TVM_TIRX_DLL PrimExpr fmod(PrimExpr x, PrimExpr y, Location loc = UnknownLoc());
   TVM_TIRX_DLL const Op& pow_op();
   TVM_TIRX_DLL const Op& fabs_op();
   TVM_TIRX_DLL const Op& fmod_op();
   TVM_TIRX_DLL const Op& floor_op();
   TVM_TIRX_DLL const Op& round_op();
   TVM_TIRX_DLL const Op& nearbyint_op();
   TVM_TIRX_DLL const Op& trunc_op();
   TVM_TIRX_DLL const Op& exp_op();
   TVM_TIRX_DLL const Op& exp2_op();
   TVM_TIRX_DLL const Op& exp10_op();
   TVM_TIRX_DLL const Op& erf_op();
   TVM_TIRX_DLL const Op& tanh_op();
   TVM_TIRX_DLL const Op& sigmoid_op();
   TVM_TIRX_DLL const Op& sqrt_op();
   TVM_TIRX_DLL const Op& rsqrt_op();
   TVM_TIRX_DLL const Op& log_op();
   TVM_TIRX_DLL const Op& log1p_op();
   TVM_TIRX_DLL const Op& log10_op();
   TVM_TIRX_DLL const Op& tan_op();
   TVM_TIRX_DLL const Op& cos_op();
   TVM_TIRX_DLL const Op& cosh_op();
   TVM_TIRX_DLL const Op& sin_op();
   TVM_TIRX_DLL const Op& sinh_op();
   TVM_TIRX_DLL const Op& asin_op();
   TVM_TIRX_DLL const Op& acos_op();
   TVM_TIRX_DLL const Op& atan_op();
   TVM_TIRX_DLL const Op& acosh_op();
   TVM_TIRX_DLL const Op& asinh_op();
   TVM_TIRX_DLL const Op& atanh_op();
   TVM_TIRX_DLL const Op& atan2_op();
   TVM_TIRX_DLL const Op& nextafter_op();
   TVM_TIRX_DLL const Op& hypot_op();
   TVM_TIRX_DLL const Op& copysign_op();
   TVM_TIRX_DLL const Op& ldexp_op();
   TVM_TIRX_DLL const Op& isnan_op();
   TVM_TIRX_DLL const Op& popcount_op();
   TVM_TIRX_DLL const Op& fma_op();
   TVM_TIRX_DLL const Op& assume_op();
   TVM_TIRX_DLL PrimExpr assume(PrimExpr condition, Location loc = UnknownLoc());
   
   inline PrimExpr fma(PrimExpr x, PrimExpr y, PrimExpr z, Location loc = UnknownLoc()) {
     return Call(x.ty(), fma_op(), {x, y, z}, {}, {}, loc).as_or_throw<PrimExpr>();
   }
   
   inline void CheckMathUnaryOpInputDType(const char* op_name, const PrimType& dtype) {
     TVM_FFI_CHECK(dtype.code() == DLDataTypeCode::kDLFloat ||
                       dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16),
                   TypeError)
         << "prim." << op_name << " only supports floating-point inputs, but got " << dtype;
   }
   
   // Intrinsic operators
   #define TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, CheckInputDType)                          \
     inline PrimExpr OpName(PrimExpr x, Location loc = UnknownLoc()) {                           \
       static const Op op = Op::Get("prim." #OpName);                                            \
       PrimType x_ty = x.ty();                                                                   \
       CheckInputDType(#OpName, x_ty);                                                           \
       if (x_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {                             \
         PrimType bf16_ty = x_ty;                                                                \
         PrimType f32_ty =                                                                       \
             x_ty.IsScalableVector()                                                             \
                 ? PrimType::ScalableVector(DLDataTypeCode::kDLFloat, 32, x_ty.VScaleFactor())   \
                 : PrimType::Float(32, x_ty.lanes());                                            \
         PrimExpr x_fp32 = prim::Cast(f32_ty, x, loc);                                           \
         PrimExpr result_fp32 = Call(f32_ty, op, {x_fp32}, {}, {}, loc).as_or_throw<PrimExpr>(); \
         return prim::Cast(bf16_ty, result_fp32, loc);                                           \
       } else {                                                                                  \
         return Call(x_ty, op, {x}, {}, {}, loc).as_or_throw<PrimExpr>();                        \
       }                                                                                         \
     }
   
   #define TVM_DECLARE_INTRIN_UNARY(OpName) \
     TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, [](const char*, const PrimType&) {})
   
   #define TVM_DECLARE_FLOAT_INTRIN_UNARY(OpName) \
     TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, CheckMathUnaryOpInputDType)
   
   TVM_DECLARE_INTRIN_UNARY(exp);
   TVM_DECLARE_INTRIN_UNARY(exp2);
   TVM_DECLARE_INTRIN_UNARY(exp10);
   TVM_DECLARE_INTRIN_UNARY(erf);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(tanh);
   TVM_DECLARE_INTRIN_UNARY(sigmoid);
   TVM_DECLARE_INTRIN_UNARY(sqrt);
   TVM_DECLARE_INTRIN_UNARY(rsqrt);
   TVM_DECLARE_INTRIN_UNARY(log);
   TVM_DECLARE_INTRIN_UNARY(log10);
   TVM_DECLARE_INTRIN_UNARY(log1p);
   TVM_DECLARE_INTRIN_UNARY(popcount);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(tan);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(cos);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(cosh);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(sin);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(sinh);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(asin);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(acos);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(atan);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(acosh);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(asinh);
   TVM_DECLARE_FLOAT_INTRIN_UNARY(atanh);
   
   #define TVM_DECLARE_INTRIN_BINARY(OpName)                                       \
     inline PrimExpr OpName(PrimExpr x, PrimExpr y, Location loc = UnknownLoc()) { \
       static const Op op = Op::Get("prim." #OpName);                              \
       return Call(x.ty(), op, {x, y}, {}, {}, loc).as_or_throw<PrimExpr>();       \
     }
   
   TVM_DECLARE_INTRIN_BINARY(atan2);
   TVM_DECLARE_INTRIN_BINARY(nextafter);
   TVM_DECLARE_INTRIN_BINARY(copysign);
   TVM_DECLARE_INTRIN_BINARY(hypot);
   TVM_DECLARE_INTRIN_BINARY(ldexp);
   
   template <typename FReduce>
   inline PrimExpr foldl(FReduce freduce, PrimExpr init_value, const ffi::Array<PrimExpr>& values,
                         Location loc = UnknownLoc()) {
     for (PrimExpr val : values) {
       init_value = freduce(init_value, val, loc);
     }
     return init_value;
   }
   
   }  // namespace prim
   
   }  // namespace tvm
   #endif  // TVM_IR_PRIM_OP_H_
