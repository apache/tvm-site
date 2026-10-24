
.. _program_listing_file_tvm_ir_prim_pattern_match.h:

Program Listing for File pattern_match.h
========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_prim_pattern_match.h>` (``tvm/ir/prim/pattern_match.h``)

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
   
   #ifndef TVM_IR_PRIM_PATTERN_MATCH_H_
   #define TVM_IR_PRIM_PATTERN_MATCH_H_
   
   #include <tvm/ir/pattern_match.h>
   #include <tvm/ir/prim/const_fold.h>
   #include <tvm/ir/prim/expr.h>
   #include <tvm/ir/prim/op.h>
   
   #include <cmath>
   #include <tuple>
   
   namespace tvm {
   template <>
   class PEqualChecker<PrimExpr> {
    public:
     bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
       if (lhs.same_as(rhs)) return true;
       return prim::ExprDeepEqual()(lhs, rhs);
     }
   };
   
   template <>
   class PEqualChecker<IntImm> {
    public:
     bool operator()(const IntImm& lhs, const IntImm& rhs) const { return lhs->value == rhs->value; }
   };
   
   template <>
   class PEqualChecker<FloatImm> {
    public:
     bool operator()(const FloatImm& lhs, const FloatImm& rhs) const {
       return std::fabs(lhs->value - rhs->value) < 1e-20;
     }
   };
   
   template <>
   class PEqualChecker<Var> {
    public:
     bool operator()(const Var& lhs, const Var& rhs) const { return lhs.same_as(rhs); }
   };
   
   template <typename T, typename DType,
             typename = std::enable_if<std::is_base_of<T, PrimExpr>::value>>
   class PVarWithDataType : public PVarWithCheck<PVarWithDataType<T, DType>, T> {
    public:
     explicit PVarWithDataType(const DType& dtype) : dtype_(dtype) {}
   
     bool Match_(const T& value) const { return dtype_.Match_(value.ty()->dtype); }
   
    protected:
     typename DType::Nested dtype_;
   };
   
   class PVecDataType : public PVarWithCheck<PVecDataType, DLDataType> {
    public:
     explicit PVecDataType(DLDataType elem_dtype) : elem_dtype_(elem_dtype) {}
   
     bool Match_(DLDataType dtype) const { return dtype.code == elem_dtype_.code; }
   
    protected:
     DLDataType elem_dtype_;
   };
   
   template <typename OpType, typename TA, typename TB, bool kFoldConstants = true>
   class PBinaryExpr : public Pattern<PBinaryExpr<OpType, TA, TB, kFoldConstants>> {
    public:
     PBinaryExpr(const TA& a, const TB& b) : a_(a), b_(b) {}
   
     void InitMatch_() const {
       a_.InitMatch_();
       b_.InitMatch_();
     }
   
     bool Match_(const ffi::ObjectRef& node) const {
       using NodeType = typename OpType::ContainerType;
       if (const NodeType* ptr = node.as<NodeType>()) {
         if (!a_.Match_(ptr->a)) return false;
         if (!b_.Match_(ptr->b)) return false;
         return true;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const {
       PrimExpr lhs = a_.Eval();
       PrimExpr rhs = b_.Eval();
       if constexpr (kFoldConstants) {
         if (auto ret = prim::TryConstFold<OpType>(lhs, rhs)) return ret.value();
       }
       return OpType(lhs, rhs);
     }
   
    private:
     typename TA::Nested a_;
     typename TB::Nested b_;
   };
   
   template <typename TA>
   class PConstWithTypeLike : public Pattern<PConstWithTypeLike<TA>> {
    public:
     PConstWithTypeLike(const TA& ref, int64_t value) : ref_(ref), value_(value) {}
   
     void InitMatch_() const {}
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const IntImmObj* ptr = node.as<IntImmObj>()) {
         return ptr->value == value_;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const { return prim::MakeConst(ref_.Eval().ty(), value_); }
   
    private:
     typename TA::Nested ref_;
     int64_t value_;
   };
   
   #define TVM_PATTERN_BINARY_OP_EX(FuncName, NodeName, CheckStep)                               \
     template <typename TA, typename TB>                                                         \
     inline PBinaryExpr<NodeName, TA, TB> FuncName(const Pattern<TA>& a, const Pattern<TB>& b) { \
       CheckStep;                                                                                \
       return PBinaryExpr<NodeName, TA, TB>(a.derived(), b.derived());                           \
     }                                                                                           \
     template <typename TA>                                                                      \
     inline PBinaryExpr<NodeName, TA, PConstWithTypeLike<TA>> FuncName(const Pattern<TA>& a,     \
                                                                       int64_t b) {              \
       CheckStep;                                                                                \
       return FuncName(a, PConstWithTypeLike<TA>(a.derived(), b));                               \
     }                                                                                           \
     template <typename TA>                                                                      \
     inline PBinaryExpr<NodeName, PConstWithTypeLike<TA>, TA> FuncName(int64_t b,                \
                                                                       const Pattern<TA>& a) {   \
       CheckStep;                                                                                \
       return FuncName(PConstWithTypeLike<TA>(a.derived(), b), a);                               \
     }
   
   #define TVM_PATTERN_BINARY_OP(FuncName, NodeName) TVM_PATTERN_BINARY_OP_EX(FuncName, NodeName, )
   
   // raise ambiguity error for operator overload of / and %
   TVM_PATTERN_BINARY_OP_EX(operator/, prim::Div, DivAmbiguityError(a));
   TVM_PATTERN_BINARY_OP_EX(operator%, prim::Mod, DivAmbiguityError(a));
   
   // arithmetic expressions
   TVM_PATTERN_BINARY_OP(operator+, prim::Add);
   TVM_PATTERN_BINARY_OP(operator-, prim::Sub);
   TVM_PATTERN_BINARY_OP(operator*, prim::Mul);
   TVM_PATTERN_BINARY_OP(min, prim::Min);
   TVM_PATTERN_BINARY_OP(max, prim::Max);
   TVM_PATTERN_BINARY_OP(div, prim::Div);
   TVM_PATTERN_BINARY_OP(truncdiv, prim::Div);
   TVM_PATTERN_BINARY_OP(truncmod, prim::Mod);
   TVM_PATTERN_BINARY_OP(floordiv, prim::FloorDiv);
   TVM_PATTERN_BINARY_OP(floormod, prim::FloorMod);
   
   // Bitwise patterns preserve nonfolding construction during evaluation.
   #define TVM_PATTERN_BITWISE_OP(FuncName, NodeName)                             \
     template <typename TA, typename TB>                                          \
     inline PBinaryExpr<NodeName, TA, TB, false> FuncName(const Pattern<TA>& a,   \
                                                          const Pattern<TB>& b) { \
       return PBinaryExpr<NodeName, TA, TB, false>(a.derived(), b.derived());     \
     }
   
   TVM_PATTERN_BITWISE_OP(operator<<, prim::LShift);
   TVM_PATTERN_BITWISE_OP(operator>>, prim::RShift);
   TVM_PATTERN_BITWISE_OP(operator&, prim::BitwiseAnd);
   TVM_PATTERN_BITWISE_OP(operator|, prim::BitwiseOr);
   TVM_PATTERN_BITWISE_OP(operator^, prim::BitwiseXor);
   
   #undef TVM_PATTERN_BITWISE_OP
   
   // logical expressions
   TVM_PATTERN_BINARY_OP(operator>, prim::GT);
   TVM_PATTERN_BINARY_OP(operator>=, prim::GE);
   TVM_PATTERN_BINARY_OP(operator<, prim::LT);
   TVM_PATTERN_BINARY_OP(operator<=, prim::LE);
   TVM_PATTERN_BINARY_OP(operator==, prim::EQ);
   TVM_PATTERN_BINARY_OP(operator!=, prim::NE);
   TVM_PATTERN_BINARY_OP(operator&&, prim::And);
   TVM_PATTERN_BINARY_OP(operator||, prim::Or);
   
   template <typename TA>
   class PNotExpr : public Pattern<PNotExpr<TA>> {
    public:
     explicit PNotExpr(const TA& value) : value_(value) {}
   
     void InitMatch_() const { value_.InitMatch_(); }
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const prim::NotObj* ptr = node.as<prim::NotObj>()) {
         if (!value_.Match_(ptr->a)) return false;
         return true;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const { return prim::Not(value_.Eval()); }
   
    private:
     typename TA::Nested value_;
   };
   
   template <typename TA>
   inline PNotExpr<TA> operator!(const Pattern<TA>& value) {
     return PNotExpr<TA>(value.derived());
   }
   
   template <typename TA>
   class PBitwiseNotExpr : public Pattern<PBitwiseNotExpr<TA>> {
    public:
     explicit PBitwiseNotExpr(const TA& value) : value_(value) {}
   
     void InitMatch_() const { value_.InitMatch_(); }
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const prim::BitwiseNotObj* ptr = node.as<prim::BitwiseNotObj>()) {
         if (!value_.Match_(ptr->a)) return false;
         return true;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const { return prim::BitwiseNot(value_.Eval()); }
   
    private:
     typename TA::Nested value_;
   };
   
   template <typename TA>
   inline PBitwiseNotExpr<TA> operator~(const Pattern<TA>& value) {
     return PBitwiseNotExpr<TA>(value.derived());
   }
   
   // select
   template <typename TCond, typename TA, typename TB>
   class PSelectExpr : public Pattern<PSelectExpr<TCond, TA, TB>> {
    public:
     PSelectExpr(const TCond& condition, const TA& true_value, const TB& false_value)
         : condition_(condition), true_value_(true_value), false_value_(false_value) {}
   
     void InitMatch_() const {
       condition_.InitMatch_();
       true_value_.InitMatch_();
       false_value_.InitMatch_();
     }
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const prim::SelectObj* ptr = node.as<prim::SelectObj>()) {
         if (!condition_.Match_(ptr->condition)) return false;
         if (!true_value_.Match_(ptr->true_value)) return false;
         if (!false_value_.Match_(ptr->false_value)) return false;
         return true;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const {
       return prim::Select(condition_.Eval(), true_value_.Eval(), false_value_.Eval());
     }
   
    private:
     typename TCond::Nested condition_;
     typename TA::Nested true_value_;
     typename TB::Nested false_value_;
   };
   
   template <typename TCond, typename TA, typename TB>
   inline PSelectExpr<TCond, TA, TB> select(const Pattern<TCond>& condition,
                                            const Pattern<TA>& true_value,
                                            const Pattern<TB>& false_value) {
     return PSelectExpr<TCond, TA, TB>(condition.derived(), true_value.derived(),
                                       false_value.derived());
   }
   
   template <typename DType, typename TA>
   class PCastExpr : public Pattern<PCastExpr<DType, TA>> {
    public:
     PCastExpr(const TA& value, const DType& dtype) : dtype_(dtype), value_(value) {}
   
     void InitMatch_() const {
       dtype_.InitMatch_();
       value_.InitMatch_();
     }
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const prim::CastObj* ptr = node.as<prim::CastObj>()) {
         if (!dtype_.Match_(ptr->ty.as_or_throw<PrimType>()->dtype)) return false;
         if (!value_.Match_(ptr->value)) return false;
         return true;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const { return prim::Cast(PrimType(dtype_.Eval()), value_.Eval()); }
   
    private:
     typename DType::Nested dtype_;
     typename TA::Nested value_;
   };
   
   template <typename DType, typename TA>
   inline PCastExpr<DType, TA> cast(const Pattern<TA>& value, const Pattern<DType>& dtype) {
     return PCastExpr<DType, TA>(value.derived(), dtype.derived());
   }
   
   template <typename TBase, typename TStride, typename TLanes>
   class PRampExpr : public Pattern<PRampExpr<TBase, TStride, TLanes>> {
    public:
     PRampExpr(const TBase& base, const TStride& stride, const TLanes& lanes)
         : base_(base), stride_(stride), lanes_(lanes) {}
   
     void InitMatch_() const {
       base_.InitMatch_();
       stride_.InitMatch_();
       lanes_.InitMatch_();
     }
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const prim::RampObj* ptr = node.as<prim::RampObj>()) {
         if (!base_.Match_(ptr->base)) return false;
         if (!stride_.Match_(ptr->stride)) return false;
         if (!lanes_.Match_(ptr->lanes)) return false;
         return true;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const { return prim::Ramp(base_.Eval(), stride_.Eval(), lanes_.Eval()); }
   
    private:
     typename TBase::Nested base_;
     typename TStride::Nested stride_;
     typename TLanes::Nested lanes_;
   };
   
   template <typename TBase, typename TStride, typename TLanes>
   inline PRampExpr<TBase, TStride, TLanes> ramp(const Pattern<TBase>& base,
                                                 const Pattern<TStride>& stride,
                                                 const Pattern<TLanes>& lanes) {
     return PRampExpr<TBase, TStride, TLanes>(base.derived(), stride.derived(), lanes.derived());
   }
   
   template <typename TBase>
   inline PRampExpr<TBase, PConstWithTypeLike<TBase>, PConstWithTypeLike<TBase>> ramp(
       const Pattern<TBase>& base, int stride, int lanes) {
     return PRampExpr<TBase, PConstWithTypeLike<TBase>, PConstWithTypeLike<TBase>>(
         base.derived(), PConstWithTypeLike<TBase>(base.derived(), stride),
         PConstWithTypeLike<TBase>(base.derived(), lanes));
   }
   
   template <typename TA, typename TLanes>
   class PBroadcastExpr : public Pattern<PBroadcastExpr<TA, TLanes>> {
    public:
     PBroadcastExpr(const TA& value, const TLanes& lanes) : value_(value), lanes_(lanes) {}
   
     void InitMatch_() const {
       value_.InitMatch_();
       lanes_.InitMatch_();
     }
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const prim::BroadcastObj* ptr = node.as<prim::BroadcastObj>()) {
         if (!value_.Match_(ptr->value)) return false;
         if (!lanes_.Match_(ptr->lanes)) return false;
         return true;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const { return prim::Broadcast(value_.Eval(), lanes_.Eval()); }
   
    private:
     typename TA::Nested value_;
     typename TLanes::Nested lanes_;
   };
   
   template <typename TA, typename TLanes>
   inline PBroadcastExpr<TA, TLanes> broadcast(const Pattern<TA>& value,
                                               const Pattern<TLanes>& lanes) {
     return PBroadcastExpr<TA, TLanes>(value.derived(), lanes.derived());
   }
   
   // internal namespace
   namespace detail {
   // implementation details for  CallExpr
   template <bool stop, std::size_t I, typename F>
   struct tuple_for_each_dispatcher {
     template <typename TTuple>
     static void run(F& f, const TTuple& tuple) {  // NOLINT(*)
       f(I, std::get<I>(tuple));
       tuple_for_each_dispatcher<(I + 1) == std::tuple_size<TTuple>::value, (I + 1), F>::run(f, tuple);
     }
   };
   
   template <std::size_t I, typename F>
   struct tuple_for_each_dispatcher<true, I, F> {
     template <typename TTuple>
     static void run(F& f, const TTuple& tuple) {}  // NOLINT(*)
   };
   
   template <typename F, typename TTuple>
   inline void tuple_for_each(F& f, const TTuple& tuple) {  // NOLINT(*)
     tuple_for_each_dispatcher<std::tuple_size<TTuple>::value == 0, 0, F>::run(f, tuple);
   }
   
   struct PCallExprInitMatchFunctor {
     template <typename T>
     void operator()(size_t i, const T& pattern) const {
       pattern.InitMatch_();
     }
   };
   
   struct PCallExprMatchFunctor {
     const CallObj* call_;
     bool matched_{true};
   
     explicit PCallExprMatchFunctor(const CallObj* call) : call_(call) {}
   
     template <typename T>
     void operator()(size_t i, const T& pattern) {
       matched_ = matched_ && pattern.Match_(call_->args[i]);
     }
   };
   
   struct PCallExprEvalArgsFunctor {
     ffi::Array<PrimExpr> args_;
   
     template <typename T>
     void operator()(size_t i, const T& pattern) {
       args_.push_back(pattern.Eval());
     }
   };
   }  // namespace detail
   
   template <typename Op, typename... TArgs>
   class PCallExpr : public Pattern<PCallExpr<Op, TArgs...>> {
    public:
     explicit PCallExpr(const TArgs&... args) : args_(args...) {}
   
     void InitMatch_() const {
       detail::PCallExprInitMatchFunctor finit;
       detail::tuple_for_each(finit, args_);
     }
   
     bool Match_(const ffi::ObjectRef& node) const {
       if (const CallObj* ptr = node.as<CallObj>()) {
         if (ptr->args.size() != sizeof...(TArgs)) return false;
         if (!ptr->op.same_as(Op::GetOp())) return false;
         detail::PCallExprMatchFunctor fmatch(ptr);
         detail::tuple_for_each(fmatch, args_);
         return fmatch.matched_;
       } else {
         return false;
       }
     }
   
     PrimExpr Eval() const {
       detail::PCallExprEvalArgsFunctor feval_args;
       detail::tuple_for_each(feval_args, args_);
       return Op::Eval(feval_args.args_);
     }
   
    private:
     std::tuple<typename TArgs::Nested...> args_;
   };
   
   // if_then_else
   struct PIfThenElseOp {
     static PrimExpr Eval(ffi::Array<PrimExpr> args) {
       return Call(args[1].ty(), GetOp(), args).as_or_throw<PrimExpr>();
     }
     static const Op& GetOp() { return prim::if_then_else_op(); }
   };
   
   template <typename TCond, typename TA, typename TB>
   inline PCallExpr<PIfThenElseOp, TCond, TA, TB> if_then_else(const Pattern<TCond>& cond,
                                                               const Pattern<TA>& true_value,
                                                               const Pattern<TB>& false_value) {
     return PCallExpr<PIfThenElseOp, TCond, TA, TB>(cond.derived(), true_value.derived(),
                                                    false_value.derived());
   }
   
   }  // namespace tvm
   #endif  // TVM_IR_PRIM_PATTERN_MATCH_H_
