
.. _program_listing_file_tvm_sym_constraint.h:

Program Listing for File constraint.h
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_sym_constraint.h>` (``tvm/sym/constraint.h``)

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
   
   #ifndef TVM_SYM_CONSTRAINT_H_
   #define TVM_SYM_CONSTRAINT_H_
   
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/prim/expr.h>
   
   #include <utility>
   
   namespace tvm {
   namespace sym {
   
   class ConstraintObj : public ffi::Object {
    public:
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO("sym.Constraint", ConstraintObj, ffi::Object);
   };
   
   class Constraint : public ffi::ObjectRef {
    public:
     explicit Constraint(ffi::ObjectPtr<ConstraintObj> node) : ffi::ObjectRef(std::move(node)) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Constraint, ffi::ObjectRef, ConstraintObj);
   };
   
   class BindingConstraintObj : public ConstraintObj {
    public:
     BindingConstraintObj(Var var, PrimExpr value) : var(std::move(var)), value(std::move(value)) {}
     explicit BindingConstraintObj(ffi::UnsafeInit tag) : var(tag), value(tag) {}
     Var var;
     PrimExpr value;
   
     static void RegisterReflection() {
       namespace refl = ffi::reflection;
       refl::ObjectDef<BindingConstraintObj>()
           .def_ro("var", &BindingConstraintObj::var)
           .def_ro("value", &BindingConstraintObj::value);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.BindingConstraint", BindingConstraintObj, ConstraintObj);
   };
   
   class BindingConstraint : public Constraint {
    public:
     BindingConstraint(Var var, PrimExpr value)
         : Constraint(ffi::make_object<BindingConstraintObj>(std::move(var), std::move(value))) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BindingConstraint, Constraint,
                                                   BindingConstraintObj);
   };
   
   class RangeConstraintObj : public ConstraintObj {
    public:
     RangeConstraintObj(Var var, Range range) : var(std::move(var)), range(std::move(range)) {}
     explicit RangeConstraintObj(ffi::UnsafeInit tag) : var(tag), range(tag) {}
     Var var;
     Range range;
   
     static void RegisterReflection() {
       namespace refl = ffi::reflection;
       refl::ObjectDef<RangeConstraintObj>()
           .def_ro("var", &RangeConstraintObj::var)
           .def_ro("range", &RangeConstraintObj::range);
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.RangeConstraint", RangeConstraintObj, ConstraintObj);
   };
   
   class RangeConstraint : public Constraint {
    public:
     RangeConstraint(Var var, Range range)
         : Constraint(ffi::make_object<RangeConstraintObj>(std::move(var), std::move(range))) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RangeConstraint, Constraint, RangeConstraintObj);
   };
   
   class PredicateConstraintObj : public ConstraintObj {
    public:
     explicit PredicateConstraintObj(PrimExpr condition, bool assume_always = false)
         : condition(std::move(condition)), assume_always(assume_always) {}
     explicit PredicateConstraintObj(ffi::UnsafeInit tag) : condition(tag) {}
     PrimExpr condition;
     bool assume_always{false};
   
     static void RegisterReflection() {
       namespace refl = ffi::reflection;
       refl::ObjectDef<PredicateConstraintObj>()
           .def_ro("condition", &PredicateConstraintObj::condition)
           .def_ro("assume_always", &PredicateConstraintObj::assume_always, refl::DefaultValue(false));
     }
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.PredicateConstraint", PredicateConstraintObj,
                                       ConstraintObj);
   };
   
   class PredicateConstraint : public Constraint {
    public:
     explicit PredicateConstraint(PrimExpr condition, bool assume_always = false)
         : Constraint(ffi::make_object<PredicateConstraintObj>(std::move(condition), assume_always)) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PredicateConstraint, Constraint,
                                                   PredicateConstraintObj);
   };
   
   class ConstraintContext;
   
   class ConstraintContextObj : public ffi::Object {
    public:
     ConstraintContextObj() = default;
     ConstraintContextObj(ffi::ObjectPtr<ConstraintContextObj> parent, int64_t parent_local_prefix)
         : parent_(std::move(parent)), parent_prefix_size_(parent_local_prefix) {
       TVM_FFI_CHECK(
           parent_ && 0 <= parent_local_prefix && parent_local_prefix <= parent_->LocalSize(),
           ValueError)
           << "Constraint context parent local prefix is out of bounds";
     }
     // Context nodes have stable identity; use Snapshot to capture their current position.
     ConstraintContextObj(const ConstraintContextObj&) = delete;
     ConstraintContextObj& operator=(const ConstraintContextObj&) = delete;
   
     void Append(const Constraint& constraint) { constraints_.push_back(constraint); }
   
     int64_t LocalSize() const { return constraints_.size(); }
   
     ConstraintContext Snapshot() const;
   
     template <typename F>
     void ForEach(F&& f) const {
       // Keep this object alive if a callback replaces the caller's last handle.
       auto context = ffi::GetObjectPtr<ConstraintContextObj>(const_cast<ConstraintContextObj*>(this));
       context->ForEach(context->LocalSize(), std::forward<F>(f));
     }
   
     ffi::Array<Constraint> ToArray() const { return ToArray(LocalSize()); }
   
     static constexpr bool _type_mutable = true;
     static void RegisterReflection();
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ConstraintContext", ConstraintContextObj, ffi::Object);
   
    private:
     friend class AnalyzerObj;
     // Materialize the fixed inherited sequence and the requested local prefix.
     ffi::Array<Constraint> ToArray(int64_t local_prefix) const {
       if (!parent_) {
         if (local_prefix == LocalSize()) return constraints_;
         return ffi::Array<Constraint>(constraints_.begin(), constraints_.begin() + local_prefix);
       }
       auto result = parent_->ToArray(parent_prefix_size_);
       if (local_prefix != 0) {
         result.insert(result.end(), constraints_.begin(), constraints_.begin() + local_prefix);
       }
       return result;
     }
   
     // Traverse a fixed prefix even if callbacks append more local constraints.
     template <typename F>
     void ForEach(int64_t local_prefix, F&& f) const {
       if (parent_) parent_->ForEach(parent_prefix_size_, f);
       for (int64_t i = 0; i < local_prefix; ++i) {
         // Read by value: a callback may append and reallocate the local array.
         const Constraint constraint = constraints_[i];
         f(constraint);
       }
     }
   
     // Fixed parent context; null for a root. Parent links are acyclic.
     const ffi::ObjectPtr<ConstraintContextObj> parent_;
     // Fixed prefix length in the parent's LOCAL sequence, not its effective count.
     const int64_t parent_prefix_size_{0};
     // Append-only local sequence; captured entries cannot be replaced or mutated through aliases.
     ffi::Array<Constraint> constraints_;
   };
   
   class ConstraintContext : public ffi::ObjectRef {
    public:
     ConstraintContext() : ffi::ObjectRef(ffi::make_object<ConstraintContextObj>()) {}
     ConstraintContext(ConstraintContext parent, int64_t parent_local_prefix)
         : ffi::ObjectRef(ffi::make_object<ConstraintContextObj>(
               ffi::GetObjectPtr<ConstraintContextObj>(parent.get()), parent_local_prefix)) {}
     explicit ConstraintContext(ffi::ObjectPtr<ConstraintContextObj> node)
         : ffi::ObjectRef(std::move(node)) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConstraintContext, ffi::ObjectRef,
                                                   ConstraintContextObj);
   };
   
   inline ConstraintContext ConstraintContextObj::Snapshot() const {
     return ConstraintContext(ffi::GetRef<ConstraintContext>(this), LocalSize());
   }
   
   }  // namespace sym
   }  // namespace tvm
   #endif  // TVM_SYM_CONSTRAINT_H_
