
.. _program_listing_file_tvm_ffi_reflection_access_path.h:

Program Listing for File access_path.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_reflection_access_path.h>` (``tvm/ffi/reflection/access_path.h``)

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
   #ifndef TVM_FFI_REFLECTION_ACCESS_PATH_H_
   #define TVM_FFI_REFLECTION_ACCESS_PATH_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/container/array.h>
   #include <tvm/ffi/container/tuple.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/reflection/registry.h>
   
   #include <vector>
   
   namespace tvm {
   namespace ffi {
   namespace reflection {
   
   enum class AccessKind : int32_t {
     kAttr = 0,
     kArrayItem = 1,
     kMapItem = 2,
     // the following two are used for error reporting when
     // the supposed access field is not available
     kAttrMissing = 3,
     kArrayItemMissing = 4,
     kMapItemMissing = 5,
   };
   
   class AccessStep;
   
   class AccessStepObj : public Object {
    public:
     AccessKind kind;
     Any key;
   
     // default constructor to enable auto-serialization
     AccessStepObj() = default;
     AccessStepObj(AccessKind kind, Any key) : kind(kind), key(key) {}
   
     inline bool StepEqual(const AccessStep& other) const;
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindConstTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.reflection.AccessStep", AccessStepObj, Object);
   };
   
   class AccessStep : public ObjectRef {
    public:
     AccessStep(AccessKind kind, Any key) : ObjectRef(make_object<AccessStepObj>(kind, key)) {}
   
     static AccessStep Attr(String field_name) { return AccessStep(AccessKind::kAttr, field_name); }
   
     static AccessStep AttrMissing(String field_name) {
       return AccessStep(AccessKind::kAttrMissing, field_name);
     }
   
     static AccessStep ArrayItem(int64_t index) { return AccessStep(AccessKind::kArrayItem, index); }
   
     static AccessStep ArrayItemMissing(int64_t index) {
       return AccessStep(AccessKind::kArrayItemMissing, index);
     }
   
     static AccessStep MapItem(Any key) { return AccessStep(AccessKind::kMapItem, key); }
   
     static AccessStep MapItemMissing(Any key = nullptr) {
       return AccessStep(AccessKind::kMapItemMissing, key);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AccessStep, ObjectRef, AccessStepObj);
   };
   
   inline bool AccessStepObj::StepEqual(const AccessStep& other) const {
     return this->kind == other->kind && AnyEqual()(this->key, other->key);
   }
   
   // forward declaration
   class AccessPath;
   
   class AccessPathObj : public Object {
    public:
     Optional<ObjectRef> parent;
     Optional<AccessStep> step;
     int32_t depth;
   
     // default constructor to enable auto-serialization
     AccessPathObj() = default;
     AccessPathObj(Optional<ObjectRef> parent, Optional<AccessStep> step, int32_t depth)
         : parent(parent), step(step), depth(depth) {}
   
     inline Optional<AccessPath> GetParent() const;
   
     inline AccessPath Extend(AccessStep step) const;
   
     inline AccessPath Attr(String field_name) const;
   
     inline AccessPath AttrMissing(String field_name) const;
   
     inline AccessPath ArrayItem(int64_t index) const;
   
     inline AccessPath ArrayItemMissing(int64_t index) const;
   
     inline AccessPath MapItem(Any key) const;
   
     inline AccessPath MapItemMissing(Any key) const;
   
     inline Array<AccessStep> ToSteps() const;
   
     inline bool PathEqual(const AccessPath& other) const;
   
     inline bool IsPrefixOf(const AccessPath& other) const;
   
     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindConstTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.reflection.AccessPath", AccessPathObj, Object);
   
    private:
     static bool PathEqual(const AccessPathObj* lhs, const AccessPathObj* rhs) {
       // fast path for same pointer
       if (lhs == rhs) return true;
       if (lhs->depth != rhs->depth) return false;
       // do deep equality checks
       while (lhs->parent.has_value()) {
         TVM_FFI_ICHECK(rhs->parent.has_value());
         TVM_FFI_ICHECK(lhs->step.has_value());
         TVM_FFI_ICHECK(rhs->step.has_value());
         if (!(*lhs->step)->StepEqual(*(rhs->step))) {
           return false;
         }
         lhs = static_cast<const AccessPathObj*>(lhs->parent.get());
         rhs = static_cast<const AccessPathObj*>(rhs->parent.get());
         // fast path for same pointer
         if (lhs == rhs) return true;
         TVM_FFI_ICHECK(lhs != nullptr);
         TVM_FFI_ICHECK(rhs != nullptr);
       }
       return true;
     }
   };
   
   class AccessPath : public ObjectRef {
    public:
     template <typename Iter>
     static AccessPath FromSteps(Iter begin, Iter end) {
       AccessPath path = AccessPath::Root();
       for (Iter it = begin; it != end; ++it) {
         path = path->Extend(*it);
       }
       return path;
     }
     static AccessPath FromSteps(Array<AccessStep> steps) {
       AccessPath path = AccessPath::Root();
       for (AccessStep step : steps) {
         path = path->Extend(step);
       }
       return path;
     }
   
     static AccessPath Root() {
       return AccessPath(make_object<AccessPathObj>(std::nullopt, std::nullopt, 0));
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AccessPath, ObjectRef, AccessPathObj);
   
    private:
     friend class AccessPathObj;
     explicit AccessPath(ObjectPtr<AccessPathObj> ptr) : ObjectRef(ptr) {}
   };
   
   using AccessPathPair = Tuple<AccessPath, AccessPath>;
   
   inline Optional<AccessPath> AccessPathObj::GetParent() const {
     if (auto opt_parent = this->parent.as<AccessPath>()) {
       return opt_parent;
     }
     return std::nullopt;
   }
   
   inline AccessPath AccessPathObj::Extend(AccessStep step) const {
     return AccessPath(make_object<AccessPathObj>(GetRef<AccessPath>(this), step, this->depth + 1));
   }
   
   inline AccessPath AccessPathObj::Attr(String field_name) const {
     return this->Extend(AccessStep::Attr(field_name));
   }
   
   inline AccessPath AccessPathObj::AttrMissing(String field_name) const {
     return this->Extend(AccessStep::AttrMissing(field_name));
   }
   
   inline AccessPath AccessPathObj::ArrayItem(int64_t index) const {
     return this->Extend(AccessStep::ArrayItem(index));
   }
   
   inline AccessPath AccessPathObj::ArrayItemMissing(int64_t index) const {
     return this->Extend(AccessStep::ArrayItemMissing(index));
   }
   
   inline AccessPath AccessPathObj::MapItem(Any key) const {
     return this->Extend(AccessStep::MapItem(key));
   }
   
   inline AccessPath AccessPathObj::MapItemMissing(Any key) const {
     return this->Extend(AccessStep::MapItemMissing(key));
   }
   
   inline Array<AccessStep> AccessPathObj::ToSteps() const {
     std::vector<AccessStep> reverse_steps;
     reverse_steps.reserve(this->depth);
     const AccessPathObj* current = this;
     while (current->parent.has_value()) {
       TVM_FFI_ICHECK(current->step.has_value());
       reverse_steps.push_back(*(current->step));
       current = static_cast<const AccessPathObj*>(current->parent.get());
       TVM_FFI_ICHECK(current != nullptr);
     }
     return Array<AccessStep>(reverse_steps.rbegin(), reverse_steps.rend());
   }
   
   inline bool AccessPathObj::PathEqual(const AccessPath& other) const {
     return PathEqual(this, other.get());
   }
   
   inline bool AccessPathObj::IsPrefixOf(const AccessPath& other) const {
     if (this->depth > other->depth) {
       return false;
     }
     const AccessPathObj* rhs_path = other.get();
     while (rhs_path->depth > this->depth) {
       TVM_FFI_ICHECK(rhs_path->parent.has_value());
       rhs_path = static_cast<const AccessPathObj*>(rhs_path->parent.get());
     }
     return PathEqual(this, rhs_path);
   }
   
   }  // namespace reflection
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_REFLECTION_ACCESS_PATH_H_
