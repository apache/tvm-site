
.. _program_listing_file_tvm_ir_object_functor.h:

Program Listing for File object_functor.h
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_object_functor.h>` (``tvm/ir/object_functor.h``)

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
   #ifndef TVM_IR_OBJECT_FUNCTOR_H_
   #define TVM_IR_OBJECT_FUNCTOR_H_
   
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/extra/structural_mutate.h>
   #include <tvm/ffi/extra/structural_visit.h>
   #include <tvm/runtime/base.h>
   
   #include <cstring>
   #include <type_traits>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   
   template <typename FType>
   class ObjectFunctor;
   
   template <typename R, typename NodeArg, typename... Args>
   class ObjectFunctor<R(NodeArg, Args...)> {
     static_assert(std::is_same_v<NodeArg, const ffi::ObjectRef&> ||
                       std::is_same_v<NodeArg, const ffi::Object*>,
                   "ObjectFunctor requires an ObjectRef reference or borrowed Object pointer");
   
    public:
     using result_type = R;
     template <typename T>
     TVM_FFI_INLINE bool CanDispatch(const T& n) const {
       uint32_t type_index = GetTypeIndex(n);
       if (type_index < begin_type_index_) return false;
       type_index -= begin_type_index_;
       return type_index < func_.size() && func_[type_index] != nullptr;
     }
     TVM_FFI_INLINE R operator()(NodeArg n, Args... args) const {
       uint32_t type_index = n->type_index();
       if (type_index >= begin_type_index_) {
         uint32_t index = type_index - begin_type_index_;
         if (index < func_.size() && func_[index] != nullptr) {
           return (*func_[index])(n, std::forward<Args>(args)...);
         }
       }
   
       const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
       for (int32_t i = type_info->type_depth - 1; i >= 0; --i) {
         type_index = type_info->type_ancestors[i]->type_index;
         if (type_index >= begin_type_index_) {
           uint32_t index = type_index - begin_type_index_;
           if (index < func_.size() && func_[index] != nullptr) {
             return (*func_[index])(n, std::forward<Args>(args)...);
           }
         }
       }
       ThrowUnregistered(n);
     }
     template <typename TNode>
     ObjectFunctor& SetDispatch(R (*f)(NodeArg n, Args...)) {
       static_assert(std::is_base_of_v<ffi::Object, TNode>,
                     "ObjectFunctor dispatch requires an object node type");
       uint32_t tindex = TNode::RuntimeTypeIndex();
       if (func_.size() <= tindex) {
         func_.resize(tindex + 1, nullptr);
       }
       TVM_FFI_ICHECK(func_[tindex] == nullptr)
           << "Dispatch for " << TNode::_type_key << " is already set";
       TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << " Cannot call SetDispatch after calling Finalize";
       func_[tindex] = f;
       return *this;
     }
     template <typename TNode>
     ObjectFunctor& ClearDispatch() {
       uint32_t tindex = TNode::RuntimeTypeIndex();
       TVM_FFI_ICHECK_LT(tindex, func_.size()) << "ClearDispatch: index out of range";
       TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << " Cannot call ClearDispatch after calling Finalize";
       func_[tindex] = nullptr;
       return *this;
     }
     void Finalize() {
       TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << "Can only call Finalize once";
       while (begin_type_index_ < func_.size() && func_[begin_type_index_] == nullptr) {
         ++begin_type_index_;
       }
       // shift up the function value
       size_t new_ftable_size = func_.size() - begin_type_index_;
       if (begin_type_index_ != 0) {
         std::memmove(func_.data(), func_.data() + begin_type_index_,
                      new_ftable_size * sizeof(FPointer));
       }
       func_.resize(new_ftable_size);
       func_.shrink_to_fit();
     }
   
    private:
     // Map null pointers and undefined handles to None so CanDispatch rejects them without
     // dereferencing.
     TVM_FFI_INLINE static uint32_t GetTypeIndex(NodeArg n) {
       if constexpr (std::is_pointer_v<NodeArg>) {
         return n != nullptr ? n->type_index() : ffi::TypeIndex::kTVMFFINone;
       } else {
         return n.defined() ? n->type_index() : ffi::TypeIndex::kTVMFFINone;
       }
     }
     TVM_FFI_INLINE static uint32_t GetTypeIndex(ffi::AnyView n) { return n.type_index(); }
   
     [[noreturn]] TVM_FFI_COLD_CODE static void ThrowUnregistered(NodeArg n) {
       TVM_FFI_THROW(InternalError) << "ObjectFunctor calls un-registered function on type "
                                    << n->GetTypeKey();
       TVM_FFI_UNREACHABLE();
     }
   
     using FPointer = R (*)(NodeArg n, Args...);
     std::vector<FPointer> func_;
     uint32_t begin_type_index_{0};
   };
   
   using ffi::Expected;
   using ffi::InplaceMode;
   using ffi::UnchangedOr;
   using ffi::VisitInterrupt;
   
   class TVM_TIRX_DLL ObjectVisitor : public ffi::StructuralVisitorObj {
    public:
     ObjectVisitor() : ObjectVisitor(GlobalVTable()) {}
     ObjectVisitor(const ObjectVisitor& other) = delete;
     ObjectVisitor& operator=(const ObjectVisitor& other) = delete;
   
     virtual ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) {
       if (native_vtable_->CanDispatch(value)) {
         const auto* object =
             ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ffi::Object>(value);
         try {
           return (*native_vtable_)(object, this);
         } catch (ffi::Error& error) {
           ffi::detail::UpdateVisitErrorContext(error, ffi::GetRef<ffi::ObjectRef>(object));
           throw;
         }
       }
       return StructuralVisitDefault(value);
     }
   
     Expected<ffi::Optional<VisitInterrupt>> VisitExpected(ffi::AnyView value) noexcept {
       try {
         return Visit(value);
       } catch (ffi::Error& error) {
         return ffi::Unexpected(std::move(error));
       } catch (const std::exception& error) {
         return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
       }
     }
   
    protected:
     using VTable = ObjectFunctor<ffi::Optional<VisitInterrupt>(const ffi::Object*, ObjectVisitor*)>;
   
     explicit ObjectVisitor(const VTable* vtable)
         : ffi::StructuralVisitorObj(StructuralVTable()), native_vtable_(vtable) {}
     static void InitVTable(VTable* vtable) {}
   
     template <typename Self, typename Node>
     static void SetDispatch(VTable* vtable) {
       vtable->template SetDispatch<Node>(DispatchNode<Self, Node>);
     }
   
    private:
     // Native table callback; core instantiations may be shared by the library.
     template <typename Self, typename Node>
     static ffi::Optional<VisitInterrupt> DispatchNode(const ffi::Object* node, ObjectVisitor* self) {
       return static_cast<Self*>(self)->Visit_(static_cast<const Node*>(node));
     }
   
     ffi::Optional<VisitInterrupt> StructuralVisitDefault(ffi::AnyView value) {
       try {
         // DefaultVisitExpected does not name the current node; DefaultMutateExpected does.
         return ffi::StructuralVisitorObj::DefaultVisitExpected(value).value();
       } catch (ffi::Error& error) {
         ffi::detail::UpdateVisitErrorContext(error, value);
         throw;
       }
     }
     static const VTable* GlobalVTable() {
       static const VTable table = [] {
         VTable table;
         InitVTable(&table);
         table.Finalize();
         return table;
       }();
       return &table;
     }
     static const ffi::StructuralVisitorVTable* StructuralVTable() {
       static const ffi::StructuralVisitorVTable table{StructuralVTableVisitImpl};
       return &table;
     }
     static TVMFFIAny StructuralVTableVisitImpl(ffi::StructuralVisitorObj* self,
                                                ffi::AnyView value) noexcept {
       return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
           static_cast<ObjectVisitor*>(self)->VisitExpected(value));
     }
   
     const VTable* const native_vtable_;
   };
   
   class TVM_TIRX_DLL ObjectMutator : public ffi::StructuralMapEngineBase {
    public:
     ObjectMutator() : ObjectMutator(GlobalVTable()) {}
     ObjectMutator(const ObjectMutator& other) = delete;
     ObjectMutator& operator=(const ObjectMutator& other) = delete;
   
     virtual UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                                          InplaceMode inplace_mode = InplaceMode::kDisallow) {
       const auto* object = value.as<ffi::Object>();
       if (object == nullptr) return ffi::Any(value);
       if (!object->unique()) inplace_mode = InplaceMode::kDisallow;
       if (native_vtable_->CanDispatch(object)) {
         try {
           return (*native_vtable_)(object, this, inplace_mode);
         } catch (ffi::Error& error) {
           ffi::detail::UpdateVisitErrorContext(error, ffi::GetRef<ffi::ObjectRef>(object));
           throw;
         }
       }
       // Default descent already supplies the current node's error context.
       return ffi::StructuralMutatorObj::DefaultMutateExpected(value, inplace_mode).value();
     }
   
     Expected<UnchangedOr<ffi::Any>> MutateExpected(
         ffi::AnyView value, InplaceMode inplace_mode = InplaceMode::kDisallow) noexcept {
       try {
         return Mutate(value, inplace_mode);
       } catch (ffi::Error& error) {
         return ffi::Unexpected(std::move(error));
       } catch (const std::exception& error) {
         return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
       }
     }
   
     ffi::Any VarRemapGet(ffi::AnyView var) { return VarRemapGetExpected(var).value(); }
     void VarRemapSet(ffi::AnyView var, ffi::AnyView value) {
       VarRemapSetExpected(var, value).value();
     }
   
     TVM_FFI_INLINE Expected<ffi::Any> VarRemapGetExpected(ffi::AnyView var) noexcept {
       return VarRemapGetImpl(var);
     }
     TVM_FFI_INLINE Expected<void> VarRemapSetExpected(ffi::AnyView var,
                                                       ffi::AnyView mapped_value) noexcept {
       return VarRemapSetImpl(var, mapped_value);
     }
   
    protected:
     using VTable =
         ObjectFunctor<UnchangedOr<ffi::Any>(const ffi::Object*, ObjectMutator*, InplaceMode)>;
   
     explicit ObjectMutator(const VTable* vtable)
         : ffi::StructuralMapEngineBase(StructuralVTable()), native_vtable_(vtable) {}
     static void InitVTable(VTable* vtable) {}
   
     template <typename Self, typename Node>
     static void SetDispatch(VTable* vtable) {
       vtable->template SetDispatch<Node>(DispatchNode<Self, Node>);
     }
   
    private:
     // Native table callback; core instantiations may be shared by the library.
     template <typename Self, typename Node>
     static UnchangedOr<ffi::Any> DispatchNode(const ffi::Object* node, ObjectMutator* self,
                                               InplaceMode inplace_mode) {
       return ffi::detail::UnchangedOrUnsafe::MoveFromTVMFFIAny<ffi::Any>(
           ffi::detail::UnchangedOrUnsafe::MoveToTVMFFIAny(
               static_cast<Self*>(self)->Mutate_(static_cast<const Node*>(node), inplace_mode)));
     }
   
     using ffi::StructuralMutatorObj::DefaultMutateExpected;
   
     static const VTable* GlobalVTable() {
       static const VTable table = [] {
         VTable table;
         InitVTable(&table);
         table.Finalize();
         return table;
       }();
       return &table;
     }
     static const ffi::StructuralMutatorVTable* StructuralVTable() {
       static const ffi::StructuralMutatorVTable table{
           StructuralVTableMutateImpl, StructuralVTableMaybeInplaceMutateImpl,
           StructuralVTableVarRemapGetImpl, StructuralVTableVarRemapSetImpl};
       return &table;
     }
     static TVMFFIAny StructuralVTableMutateImpl(ffi::StructuralMutatorObj* self,
                                                 ffi::AnyView value) noexcept {
       return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
           static_cast<ObjectMutator*>(self)->MutateExpected(value, InplaceMode::kDisallow));
     }
     static TVMFFIAny StructuralVTableMaybeInplaceMutateImpl(ffi::StructuralMutatorObj* self,
                                                             ffi::AnyView value) noexcept {
       return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
           static_cast<ObjectMutator*>(self)->MutateExpected(value, InplaceMode::kAllow));
     }
     static TVMFFIAny StructuralVTableVarRemapGetImpl(ffi::StructuralMutatorObj* self,
                                                      ffi::AnyView key) noexcept {
       return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
           static_cast<ObjectMutator*>(self)->VarRemapGetImpl(key));
     }
     static TVMFFIAny StructuralVTableVarRemapSetImpl(ffi::StructuralMutatorObj* self,
                                                      ffi::AnyView key, ffi::AnyView value) noexcept {
       return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
           static_cast<ObjectMutator*>(self)->VarRemapSetImpl(key, value));
     }
   
     const VTable* const native_vtable_;
   };
   
   #define TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(Class, Parent) \
     Class()                                                            \
         : Parent([] {                                                  \
             static const VTable table = [] {                           \
               VTable table;                                            \
               Class::InitVTable(&table);                               \
               table.Finalize();                                        \
               return table;                                            \
             }();                                                       \
             return &table;                                             \
           }()) {}
   
   }  // namespace tvm
   #endif  // TVM_IR_OBJECT_FUNCTOR_H_
