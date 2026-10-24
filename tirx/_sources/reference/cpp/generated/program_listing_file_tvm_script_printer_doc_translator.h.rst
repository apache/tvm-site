
.. _program_listing_file_tvm_script_printer_doc_translator.h:

Program Listing for File doc_translator.h
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_script_printer_doc_translator.h>` (``tvm/script/printer/doc_translator.h``)

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
   #ifndef TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_H_
   #define TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_H_
   
   #include <tvm/ffi/container/dict.h>
   #include <tvm/ffi/container/list.h>
   #include <tvm/ffi/container/map.h>
   #include <tvm/ffi/expected.h>
   #include <tvm/ffi/reflection/accessor.h>
   #include <tvm/ir/expr.h>
   #include <tvm/script/printer/doc.h>
   
   #include <optional>
   #include <utility>
   
   namespace tvm {
   
   namespace script {
   namespace printer {
   
   class DocTranslatorObj;
   
   struct DocTranslatorVTable {
     TVMFFIAny (*translate)(DocTranslatorObj* translator, ffi::AnyView value,
                            const ffi::Object* destination) noexcept;
     TVMFFIAny (*alloc_id)(DocTranslatorObj* translator, ffi::AnyView hint) noexcept;
     TVMFFIAny (*var_get_or_alloc_id)(DocTranslatorObj* translator, ffi::AnyView var,
                                      bool explicit_def) noexcept;
     TVMFFIAny (*get_implicit_defs)(DocTranslatorObj* translator) noexcept;
     TVMFFIAny (*begin_var_scope)(DocTranslatorObj* translator) noexcept;
     TVMFFIAny (*end_var_scope)(DocTranslatorObj* translator) noexcept;
   };
   
   using FDocTranslate = ffi::reflection::NativeFunctionView<ffi::Optional<ExprDoc>(
       DocTranslatorObj* translator, ffi::AnyView value, const ffi::Object* destination)>;
   
   class DocTranslatorObj : public ffi::Object {
    public:
     ffi::Optional<ExprDoc> Translate(ffi::AnyView value,
                                      const ffi::Optional<Var> bind_var = std::nullopt) {
       auto result = ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Optional<ExprDoc>>(
                         vtable_->translate(this, value,
                                            bind_var.has_value() ? bind_var.value().get() : nullptr))
                         .value();
       if (result.has_value()) {
         if (const auto* origin = value.as<ffi::Object>()) {
           RecordOrigin(result.value(), ffi::GetRef<ffi::ObjectRef>(origin));
         }
       }
       return result;
     }
     void Emit(const Doc& doc, ffi::Optional<ffi::ObjectRef> origin = std::nullopt) {
       CurrentScopeDocs().push_back(doc);
       if (origin.has_value()) RecordOrigin(doc, origin.value());
     }
   
     IdDoc VarGetOrAllocId(const Var& var, bool explicit_def) {
       return ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<IdDoc>(
                  vtable_->var_get_or_alloc_id(this, var, explicit_def))
           .value();
     }
     IdDoc AllocId(const ffi::String& hint) {
       return ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<IdDoc>(vtable_->alloc_id(this, hint))
           .value();
     }
     ffi::Dict<Var, IdDoc> GetImplicitDefs() {
       return ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Dict<Var, IdDoc>>(
                  vtable_->get_implicit_defs(this))
           .value();
     }
   
     template <typename T>
     T GetExtraConfig(const ffi::String& name, T fallback) const {
       auto it = extra_config_.find(name);
       return it == extra_config_.end() ? fallback : (*it).second.as_or_throw<T>();
     }
     template <typename T>
     T GetOrCreateExtraState(const ffi::String& name) {
       auto it = extra_state_.find(name);
       if (it != extra_state_.end()) return (*it).second.as_or_throw<T>();
       T value{};
       extra_state_.Set(name, value);
       return value;
     }
     void SetExtraState(const ffi::String& name, std::optional<ffi::Any> value) {
       if (value.has_value()) {
         extra_state_.Set(name, std::move(value.value()));
       } else {
         extra_state_.erase(name);
       }
     }
   
     void BeginVarScope() {
       ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<void>(vtable_->begin_var_scope(this)).value();
     }
     void EndVarScope() {
       ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<void>(vtable_->end_var_scope(this)).value();
     }
     template <typename Callback>
     decltype(auto) WithVarScope(Callback&& callback) {
       struct Scope {
         DocTranslatorObj* translator;
         ~Scope() {
           ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<void>(
               translator->vtable_->end_var_scope(translator));
         }
       };
       BeginVarScope();
       Scope scope{this};
       return std::forward<Callback>(callback)();
     }
   
     template <typename T, typename Callback>
     decltype(auto) WithExtraStateScope(const ffi::String& name, T value, Callback&& callback) {
       struct Scope {
         DocTranslatorObj* translator;
         ffi::String key;
         T saved;
         ~Scope() { translator->SetExtraState(key, std::move(saved)); }
       };
       Scope scope{this, name, GetOrCreateExtraState<T>(name)};
       SetExtraState(scope.key, std::move(value));
       return std::forward<Callback>(callback)();
     }
   
     void BeginDocScope() { doc_scopes_.push_back(ffi::List<Doc>()); }
     ffi::List<Doc> EndDocScope() {
       auto docs = CurrentScopeDocs();
       doc_scopes_.pop_back();
       return docs;
     }
     ffi::List<Doc> CurrentScopeDocs() const {
       TVM_FFI_CHECK(!doc_scopes_.empty(), ValueError)
           << "printer emission requires a collection scope";
       return doc_scopes_.back();
     }
     template <typename Callback>
     ffi::List<Doc> WithDocScope(Callback&& callback) {
       size_t depth = doc_scopes_.size();
       BeginDocScope();
       try {
         std::forward<Callback>(callback)();
         TVM_FFI_CHECK(doc_scopes_.size() == depth + 1, ValueError)
             << "printer unbalanced Doc collection scopes";
         return EndDocScope();
       } catch (...) {
         while (doc_scopes_.size() > depth) doc_scopes_.pop_back();
         throw;
       }
     }
   
     ffi::Dict<Doc, ffi::ObjectRef> GetDocOrigins() const { return doc_origins_; }
     void RecordOrigin(const Doc& doc, const ffi::ObjectRef& origin) { doc_origins_.Set(doc, origin); }
     ffi::Optional<ExprDoc> DefaultTranslate(ffi::AnyView value,
                                             const ffi::Object* destination = nullptr);
   
     TVM_TIRX_DLL ExprDoc DefaultTypeTranslate(const Type& type);
   
     static constexpr bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO("script.printer.DocTranslator", DocTranslatorObj, ffi::Object);
   
    protected:
     explicit DocTranslatorObj(const DocTranslatorVTable* vtable,
                               ffi::Map<ffi::String, ffi::Any> extra_config = {})
         : extra_config_(std::move(extra_config)), vtable_(vtable) {}
   
     ffi::List<ffi::List<Doc>> doc_scopes_;
   
    private:
     ffi::Map<ffi::String, ffi::Any> extra_config_;
     ffi::Dict<ffi::String, ffi::Any> extra_state_;
     ffi::Dict<Doc, ffi::ObjectRef> doc_origins_;
     const DocTranslatorVTable* vtable_;
   };
   
   inline ffi::Array<StmtDoc> ToStmtDocArray(const ffi::List<Doc>& docs) {
     ffi::Array<StmtDoc> statements;
     for (const auto& doc : docs) {
       if (auto block = doc.as<StmtBlockDoc>()) {
         for (const auto& stmt : block.value()->stmts) statements.push_back(stmt);
       } else {
         statements.push_back(doc.as_or_throw<StmtDoc>());
       }
     }
     return statements;
   }
   
   class DocTranslator : public ffi::ObjectRef {
    public:
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DocTranslator, ffi::ObjectRef, DocTranslatorObj);
   };
   
   TVM_TIRX_DLL Doc DocTranslate(ffi::AnyView ir,
                                 ffi::Dict<Doc, ffi::ObjectRef>* doc_origins = nullptr,
                                 ffi::Map<ffi::String, ffi::Any> extra_config = {});
   
   namespace type_attr {
   inline constexpr ffi::reflection::AttrKey<FDocTranslate> kDocTranslate{"__tvm_doc_translate__"};
   }  // namespace type_attr
   
   namespace op_attr {
   inline constexpr ffi::reflection::AttrKey<FDocTranslate> kDocTranslateOpCall{
       "__tvm_doc_translate_op_call__"};
   }  // namespace op_attr
   
   inline ffi::Optional<ExprDoc> DocTranslatorObj::DefaultTranslate(ffi::AnyView value,
                                                                    const ffi::Object* destination) {
     static ffi::reflection::TypeAttrColumn column(
         tvm::script::printer::type_attr::kDocTranslate.name);
     ffi::AnyView attr = column[value.type_index()];
     if (attr.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
       return type_attr::kDocTranslate.ValueFrom(attr)
           .value()
           .CallExpected(this, value, destination)
           .value();
     }
     if (attr.type_index() == ffi::TypeIndex::kTVMFFIFunction) {
       ffi::Any destination_arg = nullptr;
       if (destination) destination_arg = ffi::GetRef<ffi::ObjectRef>(destination);
       return attr.cast<ffi::Function>()
           .CallExpected<ffi::Optional<ExprDoc>>(this, value, destination_arg)
           .value();
     }
     if (attr.type_index() == ffi::TypeIndex::kTVMFFINone) {
       if (value == nullptr) return LiteralDoc::None(std::nullopt);
       if (auto string = value.as<ffi::String>()) return LiteralDoc::Str(string.value(), std::nullopt);
       if (auto boolean = value.as<bool>()) return LiteralDoc::Boolean(boolean.value(), std::nullopt);
       if (auto integer = value.as<int64_t>()) return LiteralDoc::Int(integer.value(), std::nullopt);
       if (auto floating = value.as<double>())
         return LiteralDoc::Float(floating.value(), std::nullopt);
       if (auto dtype = value.as<DLDataType>())
         return LiteralDoc::DataType(dtype.value(), std::nullopt);
       if (auto type = value.as<Type>()) return DefaultTypeTranslate(type.value());
     }
     TVM_FFI_THROW(TypeError) << (attr.type_index() == ffi::TypeIndex::kTVMFFINone
                                      ? std::string("printer has no Doc hook for ") +
                                            value.GetTypeKey()
                                      : "printer Doc hook must be a native pointer or ffi.Function");
     TVM_FFI_UNREACHABLE();
   }
   
   }  // namespace printer
   }  // namespace script
   }  // namespace tvm
   
   #endif  // TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_H_
