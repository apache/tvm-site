
.. _program_listing_file_tvm_ir_with_context.h:

Program Listing for File with_context.h
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_with_context.h>` (``tvm/ir/with_context.h``)

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
   
   #ifndef TVM_IR_WITH_CONTEXT_H_
   #define TVM_IR_WITH_CONTEXT_H_
   
   #include <exception>
   #include <memory>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   
   template <typename ContextType>
   class With {
    public:
     template <typename... Args>
     explicit With(Args&&... args) : ctx_(std::forward<Args>(args)...) {
       ctx_.EnterWithScope();
     }
     ~With() noexcept(false) { ctx_.ExitWithScope(); }
   
     // Disable copy and move construction.  `With` is intended only for
     // use in nested contexts that are exited in the reverse order of
     // entry.  Allowing context to be copied or moved would break this
     // expectation.
     With(const With& other) = delete;
     With& operator=(const With& other) = delete;
     With(With&& other) = delete;
     With& operator=(With&& other) = delete;
   
     ContextType* get() { return &ctx_; }
     const ContextType* get() const { return &ctx_; }
   
     ContextType* operator->() { return get(); }
     const ContextType* operator->() const { return get(); }
     ContextType& operator*() { return *get(); }
     const ContextType& operator*() const { return *get(); }
   
     ContextType operator()() { return ctx_; }
   
    private:
     ContextType ctx_;
   };
   
   template <typename ContextType>
   class WithGroup {
    public:
     WithGroup() = default;
     WithGroup(WithGroup&&) = default;
     WithGroup& operator=(WithGroup&&) = default;
     WithGroup(const WithGroup&) = delete;
     WithGroup& operator=(const WithGroup&) = delete;
   
     template <typename... Args>
     void Emplace(Args&&... args) {
       entries_.push_back(std::make_unique<With<ContextType>>(std::forward<Args>(args)...));
     }
   
     size_t size() const { return entries_.size(); }
   
     ~WithGroup() noexcept(false) {
       bool unwinding = std::uncaught_exceptions() > 0;
       std::exception_ptr first_exc;
       while (!entries_.empty()) {
         // Move the last entry out of the vector first, then destroy it.
         // This ensures entries_ shrinks even if ~With() throws.
         auto entry = std::move(entries_.back());
         entries_.pop_back();
         try {
           entry.reset();  // calls ~With<ContextType>() -> ExitWithScope()
         } catch (...) {
           if (!unwinding && !first_exc) {
             first_exc = std::current_exception();
           }
         }
       }
       if (first_exc) std::rethrow_exception(first_exc);
     }
   
    private:
     std::vector<std::unique_ptr<With<ContextType>>> entries_;
   };
   
   }  // namespace tvm
   #endif  // TVM_IR_WITH_CONTEXT_H_
