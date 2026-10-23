
.. _program_listing_file_tvm_ir_scope_stack.h:

Program Listing for File scope_stack.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ir_scope_stack.h>` (``tvm/ir/scope_stack.h``)

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
   
   #ifndef TVM_IR_SCOPE_STACK_H_
   #define TVM_IR_SCOPE_STACK_H_
   
   #include <tvm/ffi/error.h>
   
   #include <deque>
   #include <type_traits>
   
   namespace tvm {
   
   template <typename T>
   class ScopeStack {
    public:
     ScopeStack() { stack_.emplace_back(); }
   
     auto rbegin() const { return stack_.rbegin(); }
     auto rend() const { return stack_.rend(); }
   
     size_t size() const { return stack_.size(); }
   
     bool empty() const { return stack_.empty(); }
   
     T& Current() {
       TVM_FFI_ICHECK(!stack_.empty());
       return stack_.back();
     }
   
     const T& Current() const {
       TVM_FFI_ICHECK(!stack_.empty());
       return stack_.back();
     }
   
     template <typename F>
     auto WithNewScope(F&& body) -> decltype(body()) {
       stack_.emplace_back();
       struct Guard {
         std::deque<T>* stack;
         ~Guard() noexcept(false) { stack->pop_back(); }
       } guard{&stack_};
       if constexpr (std::is_void_v<decltype(body())>) {
         body();
       } else {
         return body();
       }
     }
   
    private:
     std::deque<T> stack_;
   };
   
   }  // namespace tvm
   
   #endif  // TVM_IR_SCOPE_STACK_H_
