
.. _program_listing_file_tvm_runtime_timer.h:

Program Listing for File timer.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_runtime_timer.h>` (``tvm/runtime/timer.h``)

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
   
   #ifndef TVM_RUNTIME_TIMER_H_
   #define TVM_RUNTIME_TIMER_H_
   
   #include <tvm/ffi/object.h>
   
   namespace tvm {
   namespace runtime {
   
   class TimerObj : public ffi::Object {
    public:
     virtual void Start() = 0;
     virtual void Stop() = 0;
     virtual int64_t SyncAndGetElapsedNanos() = 0;
   
     virtual ~TimerObj() {}
   
     static constexpr const bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO("runtime.Timer", TimerObj, ffi::Object);
   };
   
   class Timer : public ffi::ObjectRef {
    public:
     explicit Timer(ffi::UnsafeInit tag, ffi::ObjectPtr<TimerObj> n) : ffi::ObjectRef(tag) {
       data_ = std::move(n);
     }
   
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Timer, ffi::ObjectRef, TimerObj);
   };
   
   }  // namespace runtime
   }  // namespace tvm
   
   #endif  // TVM_RUNTIME_TIMER_H_
