
.. _program_listing_file_tvm_support_io.h:

Program Listing for File io.h
=============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_support_io.h>` (``tvm/support/io.h``)

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
   
   #ifndef TVM_SUPPORT_IO_H_
   #define TVM_SUPPORT_IO_H_
   
   #include <cstddef>
   #include <cstdint>
   
   namespace tvm {
   namespace support {
   
   template <typename T, typename = void>
   struct Serializer {
     static constexpr bool enabled = false;
   };
   
   class Stream {
    public:
     virtual size_t Read(void* ptr, size_t size) = 0;
   
     virtual size_t Write(const void* ptr, size_t size) = 0;
   
     virtual ~Stream() = default;
   
     template <typename T>
     void Write(const T& data) {
       static_assert(Serializer<T>::enabled, "No Serializer<T> specialization for this type");
       Serializer<T>::Write(this, data);
     }
   
     template <typename T>
     bool Read(T* out_data) {
       static_assert(Serializer<T>::enabled, "No Serializer<T> specialization for this type");
       return Serializer<T>::Read(this, out_data);
     }
   
     template <typename T>
     void WriteArray(const T* data, size_t num_elems) {
       for (size_t i = 0; i < num_elems; ++i) Write<T>(data[i]);
     }
   
     template <typename T>
     bool ReadArray(T* data, size_t num_elems) {
       for (size_t i = 0; i < num_elems; ++i) {
         if (!Read<T>(data + i)) return false;
       }
       return true;
     }
   };
   
   }  // namespace support
   }  // namespace tvm
   
   #endif  // TVM_SUPPORT_IO_H_
