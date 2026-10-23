
.. _program_listing_file_tvm_support_env.h:

Program Listing for File env.h
==============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_support_env.h>` (``tvm/support/env.h``)

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
   
   #ifndef TVM_SUPPORT_ENV_H_
   #define TVM_SUPPORT_ENV_H_
   
   #include <cstdlib>
   #include <sstream>
   #include <string>
   #include <type_traits>
   
   namespace tvm {
   namespace support {
   
   template <typename T>
   inline T GetEnv(const char* key, T default_value) {
     const char* val = std::getenv(key);
     if (val == nullptr || !*val) return default_value;
     if constexpr (std::is_same_v<T, std::string>) {
       return std::string(val);
     } else if constexpr (std::is_same_v<T, bool>) {
       // Accept "0"/"false" as false, anything else as true
       std::string s(val);
       return !(s == "0" || s == "false" || s == "False" || s == "FALSE");
     } else {
       T ret;
       std::istringstream is(val);
       is >> ret;
       return ret;
     }
   }
   
   }  // namespace support
   }  // namespace tvm
   
   #endif  // TVM_SUPPORT_ENV_H_
