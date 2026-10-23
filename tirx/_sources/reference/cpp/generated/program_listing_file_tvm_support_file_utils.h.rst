
.. _program_listing_file_tvm_support_file_utils.h:

Program Listing for File file_utils.h
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_support_file_utils.h>` (``tvm/support/file_utils.h``)

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
   
   #ifndef TVM_SUPPORT_FILE_UTILS_H_
   #define TVM_SUPPORT_FILE_UTILS_H_
   
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/string.h>
   
   #include <fstream>
   #include <string>
   #include <utility>
   
   namespace tvm {
   namespace support {
   
   inline ffi::String GetFileFormat(const ffi::String& path, const ffi::String& format) {
     if (!format.empty()) return format;
     std::string file_name = path;
     size_t pos = file_name.find_last_of('.');
     return pos == std::string::npos ? ffi::String() : ffi::String(file_name.substr(pos + 1));
   }
   
   inline ffi::Bytes ReadBytesFromFile(const ffi::String& path) {
     std::ifstream fs(path.c_str(), std::ios::in | std::ios::binary);
     TVM_FFI_ICHECK(!fs.fail()) << "Cannot open " << path;
     fs.seekg(0, std::ios::end);
     size_t size = static_cast<size_t>(fs.tellg());
     fs.seekg(0, std::ios::beg);
     std::string data(size, '\0');
     fs.read(data.data(), size);
     return ffi::Bytes(std::move(data));
   }
   
   inline void WriteBytesToFile(const ffi::String& path, const ffi::Bytes& data) {
     std::ofstream fs(path.c_str(), std::ios::out | std::ios::binary);
     TVM_FFI_ICHECK(!fs.fail()) << "Cannot open " << path;
     fs.write(data.data(), data.size());
   }
   
   }  // namespace support
   }  // namespace tvm
   #endif  // TVM_SUPPORT_FILE_UTILS_H_
