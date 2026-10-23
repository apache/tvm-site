
.. _program_listing_file_tvm_tirx_tile_op.h:

Program Listing for File op.h
=============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_tile_op.h>` (``tvm/tirx/tile/op.h``)

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
   #ifndef TVM_TIRX_TILE_OP_H_
   #define TVM_TIRX_TILE_OP_H_
   
   #include <tvm/ffi/function.h>
   #include <tvm/ir/op.h>
   
   namespace tvm::tirx::tile {
   
   TVM_TIRX_DLL void ConfigureTensorInstruction(Op op, ffi::Function validator);
   
   }  // namespace tvm::tirx::tile
   #endif  // TVM_TIRX_TILE_OP_H_
