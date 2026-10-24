
.. _program_listing_file_tvm_tirx_op_attr.h:

Program Listing for File attr.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_tirx_op_attr.h>` (``tvm/tirx/op/attr.h``)

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
   
   #ifndef TVM_TIRX_OP_ATTR_H_
   #define TVM_TIRX_OP_ATTR_H_
   
   #include <tvm/ffi/extra/structural_mutate.h>
   #include <tvm/ffi/function.h>
   #include <tvm/ffi/reflection/native_function.h>
   #include <tvm/ffi/string.h>
   #include <tvm/ir/expr.h>
   #include <tvm/ir/stmt.h>
   
   namespace tvm {
   namespace tirx {
   
   class StructuralVisitorWithConstraintContextObj;
   class StructuralMutatorWithConstraintContextObj;
   
   using FLowerIntrinsic = ffi::TypedFunction<PrimExpr(PrimExpr)>;
   
   using FLegalize = ffi::TypedFunction<PrimExpr(PrimExpr)>;
   
   namespace op_attr {
   inline constexpr ffi::reflection::AttrKey<ffi::String> kGlobalSymbol{"TGlobalSymbol"};
   inline constexpr ffi::reflection::AttrKey<bool> kVectorizable{"TVectorizable"};
   inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<ffi::Optional<
       ffi::VisitInterrupt>(StructuralVisitorWithConstraintContextObj*, ffi::AnyView value)>>
       kStructuralVisitWithConstraintContext{"__s_visit_with_constraint_context__"};
   inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<
       ffi::UnchangedOr<ffi::Any>(StructuralMutatorWithConstraintContextObj*, ffi::AnyView value)>>
       kStructuralMutateWithConstraintContext{"__s_mutate_with_constraint_context__"};
   inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<
       ffi::UnchangedOr<ffi::Any>(StructuralMutatorWithConstraintContextObj*, ffi::AnyView value)>>
       kStructuralMaybeInplaceMutateWithConstraintContext{
           "__s_maybe_inplace_mutate_with_constraint_context__"};
   }  // namespace op_attr
   
   }  // namespace tirx
   }  // namespace tvm
   #endif  // TVM_TIRX_OP_ATTR_H_
