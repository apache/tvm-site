
.. _file_tvm_ir_prim_pattern_match.h:

File pattern_match.h
====================

|exhale_lsh| :ref:`Parent directory <dir_tvm_ir_prim>` (``tvm/ir/prim``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Public tool for expression-template based pattern matching. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/ir/prim/pattern_match.h``)
--------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_ir_prim_pattern_match.h.rst



Detailed Description
--------------------

It helps to simplify pattern matching and rewrites. All the patterns are generated via expression template during compile time, so the result code should be as efficient as manually written pattern match code.

The code below shows how to use the pattern matcher.

.. code-block:: cpp


   // max(x + z, y + z) => max(x, y) + z
   tvm::PVar<PrimExpr> x, y, z;

   // The following code tries to match the declared pattern.
   // Match will fill the result of match into PVar if successful.
   // Note that z occurs twice in the pattern,
   // an equality check is performed to ensure each occurance of z
   // is equivalent to each other.
   if (max(x + z, y + z).Match(expr)) {
     // Eval evaluates a pattern with the current matched value.
     // The filled value is valid until the next call to Match.
     return (max(x, y) + z).Eval();
   }

   tvm::PrimVar tx("tx"), ty("ty");
   tvm::PVar<IntImm> c;
   tvm::PVar<Var> v;
   // We can match integer and Var, both of which are
   // special case container of Expr
   TVM_FFI_ICHECK((v * c).Match(tx * 3));
   TVM_FFI_ICHECK_EQ(c.Eval()->value, 3);
   // cannot match c to ty
   TVM_FFI_ICHECK(!(v * c).Match(tx * ty));



The pattern matcher is not threadsafe, do not use the same PVar in multiple threads.

Please be aware that the filled value in a PVar can be overriden in the next call to Match. 




Includes
--------


- ``cmath``

- ``tuple``

- ``tvm/ir/pattern_match.h`` (:ref:`file_tvm_ir_pattern_match.h`)

- ``tvm/ir/prim/const_fold.h`` (:ref:`file_tvm_ir_prim_const_fold.h`)

- ``tvm/ir/prim/expr.h`` (:ref:`file_tvm_ir_prim_expr.h`)

- ``tvm/ir/prim/op.h`` (:ref:`file_tvm_ir_prim_op.h`)






Namespaces
----------


- :ref:`namespace_tvm`


Classes
-------


- :ref:`exhale_struct_structtvm_1_1PIfThenElseOp`

- :ref:`exhale_class_classtvm_1_1PBinaryExpr`

- :ref:`exhale_class_classtvm_1_1PBitwiseNotExpr`

- :ref:`exhale_class_classtvm_1_1PBroadcastExpr`

- :ref:`exhale_class_classtvm_1_1PCallExpr`

- :ref:`exhale_class_classtvm_1_1PCastExpr`

- :ref:`exhale_class_classtvm_1_1PConstWithTypeLike`

- :ref:`exhale_class_classtvm_1_1PEqualChecker_3_01FloatImm_01_4`

- :ref:`exhale_class_classtvm_1_1PEqualChecker_3_01IntImm_01_4`

- :ref:`exhale_class_classtvm_1_1PEqualChecker_3_01PrimExpr_01_4`

- :ref:`exhale_class_classtvm_1_1PEqualChecker_3_01Var_01_4`

- :ref:`exhale_class_classtvm_1_1PNotExpr`

- :ref:`exhale_class_classtvm_1_1PRampExpr`

- :ref:`exhale_class_classtvm_1_1PSelectExpr`

- :ref:`exhale_class_classtvm_1_1PVarWithDataType`

- :ref:`exhale_class_classtvm_1_1PVecDataType`


Functions
---------


- :ref:`exhale_function_namespacetvm_1a24e781698f74f6087db5cba4c6168434`

- :ref:`exhale_function_namespacetvm_1a7daabd0d8657f2d55915420bbc3af874`

- :ref:`exhale_function_namespacetvm_1a02224802057577e4a98e692b72a47225`

- :ref:`exhale_function_namespacetvm_1a5ca8906ddad2083a761664c49de0213f`

- :ref:`exhale_function_namespacetvm_1a7be71ed571f41ce0a36e87446aa04f35`

- :ref:`exhale_function_namespacetvm_1aba7b9587003dda3fe9f353c80c2f5dfd`

- :ref:`exhale_function_namespacetvm_1a9376cf5684d043ff86fdd43767deaa59`

- :ref:`exhale_function_namespacetvm_1ad6f2b9d9c6ac80579866e62f53756e25`

- :ref:`exhale_function_namespacetvm_1a66ed8401a553ad5d4f444283cd619e85`

- :ref:`exhale_function_namespacetvm_1a6d6a1c629c51224ea66b141ff966b93f`

- :ref:`exhale_function_namespacetvm_1aaa57296cf5633ae36e8f6c1685db5d63`

- :ref:`exhale_function_namespacetvm_1a2c34094a0d90d2a554336564b1fba3a2`

- :ref:`exhale_function_namespacetvm_1a5eabe48483942dd9209e0f1970d92966`

- :ref:`exhale_function_namespacetvm_1a752f6f3c02b9ae41ce24db2d1a769105`

- :ref:`exhale_function_namespacetvm_1ac9d50c060c54980dff8f0f62736f7e1f`

- :ref:`exhale_function_namespacetvm_1a3beb6ede9dc1c8d2ac91c0ffed0b32f3`

- :ref:`exhale_function_namespacetvm_1a90215dca2b70c507d5e32ed71704b54c`

- :ref:`exhale_function_namespacetvm_1ac157e28ac096e2178d5af4f5ac124090`

- :ref:`exhale_function_namespacetvm_1a460572123ac43346571969c7c474d9b3`

- :ref:`exhale_function_namespacetvm_1a7d182ece0a5fe24bf0479f38063f0829`

- :ref:`exhale_function_namespacetvm_1ab5df457f526fb099d0d8de6a51639307`

- :ref:`exhale_function_namespacetvm_1af7b8be795dd2e9889636e2383c554c67`

- :ref:`exhale_function_namespacetvm_1afcdbd1fab3bdd58534c7fa93f3443d91`

- :ref:`exhale_function_namespacetvm_1a15550334ba622451f41e2f04d386c1a2`

- :ref:`exhale_function_namespacetvm_1ab09a39be17461ad9a4f0ec0b3cb963d9`

- :ref:`exhale_function_namespacetvm_1a253689828377e1e885a14b913f37a1c8`

- :ref:`exhale_function_namespacetvm_1a5db247e2fc116eb89596c4f2260f5fd3`

- :ref:`exhale_function_namespacetvm_1a65d320780845ca2d9cabe659baf5650d`

- :ref:`exhale_function_namespacetvm_1a66abe3176c42f9cdf33fd093da102d78`

- :ref:`exhale_function_namespacetvm_1add476d29b0ca26e54e6358ccc4b1dbeb`

- :ref:`exhale_function_namespacetvm_1a80ea8873c8723a7b89454683a972a2bd`

- :ref:`exhale_function_namespacetvm_1a9aafb09559d1108176ccc9fd5e41091d`

- :ref:`exhale_function_namespacetvm_1aeda545b338b07ae36db00ebaece10cdb`

- :ref:`exhale_function_namespacetvm_1a6740f026237b27fc522031edfe3c73ec`

- :ref:`exhale_function_namespacetvm_1ac5bf73fb1cfa124166b617fafbb8c24a`

- :ref:`exhale_function_namespacetvm_1a3d07a3069da44d3ae5b57d2e4040d4ff`

- :ref:`exhale_function_namespacetvm_1ad63e0f4007b232f08e7c15c04c170cff`

- :ref:`exhale_function_namespacetvm_1ae07cb4d386babd74a5e42d143882bfd3`

- :ref:`exhale_function_namespacetvm_1a06c876d86dd675819633003200a04c3a`

- :ref:`exhale_function_namespacetvm_1a850a6f663cf645bf1d5de9e46c074c15`

- :ref:`exhale_function_namespacetvm_1a0116e9743c4fd5d3ee6b660870e5bf4f`

- :ref:`exhale_function_namespacetvm_1ac559d7393509db4ce67a54d4edfbe8b0`

- :ref:`exhale_function_namespacetvm_1a7351c7ad62e6febd0cfff024204afd7c`

- :ref:`exhale_function_namespacetvm_1a1eaa18dbddaa07d70c9279e0bcdac05a`

- :ref:`exhale_function_namespacetvm_1a7477f9c7de0148e5580333452571ad66`

- :ref:`exhale_function_namespacetvm_1afd9566424968e15c462fb9bc2326c568`

- :ref:`exhale_function_namespacetvm_1a509224ee70e43a87c9a9f90050130067`

- :ref:`exhale_function_namespacetvm_1af6d0ad1d909999b19bf03a951f6c0083`

- :ref:`exhale_function_namespacetvm_1a5bea550dd56905267e2e2b0d72a34c62`

- :ref:`exhale_function_namespacetvm_1a81851f002c71fea8695a2e2bd29eefc7`

- :ref:`exhale_function_namespacetvm_1aabc6cf9e961e38f1c2b335bd36277487`

- :ref:`exhale_function_namespacetvm_1a93f1c7a184054d6af9aab47a636e912f`

- :ref:`exhale_function_namespacetvm_1ab480e963072a0d618a1f10d597167545`

- :ref:`exhale_function_namespacetvm_1a1bb911ef90e2f328ba2ae63387b9e936`

- :ref:`exhale_function_namespacetvm_1a643251a6b1d19348c4078bb6bbc60d1b`

- :ref:`exhale_function_namespacetvm_1a8e2950323c16d13a9504ec30071f0579`

- :ref:`exhale_function_namespacetvm_1a7a9c412ffac336a65d93d2967d60dcbe`

- :ref:`exhale_function_namespacetvm_1a8c7a3555362a34c57e7eda751b699d6a`

- :ref:`exhale_function_namespacetvm_1a7fdc1742f400de487cb807737b16aacb`

- :ref:`exhale_function_namespacetvm_1adfe0027298faaa905130ff80023b4804`

- :ref:`exhale_function_namespacetvm_1ac58e284059c962f5a96e46d98fe24e8d`

- :ref:`exhale_function_namespacetvm_1ab042ef46f754a15ae6c332babcd67907`

- :ref:`exhale_function_namespacetvm_1aa99dbf24a4d6e8c09587296bda16ff0d`

- :ref:`exhale_function_namespacetvm_1a34d3774087ee1b5cddd400921762235f`

- :ref:`exhale_function_namespacetvm_1aafc5e5fa4393becb77c4b6d7da3ace2d`

- :ref:`exhale_function_namespacetvm_1a98593b4c0d2622859925df160ca32f3d`

- :ref:`exhale_function_namespacetvm_1a3fa4f9842f6f0d7c651561a15d29f47a`

- :ref:`exhale_function_namespacetvm_1ad6a6184a0fe49fbe528783f008bb5156`

- :ref:`exhale_function_namespacetvm_1a7e76c8a0d99d28429fd531bdf9f7598e`

- :ref:`exhale_function_namespacetvm_1a44086a55195e22d4a22d637beb37963a`

- :ref:`exhale_function_namespacetvm_1aa303a127671da70407f3da30b6467be4`

- :ref:`exhale_function_namespacetvm_1aa8f0356c6e54a55b35941e21cfb1c44c`

- :ref:`exhale_function_namespacetvm_1accef0c52441f482aa49f76c6b72d97ee`

