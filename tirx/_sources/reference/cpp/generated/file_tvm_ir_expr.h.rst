
.. _file_tvm_ir_expr.h:

File expr.h
===========

|exhale_lsh| :ref:`Parent directory <dir_tvm_ir>` (``tvm/ir``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Base expr nodes in TVM. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/ir/expr.h``)
------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_ir_expr.h.rst





Includes
--------


- ``algorithm``

- ``cstddef``

- ``cstdint``

- ``functional``

- ``limits``

- ``optional``

- ``ostream``

- ``string``

- ``tvm/ffi/big_int.h``

- ``tvm/ffi/cast.h``

- ``tvm/ffi/container/array.h``

- ``tvm/ffi/container/tuple.h``

- ``tvm/ffi/container/variant.h``

- ``tvm/ffi/dtype.h``

- ``tvm/ffi/error.h``

- ``tvm/ffi/extra/dataclass.h``

- ``tvm/ffi/extra/structural_equal.h``

- ``tvm/ffi/extra/structural_hash.h``

- ``tvm/ffi/extra/structural_mutate.h``

- ``tvm/ffi/extra/structural_visit.h``

- ``tvm/ffi/function.h``

- ``tvm/ffi/optional.h``

- ``tvm/ffi/reflection/native_function.h``

- ``tvm/ffi/reflection/registry.h``

- ``tvm/ffi/string.h``

- ``tvm/ir/attrs.h`` (:ref:`file_tvm_ir_attrs.h`)

- ``tvm/ir/cow.h`` (:ref:`file_tvm_ir_cow.h`)

- ``tvm/ir/location.h`` (:ref:`file_tvm_ir_location.h`)

- ``tvm/ir/type.h`` (:ref:`file_tvm_ir_type.h`)

- ``type_traits``

- ``utility``



Included By
-----------


- :ref:`file_tvm_backend_cuda_attr.h`

- :ref:`file_tvm_codegen_intrin_rule.h`

- :ref:`file_tvm_ir_analysis.h`

- :ref:`file_tvm_ir_expr_functor.h`

- :ref:`file_tvm_ir_function.h`

- :ref:`file_tvm_ir_module.h`

- :ref:`file_tvm_ir_op.h`

- :ref:`file_tvm_ir_prim_expr.h`

- :ref:`file_tvm_ir_prim_vector_expr.h`

- :ref:`file_tvm_ir_unique_name_supply.h`

- :ref:`file_tvm_script_ir_builder_base.h`

- :ref:`file_tvm_script_ir_builder_frame.h`

- :ref:`file_tvm_script_ir_builder_ir.h`

- :ref:`file_tvm_script_printer_doc.h`

- :ref:`file_tvm_script_printer_doc_translator.h`

- :ref:`file_tvm_script_printer_support.h`

- :ref:`file_tvm_sym_analyzer.h`

- :ref:`file_tvm_sym_constraint.h`

- :ref:`file_tvm_sym_int_set.h`

- :ref:`file_tvm_sym_iter_affine_map.h`

- :ref:`file_tvm_target_target.h`

- :ref:`file_tvm_tirx_expr.h`

- :ref:`file_tvm_tirx_index_map.h`

- :ref:`file_tvm_tirx_layout.h`

- :ref:`file_tvm_tirx_op_attr.h`

- :ref:`file_tvm_tirx_type.h`




Namespaces
----------


- :ref:`namespace_tvm`

- :ref:`namespace_tvm__op_attr`

- :ref:`namespace_tvm__type_attr`


Classes
-------


- :ref:`exhale_class_classtvm_1_1Call`

- :ref:`exhale_class_classtvm_1_1CallObj`

- :ref:`exhale_class_classtvm_1_1Constant`

- :ref:`exhale_class_classtvm_1_1ConstantObj`

- :ref:`exhale_class_classtvm_1_1DataTypeImm`

- :ref:`exhale_class_classtvm_1_1DataTypeImmObj`

- :ref:`exhale_class_classtvm_1_1Expr`

- :ref:`exhale_class_classtvm_1_1ExprObj`

- :ref:`exhale_class_classtvm_1_1FloatImm`

- :ref:`exhale_class_classtvm_1_1FloatImmObj`

- :ref:`exhale_class_classtvm_1_1GenericConst`

- :ref:`exhale_class_classtvm_1_1GenericConstObj`

- :ref:`exhale_class_classtvm_1_1GlobalVar`

- :ref:`exhale_class_classtvm_1_1GlobalVarObj`

- :ref:`exhale_class_classtvm_1_1IntExpr`

- :ref:`exhale_class_classtvm_1_1IntImm`

- :ref:`exhale_class_classtvm_1_1IntImmObj`

- :ref:`exhale_class_classtvm_1_1LambdaExpr`

- :ref:`exhale_class_classtvm_1_1LambdaExprObj`

- :ref:`exhale_class_classtvm_1_1OpaqueExpr`

- :ref:`exhale_class_classtvm_1_1OpaqueExprObj`

- :ref:`exhale_class_classtvm_1_1PrimExpr`

- :ref:`exhale_class_classtvm_1_1PrimVar`

- :ref:`exhale_class_classtvm_1_1Range`

- :ref:`exhale_class_classtvm_1_1RangeObj`

- :ref:`exhale_class_classtvm_1_1StagingExpr`

- :ref:`exhale_class_classtvm_1_1StagingExprObj`

- :ref:`exhale_class_classtvm_1_1StringImm`

- :ref:`exhale_class_classtvm_1_1StringImmObj`

- :ref:`exhale_class_classtvm_1_1TensorLoad`

- :ref:`exhale_class_classtvm_1_1TensorLoadObj`

- :ref:`exhale_class_classtvm_1_1TensorRegion`

- :ref:`exhale_class_classtvm_1_1TensorRegionObj`

- :ref:`exhale_class_classtvm_1_1Tuple`

- :ref:`exhale_class_classtvm_1_1TupleGetItem`

- :ref:`exhale_class_classtvm_1_1TupleGetItemObj`

- :ref:`exhale_class_classtvm_1_1TupleObj`

- :ref:`exhale_class_classtvm_1_1TypedExpr`

- :ref:`exhale_class_classtvm_1_1Var`

- :ref:`exhale_class_classtvm_1_1VarObj`


Enums
-----


- :ref:`exhale_enum_namespacetvm_1aa80b1c23bd976ef8420358cfd4cd724a`


Functions
---------


- :ref:`exhale_function_namespacetvm_1a1bd0ed1cda03a92fcc524e6191105b8d`

- :ref:`exhale_function_namespacetvm_1a7d2c6a6983bd53ffe43dd5881f801023`


Typedefs
--------


- :ref:`exhale_typedef_namespacetvm_1aa813ea01ea7f9661b776a42817fd16bb`


Variables
---------


- :ref:`exhale_variable_namespacetvm_1_1op__attr_1a2b35e64d19937a4edb3ccaadf5911729`

- :ref:`exhale_variable_namespacetvm_1_1type__attr_1ad443e8fd6cc147752bfdfb7abb6e6f87`

- :ref:`exhale_variable_namespacetvm_1_1type__attr_1adc7503614dd7343d8a2b31eebd8d835a`

