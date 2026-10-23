
.. _file_tvm_sym_iter_affine_map.h:

File iter_affine_map.h
======================

|exhale_lsh| :ref:`Parent directory <dir_tvm_sym>` (``tvm/sym``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Iterator quasi-affine mapping patterns. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/sym/iter_affine_map.h``)
------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_sym_iter_affine_map.h.rst



Detailed Description
--------------------

This file defines a collection of mapping patterns maps a collection of independent iterators to another collection of independent iterators.

There are two main kinds of mapping patterns:



- Fuse: fuse a collection of iterators into a single one

   domain(x0) = [0, 4), domain(x1) = [0, 3), domain(x2) = [0, 2) fuse(x0, x1, x2): y = x2 * 12 + x1 * 4 + x0 domain(y) = [0, 24)

- Split: split an iterator into multiple ones

   domain(x) = [0, 24) split(x, 3, 12): [y0, y1, y2] = [x % 3, (x % 12) / 3, x / 12] domain(y0) = [0, 3), domain(y1) = [0, 4), domain(y2) = [0, 2)



We use the name "(quasi)affine" to be consistent with the terminology used in the polyhedral compilation. Notably, fuse is an affine transformation, while split corresponds to additional floordiv/mod operations that can appear in quasi-affine transformations. 




Includes
--------


- ``tvm/ffi/reflection/registry.h``

- ``tvm/ir/cow.h`` (:ref:`file_tvm_ir_cow.h`)

- ``tvm/ir/expr.h`` (:ref:`file_tvm_ir_expr.h`)

- ``tvm/sym/analyzer.h`` (:ref:`file_tvm_sym_analyzer.h`)






Namespaces
----------


- :ref:`namespace_tvm`

- :ref:`namespace_tvm__sym`


Classes
-------


- :ref:`exhale_class_classtvm_1_1sym_1_1IterMapExpr`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterMapExprNode`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterMapResult`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterMapResultObj`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterMark`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterMarkNode`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterSplitExpr`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterSplitExprNode`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterSumExpr`

- :ref:`exhale_class_classtvm_1_1sym_1_1IterSumExprNode`


Enums
-----


- :ref:`exhale_enum_namespacetvm_1_1sym_1afbbf34adac93082f4ce8a4bbc2ad1e2b`


Functions
---------


- :ref:`exhale_function_namespacetvm_1_1sym_1aa121c8b4f76bb65164b45300d41fd029`

- :ref:`exhale_function_namespacetvm_1_1sym_1a5192f7b38d858496024283f683569b94`

- :ref:`exhale_function_namespacetvm_1_1sym_1ab9cba0a13c59652fa78efcaab81a089f`

- :ref:`exhale_function_namespacetvm_1_1sym_1af8cdffe8ada537e0cf1612864ea056ef`

- :ref:`exhale_function_namespacetvm_1_1sym_1a7162654cb6035c762374cc0a53bb0693`

- :ref:`exhale_function_namespacetvm_1_1sym_1a57a8454377f5bacf6d5eeffb86fc811c`

