
.. _file_tvm_ir_pattern_match.h:

File pattern_match.h
====================

|exhale_lsh| :ref:`Parent directory <dir_tvm_ir>` (``tvm/ir``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Shared expression-template pattern and capture protocol. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/ir/pattern_match.h``)
---------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_ir_pattern_match.h.rst



Detailed Description
--------------------

.. code-block:: cpp

  #include <tvm/ir/prim/pattern_match.h>

  void MatchMultiplyAdd(const tvm::PrimExpr& expr) {
    tvm::PVar<tvm::PrimExpr> a, b, c;
    if ((a * b + c).Match(expr)) {
      tvm::PrimExpr lhs = a.Eval();
      tvm::PrimExpr rhs = b.Eval();
      tvm::PrimExpr addend = c.Eval();
    }
  }





Includes
--------


- ``optional``

- ``tuple``

- ``tvm/ffi/cast.h``

- ``type_traits``

- ``utility``



Included By
-----------


- :ref:`file_tvm_ir_prim_pattern_match.h`




Namespaces
----------


- :ref:`namespace_tvm`


Classes
-------


- :ref:`exhale_class_classtvm_1_1Pattern`

- :ref:`exhale_class_classtvm_1_1PConst`

- :ref:`exhale_class_classtvm_1_1PEqualChecker`

- :ref:`exhale_class_classtvm_1_1PMatchesOneOf`

- :ref:`exhale_class_classtvm_1_1PVar`

- :ref:`exhale_class_classtvm_1_1PVarWithCheck`

