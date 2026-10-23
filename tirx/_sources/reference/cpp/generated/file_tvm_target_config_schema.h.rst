
.. _file_tvm_target_config_schema.h:

File config_schema.h
====================

|exhale_lsh| :ref:`Parent directory <dir_tvm_target>` (``tvm/target``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Minimal schema for dynamic config canonicalization and validation. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/target/config_schema.h``)
-------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_target_config_schema.h.rst



Detailed Description
--------------------

This utility is intended for dynamic map-like configs (e.g. Target options), where we still want type checking, optional defaulting, and canonicalization. 




Includes
--------


- ``sstream``

- ``string``

- ``tvm/ffi/container/map.h``

- ``tvm/ffi/function.h``

- ``tvm/ffi/reflection/registry.h``

- ``type_traits``

- ``unordered_map``

- ``utility``

- ``vector`` (:ref:`file_tvm_tirx_op_vector.h`)



Included By
-----------


- :ref:`file_tvm_target_target_kind.h`




Namespaces
----------


- :ref:`namespace_tvm`


Classes
-------


- :ref:`exhale_struct_structtvm_1_1ConfigSchema_1_1AttrValidator`

- :ref:`exhale_struct_structtvm_1_1ConfigSchema_1_1OptionEntry`

- :ref:`exhale_class_classtvm_1_1ConfigSchema`

