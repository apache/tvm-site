
.. _file_tvm_support_serializer.h:

File serializer.h
=================

|exhale_lsh| :ref:`Parent directory <dir_tvm_support>` (``tvm/support``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Serializer<T> specializations for :ref:`exhale_class_classtvm_1_1support_1_1Stream`. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/support/serializer.h``)
-----------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_support_serializer.h.rst



Detailed Description
--------------------

Built-in support for:

- Arithmetic types (endian-aware)

- Enum types (via underlying arithmetic type)

- std::string, std::vector<T>, std::pair<A,B>, std::unordered_map<K,V>

- DLDataType, DLDevice



Custom types with Save(Stream*)/Load(Stream*) methods should define Serializer<Type> specializations directly. 




Includes
--------


- ``cstdint``

- ``cstring``

- ``dlpack/dlpack.h``

- ``string``

- ``tvm/ffi/container/array.h``

- ``tvm/ffi/container/map.h``

- ``tvm/ffi/endian.h``

- ``tvm/ffi/string.h``

- ``tvm/support/io.h`` (:ref:`file_tvm_support_io.h`)

- ``type_traits``

- ``unordered_map``

- ``utility``

- ``vector`` (:ref:`file_tvm_tirx_op_vector.h`)



Included By
-----------


- :ref:`file_tvm_runtime_tensor.h`




Namespaces
----------


- :ref:`namespace_tvm`

- :ref:`namespace_tvm__support`


Classes
-------


- :ref:`exhale_struct_structtvm_1_1support_1_1Serializer_3_01DLDataType_01_4`

- :ref:`exhale_struct_structtvm_1_1support_1_1Serializer_3_01DLDevice_01_4`

- :ref:`exhale_struct_structtvm_1_1support_1_1Serializer_3_01ffi_1_1Array_3_01T_01_4_01_4`

- :ref:`exhale_struct_structtvm_1_1support_1_1Serializer_3_01ffi_1_1Bytes_01_4`

- :ref:`exhale_struct_structtvm_1_1support_1_1Serializer_3_01ffi_1_1Map_3_01K_00_01V_01_4_01_4`

- :ref:`exhale_struct_structtvm_1_1support_1_1Serializer_3_01ffi_1_1String_01_4`

