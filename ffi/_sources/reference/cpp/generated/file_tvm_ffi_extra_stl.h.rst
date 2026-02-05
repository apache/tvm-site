
.. _file_tvm_ffi_extra_stl.h:

File stl.h
==========

|exhale_lsh| :ref:`Parent directory <dir_tvm_ffi_extra>` (``tvm/ffi/extra``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



STL container support. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/ffi/extra/stl.h``)
------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_ffi_extra_stl.h.rst



Detailed Description
--------------------

This file is an extra extension of TVM FFI, which provides support for STL containers in C++ exported functions.

Whenever possible, prefer using tvm/ffi/container/ implementations, such as ``tvm::ffi::Array`` and ``tvm::ffi::Tuple``, over STL containers.

Native ffi objects comes with stable data layout and can be directly accessed through compiled languages (Rust) and DSLs(via LLVM) with raw pointer access for better performance and compatibility. 




Includes
--------


- ``algorithm``

- ``array`` (:ref:`file_tvm_ffi_container_array.h`)

- ``cstddef``

- ``cstdint``

- ``exception``

- ``functional``

- ``iterator``

- ``map`` (:ref:`file_tvm_ffi_container_map.h`)

- ``optional`` (:ref:`file_tvm_ffi_optional.h`)

- ``tuple`` (:ref:`file_tvm_ffi_container_tuple.h`)

- ``tvm/ffi/base_details.h``

- ``tvm/ffi/c_api.h`` (:ref:`file_tvm_ffi_c_api.h`)

- ``tvm/ffi/container/array.h`` (:ref:`file_tvm_ffi_container_array.h`)

- ``tvm/ffi/container/map.h`` (:ref:`file_tvm_ffi_container_map.h`)

- ``tvm/ffi/error.h`` (:ref:`file_tvm_ffi_error.h`)

- ``tvm/ffi/function.h`` (:ref:`file_tvm_ffi_function.h`)

- ``tvm/ffi/object.h`` (:ref:`file_tvm_ffi_object.h`)

- ``tvm/ffi/type_traits.h`` (:ref:`file_tvm_ffi_type_traits.h`)

- ``type_traits`` (:ref:`file_tvm_ffi_type_traits.h`)

- ``utility``

- ``variant`` (:ref:`file_tvm_ffi_container_variant.h`)

- ``vector``






Namespaces
----------


- :ref:`namespace_tvm`

- :ref:`namespace_tvm__ffi`

