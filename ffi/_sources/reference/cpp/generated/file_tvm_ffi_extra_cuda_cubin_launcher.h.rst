
.. _file_tvm_ffi_extra_cuda_cubin_launcher.h:

File cubin_launcher.h
=====================

|exhale_lsh| :ref:`Parent directory <dir_tvm_ffi_extra_cuda>` (``tvm/ffi/extra/cuda``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



CUDA CUBIN launcher utility for loading and executing CUDA kernels. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/ffi/extra/cuda/cubin_launcher.h``)
----------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_ffi_extra_cuda_cubin_launcher.h.rst



Detailed Description
--------------------

This header provides a lightweight C++ wrapper around CUDA Runtime API for loading CUBIN modules and launching kernels. It supports:

- Loading CUBIN from memory (embedded data)

- Multi-GPU execution using CUDA primary contexts

- Kernel parameter management and launch configuration 






Includes
--------


- ``cstdint``

- ``cstring``

- ``cuda.h``

- ``cuda_runtime.h``

- ``tvm/ffi/error.h`` (:ref:`file_tvm_ffi_error.h`)

- ``tvm/ffi/extra/c_env_api.h`` (:ref:`file_tvm_ffi_extra_c_env_api.h`)

- ``tvm/ffi/extra/cuda/base.h`` (:ref:`file_tvm_ffi_extra_cuda_base.h`)

- ``tvm/ffi/extra/cuda/internal/unified_api.h`` (:ref:`file_tvm_ffi_extra_cuda_internal_unified_api.h`)

- ``tvm/ffi/string.h`` (:ref:`file_tvm_ffi_string.h`)






Namespaces
----------


- :ref:`namespace_tvm`

- :ref:`namespace_tvm__ffi`


Classes
-------


- :ref:`exhale_class_classtvm_1_1ffi_1_1CubinKernel`

- :ref:`exhale_class_classtvm_1_1ffi_1_1CubinModule`


Defines
-------


- :ref:`exhale_define_cubin__launcher_8h_1a99736a44462543179cb434ebd4512ade`

- :ref:`exhale_define_cubin__launcher_8h_1a55832b50e83cf39108f1e306f031433d`

- :ref:`exhale_define_cubin__launcher_8h_1ae8d64fa1cc7db9d38632e32054df72fc`

