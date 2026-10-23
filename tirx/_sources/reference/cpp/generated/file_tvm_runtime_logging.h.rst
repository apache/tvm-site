
.. _file_tvm_runtime_logging.h:

File logging.h
==============

|exhale_lsh| :ref:`Parent directory <dir_tvm_runtime>` (``tvm/runtime``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



logging utilities 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/runtime/logging.h``)
--------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_runtime_logging.h.rst



Detailed Description
--------------------

We use the following facilities from tvm/ffi/error.h for error handling and checking:



- TVM_FFI_THROW(ErrorKind) << "msg";

- TVM_FFI_CHECK(cond, ErrorKind) << "msg";

- TVM_FFI_CHECK_EQ(x, y, ErrorKind) << "msg";

- TVM_FFI_ICHECK(x) << "msg"; // InternalError

- TVM_FFI_ICHECK_EQ(x, y) << "msg";

- TVM_FFI_DCHECK(x) << "msg"; // Debug-only InternalError



:ref:`exhale_define_logging_8h_1aba7b09d6e8fbe414c23705ad24dde6ff`, :ref:`exhale_define_logging_8h_1aba7b09d6e8fbe414c23705ad24dde6ff`, :ref:`exhale_define_logging_8h_1aba7b09d6e8fbe414c23705ad24dde6ff` are kept for logging. :ref:`exhale_define_logging_8h_1aba7b09d6e8fbe414c23705ad24dde6ff` is kept for completeness, it throws InternalError. 




Includes
--------


- ``ctime``

- ``iomanip``

- ``iostream``

- ``sstream``

- ``string``

- ``tvm/ffi/error.h``

- ``tvm/runtime/base.h`` (:ref:`file_tvm_runtime_base.h`)

- ``unordered_map``

- ``vector`` (:ref:`file_tvm_tirx_op_vector.h`)



Included By
-----------


- :ref:`file_tvm_ir_prim_const_fold.h`




Namespaces
----------


- :ref:`namespace_tvm`

- :ref:`namespace_tvm__runtime`


Defines
-------


- :ref:`exhale_define_logging_8h_1a8a5e56158e552f01a1d9703ca21cec54`

- :ref:`exhale_define_logging_8h_1a30c3e3c313489ab5e42bb12fcfcc5968`

- :ref:`exhale_define_logging_8h_1a0581caafbd212149004303f517163b1f`

- :ref:`exhale_define_logging_8h_1aba7b09d6e8fbe414c23705ad24dde6ff`

- :ref:`exhale_define_logging_8h_1a6ff63e8955665c4a58b1598f2b07c51a`

- :ref:`exhale_define_logging_8h_1a4e35d803319a988cdc39e8ec8bbd99a2`

- :ref:`exhale_define_logging_8h_1aced66975c154ea0e2a8ec3bc818b4e08`

- :ref:`exhale_define_logging_8h_1ac2164369b4d2bf72296f8ba6ea576ecf`

- :ref:`exhale_define_logging_8h_1a09f7d88282cf92c9f231270ac113e5c6`

- :ref:`exhale_define_logging_8h_1aeb4f36db01bd128c7afeac5889dac311`

- :ref:`exhale_define_logging_8h_1adf4476a6a4ea6c74231c826e899d7189`

- :ref:`exhale_define_logging_8h_1afc28ec91d066c96a3a139441e029f326`

- :ref:`exhale_define_logging_8h_1aacc9cd093b92dffee96ffa841a6e22b4`

