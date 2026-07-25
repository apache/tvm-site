
.. _file_tvm_ffi_optional.h:

File optional.h
===============

|exhale_lsh| :ref:`Parent directory <dir_tvm_ffi>` (``tvm/ffi``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Runtime Optional container types. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``tvm/ffi/optional.h``)
-----------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_tvm_ffi_optional.h.rst



Detailed Description
--------------------

Optional<T> uses a hybrid representation. For types that enable Any storage (``TypeTraits<T>::storage_enabled``), it is backed by a single :ref:`exhale_struct_structTVMFFIAny` (Any) with nullopt represented as kTVMFFINone, mirroring Variant<...>; the layout is then independent of T (sizeof == sizeof(Any)) which keeps the ABI stable. For types that do not enable storage (e.g. non-owning view types) it falls back to std::optional<T>. 






Includes
--------


- ``optional`` (:ref:`file_tvm_ffi_optional.h`)

- ``string`` (:ref:`file_tvm_ffi_string.h`)

- ``tvm/ffi/any.h`` (:ref:`file_tvm_ffi_any.h`)

- ``tvm/ffi/error.h`` (:ref:`file_tvm_ffi_error.h`)

- ``tvm/ffi/object.h`` (:ref:`file_tvm_ffi_object.h`)

- ``tvm/ffi/string.h`` (:ref:`file_tvm_ffi_string.h`)

- ``utility``



Included By
-----------


- :ref:`file_tvm_ffi_cast.h`

- :ref:`file_tvm_ffi_container_array.h`

- :ref:`file_tvm_ffi_container_dict.h`

- :ref:`file_tvm_ffi_container_map.h`

- :ref:`file_tvm_ffi_container_map_base.h`

- :ref:`file_tvm_ffi_container_variant.h`

- :ref:`file_tvm_ffi_extra_dataclass.h`

- :ref:`file_tvm_ffi_extra_structural_equal.h`

- :ref:`file_tvm_ffi_extra_structural_visit.h`

- :ref:`file_tvm_ffi_extra_visit_error_context.h`

- :ref:`file_tvm_ffi_reflection_overload.h`

- :ref:`file_tvm_ffi_reflection_registry.h`

- :ref:`file_tvm_ffi_tvm_ffi.h`




Namespaces
----------


- :ref:`namespace_tvm`

- :ref:`namespace_tvm__ffi`

