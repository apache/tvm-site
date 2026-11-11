.. _exhale_struct_structtvm_1_1ffi_1_1is__valid__iterator_3_01Any_00_01IterType_01_4:

Template Struct is_valid_iterator< Any, IterType >
==================================================

- Defined in :ref:`file_tvm_ffi_container_array.h`


Inheritance Relationships
-------------------------

Base Types
**********

- ``public std::bool_constant< std::is_same_v< Any, std::remove_cv_t< std::remove_reference_t< decltype(*std::declval< IterType >())> > >||std::is_base_of_v< Any, std::remove_cv_t< std::remove_reference_t< decltype(*std::declval< IterType >())> > > >``
- ``public std::true_type``


Struct Documentation
--------------------


.. doxygenstruct:: tvm::ffi::is_valid_iterator< Any, IterType >
   :project: tvm-ffi
   :members:
   :protected-members:
   :undoc-members: