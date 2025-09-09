.. _exhale_struct_structtvm_1_1ffi_1_1is__valid__iterator:

Template Struct is_valid_iterator
=================================

- Defined in :ref:`file_tvm_ffi_container_array.h`


Inheritance Relationships
-------------------------

Base Type
*********

- ``public std::bool_constant< std::is_same_v< T, std::remove_cv_t< std::remove_reference_t< decltype(*std::declval< IterType >())> > >||std::is_base_of_v< T, std::remove_cv_t< std::remove_reference_t< decltype(*std::declval< IterType >())> > > >``


Derived Type
************

- ``public tvm::ffi::is_valid_iterator< Optional< T >, IterType >`` (:ref:`exhale_struct_structtvm_1_1ffi_1_1is__valid__iterator_3_01Optional_3_01T_01_4_00_01IterType_01_4`)


Struct Documentation
--------------------


.. doxygenstruct:: tvm::ffi::is_valid_iterator
   :project: tvm-ffi
   :members:
   :protected-members:
   :undoc-members: