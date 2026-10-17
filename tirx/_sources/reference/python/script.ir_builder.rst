..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

tvm_tirx.script.ir_builder
==========================

.. automodule:: tvm_tirx.script.ir_builder
   :no-members:

.. currentmodule:: tvm_tirx.script.ir_builder

Classes
-------

.. autosummary::
   :toctree: generated/

   AlreadyEmitted
   IRBuilder
   IRModuleFrame

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Any``
     - :obj:`tvm_tirx.ir.AnyType`
   * - ``Call``
     - :obj:`tvm_tirx.ir.Call`
   * - ``DataTypeImm``
     - :obj:`tvm_tirx.ir.DataTypeImm`
   * - ``Func``
     - :obj:`tvm_tirx.ir.FuncType`
   * - ``FuncType``
     - :obj:`tvm_tirx.ir.FuncType`
   * - ``GenericConst``
     - :obj:`tvm_tirx.ir.GenericConst`
   * - ``Missing``
     - :obj:`tvm_tirx.ir.MissingType`
   * - ``MissingType``
     - :obj:`tvm_tirx.ir.MissingType`
   * - ``Op``
     - :obj:`tvm_tirx.ir.Op`
   * - ``Opaque``
     - :obj:`tvm_tirx.ir.OpaqueType`
   * - ``Prim``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``PrimType``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``Ptr``
     - :obj:`tvm_tirx.ir.PtrType`
   * - ``Range``
     - :obj:`tvm_tirx.ir.Range`
   * - ``String``
     - :obj:`tvm_tirx.ir.StringType`
   * - ``StringImm``
     - :obj:`tvm_tirx.ir.StringImm`
   * - ``StringType``
     - :obj:`tvm_tirx.ir.StringType`
   * - ``TensorRegion``
     - :obj:`tvm_tirx.ir.TensorRegion`
   * - ``TensorRegionType``
     - :obj:`tvm_tirx.ir.TensorRegionType`
   * - ``Tuple``
     - :obj:`tvm_tirx.ir.Tuple`
   * - ``Type``
     - :obj:`tvm_tirx.ir.Type`
   * - ``Var``
     - :obj:`tvm_tirx.ir.Var`

Functions
---------

.. autosummary::
   :toctree: generated/

   Lambda
   at_
   check_well_formed_
   constexpr
   decl_function
   dtype
   dynamic
   ir_module
   meta_var
   module_attrs
   module_get_attr
   module_global_infos
   module_member_
   module_set_attr
   resolve_global_info_
   resolve_global_info_args
   with_at_group_

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``make_node``
     - :obj:`tvm_tirx.ir.make_node`

Namespaces
----------

.. toctree::
   :maxdepth: 1

   script.ir_builder.frame
   script.ir_builder.ir
   script.ir_builder.stmt
   script.ir_builder.tracing
