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

tvm_tirx.script.ir_builder.parser_protocol
==========================================

.. automodule:: tvm_tirx.script.ir_builder.parser_protocol
   :no-members:

.. currentmodule:: tvm_tirx.script.ir_builder.parser_protocol

Functions and Modules
---------------------

.. autosummary::
   :toctree: generated/

   ir_module
   decl_function
   function_
   arg_
   func_name_
   func_ret_type_

Bindings and Statements
-----------------------

.. autosummary::
   :toctree: generated/

   resolve_global_info_
   call_global_var_
   module_member_
   bind_
   decl_mutable_cell_
   set_mutable_cell_
   unpack_
   emit_
   setitem_
   setattr_

Control Flow
------------

.. autosummary::
   :toctree: generated/

   if_
   then_
   else_
   for_
   while_
   range_
   break_
   continue_
   return_
   assert_

Operators
---------

.. autosummary::
   :toctree: generated/

   if_then_else_
   and_
   or_
   not_
   lt_
   le_
   gt_
   ge_
   eq_
   ne_
