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

tvm_tirx.backend.cuda.tile
==========================

.. automodule:: tvm_tirx.backend.cuda.tile
   :no-members:

.. currentmodule:: tvm_tirx.backend.cuda.tile

The CUDA tile backend lowers explicit instructions over tensor regions.
See :doc:`/programming_guides/tile/tile_patterns` for an introduction and the
:doc:`instruction reference <backend.cuda.tile.instructions>` for
parameters, operand coordinates, and storage requirements. The public
constructors are :doc:`apply and ewise <backend.cuda.script.tile>`.

Instruction reference
---------------------

.. toctree::
   :maxdepth: 1

   backend.cuda.tile.instructions

Namespaces
----------

.. toctree::
   :maxdepth: 1

   backend.cuda.tile.exec_context
