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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/tile_ops.rst

Tile Operations
===============

TIRx provides two CUDA tile operations: ``T.cuda.tile.apply`` and
``T.cuda.tile.ewise``. ``apply`` repeats a specified PTX instruction, such as
a vector load or matrix multiplication, while ``ewise`` applies a scalar
TIRx function to register elements. You choose the operand views,
instruction variants, and synchronization required by the kernel.

The guides below cover data movement, arithmetic, asynchronous transfers,
and matrix instructions. Start with :doc:`tile_patterns` for a load,
compute, and store example that shows how operand axes divide the work
across threads.

For each instruction family's parameters, operand layouts, and alignment
requirements, see the
:doc:`instruction reference </reference/python/backend.cuda.tile.instructions>`.

.. toctree::
   :maxdepth: 1

   tile_patterns
   data_movement
   arithmetic

Asynchronous Transfers
----------------------

.. toctree::
   :maxdepth: 1

   async_copies
   tma

Matrix Instructions
-------------------

.. toctree::
   :maxdepth: 1

   warp_matrix
   blackwell_tensor_cores
