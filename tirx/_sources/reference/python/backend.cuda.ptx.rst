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

tvm_tirx.backend.cuda.ptx
=========================

.. automodule:: tvm_tirx.backend.cuda.ptx
   :no-members:

.. currentmodule:: tvm_tirx.backend.cuda.ptx

Classes
-------

.. autosummary::
   :toctree: generated/

   PTXNamespace

Functions
---------

.. autosummary::
   :toctree: generated/

   instruction_forms
   instruction_name

Instruction entries
-------------------

``python/tvm_tirx/backend/cuda/ptx/table.py`` defines ``TABLE`` and its
``InstructionEntry`` records. Each entry describes one operand shape;
multiple entries may share a user-facing instruction mnemonic.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``operands``
     - Operand order, register or address kind, read/write direction, dtype, and register-group size.
   * - ``slots``
     - Modifier positions, legal tokens, and optional positions.
   * - ``check``, ``imm_check``
     - Checks for modifier combinations and concrete immediate values.
   * - ``asm_volatile``, ``orders_memory``
     - Inline-assembly retention and compiler memory-ordering requirements.
   * - ``cert_arch``
     - Architecture used by instruction certification tests.

Predication and special operands
--------------------------------

PTX call destinations are leading operands, and the call itself returns
no value. Inside TVMScript, the keyword-only ``pred`` gates an instruction.
For example, a predicated load can preserve the destination's existing
value when the predicate is false:

.. code-block:: python

   T.ptx.ld.global_.b32(value, pointer, pred=predicate, preserve_dst=True)

Here ``value`` must be an initialized writable destination, and
``pointer`` selects the memory to read. Special operand helpers include:

.. list-table::
   :header-rows: 1

   * - Helper
     - Meaning
   * - ``T.ptx.pred(value)``
     - Mark an operand as a PTX ``.pred`` register.
   * - ``T.ptx.addr(base, byte_offset)``
     - Form an immediate-offset address for an instruction that accepts it.
   * - ``T.ptx.SINK``
     - Supply PTX's ``_`` sink in a table-marked operand position.

Developer generators
--------------------

Run these commands from the repository root with the Python package
available:

.. code-block:: bash

   python -m tvm_tirx.backend.cuda.ptx.gen_helpers add -o /tmp/ptx_add.cuh
   python -m tvm_tirx.backend.cuda.ptx.gen_stubs -o /tmp/ptx_stubs.pyi
   python -m tvm_tirx.backend.cuda.ptx.gen_coverage > /tmp/ptx_coverage.txt

``gen_helpers`` renders the selected family's supported instruction forms.
``gen_stubs`` generates signatures for editor tooling, and ``gen_coverage``
reports instruction families and variants. These files support inspection;
runtime code generation reads the instruction table directly. Each command's
``--help`` describes its available options. The :doc:`PTX Tablegen guide
</cuda/ptx_tablegen>` follows an instruction through the implementation.
