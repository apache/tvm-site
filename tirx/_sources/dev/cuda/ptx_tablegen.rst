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

PTX TableGen
============

This guide follows ``add.rn.f32`` from a TVMScript call to generated inline
PTX. TIRx uses table-driven generation: one instruction description defines its
operands and modifiers, and the backend uses that description to check calls
and generate CUDA helpers. We will inspect this description and its generated
code, then see how to add support for another instruction form.

Operands and Modifiers
-----------------------

The :ref:`direct PTX example <cuda-direct-ptx>` adds two float32 values
with a writable destination:

.. code-block:: python

   T.ptx.add.rn.f32(result, A[tid], B[tid])

The instruction descriptions live in
`python/tvm_tirx/backend/cuda/ptx/table.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/ptx/table.py>`__.
Each ``InstructionEntry`` records the operand order and allowed modifier
values. The floating-point entry ``TABLE["add"]`` covers this form.
``tokens_for`` checks the chosen modifiers and puts them in slot order:

.. code-block:: python

   from tvm_tirx.backend.cuda.ptx.table import TABLE, tokens_for

   entry = TABLE["add"]
   tokens = tokens_for(entry, rnd="rn", type="f32")
   modifiers = {slot.name: token for slot, token in zip(entry.slots, tokens) if token}
   print(entry.op_name)
   print([operand.name for operand in entry.operands])
   print(modifiers)

The output is:

.. code-block:: text

   tirx.ptx.add
   ['d', 'a', 'b']
   {'rnd': 'rn', 'type': 'f32'}

``tirx.ptx.add`` is the IR operator name for this entry. ``result``
corresponds to destination ``d``; ``A[tid]`` and ``B[tid]``
correspond to sources ``a`` and ``b``. The ``rnd`` slot selects ``rn``
(round to nearest, ties to even), and the ``type`` slot selects ``f32``.
The printed modifier map contains the choices used by this call.

Generated CUDA Helper
----------------------

`python/tvm_tirx/backend/cuda/ptx/engine.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/ptx/engine.py>`__
registers the table entries as IR operators and constructs an ``ir.Call``
with result type ``void`` for the instruction. Its arguments include the
destination and both inputs. During CUDA source generation, the backend uses
`python/tvm_tirx/backend/cuda/ptx/render.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/ptx/render.py>`__
to emit a helper for the selected modifiers and operand types:

.. code-block:: cuda

   __forceinline__ __device__ void tvm_builtin_ptx_add_rn_f32(
       float& __d, float __a, float __b) {
     asm volatile("add.rn.f32 %0, %1, %2;"
                  : "=f"(__d) : "f"(__a), "f"(__b));
   }

The generated kernel calls it with the result's storage and the two
input values:

.. code-block:: cuda

   tvm_builtin_ptx_add_rn_f32(
       result_ptr[0], A_ptr[((int)threadIdx.x)], B_ptr[((int)threadIdx.x)]);

The reference parameter ``__d`` refers to ``result_ptr[0]``. The ``"=f"``
constraint writes that destination, while the two ``"f"`` constraints read
``__a`` and ``__b``. This is how a void instruction call produces a result
through its destination argument. The next tensor store writes it to
``C[tid]`` in the original example.

To inspect the helpers generated from this entry, run:

.. code-block:: bash

   python -m tvm_tirx.backend.cuda.ptx.gen_helpers add -o /tmp/ptx_add.cuh

Extend the instruction table
----------------------------

If a kernel needs a PTX form that ``T.ptx`` does not yet recognize, the
backend needs a description of its operands and modifiers. That description
allows the backend to recognize the call, check its arguments, and generate
the corresponding CUDA helper.

The full ``add`` entry already supports our example. A reduced, standalone
description of ``add.rn.f32`` lets us inspect the fields we would use for
a new form and the helper source they produce:

.. code-block:: python

   from tvm_tirx.backend.cuda.ptx.render import render_variant
   from tvm_tirx.backend.cuda.ptx.table import (
       InstructionEntry,
       ModifierSlot,
       OperandSlot,
       tokens_for,
   )

   demo = InstructionEntry(
       name="add_demo",
       mnemonic="add",
       operands=(
           OperandSlot("d", rw="w"),
           OperandSlot("a"),
           OperandSlot("b"),
       ),
       slots=(
           ModifierSlot("rnd", ("rn",)),
           ModifierSlot("type", ("f32",)),
       ),
       cert_arch="sm_90",
   )
   tokens = tokens_for(demo, rnd="rn", type="f32")
   opcode, helper, source = render_variant(demo, tokens)
   print(opcode)
   print(helper)

The output is:

.. code-block:: text

   add.rn.f32
   tvm_builtin_ptx_add_demo_rn_f32

``mnemonic`` sets ``add`` in the emitted instruction, while ``name``
identifies the description and appears in the helper name. ``rw="w"``
marks ``d`` as the output, and the slots restrict this description to
``rn.f32``. The returned ``source`` is the CUDA C++ helper definition as a
string, containing the inline PTX statement shown earlier. At this stage,
``demo`` is a standalone description used to generate that string.

To make an unsupported form available through ``T.ptx``, add its
``InstructionEntry`` to ``_ENTRIES`` in ``table.py``. The backend registers
the table entries as IR operators and code-generation callbacks when it
loads them. If an existing entry already has the required operand
arrangement, extend its modifier choices and rules. Use ``check`` for
rules involving several modifiers, and ``imm_check`` for restrictions on
constant operand values. A different operand arrangement can use another
entry with the same mnemonic.

The new form then needs its own test in
`tests/python/tirx/codegen/test_ptx_dialect.py
<https://github.com/apache/tvm-tirx/blob/main/tests/python/tirx/codegen/test_ptx_dialect.py>`__
that constructs a ``T.ptx`` call in a small kernel, checks the generated
instruction, and checks that invalid inputs are rejected. Use the file's
``_assert_ptxas_ok`` helper on the generated CUDA source to compile the
kernel to a CUBIN with NVCC. The caller ensures the new instruction reaches
assembly, where its validity is checked for the selected GPU architecture.
The entry's ``cert_arch`` selects a target supporting all its variants
for the table-wide assembly tests.

The :doc:`PTX reference </reference/python/backend.cuda.ptx>` covers
additional entry fields, special operands, and stub-generation tools.
