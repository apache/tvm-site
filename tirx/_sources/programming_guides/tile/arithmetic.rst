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

.. Adapted from Apache TVM:
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/add_sub_mul.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/div_rcp_sqrt.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/min_max.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/fma.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/cvt.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/ex2_lg2.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/abs_neg.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/ewise.rst

Arithmetic and Type Conversion
==============================

This guide shows how to compute with the register values loaded by tile
operations. ``apply`` selects a native arithmetic or conversion instruction;
``ewise`` applies a scalar TIRx function to each value. We will use an affine
transformation followed by ReLU, then convert float32 values to float16 and
express a SiLU activation as a scalar function.

An Affine Transformation and ReLU
---------------------------------

A common activation step scales each value, adds a bias, and clamps the
result at zero: ``B = max(2*A + 1, 0)``. The following kernel assigns four
values to each of 64 threads:

.. code-block:: python

    from tvm_tirx.script import tirx as T
    from tvm_tirx.tirx.layout import S, TileLayout
    from tvm_tirx.backend.cuda.axis import laneid, warpid

    register_layout = TileLayout(S[(2, 32, 4) : (1 @ warpid, 1 @ laneid, 1)])

    @T.function
    def affine_relu(
        A: T.Tensor((2, 32, 4), "float32", align=16),
        B: T.Tensor((2, 32, 4), "float32", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        registers = T.alloc_local((2, 32, 4), "float32", layout=register_layout)
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, A, scope="cta")
        scalars = registers.view(2, 32, 4, 1)
        T.cuda.tile.apply(
            T.ptx["fma.rn.f32"], scalars, scalars, T.float32(2), T.float32(1), scope="cta"
        )
        T.cuda.tile.apply(T.ptx["max.f32"], scalars, scalars, T.float32(0), scope="cta")
        T.cuda.tile.apply(T.ptx["st.global.v4.b32"], B, registers, scope="cta")

The load and store each use a four-word atom. The arithmetic view
``(2, 32, 4, 1)`` keeps the warp and lane group axes, but exposes four
repeats of a single-register atom. ``fma.rn.f32`` computes ``2*x + 1`` with
one rounding per result, and ``max.f32`` applies ReLU. The scalar constants
are reused for every repeat. Both arithmetic calls write back to the same
register storage.

We can check the result with finite float32 inputs:

.. code-block:: python

    import torch
    import tvm_tirx

    a = torch.linspace(-4, 4, 256, device="cuda").reshape(2, 32, 4)
    b = torch.empty_like(a)
    compiled = tvm_tirx.compile(affine_relu, target="cuda")
    compiled(a, b)
    torch.testing.assert_close(b, torch.clamp_min(a * 2 + 1, 0))

Other scalar instructions use the same single-register atom:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Operation
     - Example PTX forms
     - Reference
   * - Addition, subtraction, multiplication
     - ``add.rn.f32``, ``sub.rn.f32``, ``mul.rn.f32``
     - :ref:`Operand forms <cuda-tile-contract-add-sub-mul>`
   * - Division, reciprocal, square root
     - ``div.rn.f32``, ``rcp.approx.ftz.f32``, ``sqrt.rn.f32``
     - :ref:`Accuracy and operands <cuda-tile-contract-div-rcp-sqrt>`
   * - Elementwise minimum and maximum
     - ``min.f32``, ``max.f32``
     - :ref:`Binary and three-input forms <cuda-tile-contract-min-max>`
   * - Fused multiply-add
     - ``fma.rn.f32``
     - :ref:`Rounding and packing <cuda-tile-contract-fma>`
   * - Base-two exponential and logarithm
     - ``ex2.approx.ftz.f32``, ``lg2.approx.ftz.f32``
     - :ref:`Approximate forms <cuda-tile-contract-ex2-lg2>`
   * - Absolute value and negation
     - ``abs.f32``, ``neg.f32``
     - :ref:`Types and operands <cuda-tile-contract-abs-neg>`

Each repeat produces an independent result; an elementwise ``max`` does
not reduce a tile to one value. The instruction spelling selects rounding,
approximation, and subnormal handling. Packed forms such as
``add.rn.f16x2`` operate on two half values carried in a uint32 register;
the :ref:`register instruction reference <cuda-tile-contract-registers>`
specifies those representations and the predication options.

Numeric Conversion
------------------

The same flow can produce float16 output. ``cvt.rn.f16.f32`` converts one
float32 value with round-to-nearest-even. Its destination is a 16-bit
register container, exposed below through a uint16 view:

.. code-block:: python

    @T.function
    def to_float16(
        A: T.Tensor((2, 32, 4), "float32", align=16),
        B: T.Tensor((2, 32, 4), "float16", align=8),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        registers = T.alloc_local((2, 32, 4), "float32", layout=register_layout)
        converted = T.alloc_local((2, 32, 4), "float16", layout=register_layout)
        words = converted.view("uint16")
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, A, scope="cta")
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            words.view(2, 32, 4, 1), registers.view(2, 32, 4, 1), scope="cta",
        )
        T.cuda.tile.apply(T.ptx["st.global.v4.b16"], B.view("uint16"), words, scope="cta")

``cvt`` performs the numeric conversion. The uint16 views expose the
resulting bits for the PTX operands and reuse the float16 storage. The
store writes four 16-bit words per thread, with an 8-byte-aligned output
row. With ``a`` from the preceding example, we can verify the converted
values:

.. code-block:: python

    half_result = torch.empty_like(a, dtype=torch.float16)
    tvm_tirx.compile(to_float16, target="cuda")(a, half_result)
    torch.testing.assert_close(half_result, a.to(torch.float16))

Packed conversions such as ``cvt.rn.f16x2.f32`` place two converted values
in one uint32 carrier. Their operand order and supported inverse forms
are specified in the :ref:`conversion reference <cuda-tile-contract-cvt>`.

Scalar Function Compositions
----------------------------

A scalar TIRx function can describe a longer computation. For example,
SiLU multiplies a value by its sigmoid activation:

.. code-block:: python

    @T.function(private=True)
    def silu(value: T.float32) -> T.float32:
        return value / (T.float32(1) + T.exp(-value))

Inside ``affine_relu``, replace the two arithmetic calls with the following
call to apply SiLU to the loaded values:

.. code-block:: python

    T.cuda.tile.ewise(silu, registers, registers, scope="cta")

``ewise`` uses the original ``(2, 32, 4)`` view: the last axis now selects
four scalar invocations per thread. The scalar function is inlined for
each value and may expand to several instructions. Its parameter and
result types must match the register elements. The load and store remain
in the kernel, and the :ref:`scalar function reference
<cuda-tile-contract-ewise>` describes the supported function bodies.
