# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""This example defines, compiles, and runs a bias + sigmoid program in TVMScript.

TIRx needs LLVM support. The example runs from the repository root with:
    uv run --no-sync python docs/ir/examples/tvmscript_sigmoid.py

X and Y have shape (2*n, channels); bias has one value per channel.
The code in bias_sigmoid processes one element at a time. run() compiles a
130-channel specialization and compares the output against NumPy.
"""

# begin program
from __future__ import annotations

from tvm_tirx import ir, tirx
from tvm_tirx.script import tirx as T


def make_bias_sigmoid(channels=130):
    n = T.dynamic("n", "int64")

    @T.function
    def bias_sigmoid(
        X: T.Tensor((2 * n, channels), T.float32),
        bias: T.Tensor((channels,), T.float32),
        Y: T.Tensor((2 * n, channels), T.float32),
    ):
        for i in range(2 * n):
            for j in range(T.int64(channels)):
                biased = X[i, j] + bias[j]
                Y[i, j] = T.float32(1) / (T.float32(1) + T.exp(-biased))

    return bias_sigmoid


program = ir.IRModule({"bias_sigmoid": make_bias_sigmoid(130)})


def run():
    import numpy as np

    from tvm_tirx import runtime

    function = program["bias_sigmoid"]
    input_var = function.params[0]  # ir.Var named X.
    tensor_type = input_var.ty  # tirx.TensorType: shape=(2 * n, 130), dtype=float32.
    row_extent = tensor_type.shape[0]  # ir.prim.Mul: a=2, b=n, ty=ir.PrimType("int64").
    location = row_extent.loc  # Source location of 2 * n in X's type annotation.
    print("Input type:", tensor_type.script())
    print("Rows:", row_extent.script())
    print("Rows source:", location.source_name.name)
    print("Rows start:", location.start_line, location.start_column)

    print(program.script())
    compiled = tirx.compile(program, target="llvm")
    batch, channels = 16, 130
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(batch, channels)).astype("float32")
    offsets = rng.normal(size=channels).astype("float32")
    x = runtime.tensor(inputs)
    bias = runtime.tensor(offsets)
    output = runtime.tensor(np.full_like(inputs, np.nan))
    compiled["bias_sigmoid"](x, bias, output)

    expected = 1.0 / (1.0 + np.exp(-(inputs + offsets)))
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-7)
    print("bias + sigmoid matches NumPy.")


if __name__ == "__main__":
    run()
# end program
