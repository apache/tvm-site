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
"""Compile a function that multiplies complex numbers with my_ext, and check it against NumPy."""

import my_ext
import numpy as np

import tvm_tirx
from tvm_tirx import runtime
from tvm_tirx.script import tirx as T

# begin kernel
N = 16


@T.function
def complex_product(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    C: T.Tensor((N,), "float32"),
    D: T.Tensor((N,), "float32"),
    R: T.Tensor((N,), "float32"),
    Im: T.Tensor((N,), "float32"),
):
    for i in range(N):
        z = my_ext.complex(A[i], B[i])
        w = my_ext.complex(C[i], D[i])
        product = my_ext.cmul(z, w)
        R[i] = my_ext.real(product)
        Im[i] = my_ext.imag(product)


# end kernel

# begin compile
compiled = tvm_tirx.compile(complex_product, target="llvm", pipeline="my_ext")
# end compile

# begin comparison
rng = np.random.default_rng(0)
a, b, c, d = rng.standard_normal((4, N)).astype("float32")
inputs = [runtime.tensor(x) for x in (a, b, c, d)]
real_out = runtime.empty((N,), "float32")
imag_out = runtime.empty((N,), "float32")
compiled(*inputs, real_out, imag_out)

expected = (a + 1j * b) * (c + 1j * d)
np.testing.assert_allclose(real_out.numpy(), expected.real, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(imag_out.numpy(), expected.imag, rtol=1e-5, atol=1e-6)
# end comparison
print("Complex multiplication matches NumPy.")
