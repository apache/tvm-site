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
"""Add one to each element on an NVIDIA GPU. Requires LLVM and CUDA support."""

# begin program
import numpy as np

from tvm_tirx import cuda, runtime, tirx
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,)))
    i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
    if i < n:
        B[i] = A[i] + 1


compiled = tirx.compile(add_one, target="cuda")
dev = cuda(0)
a = runtime.tensor(np.arange(10, dtype="float32"), device=dev)
b = runtime.empty(a.shape, a.dtype, device=dev)
compiled(a, b)
np.testing.assert_allclose(b.numpy(), a.numpy() + 1)
print(b.numpy())
# end program
