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
"""Add one to each element on the CPU. Requires LLVM support."""

# begin program
import torch

from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    for i in range(n):
        B[i] = A[i] + 1


add_one_compiled = tirx.compile(add_one, target="llvm")
a = torch.arange(10, dtype=torch.float32)
b = torch.empty(a.shape, dtype=a.dtype)
add_one_compiled(a, b)
torch.testing.assert_close(b, a + 1)
print(b)
# end program
