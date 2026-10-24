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
"""Generate tensor programs, inspect their IR, and check source round trips."""

# TVMScript resolves dependent parameter annotations when constructing IR.

# begin tvmscript example
from __future__ import annotations

from tvm_tirx import ir, tirx
from tvm_tirx.script import from_source
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@T.function
def f(n: T.int64, A: T.Tensor((2 * n, 128), "float32")) -> T.void:
    for i, j in T.grid(2 * n, T.int64(128)):
        A[i, j] = A[i, j] + T.float32(1)


# end tvmscript example


# begin script helpers
@T.inline
def increment(value):
    return value + ir.const(1, value.ty)


@T.macro
def increment_at(A, i, j):
    A[i, j] = increment(A[i, j])


# end script helpers


# begin program factory
def make_increment(width, dtype="float32"):
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive Python integer")

    @T.function
    def generated(n: T.int64, A: T.Tensor((2 * n, width), dtype)) -> T.void:
        for i, j in T.grid(2 * n, T.int64(width)):
            increment_at(A, i, j)

    return generated


wide = make_increment(128)
narrow = make_increment(64, "float16")
# end program factory

# begin print generated function
source = wide.script()
print(source)
# end print generated function

# begin diagnostic display
wide.show(
    print_line_numbers=True,
    obj_to_underline=[wide.body[0]],
)
# end diagnostic display

# begin printing and parsing
restored = from_source(source, extra_vars={"T": T})
ir.assert_structural_equal(wide, restored)
# end printing and parsing


# begin module round trip
@I.ir_module
class Kernels:
    wide = wide.with_attr("global_symbol", "wide")
    narrow = narrow.with_attr("global_symbol", "narrow")


module_source = Kernels.script()
restored_module = from_source(module_source, extra_vars={"I": I, "T": T})
ir.assert_structural_equal(Kernels, restored_module)
# end module round trip

# begin standalone source
standalone_source = Kernels.script(extra_config={"ir.comment_imports": False})
standalone_module = from_source(standalone_source)
ir.assert_structural_equal(Kernels, standalone_module)
# end standalone source

assert isinstance(f, tirx.Function)
ir.assert_structural_equal(f.with_attr("global_symbol", ""), wide.with_attr("global_symbol", ""))
for program, width, dtype in ((wide, 128, "float32"), (narrow, 64, "float16")):
    n, tensor = program.params
    assert tensor.ty.shape[0].b.same_as(n)
    assert program.body[0].end.b.same_as(n)
    assert int(tensor.ty.shape[1]) == width
    assert tensor.ty.dtype == ir.PrimType(dtype)
    assert tirx.analysis.check_well_formed(program)
assert tirx.analysis.check_well_formed(Kernels)
assert "increment_at(" not in source and "increment(" not in source
assert wide.params[1].ty.script().startswith("T.Tensor(")
assert "for " in wide.body[0].script()
assert source == wide.script()
try:
    from_source(source)
except SyntaxError:
    pass
else:
    raise AssertionError("Source text must supply its namespace through imports or extra_vars")
try:
    make_increment(0)
except ValueError:
    pass
else:
    raise AssertionError("The factory should reject a zero width")
print("Printing and parsing example passed.")
