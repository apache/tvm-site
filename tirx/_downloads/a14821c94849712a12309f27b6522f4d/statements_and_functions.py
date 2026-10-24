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
"""Assemble an integer increment function from expressions and statements."""

from tvm_tirx import ir, tirx

# begin statements
x = ir.Var("x", "int32")
added = x + 1
result = ir.Var("result", added.ty)
body = ir.SeqStmt(
    [
        ir.Bind(result, added),
        ir.Return(result),
    ]
)
# end statements

# begin branch
branch = ir.If(x > 0, ir.Return(added), ir.Return(x))
# end branch

# begin loop
i = ir.Var("i", "int32")
loop = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, ir.Evaluate(i))
# end loop

# begin function construction
value = ir.Var("value", "int32")
function = tirx.Function([value], ir.Return(value + 1), ret_type=value.ty)
# end function construction

# begin module construction
function = function.with_attr("global_symbol", "main")
entry = ir.GlobalVar("main")
module = ir.IRModule({entry: function})
# end module construction

# begin global call
global_call = ir.Call(entry, [ir.const(3, "int32")], ty=function.ret_type)
# end global call

# begin module serialization
restored = ir.load_json(ir.save_json(module))
ir.assert_structural_equal(module, restored)
# end module serialization

assert global_call.ty == x.ty
assert module[entry].same_as(module["main"])
assert tirx.analysis.check_well_formed(module)
assert tirx.analysis.check_well_formed(tirx.Function([x], body, ret_type=x.ty))
assert tirx.analysis.check_well_formed(tirx.Function([x], branch, ret_type=x.ty))
# begin module update
variant = module.clone()
variant["main"] = function.with_attr("example.note", "clone")
assert "example.note" not in module["main"].attrs
# end module update

escaped = tirx.Function([], [loop, ir.Evaluate(i)])
assert not tirx.analysis.check_well_formed(escaped, assert_mode=False)
executable = tirx.compile(module, target="llvm")
for value in (-2, 0, 3):
    assert executable(value) == value + 1
print("Statement, function, and execution examples passed.")
