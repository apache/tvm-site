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
"""Inspect an operator, construct calls, and check result types."""

from tvm_tirx import ir

# begin operator lookup
op = ir.Op.get("prim.exp")
argument_names = [entry.name for entry in op.args_info]
effect = op.get_attr("TCallEffectKind")
# end operator lookup

# begin call contract
value = ir.const(1.0, "float32")
call = ir.Call(op, [value])
# end call contract

# begin function call type
callee = ir.Var("f", ir.FuncType([value.ty], value.ty))
function_call = ir.Call(callee, [value])
# end function call type

# begin validation failure
invalid = ir.Call(op, [], ty="float32")
# end validation failure
try:
    invalid.validate()
except TypeError as error:
    assert "expected 1 argument, got 0" in str(error)
else:
    raise AssertionError("Expected an argument-count error")

# begin provisional call
provisional = ir.Call(op, [value], ty=ir.Type.missing())
inferred = ir.reinfer_type(provisional)
# end provisional call

assert argument_names == ["x"] and effect == ir.CallEffectKind.Pure
assert call.ty == function_call.ty == inferred == value.ty
assert call.validate() is None
assert isinstance(provisional.ty, ir.MissingType)
print("Operator and call examples passed.")
