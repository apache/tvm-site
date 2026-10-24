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
"""Build and inspect typed scalar and tensor expressions.

Run ``python types_and_expressions.py`` with TIRx installed.

The four steps show scalar expressions, symbolic tensor shapes, operator
calls, and a custom type. Each step prints the constructed IR and its type.
For example, the tensor step prints::

    2. Tensor expressions
       Fixed tensor type: T.Tensor((8, 128), T.float32)
       Symbolic tensor type: T.Tensor((T.int64(2) * n, 128), T.float32)
       A[0, 0] : T.float32

The printed forms use TVMScript notation, such as T.int32 and T.Tensor(...).
"""

from tvm_ffi.dataclasses import py_class

from tvm_tirx import ir, tirx

# 1. Describe an integer computation, then construct its input and addition.
# begin composite type
integer_type = ir.PrimType("int32")
signature = ir.FuncType([integer_type], integer_type)
# end composite type

# begin scalar expressions
x = ir.Var("x", integer_type)
added = x + ir.const(1, integer_type)
# end scalar expressions

print("1. Scalar expressions")
print(f"   {added.script()} : {added.ty.script()}")
print(f"   Function signature: {signature.script()}")

# 2. A tensor type can use fixed dimensions or reuse integer expressions.
# begin constant tensor type
tile_type = tirx.TensorType((8, 128), "float32")
# end constant tensor type

# begin tensor expressions
n = ir.Var("n", "int64")
tensor_type = tirx.TensorType((2 * n, 128), "float32")
tensor = ir.Var("A", tensor_type)
element = tensor[0, 0]
# end tensor expressions

# The shape retains the same n object; no numeric value for n is needed here.
assert tensor.ty.shape[0].b.same_as(n)
print("\n2. Tensor expressions")
print(f"   Fixed tensor type: {tile_type.script()}")
print(f"   Symbolic tensor type: {tensor_type.script()}")
print(f"   {element.script()} : {element.ty.script()}")

# 3. Retrieve one operator and use it in calls with different input expressions.
# begin operator expressions
exp_op = ir.Op.get("prim.exp")
value = ir.Var("value", "float32")
exp_call = ir.Call(exp_op, [value])
# end operator expressions

tensor_exp = ir.Call(exp_op, [element])
assert tensor_exp.op.same_as(exp_call.op)
print("\n3. Operator calls")
print(f"   Registered operator: {exp_op.name}")
print(f"   {exp_call.script()} : {exp_call.ty.script()}")
print(f"   {tensor_exp.script()} : {tensor_exp.ty.script()}")


# 4. Define a type for a pair of values using existing Type objects as fields.
# begin custom type
@py_class("my_ext.PairType")
class PairType(ir.Type):
    first: ir.Type
    second: ir.Type


pair = ir.Var("p", PairType(ir.PrimType("int32"), ir.PrimType("float32")))
# end custom type

print("\n4. A custom type")
print(f"   {pair.name} : {type(pair.ty).__name__}")
print(f"   Component types: {pair.ty.first.script()}, {pair.ty.second.script()}")
print(f"   Inherited source location: {type(pair.ty.loc).__name__}")
