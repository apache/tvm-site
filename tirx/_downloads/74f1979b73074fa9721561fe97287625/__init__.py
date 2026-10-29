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
"""The complex type and operators of the Rust library, and a pipeline that lowers them."""

from pathlib import Path

from tvm_ffi.libinfo import load_lib_module

from tvm_tirx import ir, tirx
from tvm_tirx.ir._expose_api import _init_exposed_op_api, _init_exposed_type_api
from tvm_tirx.script.ir_builder.base import auto_emit_void_call

# begin load
_init_exposed_op_api("my_ext", __name__, wrapper=auto_emit_void_call)
_init_exposed_type_api("my_ext", __name__)
_LIB = load_lib_module("my-ext", "my_ext", extra_lib_paths=[Path(__file__).parent])
# end load


def lower_complex() -> ir.transform.Pass:
    """The pass that lowers complex values to scalar arithmetic."""
    return _LIB.lower_complex()


# begin pipeline
def _pipeline(*, target=None):
    lower, finalize_host, finalize_device = tirx.get_tir_pipeline("default", target=target)
    return ir.transform.Sequential([lower_complex(), lower]), finalize_host, finalize_device


tirx.register_tir_pipeline("my_ext", _pipeline)
# end pipeline
