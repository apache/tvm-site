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

tvm_tirx.tirx.transform
=======================

.. automodule:: tvm_tirx.tirx.transform
   :no-members:

.. currentmodule:: tvm_tirx.tirx.transform

Classes
-------

.. autosummary::
   :toctree: generated/

   FunctionPass
   RemoveNoOpConfig
   UnrollLoopConfig

Functions
---------

.. autosummary::
   :toctree: generated/

   AnnotateEntryFunc
   Apply
   BF16ComputeLegalize
   BF16StorageLegalize
   BindTarget
   CommonSubexprElim
   ConvertSSA
   DecorateDeviceScope
   FP8ComputeLegalize
   FP8StorageLegalize
   Filter
   FlattenTensor
   ForceNarrowIndexToInt32
   HoistIf
   LowerIntrin
   LowerTIRx
   LowerTVMBuiltin
   LowerThreadAllreduce
   MakePackedAPI
   NarrowDataType
   PointerValueTypeRewrite
   RemoveNoOp
   RenormalizeSplitPattern
   SkipAssert
   SplitHostDevice
   StmtSimplify
   StorageRewrite
   TileDispatch
   UnrollLoop
   VectorizeLoop
   VerifyMemory
   function_pass
