#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Python code to construct AscGraph
from pyautofuse import ascir
from pyautofuse import Autofuser, AutofuserOptions
NpuKernel0Graph = ascir.HintGraph('fused_graph')
A0 = ascir.SizeExpr(50)
R0 = ascir.SizeExpr(128)
buf8_a0 = NpuKernel0Graph.create_axis("buf8_z0", A0)
buf8_r0 = NpuKernel0Graph.create_axis("buf8_z1", R0)
arg2_1 = ascir.ops.Data('arg2_1', NpuKernel0Graph)
arg2_1.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('load', NpuKernel0Graph)
load.attr.sched.axis = [buf8_a0, buf8_r0]
load.x = arg2_1.y
load.y.axis = [buf8_a0, buf8_r0]
load.y.strides = [R0, 1]
load.y.size = [A0, R0]
abs = ascir.ops.Abs('abs', NpuKernel0Graph)
abs.attr.sched.axis = [buf8_a0, buf8_r0]
abs.x = load.y
abs.y.axis = [buf8_a0, buf8_r0]
abs.y.strides = [R0, 1]
abs.y.size = [A0, R0]
t = ascir.ops.Max('t', NpuKernel0Graph)
t.attr.sched.axis = [buf8_a0, buf8_r0]
t.x = abs.y
t.y.axis = [buf8_a0, buf8_r0]
t.y.strides = [1, 0]
t.y.size = [A0, 1]
store8 = ascir.ops.Store('store8', NpuKernel0Graph)
store8.attr.sched.axis = [buf8_a0, buf8_r0]
store8.x = t.y
store8.y.axis = [buf8_a0, buf8_r0]
store8.y.strides = [1, 0]
store8.y.size = [A0, 1]
buf8 = ascir.ops.Output('buf8', NpuKernel0Graph)
buf8.x = store8.y
buf8.y.dtype = ascir.dtypes.float32

fuser = Autofuser(AutofuserOptions())
fused_NpuKernel0Graph = fuser.autofuse(NpuKernel0Graph)
op_proto, tiling_def, host_impl, device_impl = fuser.codegen(NpuKernel0Graph, fused_NpuKernel0Graph)
print("=================================")
print(device_impl)