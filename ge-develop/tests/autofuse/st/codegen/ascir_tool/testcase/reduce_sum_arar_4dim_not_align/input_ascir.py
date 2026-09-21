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
from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions
NpuKernel0Graph = ascir.HintGraph('fused_graph_0_arar')
A0 = ascir.SizeExpr(2)
R0 = ascir.SizeExpr(1000)
A1 = ascir.SizeExpr(50)
R1 = ascir.SizeExpr(63)
buf8_a0 = NpuKernel0Graph.create_axis("buf8_z0", A0)
buf8_r0 = NpuKernel0Graph.create_axis("buf8_z1", R0)
buf8_a1 = NpuKernel0Graph.create_axis("buf8_z2", A1)
buf8_r1 = NpuKernel0Graph.create_axis("buf8_z3", R1)
arg2_1 = ascir.ops.Data('arg2_1', NpuKernel0Graph)
#arg2_1.attr.sched.exec_order = 0
arg2_1.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('load', NpuKernel0Graph)
#load.attr.sched.exec_order = 1
load.attr.sched.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
load.x = arg2_1.y
load.y.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
load.y.strides = [R0 * A1 * R1, A1 * R1, R1, ascir.SizeExpr(1)]
load.y.size = [A0, R0, A1, R1]
load.y.dtype = ascir.dtypes.float32
abs = ascir.ops.Abs('abs', NpuKernel0Graph)
#abs.attr.sched.exec_order = 2
abs.attr.sched.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
abs.x = load.y
abs.y.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
abs.y.strides = [R0 * A1 * R1, A1 * R1, R1, ascir.SizeExpr(1)]
abs.y.size = [A0, R0, A1, R1]
abs.y.dtype = ascir.dtypes.float32
sum = ascir.ops.Sum('sum', NpuKernel0Graph)
#sum.attr.sched.exec_order = 3
sum.attr.sched.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
sum.x = abs.y
sum.y.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
sum.y.strides = [A1, 0, 1, 0]
sum.y.size = [A0, 1, A1, 1]
sum.y.dtype = ascir.dtypes.float32
store8 = ascir.ops.Store('store8', NpuKernel0Graph)
#store8.attr.sched.exec_order = 4
store8.attr.sched.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
store8.x = sum.y
store8.y.axis = [buf8_a0, buf8_r0, buf8_a1, buf8_r1]
store8.y.strides = [A1, 0, 1, 0]
store8.y.size = [A0, 1, A1, 1]
store8.y.dtype = ascir.dtypes.float32
buf8 = ascir.ops.Output('buf8', NpuKernel0Graph)
#buf8.attr.sched.exec_order = 5
buf8.x = store8.y
buf8.y.dtype = ascir.dtypes.float32

fuser = Autofuser(AutofuserOptions())
fused_NpuKernel0Graph = fuser.schedule(NpuKernel0Graph)
tiling_def, host_impl, device_impl = fuser.codegen(fused_NpuKernel0Graph)
print("=================================")
print(device_impl)