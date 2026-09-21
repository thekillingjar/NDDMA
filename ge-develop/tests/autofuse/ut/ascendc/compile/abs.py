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

from pyautofuse import ascir, AutofuserOptions, Autofuser
import compile_adapter

graph = ascir.HintGraph('graph')
s0 = graph.create_size("s0")
z0 = graph.create_axis("z0", ascir.SizeExpr([s0]))

x = ascir.ops.Data("x")
y = ascir.ops.Output("y")

x.y.dtype = ascir.dtypes.float16

load = ascir.ops.Load("load")
load.attr.sched.axis = [z0]
load.x = x
load.y.axis = [z0]
load.y.size = [ascir.SizeExpr([s0])]
load.y.strides = [ascir.SizeExpr()]

abs = ascir.ops.Abs("abs")
abs.attr.sched.axis = [z0]
abs.x = load

store = ascir.ops.Store("store")
store.attr.sched.axis = [z0]
store.x = abs
store.y.axis = [z0]
store.y.size = [ascir.SizeExpr([s0])]
store.y.strides = [ascir.SizeExpr()]

y.x = store

graph.set_inputs([x])
graph.set_outputs([y])

options = AutofuserOptions(tiling_lib_path="", tiling_lib_codegen_symbol="")
fuser = Autofuser(options)
impl_graphs = fuser.schedule(graph)
op_proto, tiling_def, host_tiling, op_kernel = fuser.codegen(graph, impl_graphs)

#print(host_tiling)
compile_adapter.jit_compile('graph', tiling_def, host_tiling, op_kernel,['--soc_version=Ascend910B2', '--output_file=my_add'])
