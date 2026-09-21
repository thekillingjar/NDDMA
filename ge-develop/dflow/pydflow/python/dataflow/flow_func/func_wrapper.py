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

import functools
import inspect
import traceback
from dataflow.flow_func.func_register import FlowFuncRegister
from . import flow_func as fw


def proc_wrapper(func_list=None):
    def wrapper(func):
        if func_list is not None:
            module_name = inspect.getmodule(func).__name__
            clz_name = func.__qualname__.split(".")[0]
            FlowFuncRegister.register_flow_func(
                module_name, clz_name, func.__name__, func_list
            )

        @functools.wraps(func)
        def proc_func(self, run_context, input_flow_msgs):
            logger = fw.FlowFuncLogger()
            try:
                py_run_context = fw.MetaRunContext(run_context)
                inputs = [fw.FlowMsg(msg) for msg in input_flow_msgs]
                logger.info("trans to context and inputs success.")
                return func(self, py_run_context, inputs)
            except Exception as e:
                traceback.print_exc()
                logger.error("proc wrapper exception")
                return fw.FLOW_FUNC_FAILED

        return proc_func

    return wrapper


def init_wrapper():
    def wrapper(func):
        @functools.wraps(func)
        def proc_init(self, meta_params):
            logger = fw.FlowFuncLogger()
            try:
                py_meta_params = fw.PyMetaParams(meta_params)
                logger.info("trans to meta parmas wrapper success.")
                return func(self, py_meta_params)
            except Exception as e:
                traceback.print_exc()
                logger.error("init wrapper exception")
                return fw.FLOW_FUNC_FAILED

        return proc_init

    return wrapper
