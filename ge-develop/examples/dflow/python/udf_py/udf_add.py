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

import dataflow.flow_func as ff
import dataflow.data_type as dt


class UserFunc1():
    def __init__(self):
        self.count = 0

    @ff.init_wrapper()
    def init_flow_func(self, meta_params):
        logger = ff.FlowFuncLogger()
        logger.info("Start init")
        name = meta_params.get_name()
        logger.info("Func name [%s]" % name)
        name_attr = meta_params.get_attr_int("name")
        logger.info("Get attr int [%d]", name_attr[1])
        input_num = meta_params.get_input_num()
        logger.info("Input num [%d]", input_num)

        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i0,i1,o0")
    def add1(self, run_context, input_flow_msgs):
        logger = ff.FlowFuncLogger()
        for msg in input_flow_msgs:
            logger.info("Get msg code [%d]", msg.get_ret_code())
            if msg.get_ret_code() != ff.FLOW_FUNC_SUCCESS:
                logger.error("Invalid inputs: return code is not success.")
                return ff.FLOW_FUNC_FAILED

            if msg.get_msg_type() != ff.MSG_TYPE_TENSOR_DATA:
                logger.error("invalid input: msg type should be tensor data.")
                return ff.FLOW_FUNC_FAILED

        if input_flow_msgs.__len__() != 2:
            logger.error("Inputs should be 2.")
            return ff.FLOW_FUNC_FAILED

        tensor1 = input_flow_msgs[0].get_tensor()
        tensor2 = input_flow_msgs[1].get_tensor()
        np1 = tensor1.numpy()
        np2 = tensor2.numpy()
        dtype1 = tensor1.get_data_type()
        dtype2 = tensor2.get_data_type()
        if dtype1 != dtype2:
            logger.error("Input type should be same.")
            return ff.FLOW_FUNC_FAILED
        logger.info("Input is same.")
        shape1 = tensor1.get_shape()
        shape2 = tensor2.get_shape()
        if shape1 != shape2:
            logger.error("Input shape should be same.")
            return ff.FLOW_FUNC_FAILED
        out = run_context.alloc_tensor_msg(shape1, dt.DT_INT32)
        data_size = out.get_tensor().get_data_size()
        logger.info("Output data size is [%d].", data_size)
        ele_cnt = out.get_tensor().get_element_cnt()
        logger.info("Element count is [%d].", ele_cnt)

        a = out.get_tensor().numpy()
        a[...] = np1 + np2
        logger.info("Prepare to set output.")

        if run_context.set_output(0, out) != ff.FLOW_FUNC_SUCCESS:
            logger.error("Set output failed.")
            return ff.FLOW_FUNC_FAILED
        self.count += 1
        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i2,i3,o1")
    def add2(self, run_context, input_flow_msgs):
        logger = ff.FlowFuncLogger()
        for msg in input_flow_msgs:
            logger.info("Get msg code [%d]", msg.get_ret_code())
            if msg.get_ret_code() != ff.FLOW_FUNC_SUCCESS:
                logger.error("Invalid inputs: return code is not success.")
                return ff.FLOW_FUNC_FAILED

            if msg.get_msg_type() != ff.MSG_TYPE_TENSOR_DATA:
                logger.error("invalid input: msg type should be tensor data.")
                return ff.FLOW_FUNC_FAILED

        if input_flow_msgs.__len__() != 2:
            logger.error("Inputs should be 2.")
            return ff.FLOW_FUNC_FAILED

        tensor1 = input_flow_msgs[0].get_tensor()
        tensor2 = input_flow_msgs[1].get_tensor()
        np1 = tensor1.numpy()
        np2 = tensor2.numpy()
        dtype1 = tensor1.get_data_type()
        dtype2 = tensor2.get_data_type()
        if dtype1 != dtype2:
            logger.error("Input type should be same.")
            return ff.FLOW_FUNC_FAILED
        logger.info("Input is same.")
        shape1 = tensor1.get_shape()
        shape2 = tensor2.get_shape()
        if shape1 != shape2:
            logger.error("Input shape should be same.")
            return ff.FLOW_FUNC_FAILED
        out = run_context.alloc_tensor_msg(shape1, dt.DT_INT32)
        data_size = out.get_tensor().get_data_size()
        logger.info("Output data size is [%d].", data_size)
        ele_cnt = out.get_tensor().get_element_cnt()
        logger.info("Element count is [%d].", ele_cnt)

        a = out.get_tensor().numpy()
        a[...] = np1 + 2 * np2
        logger.info("Prepare to set output.")
        if run_context.set_output(1, out) != ff.FLOW_FUNC_SUCCESS:
            logger.error("Set output failed.")
            return ff.FLOW_FUNC_FAILED
        self.count += 1
        return ff.FLOW_FUNC_SUCCESS
