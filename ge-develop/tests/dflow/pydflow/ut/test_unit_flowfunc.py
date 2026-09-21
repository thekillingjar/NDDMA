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

import queue
import unittest
import numpy as np
import dataflow.flow_func.flow_func as ff
import dataflow.flow_func.flowfunc_wrapper as fw
import dataflow.data_type as dt
import dataflow as df
import dataflow.utils.utils as utils


class FlowFuncLoggerTest(unittest.TestCase):
    def test_log_is_enable(self):
        self.assertEqual(ff.logger.is_log_enable(0, 0), False)
        self.assertEqual(ff.logger.is_log_enable(ff.DEBUG_LOG, 0), False)
        self.assertEqual(ff.logger.is_log_enable(ff.DEBUG_LOG, ff.DEBUG), True)

    def test_get_log_header(self):
        header = ff.logger.get_log_header()
        self.assertEqual(header[-5:-1], "USER")

    def test_print_log(self):
        ff.logger.debug("debug log :%s %d %f", "test", 100, 0.1)
        ff.logger.error("error log :%s %d %f", "test", 100, 0.1)
        ff.logger.info("info log :%s %d %f", "test", 100, 0.1)
        ff.logger.warn("warn log :%s %d %f", "test", 100, 0.1)
        ff.logger.run_error("run_error log :%s %d %f", "test", 100, 0.1)
        ff.logger.run_info("run_info log :%s %d %f", "test", 100, 0.1)
        ff.logger.debug("debug log :%s %d %f %f", "test", 100, 0.1)
        ff.logger.error("error log :%s %d %f %f", "test", 100, 0.1)
        ff.logger.info("info log :%s %d %f %f", "test", 100, 0.1)
        ff.logger.warn("warn log :%s %d %f %f", "test", 100, 0.1)
        ff.logger.run_error("run_error log :%s %d %f %f", "test", 100, 0.1)
        ff.logger.run_info("run_info log :%s %d %f %f", "test", 100, 0.1)


class FlowMsgTest(unittest.TestCase):
    def test_msg_get_tensor_stub(self):
        orig_msg = fw.FlowMsg()
        msg = ff.FlowMsg(orig_msg)
        self.assertEqual(msg.get_msg_type(), fw.MSG_TYPE_TENSOR_DATA)
        tensor = msg.get_tensor()
        self.assertEqual(tensor.get_shape(), [2, 5])
        self.assertEqual(tensor.get_data_type(), np.uint32)
        self.assertEqual(tensor.get_data_size(), 40)
        self.assertEqual(tensor.get_element_cnt(), 10)
        self.assertEqual(tensor.reshape([5, 2]), ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(tensor.get_shape(), [5, 2])
        self.assertNotEqual(tensor.reshape([5, 2, 5]), ff.FLOW_FUNC_SUCCESS)

    def test_msg_get_and_set_func(self):
        orig_msg = fw.FlowMsg()
        msg = ff.FlowMsg(orig_msg)
        msg.get_transaction_id()
        msg.set_ret_code(100)
        self.assertEqual(msg.get_ret_code(), 100)
        msg.set_start_time(10000)
        self.assertEqual(msg.get_start_time(), 10000)
        msg.set_end_time(20000)
        self.assertEqual(msg.get_end_time(), 20000)
        msg.set_flow_flags(3)
        self.assertEqual(msg.get_flow_flags(), 3)
        msg.set_route_label(3)
        print(msg)


class PyMetaParamsTest(unittest.TestCase):
    def test_get_attr_from_stub(self):
        orig_params = fw.MetaParams()
        params = ff.PyMetaParams(orig_params)
        print(params)
        self.assertEqual(params.get_name(), "stubName")
        self.assertEqual(params.get_work_path(), "stubWorkPath")
        self.assertEqual(params.get_input_num(), 2)
        self.assertEqual(params.get_output_num(), 1)
        self.assertEqual(params.get_running_device_id(), 0)
        self.assertEqual(params.get_running_instance_id(), 0)
        self.assertEqual(params.get_running_instance_num(), 1)
        int_attr = params.get_attr_int("AttrStub")
        self.assertEqual(int_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(int_attr[1], 0)
        int_attr = params.get_attr_int_list("AttrStub")
        self.assertEqual(int_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(int_attr[1], [0])
        int_attr = params.get_attr_int_list_list("AttrStub")
        self.assertEqual(int_attr[0], ff.FLOW_FUNC_SUCCESS)
        bool_attr = params.get_attr_bool("Not")
        self.assertEqual(bool_attr[0], ff.FLOW_FUNC_ERR_ATTR_NOT_EXITS)
        bool_attr = params.get_attr_bool("AttrStub")
        self.assertEqual(bool_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(bool_attr[1], True)
        bool_attr = params.get_attr_bool_list("AttrStub")
        self.assertEqual(bool_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(bool_attr[1], [True])
        float_attr = params.get_attr_float("AttrStub")
        self.assertEqual(float_attr[0], ff.FLOW_FUNC_SUCCESS)
        float_attr = params.get_attr_float_list("AttrStub")
        self.assertEqual(float_attr[0], ff.FLOW_FUNC_SUCCESS)
        dtype_attr = params.get_attr_tensor_dtype("AttrStub")
        self.assertEqual(dtype_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(dtype_attr[1], np.int16)
        dtype_attr = params.get_attr_tensor_dtype_list("AttrStub")
        self.assertEqual(dtype_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(dtype_attr[1], [np.int16])
        str_attr = params.get_attr_str("AttrStub")
        self.assertEqual(str_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(str_attr[1], "stub")
        str_attr = params.get_attr_str_list("AttrStub")
        self.assertEqual(str_attr[0], ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(str_attr[1], ["stub"])


class MetaRunContextTest(unittest.TestCase):
    def test_alloc_tensor(self):
        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        tensor1 = msg1.get_tensor()
        self.assertEqual(tensor1.get_shape(), [2, 3])
        self.assertEqual(tensor1.get_data_type(), np.float16)
        msg2 = context.alloc_tensor_msg([1, 3], np.int32)
        tensor1 = msg2.get_tensor()
        self.assertEqual(tensor1.get_shape(), [1, 3])
        self.assertEqual(tensor1.get_data_type(), np.int32)
        self.assertRaises(TypeError, context.alloc_tensor_msg, [2, 3], 1)
        msg3 = context.alloc_empty_data_msg()
        print(context)

    def test_set_output(self):
        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        self.assertEqual(context.set_output(0, msg1), ff.FLOW_FUNC_SUCCESS)
        out = np.array([[1, 2]], dtype=np.int32)
        self.assertEqual(context.set_output(0, out), ff.FLOW_FUNC_SUCCESS)

    def test_set_output_with_balance_option(self):
        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        cfg = ff.BalanceConfig(3, 4, ff.AffinityPolicy.COL_AFFINITY)
        cfg.set_data_pos([(1, 2)])
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        self.assertEqual(context.set_output(0, msg1, cfg), ff.FLOW_FUNC_SUCCESS)
        out = np.array([[1, 2]], dtype=np.int32)
        self.assertEqual(context.set_output(0, out, cfg), ff.FLOW_FUNC_SUCCESS)

    def test_set_multi_outputs(self):
        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        cfg = ff.BalanceConfig(3, 4, ff.AffinityPolicy.COL_AFFINITY)
        cfg.set_data_pos([(1, 2)])
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        out = np.array([[1, 2]], dtype=np.int32)
        self.assertEqual(
            context.set_multi_outputs(0, [msg1, out], cfg), ff.FLOW_FUNC_SUCCESS
        )
        self.assertRaises(
            TypeError, context.set_multi_outputs, 0, [msg1, out, cfg], cfg
        )

    def test_run_flow_model(self):
        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        msg2 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        ret = context.run_flow_model("invoke_graph", [msg1, msg2], 1000)
        self.assertEqual(ret[0], ff.FLOW_FUNC_SUCCESS)

    def test_get_user_data(self):
        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        stub_str = "StubUserData"
        str_len = len(stub_str)
        user_data = context.get_user_data(len(stub_str) + 1, 0)
        name = user_data.decode("utf-8")
        ff.logger.info("user data %s", name)
        self.assertEqual(name[0:str_len], stub_str[0:str_len])

    def test_raise_and_get_exception(self):
        wrapper_context = fw.MetaRunContext()
        context = ff.MetaRunContext(wrapper_context)
        ret = context.get_exception()
        self.assertEqual(ret[0], False)
        context.raise_exception(-2, 200)
        ret = context.get_exception()
        self.assertEqual(ret[0], True)
        self.assertEqual(ret[1], -2)
        self.assertEqual(ret[2], 200)
        context.raise_exception(-1, 200)

    def test_flow_msg_queue(self):
        ff.logger.info("Start to test flow msg queue")
        orig_flow_msg_queue = fw.FlowMsgQueue()
        flow_msg_queue = ff.FlowMsgQueue(orig_flow_msg_queue)
        self.assertRaises(NotImplementedError, flow_msg_queue.task_done)
        self.assertRaises(NotImplementedError, flow_msg_queue.join)
        self.assertEqual(flow_msg_queue.qsize(), 10)
        self.assertFalse(flow_msg_queue.empty())
        self.assertFalse(flow_msg_queue.full())
        self.assertRaises(NotImplementedError, flow_msg_queue.put, 1)
        self.assertRaises(NotImplementedError, flow_msg_queue.put_nowait, 1)
        self.assertEqual(type(flow_msg_queue.get_nowait()), df.Tensor)
        self.assertEqual(type(flow_msg_queue.get()), df.Tensor)
        self.assertRaises(ValueError, flow_msg_queue.get, timeout=-1)
        self.assertEqual(type(flow_msg_queue.get(timeout=1)), df.Tensor)
        self.assertRaises(utils.DfAbortException, flow_msg_queue.get, timeout=2)
        self.assertRaises(queue.Empty, flow_msg_queue.get, timeout=3)
        self.assertRaises(utils.DfException, flow_msg_queue.get, timeout=4)
