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

import unittest
import os.path
import shutil
import queue
import numpy as np
import dataflow as df
import dataflow.flow_func as ff
import dataflow.data_type as dt
import dataflow.flow_func.flowfunc_wrapper as fw
import dataflow.utils.log as df_log
import dataflow.utils.utils as utils


class AssignFunc:
    @ff.init_wrapper()
    def init_flow_func(self, meta_params):
        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i0,i1,o0")
    def assign(self, run_context, input_flow_msgs):
        logger = ff.FlowFuncLogger()
        for msg in input_flow_msgs:
            if run_context.set_output(0, msg) != ff.FLOW_FUNC_SUCCESS:
                logger.error("set output failed")
            return ff.FLOW_FUNC_FAILED
        return ff.FLOW_FUNC_SUCCESS


class AssignFuncInputNotContinuous:
    @ff.proc_wrapper("i0,i2,o0")
    def assign(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS


class AssignFuncInputRepeat:
    @ff.proc_wrapper("i0,i2,o0")
    def assign(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i1,i2,o0")
    def assign1(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS


class AssignFuncOutputNotContinuous:
    @ff.proc_wrapper("i0,i1,o0,o2")
    def assign(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS


class TestDataflowFlowFunc(unittest.TestCase):

    def test_create_ws(self):
        ws_dir = "./py_assign_ws"
        func_pp = df.FuncProcessPoint(py_func=AssignFunc, workspace_dir=ws_dir)
        self.assertTrue(os.path.exists(ws_dir))
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "CMakeLists.txt")))
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "func_assign_func.json")))
        shutil.rmtree(ws_dir)

    def test_create_ws_input_not_continuous(self):
        ws_dir = "./py_assign_ws_invalid"
        self.assertRaises(
            ValueError,
            df.FuncProcessPoint,
            py_func=AssignFuncInputNotContinuous,
            workspace_dir=ws_dir,
        )
        shutil.rmtree(ws_dir)

    def test_create_ws_input_repeat(self):
        ws_dir = "./py_assign_ws_invalid"
        self.assertRaises(
            ValueError,
            df.FuncProcessPoint,
            py_func=AssignFuncInputRepeat,
            workspace_dir=ws_dir,
        )
        shutil.rmtree(ws_dir)

    def test_create_ws_output_not_continuous(self):
        ws_dir = "./py_assign_ws_invalid"
        self.assertRaises(
            ValueError,
            df.FuncProcessPoint,
            py_func=AssignFuncOutputNotContinuous,
            workspace_dir=ws_dir,
        )
        shutil.rmtree(ws_dir)


class TestFlowFunc(unittest.TestCase):
    def test_init_sample(self):
        orig_params = fw.MetaParams()
        params = ff.PyMetaParams(orig_params)
        self.assertEqual(params.get_name(), "stubName")
        self.assertEqual(params.get_work_path(), "stubWorkPath")
        self.assertEqual(params.get_input_num(), 2)
        self.assertEqual(params.get_output_num(), 1)
        self.assertEqual(params.get_running_device_id(), 0)
        self.assertEqual(params.get_running_instance_id(), 0)
        self.assertEqual(params.get_running_instance_num(), 1)

    def test_run_sample(self):
        orig_context = fw.MetaRunContext()
        ff.logger.info("Start flow func st test")
        context = ff.MetaRunContext(orig_context)
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        self.assertEqual(context.set_output(0, msg1), ff.FLOW_FUNC_SUCCESS)
        out = np.array([[1, 2]], dtype=np.int32)
        self.assertEqual(context.set_output(0, out), ff.FLOW_FUNC_SUCCESS)
        stub_str = "StubUserData"
        str_len = len(stub_str)
        user_data = context.get_user_data(len(stub_str) + 1, 0)
        name = user_data.decode("utf-8")
        ff.logger.info("user data %s", name)
        self.assertEqual(name[0:str_len], stub_str[0:str_len])

    def test_run_with_balance_options(self):
        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        msg1.get_tensor().reshape([3, 2])
        cfg = ff.BalanceConfig(3, 4, ff.AffinityPolicy.COL_AFFINITY)
        cfg.set_data_pos([(1, 2)])
        self.assertEqual(context.set_output(0, msg1, cfg), ff.FLOW_FUNC_SUCCESS)

        out = np.array([[1, 2]], dtype=np.int32)
        self.assertEqual(context.set_output(0, out, cfg), ff.FLOW_FUNC_SUCCESS)

        cfg.set_data_pos([(0, 0), (1, 2)])
        self.assertEqual(
            context.set_multi_outputs(0, [msg1, out], cfg), ff.FLOW_FUNC_SUCCESS
        )
        self.assertRaises(
            TypeError, context.set_multi_outputs, 0, [msg1, out, cfg], cfg
        )

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
