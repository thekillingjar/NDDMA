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
import numpy as np
import dataflow as df
import dataflow.utils.log as df_log
import dataflow.dflow_wrapper as dwrapper

PARAM_INVALID = 145000
NOT_INIT = 145001
DATATYPE_INVALID = 145022
SHAPE_INVALID = 145021


class TestDataFlow(unittest.TestCase):
    def test_with_error_type_flow_options(self):
        flow_options = 0
        self.assertRaises(df.DfException, df.init, flow_options)
        try:
            df.init(flow_options)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
            print(e)

    def test_with_error_value_flow_options(self):
        flow_options = {"0": 0}
        try:
            df.init(flow_options)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
            print(e)

    def test_create_with_error_framework(self):
        framework = 0
        graph_file = "xxxx.pb"
        try:
            df.GraphProcessPoint(framework, graph_file)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
            print(e)

    def test_create_tensor_desc(self):
        tensor_desc = df.TensorDesc(dtype=df.DT_INT64, shape=[])
        self.assertEqual(tensor_desc._dtype, df.DT_INT64)
        self.assertEqual(tensor_desc._shape, [])
        tensor_desc = df.TensorDesc(dtype=df.DT_INT64, shape=[1, 2])
        self.assertEqual(tensor_desc._dtype, df.DT_INT64)
        self.assertEqual(tensor_desc._shape, [1, 2])
        tensor_desc = df.TensorDesc(dtype=df.DT_INT64, shape=(1, 2))
        self.assertEqual(tensor_desc._dtype, df.DT_INT64)
        self.assertEqual(tensor_desc._shape, [1, 2])
        try:
            df.TensorDesc("xxx", "xxx")
        except df.DfException as e:
            self.assertEqual(e.error_code, DATATYPE_INVALID)
        self.assertRaises(df.DfException, df.TensorDesc, "xxx", "xxx")
        self.assertRaises(df.DfException, df.TensorDesc, df.DT_INT64, "xxx")
        self.assertRaises(df.DfException, df.TensorDesc, df.DT_INT64, ["xxx"])
        print(tensor_desc)

    def test_create_tensor(self):
        t = df.Tensor(1)
        t = df.Tensor(t)
        self.assertRaises(df.DfException, t.get_shape)
        self.assertRaises(df.DfException, t.get_data_type)
        self.assertRaises(df.DfException, t.get_data_size)
        self.assertRaises(df.DfException, t.get_element_cnt)
        tensor_desc = df.TensorDesc(dtype=df.DT_INT64, shape=[])
        self.assertEqual(t._tensor_desc, tensor_desc)
        t = df.Tensor(t, tensor_desc=tensor_desc)
        self.assertEqual(t._tensor_desc, tensor_desc)
        tensor_desc = df.TensorDesc(dtype=df.DT_INT64, shape=[1])
        try:
            df.Tensor(t, tensor_desc=tensor_desc)
        except df.DfException as e:
            self.assertEqual(e.error_code, SHAPE_INVALID)
        tensor_desc = df.TensorDesc(dtype=df.DT_FLOAT16, shape=[])
        t = df.Tensor(t, tensor_desc=tensor_desc)
        self.assertEqual(t._tensor_desc, tensor_desc)
        print(t)
        a = np.array([[1, 0, 2, 4]], dtype=np.int32)
        strided_array = np.lib.stride_tricks.as_strided(
            a, shape=(1, 2), strides=(16, 8)
        )
        self.assertRaises(df.DfException, df.Tensor, strided_array)
        try:
            df.Tensor(strided_array)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        try:
            df.TensorDesc(dtype=df.DT_INT64, shape=1)
        except df.DfException as e:
            self.assertEqual(e.error_code, SHAPE_INVALID)
            print(e, flush=True)
        try:
            df.TensorDesc(dtype=df.DT_INT64, shape=["a"])
        except df.DfException as e:
            self.assertEqual(e.error_code, SHAPE_INVALID)
            print(e, flush=True)

    def test_create_string_tensor(self):
        # construct by single string
        t = df.Tensor("test_string")
        try:
            a = t.numpy(True)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        # construct by dwrapper.TestTensor
        tmp_value = np.array(["new", "hello"], dtype=np.str_)
        ti = dwrapper.Tensor(tmp_value)
        t = df.Tensor(ti)
        t.numpy(True)
        # construct by df tensor
        df_t = df.Tensor(t)
        t.numpy(True)
        # construct by numpy
        dt_np = df.Tensor(tmp_value)
        t.numpy(True)

    def test_create_flow_data_with_invalid_params(self):
        try:
            df.FlowData("xxx", "xxx")
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
            print(e.__repr__())
        try:
            df.FlowData(df.Tensor, "xxx")
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
            print(e.__repr__())

    def test_dataflow(self):
        df_log.set_log_level(df_log.INFO)
        data = df.FlowData(
            name="test_flow_graph_data", schema=df.TensorDesc(df.DT_FLOAT16, [])
        )
        data1 = df.FlowData(
            name="test_flow_graph_data1", schema=df.TensorDesc(df.DT_FLOAT16, [])
        )
        flow_node = df.FlowNode(2, 1, "test_flow_graph_node")
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "funcxxx")
        framework = df.Framework.TENSORFLOW
        graph_file = "xxxx.pb"
        graph_pp = df.GraphProcessPoint(framework, graph_file)
        func_pp.add_invoked_closure("graph_key", graph_pp)

        data2 = df.FlowData(name="test_flow_graph_data2")
        data3 = df.FlowData(name="test_flow_graph_data3")
        flow_node1 = df.FlowNode(2, 1, "test_flow_graph_node1")
        output1 = flow_node(data2, data3)
        self.assertRaises(df.DfException, df.FlowGraph, [output1])
        try:
            df.FlowGraph([output1])
            print(e)
            print(e.__repr__())
        except df.DfException as e:
            self.assertEqual(e.error_code, NOT_INIT)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        invoke_graph = df.FlowGraph([output1])
        flow_graph_pp = df.FlowGraphProcessPoint(invoke_graph)
        func_pp.add_invoked_closure("flow_graph_key", flow_graph_pp)

        flow_node.add_process_point(func_pp)
        time_batch = df.TimeBatch()
        flow_node.map_input(0, func_pp, 0, [time_batch])
        flow_node.map_output(0, func_pp, 0)
        flow_node.set_balance_scatter()
        flow_node.set_balance_gather()
        output = flow_node(data, data)

        graph = df.FlowGraph([output], graphpp_builder_async=True)
        graph.set_contains_n_mapping_node(True)
        graph.set_inputs_align_attrs(256, 600 * 1000, True)
        graph.set_exception_catch(True)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, -1, 60 * 1000)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, 256, -2)

        flow_info = df.FlowInfo()
        flow_info.flow_flags = df.FlowFlag.DATA_FLOW_FLAG_EOS

        empty_array = bytearray()
        self.assertRaises(df.DfException, flow_info.set_user_data, empty_array)
        user_data_str = "UserData123"
        user_data_array = bytearray(user_data_str, "utf-8")
        self.assertRaises(df.DfException, flow_info.set_user_data, user_data_array, 60)
        flow_info.set_user_data(user_data_array)
        fetch_user_data = flow_info.get_user_data(len(user_data_str))
        name = fetch_user_data.decode("utf-8")
        self.assertEqual(name, user_data_str)
        ret = graph.feed_data({data: 1})
        self.assertEqual(ret, 0)
        t = df.Tensor(1)
        ret = graph.feed_data({data: t})
        self.assertEqual(ret, 0)
        t2 = df.Tensor(t, tensor_desc=df.TensorDesc(df.DT_FLOAT16, []))
        ret = graph.feed_data({data: t2})
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data: np.array(1, dtype=np.float16)})
        self.assertEqual(ret, 0)
        ret = graph.feed_data(
            {data: df.Tensor(1, tensor_desc=df.TensorDesc(df.DT_FLOAT16, []))}
        )
        self.assertEqual(ret, 0)
        self.assertRaises(df.DfException, df.TensorDesc, "", [])
        try:
            df.TensorDesc("", [])
        except df.DfException as e:
            self.assertEqual(e.error_code, DATATYPE_INVALID)
        ret = graph.feed_data({data: ["xxx"]})
        self.assertEqual(ret, SHAPE_INVALID)
        ret = graph.feed_data({}, flow_info)
        self.assertEqual(ret, 0)
        ret = graph.feed_data({}, 1)
        self.assertEqual(ret, PARAM_INVALID)

        ret = graph.feed_data({data1: 1})
        self.assertEqual(ret, PARAM_INVALID)

        ret = graph.feed_data({data1: 1}, partial_inputs=True)
        self.assertEqual(ret, PARAM_INVALID)

        flow_info.flow_flags = 0
        ret = graph.feed_data({}, flow_info)
        self.assertEqual(ret, PARAM_INVALID)

        output = graph.fetch_data([2])
        self.assertEqual(output[2], PARAM_INVALID)

        self.assertRaises(df.DfException, graph.set_contains_n_mapping_node, True)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, 256, 600 * 1000)
        self.assertRaises(df.DfException, graph.set_exception_catch, True)
        output = graph.fetch_data()
        df.finalize()
