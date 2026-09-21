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
import os
import shutil
import numpy as np
import dataflow as df
import dataflow.flow_func as ff
import dataflow.data_type as df_dt
import dataflow.dflow_wrapper as dwrapper

PARAM_INVALID = 145000
NOT_INIT = 145001
DATATYPE_INVALID = 145022
SHAPE_INVALID = 145021


class TestInit(unittest.TestCase):
    def test_with_ok_flow_options(self):
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        self.assertIn("ge.exec.deviceId", df.dataflow._global_options)
        self.assertEqual(df.dataflow._global_options["ge.exec.deviceId"], "0")
        self.assertIn("ge.graphRunMode", df.dataflow._global_options)
        self.assertEqual(df.dataflow._global_options["ge.graphRunMode"], "0")

    def test_with_error_type_flow_options(self):
        flow_options = 0
        try:
            df.init(flow_options)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)

    def test_with_error_key_flow_options(self):
        flow_options = {0: "xxx"}
        try:
            df.init(flow_options)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)

    def test_with_error_value_flow_options(self):
        flow_options = {"0": 0}
        try:
            df.init(flow_options)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)


class TestFinalize(unittest.TestCase):
    def test_ok(self):
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        self.assertIsNotNone(df.dataflow._global_options)
        df.finalize()
        self.assertIsNone(df.dataflow._global_options)


class TestGraphProcessPoint(unittest.TestCase):
    def test_create_ok(self):
        framework = df.Framework.TENSORFLOW
        graph_file = "xxxx.pb"
        graph_pp = df.GraphProcessPoint(framework, graph_file)
        self.assertEqual(graph_pp.framework, framework)
        self.assertEqual(graph_pp.graph_file, graph_file)
        self.assertEqual(graph_pp.load_params, {})
        self.assertEqual(graph_pp.compile_config_path, "")
        self.assertEqual(graph_pp.name, "GraphProcessPoint")
        graph_pp1 = df.GraphProcessPoint(framework, graph_file)
        self.assertEqual(graph_pp1.name, "GraphProcessPoint_1")
        graph_pp2 = df.GraphProcessPoint(framework, graph_file, name="graphpp")
        self.assertEqual(graph_pp2.name, "graphpp")
        graph_pp3 = df.GraphProcessPoint(framework, graph_file, name="graphpp")
        self.assertEqual(graph_pp3.name, "graphpp_1")
        graph_pp3_fnode = graph_pp3.fnode(
            input_num=2, output_num=1, name="test_graphpp_fnode"
        )
        self.assertEqual(graph_pp3_fnode.input_num, 2)
        self.assertEqual(graph_pp3_fnode.output_num, 1)

    def test_create_with_error_framework(self):
        framework = 0
        graph_file = "xxxx.pb"
        try:
            df.GraphProcessPoint(framework, graph_file)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)


class TestFlowGraphProcessPoint(unittest.TestCase):
    def test_create_ok(self):
        data = df.FlowData(name="test_flow_graph_data")
        data1 = df.FlowData(name="test_flow_graph_data1")
        flow_node = df.FlowNode(2, 1, "test_flow_graph_node")
        output = flow_node(data, data1)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output])
        flow_graph_pp = df.FlowGraphProcessPoint(graph)
        self.assertEqual(flow_graph_pp.compile_config_path, "")
        self.assertEqual(flow_graph_pp.name, "FlowGraphProcessPoint")
        flow_graph_pp2 = df.FlowGraphProcessPoint(graph, name="flowgraphpp")
        self.assertEqual(flow_graph_pp2.name, "flowgraphpp")
        flow_graph_pp3 = df.FlowGraphProcessPoint(graph, name="flowgraphpp")
        self.assertEqual(flow_graph_pp3.name, "flowgraphpp_1")
        df.finalize()


class AssignFunc:
    @ff.init_wrapper()
    def init_flow_func(self, meta_params):
        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i0,i1,o0")
    def assign(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS


class TestFuncProcessPoint(unittest.TestCase):
    def test_create_with_compile_config_path(self):
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path)
        self.assertEqual(func_pp.compile_config_path, compile_config_path)
        self.assertEqual(func_pp.name, "FuncProcessPoint")
        func_pp1 = df.FuncProcessPoint(compile_config_path)
        self.assertEqual(func_pp1.name, "FuncProcessPoint_1")
        func_pp2 = df.FuncProcessPoint(compile_config_path, "funcpp")
        self.assertEqual(func_pp2.name, "funcpp")
        func_pp3 = df.FuncProcessPoint(compile_config_path, "funcpp")
        self.assertEqual(func_pp3.name, "funcpp_1")

    def test_create_with_error_compile_config_path(self):
        self.assertRaises(TypeError, df.FuncProcessPoint, 0)

    def test_create_with_error_name(self):
        compile_config_path = "./func.json"
        self.assertRaises(TypeError, df.FuncProcessPoint, compile_config_path, 0)

    def test_create_with_py_func(self):
        ws_dir = "./py_assign_ws"
        func_pp = df.FuncProcessPoint(py_func=AssignFunc, workspace_dir=ws_dir)
        self.assertTrue(os.path.exists(ws_dir))
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "CMakeLists.txt")))
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "func_assign_func.json")))
        self.assertEqual(
            os.path.abspath(func_pp.compile_config_path),
            os.path.abspath(os.path.join(ws_dir, "func_assign_func.json")),
        )
        shutil.rmtree(ws_dir)

    def test_set_init_param_with_error_attr_name(self):
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "funcxxx")
        self.assertRaises(TypeError, func_pp.set_init_param, 0, 0)

    def test_set_init_param_with_error_attr_value(self):
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "funcxxx")
        self.assertRaises(TypeError, func_pp.set_init_param, "attr_name", func_pp)

    def test_set_init_param_ok(self):
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "funcxxx")
        self.assertIsNone(func_pp.set_init_param("attr_name", df.DT_UINT32))
        self.assertIsNone(func_pp.set_init_param("attr_name", "attr_value"))
        self.assertIsNone(func_pp.set_init_param("attr_name", ["aaaaaa", "bbbbbb"]))
        self.assertIsNone(func_pp.set_init_param("attr_name", 1844674407370955161))
        self.assertIsNone(func_pp.set_init_param("attr_name", 1))
        self.assertIsNone(func_pp.set_init_param("attr_name", 1.0))
        self.assertIsNone(func_pp.set_init_param("attr_name", [1.0]))
        self.assertIsNone(func_pp.set_init_param("attr_name", [df.DT_UINT32]))
        self.assertIsNone(func_pp.set_init_param("attr_name", []))
        self.assertIsNone(func_pp.set_init_param("attr_name", [0]))
        self.assertIsNone(func_pp.set_init_param("attr_name", [[0]]))
        self.assertIsNone(func_pp.set_init_param("attr_name", True))
        self.assertIsNone(func_pp.set_init_param("attr_name", [True]))

    def test_add_invoked_closure(self):
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "funcxxx")
        framework = df.Framework.TENSORFLOW
        graph_file = "xxxx.pb"
        graph_pp = df.GraphProcessPoint(framework, graph_file, name="graphxxx")
        self.assertIsNone(func_pp.add_invoked_closure("graph_key", graph_pp))


class TestFlowData(unittest.TestCase):
    def test_create_flow_data_ok(self):
        data = df.FlowData()
        self.assertEqual(data.name, "FlowData")
        tensor_desc = df.TensorDesc(dtype=df.DT_UINT32, shape=[1, 2])
        data1 = df.FlowData(schema=tensor_desc)
        self.assertEqual(data1._schema, tensor_desc)
        self.assertEqual(data1.name, "FlowData_1")
        data2 = df.FlowData(schema=tensor_desc, name="data")
        self.assertEqual(data2._schema, tensor_desc)
        self.assertEqual(data2.name, "data")
        data3 = df.FlowData(name="data")
        self.assertEqual(data3.name, "data_1")
        try:
            df.FlowData(df.TensorDesc(df.DT_INT32, [1]))
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)

    def test_create_flow_data_with_invalid_params(self):
        try:
            df.FlowData("xxx", "xxx")
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        try:
            df.FlowData(df.Tensor, "xxx")
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)


class TestFlowInfo(unittest.TestCase):
    def test_set_get(self):
        flow_info = df.FlowInfo()
        self.assertEqual(flow_info.start_time, 0)
        self.assertEqual(flow_info.end_time, 0)
        self.assertEqual(flow_info.flow_flags, 0)
        self.assertEqual(flow_info.transaction_id, 0)
        # set value by setter
        flow_info.start_time = 1
        flow_info.end_time = 2
        flow_info.flow_flags = 3
        flow_info.transaction_id = 100
        self.assertEqual(flow_info.start_time, 1)
        self.assertEqual(flow_info.end_time, 2)
        self.assertEqual(flow_info.flow_flags, 3)
        self.assertEqual(flow_info.transaction_id, 100)


class TestFlowNode(unittest.TestCase):
    def test_create_flow_node_ok(self):
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num)
        self.assertEqual(flow_node.input_num, input_num)
        self.assertEqual(flow_node.output_num, output_num)
        self.assertEqual(flow_node.name, "FlowNode")
        flow_node1 = df.FlowNode(input_num, output_num)
        self.assertEqual(flow_node1.name, "FlowNode_1")
        flow_node2 = df.FlowNode(input_num, output_num, "flownodexxx")
        self.assertEqual(flow_node2.name, "flownodexxx")
        flow_node3 = df.FlowNode(input_num, output_num, "flownodexxx")
        self.assertEqual(flow_node3.name, "flownodexxx_1")

    def test_create_flow_node_with_invalid_params(self):
        self.assertRaises(TypeError, df.FlowNode, "xxx", "xxx", "xxx")

    def test_add_process_point(self):
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "add_process_point")
        self.assertIsNone(flow_node.add_process_point(func_pp))
        framework = df.Framework.TENSORFLOW
        graph_file = "xxxx.pb"
        graph_pp = df.GraphProcessPoint(framework, graph_file, name="add_process_point")
        self.assertIsNone(flow_node.add_process_point(graph_pp))
        try:
            flow_node.add_process_point("xxx")
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)

    def test_map_input(self):
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "add_process_point")
        self.assertIsNone(flow_node.map_input(0, func_pp, 0))
        try:
            flow_node.map_input(2, func_pp, 0)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        try:
            flow_node.map_input(0, "xxx", 0)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)

    def test_map_input_with_attr(self):
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "add_process_point")
        time_batch = df.TimeBatch()
        self.assertIsNone(flow_node.map_input(0, func_pp, 0, [time_batch]))
        count_batch = df.CountBatch()
        self.assertIsNone(flow_node.map_input(0, func_pp, 0, [count_batch]))
        try:
            flow_node.map_input(0, func_pp, 0, [0])
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        self.assertIsNone(flow_node.map_input(0, func_pp, 0, [time_batch, count_batch]))

    def test_map_output(self):
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        compile_config_path = "./func.json"
        func_pp = df.FuncProcessPoint(compile_config_path, "add_process_point")
        self.assertIsNone(flow_node.map_output(0, func_pp, 0))
        try:
            flow_node.map_output(2, func_pp, 0)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        try:
            flow_node.map_output(0, "xxx", 0)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)

    def test_set_attr(self):
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        self.assertIsNone(flow_node.set_attr("_flow_attr", True))
        self.assertIsNone(flow_node.set_attr("_flow_attr_depth", 108))
        self.assertIsNone(flow_node.set_attr("_flow_attr_enqueue_policy", "FIFO"))
        self.assertRaises(TypeError, flow_node.set_attr, "name", [0])
        self.assertRaises(TypeError, flow_node.set_attr, 0, 0)

    def test_call(self):
        flow_data = df.FlowData(name="test_call")
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        self.assertRaises(df.DfException, flow_node, flow_data)
        self.assertRaises(df.DfException, flow_node, flow_data, "xxx")
        output0 = flow_node(flow_data, flow_data)
        try:
            flow_node(flow_data, output0)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        flow_node1 = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        flow_node1.set_balance_scatter()
        flow_node1.set_balance_gather()
        output1 = flow_node1(flow_data, output0)
        self.assertEqual(output1.node, flow_node1)
        self.assertEqual(output1.index, 0)
        flow_node2 = df.FlowNode(input_num, output_num, "flownodexxxxxx")
        output2 = flow_node2(output1, output1)
        self.assertEqual(output2.node, flow_node2)
        self.assertEqual(output2.index, 0)
        flow_node3 = df.FlowNode(input_num, 2, "flownodexxxxxx")
        output3 = flow_node3(output2, output2)
        self.assertEqual(len(output3), 2)
        self.assertEqual(output3[0].node, flow_node3)
        self.assertEqual(output3[0].index, 0)
        self.assertEqual(output3[1].node, flow_node3)
        self.assertEqual(output3[1].index, 1)
        flow_node4 = df.FlowNode(input_num, 0, "flownodexxxxxx")
        output4 = flow_node4(output3[0], output3[1])
        self.assertIsNone(output4)

    def test_build_flow_node(self):
        flow_data = df.FlowData(name="test_build_flow_node")
        input_num = 2
        output_num = 1
        flow_node = df.FlowNode(input_num, output_num, "flow_node_test_build")
        self.assertRaises(
            df.DfException,
            flow_node._build_flow_node,
            flow_data,
            "xxx",
            input_indexes=[0, 1],
            output_indexes=[0],
        )
        output0 = flow_node._build_flow_node(
            flow_data, flow_data, input_indexes=[0, 1], output_indexes=[0]
        )
        try:
            flow_node(flow_data, output0)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)


class TestFlowGraph(unittest.TestCase):
    def test_creat_flow_graph(self):
        data = df.FlowData(name="test_flow_graph_data")
        flow_node = df.FlowNode(2, 1, "test_flow_graph_node")
        output = flow_node(data, data)
        try:
            df.FlowGraph([output])
        except df.DfException as e:
            self.assertEqual(e.error_code, NOT_INIT)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output])
        self.assertRaises(df.DfException, df.FlowGraph, [1])
        df.finalize()

    def test_set_get_user_data(self):
        flow_info = df.FlowInfo()
        empty_array = bytearray()
        self.assertRaises(df.DfException, flow_info.set_user_data, empty_array)
        user_data_str = "UserData123"
        size = len(user_data_str)
        user_data_array = bytearray(user_data_str, "utf-8")
        self.assertRaises(df.DfException, flow_info.set_user_data, user_data_array, 60)
        flow_info.set_user_data(user_data_array)
        fetch_user_data = flow_info.get_user_data(size)
        name = fetch_user_data.decode("utf-8")
        self.assertEqual(name, user_data_str)

    def test_feed_data(self):
        data = df.FlowData(name="test_flow_graph_data")
        data1 = df.FlowData(name="test_flow_graph_data1")
        data2 = df.FlowData(name="test_flow_graph_data2")
        flow_node = df.FlowNode(2, 1, "test_flow_graph_node")
        output = flow_node(data, data1)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output])
        flow_info = df.FlowInfo()
        flow_info.flow_flags = (
                df.FlowFlag.DATA_FLOW_FLAG_EOS | df.FlowFlag.DATA_FLOW_FLAG_SEG
        )
        user_data_str = "UserData123"
        user_data_array = bytearray(user_data_str, "utf-8")
        flow_info.set_user_data(user_data_array)
        ret = graph.feed_data({data: 1, data1: 1}, flow_info)
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data: [1], data1: [1]})
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data: np.array([1]), data1: [1]})
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data: df.Tensor([1]), data1: [1]})
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data: df.Tensor([1])})
        self.assertEqual(ret, PARAM_INVALID)
        ret = graph.feed_data({data: df.Tensor([1])}, partial_inputs=True)
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data1: df.Tensor([1])}, partial_inputs=True)
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data2: df.Tensor([1])}, partial_inputs=True)
        self.assertEqual(ret, PARAM_INVALID)
        ret = graph.feed_data(
            {data: df.Tensor([1], tensor_desc=df.TensorDesc(df.DT_FLOAT, [1]))},
            partial_inputs=True,
        )
        self.assertEqual(ret, 0)
        ret = graph.feed_data({data: [184467440737095516150]}, partial_inputs=True)
        self.assertEqual(ret, DATATYPE_INVALID)
        ret = graph.feed_data({data: {"1": 1}}, partial_inputs=True)
        self.assertEqual(ret, DATATYPE_INVALID)
        ret = graph.feed_data({}, flow_info, partial_inputs=True)
        self.assertEqual(ret, 0)
        ret = graph.feed_data(
            {data: df.Tensor([1]), data1: df.Tensor(["nice"])}, flow_info
        )
        self.assertEqual(ret, 0)
        df.finalize()

    def test_feed_data_failed_invalid_flow_info(self):
        data = df.FlowData(name="test_flow_graph_data")
        flow_node = df.FlowNode(2, 1, "test_flow_graph_node")
        output = flow_node(data, data)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output])
        flow_info = df.FlowInfo()
        flow_info.flow_flags = (
                df.FlowFlag.DATA_FLOW_FLAG_EOS | df.FlowFlag.DATA_FLOW_FLAG_SEG
        )
        user_data_str = "UserData123"
        user_data_array = bytearray(user_data_str, "utf-8")
        flow_info.set_user_data(user_data_array)
        ret = graph.feed_data({data: 1}, "aaa")
        self.assertEqual(ret, PARAM_INVALID)

    def test_fetch_data(self):
        data = df.FlowData(name="test_flow_graph_data")
        flow_node = df.FlowNode(2, 2, "test_flow_graph_node")
        output0, output1 = flow_node(data, data)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output0, output1])
        ret = graph.feed_data({data: 1})
        self.assertEqual(ret, 0)
        output = graph.fetch_data()
        output = graph.fetch_data([0])
        output = graph.fetch_data([1])
        output = graph.fetch_data([0, 1])
        output = graph.fetch_data([1, 0])
        self.assertEqual(len(output), 3)
        self.assertEqual(output[2], 0)
        output = graph.fetch_data([2])
        self.assertEqual(len(output), 3)
        self.assertEqual(output[2], PARAM_INVALID)
        self.assertRaises(TypeError, graph.fetch_data, ["2"])
        df.finalize()

    def test_set_contains_n_mapping_node(self):
        data = df.FlowData(name="test_set_contains_n_mapping_node")
        flow_node = df.FlowNode(2, 2, "test_flow_graph_node")
        output0, output1 = flow_node(data, data)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output0, output1])
        graph.set_contains_n_mapping_node(True)
        ret = graph.feed_data({data: 1})
        self.assertEqual(ret, 0)
        self.assertRaises(df.DfException, graph.set_contains_n_mapping_node, True)
        df.finalize()

    def test_set_exception_catch(self):
        data = df.FlowData(name="test_set_exception_catch")
        flow_node = df.FlowNode(2, 2, "test_flow_graph_node")
        output0, output1 = flow_node(data, data)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output0, output1])
        graph.set_exception_catch(True)
        ret = graph.feed_data({data: 1})
        self.assertEqual(ret, 0)
        self.assertRaises(df.DfException, graph.set_exception_catch, True)
        df.finalize()

    def test_set_inputs_align_attrs(self):
        data = df.FlowData(name="test_set_inputs_align_attrs")
        flow_node = df.FlowNode(2, 2, "test_flow_graph_node")
        output0, output1 = flow_node(data, data)
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        graph = df.FlowGraph([output0, output1], graphpp_builder_async=True)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, -1, 60 * 1000)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, 1025, 60 * 1000)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, 256, 601 * 1000)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, 256, 0)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, 256, -2)
        graph.set_inputs_align_attrs(256, -1)
        graph.set_inputs_align_attrs(256, 600 * 1000)
        graph.set_inputs_align_attrs(0, 600 * 1000)
        graph.set_inputs_align_attrs(1024, 600 * 1000, True)
        ret = graph.feed_data({data: 1})
        self.assertEqual(ret, 0)
        self.assertRaises(df.DfException, graph.set_inputs_align_attrs, 256, 600 * 1000)
        df.finalize()

    def test_output_mapping(self):
        flow_options = {"ge.exec.deviceId": "0"}
        df.init(flow_options)
        data = df.FlowData(name="test_data")
        node0 = df.FlowNode(2, 3, "test_flow_graph_node0")
        node1 = df.FlowNode(2, 3, "test_flow_graph_node1")
        out0, out1, out2 = node0(data, data)
        out3, out4, out5 = node1(data, data)
        # in GE graph node0:0,0,0,1,2 node1:0,2
        graph = df.FlowGraph([out2, out1, out0, out0, out5, out3, out0])
        self.assertEqual(graph._output_idx_mapping, [4, 3, 0, 1, 6, 5, 2])
        df.finalize()


class TestTensor(unittest.TestCase):
    def test_create_tensor(self):
        t = df.Tensor(1)
        t = df.Tensor(t)
        self.assertRaises(df.DfException, t.get_shape)
        self.assertRaises(df.DfException, t.get_data_type)
        self.assertRaises(df.DfException, t.get_data_size)
        self.assertRaises(df.DfException, t.get_element_cnt)
        self.assertRaises(df.DfException, t.reshape, [])
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
        t = df.Tensor([["aa", "aaaaaaaaaaaaaaa"], ["b", "555"]])
        self.assertEqual(t._tensor_desc._dtype, df_dt.DT_STRING)
        try:
            a = t.numpy()
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        t = df.Tensor(["nice"])
        self.assertEqual(t._tensor_desc._dtype, df_dt.DT_STRING)
        try:
            a = t.numpy(True)
        except df.DfException as e:
            self.assertEqual(e.error_code, PARAM_INVALID)
        # construct by dwrapper.TestTensor
        tmp_value = np.array(["new", "hello"], dtype=np.bytes_)
        ti = dwrapper.Tensor(tmp_value)
        t = df.Tensor(ti)
        # construct by df tensor
        df_t = df.Tensor(t)
        # construct by numpy
        dt_np = df.Tensor(tmp_value)
        self.assertEqual(dt_np._tensor_desc._dtype, df_dt.DT_STRING)
        tmp_value = np.array(["new", "hello"], dtype=np.bytes_)
        ti = dwrapper.Tensor(tmp_value)
        t = df.Tensor(ti)
        # construct by df tensor
        df_t = df.Tensor(t)
        # construct by numpy
        dt_np = df.Tensor(tmp_value)
        self.assertEqual(dt_np._tensor_desc._dtype, df_dt.DT_STRING)


class TestTensorDesc(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
