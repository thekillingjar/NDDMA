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
from typing import Tuple
from pathlib import Path
import numpy as np
import cloudpickle
import dataflow as df
import dataflow.data_type as dt
from dataflow.pyflow import PyActorProcessPoint
import dataflow.flow_func.flowfunc_wrapper as fw
import dataflow.flow_func as ff
import dataflow.utils.utils as utils


@df.pyflow(
    resources={"memory": 100, "num_cpus": 1, "num_npus": 1}, visible_device_enable=True
)
class Foo:
    def __init__(self, name, val):
        self.name = name
        self.val = val

    @df.method()
    def sub_1(self, in0):
        return in0

    @df.method(num_returns=2)
    def add_1(self, in0):
        return in0, self.val


def hook_func():
    pass


@df.pyflow(env_hook_func=hook_func)
def some_func(in0, in1) -> Tuple[df.Tensor, df.Tensor, int]:
    return df.alloc_tensor([2, 3], dt.DT_FLOAT16), in1, 1


class SomeClass:
    def __init__(self, name, val):
        self.name = name
        self.val = val


@df.pyflow
class TestFooInvalid:
    def __init__(self, name, val):
        self.name = name
        self.val = val

    @df.method()
    def some_func(self, in0):
        return in0


@df.pyflow(stream_input="Queue")
def func_with_stream_input(queue0, queue1):
    in0 = queue0.get()
    in1 = queue1.get()
    return df.alloc_tensor([2, 3], dt.DT_FLOAT16)


@df.pyflow
class MyClass:
    @df.method(stream_input="Queue")
    def func_with_stream_input(self, queue0):
        in0 = queue0.get()
        return df.alloc_tensor([2, 3], dt.DT_FLOAT16)

    @df.method(choice_output=lambda e: e is not None)
    def normal_func(self, in0):
        return in0


@df.pyflow(choice_output=lambda e: e is not None)
def func_with_choice_output(in0, in1):
    return None, 1


@df.pyflow
class MyClassWithStreamOutput:
    @df.method()
    def func_with_stream_ouput(self, in0):
        yield 1


@df.pyflow
def func_with_stream_ouput(in0, in1):
    yield 1
    yield 2
    yield df.alloc_tensor([2, 3], dt.DT_FLOAT16)


@df.pyflow(num_returns=3)
def func_with_invalid_output_num(in0, in1):
    return in0, in1


class TestPyFlow(unittest.TestCase):
    def test_pyflow_success(self):
        df.init({"ge.exec.deviceId": "0"})

        df.msg_type_register.register(
            1026,
            SomeClass,
            lambda obj: cloudpickle.dumps(obj),
            lambda buffer: cloudpickle.loads(buffer),
        )

        data0 = df.FlowData(name="d0")
        data1 = df.FlowData(name="d1")
        data2 = df.FlowData(name="d2")
        foo = Foo.fnode("some_name", 1).set_alias("Foo")
        sub1_out = foo.sub_1(data0)
        add1_out0, add1_out1 = foo.add_1(data1)
        # input index is defined by func inner order, add_1/sub_1
        self.assertEqual(foo._input_anchors[0].name, "d1")
        self.assertEqual(foo._input_anchors[1].name, "d0")
        func_out0, func_out1, func_out2 = some_func.fnode().set_alias("some_func")(
            add1_out0, add1_out1
        )
        graph = df.FlowGraph([sub1_out, func_out0, func_out1, func_out2])
        self.assertEqual(Path("./some_func_fnode_ws").exists(), True)
        self.assertEqual(Path(foo.name + "_ws").exists(), True)

        df.utils.generate_deploy_template(graph, "./test_generate_deploy_info.json")
        self.assertEqual(Path("./test_generate_deploy_info.json").exists(), True)

        d0 = df.alloc_tensor([1, 2], np.int32)
        ret = graph.feed({data0: d0, data1: 2, data2: 3})
        self.assertEqual(ret, 0)

        output = graph.fetch()
        self.assertEqual(output[1], 0)

        ret = graph.feed({data0: SomeClass("test", 1)}, partial_inputs=True)
        self.assertEqual(ret, 0)

        output = graph.fetch([0])
        self.assertEqual(output[1], 0)

        df.finalize()

        os.system("rm -rf ./*_fnode_ws")
        os.remove("./test_generate_deploy_info.json")

    def test_pyflow_invalid(self):
        # can not invoke class __init__ and __call__ directly
        self.assertRaises(TypeError, TestFooInvalid, "test", 1)
        foo = TestFooInvalid.fnode("test", 1)
        # fnode input num and func input num not equal
        self.assertRaises(df.DfException, foo.some_func, df.FlowData(), df.FlowData())
        os.system("rm -rf ./*_fnode_ws")

    def test_pyflow_run(self):
        ret = some_func(fw.MetaRunContext(), [fw.FlowMsg(), fw.FlowMsg()])
        self.assertEqual(ret, 0)
        foo_node = Foo.fnode()
        self.assertEqual(Path(foo_node.name + "_ws/src_python/Foo.pkl").exists(), True)
        with open(foo_node.name + "_ws/src_python/Foo.pkl", "rb") as f:
            foo = cloudpickle.loads(f.read())
            ret = foo.sub_1(fw.MetaRunContext(), [fw.FlowMsg()])
            self.assertEqual(ret, 0)
        os.system("rm -rf ./*_fnode_ws")

    def test_init_not_called_directly(self):
        with self.assertRaises(TypeError) as cm:
            PyActorProcessPoint()
        self.assertEqual(
            str(cm.exception),
            "PyActorProcessPoint.__init__ should not be called. Please use the @df.pyflow decorator instead.",
        )

    def test_stream_input(self):
        df.init({"ge.exec.deviceId": "0"})

        data0 = df.FlowData()
        data1 = df.FlowData()
        fnode0 = MyClass.fnode()
        fnode1 = func_with_stream_input.fnode()
        fnode0_out_0 = fnode0.func_with_stream_input(data0)
        fnode0_out_1 = fnode0.normal_func(data1)
        fnode1_out = fnode1(fnode0_out_0, fnode0_out_1)

        graph = df.FlowGraph([fnode1_out])

        df.utils.generate_deploy_template(graph, "./test_generate_deploy_info.json")

        d0 = df.alloc_tensor([1, 2], np.int32)
        ret = graph.feed({data0: d0, data1: 2})
        self.assertEqual(ret, 0)

        output = graph.fetch()
        self.assertEqual(output[1], 0)
        df.finalize()

        os.system("rm -rf ./*_fnode_ws")
        os.remove("./test_generate_deploy_info.json")

    def test_pyflow_run_with_stream_input(self):
        ret = func_with_stream_input(
            fw.MetaRunContext(), [fw.FlowMsgQueue(), fw.FlowMsgQueue()]
        )
        self.assertEqual(ret, 0)
        fnode0 = MyClass.fnode()
        self.assertEqual(
            Path(fnode0.name + "_ws/src_python/MyClass.pkl").exists(), True
        )
        with open(fnode0.name + "_ws/src_python/MyClass.pkl", "rb") as f:
            my_class = cloudpickle.loads(f.read())
            ret = my_class.func_with_stream_input(
                fw.MetaRunContext(), [fw.FlowMsgQueue()]
            )
            self.assertEqual(ret, 0)
        os.system("rm -rf ./*_fnode_ws")

    def test_invalid_stream_input_type(self):
        def decorate_func():
            @df.pyflow(stream_input="Iterator")
            def func_with_invalid_stream_input(queue0, queue1):
                in0 = queue0.get()
                in1 = queue1.get()
                return df.alloc_tensor([2, 3], dt.DT_FLOAT16)

        self.assertRaises(TypeError, decorate_func)

        def decorate_class_method():
            @df.pyflow
            class MyClass:
                @df.method(stream_input="Iterator")
                def func_with_invalid_stream_input(self, queue0):
                    in0 = queue0.get()
                    return in0

        self.assertRaises(TypeError, decorate_class_method)

    def test_choice_ouput(self):
        ret = func_with_choice_output(fw.MetaRunContext(), [fw.FlowMsg(), fw.FlowMsg()])
        self.assertEqual(ret, 0)

    def test_stream_output(self):
        ret = func_with_stream_ouput(fw.MetaRunContext(), [fw.FlowMsg(), fw.FlowMsg()])
        self.assertEqual(ret, 0)
        fnode0 = MyClassWithStreamOutput.fnode()
        with open(
                fnode0.name + "_ws/src_python/MyClassWithStreamOutput.pkl", "rb"
        ) as f:
            my_class = cloudpickle.loads(f.read())
            ret = my_class.func_with_stream_ouput(fw.MetaRunContext(), [fw.FlowMsg()])
            self.assertEqual(ret, 0)
        os.system("rm -rf ./*_fnode_ws")

    def test_func_with_invalid_output_num(self):
        ret = func_with_invalid_output_num(
            fw.MetaRunContext(), [fw.FlowMsg(), fw.FlowMsg()]
        )
        self.assertEqual(ret, ff.FLOW_FUNC_FAILED)

    def test_pyflow_run_with_invalid_input(self):
        ret = func_with_stream_input(fw.MetaRunContext(), [0, 1])
        self.assertEqual(ret, ff.FLOW_FUNC_ERR_PARAM_INVALID)
        fnode0 = MyClass.fnode()
        with open(fnode0.name + "_ws/src_python/MyClass.pkl", "rb") as f:
            my_class = cloudpickle.loads(f.read())
            ret = my_class.func_with_stream_input(fw.MetaRunContext(), [0])
            self.assertEqual(ret, ff.FLOW_FUNC_ERR_PARAM_INVALID)
        os.system("rm -rf ./*_fnode_ws")

    def test_utils(self):
        instance_id = 0
        # Act
        utils.set_running_instance_id(instance_id)
        result = utils.get_running_instance_id()
        # Assert
        self.assertEqual(result, instance_id)

        instance_num = 1
        # Act
        utils.set_running_instance_num(instance_num)
        result = utils.get_running_instance_num()
        # Assert
        self.assertEqual(result, instance_num)
