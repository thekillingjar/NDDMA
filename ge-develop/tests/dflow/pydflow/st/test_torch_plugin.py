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
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import dataflow.plugin.torch.torch_plugin as torch_plugin
import dataflow.data_type as dt
import dataflow.utils.utils as utils
import dataflow as df


class TestTorchPlugin(unittest.TestCase):
    def setUp(self):
        utils.set_running_device_id(0)
        return super().setUp()

    @staticmethod
    def MockTorchType(torch):
        torch.float32 = "float32"
        torch.float16 = "float16"
        torch.bfloat16 = "bfloat16"
        torch.int8 = "int8"
        torch.int16 = "int16"
        torch.uint16 = "uint16"
        torch.uint8 = "uint8"
        torch.int32 = "int32"
        torch.int64 = "int64"
        torch.uint32 = "uint32"
        torch.uint64 = "uint64"
        torch.bool = "bool"
        torch.float64 = "float64"
        torch.__version__ = "2.5.0"

    @patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch_npu": MagicMock(),
            "torchair": MagicMock(),
            "cloudpickle": MagicMock(),
        },
    )
    @patch("dataflow.plugin.torch.torch_plugin.msg_type_register")
    def test_func_with_npu_model(self, mock_msg_type_register):
        import torch
        import torch_npu
        import torchair
        import cloudpickle

        TestTorchPlugin.MockTorchType(torch)

        cloudpickle.dumps = MagicMock(return_value=bytes(1024))

        @torch_plugin.npu_model(num_returns=2)
        def func_with_npu_model(in0, in1):
            return in0, in1

        data0 = df.FlowData(name="d0")
        data1 = df.FlowData(name="d1")
        fnode = func_with_npu_model.fnode()
        out = fnode(data0, data1)
        self.assertEqual(len(out), 2)
        self.assertEqual(Path("./func_with_npu_model_fnode_ws").exists(), True)

    @patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch_npu": MagicMock(),
            "torchair": MagicMock(),
            "cloudpickle": MagicMock(),
        },
    )
    @patch("dataflow.plugin.torch.torch_plugin.msg_type_register")
    def test_class_with_npu_model_1(self, mock_msg_type_register):
        import torch
        import torch_npu
        import torchair
        import cloudpickle

        TestTorchPlugin.MockTorchType(torch)

        cloudpickle.dumps = MagicMock(return_value=bytes(1024))

        @torch_plugin.npu_model(optimize_level=1)
        class TestNpuModel:
            @df.method()
            def sub_1(self, in0, in1):
                return in0 + in1

        data0 = df.FlowData(name="d0")
        data1 = df.FlowData(name="d1")
        fnode = TestNpuModel.fnode()
        out = fnode(data0, data1)
        self.assertTrue(isinstance(out, df.dataflow.FlowOutput))
        self.assertEqual(Path("./TestNpuModel_fnode_ws").exists(), True)

    @patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch_npu": MagicMock(),
            "torchair": MagicMock(),
            "cloudpickle": MagicMock(),
        },
    )
    @patch("dataflow.plugin.torch.torch_plugin.msg_type_register")
    def test_class_with_npu_model_2_dynamic(self, mock_msg_type_register):
        import torch
        import torch_npu
        import torchair
        import cloudpickle

        TestTorchPlugin.MockTorchType(torch)

        cloudpickle.dumps = MagicMock(return_value=bytes(1024))

        @torch_plugin.npu_model(
            optimize_level=2,
            input_descs=[
                df.TensorDesc(dt.DT_FLOAT, [-1, 1024]),
                df.TensorDesc(dt.DT_FLOAT, [-1, 1024]),
            ],
        )
        class TestNpuModelLevel2Dynamic:
            def __init__(self):
                pass

            @df.method()
            def forward(self, in0, in1):
                return in0

        workspace_dir = "./TestNpuModelLevel2Dynamic_fnode_ws"
        data0 = df.FlowData(name="d0")
        data1 = df.FlowData(name="d1")
        fnode = TestNpuModelLevel2Dynamic.fnode()
        out = fnode(data0, data1)
        self.assertTrue(isinstance(out, df.dataflow.FlowOutput))
        torchair.dynamo_export.assert_called_once()
        _, kwargs = torchair.dynamo_export.call_args
        self.assertEqual(kwargs["export_path"], workspace_dir)
        self.assertEqual(kwargs["dynamic"], True)

    @patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch_npu": MagicMock(),
            "torchair": MagicMock(),
            "cloudpickle": MagicMock(),
        },
    )
    @patch("dataflow.plugin.torch.torch_plugin.msg_type_register")
    def test_class_with_npu_model_2_static(self, mock_msg_type_register):
        import torch
        import torch_npu
        import torchair
        import cloudpickle

        TestTorchPlugin.MockTorchType(torch)

        cloudpickle.dumps = MagicMock(return_value=bytes(1024))

        @torch_plugin.npu_model(
            optimize_level=2,
            input_descs=[
                df.TensorDesc(dt.DT_FLOAT, [2, 1024]),
                df.TensorDesc(dt.DT_FLOAT, [2, 1024]),
            ],
        )
        class TestNpuModelLevel2Static:
            def __init__(self):
                pass

            @df.method()
            def forward(self, in0, in1):
                return in0

        workspace_dir = "./TestNpuModelLevel2Static_fnode_ws"
        data0 = df.FlowData(name="d0")
        data1 = df.FlowData(name="d1")
        fnode = TestNpuModelLevel2Static.fnode()
        out = fnode(data0, data1)
        self.assertTrue(isinstance(out, df.dataflow.FlowOutput))
        torchair.dynamo_export.assert_called_once()
        _, kwargs = torchair.dynamo_export.call_args
        self.assertEqual(kwargs["export_path"], workspace_dir)
        self.assertEqual(kwargs["dynamic"], False)

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_serialize_with_torch_tensor(self):
        import torch

        TestTorchPlugin.MockTorchType(torch)
        torch_tensor = MagicMock()
        torch_tensor.device = torch.device("cpu")
        torch_tensor.is_contiguous.return_value = True
        torch_tensor.numpy.return_value = b"mock_data"
        torch_tensor.data_ptr.return_value = 0
        torch_tensor.shape = [1]
        torch_tensor.dtype = torch.float32
        torch_tensor.numel.return_value = 1
        torch_tensor.element_size.return_value = 4

        serialized = torch_plugin._serialize_with_torch_tensor(torch_tensor)
        self.assertIsInstance(serialized, bytes)

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_deserialize_with_torch_tensor(self):
        import torch

        buffer = bytes(1024)
        deserialized = torch_plugin._deserialize_with_torch_tensor(buffer)
        torch.frombuffer.assert_called_once()

    @patch.dict(
        "sys.modules",
        {"torch": MagicMock(), "torch_npu": MagicMock(), "torchair": MagicMock()},
    )
    def test_prepare_inputs(self):
        import torch
        import torch_npu
        import torchair

        TestTorchPlugin.MockTorchType(torch)

        torchair.llm_datadist.create_npu_tensors.return_value = [MagicMock()]
        import dataflow.flow_func.flowfunc_wrapper as fw
        import dataflow.flow_func as ff

        orig_context = fw.MetaRunContext()
        context = ff.MetaRunContext(orig_context)
        msg1 = context.alloc_tensor_msg([2, 3], dt.DT_FLOAT16)
        inputs = [msg1._flow_msg]
        input_num = 1
        ret, input_list = torch_plugin._prepare_inputs(inputs, input_num)
        self.assertEqual(ret, torch_plugin.ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(len(input_list), 1)

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_prepare_outputs_single(self):
        import dataflow.flow_func.flowfunc_wrapper as fw
        import dataflow.flow_func as ff

        orig_context = fw.MetaRunContext()
        runtime_context = ff.MetaRunContext(orig_context)
        import torch

        TestTorchPlugin.MockTorchType(torch)
        torch.Tensor = type("MockTensor", (object,), {})

        mock_tensor = MagicMock()
        mock_tensor.__class__ = torch.Tensor
        mock_tensor.dtype = torch.float32
        mock_tensor.shape = [2, 3]
        mock_tensor.device.type = "npu"
        outputs = mock_tensor
        output_num = 1
        ret, runtime_tensor_descs = torch_plugin._prepare_outputs(
            runtime_context, outputs, output_num
        )
        self.assertEqual(ret, torch_plugin.ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(len(runtime_tensor_descs), 1)
        mock_tensor.device.type = "cpu"
        err_outputs = mock_tensor
        ret, runtime_tensor_descs = torch_plugin._prepare_outputs(
            runtime_context, err_outputs, output_num
        )
        self.assertNotEqual(ret, torch_plugin.ff.FLOW_FUNC_SUCCESS)

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_prepare_outputs_multi(self):
        import dataflow.flow_func.flowfunc_wrapper as fw
        import dataflow.flow_func as ff

        orig_context = fw.MetaRunContext()
        runtime_context = ff.MetaRunContext(orig_context)
        import torch

        TestTorchPlugin.MockTorchType(torch)
        torch.Tensor = type("MockTensor", (object,), {})

        mock_tensor = MagicMock()
        mock_tensor.__class__ = torch.Tensor
        mock_tensor.dtype = torch.float32
        mock_tensor.shape = [2, 3]
        mock_tensor.device.type = "npu"
        outputs = (mock_tensor, mock_tensor, mock_tensor)
        output_num = 3
        ret, runtime_tensor_descs = torch_plugin._prepare_outputs(
            runtime_context, outputs, output_num
        )
        self.assertEqual(ret, torch_plugin.ff.FLOW_FUNC_SUCCESS)
        self.assertEqual(len(runtime_tensor_descs), 1)
        err_outputs = ("aaa", mock_tensor, mock_tensor)
        ret, runtime_tensor_descs = torch_plugin._prepare_outputs(
            runtime_context, err_outputs, output_num
        )
        self.assertNotEqual(ret, torch_plugin.ff.FLOW_FUNC_SUCCESS)
