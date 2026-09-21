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
import threading
from typing import Union, List
import dataflow.data_type as dt
import dataflow.dataflow as df
import dataflow.flow_func as ff
import dataflow.utils.utils as utils
import dataflow.utils.log as log
from dataflow.flow_func import flowfunc_wrapper as fw
from dataflow.utils.msg_type_register import msg_type_register
from dataflow.pyflow import PyActorProcessPoint, PyFunctionProcessPoint


_npu_model_support_args_ = [
    "num_returns",
    "resources",
    "env_hook_func",
    "visible_device_enable",
]
_npu_actor_model_support_args_ = _npu_model_support_args_ + [
    "optimize_level",
    "input_descs",
]

_df_to_torch_dtype = None
_torch_to_df_dtype = None
_lock = threading.Lock()


def _initialize_torch_to_df_dtype():
    import torch

    global _torch_to_df_dtype
    global _df_to_torch_dtype
    _torch_to_df_dtype = {
        torch.float32: dt.DT_FLOAT,
        torch.float16: dt.DT_FLOAT16,
        torch.bfloat16: dt.DT_BF16,
        torch.int8: dt.DT_INT8,
        torch.int16: dt.DT_INT16,
        torch.uint8: dt.DT_UINT8,
        torch.int32: dt.DT_INT32,
        torch.int64: dt.DT_INT64,
        torch.bool: dt.DT_BOOL,
        torch.float64: dt.DT_DOUBLE,
    }
    if torch.__version__ >= "2.3":
        _torch_to_df_dtype.update(
            {
                torch.uint16: dt.DT_UINT16,
                torch.uint32: dt.DT_UINT32,
                torch.uint64: dt.DT_UINT64,
            }
        )
    _df_to_torch_dtype = {v: k for k, v in _torch_to_df_dtype.items()}


def _convert_df_to_torch_tensor_dtype(df_dtype):
    global _df_to_torch_dtype
    # 使用锁来确保初始化操作是线程安全的
    if _df_to_torch_dtype is None:
        with _lock:  # 获取锁
            if _df_to_torch_dtype is None:  # 双重检查，确保只有一个线程初始化
                _initialize_torch_to_df_dtype()

    if df_dtype not in _df_to_torch_dtype:
        raise ValueError(f"df_dtype {df_dtype} is not supported")
    return _df_to_torch_dtype[df_dtype]


def _convert_torch_to_df_tensor_dtype(torch_dtype):
    global _torch_to_df_dtype
    # 使用锁来确保初始化操作是线程安全的
    if _torch_to_df_dtype is None:
        with _lock:  # 获取锁
            if _torch_to_df_dtype is None:  # 双重检查，确保只有一个线程初始化
                _initialize_torch_to_df_dtype()

    if torch_dtype not in _torch_to_df_dtype:
        raise ValueError(f"torch_dtype {torch_dtype} is not supported")
    return _torch_to_df_dtype[torch_dtype]


def _prepare_inputs(inputs: Union[List[fw.FlowMsg]], input_num):
    import torch
    import torch_npu
    import torchair

    # make sure device is running device
    torch.npu.set_device(utils.get_running_device_id())
    input_list = []
    ret, runtime_tensor_descs = (
        fw.RuntimeTensorDescMsgProcessor.get_runtime_tensor_descs(inputs[0], input_num)
    )
    if ret != ff.FLOW_FUNC_SUCCESS:
        ff.logger.error("get_runtime_tensor_descs failed, ret=%d", ret)
        return ret, input_list
    for runtime_tensor_desc in runtime_tensor_descs:
        torch_tensor_dtype = _convert_df_to_torch_tensor_dtype(
            dt.get_python_dtype_from_dwrapper_dtype(runtime_tensor_desc.dtype)
        )
        npu_tensors = torchair.llm_datadist.create_npu_tensors(
            runtime_tensor_desc.shape, torch_tensor_dtype, [runtime_tensor_desc.address]
        )
        input_list.append(npu_tensors[0])
    return ff.FLOW_FUNC_SUCCESS, input_list


def _check_torch_output(output, idx=0):
    import torch

    if not isinstance(output, (torch.Tensor)):
        ff.logger.error(f"output:{idx} type:{type(output)} is not torch tensor")
        return ff.FLOW_FUNC_FAILED

    if output.device.type != "npu":
        ff.logger.error(f"output:{idx} device type:{output.device.type} is not npu")
        return ff.FLOW_FUNC_FAILED

    return ff.FLOW_FUNC_SUCCESS


def _prepare_outputs(runtime_context, outputs, output_num):
    runtime_tensor_descs = []
    if output_num > 1:
        if not isinstance(outputs, (tuple)):
            ff.logger.error(
                "output num error, num_returns = %d, but return is not tuple.",
                output_num,
            )
            return ff.FLOW_FUNC_FAILED, []
        elif len(outputs) != output_num:
            ff.logger.error(
                "output num error, num_returns = %d, but real return num = %d.",
                output_num,
                len(outputs),
            )
            return ff.FLOW_FUNC_FAILED, []
        else:
            for index, output in enumerate(outputs):
                if _check_torch_output(output, index) != ff.FLOW_FUNC_SUCCESS:
                    return ff.FLOW_FUNC_FAILED, []
                df_tensor_dtype = _convert_torch_to_df_tensor_dtype(output.dtype)
                desc = fw.RuntimeTensorDesc()
                desc.address = output.data_ptr()
                desc.shape = list(output.shape)
                desc.dtype = int(df_tensor_dtype.dtype)
                desc.size = output.numel() * output.element_size()
                runtime_tensor_descs.append(desc)
    else:
        if _check_torch_output(outputs) != ff.FLOW_FUNC_SUCCESS:
            return ff.FLOW_FUNC_FAILED, []
        df_tensor_dtype = _convert_torch_to_df_tensor_dtype(outputs.dtype)
        desc = fw.RuntimeTensorDesc()
        desc.address = outputs.data_ptr()
        desc.shape = list(outputs.shape)
        desc.dtype = int(df_tensor_dtype.dtype)
        desc.size = outputs.numel() * outputs.element_size()
        runtime_tensor_descs.append(desc)
    desc_msg = fw.RuntimeTensorDescMsgProcessor.create_runtime_tensor_desc_msg(
        runtime_context._context, runtime_tensor_descs
    )
    if desc_msg is None:
        ff.logger.error("create_runtime_tensor_desc_msg failed")
        return ff.FLOW_FUNC_FAILED, []
    desc_msg.set_msg_type(int(ff.MSG_TYPE_TORCH_TENSOR_MSG))
    return ff.FLOW_FUNC_SUCCESS, [desc_msg]


def _dynamo_export(class_ins, input_descs, workspace_dir):
    import torch
    import torch_npu
    import torchair

    model = class_ins._decorated_class(
        *class_ins._saved_args[0], **class_ins._saved_args[1]
    )
    input_list = []
    is_dynamic = False
    for input_desc in input_descs:
        desc_shape = [1 if item < 0 else item for item in input_desc._shape]
        print(f"input_desc._dtype={input_desc._dtype}")
        input_list.append(
            torch.ones(
                *desc_shape, dtype=_convert_df_to_torch_tensor_dtype(input_desc._dtype)
            )
        )
        if input_desc._shape != desc_shape:
            is_dynamic = True

    torchair.dynamo_export(
        *input_list, model=model, export_path=workspace_dir, dynamic=is_dynamic
    )


def _serialize_with_torch_tensor(torch_tensor):
    import torch

    if torch_tensor.device != torch.device("cpu"):
        raise TypeError(
            f"torch tensor device:{torch_tensor.device} is not support when df use @npu_model, please to cpu."
        )
    if not torch_tensor.is_contiguous():
        torch_tensor = torch_tensor.contiguous()
    df_tensor_dtype = _convert_torch_to_df_tensor_dtype(torch_tensor.dtype)

    desc = fw.RuntimeTensorDesc()
    desc.address = torch_tensor.data_ptr()
    desc.shape = list(torch_tensor.shape)
    desc.dtype = int(df_tensor_dtype.dtype)
    desc.size = torch_tensor.numel() * torch_tensor.element_size()
    return desc.to_bytes() + memoryview(torch_tensor.numpy())


def _deserialize_with_torch_tensor(buffer):
    import torch

    desc = fw.RuntimeTensorDesc.from_memory(buffer)
    dtype = _convert_df_to_torch_tensor_dtype(
        dt.get_python_dtype_from_dwrapper_dtype(desc.dtype)
    )
    return torch.frombuffer(buffer=buffer, dtype=dtype, offset=1024).reshape(desc.shape)


def _register_torch_tensor():
    import torch

    msg_type_register.register(
        ff.MSG_TYPE_TORCH_TENSOR_MSG,
        torch.Tensor,
        _serialize_with_torch_tensor,
        _deserialize_with_torch_tensor,
    )


class NpuActorProcessPoint(PyActorProcessPoint):
    @classmethod
    def check_options_supported(cls, node_options):
        for key in node_options.keys():
            if key not in _npu_actor_model_support_args_:
                raise TypeError(f"param:{key} is not support in @npu_model.")

    @classmethod
    def add_process_point(cls, flow_node, class_ins):
        workspace_dir = "./" + flow_node.name + "_ws"
        optimize_level = 1
        if "optimize_level" in class_ins._default_options:
            optimize_level = class_ins._default_options["optimize_level"]
            if not isinstance(optimize_level, int):
                raise TypeError(
                    f"optimize_level must be a number, but got {type(optimize_level)}."
                )
        if optimize_level == 1:
            pp = df.FuncProcessPoint(py_func=class_ins, workspace_dir=workspace_dir)
        elif optimize_level == 2:
            input_descs = class_ins._default_options.get("input_descs")
            if input_descs is None:
                raise TypeError(
                    f"optimize_level is {optimize_level}, but input_descs is None."
                )
            _dynamo_export(class_ins, input_descs, workspace_dir)
            pp = df.GraphProcessPoint(
                df.Framework.MINDSPORE, workspace_dir + "/export.air"
            )
        else:
            raise TypeError(
                f"optimize_level:{optimize_level} is not support in @npu_model."
            )
        flow_node.add_process_point(pp)
        flow_node.set_attr("_npu_sched_model", 1)

    @classmethod
    def get_redefined_method(
        cls,
        class_ins,
        method_name,
        method_def,
        input_num,
        output_idx_offset,
        output_num,
    ):
        class MethodClass:
            def __init__(
                self,
                class_ins,
                method_name,
                method_def,
                input_num,
                output_idx_offset,
                output_num,
            ):
                self._class_ins = class_ins
                self._method = method_def
                self._method_name = method_name
                self._input_num = input_num
                self._output_num = output_num
                self._result = []

            def __call__(
                self,
                run_context: fw.MetaRunContext,
                inputs: Union[List[fw.FlowMsg], List[fw.FlowMsgQueue]],
            ):
                try:
                    prepare_ret, input_list = _prepare_inputs(inputs, self._input_num)
                    if prepare_ret != ff.FLOW_FUNC_SUCCESS:
                        ff.logger.error(
                            "failed to prepare input, ret = %d", prepare_ret
                        )
                        return prepare_ret

                    runtime_context = ff.MetaRunContext(run_context)
                    # cache result until next call
                    self._result = self._method(self._class_ins, *tuple(input_list))
                    import torch

                    torch.npu.synchronize()
                    return self._set_output(runtime_context, self._result)
                except utils.DfAbortException as e:
                    ff.logger.warn("proc is aborted, %s", str(e))
                    return ff.FLOW_FUNC_SUCCESS
                except Exception as e:
                    traceback.print_exc()
                    ff.logger.error("proc wrapper exception, %s", str(e))
                    return ff.FLOW_FUNC_FAILED
                return ff.FLOW_FUNC_SUCCESS

            def _set_output(self, runtime_context, result):
                ret = _prepare_outputs(runtime_context, result, self._output_num)
                if ret[0] != ff.FLOW_FUNC_SUCCESS:
                    ff.logger.error("failed to prepare output, ret = %d", ret[0])
                    return ret[0]
                result_list = ret[1]
                if (
                    runtime_context.set_output(0, result_list[0])
                    != ff.FLOW_FUNC_SUCCESS
                ):
                    ff.logger.error("failed to set output")
                    return ff.FLOW_FUNC_FAILED
                return ff.FLOW_FUNC_SUCCESS

        return MethodClass(
            class_ins, method_name, method_def, input_num, output_idx_offset, output_num
        )


class NpuFunctionProcessPoint(PyFunctionProcessPoint):
    def __init__(
        self,
        function,
        node_options,
    ):
        super().__init__(function, node_options)
        self._outputs = None

    @classmethod
    def check_options_supported(cls, node_options):
        for key in node_options.keys():
            if key not in _npu_model_support_args_:
                raise TypeError(
                    f"param:{key} is not support in @npu_model when applied to a function."
                )

    def add_process_point(self, flow_node):
        pp = df.FuncProcessPoint(
            py_func=self, workspace_dir="./" + flow_node.name + "_ws"
        )
        flow_node.add_process_point(pp)
        flow_node.set_attr("_npu_sched_model", 1)

    def prepare_inputs(self, inputs: Union[List[fw.FlowMsg]], input_num):
        return _prepare_inputs(inputs, input_num)

    def prepare_outputs(self, runtime_context, outputs, output_num):
        import torch

        torch.npu.synchronize()
        # cache result until next call
        self._outputs = outputs
        return _prepare_outputs(runtime_context, outputs, output_num)


def _make_npu_model(function_or_class, options):
    import torch
    import torch_npu
    import torchair

    _register_torch_tensor()
    if inspect.isfunction(function_or_class):
        return NpuFunctionProcessPoint(function_or_class, options)

    if inspect.isclass(function_or_class):
        return NpuActorProcessPoint._df_from_class(function_or_class, options)

    raise TypeError(
        f"The @npu_model decorator must be applied to either a function or a class."
    )


def npu_model(*args, **kwargs) -> Union[NpuFunctionProcessPoint]:
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return _make_npu_model(args[0], {})
    return functools.partial(_make_npu_model, options=kwargs)
