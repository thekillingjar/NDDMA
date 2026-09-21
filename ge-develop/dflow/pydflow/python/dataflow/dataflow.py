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

from __future__ import annotations
import atexit
from enum import auto, Enum, IntEnum
from typing import Any, Optional, Dict, Union, List, Tuple
import threading
import os
import sys
import numpy as np
import dataflow.utils.log as log
from dataflow.utils.name_scope import name_scope, reset_name_scope
import dataflow.utils.utils as utils
from dataflow.utils.msg_type_register import msg_type_register
from dataflow.tools.func_ws_creator import FuncWsCreator
from dataflow.flow_func.func_register import FlowFuncRegister
from dataflow.flow_func import flowfunc_wrapper
import dataflow.data_type as df_dt
import dataflow.dflow_wrapper as dwrapper
import dataflow.data_wrapper as data_wrapper

_global_options = None
_global_options_lock = threading.Lock()
_global_context = None
_global_context_lock = threading.Lock()
_global_graph_id_seed = 0
_global_graph_id_seed_lock = threading.Lock()

_support_np_dtype = [
    np.float32,
    np.float16,
    np.int8,
    np.int16,
    np.uint16,
    np.uint8,
    np.int32,
    np.int64,
    np.uint32,
    np.uint64,
    np.bool_,
    np.float64,
]

dwrapper.init_datatype_manager(
    {
        data_wrapper.DataType.DT_FLOAT: np.array([], np.float32),
        data_wrapper.DataType.DT_FLOAT16: np.array([], np.float16),
        data_wrapper.DataType.DT_INT8: np.array([], np.int8),
        data_wrapper.DataType.DT_INT16: np.array([], np.int16),
        data_wrapper.DataType.DT_UINT16: np.array([], np.uint16),
        data_wrapper.DataType.DT_UINT8: np.array([], np.uint8),
        data_wrapper.DataType.DT_INT32: np.array([], np.int32),
        data_wrapper.DataType.DT_INT64: np.array([], np.int64),
        data_wrapper.DataType.DT_UINT32: np.array([], np.uint32),
        data_wrapper.DataType.DT_UINT64: np.array([], np.uint64),
        data_wrapper.DataType.DT_BOOL: np.array([], np.bool_),
        data_wrapper.DataType.DT_DOUBLE: np.array([], np.double),
    }
)

# flow msg type
MSG_TYPE_TENSOR_DATA = dwrapper.MsgType.MSG_TYPE_TENSOR_DATA
MSG_TYPE_PICKLED_MSG = 65535
MSG_TYPE_USER_DEFINE_START = 1024


class _Context(object):
    """
    Internal class. Only one class object is created.
    """

    def __init__(self, options):
        ret = dwrapper.ge_initialize(options)
        if ret != 0:
            raise utils.DfException(
                "Failed to ge initialize, for details about the error information, see the ascend log.",
                ret,
            )
        try:
            self.session = dwrapper.DFlowSession(options)
        except dwrapper.DataFlowRuntimeError as e:
            raise utils.DfException(e.__str__(), e.GetErrorCode())

    def __del__(self):
        dwrapper.ge_finalize()


@utils.assert_args_type(Dict[str, str])
def init(flow_options: Optional[Dict[str, str]]) -> None:
    """
    config data flow initialize options"
    """
    global _global_options
    if _global_options is not None:
        log.warning("It's already initialized. flow_options=%s", _global_options)
        return
    with _global_options_lock:
        if _global_options:
            log.warning("It's already initialized. flow_options=%s", _global_options)
            return
        _global_options = flow_options
        # add dataflow default config
        _global_options["ge.graphRunMode"] = "0"
        log.info("Init succeeded, set flow_options=%s", _global_options)


@atexit.register
def _clear():
    finalize()


def finalize() -> None:
    with _global_context_lock:
        global _global_context
        if _global_context is not None:
            del _global_context
            _global_context = None
    with _global_options_lock:
        global _global_options
        if _global_options is not None:
            _global_options = None
    reset_name_scope()


class TimeBatch(dwrapper.TimeBatch):
    @utils.assert_args_type(time_window=int, batch_dim=int, drop_remainder=bool)
    def __init__(
        self, time_window: int = 0, batch_dim: int = -1, drop_remainder: bool = False
    ) -> None:
        super().__init__()
        self.time_window: int = time_window
        self.batch_dim: int = batch_dim
        self.drop_remainder: bool = drop_remainder
        self._attr_type = dwrapper.DataFlowAttrType.TIME_BATCH


class CountBatch(dwrapper.CountBatch):
    @utils.assert_args_type(batch_size=int, slide_stride=int, timeout=int, padding=bool)
    def __init__(
        self,
        batch_size: int = 0,
        slide_stride: int = 0,
        timeout: int = 0,
        padding: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size: int = batch_size
        self.slide_stride: int = slide_stride
        self.timeout: int = timeout
        self.padding: bool = padding
        self._attr_type = dwrapper.DataFlowAttrType.COUNT_BATCH


class Framework(Enum):
    TENSORFLOW = auto()
    ONNX = auto()
    MINDSPORE = auto()


class FlowFlag(IntEnum):
    DATA_FLOW_FLAG_EOS = 1 << 0  # data flag end
    DATA_FLOW_FLAG_SEG = 1 << 1  # segment flag for discontinuous


_framework_str = {
    Framework.TENSORFLOW: "tensorflow",
    Framework.ONNX: "onnx",
    Framework.MINDSPORE: "mindspore",
}


class BaseData(object):
    def __init__(self, cls_type):
        self.cls_type = cls_type


class TensorDesc(object):
    def __init__(self, dtype: df_dt.DType, shape: Union[List[int], Tuple[int]]):
        if not isinstance(dtype, df_dt.DType):
            raise utils.DfException(
                f"Argument dtype must be {df_dt.DType}, but got {dtype}",
                dwrapper.DATATYPE_INVALID,
            )
        if (not isinstance(shape, List)) and (not isinstance(shape, Tuple)):
            raise utils.DfException(
                f"Argument shape must be {List[int]} or {Tuple[int]}, but got {shape}",
                dwrapper.SHAPE_INVALID,
            )
        for dim in shape:
            if not isinstance(dim, int):
                raise utils.DfException(
                    f"Argument shape must be {List[int]} or {Tuple[int]}, but got {shape}",
                    dwrapper.SHAPE_INVALID,
                )
        self._dtype = dtype
        self._shape = list(shape)

    def __str__(self):
        return (
            "TensorDesc("
            + "dtype="
            + str(self._dtype)
            + ", shape="
            + str(self._shape)
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, TensorDesc):
            return False
        return (self._dtype == other._dtype) and (self._shape == other._shape)


class Tensor(BaseData):
    def __init__(self, data: Any, *, tensor_desc: TensorDesc = None) -> None:
        super().__init__(Tensor)
        self._flow_msg = None
        # Invoked internally
        if isinstance(data, dwrapper.Tensor):
            self._impl = data
            self._tensor_desc = TensorDesc(
                df_dt.get_python_dtype_from_dwrapper_dtype(data.get_dtype()),
                data.get_shape(),
            )
        elif isinstance(data, flowfunc_wrapper.Tensor):
            self._impl = data
            self._tensor_desc = None
        elif isinstance(data, dwrapper.FlowMsg) or isinstance(
            data, flowfunc_wrapper.FlowMsg
        ):
            self._flow_msg = data
            self._impl = data.get_tensor()
            self._tensor_desc = TensorDesc(
                df_dt.get_python_dtype_from_dwrapper_dtype(self._impl.get_dtype()),
                self._impl.get_shape(),
            )
        else:
            if isinstance(data, Tensor):
                if (not tensor_desc) or (tensor_desc == data._tensor_desc):
                    self._tensor_desc = data._tensor_desc
                    self._impl = data._impl.clone()
                    return
                else:
                    copy = (
                        True if data._tensor_desc._dtype == df_dt.DT_STRING else False
                    )
                    data = data.numpy(copy)
            elif not isinstance(data, np.ndarray):
                # Use the shape and data type inference function of numpy
                data = np.array(data)
            else:
                pass
            if tensor_desc:
                if list(data.shape) != tensor_desc._shape:
                    raise utils.DfException(
                        f"The shape of argument data is {data.shape}, "
                        f"but the argument tensor_desc is {tensor_desc}.",
                        dwrapper.SHAPE_INVALID,
                    )
                tensor_desc_dtype = df_dt._dflow_dtype_to_np_dtype.get(
                    tensor_desc._dtype, None
                )
                if not tensor_desc_dtype:
                    raise utils.DfException(
                        f"The dtype of argument tensor_desc is "
                        f" {tensor_desc._dtype} which is not supported.",
                        dwrapper.DATATYPE_INVALID,
                    )
                if data.dtype != tensor_desc_dtype:
                    log.warning(
                        f"The dtype of argument tensor_desc is {tensor_desc._dtype}, but dtype "
                        f"{data.dtype} data is fed, the data will be tried to convert from type "
                        f"{data.dtype} to type {tensor_desc._dtype}."
                    )
                    try:
                        data = np.asarray(data, tensor_desc_dtype)
                    except BaseException as e:
                        raise ValueError(
                            f"Failed to convert type {data.dtype} to type {tensor_desc._dtype}."
                        ) from e

            else:
                if (
                    data.dtype not in _support_np_dtype
                    and not self._is_origin_dtype_str(data.dtype)
                ):
                    raise utils.DfException(
                        f"The dtype of data is {data.dtype}, should be in {_support_np_dtype}",
                        dwrapper.DATATYPE_INVALID,
                    )
            try:
                if self._is_origin_dtype_str(data.dtype):
                    self._impl = self._convert_str_dtype_to_wrapper_tensor(data)
                else:
                    self._impl = dwrapper.Tensor(data)
            except BaseException as e:
                raise utils.DfException(e.__str__())

            if tensor_desc:
                self._tensor_desc = tensor_desc
            else:
                if self._is_origin_dtype_str(data.dtype):
                    self._tensor_desc = TensorDesc(df_dt.DT_STRING, data.shape)
                else:
                    self._tensor_desc = TensorDesc(
                        df_dt._np_dtype_to_dflow_dtype.get(data.dtype, None), data.shape
                    )

    def __str__(self):
        if self._is_inner_dtype_str():
            return (
                "Tensor("
                + str(self.numpy(True))
                + ", tensor_desc="
                + str(self._tensor_desc)
                + ")"
            )
        return (
            "Tensor("
            + str(self.numpy())
            + ", tensor_desc="
            + str(self._tensor_desc)
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    def numpy(self, copy=False) -> np.ndarray:
        if self._is_inner_dtype_str():
            if not copy:
                raise utils.DfException("String tensor must set copy param true.")
            # Due to different memory allocation modes, memory copy must be performed for the string type.
            # The buffinfo function is a zero copy function
            return self._str_numpy()
        try:
            if copy:
                ret = np.array(self._impl, copy=True)
            else:
                ret = np.asarray(self._impl)
        except BaseException as e:
            raise utils.DfException(e.__str__())
        return ret

    def get_shape(self):
        if isinstance(self._impl, flowfunc_wrapper.Tensor):
            return self._impl.get_shape()
        else:
            raise utils.DfException(f"None flow func tensor is not supported.")

    def get_data_type(self):
        if isinstance(self._impl, flowfunc_wrapper.Tensor):
            ge_data_type = self._impl.get_dtype()
            wrapper_type = df_dt.get_python_dtype_from_dwrapper_dtype(ge_data_type)
            return df_dt._dflow_dtype_to_np_dtype.get(wrapper_type, None)
        else:
            raise utils.DfException(f"None flow func tensor is not supported.")

    def get_data_size(self):
        if isinstance(self._impl, flowfunc_wrapper.Tensor):
            return self._impl.get_data_size()
        else:
            raise utils.DfException(f"None flow func tensor is not supported.")

    def get_element_cnt(self):
        if isinstance(self._impl, flowfunc_wrapper.Tensor):
            return self._impl.get_element_cnt()
        else:
            raise utils.DfException(f"None flow func tensor is not supported.")

    def reshape(self, shape: Union[List[int], Tuple[int]]):
        if isinstance(self._impl, flowfunc_wrapper.Tensor):
            return self._impl.reshape(shape)
        else:
            raise utils.DfException(f"None flow func tensor is not supported.")

    def _is_origin_dtype_str(self, dtype):
        return np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_)

    def _is_inner_dtype_str(self):
        return (
            self._tensor_desc is not None
            and self._tensor_desc._dtype == df_dt.DT_STRING
        )

    def _convert_str_dtype_to_wrapper_tensor(self, data):
        # change to ascii encode
        format_data = np.asarray([data], dtype=np.bytes_)
        format_data = format_data.astype(np.bytes_)
        end_point = "\0".encode("ascii", errors="ignore")
        new_data = np.char.add(format_data, end_point)
        return dwrapper.Tensor(new_data)

    def _str_numpy(self) -> np.ndarray:
        try:
            ret = np.array(self._impl.get_string_tensor()).reshape(
                self._impl.get_shape()
            )
        except BaseException as e:
            raise utils.DfException(e.__str__())
        return ret


def alloc_tensor(
    shape: Union[List[int], Tuple[int]], dtype, align: Optional[int] = 64
) -> Tensor:
    if isinstance(dtype, df_dt.DType):
        dtype = dtype.dtype
    elif dtype in _support_np_dtype:
        dtype = df_dt._np_dtype_to_dflow_dtype.get(np.dtype(dtype)).dtype
    else:
        raise TypeError(f"The dtype of data is {dtype} is invalid.")
    if utils.get_running_in_udf():
        return Tensor(
            flowfunc_wrapper.FlowBufferFactory.alloc_tensor(shape, dtype, align)
        )
    else:
        return Tensor(dwrapper.FlowBufferFactory.alloc_tensor_msg(shape, dtype, align))


class GraphProcessPoint(object):
    @utils.assert_args_type(framework=Framework)
    def __init__(
        self,
        framework: Framework,
        graph_file: str,
        load_params: Optional[Dict[str, str]] = None,
        compile_config_path: Optional[str] = "",
        name: Optional[str] = None,
    ) -> None:
        self.framework = framework
        self.graph_file = graph_file
        self.load_params = load_params or {}
        self.compile_config_path = compile_config_path
        with name_scope(name, "GraphProcessPoint") as name:
            self.name = name

        ret = dwrapper.load_graph_pp(
            _framework_str.get(self.framework, "None"),
            self.graph_file,
            self.load_params,
            self.compile_config_path,
            self.name,
        )
        if len(ret) != 2:
            raise utils.DfException(
                "Failed to load graph pp, for details about the error information, "
                "see the ascend log.",
                dwrapper.INNER_ERROR,
            )
        if ret[0].ret_code != 0:
            raise utils.DfException(ret[0].error_msg, ret[0].ret_code)
        else:
            self._impl = ret[1]

    def fnode(self, input_num: int, output_num: int, name: Optional[str] = None):
        flow_node = FlowNode(input_num=input_num, output_num=output_num, name=name)
        flow_node.add_process_point(self)
        return flow_node


class FlowGraphProcessPoint(object):
    def __init__(
        self,
        flow_graph: FlowGraph,
        compile_config_path: Optional[str] = "",
        name: Optional[str] = None,
    ) -> None:
        self.compile_config_path = compile_config_path
        self.flow_graph = flow_graph
        with name_scope(name, "FlowGraphProcessPoint") as name:
            self.name = name

        ret = dwrapper.load_flow_graph_pp(
            self.flow_graph._impl, self.compile_config_path, self.name
        )
        if len(ret) != 2:
            raise utils.DfException(
                "Failed to load flow graph pp, for details about the error information, "
                "see the ascend log.",
                dwrapper.INNER_ERROR,
            )
        if ret[0].ret_code != 0:
            raise utils.DfException(ret[0].error_msg, ret[0].ret_code)
        else:
            self._impl = ret[1]


class FuncProcessPoint(object):
    def __init__(
        self,
        compile_config_path: Optional[str] = None,
        name: Optional[str] = None,
        py_func: Optional = None,
        workspace_dir: Optional = None,
    ) -> None:
        if py_func is not None:
            if isinstance(py_func, type):
                module_name = py_func.__module__
                class_name = py_func.__name__
                py_def = py_func
            else:
                module_name = py_func._module_name
                class_name = py_func._clz_name
                py_def = py_func.__class__

            register_params = FlowFuncRegister.get_flow_func(module_name, class_name)
            if register_params is None:
                raise utils.DfException(
                    "flow func {} is not decorator "
                    "by flow_func_annotation".format(py_def),
                    flowfunc_wrapper.FLOW_FUNC_FAILED,
                )
            ws_dir = ""
            if workspace_dir is None:
                ws_dir = "./" + class_name + "_ws"
            else:
                ws_dir = workspace_dir

            func_param = []
            for func in register_params:
                func_param.append(func + ":" + register_params[func].replace(",", ":"))
            funcs_param = ",".join(func_param)

            flow_func_infos = FlowFuncRegister.get_flow_func_infos(
                module_name, class_name
            )
            func_ws_creator = FuncWsCreator(
                funcs_param=funcs_param,
                clz_name=class_name,
                ws_dir=ws_dir,
                flow_func_infos=flow_func_infos,
            )
            func_ws_creator.generate()
            func_ws_creator.copy_py_src_file(
                os.path.abspath(sys.modules[module_name].__file__)
            )
            self.compile_config_path = func_ws_creator.get_config_file()
        else:
            self.compile_config_path = compile_config_path

        with name_scope(name, "FuncProcessPoint") as name:
            self.name = name
        self._impl = dwrapper.FunctionPp(self.name)
        self._impl.set_compile_config(self.compile_config_path)

    def set_init_param(
        self,
        attr_name: str,
        attr_value: Union[
            str,
            List[str],
            int,
            List[int],
            List[List[int]],
            float,
            List[float],
            bool,
            List[bool],
            df_dt.DType,
            List[df_dt.DType],
        ],
    ) -> None:
        if isinstance(attr_value, df_dt.DType):
            self._impl.set_init_param(attr_name, attr_value.dtype)
        elif (
            isinstance(attr_value, List)
            and len(attr_value) > 0
            and isinstance(attr_value[0], df_dt.DType)
        ):
            self._impl.set_init_param(attr_name, [value.dtype for value in attr_value])
        else:
            self._impl.set_init_param(attr_name, attr_value)

    def add_invoked_closure(self, graph_key: str, graph_pp: GraphProcessPoint) -> None:
        self._impl.add_invoked_closure(graph_key, graph_pp._impl)

    def add_invoked_closure(
        self, graph_key: str, flow_graph_pp: FlowGraphProcessPoint
    ) -> None:
        self._impl.add_invoked_closure(graph_key, flow_graph_pp._impl)


class FlowData(object):
    def __init__(
        self,
        data_cls=Tensor,
        schema: Optional[TensorDesc] = None,
        name: Optional[str] = None,
    ) -> None:
        if data_cls != Tensor:
            raise utils.DfException(
                f"Argument data_cls must be equal to Tensor, but got {data_cls}"
            )
        if schema and type(schema) != TensorDesc:
            raise utils.DfException(
                f"Argument schema type must be TensorDesc, but got {type(schema)}"
            )
        self._data_cls = data_cls
        self._schema = schema
        with name_scope(name, "FlowData") as name:
            self.name = name
        self._impl = dwrapper.FlowData(self.name, 0)


class FlowNode(object):
    def __init__(
        self, input_num: int, output_num: int, name: Optional[str] = None
    ) -> None:
        self.input_num = input_num
        self.output_num = output_num
        with name_scope(name, "FlowNode", "FlowNode") as name:
            self.name = name
        self._impl = dwrapper.FlowNode(self.name, self.input_num, self.output_num)
        # add default attr
        self._impl.set_attr("_flow_attr", True)
        self._input_anchors = [None] * input_num
        self._output_anchors = [None] * output_num
        self.alias = None

    def __call__(
        self, *inputs: Union[FlowData, "FlowOutput"]
    ) -> Union["FlowOutput", Tuple["FlowOutput", ...]]:
        if len(inputs) != self.input_num:
            raise utils.DfException(
                f"Flow node need {self.input_num} input, but got {len(inputs)} input"
            )
        for idx, arg in enumerate(inputs):
            self._input_anchors[idx] = arg
            if isinstance(arg, FlowData):
                # connect FlowData with me
                self._impl.set_input(idx, arg._impl, 0)
            elif isinstance(arg, FlowOutput):
                # connect FlowOutput with me
                if arg.node == self:
                    raise utils.DfException(
                        f"The {idx}th input is the output of current flow node, so cannot be used as an input"
                    )
                self._impl.set_input(idx, arg.node._impl, arg.index)
            else:
                raise utils.DfException(
                    f"Argument:{type(arg)} inputs must be {Union[FlowData, FlowOutput]}"
                )

        if self.output_num == 0:
            return None
        elif self.output_num == 1:
            output = FlowOutput(self, 0)
            self._output_anchors[0] = output
            return output
        else:
            for i in range(self.output_num):
                self._output_anchors[i] = FlowOutput(self, i)
            return tuple(self._output_anchors)

    @utils.assert_args_type(
        pp=Union[GraphProcessPoint, FuncProcessPoint, FlowGraphProcessPoint]
    )
    def add_process_point(
        self, pp: Union[GraphProcessPoint, FuncProcessPoint, FlowGraphProcessPoint]
    ) -> None:
        self._impl.add_pp(pp._impl)

    @utils.assert_args_type(
        pp=Union[GraphProcessPoint, FuncProcessPoint, FlowGraphProcessPoint]
    )
    def map_input(
        self,
        node_input_index: int,
        pp: Union[GraphProcessPoint, FuncProcessPoint, FlowGraphProcessPoint],
        pp_input_index: int,
        input_attrs: Optional[List[Union[TimeBatch, CountBatch]]] = None,
    ) -> None:
        if node_input_index >= self.input_num:
            raise utils.DfException(
                f"The arg node_input_index should in [0,{self.input_num})"
            )
        input_attrs = input_attrs or []
        attrs = self._convert_input_attrs(input_attrs)
        self._impl.map_input(node_input_index, pp._impl, pp_input_index, attrs)

    @utils.assert_args_type(
        pp=Union[GraphProcessPoint, FuncProcessPoint, FlowGraphProcessPoint]
    )
    def map_output(
        self,
        node_output_index: int,
        pp: Union[GraphProcessPoint, FuncProcessPoint, FlowGraphProcessPoint],
        pp_output_index: int,
    ) -> None:
        if node_output_index >= self.output_num:
            raise utils.DfException(
                f"The arg node_output_index should in [0,{self.output_num})"
            )
        self._impl.map_output(node_output_index, pp._impl, pp_output_index)

    def set_balance_scatter(self) -> None:
        self._impl.set_balance_scatter()

    def set_balance_gather(self) -> None:
        self._impl.set_balance_gather()

    def set_attr(self, attr_name: str, attr_value: Union[bool, str, int]) -> None:
        self._impl.set_attr(attr_name, attr_value)

    def set_alias(self, name):
        if name == self.name:
            self.alias = name
        else:
            with name_scope(name, "FlowNode", "FlowNode", False) as name:
                self.alias = name
        self.set_attr("_flow_attr_flow_node_alias", self.alias)
        return self

    def _convert_input_attrs(self, input_attrs):
        ret_attrs = []
        for input_attr in input_attrs:
            if not (
                isinstance(input_attr, TimeBatch) or isinstance(input_attr, CountBatch)
            ):
                raise utils.DfException(
                    "Argument input_attrs must be List[Union[TimeBatch, CountBatch]]"
                )
            attr = dwrapper.DataFlowInputAttr()
            attr.attr_type = input_attr._attr_type
            attr.attr_value = input_attr
            ret_attrs.append(attr)
        return ret_attrs

    def _build_flow_node(
        self, *inputs: Union[FlowData, "FlowOutput"], input_indexes, output_indexes
    ) -> Union["FlowOutput", Tuple["FlowOutput", ...]]:
        if len(inputs) != len(input_indexes):
            raise utils.DfException(
                f"Flow node func need {len(input_indexes)} input, but got {len(inputs)} input"
            )
        for idx, arg in enumerate(inputs):
            input_idx = input_indexes[idx]
            self._input_anchors[input_idx] = arg
            if isinstance(arg, FlowData):
                # connect FlowData with me
                self._impl.set_input(input_idx, arg._impl, 0)
            elif isinstance(arg, FlowOutput):
                # connect FlowOutput with me
                if arg.node == self:
                    raise utils.DfException(
                        f"The {idx}th input is the output of current flow node, so cannot be used as an input"
                    )
                self._impl.set_input(input_idx, arg.node._impl, arg.index)
            else:
                raise utils.DfException(
                    f"Argument inputs must be {Union[FlowData, FlowOutput]}"
                )

        if len(output_indexes) == 0:
            return None
        elif len(output_indexes) == 1:
            output_idx = output_indexes[0]
            output = FlowOutput(self, output_idx)
            self._output_anchors[output_idx] = output
            return output
        else:
            output_anchors = []
            for output_idx in output_indexes:
                flow_output = FlowOutput(self, output_idx)
                self._output_anchors[output_idx] = flow_output
                output_anchors.append(flow_output)
            return tuple(output_anchors)


class FlowOutput(object):
    def __init__(self, node: FlowNode, index: int) -> None:
        self.node = node
        self.index = index


class FlowInfo(object):
    MAX_USER_DATA_SIZE = 64

    def __init__(
        self,
        start_time: int = 0,
        end_time: int = 0,
        flow_flags: int = 0,
        transaction_id: int = 0,
    ) -> None:
        self._start_time: int = start_time
        self._end_time: int = end_time
        self._flow_flags: int = flow_flags
        self._transaction_id: int = transaction_id
        self._user_data: bytearray = bytearray(FlowInfo.MAX_USER_DATA_SIZE)
        self._data_size: int = 0

    def __str__(self):
        return (
            "FlowInfo(start_time="
            + str(self._start_time)
            + ", end_time="
            + str(self._end_time)
            + ", flow_flags="
            + str(self._flow_flags)
            + ", transaction_id="
            + str(self._transaction_id)
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def user_data(self):
        return self._user_data

    @property
    def data_size(self):
        return self._data_size

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, new_value):
        self._start_time = new_value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, new_value: int):
        self._end_time = new_value

    @property
    def flow_flags(self):
        return self._flow_flags

    @flow_flags.setter
    def flow_flags(self, new_value: int):
        self._flow_flags = new_value

    @property
    def transaction_id(self):
        return self._transaction_id

    @transaction_id.setter
    def transaction_id(self, new_value: int):
        self._transaction_id = new_value

    @staticmethod
    def _check_params_for_user_data(size: int, offset: int) -> bool:
        if size == 0:
            raise utils.DfException(
                f"The size is 0 or empty. It should in (0, {FlowInfo.MAX_USER_DATA_SIZE}]"
            )
        if (
            offset >= FlowInfo.MAX_USER_DATA_SIZE
            or FlowInfo.MAX_USER_DATA_SIZE - offset < size
        ):
            raise utils.DfException(
                f"Offset {offset} add userData size {size} can not be greater "
                f"than {FlowInfo.MAX_USER_DATA_SIZE}."
            )
        return True

    def set_user_data(self, user_data: bytearray, offset: int = 0):
        if self._check_params_for_user_data(len(user_data), offset):
            self._user_data[offset : offset + len(user_data)] = user_data
            self._data_size = (
                len(user_data) + offset
                if self._data_size <= len(user_data) + offset
                else self._data_size
            )

    def get_user_data(self, size: int = 0, offset: int = 0) -> bytearray:
        if self._check_params_for_user_data(size, offset):
            return self._user_data[offset : offset + size]
        else:
            return None


class FlowGraph(object):
    @utils.assert_args_type(outputs=List[FlowOutput])
    def __init__(
        self,
        outputs: List[FlowOutput],
        graph_options: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        graphpp_builder_async=False,
    ) -> None:
        self._impl = None
        self._outputs = outputs
        self._graph_options = graph_options or {}
        with name_scope(name, "FlowGraph") as name:
            self.name = name
        self._graphpp_builder_async = graphpp_builder_async
        self._out_nodes_idxes = {}
        self._output_idx_mapping = []
        self._input_datas = []
        self._graph_mode_init()

    def _graph_mode_init(self):
        self._extract_output_nodes()

        self._extract_input_datas()

        # make sure ge and session are initialized
        global _global_context
        if _global_context is None:
            with _global_context_lock:
                if _global_context is None:
                    self._init_ge()

        # create flow graph
        self._impl = dwrapper.FlowGraph(self.name)
        self._impl.set_graphpp_builder_async(self._graphpp_builder_async)
        self._impl.set_inputs([input_data._impl for input_data in self._input_datas])
        self._impl.set_outputs(
            [
                (node_2_indexes[0]._impl, node_2_indexes[1])
                for node_2_indexes in self._out_nodes_idxes.values()
            ]
        )

        # gen graph id
        def _gen_graph_id():
            global _global_graph_id_seed
            with _global_graph_id_seed_lock:
                graph_id = _global_graph_id_seed
                _global_graph_id_seed += 1
            return graph_id

        self._graph_id = _gen_graph_id()

    def _add_graph(self):
        if self._impl is not None:
            # add graph to session
            result = _global_context.session.add_flow_graph(
                self._graph_id, self._impl, self._graph_options
            )
            if result.ret_code != 0:
                raise utils.DfException(result.error_msg, result.ret_code)
            log.info(
                f"The flow graph(graph_id={self._graph_id}, graph_name={self.name}, "
                f"graph_options={self._graph_options} is added successfully."
            )
            # release flow model impl after add graph
            self._impl = None

    def _extract_output_nodes(self):
        """
        extract output FlowNode as per FlowOutput defined by user
        meanwhile check each FlowNode's all outputs should be all defined by user
        eg.

            flowdata --> node(a) --> a_out0
                    |
                     ---> node(b) --> b_out0
                                |---> b_out1
        user define outputs are:  [b_out0, a_out0, b_out1]
        output FlowNode will be [node(b), node(a)] after extracting
        """
        out_nodes_idxes = {}
        out_nodes_name_to_index = {}
        for output in self._outputs:
            if out_nodes_idxes.get(output.node.name, None) is None:
                out_nodes_idxes[output.node.name] = (output.node, [output.index])
                out_nodes_name_to_index[output.node.name] = [output.index]
            else:
                _, indexes = out_nodes_idxes[output.node.name]
                indexes.append(output.index)
                indexes_for_mapping = out_nodes_name_to_index[output.node.name]
                indexes_for_mapping.append(output.index)

        self._out_nodes_idxes = out_nodes_idxes
        self._out_nodes_name_to_index = out_nodes_name_to_index
        self._map_outputs_index()

    def _map_outputs_index(self):
        """
        after extracting output FlowNode, map sequence of user defined outputs with sequence of output FlowNodes
        eg.

            flowdata --> node(a) --> a_out0
                   |
                    ---> node(b) --> b_out0
                                |---> b_out1
        user define outputs is:  [b_out0, a_out0, b_out1]
        output FlowNode will be [node(b), node(a)] after extracting, the flowgraph output sequence will be
        [b_out0, b_out1, a_out0]. output index will be [0, 2, 1]
        in case user define outputs is:  [b_out1, a_out0], output index will be [0, 1] instead of [1, 2]
        """
        node_2_seq = {}
        seq = 0
        for output in self._outputs:
            if output.node.name not in node_2_seq:
                node_2_seq[output.node.name] = seq
                seq += len(self._out_nodes_idxes[output.node.name][1])
                indexes = self._out_nodes_name_to_index[output.node.name]
                indexes.sort()
                _, indexes_out = self._out_nodes_idxes[output.node.name]
                indexes_out.sort()
        self._output_idx_mapping = []
        for output in self._outputs:
            indexes = self._out_nodes_name_to_index[output.node.name]
            """
            if same node set for dataflow graph output. eg.
            flowdata --> node(a) --> a_out0
                   |
                    ---> node(a) --> a_out0
            indexes=[0 , 0]
            indexes.index(output.index)==> indexes.index(0) will always get position 0
            so setting indexes[pos] = -1 to mark already found
            """
            pos = indexes.index(output.index)
            self._output_idx_mapping.append(pos + node_2_seq[output.node.name])
            indexes[pos] = -1

    def _extract_input_datas(self):
        """
        extract FlowData input as per FlowOutput defined by user
        meanwhile check each FlowData's index should be start from 0 and continugous
        """
        # extract input FlowData
        checked_nodes = set()
        need_check_nodes = []
        input_datas = set()
        for _, output in enumerate(self._outputs):
            need_check_nodes.append(output.node)

        while len(need_check_nodes) > 0:
            node = need_check_nodes.pop()
            if node.name in checked_nodes:
                # skip this node
                continue
            checked_nodes.add(node.name)
            for anchor in node._input_anchors:
                if isinstance(anchor, FlowData):
                    input_datas.add(anchor)
                else:
                    need_check_nodes.append(anchor.node)
        self._input_datas = list(input_datas)
        # sort by flow data name
        self._input_datas.sort(key=lambda e: e.name)
        for i, input_data in enumerate(self._input_datas):
            input_data._impl.set_attr("index", i)

    def _init_ge(self):
        """
        initialize ge and session, data flow
        """
        global _global_context, _global_options
        if _global_options is None:
            raise utils.DfException("Should invoke init first.", dwrapper.NOT_INIT)
        _global_context = _Context(_global_options)

    def _convert_feed_data_to_tensor(self, input_data: FlowData, feed_data: Any):
        if isinstance(feed_data, Tensor):
            if input_data._schema and (feed_data._tensor_desc != input_data._schema):
                return Tensor(feed_data, tensor_desc=input_data._schema)
            else:
                return feed_data
        else:
            return Tensor(feed_data, tensor_desc=input_data._schema)

    def _convert_object_to_flow_msg(self, input_data: FlowData, feed_data: Any):
        serialized_data = None
        if isinstance(feed_data, Tensor):
            if input_data._schema and (feed_data._tensor_desc != input_data._schema):
                raise utils.DfException(
                    "The desc of the tensor is different from "
                    f"that of the data:{input_data.name}."
                )
            else:
                if feed_data._flow_msg is not None:
                    # from df.alloc_tensor
                    return (feed_data._flow_msg, serialized_data)
                else:
                    return (
                        dwrapper.FlowBufferFactory.to_tensor_flow_msg(feed_data._impl),
                        serialized_data,
                    )
        elif feed_data is None:
            return (
                dwrapper.FlowBufferFactory.alloc_empty_data_msg(MSG_TYPE_TENSOR_DATA),
                serialized_data,
            )
        else:
            msg_type = msg_type_register.get_msg_type(type(feed_data))
            if msg_type is None:
                msg_type = MSG_TYPE_PICKLED_MSG
            serialize_func = msg_type_register.get_serialize_func(msg_type)
            serialized_data = serialize_func(feed_data)
            flow_msg = dwrapper.FlowBufferFactory.to_raw_data_flow_msg(serialized_data)
            flow_msg.set_msg_type(msg_type)
            return (flow_msg, serialized_data)

    def _check_flow_msg(self, flow_msg):
        if flow_msg.get_ret_code() != 0:
            log.error("model execute failed, ret = %d", flow_msg.get_ret_code())
            return flow_msg.get_ret_code()
        if (
            flow_msg.get_msg_type() != MSG_TYPE_TENSOR_DATA
            and int(flow_msg.get_msg_type()) != 1023
            and int(flow_msg.get_msg_type()) < MSG_TYPE_USER_DEFINE_START
        ):
            log.error("invalid flow msg type:%d", int(flow_msg.get_msg_type()))
            return dwrapper.PARAM_INVALID
        if int(
            flow_msg.get_msg_type()
        ) >= MSG_TYPE_USER_DEFINE_START and not msg_type_register.registered(
            flow_msg.get_msg_type()
        ):
            log.error(
                "user flow msg type:%d is not registered", int(flow_msg.get_msg_type())
            )
            return dwrapper.PARAM_INVALID
        return 0

    def _convert_flow_msg_to_object(self, flow_msg):
        msg_type = flow_msg.get_msg_type()
        if msg_type == MSG_TYPE_TENSOR_DATA:
            if flow_msg.get_tensor() is None:
                return None
            else:
                return Tensor(flow_msg)
        elif msg_type_register.registered(msg_type):
            deserialize_func = msg_type_register.get_deserialize_func(msg_type)
            return deserialize_func(flow_msg.get_raw_data())
        else:
            raise utils.DfException(f"msg type:{msg_type} is not registered")

    def set_contains_n_mapping_node(self, contains_n_mapping_node: bool):
        if self._impl is not None:
            self._impl.set_contains_n_mapping_node(contains_n_mapping_node)
        else:
            raise utils.DfException(
                "can not call set_contains_n_mapping_node after feed data"
            )

    def set_inputs_align_attrs(
        self,
        align_max_cache_num: int,
        align_timeout: int,
        dropout_when_not_align: bool = False,
    ):
        if self._impl is not None:
            if align_max_cache_num < 0 or align_max_cache_num > 1024:
                raise utils.DfException(
                    f"align_max_cache_num={align_max_cache_num} is out of range [0, 1024]"
                )
            if align_timeout < -1 or align_timeout == 0 or align_timeout > 600 * 1000:
                raise utils.DfException(
                    f"align_timeout={align_timeout} must be -1 or in range(0, 600 * 1000]"
                )
            self._impl.set_inputs_align_attrs(
                align_max_cache_num, align_timeout, dropout_when_not_align
            )
        else:
            raise utils.DfException(
                "can not call set_inputs_align_attrs after feed data"
            )

    def set_exception_catch(self, enable_exception_catch: bool = False):
        if self._impl is not None:
            self._impl.set_exception_catch(enable_exception_catch)
        else:
            raise utils.DfException("can not call set_exception_catch after feed data")

    def feed_data(
        self,
        feed_dict: Dict[FlowData, Any],
        flow_info: Optional[FlowInfo] = None,
        timeout: Optional[int] = -1,
        partial_inputs: Optional[bool] = False,
    ) -> int:
        df_flow_info = dwrapper.FlowInfo()
        if flow_info is not None:
            if isinstance(flow_info, FlowInfo):
                df_flow_info.start_time = flow_info.start_time
                df_flow_info.end_time = flow_info.end_time
                df_flow_info.flow_flags = flow_info.flow_flags
                df_flow_info.transaction_id = flow_info.transaction_id
                if flow_info.data_size != 0:
                    df_flow_info.set_user_data(
                        flow_info.user_data, flow_info.data_size, 0
                    )
            else:
                log.error(f"Argument flow_info must be {FlowInfo}")
                return dwrapper.PARAM_INVALID
        # ensure graph is added
        self._add_graph()
        indexes = []
        inputs = []
        if not feed_dict:
            if (
                df_flow_info.flow_flags & FlowFlag.DATA_FLOW_FLAG_EOS
                != FlowFlag.DATA_FLOW_FLAG_EOS
            ):
                log.error(
                    f"feed_dict is empty but flow_flags={df_flow_info.flow_flags} "
                    f"is not with eos flag={FlowFlag.DATA_FLOW_FLAG_EOS}"
                )
                return dwrapper.PARAM_INVALID

            log.info("Feed eos empty data.")
            result = _global_context.session.feed_data(
                self._graph_id, indexes, inputs, df_flow_info, timeout
            )
            if result.ret_code != 0:
                log.error("failed to feed data, error msg = %s", result.error_msg)
            return result.ret_code

        for i, input_data in enumerate(self._input_datas):
            if input_data not in feed_dict:
                if not partial_inputs:
                    log.error(
                        f"Missing input {input_data.name} in feed_dict {[key.name for key in feed_dict.keys()]}"
                    )
                    return dwrapper.PARAM_INVALID
                else:
                    continue

            try:
                inputs.append(
                    self._convert_feed_data_to_tensor(
                        input_data, feed_dict[input_data]
                    )._impl
                )
                indexes.append(i)
            except utils.DfException as e:
                log.error("failed to convert data to tensor, error msg = %s", e.message)
                return e.error_code

        if not inputs:
            log.error(
                f"no inputs {[data.name for data in self._input_datas]} found in "
                f"feed_dict {[key.name for key in feed_dict.keys()]}"
            )
            return dwrapper.PARAM_INVALID

        result = _global_context.session.feed_data(
            self._graph_id, indexes, inputs, df_flow_info, timeout
        )
        return result.ret_code

    def fetch_data(
        self, indexes: Optional[List[int]] = None, timeout: Optional[int] = -1
    ) -> Tuple[List[Any], FlowInfo, int]:
        flow_info = FlowInfo()
        outputs = []
        if indexes is None:
            ret = _global_context.session.fetch_data(
                self._graph_id, self._output_idx_mapping, timeout, flow_info.user_data
            )
        else:
            convert_idx = []
            for idx in indexes:
                if idx > (len(self._output_idx_mapping) - 1):
                    log.error(
                        f"The value of indexes should in [0, {len(self._output_idx_mapping)})."
                    )
                    return (outputs, flow_info, dwrapper.PARAM_INVALID)
                convert_idx.append(self._output_idx_mapping[idx])
            ret = _global_context.session.fetch_data(
                self._graph_id, convert_idx, timeout, flow_info.user_data
            )
        if len(ret) != 3:
            raise utils.DfException(
                "Failed to fetch data. Return result length must be 3. "
                "for details about the error information, see the ascend log.",
                dwrapper.INNER_ERROR,
            )
        flow_info.start_time = ret[2].start_time
        flow_info.end_time = ret[2].end_time
        flow_info.flow_flags = ret[2].flow_flags
        flow_info.transaction_id = ret[2].transaction_id
        if ret[0].ret_code != 0 and ret[0].ret_code != dwrapper.SUBHEALTHY:
            log.error("failed to fetch data, error msg = %s", ret[0].error_msg)
            return (outputs, flow_info, ret[0].ret_code)

        for output in ret[1]:
            outputs.append(Tensor(output))
        return (outputs, flow_info, ret[0].ret_code)

    def feed(
        self,
        feed_dict: Dict[FlowData, Any],
        timeout: Optional[int] = -1,
        partial_inputs: Optional[bool] = False,
    ) -> int:
        # ensure graph is added
        self._add_graph()
        indexes = []
        inputs = []
        serialized_inputs = []

        for i, input_data in enumerate(self._input_datas):
            if input_data not in feed_dict:
                if not partial_inputs:
                    log.error(
                        f"Missing input {input_data.name} in feed_dict {[key.name for key in feed_dict.keys()]}"
                    )
                    return dwrapper.PARAM_INVALID
                else:
                    continue

            try:
                convert_ret = self._convert_object_to_flow_msg(
                    input_data, feed_dict[input_data]
                )
                inputs.append(convert_ret[0])
                serialized_inputs.append(convert_ret[1])
                indexes.append(i)
            except utils.DfException as e:
                log.error(
                    "failed to convert data to flow msg, error msg = %s", e.message
                )
                return e.error_code

        if not inputs:
            log.error(
                f"no inputs {[data.name for data in self._input_datas]} found in "
                f"feed_dict {[key.name for key in feed_dict.keys()]}"
            )
            return dwrapper.PARAM_INVALID

        result = _global_context.session.feed_flow_msg(
            self._graph_id, indexes, inputs, timeout
        )
        return result.ret_code

    def fetch(
        self, indexes: Optional[List[int]] = None, timeout: Optional[int] = -1
    ) -> Tuple[List[Any], int]:
        flow_info = FlowInfo()
        outputs = []
        if indexes is None:
            ret = _global_context.session.fetch_flow_msg(
                self._graph_id, self._output_idx_mapping, timeout
            )
        else:
            convert_idx = []
            for idx in indexes:
                if idx > (len(self._output_idx_mapping) - 1):
                    log.error(
                        f"The value of indexes should in [0, {len(self._output_idx_mapping)})."
                    )
                    return (outputs, dwrapper.PARAM_INVALID)
                convert_idx.append(self._output_idx_mapping[idx])
            ret = _global_context.session.fetch_flow_msg(
                self._graph_id, convert_idx, timeout
            )
        if len(ret) != 2:
            raise utils.DfException(
                "Failed to fetch flow msg. Return result length must be 2. "
                "for details about the error information, see the ascend log.",
                dwrapper.INNER_ERROR,
            )
        if ret[0].ret_code != 0 and ret[0].ret_code != dwrapper.SUBHEALTHY:
            log.error("failed to fetch flow msg, error msg = %s", ret[0].error_msg)

        for output in ret[1]:
            check_ret = self._check_flow_msg(output)
            if check_ret != 0:
                log.error("failed to check output flow msg")
                return ([], dwrapper.PARAM_INVALID)
            else:
                output_object = self._convert_flow_msg_to_object(output)
                outputs.append(output_object)
        return (outputs, ret[0].ret_code)
