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

from enum import Enum
import threading
import functools
import traceback
import logging
from typing import Any, Optional, Dict, Union, List, Tuple
import queue
import numpy as np
import dataflow.dataflow as df
import dataflow.data_type as dt
import dataflow.utils.utils as utils
from dataflow.flow_func import flowfunc_wrapper as fw
from dataflow import data_wrapper
from dataflow import dflow_wrapper as dw

fw.init_func_datatype_manager(
    {
        data_wrapper.FuncDataType.DT_FLOAT: np.array([], np.float32),
        data_wrapper.FuncDataType.DT_FLOAT16: np.array([], np.float16),
        data_wrapper.FuncDataType.DT_INT8: np.array([], np.int8),
        data_wrapper.FuncDataType.DT_INT16: np.array([], np.int16),
        data_wrapper.FuncDataType.DT_UINT16: np.array([], np.uint16),
        data_wrapper.FuncDataType.DT_UINT8: np.array([], np.uint8),
        data_wrapper.FuncDataType.DT_INT32: np.array([], np.int32),
        data_wrapper.FuncDataType.DT_INT64: np.array([], np.int64),
        data_wrapper.FuncDataType.DT_UINT32: np.array([], np.uint32),
        data_wrapper.FuncDataType.DT_UINT64: np.array([], np.uint64),
        data_wrapper.FuncDataType.DT_BOOL: np.array([], np.bool_),
        data_wrapper.FuncDataType.DT_DOUBLE: np.array([], np.double),
    }
)

FLOW_FUNC_SUCCESS = fw.FLOW_FUNC_SUCCESS
FLOW_FUNC_FAILED = fw.FLOW_FUNC_FAILED
FLOW_FUNC_ERR_PARAM_INVALID = fw.FLOW_FUNC_ERR_PARAM_INVALID
FLOW_FUNC_ERR_ATTR_NOT_EXITS = fw.FLOW_FUNC_ERR_ATTR_NOT_EXITS
FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH = fw.FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH
FLOW_FUNC_ERR_TIME_OUT_ERROR = fw.FLOW_FUNC_ERR_TIME_OUT_ERROR
FLOW_FUNC_STATUS_REDEPLOYING = fw.FLOW_FUNC_STATUS_REDEPLOYING
FLOW_FUNC_STATUS_EXIT = fw.FLOW_FUNC_STATUS_EXIT
FLOW_FUNC_ERR_DRV_ERROR = fw.FLOW_FUNC_ERR_DRV_ERROR
FLOW_FUNC_ERR_QUEUE_ERROR = fw.FLOW_FUNC_ERR_QUEUE_ERROR
FLOW_FUNC_ERR_MEM_BUF_ERROR = fw.FLOW_FUNC_ERR_MEM_BUF_ERROR
FLOW_FUNC_ERR_EVENT_ERROR = fw.FLOW_FUNC_ERR_EVENT_ERROR
FLOW_FUNC_ERR_USER_DEFINE_START = fw.FLOW_FUNC_ERR_USER_DEFINE_START
FLOW_FUNC_ERR_USER_DEFINE_END = fw.FLOW_FUNC_ERR_USER_DEFINE_END
MAX_USER_DATA_SIZE = 64

# flow msg type
MSG_TYPE_TENSOR_DATA = fw.MsgType.MSG_TYPE_TENSOR_DATA
MSG_TYPE_RAW_MSG = fw.MsgType.MSG_TYPE_RAW_MSG
MSG_TYPE_TORCH_TENSOR_MSG = fw.MsgType.MSG_TYPE_TORCH_TENSOR_MSG
MSG_TYPE_PICKLED_MSG = fw.MsgType.MSG_TYPE_PICKLED_MSG
MSG_TYPE_USER_DEFINE_START = fw.MsgType.MSG_TYPE_USER_DEFINE_START
# flow flag type
FLOW_FLAG_EOS = fw.FlowFlag.FLOW_FLAG_EOS
FLOW_FLAG_SEG = fw.FlowFlag.FLOW_FLAG_SEG
# log type
DEBUG_LOG = fw.FlowFuncLogType.DEBUG_LOG
RUN_LOG = fw.FlowFuncLogType.RUN_LOG
# log level
DEBUG = fw.FlowFuncLogLevel.DEBUG
INFO = fw.FlowFuncLogLevel.INFO
WARN = fw.FlowFuncLogLevel.WARN
ERROR = fw.FlowFuncLogLevel.ERROR


class FlowFuncLogger:
    def __init__(self) -> None:
        self._impl = fw.FlowFuncLogger()

    @staticmethod
    def _get_caller_info() -> str:
        frame = logging.currentframe()
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        func_name = frame.f_code.co_name
        location_message = ("[{}:{}][{}]").format(filename, line_number, func_name)
        return location_message

    def is_log_enable(
        self, log_type: fw.FlowFuncLogType, log_level: fw.FlowFuncLogLevel
    ) -> bool:
        if not isinstance(log_type, fw.FlowFuncLogType):
            return False
        if not isinstance(log_level, fw.FlowFuncLogLevel):
            return False
        return self._impl.is_log_enable(log_type, log_level)

    def get_log_header(self) -> str:
        return self._impl.get_log_header()

    def debug(self, message: str, *args: tuple) -> None:
        if not self.is_log_enable(DEBUG_LOG, DEBUG):
            return
        try:
            user_message = message % args
            self._impl.debug_log_debug(self._get_caller_info(), user_message)
        except BaseException as e:
            self._impl.debug_log_debug(self._get_caller_info(), message)

    def error(self, message: str, *args: tuple) -> None:
        if not self.is_log_enable(DEBUG_LOG, ERROR):
            return
        try:
            user_message = message % args
            self._impl.debug_log_error(self._get_caller_info(), user_message)
        except BaseException as e:
            self._impl.debug_log_error(self._get_caller_info(), message)

    def info(self, message: str, *args: tuple) -> None:
        if not self.is_log_enable(DEBUG_LOG, INFO):
            return
        try:
            user_message = message % args
            self._impl.debug_log_info(self._get_caller_info(), user_message)
        except BaseException as e:
            self._impl.debug_log_info(self._get_caller_info(), message)

    def warn(self, message: str, *args: tuple) -> None:
        if not self.is_log_enable(DEBUG_LOG, WARN):
            return
        try:
            user_message = message % args
            self._impl.debug_log_warn(self._get_caller_info(), user_message)
        except BaseException as e:
            self._impl.debug_log_warn(self._get_caller_info(), message)

    def run_error(self, message: str, *args: tuple) -> None:
        if not self.is_log_enable(RUN_LOG, WARN):
            return
        try:
            user_message = message % args
            self._impl.run_log_error(self._get_caller_info(), user_message)
        except BaseException as e:
            self._impl.run_log_error(self._get_caller_info(), message)

    def run_info(self, message: str, *args: tuple) -> None:
        if not self.is_log_enable(RUN_LOG, INFO):
            return
        try:
            user_message = message % args
            self._impl.run_log_info(self._get_caller_info(), user_message)
        except BaseException as e:
            self._impl.run_log_info(self._get_caller_info(), message)


logger = FlowFuncLogger()


class FlowMsg:
    def __init__(self, flow_msg: Union[fw.FlowMsg, dw.FlowMsg]) -> None:
        self._flow_msg = flow_msg

    def __repr__(self):
        return self._flow_msg.__repr__()

    def get_msg_type(self) -> fw.MsgType:
        return self._flow_msg.get_msg_type()

    def set_msg_type(self, msg_type) -> None:
        return self._flow_msg.set_msg_type(msg_type)

    def get_tensor(self):
        return (
            None
            if self._flow_msg.get_tensor() is None
            else df.Tensor(self._flow_msg.get_tensor())
        )

    def get_raw_data(self):
        return self._flow_msg.get_raw_data()

    def get_ret_code(self) -> int:
        return self._flow_msg.get_ret_code()

    def set_ret_code(self, ret_code) -> None:
        return self._flow_msg.set_ret_code(ret_code)

    def get_start_time(self) -> int:
        return self._flow_msg.get_start_time()

    def set_start_time(self, start_time: int) -> None:
        return self._flow_msg.set_start_time(start_time)

    def get_end_time(self) -> int:
        return self._flow_msg.get_end_time()

    def set_end_time(self, end_time: int):
        return self._flow_msg.set_end_time(end_time)

    def get_flow_flags(self) -> int:
        return self._flow_msg.get_flow_flags()

    def set_flow_flags(self, flow_flags: int) -> None:
        return self._flow_msg.set_flow_flags(flow_flags)

    def set_route_label(self, route_label) -> None:
        return self._flow_msg.set_route_label(route_label)

    def get_transaction_id(self):
        return self._flow_msg.get_transaction_id()

    def set_transaction_id(self, transaction_id) -> None:
        return self._flow_msg.set_transaction_id(transaction_id)


class FlowMsgQueue(queue.Queue):
    def __init__(self, flow_msg_queue: fw.FlowMsgQueue) -> None:
        super().__init__()
        self.mutex = threading.Lock()
        self._flow_msg_queue = flow_msg_queue
        self.maxsize = self._flow_msg_queue.depth()

    def task_done(self):
        raise NotImplementedError("This method is not allowed.")

    def join(self):
        raise NotImplementedError("This method is not allowed.")

    def qsize(self):
        with self.mutex:
            return self._flow_msg_queue.size()

    def empty(self):
        with self.mutex:
            return not self._flow_msg_queue.size()

    def full(self):
        with self.mutex:
            return 0 < self.maxsize <= self._flow_msg_queue.size()

    def put(self, item, block=True, timeout=None):
        raise NotImplementedError("This method is not allowed.")

    def get(self, block=True, timeout=None):
        if not block:
            ret = self._flow_msg_queue.dequeue(0)
            if ret[0] == FLOW_FUNC_SUCCESS:
                if int(ret[1].get_msg_type()) == int(MSG_TYPE_TENSOR_DATA):
                    return None if ret[1].get_tensor() is None else df.Tensor(ret[1])
                return utils.convert_flow_msg_to_object(FlowMsg(ret[1]))
            raise queue.Empty
        elif timeout is None:
            return self._get_with_timeout(-1)
        elif timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        else:
            return self._get_with_timeout(timeout)

    def put_nowait(self, item):
        raise NotImplementedError("This method is not allowed.")

    def get_nowait(self):
        return self.get(block=False)

    def _get_with_timeout(self, timeout):
        ret = self._flow_msg_queue.dequeue(timeout)
        if ret[0] == FLOW_FUNC_STATUS_REDEPLOYING:
            raise utils.DfAbortException("Redeploying, schedule aborted!", ret[0])
        elif ret[0] == FLOW_FUNC_STATUS_EXIT:
            raise utils.DfAbortException("Process exiting, schedule aborted!", ret[0])
        elif ret[0] == FLOW_FUNC_ERR_TIME_OUT_ERROR:
            raise queue.Empty
        elif ret[0] == FLOW_FUNC_SUCCESS:
            if int(ret[1].get_msg_type()) == int(MSG_TYPE_TENSOR_DATA):
                return None if ret[1].get_tensor() is None else df.Tensor(ret[1])
            return utils.convert_flow_msg_to_object(FlowMsg(ret[1]))
        else:
            raise utils.DfException("Queue get item failed", ret[0])


class PyMetaParams:
    def __init__(self, meta_params: fw.MetaParams) -> None:
        self._meta_params = meta_params

    def __repr__(self):
        return self._meta_params.__repr__()

    def get_name(self) -> str:
        return self._meta_params.get_name()

    def get_attr_int(self, name: str) -> Tuple[int, int]:
        return self._meta_params.get_int64(name)

    def get_attr_int_list(self, name: str) -> Tuple[int, List[int]]:
        return self._meta_params.get_int64_vector(name)

    def get_attr_int_list_list(self, name: str) -> Tuple[int, List[List[int]]]:
        return self._meta_params.get_int64_vector_vector(name)

    def get_attr_bool(self, name: str) -> Tuple[int, bool]:
        return self._meta_params.get_bool(name)

    def get_attr_bool_list(self, name: str) -> Tuple[int, List[bool]]:
        return self._meta_params.get_bool_list(name)

    def get_attr_float(self, name: str) -> Tuple[int, float]:
        return self._meta_params.get_float(name)

    def get_attr_float_list(self, name: str) -> Tuple[int, List[float]]:
        return self._meta_params.get_float_list(name)

    def get_attr_tensor_dtype(self, name: str):
        dtype = self._meta_params.get_tensor_dtype(name)
        if dtype[0] == fw.FLOW_FUNC_SUCCESS:
            wrapper_type = dt.get_python_dtype_from_dwrapper_dtype(dtype[1])
            return (
                fw.FLOW_FUNC_SUCCESS,
                dt._dflow_dtype_to_np_dtype.get(wrapper_type, None),
            )
        else:
            return (fw.FLOW_FUNC_ERR_ATTR_NOT_EXITS, None)

    def get_attr_tensor_dtype_list(self, name: str):
        dtype_list = self._meta_params.get_tensor_dtype_list(name)
        result = []
        if dtype_list[0] == fw.FLOW_FUNC_SUCCESS:
            for i in dtype_list[1]:
                wrapper_type = dt.get_python_dtype_from_dwrapper_dtype(i)
                result.append(dt._dflow_dtype_to_np_dtype.get(wrapper_type, None))
            return (fw.FLOW_FUNC_SUCCESS, result)
        else:
            return (fw.FLOW_FUNC_ERR_ATTR_NOT_EXITS, result)

    def get_attr_str(self, name: str) -> Tuple[int, str]:
        return self._meta_params.get_string(name)

    def get_attr_str_list(self, name: str) -> Tuple[int, List[str]]:
        return self._meta_params.get_string_list(name)

    def get_input_num(self) -> int:
        return self._meta_params.get_input_number()

    def get_output_num(self) -> int:
        return self._meta_params.get_output_number()

    def get_work_path(self) -> str:
        return self._meta_params.get_work_path()

    def get_running_device_id(self) -> int:
        return self._meta_params.get_running_device_id()
    
    def get_running_instance_id(self) -> int:
        return self._meta_params.get_running_instance_id()
    
    def get_running_instance_num(self) -> int:
        return self._meta_params.get_running_instance_num()


class AffinityPolicy(Enum):
    NO_AFFINITY = fw.AffinityPolicy.NO_AFFINITY
    ROW_AFFINITY = fw.AffinityPolicy.ROW_AFFINITY
    COL_AFFINITY = fw.AffinityPolicy.COL_AFFINITY

    def __init__(self, inner_type):
        self.inner_type = inner_type


class BalanceConfig:
    def __init__(
        self,
        row_num: int,
        col_num: int,
        affinity_policy: AffinityPolicy = AffinityPolicy.NO_AFFINITY,
    ) -> None:
        self._impl = fw.BalanceConfig()
        self._impl.set_balance_weight(row_num, col_num)
        self._impl.set_affinity_policy(affinity_policy.inner_type)

    def set_data_pos(self, pos: List[Tuple[int, int]]):
        self._impl.set_data_pos(pos)

    def get_inner_config(self):
        return self._impl


class MetaRunContext:
    def __init__(self, run_context: fw.MetaRunContext) -> None:
        self._context = run_context
        self._user_data = bytes(MAX_USER_DATA_SIZE)

    def __repr__(self):
        return self._context.__repr__()

    def alloc_tensor_msg(
        self, shape: Union[List[int], Tuple[int]], dtype, align: Optional[int] = 64
    ) -> FlowMsg:
        if isinstance(dtype, dt.DType):
            dtype = dtype.dtype
        elif dtype in df._support_np_dtype:
            dtype = dt._np_dtype_to_dflow_dtype.get(np.dtype(dtype)).dtype
        else:
            raise TypeError(f"The dtype of data is {dtype} is invalid.")
        return FlowMsg(self._context.alloc_tensor_msg(shape, dtype, align))

    def alloc_raw_data_msg(self, size, align: Optional[int] = 64) -> FlowMsg:
        return FlowMsg(self._context.alloc_raw_data_msg(size, align))

    def to_flow_msg(self, tensor):
        return FlowMsg(self._context.to_flow_msg(tensor._impl))

    def set_output(
        self,
        index,
        output: Union[FlowMsg, np.ndarray, fw.FlowMsg],
        balance_config: BalanceConfig = None,
    ) -> int:
        if isinstance(output, FlowMsg):
            output = output._flow_msg
        elif isinstance(output, np.ndarray):
            output_b = output
            output = self.alloc_tensor_msg(
                output.shape,
                dt._np_dtype_to_dflow_dtype.get(np.dtype(output.dtype), None),
            )
            out_np = output.get_tensor().numpy()
            out_np[...] = output_b
            output = output._flow_msg

        if balance_config is None:
            return self._context.set_output(index, output)
        else:
            return self._context.set_output(
                index, output, balance_config.get_inner_config()
            )

    def set_multi_outputs(
        self,
        index,
        outputs: List[Union[FlowMsg, np.ndarray, fw.FlowMsg]],
        balance_config: BalanceConfig,
    ) -> int:
        output_msgs = [None] * len(outputs)
        for i, output in enumerate(outputs):
            if isinstance(output, fw.FlowMsg):
                output_msgs[i] = output
            elif isinstance(output, FlowMsg):
                output_msgs[i] = output._flow_msg
            elif isinstance(output, np.ndarray):
                output_b = output
                output = self.alloc_tensor_msg(
                    output.shape,
                    dt._np_dtype_to_dflow_dtype.get(np.dtype(output.dtype), None),
                )
                out_np = output.get_tensor().numpy()
                out_np[...] = output_b
                output_msgs[i] = output._flow_msg
            else:
                raise TypeError(
                    f"The outputs:{i} type { type(output) } is unsupported."
                )
        return self._context.set_multi_outputs(
            index, output_msgs, balance_config.get_inner_config()
        )

    def alloc_empty_data_msg(self) -> FlowMsg:
        return FlowMsg(self._context.alloc_empty_msg(fw.MsgType.MSG_TYPE_TENSOR_DATA))

    def run_flow_model(
        self, model_key: str, input_msgs: List[FlowMsg], timeout: int
    ) -> Tuple[int, List[FlowMsg]]:
        wrapper_flow_msg = [msg._flow_msg for msg in input_msgs]
        outputs = self._context.run_flow_model(model_key, wrapper_flow_msg, timeout)
        return outputs

    def get_user_data(self, size: int, offset: int = 0):
        ret = self._context.get_user_data(self._user_data, size, offset)
        if ret == fw.FLOW_FUNC_SUCCESS:
            return self._user_data
        else:
            return bytearray(MAX_USER_DATA_SIZE)

    def raise_exception(self, exception_code: int, user_context_id: int):
        return self._context.raise_exception(exception_code, user_context_id)

    def get_exception(self) -> tuple:
        r"""
        tuple(bool, exception_code, user_context_id)
        the first element in tuple : False(exception no existed). True(exception existed)
        """
        return self._context.get_exception()
