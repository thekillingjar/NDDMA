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
import copy
import traceback
from typing import Union, List
import numpy as np
from dataflow.flow_func.func_register import FlowFuncRegister, FlowFuncInfos
import dataflow.flow_func as ff
import dataflow.dataflow as df
import dataflow.utils.utils as utils
import dataflow.utils.log as log
import dataflow.data_type as dt
from dataflow.flow_func import flowfunc_wrapper as fw


_pyflow_support_args_ = [
    "num_returns",
    "resources",
    "env_hook_func",
    "visible_device_enable",
    "stream_input",
    "choice_output",
]


def _gen_func_params(
    input_num, output_num, base_input_num=0, base_output_num=0, stream_input=None
):
    func_list = ""
    for i in range(input_num):
        func_list += f"i{i + base_input_num}"
        func_list += ","

    for i in range(output_num):
        func_list += f"o{i + base_output_num}"
        if i < output_num - 1:
            func_list += ","
    if stream_input is not None:
        func_list += f",stream_input"
    return func_list


def _get_func_num_returns(func, options):
    typing_num_returns = utils.get_typing_num_returns(func)
    if "num_returns" in options:
        arg_num_returns = options["num_returns"]
        if typing_num_returns != -1:
            key_error_string = (
                f"The number:{typing_num_returns} of outputs specified by typing is different from "
                f"the number:{arg_num_returns} specified by num_returns."
            )
            assert arg_num_returns == typing_num_returns, key_error_string
        assert isinstance(arg_num_returns, int), "The 'num_returns' option must be an integer."
        assert arg_num_returns > 0, "The 'num_returns' option must be greater than 0."
        return arg_num_returns
    else:
        if typing_num_returns != -1:
            return typing_num_returns
        else:
            return 1


def _parse_option_resource(option, flow_func_infos):
    if "resources" in option:
        for key, value in option["resources"].items():
            if key == "num_cpus":
                key = "cpu"
            flow_func_infos.add_running_resources_info(key, value)


def _parse_option_env_hook_func(option, flow_func_infos):
    if "env_hook_func" in option:
        flow_func_infos.set_env_hook_func(option["env_hook_func"])


def _parse_option_visible_device_enable(option, flow_func_infos):
    if "visible_device_enable" in option:
        flow_func_infos.set_visible_device_enable(option["visible_device_enable"])


def _init_process_point(meta_params):
    py_meta_params = ff.PyMetaParams(meta_params)
    utils.set_running_device_id(py_meta_params.get_running_device_id())
    utils.set_running_instance_id(py_meta_params.get_running_instance_id())
    utils.set_running_instance_num(py_meta_params.get_running_instance_num())
    utils.set_running_in_udf()
    work_path = py_meta_params.get_work_path()


def _check_flow_msg(flow_msg):
    logger = ff.FlowFuncLogger()
    if flow_msg.get_ret_code() != ff.FLOW_FUNC_SUCCESS:
        logger.error("invalid input, ret_code=%d", flow_msg.get_ret_code())
        return ff.FLOW_FUNC_FAILED
    msg_type = flow_msg.get_msg_type()
    if msg_type in (ff.MSG_TYPE_TENSOR_DATA, ff.MSG_TYPE_RAW_MSG, ff.MSG_TYPE_TORCH_TENSOR_MSG):
        return ff.FLOW_FUNC_SUCCESS
    if msg_type < ff.MSG_TYPE_USER_DEFINE_START:
        logger.error("invalid flow msg type:%d", int(msg_type))
        return ff.FLOW_FUNC_FAILED
    return ff.FLOW_FUNC_SUCCESS


def _convert_object_to_flow_msg(run_context, py_object):
    if isinstance(py_object, df.Tensor):
        if isinstance(py_object._impl, fw.Tensor):
            try:
                return run_context.to_flow_msg(py_object)
            except BaseException as e:
                ff.logger.info(
                    "the alloc_tensor interface is recommended, "
                    "non-shared tensors cannot be directly converted to flow msg, e:%s. ",
                    e,
                )

        out_np = py_object.numpy()
        msg = run_context.alloc_tensor_msg(
            out_np.shape, dt._np_dtype_to_dflow_dtype.get(np.dtype(out_np.dtype), None)
        )
        msg_np = msg.get_tensor().numpy()
        msg_np[...] = out_np
        return msg
    elif py_object is None:
        return run_context.alloc_empty_data_msg()
    else:
        msg_type = utils.get_msg_type_register().get_msg_type(type(py_object))
        if msg_type is None:
            msg_type = ff.MSG_TYPE_PICKLED_MSG
        serialize_func = utils.get_msg_type_register().get_serialize_func(msg_type)
        serialize_buffer = serialize_func(py_object)
        msg = run_context.alloc_raw_data_msg(len(serialize_buffer))
        msg_mv = msg.get_raw_data()
        msg_mv[:] = serialize_buffer[:]
        msg.set_msg_type(msg_type)
        return msg


def _process_results(run_context, results, output_num, choice_output):
    logger = ff.FlowFuncLogger()
    output_msg_list = []
    if output_num > 1:
        if not isinstance(results, (tuple)):
            logger.error(
                "output num error, num_returns = %d, but return is not tuple.",
                output_num,
            )
            return (ff.FLOW_FUNC_FAILED, output_msg_list)
        elif len(results) != output_num:
            logger.error(
                "output num error, num_returns = %d, but real return num = %d.",
                len(results),
                output_num,
            )
            return (ff.FLOW_FUNC_FAILED, output_msg_list)
        else:
            for _, item in enumerate(results):
                if choice_output is None:
                    output_msg = _convert_object_to_flow_msg(run_context, item)
                else:
                    output_msg = (
                        _convert_object_to_flow_msg(run_context, item)
                        if choice_output(item)
                        else None
                    )
                output_msg_list.append(output_msg)
    else:
        if choice_output is None:
            output_msg = _convert_object_to_flow_msg(run_context, results)
        else:
            output_msg = (
                _convert_object_to_flow_msg(run_context, results)
                if choice_output(results)
                else None
            )
        output_msg_list.append(output_msg)
    return (ff.FLOW_FUNC_SUCCESS, output_msg_list)


def method(*args, **kwargs):
    valid_kwargs = [
        "",
        "num_returns",
        "choice_output",
        "stream_input",
    ]
    error_string = (
        f"Unexpected args argument to @df.method, args='{args}', please use kwargs"
        f"the arguments in the list {valid_kwargs}, for example "
        "'@df.method() or df.method(num_returns=2)'."
    )
    assert len(args) == 0, error_string
    for key in kwargs:
        key_error_string = (
            f"Unexpected keyword argument to @df.method: '{key}'. The "
            f"supported keyword arguments are {valid_kwargs}"
        )
        assert key in valid_kwargs, key_error_string

    def decorator(func):
        num_returns = _get_func_num_returns(func, kwargs)
        func.__df_num_returns__ = num_returns

        if "choice_output" in kwargs:
            func.__df_choice_output__ = kwargs["choice_output"]
        if "stream_input" in kwargs:
            func.__df_stream_input__ = kwargs["stream_input"]
            if kwargs["stream_input"] != "Queue":
                raise TypeError(
                    f"Invalid stream input type: {func.__df_stream_input__}, only support 'Queue' now."
                )
        func.__df_method__ = True
        return func

    return decorator


class PyActorProcessPoint:
    def __init__(self, *args, **kwargs):
        self._module_name = ""
        self._clz_name = ""
        self._default_options = None
        self._decorated_class = None
        self._flow_func_infos = None
        self._saved_args = None
        self._input_num = 0
        self._output_num = 0
        raise TypeError(
            "PyActorProcessPoint.__init__ should not be called. Please use "
            "the @df.pyflow decorator instead."
        )

    def __call__(self, *args, **kwargs):
        raise TypeError(
            "FlowNode functions cannot be called directly. Instead "
            f"of running '{self._clz_name}()', "
            f"try '{self._clz_name}.fnode()'."
        )

    @classmethod
    def check_options_supported(cls, node_options):
        for key in node_options.keys():
            if key not in _pyflow_support_args_:
                raise TypeError(f"param:{key} is not support in @pyflow.")

    @classmethod
    def _df_from_class(cls, decorated_class, node_options):

        for attribute in [
            "fnode",
            "_fnode",
            "_df_from_class",
        ]:
            if hasattr(decorated_class, attribute):
                raise TypeError(
                    "Creating an process point class from class "
                    f"{decorated_class.__name__} can not overwrites "
                    f"attribute {attribute}, please rename"
                )
        cls.check_options_supported(node_options)

        class DerivedNodeClass(cls, decorated_class):
            pass

        name = f"{decorated_class.__name__}"
        DerivedNodeClass.__module__ = decorated_class.__module__
        DerivedNodeClass.__name__ = name
        DerivedNodeClass.__qualname__ = name
        # Construct the base object.
        derived_cls = DerivedNodeClass

        derived_cls._module_name = inspect.getmodule(derived_cls).__name__
        derived_cls._clz_name = derived_cls.__name__
        derived_cls._default_options = node_options
        derived_cls._decorated_class = decorated_class
        derived_cls._flow_func_infos = FlowFuncInfos()
        _parse_option_resource(node_options, derived_cls._flow_func_infos)
        _parse_option_env_hook_func(node_options, derived_cls._flow_func_infos)
        _parse_option_visible_device_enable(node_options, derived_cls._flow_func_infos)
        return derived_cls

    @classmethod
    def fnode(cls, *args, **kwargs):
        return cls._fnode(args=args, kwargs=kwargs, **cls._default_options)

    @classmethod
    def add_process_point(cls, flow_node, class_ins):
        pp = df.FuncProcessPoint(
            name=cls._clz_name, py_func=class_ins, workspace_dir="./" + flow_node.name + "_ws"
        )
        flow_node.add_process_point(pp)

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
                self._output_idx_offset = output_idx_offset
                self._output_num = output_num
                self._default_options = self._class_ins._default_options
                self._choice_output = None
                if hasattr(self._method, "__df_choice_output__"):
                    self._choice_output = self._method.__df_choice_output__

            def __call__(
                self,
                run_context: fw.MetaRunContext,
                inputs: Union[List[fw.FlowMsg], List[fw.FlowMsgQueue]],
            ):
                input_list = []
                logger = ff.FlowFuncLogger()
                try:
                    for input in inputs:
                        if isinstance(input, fw.FlowMsg):
                            if _check_flow_msg(input) != ff.FLOW_FUNC_SUCCESS:
                                logger.error("invalid input")
                                return ff.FLOW_FUNC_FAILED
                            input_list.append(
                                utils.convert_flow_msg_to_object(ff.FlowMsg(input))
                            )
                        elif isinstance(input, fw.FlowMsgQueue):
                            input_list.append(ff.FlowMsgQueue(input))
                        else:
                            logger.error("invalid input type:%s", type(input))
                            return ff.FLOW_FUNC_ERR_PARAM_INVALID

                    runtime_context = ff.MetaRunContext(run_context)
                    if inspect.isgeneratorfunction(self._method):  # stream output
                        gen = self._method(
                            self._class_ins, *tuple(input_list)
                        )  # unpacking tuple into args
                        for results in gen:
                            if (
                                self._set_output(runtime_context, results)
                                != ff.FLOW_FUNC_SUCCESS
                            ):
                                return ff.FLOW_FUNC_FAILED
                        return ff.FLOW_FUNC_SUCCESS
                    else:
                        results = self._method(
                            self._class_ins, *tuple(input_list)
                        )  # unpacking tuple into args
                        return self._set_output(runtime_context, results)
                except utils.DfAbortException as e:
                    logger.warn("proc is aborted, %s", str(e))
                    return ff.FLOW_FUNC_SUCCESS
                except Exception as e:
                    traceback.print_exc()
                    logger.error("proc wrapper exception, %s", str(e))
                    return ff.FLOW_FUNC_FAILED
                return ff.FLOW_FUNC_SUCCESS

            def _set_output(self, runtime_context, results):
                logger = ff.FlowFuncLogger()
                ret = _process_results(
                    runtime_context, results, self._output_num, self._choice_output
                )
                if ret[0] != ff.FLOW_FUNC_SUCCESS:
                    logger.error("failed to process outputs.")
                    return ff.FLOW_FUNC_FAILED
                result_list = ret[1]
                for index, result in enumerate(result_list):
                    if result is None:
                        continue
                    if (
                        runtime_context.set_output(
                            index + self._output_idx_offset, result
                        )
                        != ff.FLOW_FUNC_SUCCESS
                    ):
                        logger.error("set output failed")
                        return ff.FLOW_FUNC_FAILED
                return ff.FLOW_FUNC_SUCCESS

        return MethodClass(
            class_ins, method_name, method_def, input_num, output_idx_offset, output_num
        )

    @classmethod
    def _fnode(cls, args=None, kwargs=None, **node_options):
        kwargs = {} if kwargs is None else kwargs
        args = [] if args is None else args
        class_ins = cls.__new__(cls)
        class_ins._saved_args = (args, kwargs)
        class_ins._input_num = 0
        class_ins._output_num = 0

        class ActorFlowNodeMethod:
            def __init__(self, class_ins, method_name, method_def):
                self._class_ins = class_ins
                self._method = method_def
                self._method_name = method_name
                self._default_options = copy.deepcopy(self._class_ins._default_options)
                if hasattr(self._method, "__df_num_returns__"):
                    self._default_options["num_returns"] = (
                        self._method.__df_num_returns__
                    )
                if hasattr(self._method, "__df_stream_input__"):
                    self._default_options["stream_input"] = (
                        self._method.__df_stream_input__
                    )
                if hasattr(self._method, "__df_choice_output__"):
                    self._default_options["choice_output"] = (
                        self._method.__df_choice_output__
                    )
                self._input_num = utils.get_param_count(self._method)
                self._input_indexes = [
                    self._class_ins._input_num + i for i in range(self._input_num)
                ]
                self._output_num = 1
                self._output_idx_offset = self._class_ins._output_num
                self._stream_input = None
                self._flow_node = None
                if "num_returns" in self._default_options:
                    self._output_num = self._default_options["num_returns"]
                if "stream_input" in self._default_options:
                    self._stream_input = self._default_options["stream_input"]
                self._output_indexes = [
                    self._class_ins._output_num + i for i in range(self._output_num)
                ]

                func_params = _gen_func_params(
                    self._input_num,
                    self._output_num,
                    self._class_ins._input_num,
                    self._class_ins._output_num,
                    self._stream_input,
                )
                self._class_ins._flow_func_infos.add_func_params(
                    method_name, func_params
                )
                self._class_ins._input_num += self._input_num
                self._class_ins._output_num += self._output_num

            def __call__(self, *inputs):
                if len(inputs) != self._input_num:
                    raise utils.DfException(
                        f"Func:{self._method_name} need {self._input_num} input, "
                        f"but fnode got {len(inputs)} input"
                    )
                return self._flow_node._build_flow_node(
                    *inputs,
                    input_indexes=self._input_indexes,
                    output_indexes=self._output_indexes,
                )

            def __call__(self, *inputs):
                if len(inputs) != self._input_num:
                    raise utils.DfException(
                        f"Func:{self._method_name} need {self._input_num} input, "
                        f"but fnode got {len(inputs)} input"
                    )
                return self._flow_node._build_flow_node(
                    *inputs,
                    input_indexes=self._input_indexes,
                    output_indexes=self._output_indexes,
                )

            def set_fnode(self, flow_node):
                self._flow_node = flow_node

        actor_fnode_methods = {}
        for method_name in dir(class_ins._decorated_class):
            method_def = getattr(class_ins._decorated_class, method_name)
            if callable(method_def) and not method_name.startswith("__"):
                if hasattr(method_def, "__df_method__"):
                    flow_node_method_ins = ActorFlowNodeMethod(
                        class_ins, method_name, method_def
                    )
                    actor_fnode_methods[method_name] = flow_node_method_ins

                    method_class_ins = cls.get_redefined_method(
                        class_ins,
                        method_name,
                        method_def,
                        flow_node_method_ins._input_num,
                        flow_node_method_ins._output_idx_offset,
                        flow_node_method_ins._output_num,
                    )
                    setattr(class_ins, method_name, method_class_ins)

        flow_node = df.FlowNode(
            input_num=class_ins._input_num,
            output_num=class_ins._output_num,
            name=class_ins._clz_name + "_fnode",
        )
        class_ins._flow_func_infos.set_func_object(class_ins)
        FlowFuncRegister.register_flow_func_infos(
            class_ins._module_name, class_ins._clz_name, class_ins._flow_func_infos
        )
        cls.add_process_point(flow_node, class_ins)
        for method_name, actor_method in actor_fnode_methods.items():
            actor_method.set_fnode(flow_node)
            setattr(flow_node, method_name, actor_method)
        return flow_node

    def _super_init(self, meta_params):
        _init_process_point(meta_params)
        args = self._saved_args[0]
        kwargs = self._saved_args[1]
        super().__init__(*args, **kwargs)


class PyFunctionProcessPoint:
    def __init__(
        self,
        function,
        node_options,
    ):
        self.check_options_supported(node_options)
        self._default_options = node_options
        self._function = function
        self._stream_input = None
        self._choice_output = None
        if "stream_input" in node_options:
            self._stream_input = node_options["stream_input"]
            if self._stream_input != "Queue":
                raise TypeError(
                    f"Invalid stream input type: {self._stream_input}, only support 'Queue' now."
                )
        if "choice_output" in node_options:
            self._choice_output = node_options["choice_output"]
        self._module_name = (
            inspect.getmodule(function).__name__ if inspect.getmodule(function) else ""
        )
        self._clz_name = function.__name__
        self.func_name = function.__name__
        self.output_num = _get_func_num_returns(function, node_options)
        flow_func_infos = FlowFuncInfos()
        _parse_option_resource(node_options, flow_func_infos)
        _parse_option_env_hook_func(node_options, flow_func_infos)
        _parse_option_visible_device_enable(node_options, flow_func_infos)

        self.input_num = utils.get_param_count(self._function)
        func_params = _gen_func_params(
            self.input_num, self.output_num, stream_input=self._stream_input
        )
        flow_func_infos.add_func_params(self.func_name, func_params)

        flow_func_infos.set_func_object(self)
        FlowFuncRegister.register_flow_func_infos(
            self._module_name, self._clz_name, flow_func_infos
        )

        @functools.wraps(function)
        def _fnode_proxy(*args, **kwargs):
            return self._fnode(args=args, kwargs=kwargs, **self._default_options)

        self.fnode = _fnode_proxy

    def __call__(
        self,
        run_context: fw.MetaRunContext,
        inputs: Union[List[fw.FlowMsg], List[fw.FlowMsgQueue]],
    ):
        try:
            prepare_ret, input_list = self.prepare_inputs(inputs, self.input_num)
            if prepare_ret != ff.FLOW_FUNC_SUCCESS:
                ff.logger.error("failed to prepare input, ret = %d", prepare_ret)
                return prepare_ret

            runtime_context = ff.MetaRunContext(run_context)
            if inspect.isgeneratorfunction(self._function):  # stream output
                gen = self._function(*tuple(input_list))  # unpacking tuple into args
                for results in gen:
                    if (
                        self._set_output(runtime_context, results)
                        != ff.FLOW_FUNC_SUCCESS
                    ):
                        return ff.FLOW_FUNC_FAILED
                return ff.FLOW_FUNC_SUCCESS
            else:
                results = self._function(
                    *tuple(input_list)
                )  # unpacking tuple into args
                return self._set_output(runtime_context, results)
        except utils.DfAbortException as e:
            ff.logger.warn("proc is aborted, %s", str(e))
            return ff.FLOW_FUNC_SUCCESS
        except Exception as e:
            traceback.print_exc()
            ff.logger.error("proc wrapper exception, %s", str(e))
            return ff.FLOW_FUNC_FAILED
        return ff.FLOW_FUNC_SUCCESS

    @classmethod
    def check_options_supported(cls, node_options):
        for key in node_options.keys():
            if key not in _pyflow_support_args_:
                raise TypeError(f"param:{key} is not support in @pyflow.")

    def prepare_inputs(
        self, inputs: Union[List[fw.FlowMsg], List[fw.FlowMsgQueue]], input_num
    ):
        input_list = []
        logger = ff.FlowFuncLogger()
        for input_msg in inputs:
            if isinstance(input_msg, fw.FlowMsg):
                if _check_flow_msg(input_msg) != ff.FLOW_FUNC_SUCCESS:
                    logger.error("invalid input")
                    return ff.FLOW_FUNC_FAILED, []
                input_list.append(
                    utils.convert_flow_msg_to_object(ff.FlowMsg(input_msg))
                )
            elif isinstance(input_msg, fw.FlowMsgQueue):
                input_list.append(ff.FlowMsgQueue(input_msg))
            else:
                logger.error("invalid input type:%s", type(input_msg))
                return ff.FLOW_FUNC_ERR_PARAM_INVALID, []
        return ff.FLOW_FUNC_SUCCESS, input_list

    def prepare_outputs(self, runtime_context, outputs, output_num):
        return _process_results(
            runtime_context, outputs, output_num, self._choice_output
        )

    def add_process_point(self, flow_node):
        pp = df.FuncProcessPoint(
            py_func=self, workspace_dir="./" + flow_node.name + "_ws"
        )
        flow_node.add_process_point(pp)

    def _super_init(self, meta_params):
        _init_process_point(meta_params)

    def _set_output(self, runtime_context, results):
        ret = self.prepare_outputs(runtime_context, results, self.output_num)
        if ret[0] != ff.FLOW_FUNC_SUCCESS:
            ff.logger.error("failed to prepare outputs, ret = %d.", ret[0])
            return ff.FLOW_FUNC_FAILED
        result_list = ret[1]
        for index, result in enumerate(result_list):
            if result is None:
                continue
            if runtime_context.set_output(index, result) != ff.FLOW_FUNC_SUCCESS:
                ff.logger.error("set output failed")
                return ff.FLOW_FUNC_FAILED
        return ff.FLOW_FUNC_SUCCESS

    def _fnode(self, args=None, kwargs=None, **node_options):
        kwargs = {} if kwargs is None else kwargs
        args = [] if args is None else args
        flow_node = df.FlowNode(
            input_num=self.input_num,
            output_num=self.output_num,
            name=self.func_name + "_fnode",
        )
        self.add_process_point(flow_node)
        return flow_node


def _make_pyflow(function_or_class, options):
    if inspect.isfunction(function_or_class):
        return PyFunctionProcessPoint(function_or_class, options)

    if inspect.isclass(function_or_class):
        return PyActorProcessPoint._df_from_class(function_or_class, options)

    raise TypeError(
        "The @flow_node decorator must be applied to either a function or a class."
    )


def pyflow(*args, **kwargs) -> Union[PyFunctionProcessPoint, PyActorProcessPoint]:
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return _make_pyflow(args[0], {})
    return functools.partial(_make_pyflow, options=kwargs)
