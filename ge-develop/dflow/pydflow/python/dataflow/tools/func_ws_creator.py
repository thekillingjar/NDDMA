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

import os
import json
import re
import shutil
import inspect
import sys
from typing import Optional
import dataflow.utils.log as log
from dataflow.tools.tpl.tpl_wrapper_code import gen_wrapper_code
from dataflow.tools.tpl.tpl_cmake import gen_func_cmake
from dataflow.tools.tpl.tpl_py_func_code import gen_py_func_code
from dataflow.utils.msg_type_register import msg_type_register

MSG_TYPE_PICKLED_MSG = 65535


class FuncWsCreator:
    def __init__(
            self, funcs_param, clz_name="", ws_dir="", flow_func_infos: Optional = None
    ) -> None:
        if ws_dir != "":
            self.ws_dir = os.path.abspath(ws_dir)
        else:
            self.ws_dir = os.getcwd()

        self.cmake = "CMakeLists.txt"
        self.src_dir = "src_cpp"
        self.py_src_dir = "src_python"
        self._extract_func_indexes(funcs_param)
        self.clz_name = clz_name
        self._gen_names()
        self.flow_func_infos = flow_func_infos

    @staticmethod
    def _convert_camel_to_snake(value):
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", value)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def generate(self):
        # 1. generate udf function code
        self._gen_func_code()
        # 2. generate udf CMakeLists.txt (must after gen_func_code)
        self._gen_cmake()
        # 3. generate udf compile config (last step)
        self._gen_compile_cfg()

    def copy_py_src_file(self, py_src_file):
        if self._has_func_object():
            pkl_file_path = os.path.join(self.ws_dir, self.py_src_dir)
            pkl_file_name = pkl_file_path + "/" + self.clz_name + ".pkl"
            serialize_func = msg_type_register.get_serialize_func(MSG_TYPE_PICKLED_MSG)
            serialized_data = serialize_func(self.flow_func_infos.get_func_object())
            with open(pkl_file_name, "wb") as f:
                f.write(serialized_data)

            msg_type_register_file_name = pkl_file_path + "/_msg_type_register.pkl"
            serialized_data = serialize_func(msg_type_register)
            with open(msg_type_register_file_name, "wb") as f:
                f.write(serialized_data)

            if self.flow_func_infos.get_env_hook_func() is not None:
                env_hook_func_file_name = pkl_file_path + "/_env_hook_func.pkl"
                module = inspect.getmodule(self.flow_func_infos.get_env_hook_func())
                if module:
                    import cloudpickle

                    cloudpickle.register_pickle_by_value(module)
                serialized_data = serialize_func(
                    self.flow_func_infos.get_env_hook_func()
                )
                with open(env_hook_func_file_name, "wb") as f:
                    f.write(serialized_data)
        else:
            file_path = os.path.join(self.ws_dir, self.py_src_dir, self.py_name)
            shutil.copy(py_src_file, file_path)

    def get_config_file(self):
        return os.path.join(self.ws_dir, self.cfg_name)

    def _has_func_object(self):
        return (
                self.flow_func_infos is not None
                and self.flow_func_infos.get_func_object() is not None
        )

    def _extract_func_indexes(self, funcs_param):
        self.func_names = []
        self.input_output_indexes = []
        self.stream_inputs = []

        func_cfgs = funcs_param.split(",")
        for cfg in func_cfgs:
            f_params = cfg.split(":")
            num = len(f_params)
            if num < 3:
                raise ValueError(f"params {cfg} format invalid")
            # extract func name
            f_name = f_params[0].strip()
            if not f_name.isidentifier():
                raise ValueError(f"params {cfg} function name format invalid")

            self.func_names.append(f_name)

            # param eg.i1, o2, extract input and output index
            input_idx = []
            output_idx = []
            stream_input = False
            for i in range(1, num):
                io = f_params[i].strip()
                if io.startswith("i"):
                    input_idx.append(int(io[1:]))
                elif io.startswith("o"):
                    output_idx.append(int(io[1:]))
                elif io == "stream_input":
                    stream_input = True
                else:
                    raise ValueError(f"params {cfg} input/output format invalid")

            self.input_output_indexes.append((input_idx, output_idx))
            self.stream_inputs.append(stream_input)

    def _gen_names(self):
        if self.clz_name == "":
            for i, f_name in enumerate(self.func_names):
                # to avoid too many funcs, only use most 4 func names to gen bin/cpp/cfg names
                if i < 4:
                    self.clz_name += f_name.title()
        if not self.clz_name.isidentifier():
            raise ValueError(f"clz_name {self.clz_name} format invalid")

        clz_name_snake = FuncWsCreator._convert_camel_to_snake(self.clz_name)

        self.bin_name = "libfunc_" + clz_name_snake + ".so"
        self.cfg_name = "func_" + clz_name_snake + ".json"
        self.cpp_name = "func_" + clz_name_snake + ".cpp"
        self.py_name = "func_" + clz_name_snake + ".py"
        self.prj_name = "func_" + clz_name_snake
        self.py_module_name = "func_" + clz_name_snake

    def _gen_func_code(self):
        f_path = os.path.join(self.ws_dir, self.src_dir)
        if not os.path.exists(f_path):
            os.makedirs(f_path)
        f_path = os.path.join(f_path, self.cpp_name)
        with open(f_path, "w") as f:
            clz_name = self.clz_name
            py_module_name = self.py_module_name
            if self._has_func_object():
                func_obj = self.flow_func_infos.get_func_object()
                clz_name = func_obj._clz_name
                py_module_name = func_obj._module_name
            f.write(
                gen_wrapper_code(
                    clz_name, self.func_names, py_module_name, self.stream_inputs
                )
            )
        log.info("create cpp file %s end", f_path)
        f_path = os.path.join(self.ws_dir, self.py_src_dir)
        if not os.path.exists(f_path):
            os.makedirs(f_path)
        if not self._has_func_object():
            f_path = os.path.join(f_path, self.py_name)
            with open(f_path, "w") as f:
                f.write(gen_py_func_code(self.clz_name, self.func_names))
            log.info("create python file %s end", f_path)

    def _gen_cmake(self):
        f_path = os.path.join(self.ws_dir, self.cmake)
        with open(f_path, "w") as f:
            f.write(gen_func_cmake(self.prj_name, self.src_dir, self.py_src_dir))
        log.info("create cmake file %s end", f_path)

    def _gen_compile_cfg(self):
        input_num = self._extract_input_num()
        output_num = self._extract_output_num()
        udf_desc = {}
        num = len(self.func_names)
        funcs = []
        for i in range(num):
            func_dict = {}
            func_dict["func_name"] = self.func_names[i]
            input_indexes, output_indexes = self.input_output_indexes[i]
            func_dict["inputs_index"] = input_indexes
            func_dict["outputs_index"] = output_indexes
            func_dict["stream_input"] = self.stream_inputs[i]
            funcs.append(func_dict)
        udf_desc["func_list"] = funcs
        udf_desc["input_num"] = input_num
        udf_desc["output_num"] = output_num
        udf_desc["target_bin"] = self.bin_name
        udf_desc["workspace"] = self.ws_dir
        udf_desc["cmakelist_path"] = self.cmake
        udf_desc["heavy_load"] = True
        if self.flow_func_infos is not None:
            resources_info = self.flow_func_infos.get_running_resources_info()
            if len(resources_info) != 0:
                udf_desc["running_resources_info"] = resources_info
            visible_device_enable = self.flow_func_infos.get_visible_device_enable()
            if visible_device_enable is not None:
                udf_desc["visible_device_enable"] = visible_device_enable

        f_path = os.path.join(self.ws_dir, self.cfg_name)
        with open(f_path, "w") as f:
            # use indent 4
            json.dump(udf_desc, f, indent=4)
        log.info("create compile config file %s end", f_path)

    def _extract_input_num(self):
        input_indexes = []
        for indexes, _ in self.input_output_indexes:
            for input in indexes:
                input_indexes.append(input)
        # sort input index
        input_indexes.sort()
        # check input index should start from 0 and continuous
        for i, idx in enumerate(input_indexes):
            if i < idx:
                raise ValueError(f"input index {i} is not configured")
            if i > idx:
                raise ValueError(f"input index {idx} is configured repeatedly")
        return len(input_indexes)

    def _extract_output_num(self):
        output_indexes = set()
        for _, indexes in self.input_output_indexes:
            for output in indexes:
                output_indexes.add(output)

        output_indexes_list = list(output_indexes)
        # sort input index
        output_indexes_list.sort()
        # check input index should start from 0 and continuous
        for i, idx in enumerate(output_indexes_list):
            if i != idx:
                raise ValueError(f"output index {i} is not configured")

        return len(output_indexes_list)
