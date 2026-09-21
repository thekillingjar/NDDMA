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

from jinja2 import Template
import os

CONTENT = '''
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <functional>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include "pybind11/stl.h"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "meta_multi_func.h"
#include "flow_func_log.h"

namespace py = pybind11;

namespace FlowFunc {
namespace {
class ScopeGuard {
 public:
  explicit ScopeGuard(std::function<void()> callback) : exit_callback_(callback) {}

  ~ScopeGuard() noexcept
  {
    if (exit_callback_ != nullptr) {
      exit_callback_();
    }
  }
 private:
  std::function<void()> exit_callback_;
};
}
class {{clz_name}} : public MetaMultiFunc {
 public:
  {{clz_name}}()
  {
    Py_Initialize();
    Py_DECREF(PyImport_ImportModule("threading"));
    thread_state_ = PyEval_SaveThread();
    FLOW_FUNC_LOG_INFO("init python interpreter");
  }

  ~{{clz_name}}() override
  {
    {
      py::gil_scoped_acquire acquire;
      py_obj_ = py::object();
      py::module gc = py::module_::import("gc");
      gc.attr("collect")();
    }

    PyEval_RestoreThread(thread_state_);
    Py_Finalize();
    thread_state_ = nullptr;
    FLOW_FUNC_LOG_INFO("finalize python interpreter");
  }

  int32_t Init(const std::shared_ptr<MetaParams> &params) override
  {
    FLOW_FUNC_LOG_DEBUG("Init enter");
    py::gil_scoped_acquire acquire;
    int32_t result = FLOW_FUNC_SUCCESS;
    try {
      PyRun_SimpleString("import sys");
      std::string append = std::string("sys.path.append('") + params->GetWorkPath() + "')";
      PyRun_SimpleString(append.c_str());
      result = InitPyObj(params);
      if (result != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("%s.%s init py object failed, result=%d", py_module_name, py_clz_name, result);
        return result;
      }
      FLOW_FUNC_LOG_INFO("%s.%s init py object success", py_module_name, py_clz_name);
    } catch (std::exception &e) {
      FLOW_FUNC_LOG_ERROR("%s.%s init exception: %s", py_module_name, py_clz_name, e.what());
      result = FLOW_FUNC_FAILED;
    }
    FLOW_FUNC_LOG_DEBUG("FlowFunc Init end.");
    return result;
  }

  int32_t ResetFlowFuncState(const std::shared_ptr<MetaParams> &params) override
  {
    py::gil_scoped_acquire acquire;
    int32_t result = FLOW_FUNC_SUCCESS;
    try {
      result = InitPyObj(params);
      if (result != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("reset flow func[%s.%s] state failed, result=%d", py_module_name, py_clz_name, result);
        return result;
      }
      FLOW_FUNC_LOG_INFO("reset flow func[%s.%s] state success", py_module_name, py_clz_name);
    } catch (std::exception &e) {
      FLOW_FUNC_LOG_ERROR("reset flow func[%s.%s] state exception: %s", py_module_name, py_clz_name, e.what());
      result = FLOW_FUNC_FAILED;
    }
    return result;
  }

  {% for f_name in f_names %}
  {% if stream_inputs[loop.index0] %}
  int32_t {{f_name|replace("_", " ")|title|replace(" ", "")}}Proc(
      const std::shared_ptr<MetaRunContext> &runContext, const std::vector<std::shared_ptr<FlowMsgQueue>> &inputFlowMsgQueues)
  {
    py::gil_scoped_acquire acquire;
    int32_t result = FLOW_FUNC_SUCCESS;
    std::string proc_name = "{{py_module_name}}.{{clz_name}}";
    try {
      auto proc_func = py_obj_;
      if (py::hasattr(py_obj_, "{{f_name}}")) {
        proc_func = py_obj_.attr("{{f_name}}");
        proc_name += "{{f_name}}";
      }
      FLOW_FUNC_LOG_INFO("call %s begin.", proc_name.c_str());
      result = proc_func(runContext, inputFlowMsgQueues).cast<int32_t>();
      if (result != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("call %s return %d.", proc_name.c_str(), result);
        return result;
      }
      FLOW_FUNC_LOG_INFO("call %s success.", proc_name.c_str());
    } catch (std::exception &e) {
      FLOW_FUNC_LOG_ERROR("call %s exception: %s", proc_name.c_str(), e.what());
      result = FLOW_FUNC_FAILED;
    }
    return result;
  }
  {% else %}
  int32_t {{f_name|replace("_", " ")|title|replace(" ", "")}}Proc(
      const std::shared_ptr<MetaRunContext> &runContext, const std::vector<std::shared_ptr<FlowMsg>> &inputFlowMsgs)
  {
    py::gil_scoped_acquire acquire;
    int32_t result = FLOW_FUNC_SUCCESS;
    std::string proc_name = "{{py_module_name}}.{{clz_name}}";
    try {
      auto proc_func = py_obj_;
      // function has not attr
      if (py::hasattr(py_obj_, "{{f_name}}")) {
        proc_func = py_obj_.attr("{{f_name}}");
        proc_name += "{{f_name}}";
      }
      FLOW_FUNC_LOG_INFO("call %s begin.", proc_name.c_str());
      result = proc_func(runContext, inputFlowMsgs).cast<int32_t>();
      if (result != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("call %s return %d.", proc_name.c_str(), result);
        return result;
      }
      FLOW_FUNC_LOG_INFO("call %s success.", proc_name.c_str());
    } catch (std::exception &e) {
      FLOW_FUNC_LOG_ERROR("call %s exception: %s", proc_name.c_str(), e.what());
      result = FLOW_FUNC_FAILED;
    }
    return result;
  }
  {% endif %}
  {% endfor %}
 private:
  int32_t InitPyObj(const std::shared_ptr<MetaParams> &params)
  {
    int32_t result = FLOW_FUNC_SUCCESS;
    std::string pkl_file = std::string(params->GetWorkPath()) + "/" + py_clz_name + ".pkl";
    if (CheckFileExists(pkl_file)) {
      return InitPyObjFromPkl(params);
    }
    FLOW_FUNC_LOG_INFO("Load py module name: %s", py_module_name);
    auto module = py::module_::import(py_module_name);
    FLOW_FUNC_LOG_INFO("%s.%s import success", py_module_name, py_clz_name);
    py_obj_ = module.attr(py_clz_name)();
    if (CheckProcExists() != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("%s.%s check proc exists failed", py_module_name, py_clz_name);
      return FLOW_FUNC_FAILED;
    }
    if (py::hasattr(py_obj_, "init_flow_func")) {
      result = py_obj_.attr("init_flow_func")(params).cast<int32_t>();
      if (result != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("%s.%s init_flow_func result=%d", py_module_name, py_clz_name, result);
        return result;
      }
      FLOW_FUNC_LOG_INFO("%s.%s init_flow_func success", py_module_name, py_clz_name);
    } else {
      FLOW_FUNC_LOG_INFO("%s.%s has no init_flow_func method, no need init", py_module_name, py_clz_name);
    }
    return result;
  }
  int32_t CheckProcExists() const
  {
    {% for f_name in f_names %}
    if (!py::hasattr(py_obj_, "{{f_name}}")) {
      FLOW_FUNC_LOG_ERROR("{{py_module_name}}.{{clz_name}} has not proc method {{f_name}}");
      return FLOW_FUNC_FAILED;
    }
    {% endfor %}
    return FLOW_FUNC_SUCCESS;
  }
  bool CheckFileExists(const std::string &file_path) const
  {
    return (access(file_path.c_str(), F_OK) != -1);
  }

  int32_t GetFileBuffer(const std::string &file_name, std::vector<uint8_t> &buffer)
  {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
      FLOW_FUNC_LOG_ERROR("failed to open file:%s", file_name.c_str());
      return FLOW_FUNC_FAILED;
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
      file.close();
      FLOW_FUNC_LOG_ERROR("failed to read file:%s", file_name.c_str());
      return FLOW_FUNC_FAILED;
    }
    file.close();
    return FLOW_FUNC_SUCCESS;
  }

  int32_t InitPyObjFromPkl(const std::shared_ptr<MetaParams> &params)
  {
    int32_t result = FLOW_FUNC_SUCCESS;
    auto df = py::module_::import("dataflow");
    auto type_register = df.attr("msg_type_register");
    FLOW_FUNC_LOG_INFO("msg_type_register import success");

    PyRun_SimpleString("import sys");
    std::string append = std::string("sys.path.append('") + "{{work_path}}" + "')";
    PyRun_SimpleString(append.c_str());

    std::string register_pkl_file = std::string(params->GetWorkPath()) + "/_msg_type_register.pkl";
    std::vector<uint8_t> reg_buf;
    if (GetFileBuffer(register_pkl_file, reg_buf) != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("failed to get file:%s buffer", register_pkl_file.c_str());
      return FLOW_FUNC_FAILED;
    }

    auto deserialize_func = type_register.attr("get_deserialize_func")(65535);
    if (deserialize_func == nullptr) {
      FLOW_FUNC_LOG_ERROR("failed to get register deserialize func");
      return FLOW_FUNC_FAILED;
    }
    type_register = deserialize_func(py::memoryview::from_memory(&reg_buf[0], reg_buf.size(), false));
    auto df_utils = py::module_::import("dataflow.utils.utils");
    df_utils.attr("set_msg_type_register")(type_register);

    std::string env_hook_func_pkl_file = std::string(params->GetWorkPath()) + "/_env_hook_func.pkl";
    if (CheckFileExists(env_hook_func_pkl_file)) {
      std::vector<uint8_t> hook_buffer;
      if (GetFileBuffer(env_hook_func_pkl_file, hook_buffer) != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("failed to get file:%s buffer", env_hook_func_pkl_file.c_str());
        return FLOW_FUNC_FAILED;
      }
      deserialize_func(py::memoryview::from_memory(&hook_buffer[0], hook_buffer.size(), false))();
    }

    std::string pkl_file = std::string(params->GetWorkPath()) + "/" + py_clz_name + ".pkl";
    std::vector<uint8_t> buffer;
    if (GetFileBuffer(pkl_file, buffer) != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("failed to get file:%s buffer", pkl_file.c_str());
      return FLOW_FUNC_FAILED;
    }

    py_obj_ = deserialize_func(py::memoryview::from_memory(&buffer[0], buffer.size(), false));
    if (IsClass(py_obj_) && CheckProcExists() != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("%s.%s check proc exists failed", py_module_name, py_clz_name);
      return FLOW_FUNC_FAILED;
    }

    if (py::hasattr(py_obj_, "_super_init")) {
      py_obj_.attr("_super_init")(params);
      FLOW_FUNC_LOG_INFO("%s.%s _super_init success", py_module_name, py_clz_name);
    }
    return result;
  }
  bool IsClass(const py::object& obj) {
    return py::isinstance<py::type>(obj);
  }

  static constexpr const char *py_module_name = "{{py_module_name}}";
  static constexpr const char *py_clz_name = "{{clz_name}}";
  PyThreadState *thread_state_ = nullptr;
  py::object py_obj_;
};

FLOW_FUNC_REGISTRAR({{clz_name}})   {% for f_name in f_names %}
    .RegProcFunc("{{f_name}}", &{{clz_name}}::{{f_name|replace("_", " ")|title|replace(" ", "")}}Proc){% endfor %};
}

'''

TPL = Template(CONTENT)


def gen_wrapper_code(clz_name, f_names, py_module_name, stream_inputs):
    global TPL
    return TPL.render(clz_name=clz_name, f_names=f_names, py_module_name=py_module_name, work_path=os.getcwd(), 
                      stream_inputs=stream_inputs)
