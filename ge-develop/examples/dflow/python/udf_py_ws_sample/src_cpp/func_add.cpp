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
#include "pybind11/stl.h"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "meta_multi_func.h"
#include "flow_func_log.h"
__attribute__((constructor)) static void SoInit(void);
__attribute__((destructor)) static void SoDeinit(void);
void SoInit(void) {
  Py_Initialize();
}

void SoDeinit(void) {
  Py_Finalize();
}

namespace py = pybind11;

namespace FlowFunc {
namespace {
class ScopeGuard {
 public:
  explicit ScopeGuard(std::function<void()> callback) : callback_(callback) {}

  ~ScopeGuard() noexcept {
    if (callback_ != nullptr) {
      callback_();
    }
  }

 private:
  std::function<void()> callback_;
};
}  // namespace

class AddFlowFunc : public MetaMultiFunc {
 public:
  AddFlowFunc() {
    Py_DECREF(PyImport_ImportModule("threading"));
    thread_state_ = PyEval_SaveThread();
  }

  ~AddFlowFunc() override {
    py_module_.release();
    PyEval_RestoreThread(thread_state_);
    FLOW_FUNC_LOG_INFO("Finalize python interpreter.");
  }

  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    FLOW_FUNC_LOG_DEBUG("Init enter.");
    PyGILState_STATE gil_state = PyGILState_Ensure();
    ScopeGuard gil_guard([&gil_state]() {
      PyGILState_Release(gil_state);
      FLOW_FUNC_LOG_INFO("PyGILState_Release.");
    });
    FLOW_FUNC_LOG_INFO("PyGILState_Ensure.");
    int32_t result = FLOW_FUNC_SUCCESS;
    try {
      PyRun_SimpleString("import sys");
      std::string append = std::string("sys.path.append('") + params->GetWorkPath() + "')";
      PyRun_SimpleString(append.c_str());

      constexpr const char *py_module_name = "func_add";
      constexpr const char *py_clz_name = "Add";
      auto module = py::module_::import(py_module_name);
      py_module_ = module.attr(py_clz_name)();
      if (CheckProcExists() != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("%s %s check proc exists failed.", py_module_name, py_clz_name);
        return FLOW_FUNC_FAILED;
      }
      if (py::hasattr(py_module_, "init_flow_func")) {
        result = py_module_.attr("init_flow_func")(params).cast<int32_t>();
        if (result != FLOW_FUNC_SUCCESS) {
          FLOW_FUNC_LOG_ERROR("%s %s Init failed: %d", py_module_name, py_clz_name, result);
          return result;
        }
        FLOW_FUNC_LOG_INFO("%s %s Init success.", py_module_name, py_clz_name);
      } else {
        FLOW_FUNC_LOG_INFO("%s %s has no init_flow_func method.", py_module_name, py_clz_name);
      }
    } catch (std::exception &e) {
      FLOW_FUNC_LOG_ERROR("Init failed: %s", e.what());
      result = FLOW_FUNC_FAILED;
    }
    return result;
  }

  int32_t Add1Proc(const std::shared_ptr<MetaRunContext> &run_context,
                   const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    FLOW_FUNC_LOG_DEBUG("Add1Proc enter.");
    PyGILState_STATE gil_state = PyGILState_Ensure();
    ScopeGuard gil_guard([&gil_state]() {
      PyGILState_Release(gil_state);
      FLOW_FUNC_LOG_INFO("PyGILState_Release.");
    });
    FLOW_FUNC_LOG_INFO("PyGILState_Ensure.");
    int32_t result = FLOW_FUNC_SUCCESS;
    try {
      result = py_module_.attr("add1")(run_context, input_flow_msgs).cast<int32_t>();
      if (result != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("Add1 failed: %d", result);
        return result;
      }
      FLOW_FUNC_LOG_INFO("Add1 success.");
    } catch (const std::exception &e) {
      FLOW_FUNC_LOG_ERROR("Proc failed: %s", e.what());
      result = FLOW_FUNC_FAILED;
    }
    return result;
  }

  int32_t Add2Proc(const std::shared_ptr<MetaRunContext> &run_context,
                   const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    FLOW_FUNC_LOG_DEBUG("Add2Proc enter.");
    PyGILState_STATE gil_state = PyGILState_Ensure();
    ScopeGuard gil_guard([&gil_state]() {
      PyGILState_Release(gil_state);
      FLOW_FUNC_LOG_INFO("PyGILState_Release.");
    });
    FLOW_FUNC_LOG_INFO("PyGILState_Ensure.");
    int32_t result = FLOW_FUNC_SUCCESS;
    try {
      result = py_module_.attr("add2")(run_context, input_flow_msgs).cast<int32_t>();
      if (result != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("Add2 failed: %d", result);
        return result;
      }
      FLOW_FUNC_LOG_INFO("Add2 success.");
    } catch (const std::exception &e) {
      FLOW_FUNC_LOG_ERROR("Proc failed: %s", e.what());
      result = FLOW_FUNC_FAILED;
    }
    return result;
  }

 private:
  int32_t CheckProcExists() const {
    if (!py::hasattr(py_module_, "add1")) {
      FLOW_FUNC_LOG_ERROR("There is no method named Add1.");
    }
    if (!py::hasattr(py_module_, "add2")) {
      FLOW_FUNC_LOG_ERROR("There is no method named Add2.");
    }
    return FLOW_FUNC_SUCCESS;
  }

  py::object py_module_;
  PyThreadState *thread_state_ = nullptr;
};

FLOW_FUNC_REGISTRAR(AddFlowFunc)
    .RegProcFunc("Add1", &AddFlowFunc::Add1Proc)
    .RegProcFunc("Add2", &AddFlowFunc::Add2Proc);
}  // namespace FlowFunc
