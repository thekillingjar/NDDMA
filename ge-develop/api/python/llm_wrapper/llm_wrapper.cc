/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "Python.h"
#ifdef ASCEND_CI_LIMITED_PY37
#undef PyCFunction_NewEx
#endif

#include <map>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "acl/acl.h"
#include "ge/ge_api.h"
#include "graph/types.h"
#include "llm_datadist/llm_error_codes.h"
#include "llm_datadist_wrapper.h"
#include "llm_tensor.h"
#include "common/mem_utils.h"

#undef PYBIND11_CHECK_PYTHON_VERSION
#define PYBIND11_CHECK_PYTHON_VERSION

namespace llm {
namespace {
namespace py = pybind11;

std::vector<std::pair<int64_t, int64_t>> PyDictToVector(const py::dict &py_dict) {
  std::vector<std::pair<int64_t, int64_t>> result;

  for (const auto &item : py_dict) {
    int64_t key = item.first.cast<int64_t>();
    int64_t value = item.second.cast<int64_t>();
    result.emplace_back(key, value);
  }
  return result;
}

void BindStatusCodes(py::module &m) {
  m.attr("kSuccess") = py::int_(ge::SUCCESS);
  m.attr("kFailed") = py::int_(ge::FAILED);
  m.attr("kParamInvalid") = py::int_(ge::LLM_PARAM_INVALID);
  m.attr("kLLMWaitProcessTimeOut") = py::int_(ge::LLM_WAIT_PROC_TIMEOUT);
  m.attr("kLLMKVCacheNotExist") = py::int_(ge::LLM_KV_CACHE_NOT_EXIST);
  m.attr("kLLMRepeatRequest") = py::int_(ge::LLM_REPEAT_REQUEST);
  m.attr("kLLMRequestAlreadyCompleted") = py::int_(ge::LLM_REQUEST_ALREADY_COMPLETED);
  m.attr("kLLMEngineFinalized") = py::int_(ge::LLM_ENGINE_FINALIZED);
  m.attr("kLLMNotYetLink") = py::int_(ge::LLM_NOT_YET_LINK);
  m.attr("kAlreadyLink") = py::int_(ge::LLM_ALREADY_LINK);
  m.attr("kLinKFailed") = py::int_(ge::LLM_LINK_FAILED);
  m.attr("kUnlinkFailed") = py::int_(ge::LLM_UNLINK_FAILED);
  m.attr("kNotifyPromptUnlinkFailed") = py::int_(ge::LLM_NOTIFY_PROMPT_UNLINK_FAILED);
  m.attr("kLLMClusterNumExceedLimit") = py::int_(ge::LLM_CLUSTER_NUM_EXCEED_LIMIT);
  m.attr("kProcessingLink") = py::int_(ge::LLM_PROCESSING_LINK);
  m.attr("kDeviceOutOfMemory") = py::int_(ge::LLM_DEVICE_OUT_OF_MEMORY);
  m.attr("kPrefixAlreadyExist") = py::int_(ge::LLM_PREFIX_ALREADY_EXIST);
  m.attr("kPrefixNotExist") = py::int_(ge::LLM_PREFIX_NOT_EXIST);
  m.attr("kLLMSeqLenOverLimit") = py::int_(ge::LLM_SEQ_LEN_OVER_LIMIT);
  m.attr("kLLMNoFreeBlock") = py::int_(ge::LLM_NO_FREE_BLOCK);
  m.attr("kLLMBlocksOutOfMemory") = py::int_(ge::LLM_BLOCKS_OUT_OF_MEMORY);
  m.attr("kLLMExistLink") = py::int_(ge::LLM_EXIST_LINK);
  m.attr("kLLMFeatureNotEnabled") = py::int_(ge::LLM_FEATURE_NOT_ENABLED);
  m.attr("kLLMTimeout") = py::int_(ge::LLM_TIMEOUT);
  m.attr("kLLMLinkBusy") = py::int_(ge::LLM_LINK_BUSY);
  m.attr("kLLMOutOfMemory") = py::int_(ge::LLM_OUT_OF_MEMORY);
  m.attr("kLLMDeviceMemError") = py::int_(ACL_ERROR_RT_DEVICE_MEM_ERROR);
  m.attr("kLLMSuspectRemoteError") = py::int_(ACL_ERROR_RT_SUSPECT_REMOTE_ERROR);
}

void BindTensorFuncs(py::module &m) {
  (void)m.def("calc_tensor_size", &LLMTensor::CalcTensorSize);
  (void)m.def("build_npu_tensors", &LLMTensor::BuildNpuTensors);
  (void)m.def("build_tensor", &LLMTensor::BuildTensor);
  (void)m.def("clone_tensor", &LLMTensor::CloneTensor);
  (void)m.def("destroy_tensor", &LLMTensor::DestroyTensor);
  (void)m.def("tensor_get_buffer", &LLMTensor::GetBuffer);
  (void)m.def("get_string_tensor", &LLMTensor::GetStringTensor);
}

void BuildDataDistFuncs(py::module &m) {
  (void) m.def("initialize", &LLMDataDistWrapper::Init, py::call_guard<py::gil_scoped_release>());
  (void) m.def("finalize", &LLMDataDistWrapper::Finalize, py::call_guard<py::gil_scoped_release>());
  (void) m.def("check_link_status", &LLMDataDistWrapper::CheckLinkStatus, py::call_guard<py::gil_scoped_release>());
  (void) m.def("link_clusters", &LLMDataDistWrapper::LinkClusters, py::call_guard<py::gil_scoped_release>());
  (void) m.def("unlink_clusters", &LLMDataDistWrapper::UnlinkClusters, py::call_guard<py::gil_scoped_release>());
  (void) m.def("allocate_cache", &LLMDataDistWrapper::AllocateCache, py::call_guard<py::gil_scoped_release>());
  (void) m.def("deallocate_cache", &LLMDataDistWrapper::DeallocateCache, py::call_guard<py::gil_scoped_release>());
  (void) m.def("remove_cache_key", &LLMDataDistWrapper::RemoveCacheKey, py::call_guard<py::gil_scoped_release>());
  (void) m.def("pull_cache", &LLMDataDistWrapper::PullCache, py::call_guard<py::gil_scoped_release>());
  (void) m.def("copy_cache", &LLMDataDistWrapper::CopyCache, py::call_guard<py::gil_scoped_release>());
  (void) m.def("get_tensor", &LLMDataDistWrapper::GetCachedTensor, py::call_guard<py::gil_scoped_release>());
  (void) m.def("switch_role", &LLMDataDistWrapper::SwitchRole, py::call_guard<py::gil_scoped_release>());
  (void) m.def("swap_blocks", &LLMDataDistWrapper::SwapBlocks, py::call_guard<py::gil_scoped_release>());
  (void) m.def("dict_to_vector", &PyDictToVector, py::call_guard<py::gil_scoped_release>());
  (void) m.def("transfer_cache", &LLMDataDistWrapper::TransferCache, py::call_guard<py::gil_scoped_release>());
}

void BuildDataType(py::module &m) {
  m.attr("DT_FLOAT") = py::int_(static_cast<int32_t>(ge::DataType::DT_FLOAT));
  m.attr("DT_FLOAT16") = py::int_(static_cast<int32_t>(ge::DataType::DT_FLOAT16));
  m.attr("DT_BF16") = py::int_(static_cast<int32_t>(ge::DataType::DT_BF16));
  m.attr("DT_INT8") = py::int_(static_cast<int32_t>(ge::DataType::DT_INT8));
  m.attr("DT_INT16") = py::int_(static_cast<int32_t>(ge::DataType::DT_INT16));
  m.attr("DT_UINT16") = py::int_(static_cast<int32_t>(ge::DataType::DT_UINT16));
  m.attr("DT_UINT8") = py::int_(static_cast<int32_t>(ge::DataType::DT_UINT8));
  m.attr("DT_INT32") = py::int_(static_cast<int32_t>(ge::DataType::DT_INT32));
  m.attr("DT_INT64") = py::int_(static_cast<int32_t>(ge::DataType::DT_INT64));
  m.attr("DT_UINT32") = py::int_(static_cast<int32_t>(ge::DataType::DT_UINT32));
  m.attr("DT_UINT64") = py::int_(static_cast<int32_t>(ge::DataType::DT_UINT64));
  m.attr("DT_BOOL") = py::int_(static_cast<int32_t>(ge::DataType::DT_BOOL));
  m.attr("DT_DOUBLE") = py::int_(static_cast<int32_t>(ge::DataType::DT_DOUBLE));
  m.attr("DT_STRING") = py::int_(static_cast<int32_t>(ge::DataType::DT_STRING));
}
}  // namespace

PYBIND11_MODULE(llm_wrapper, m) {
  BindStatusCodes(m);
  BuildDataDistFuncs(m);
  BindTensorFuncs(m);
  BuildDataType(m);
}
}  // namespace llm
