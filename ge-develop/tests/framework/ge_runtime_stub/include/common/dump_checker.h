/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_DUMP_CHECKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_DUMP_CHECKER_H_
#include <string>
#include <set>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include "proto/op_mapping.pb.h"
#include "aicpu_task_struct.h"
#include "framework/common/debug/ge_log.h"
#include "aicpu/aicpu_schedule/aicpusd_info.h"
#include "common/checker.h"
#include "stub/runtime_stub_impl.h"
#include "stub/acl_runtime_stub_impl.h"

using OpMappingInfo = toolkit::aicpu::dump::OpMappingInfo;
using DumpTask = toolkit::aicpu::dump::Task;
using DumpData = toolkit::aicpu::dump::DumpData;
constexpr uint32_t kModelName = 2U;
constexpr uint32_t kModelId = 3U;
constexpr uint32_t kStepIdAddr = 4U;

namespace ge {

struct OpMappingInfoWithFlag {
  OpMappingInfo op_mapping_info;
  bool need_unload;
  OpMappingInfoWithFlag(OpMappingInfo info, bool flag) : op_mapping_info(info), need_unload(flag) {}
};

class DumpChecker {
 public:
  /**
   * 解析、校验、保存OpMappingInfo
   * @param dump_info
   * @return
   */
  ge::Status LoadOpMappingInfo(const char_t *opMappingInfoStr, uint32_t length, bool need_unload = false) {
    const std::string proto_info(opMappingInfoStr, static_cast<uint64_t>(length));
    OpMappingInfo op_mapping_info;
    GE_ASSERT_TRUE(op_mapping_info.ParseFromString(proto_info));
    if (op_mapping_info.step_id_case() == kStepIdAddr) {
      step_id_addr_ = reinterpret_cast<uint32_t *>(reinterpret_cast<uintptr_t>(op_mapping_info.step_id_addr()));
    }

    if (step_id_addr_ != nullptr && step_id_ != *step_id_addr_) {
      GELOGI("Updating step_id to [%u]", *step_id_addr_);
      step_id_ = *step_id_addr_;
    }

    if (op_mapping_info.flag() == 1U) {
      load_op_mapping_infos_.emplace_back(op_mapping_info, need_unload);
      GELOGI("Insert LoadOpMappingInfo success, cur size: %u.", load_op_mapping_infos_.size());
    } else if (op_mapping_info.flag() == 0U) {
      unload_op_mapping_infos_.emplace_back(op_mapping_info);
      GELOGI("Insert UnLoadOpMappingInfo success, cur size: %u.", unload_op_mapping_infos_.size());
    } else {
      GELOGE(FAILED, "[LoadOpMappingInfo] failed, load_flag: %u is invalid", op_mapping_info.flag());
      return PARAM_INVALID;
    }
    return ge::SUCCESS;
  }

  uint32_t GetLoadOpMappingInfoSize() {
    return load_op_mapping_infos_.size();
  }

  uint32_t GetUnLoadOpMappingInfoSize() {
    return unload_op_mapping_infos_.size();
  }

  void ClearAllOpMappingInfos() {
    load_op_mapping_infos_.clear();
    unload_op_mapping_infos_.clear();
    GELOGI("Clear OpMappingInfo success, loadMap size: %u, unloadMap size: %u", load_op_mapping_infos_.size(),
      unload_op_mapping_infos_.size());
  }

  uint32_t GetOpMappingInfoTaskSize(const std::string op_name) {
    OpMappingInfo op_mapping_info;
    if (GetOpMappingInfoByOpName(op_name, op_mapping_info) == ge::SUCCESS) {
      return op_mapping_info.task_size();
    }
    return UINT32_MAX;
  }

  std::string GetDumpPath(const std::string op_name) {
    OpMappingInfo op_mapping_info;
    if (GetOpMappingInfoByOpName(op_name, op_mapping_info) == ge::SUCCESS) {
      return op_mapping_info.dump_path();
    }
    return std::string();
  }

  std::string GetModelName(const std::string op_name) {
    OpMappingInfo op_mapping_info;
    if ((GetOpMappingInfoByOpName(op_name, op_mapping_info) == ge::SUCCESS) &&
      (op_mapping_info.model_name_param_case() == kModelName)) {
      return op_mapping_info.model_name();
    }
    return std::string();
  }

  uint32_t GetModelId(const std::string op_name) {
    OpMappingInfo op_mapping_info;
    if ((GetOpMappingInfoByOpName(op_name, op_mapping_info) == ge::SUCCESS) &&
      (op_mapping_info.model_id_param_case() == kModelId)) {
      return op_mapping_info.model_id();
    }
    return UINT32_MAX;
  }

  uint32_t GetStepId() {
    return step_id_;
  }

  std::string GetDumpStep(const std::string op_name) {
    OpMappingInfo op_mapping_info;
    if (GetOpMappingInfoByOpName(op_name, op_mapping_info) == ge::SUCCESS) {
      return op_mapping_info.dump_step();
    }
    return std::string();
  }

  int32_t GetDumpData(const std::string op_name) {
    OpMappingInfo op_mapping_info;
    if (GetOpMappingInfoByOpName(op_name, op_mapping_info) == ge::SUCCESS) {
      return static_cast<int32_t>(op_mapping_info.dump_data());
    }
    return INT32_MAX;
  }

  bool GetEndGraph(const std::string op_name) {
    DumpTask task;
    if (GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) {
      return task.end_graph();
    }
    return false;
  }

  uint32_t GetStreamId(const std::string op_name) {
    DumpTask task;
    if (GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) {
      return task.stream_id();
    }
    return UINT32_MAX;
  }

  uint32_t GetTaskId(const std::string op_name) {
    DumpTask task;
    if (GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) {
      return task.task_id();
    }
    return UINT32_MAX;
  }

  uint32_t GetContextId(const std::string op_name) {
    DumpTask task;
    if (GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) {
      return task.context_id();
    }
    return UINT32_MAX;
  }

  std::string GetOpType(const std::string op_name) {
    DumpTask task;
    if (GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) {
      return task.op().op_type();
    }
    return std::string();
  }

  int32_t GetTaskType(const std::string op_name) {
    DumpTask task;
    if (GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) {
      return static_cast<int32_t>(task.task_type());
    }
    return INT32_MAX;
  }

  std::string GetOpAttrName(const std::string op_name, const int32_t attr_id) {
    DumpTask task;
    if ((GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) && (attr_id < task.attr().size())) {
      return task.attr(attr_id).name();
    }
    return std::string();
  }

  std::string GetOpAttrValue(const std::string op_name, const int32_t attr_id) {
    DumpTask task;
    if ((GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) && (attr_id < task.attr().size())) {
      return task.attr(attr_id).value();
    }
    return std::string();
  }

  uint64_t GetTaskInputAddr(const std::string op_name, const int32_t input_id) {
    for (auto &load_op_mapping_info : load_op_mapping_infos_) {
      for (int32_t id = 0; id < load_op_mapping_info.op_mapping_info.task_size(); id++) {
        const DumpTask &task = load_op_mapping_info.op_mapping_info.task(id);
        if ((task.op().op_name() == op_name) && (input_id < task.input().size())) {
          return task.input(input_id).address();
        }
      }
    }
    return UINT64_MAX;
  }

  uint64_t GetTaskOutputAddr(const std::string op_name, const int32_t output_id) {
    for (auto &load_op_mapping_info : load_op_mapping_infos_) {
      for (int32_t id = 0; id < load_op_mapping_info.op_mapping_info.task_size(); id++) {
        const DumpTask &task = load_op_mapping_info.op_mapping_info.task(id);
        if ((task.op().op_name() == op_name) && (output_id < task.output().size())) {
          return task.output(output_id).address();
        }
      }
    }
    return UINT64_MAX;
  }

  uint64_t GetTaskInputSize(const std::string op_name) {
    for (auto &load_op_mapping_info : load_op_mapping_infos_) {
      for (int32_t id = 0; id < load_op_mapping_info.op_mapping_info.task_size(); id++) {
        const DumpTask &task = load_op_mapping_info.op_mapping_info.task(id);
        if ((task.op().op_name() == op_name) && (task.input().size() > 0)) {
          return task.input().size();
        }
      }
    }
    return 0UL;
  }

  uint64_t GetTaskOutputSize(const std::string op_name, const int32_t output_id) {
    for (auto &load_op_mapping_info : load_op_mapping_infos_) {
      for (int32_t id = 0; id < load_op_mapping_info.op_mapping_info.task_size(); id++) {
        const DumpTask &task = load_op_mapping_info.op_mapping_info.task(id);
        if ((task.op().op_name() == op_name) && (task.output().size() > 0)) {
          return task.output().size();
        }
      }
    }
    return 0UL;
  }

  uint64_t GetTaskWorkspaceAddr(const std::string op_name, const int32_t workspace_id) {
    DumpTask task;
    if ((GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) && (workspace_id < task.space().size())) {
      return task.space(workspace_id).data_addr();
    }
    return UINT64_MAX;
  }

  uint64_t GetTaskContextInputAddr(const std::string op_name, const int32_t context_id, const int32_t input_id) {
    DumpTask task;
    if ((GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) && (context_id < task.context().size()) &&
      (input_id < task.context(context_id).input().size())) {
      return task.context(context_id).input(input_id).address();
    }
    return UINT64_MAX;
  }

  uint64_t GetTaskContextOutputAddr(const std::string op_name, const int32_t context_id, const int32_t output_id) {
    DumpTask task;
    if ((GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) && (context_id < task.context().size()) &&
      (output_id < task.context(context_id).output().size())) {
      return task.context(context_id).output(output_id).address();
    }
    return UINT64_MAX;
  }

  uint64_t GetTaskOpBufferAddr(const std::string op_name, const int32_t buffer_id) {
    DumpTask task;
    if ((GetDumpTaskByOpName(op_name, task) == ge::SUCCESS) && (buffer_id < task.buffer().size())) {
      return task.buffer(buffer_id).address();
    }
    return UINT64_MAX;
  }

  bool CheckOpMappingInfoClear() {
    for (auto &load_op_mapping_info : load_op_mapping_infos_) {
      if (load_op_mapping_info.need_unload && (!IsOpMappingInfoCleared(load_op_mapping_info.op_mapping_info))) {
        return false;
      }
    }
    return true;
  }

  ge::Status CheckStreamIdAndTaskId(const char_t *opMappingInfoStr, uint32_t length) {
    const std::string proto_info(opMappingInfoStr, static_cast<uint64_t>(length));
    OpMappingInfo op_mapping_info;
    GE_ASSERT_TRUE(op_mapping_info.ParseFromString(proto_info));
    for (int32_t id = 0; id < op_mapping_info.task_size(); id++) {
      if ((op_mapping_info.task(id).stream_id() > UINT16_MAX) ||
         (op_mapping_info.task(id).task_id() > UINT16_MAX)) {
        GELOGE(FAILED, "[CheckStreamIdAndTaskId] failed, stream_id: %u task_id: %u",
          op_mapping_info.task(id).stream_id(), op_mapping_info.task(id).task_id());
        return PARAM_INVALID;
      }
    }
    return ge::SUCCESS;
  }
 private:
  bool IsOpMappingInfoClearByModelId(OpMappingInfo &op_mapping_info) {
    // 判断上层有没有传递model_id
    if (op_mapping_info.model_id_param_case() == 0) {
      return false;
    }
    for (auto &unload_op_mapping_info : unload_op_mapping_infos_) {
      if ((unload_op_mapping_info.model_id_param_case() != 0) &&
        (op_mapping_info.model_id() == unload_op_mapping_info.model_id())) {
          GELOGI("Model: %s OpMappingInfo clear by model id: %u success.", op_mapping_info.model_name(),
            op_mapping_info.model_id());
          return true;
        }
    }
    return false;
  }

  bool IsDumpTaskClearByUnloadInfo(const DumpTask &task) const {
    for (auto &unload_op_mapping_info : unload_op_mapping_infos_) {
      for (int32_t id = 0; id < unload_op_mapping_info.task_size(); id++) {
        const DumpTask &unload_task = unload_op_mapping_info.task(id);
        if ((unload_task.stream_id() == task.stream_id()) && (unload_task.task_id() == task.task_id())) {
          return true;
        }
      }
    }
    GELOGE(FAILED, "[CheckDumpTaskClear] failed, op: %s, stream_id: %u, task_id: %u",
      task.op().op_name().c_str(), task.stream_id(), task.task_id());
    return false;
  }

  bool IsOpMappingInfoCleared(OpMappingInfo &op_mapping_info) {
    if (IsOpMappingInfoClearByModelId(op_mapping_info)) {
      return true;
    }
    for (int32_t id = 0; id < op_mapping_info.task_size(); id++) {
      const DumpTask &task = op_mapping_info.task(id);
      if (!IsDumpTaskClearByUnloadInfo(task)) {
        return false;
      }
    }
    return true;
  }

  ge::Status GetOpMappingInfoByOpName(const std::string op_name, OpMappingInfo &op_mapping_info) {
    for (auto &load_op_mapping_info : load_op_mapping_infos_) {
      for (int32_t id = 0; id < load_op_mapping_info.op_mapping_info.task_size(); id++) {
        const DumpTask &task = load_op_mapping_info.op_mapping_info.task(id);
        if (task.op().op_name() == op_name) {
          op_mapping_info = load_op_mapping_info.op_mapping_info;
          return ge::SUCCESS;
        }
      }
    }
    GELOGE(FAILED, "[GetOpMappingInfoByOpName] failed, op: %s is not in load_op_mapping_infos_", op_name.c_str());
    return PARAM_INVALID;
  }

  ge::Status GetDumpTaskByOpName(const std::string op_name, DumpTask &dump_task) {
    for (auto &load_op_mapping_info : load_op_mapping_infos_) {
      for (int32_t id = 0; id < load_op_mapping_info.op_mapping_info.task_size(); id++) {
        const DumpTask &task = load_op_mapping_info.op_mapping_info.task(id);
        if (task.op().op_name() == op_name) {
          dump_task = task;
          return ge::SUCCESS;
        }
      }
    }
    GELOGE(FAILED, "[GetDumpTaskByOpName] failed, op: %s is not in load_op_mapping_infos_", op_name.c_str());
    return PARAM_INVALID;
  }

 private:
  std::vector<OpMappingInfoWithFlag> load_op_mapping_infos_;
  std::vector<OpMappingInfo> unload_op_mapping_infos_;
  friend class DumpCheckRuntimeStub;
  uint32_t *step_id_addr_{nullptr};
  uint32_t step_id_{UINT32_MAX};
};

class DumpCheckRuntimeStub : public gert::RuntimeStubImpl {
 public:
  rtError_t rtDatadumpInfoLoad(const void *dump_info, uint32_t length) {
    GE_ASSERT_NOTNULL(dump_info);
    const char_t *opMappingInfoStr = static_cast<const char_t *>(dump_info);
    return dump_checker_.LoadOpMappingInfo(opMappingInfoStr, length, true);
  }

  rtError_t rtCpuKernelLaunchWithFlag(const void *so_name, const void *kernel_name, uint32_t block_dim,
                                      const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags) {
    std::string kernel(reinterpret_cast<const char *>(kernel_name));
    if (kernel != "DumpDataInfo") {
      if (std::string(__FUNCTION__) == g_runtime_stub_mock) {
        return -1;
      }
      if (kernel == "CheckKernelSupported") {
        const auto *ptr = reinterpret_cast<const CheckKernelSupportedConfig *>(args->args);
        *(reinterpret_cast<int32_t *>(reinterpret_cast<uintptr_t>(ptr->checkResultAddr))) = 0;
      }
      return RT_ERROR_NONE;
    }
    GE_ASSERT_NOTNULL(args->args);
    size_t args_pos = sizeof(aicpu::AicpuParamHead);
    auto proto_arg = *reinterpret_cast<const uint64_t *>(reinterpret_cast<const uint8_t *>(args->args) + args_pos);
    args_pos += sizeof(uint64_t);
    auto size_arg = *reinterpret_cast<const uint64_t *>(reinterpret_cast<const uint8_t *>(args->args) + args_pos);
    auto proto_mem = reinterpret_cast<void *>(proto_arg);
    auto proto_size = *reinterpret_cast<uint64_t *>(size_arg);

    char opMappingInfoStr[proto_size + 1];
    opMappingInfoStr[proto_size] = '\0';
    (void)memcpy_s(opMappingInfoStr, proto_size + 1, proto_mem, proto_size);
    GE_ASSERT_SUCCESS(dump_checker_.CheckStreamIdAndTaskId(opMappingInfoStr, proto_size));
    return dump_checker_.LoadOpMappingInfo(opMappingInfoStr, proto_size, false);
  }

  rtError_t rtGeneralCtrl(uintptr_t *ctrl, uint32_t num, uint32_t type) {
    GELOGI("Enter rtGeneralCtrl.");
    if (type == RT_GNL_CTRL_TYPE_FFTS_PLUS_FLAG) {
      rtFftsPlusTaskInfo_t *task_info = PtrToPtr<void, rtFftsPlusTaskInfo_t>(ValueToPtr(*ctrl));
      const char_t *opMappingInfoStr = static_cast<const char_t *>(task_info->fftsPlusDumpInfo.loadDumpInfo);
      uint32_t length = task_info->fftsPlusDumpInfo.loadDumpInfolen;
      GE_ASSERT_SUCCESS(dump_checker_.LoadOpMappingInfo(opMappingInfoStr, length, true));

      opMappingInfoStr = static_cast<const char_t *>(task_info->fftsPlusDumpInfo.unloadDumpInfo);
      length = task_info->fftsPlusDumpInfo.unloadDumpInfolen;
      GE_ASSERT_SUCCESS(dump_checker_.CheckStreamIdAndTaskId(opMappingInfoStr, length));
      GE_ASSERT_SUCCESS(dump_checker_.LoadOpMappingInfo(opMappingInfoStr, length));
    }
    return RT_ERROR_NONE;
  }

  rtError_t rtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
    if (count == 0) {
      return RT_ERROR_NONE;
    }
    return memcpy_s(dst, dest_max, src, count);
  }

  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
    *dev_ptr = new uint8_t[size];
    memset_s(*dev_ptr, size, 0, size);
    return RT_ERROR_NONE;
  }

  rtError_t rtFree(void *dev_ptr) override {
    if (dev_ptr == nullptr) {
      return RT_ERROR_NONE;
    }
    // RT2 frees step_id_addr before unloading DatadumpInfo.
    if (dev_ptr == dump_checker_.step_id_addr_) {
      GELOGI("step_id_addr_ about to be freed, with value[%u]", *dump_checker_.step_id_addr_);
      dump_checker_.step_id_addr_ = nullptr;
    }
    delete[](uint8_t *) dev_ptr;
    return RT_ERROR_NONE;
  }
  DumpChecker &GetDumpChecker() {
    return dump_checker_;
  }

 private:
  DumpChecker dump_checker_;
};

class DumpCheckAclRuntimeStub : public gert::AclRuntimeStubImpl {
 public:
  rtError_t rtDatadumpInfoLoad(const void *dump_info, uint32_t length) {
    GE_ASSERT_NOTNULL(dump_info);
    const char_t *opMappingInfoStr = static_cast<const char_t *>(dump_info);
    return dump_checker_.LoadOpMappingInfo(opMappingInfoStr, length, true);
  }

  rtError_t rtCpuKernelLaunchWithFlag(const void *so_name, const void *kernel_name, uint32_t block_dim,
                                      const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags) {
    std::string kernel(reinterpret_cast<const char *>(kernel_name));
    if (kernel != "DumpDataInfo") {
      if (std::string(__FUNCTION__) == g_runtime_stub_mock) {
        return -1;
      }
      if (kernel == "CheckKernelSupported") {
        const auto *ptr = reinterpret_cast<const CheckKernelSupportedConfig *>(args->args);
        *(reinterpret_cast<int32_t *>(reinterpret_cast<uintptr_t>(ptr->checkResultAddr))) = 0;
      }
      return RT_ERROR_NONE;
    }
    GE_ASSERT_NOTNULL(args->args);
    size_t args_pos = sizeof(aicpu::AicpuParamHead);
    auto proto_arg = *reinterpret_cast<const uint64_t *>(reinterpret_cast<const uint8_t *>(args->args) + args_pos);
    args_pos += sizeof(uint64_t);
    auto size_arg = *reinterpret_cast<const uint64_t *>(reinterpret_cast<const uint8_t *>(args->args) + args_pos);
    auto proto_mem = reinterpret_cast<void *>(proto_arg);
    auto proto_size = *reinterpret_cast<uint64_t *>(size_arg);

    char opMappingInfoStr[proto_size + 1];
    opMappingInfoStr[proto_size] = '\0';
    (void)memcpy_s(opMappingInfoStr, proto_size + 1, proto_mem, proto_size);
    GE_ASSERT_SUCCESS(dump_checker_.CheckStreamIdAndTaskId(opMappingInfoStr, proto_size));
    return dump_checker_.LoadOpMappingInfo(opMappingInfoStr, proto_size, false);
  }

  rtError_t rtGeneralCtrl(uintptr_t *ctrl, uint32_t num, uint32_t type) {
    GELOGI("Enter rtGeneralCtrl.");
    if (type == RT_GNL_CTRL_TYPE_FFTS_PLUS_FLAG) {
      rtFftsPlusTaskInfo_t *task_info = PtrToPtr<void, rtFftsPlusTaskInfo_t>(ValueToPtr(*ctrl));
      const char_t *opMappingInfoStr = static_cast<const char_t *>(task_info->fftsPlusDumpInfo.loadDumpInfo);
      uint32_t length = task_info->fftsPlusDumpInfo.loadDumpInfolen;
      GE_ASSERT_SUCCESS(dump_checker_.LoadOpMappingInfo(opMappingInfoStr, length, true));

      opMappingInfoStr = static_cast<const char_t *>(task_info->fftsPlusDumpInfo.unloadDumpInfo);
      length = task_info->fftsPlusDumpInfo.unloadDumpInfolen;
      GE_ASSERT_SUCCESS(dump_checker_.CheckStreamIdAndTaskId(opMappingInfoStr, length));
      GE_ASSERT_SUCCESS(dump_checker_.LoadOpMappingInfo(opMappingInfoStr, length));
    }
    return RT_ERROR_NONE;
  }

  aclError aclrtMemcpy(void *dst, size_t dest_max, const void *src, size_t count, aclrtMemcpyKind kind) override {
    if (count == 0) {
      return ACL_SUCCESS;
    }
    return memcpy_s(dst, dest_max, src, count);
  }

  aclError aclrtMalloc(void **dev_ptr, size_t size, aclrtMemMallocPolicy policy) override {
    *dev_ptr = new uint8_t[size];
    memset_s(*dev_ptr, size, 0, size);
    return ACL_SUCCESS;
  }

  aclError aclrtFree(void *dev_ptr) override {
    if (dev_ptr == nullptr) {
      return ACL_SUCCESS;
    }
    // RT2 frees step_id_addr before unloading DatadumpInfo.
    if (dev_ptr == dump_checker_.step_id_addr_) {
      GELOGI("step_id_addr_ about to be freed, with value[%u]", *dump_checker_.step_id_addr_);
      dump_checker_.step_id_addr_ = nullptr;
    }
    delete[] (uint8_t *)dev_ptr;
    return ACL_SUCCESS;
  }

  DumpChecker &GetDumpChecker() {
    return dump_checker_;
  }

 private:
  DumpChecker dump_checker_;
};
}  // namespace ge
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_DUMP_CHECKER_H_
