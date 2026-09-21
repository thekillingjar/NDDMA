/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_TASK_RUN_PARAM_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_TASK_RUN_PARAM_FAKER_H_
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/model_utils.h"
namespace ge {
class TaskRunParamFaker {
 public:
  TaskRunParamFaker &Input(uint64_t logical_addr, uint64_t memory_type, bool support_refresh) {
    param_.parsed_input_addrs.push_back({logical_addr, memory_type, support_refresh, {}});
    return *this;
  }
  TaskRunParamFaker &Output(uint64_t logical_addr, uint64_t memory_type, bool support_refresh) {
    param_.parsed_output_addrs.push_back({logical_addr, memory_type, support_refresh, {}});
    return *this;
  }
  TaskRunParamFaker &Workspace(uint64_t logical_addr, uint64_t memory_type, bool support_refresh) {
    param_.parsed_workspace_addrs.push_back({logical_addr, memory_type, support_refresh, {}});
    return *this;
  }
  TaskRunParamFaker &Inputs(std::initializer_list<uint64_t> logical_addrs, uint64_t memory_type, bool support_refresh) {
    for (auto addr : logical_addrs) {
      Input(addr, memory_type, support_refresh);
    }
    return *this;
  }
  TaskRunParamFaker &Outputs(std::initializer_list<uint64_t> logical_addrs, uint64_t memory_type,
                             bool support_refresh) {
    for (auto addr : logical_addrs) {
      Output(addr, memory_type, support_refresh);
    }
    return *this;
  }
  TaskRunParamFaker &RefreshInput(uint64_t logical_addr, uint64_t memory_type) {
    return Input(logical_addr, memory_type, true);
  }
  TaskRunParamFaker &RefreshHbmInput(uint64_t logical_addr) {
    return Input(logical_addr, RT_MEMORY_HBM, true);
  }
  TaskRunParamFaker &RefreshWeightInput(uint64_t logical_addr) {
    return Input(logical_addr, kWeightMemType, true);
  }
  TaskRunParamFaker &RefreshOutput(uint64_t logical_addr, uint64_t memory_type) {
    return Output(logical_addr, memory_type, true);
  }
  TaskRunParamFaker &RefreshHbmOutput(uint64_t logical_addr) {
    return Output(logical_addr, RT_MEMORY_HBM, true);
  }
  TaskRunParamFaker &RefreshWeightOutput(uint64_t logical_addr) {
    return Output(logical_addr, kWeightMemType, true);
  }
  TaskRunParamFaker &RefreshWorkspace(uint64_t logical_addr, uint64_t memory_type) {
    return Workspace(logical_addr, memory_type, true);
  }
  TaskRunParamFaker &RefreshHbmWorkspace(uint64_t logical_addr) {
    return Workspace(logical_addr, RT_MEMORY_HBM, true);
  }
  TaskRunParamFaker &TaskArgs(int64_t len, ArgsPlacement placement) {
    param_.args_descs.emplace_back(TaskArgsDesc{len, placement});
    return *this;
  }
  TaskRunParamFaker &HbmTaskArgsLen(int64_t len) {
    return TaskArgs(len, ArgsPlacement::kArgsPlacementHbm);
  }
  TaskRunParamFaker &TsTaskArgsLen(int64_t len) {
    return TaskArgs(len, ArgsPlacement::kArgsPlacementTs);
  }
  TaskRunParamFaker &SqeTaskArgsLen(int64_t len) {
    return TaskArgs(len, ArgsPlacement::kArgsPlacementSqe);
  }
  TaskRunParamFaker &HostSvmTaskArgsLen(int64_t len) {
    return TaskArgs(len, ArgsPlacement::kArgsPlacementHostSvm);
  }
  TaskRunParam Build() const {
    return param_;
  }

 private:
  TaskRunParam param_ = {{}, {}, {}, {{0, ArgsPlacement::kArgsPlacementHbm}}};
};
}  // namespace ge

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_TASK_RUN_PARAM_FAKER_H_
