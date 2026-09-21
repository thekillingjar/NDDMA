/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/task_def_faker.h"
#include "framework/common/taskdown_common.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
namespace {
std::array<ge::ccKernelType, TaskDefFaker::kKernelTypeEnd> kernel_types_map = {
    ge::ccKernelType::TE,          ge::ccKernelType::AI_CPU, ge::ccKernelType::AI_CPU,
    ge::ccKernelType::CUST_AI_CPU, ge::ccKernelType::DVPP,   ge::ccKernelType::CCE_AI_CORE,
};

std::array<ge::ModelTaskType, TaskDefFaker::kTaskTypeEnd> task_types_map = {
    ge::ModelTaskType::MODEL_TASK_ALL_KERNEL,
    ge::ModelTaskType::MODEL_TASK_KERNEL,
    ge::ModelTaskType::MODEL_TASK_KERNEL_EX,
    ge::ModelTaskType::MODEL_TASK_KERNEL,
    ge::ModelTaskType::MODEL_TASK_KERNEL,
    ge::ModelTaskType::MODEL_TASK_DSA,
    ge::ModelTaskType::MODEL_TASK_HCCL,
    ge::ModelTaskType::MODEL_TASK_MEMCPY_ASYNC,                  // kRts
    ge::ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX,  // kLabelSwitch
    ge::ModelTaskType::MODEL_TASK_EVENT_RECORD, // kEvent
};
}  // namespace

TaskDefFaker &TaskDefFaker::AddTask(const TaskTypeInfo &taskInfo) {
  task_infos.emplace_back(taskInfo);
  return *this;
}

std::vector<domi::TaskDef> TaskDefFaker::CreateTaskDef(uint64_t op_index) {
  std::vector<domi::TaskDef> task_defs(task_infos.size());
  for (size_t i = 0; i < task_defs.size(); ++i) {
    auto task_type = task_infos[i].task_type;
    auto kernel_type = task_infos[i].kernel_type;
    task_defs[i].set_type(static_cast<uint32_t>(task_types_map[task_type]));
    uint16_t offset = 0;
    uint64_t addrs[20] = {0};
    if (task_type == kWithHandle) {
      task_defs[i].mutable_kernel_with_handle()->mutable_context()->set_op_index(op_index);
      task_defs[i].mutable_kernel_with_handle()->mutable_context()->set_kernel_type(
          static_cast<uint32_t>(kernel_types_map[kernel_type]));
      task_defs[i].mutable_kernel_with_handle()->mutable_context()->set_args_offset(&offset, sizeof(offset));
      task_defs[i].mutable_kernel_with_handle()->set_args(addrs, sizeof(addrs));
      task_defs[i].mutable_kernel_with_handle()->set_block_dim(8);
    } else if (task_type == kWithoutHandle || task_type == kCCAicpu) {
      task_defs[i].mutable_kernel()->mutable_context()->set_op_index(op_index);
      task_defs[i].mutable_kernel()->mutable_context()->set_kernel_type(
          static_cast<uint32_t>(kernel_types_map[kernel_type]));
      task_defs[i].mutable_kernel()->mutable_context()->set_args_offset(&offset, sizeof(offset));
      task_defs[i].mutable_kernel()->set_args(addrs, sizeof(addrs));
      task_defs[i].mutable_kernel()->set_block_dim(8);
      task_defs[i].mutable_kernel()->set_kernel_name("name");
    } else if (task_type == kTfAicpu) {
      task_defs[i].mutable_kernel_ex()->set_op_index(op_index);
      task_defs[i].mutable_kernel()->mutable_context()->set_kernel_type(
          static_cast<uint32_t>(kernel_types_map[kernel_type]));
      task_defs[i].mutable_kernel()->set_kernel_name("name");
    } else if (task_type == kDvppTask) {
      task_defs[i].mutable_kernel()->mutable_context()->set_op_index(op_index);
      task_defs[i].mutable_kernel()->mutable_context()->set_kernel_type(
          static_cast<uint32_t>(kernel_types_map[kernel_type]));
      task_defs[i].mutable_kernel()->set_kernel_name("name");
    } else if (task_type == kDsaRandomNormal) {
      task_defs[i].mutable_kernel()->mutable_context()->set_op_index(op_index);
      task_defs[i].mutable_kernel()->mutable_context()->set_kernel_type(
          static_cast<uint32_t>(kernel_types_map[kernel_type]));
      task_defs[i].mutable_kernel()->set_kernel_name("name");
    } else if (task_type == kHccl) {
      auto &kernel_hccl_def = *task_defs[i].mutable_kernel_hccl();
      kernel_hccl_def.set_op_index(op_index);
      kernel_hccl_def.set_hccl_type("HcomBroadcast");
    } else if (task_type == kRts) {
      auto &kernel_rts_def = *(task_defs[i].mutable_memcpy_async());
      kernel_rts_def.set_op_index(op_index);
    } else if (task_type == kLabelSwitch) {
      task_defs[i].mutable_label_switch_by_index()->set_op_index(op_index);
    } else if (task_type == kEvent) {
      task_defs[i].mutable_event_ex()->set_op_index(std::numeric_limits<uint32_t>::max());
    } else {
      throw std::invalid_argument("Invalid task type");
    }
  }

  return task_defs;
}

std::unique_ptr<TaskDefFaker> TaskDefFaker::Clone() const {
  return std::unique_ptr<TaskDefFaker>(new TaskDefFaker(*this));
}
}  // namespace gert
