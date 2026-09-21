/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/npu_get_float_status_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "common/checker.h"
#include "common/runtime_api_wrapper.h"

namespace ge {
Status NpuGetFloatStatusTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                       const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                       const IowAddrs &iow_addrs) {
  GE_CHECK_NOTNULL(davinci_model);
  (void)persistent_workspace;
  davinci_model_ = davinci_model;
  GELOGI("NpuGetFloatStatusTaskInfo Init Start.");
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));
  const domi::NpuGetFloatStatusDef &npu_get_float_status = task_def.npu_get_float_status();
  output_size_ = npu_get_float_status.output_size();
  mode_ = npu_get_float_status.mode();
  const int64_t op_index = ParseOpIndex(task_def);
  const OpDescPtr op_desc = davinci_model_->GetOpByIndex(static_cast<uint32_t>(op_index));
  GE_CHECK_NOTNULL(op_desc);
  // todo: model args manager 完成后，此处需增加对 iow_addrs中output_logic_addrs size的校验
  // todo: 0 is ArgsPlacement::kArgsPlacementHbm
  GE_ASSERT_TRUE((args[0].dev_addr != 0U), "[Check][Param] dev addr is nullptr.");
  args_ = ValueToPtr(args[0U].dev_addr);
  output_addr_mem_type_ = iow_addrs.output_logic_addrs[0].memory_type;;
  output_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(iow_addrs.output_logic_addrs[0].logic_addr));
  GELOGD("op_name:%s, output_addr 0x%llx, mem_type 0x%llx, args %p",
      op_desc->GetName().c_str(), PtrToValue(output_addr_), output_addr_mem_type_, args_);
  io_addrs_.emplace_back(output_addr_);
  io_addr_mem_types_.emplace_back(output_addr_mem_type_);
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), VPtrToValue(io_addrs_),
      io_addr_mem_types_, {op_desc->GetName(), op_desc->GetType()}));
  GELOGI("NpuGetFloatStatusTaskInfo Init Success, logic stream id: %u, stream: %p.", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status NpuGetFloatStatusTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                                    TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);
  constexpr uint32_t args_size = static_cast<uint32_t>(sizeof(uint8_t *));
  task_run_param.args_descs.push_back({static_cast<int64_t>(args_size), ArgsPlacement::kArgsPlacementHbm});
  const domi::NpuGetFloatStatusDef &npu_get_float_status = task_def.npu_get_float_status();
  const auto ret = ModelUtils::GetRtAddress(davinci_model->GetRuntimeParam(),
                                            static_cast<uintptr_t>(npu_get_float_status.output_addr()),
                                            output_addr_, output_addr_mem_type_);
  if (ret != SUCCESS) {
    return ret;
  }
  task_run_param.parsed_output_addrs.push_back({PtrToValue(output_addr_), output_addr_mem_type_, true, {0}});

  GELOGD("output_addr 0x%llx, mem_type 0x%llx", PtrToValue(output_addr_), output_addr_mem_type_);
  return SUCCESS;
}

Status NpuGetFloatStatusTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL);
  return SUCCESS;
}

Status NpuGetFloatStatusTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                                                 void *const host_args,
                                                 const size_t host_args_max_len) {
  GELOGI("NpuGetFloatStatusTaskInfo::UpdateArgs in.");
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr, host_args, host_args_max_len));
  GELOGI("NpuGetFloatStatusTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status NpuGetFloatStatusTaskInfo::Distribute() {
  GELOGI("NpuGetFloatStatusTaskInfo Distribute Start.");
  GE_CHK_RT_RET(ge::rtNpuGetFloatStatus(args_, output_size_, mode_, stream_));
  is_support_redistribute_ = true;
  GELOGI("NpuGetFloatStatusTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

int64_t NpuGetFloatStatusTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const auto &npu_get_float_status = task_def.npu_get_float_status();
  return static_cast<int64_t>(npu_get_float_status.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_NPU_GET_FLOAT_STATUS, NpuGetFloatStatusTaskInfo);
}  // namespace ge
