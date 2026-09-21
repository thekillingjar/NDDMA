/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/dvpp/dvpp_task_info.h"

#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "opskernel_executor/ops_kernel_executor_manager.h"
#include "common/runtime_api_wrapper.h"


namespace ge {
Status DvppTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                          const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                          const IowAddrs &iow_addrs) {
  GELOGI("DvppTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  const domi::DvppTaskDef &dvpp_task = task_def.dvpp_task();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(dvpp_task.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Can't get op_desc from davinci_model by index:%u", dvpp_task.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range!", dvpp_task.op_index());
    return INTERNAL_ERROR;
  }

  ops_kernel_store_ = op_desc->TryGetExtAttr<OpsKernelExecutor *>("OpsKernelInfoStorePtr", nullptr);
  if ((ops_kernel_store_ == nullptr) &&
      (OpsKernelExecutorManager::GetInstance().GetExecutor(op_desc->GetOpKernelLibName(), ops_kernel_store_) !=
       SUCCESS)) {
    REPORT_INNER_ERR_MSG("E19999", "Can't get op_desc from davinci_model by index:%u", dvpp_task.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range!", dvpp_task.op_index());
    return INTERNAL_ERROR;
  }
  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  const std::vector<void *> input_data_addrs = ModelUtils::GetInputAddrs(rts_param, op_desc);
  const std::vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  const std::vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
  (void)io_addrs_.insert(io_addrs_.cend(), input_data_addrs.cbegin(), input_data_addrs.cend());
  (void)io_addrs_.insert(io_addrs_.cend(), output_data_addrs.cbegin(), output_data_addrs.cend());
  (void)io_addrs_.insert(io_addrs_.cend(), workspace_data_addrs.cbegin(), workspace_data_addrs.cend());

  dvpp_info_.op_desc = op_desc;
  dvpp_info_.io_addrs = io_addrs_;
  ge_task_.dvpp_info = dvpp_info_;
  const auto result = ops_kernel_store_->LoadTask(ge_task_);
  if (result != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call ops_kernel_info_store LoadTask fail");
    GELOGE(INTERNAL_ERROR, "[Load][Task] fail, return ret:%u", result);
    return INTERNAL_ERROR;
  }
  GELOGI("DvppTaskInfo %s Init Success, logic stream id: %u, stream: %p",
    op_desc->GetNamePtr(), task_def.stream_id(), stream_);
  return SUCCESS;
}

Status DvppTaskInfo::Distribute() {
  GELOGI("DvppTaskInfo Distribute Start.");
  GE_CHK_RT_RET(
      ge::rtStarsTaskLaunch(ge_task_.dvpp_info.sqe, static_cast<uint32_t>(sizeof(ge_task_.dvpp_info.sqe)), stream_));
  is_support_redistribute_ = true;
  GELOGI("DvppTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_DVPP, DvppTaskInfo);
}  // namespace ge
