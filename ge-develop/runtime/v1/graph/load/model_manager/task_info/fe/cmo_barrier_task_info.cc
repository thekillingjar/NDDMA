/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/cmo_barrier_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/runtime_api_wrapper.h"

namespace ge {
Status CmoBarrierTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                const IowAddrs &iow_addrs) {
  GELOGI("CmoBarrierTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  const domi::CmoBarrierTaskDef &cmo_barrier_task_def = task_def.cmo_barrier_task();
  barrier_task_info_.logicIdNum = static_cast<uint8_t>(cmo_barrier_task_def.logic_id_num());
  const int32_t barrier_info_size = cmo_barrier_task_def.barrier_info_size();
  if (barrier_info_size > static_cast<int32_t>(RT_CMO_MAX_BARRIER_NUM)) {
    GELOGE(PARAM_INVALID, "Current barrier info size is %d, should not > %d", barrier_info_size,
           static_cast<int32_t>(RT_CMO_MAX_BARRIER_NUM));
    return PARAM_INVALID;
  }
  for (int32_t index = 0; index < barrier_info_size; ++index) {
    const domi::CmoBarrierInfoDef &barrier_info_def = cmo_barrier_task_def.barrier_info(index);
    barrier_task_info_.cmoInfo[index].cmoType = static_cast<uint16_t>(barrier_info_def.cmo_type());
    barrier_task_info_.cmoInfo[index].logicId = barrier_info_def.logic_id();
  }

  GELOGI("CmoBarrierTaskInfo Init Success, logic stream id: %u, stream: %p", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status CmoBarrierTaskInfo::Distribute() {
  GELOGI("CmoBarrierTaskInfo Distribute Start.");
  GE_CHK_RT_RET(ge::rtBarrierTaskLaunch(&barrier_task_info_, stream_, 0U));
  is_support_redistribute_ = true;
  GELOGI("CmoBarrierTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_BARRIER, CmoBarrierTaskInfo);
}  // namespace ge
