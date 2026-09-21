/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/fusion_stop_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
Status FusionStopTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                const IowAddrs &iow_addrs) {
  GELOGI("FusionStopTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  GELOGI("FusionStopTaskInfo Init Success, logic stream id: %u, stream: %p.", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status FusionStopTaskInfo::Distribute() {
  GELOGI("FusionStopTaskInfo Distribute Start.");
  const rtError_t rt_ret = rtKernelFusionEnd(stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtKernelFusionEnd failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtKernelFusionEnd] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  is_support_redistribute_ = true;
  GELOGI("FusionStopTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_FUSION_END, FusionStopTaskInfo);
}  // namespace ge
