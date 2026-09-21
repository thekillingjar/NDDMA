/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/event_mem_wait_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
Status EventMemWaitTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                               const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                               const IowAddrs &iow_addrs) {
  mem_event_id_ = task_def.event_id();
  GELOGI("EventMemWaitTaskInfo Init Start mem_event_id_ %u", mem_event_id_);
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  if (task_def.has_event_ex()) {
    op_index_ = task_def.event_ex().op_index();
    event_type_ = task_def.event_ex().event_type();
    GELOGI("EventMemWaitTaskInfo get op_index_ %u, event_type_ %u", op_index_, event_type_);
  }

  return SUCCESS;
}

Status EventMemWaitTaskInfo::Distribute() {
  GELOGI("EventMemWaitTaskInfo Distribute Start event_id %u", mem_event_id_);
  const auto op_desc = davinci_model_->GetOpByIndex(op_index_);
  if (op_desc != nullptr) {
    SetTaskTag(op_desc->GetName().c_str());
  }
  void *cur_mem = davinci_model_->GetMemEventIdAddr(mem_event_id_);
  GE_ASSERT_NOTNULL(cur_mem);

  rtError_t rt_ret = rtsValueWait(cur_mem, 1, 1, stream_);
  GE_ASSERT_TRUE((rt_ret == RT_ERROR_NONE), "Call rtsValueWait failed, ret:%d", rt_ret);

  if (op_desc != nullptr) {
    SetTaskTag(op_desc->GetName().c_str());
  }
  // reset mem event ,use rtMemWriteVal instead of rtMemsetAsync due to performance
  rt_ret = rtsValueWrite(cur_mem, 0, 0, stream_);
  GE_ASSERT_TRUE((rt_ret == RT_ERROR_NONE), "Call rtsValueWrite failed, ret:%d", rt_ret);

  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_MEM_EVENT_WAIT, EventMemWaitTaskInfo);
}  // namespace ge
