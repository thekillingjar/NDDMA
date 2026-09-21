/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/cmo_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "common/runtime_api_wrapper.h"

namespace ge {
Status CmoTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                         const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                         const IowAddrs &iow_addrs) {
  GELOGI("CmoTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  const domi::CmoTaskDef &cmo_task_def = task_def.cmo_task();
  cmo_task_info_.cmoType = static_cast<uint16_t>(cmo_task_def.cmo_type());
  cmo_task_info_.logicId = cmo_task_def.logic_id();
  cmo_task_info_.qos = static_cast<uint8_t>(cmo_task_def.qos());
  cmo_task_info_.partId = static_cast<uint8_t>(cmo_task_def.part_id());
  cmo_task_info_.pmg = static_cast<uint8_t>(cmo_task_def.pmg());
  cmo_task_info_.opCode = static_cast<uint16_t>(cmo_task_def.op_code());
  cmo_task_info_.numInner = static_cast<uint16_t>(cmo_task_def.num_inner());
  cmo_task_info_.numOuter = static_cast<uint16_t>(cmo_task_def.num_outer());
  cmo_task_info_.lengthInner = cmo_task_def.length_inner();
  const auto &rts_param = davinci_model->GetRuntimeParam();
  uint8_t *source_mem_addr = nullptr;
  if (ModelUtils::GetRtAddress(rts_param, static_cast<uintptr_t>(cmo_task_def.source_addr()),
                               source_mem_addr) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress]failed, logic write addr base is 0x%" PRIx64 ".",
      cmo_task_def.source_addr());
    return INTERNAL_ERROR;
  }
  cmo_task_info_.sourceAddr = PtrToValue(source_mem_addr);
  cmo_task_info_.striderOuter = cmo_task_def.strider_outer();
  cmo_task_info_.striderInner = cmo_task_def.strider_inner();
  GELOGI("cmoType[%u], logicId[%u], qos[%u], partid[%u], pmg[%u], opCode[%u], numInner[%u], numOuter[%u], "
         "lengthInner[%u], striderOuter[%u], striderInner[%u]", cmo_task_def.cmo_type(), cmo_task_def.logic_id(),
         cmo_task_def.qos(), cmo_task_def.part_id(), cmo_task_def.pmg(), cmo_task_def.op_code(),
         cmo_task_def.num_inner(), cmo_task_def.num_outer(), cmo_task_def.length_inner(), cmo_task_def.strider_outer(),
         cmo_task_def.strider_inner());

  GELOGI("CmoTaskInfo Init Success, logic stream id: %u, stream: %p", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status CmoTaskInfo::Distribute() {
  GELOGI("CmoTaskInfo Distribute Start.");
  GE_CHK_RT_RET(ge::rtCmoTaskLaunch(&cmo_task_info_, stream_, 0U));
  is_support_redistribute_ = true;
  GELOGI("CmoTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_CMO, CmoTaskInfo);
}  // namespace ge
