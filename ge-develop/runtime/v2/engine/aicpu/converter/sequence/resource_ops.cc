/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "exe_graph/runtime/kernel_context.h"
#include "register/kernel_registry_impl.h"
#include "tensor_sequence.h"
#include "resource_mgr.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
ge::graphStatus CreateSession(KernelContext* context) {
  GELOGD("CreateSession begin");
  auto session_id = context->GetInputPointer<uint64_t>(0UL);
  if (session_id == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get input session_id failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get input session_id failed.");
    return ge::PARAM_INVALID;
  }
  auto output = context->GetOutputPointer<uint64_t>(0UL);
  if (output == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get output failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get output failed.");
    return ge::PARAM_INVALID;
  }

  *output = *session_id;
  GELOGD("CreateSession session_id:%llu", *session_id);
  return SessionMgr::GetInstance()->CreateSession(*session_id);
}
REGISTER_KERNEL(CreateSession).RunFunc(CreateSession);

uint32_t DestroySession(KernelContext* context) {
  GELOGD("DestroySession begin");
  auto session_id = context->GetInputPointer<uint64_t>(0UL);
  if (session_id == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get input session_id failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get input session_id failed.");
    return ge::PARAM_INVALID;
  }
  GELOGD("DestroySession session_id:%llu", *session_id);
  return SessionMgr::GetInstance()->DestroySession(*session_id);
}
REGISTER_KERNEL(DestroySession).RunFunc(DestroySession);

uint32_t ClearContainer(KernelContext* context) {
  GELOGD("ClearContainer begin");
  auto session_id = context->GetInputPointer<uint64_t>(0UL);
  if (session_id == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get input session_id failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get input session_id failed.");
    return ge::PARAM_INVALID;
  }
  auto container_id = context->GetInputPointer<uint64_t>(1UL);
  if (container_id == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get input container_id failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get input container_id failed.");
    return ge::PARAM_INVALID;
  }

  GELOGD("ClearContainer session_id:%llu container_id:%llu", *session_id, *container_id);
  ResourceMgrPtr rm;
  SessionMgr::GetInstance()->GetRm(*session_id, *container_id, rm);
  return rm->ClearStepResource();
}
REGISTER_KERNEL(ClearContainer).RunFunc(ClearContainer);
}  // namespace aicpu