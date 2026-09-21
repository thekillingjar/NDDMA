/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/kernel_registry.h"
#include "common/checker.h"
#include "framework/runtime/rt_session.h"
namespace gert {
ge::graphStatus GetSessionId(KernelContext *context) {
  auto rt_session = context->GetInputValue<RtSession *>(0);
  GE_ASSERT_NOTNULL(rt_session);
  auto session_id_ptr = context->GetOutputPointer<uint64_t>(0);
  GE_ASSERT_NOTNULL(session_id_ptr);
  *session_id_ptr = rt_session->GetSessionId();
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetSessionId).RunFunc(GetSessionId);
}
