/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/rt_session.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "framework/runtime/model_rt_var_manager.h"
#include "graph/ge_context.h"

namespace gert {
void RtSession::DestroyResources() const {
  // 原来是在ge_executor的unload和rt2的unload单独调用，但因为bundle场景下多个模型是共享同一份rtsession
  // 卸载单个模型不能释放rtsession粒度的资源
  // 所以提供该接口由创建管理rtsession方调用（当前主要是离线场景acl), 类比在线场景下的innersession的销毁
  const uint32_t device_id = ge::GetContext().DeviceId();
  GELOGI("destroy rt session resources, session id %lu, device id %u", session_id_, device_id);
  ge::VarManagerPool::Instance().RemoveVarManager(session_id_);
  gert::RtVarManagerPool::Instance().RemoveRtVarManager(session_id_);
  ge::ExternalWeightManagerPool::Instance().RemoveManager(session_id_);
  ge::SessionMemAllocator<ge::ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(session_id_, device_id);
  ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id_, device_id);
  ge::SessionMemAllocator<ge::ActiveMemoryAllocator>::Instance().RemoveAllocator(session_id_, device_id);
}

}  // namespace gert