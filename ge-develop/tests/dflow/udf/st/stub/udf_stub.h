/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_ST_DRV_STUB_H
#define UDF_ST_DRV_STUB_H

#include "flow_func/mbuf_flow_msg.h"

namespace FlowFunc {

constexpr uint32_t UDF_ST_QUEUE_MAX_DEPTH = 10;

#pragma pack(push, 1)
struct MbufStImpl {
  uint32_t mbuf_size;
  uint32_t reserve; // just for make sure data ptr aligned by 8
  uint8_t reserve_head[256 - sizeof(MbufHeadMsg)];
  MbufHeadMsg head_msg;
  RuntimeTensorDesc tensor_desc;
  uint8_t data[4096];
};

#pragma pack(pop)

void ClearStubEschedEvents();

uint32_t CreateQueue(uint32_t depth = UDF_ST_QUEUE_MAX_DEPTH);
void DestroyQueue(uint32_t queue_id);

int32_t LinkQueueInTest(uint32_t queue_id, uint32_t linked_queue_id);

void UnlinkQueueInTest(uint32_t queue_id);
}  // namespace FlowFunc
#endif  // UDF_ST_DRV_STUB_H
