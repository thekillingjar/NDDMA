/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_FLOW_MSG_QUEUE_H
#define FLOW_FUNC_FLOW_MSG_QUEUE_H

#include <vector>
#include "flow_func_defines.h"
#include "tensor_data_type.h"
#include "flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowMsgQueue {
 public:
  FlowMsgQueue() = default;

  ~FlowMsgQueue() = default;

  virtual int32_t Dequeue(std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout = -1) {
    if (timeout == 2) {
      return FLOW_FUNC_STATUS_REDEPLOYING;
    } else if (timeout == 3) {
      return FLOW_FUNC_ERR_TIME_OUT_ERROR;
    } else if (timeout == 4) {
      return FLOW_FUNC_ERR_QUEUE_ERROR;
    } else {
      flow_msg = std::make_shared<FlowMsg>();
      return FLOW_FUNC_SUCCESS;
    }
  }

  virtual int32_t Depth() const {
    return 128;
  }

  virtual int32_t Size() const {
    return 10;
  }
};
}  // namespace FlowFunc

#endif  // FLOW_FUNC_FLOW_MSG_QUEUE_H
