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

#include "flow_func_defines.h"
#include "flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowMsgQueue {
public:
    FlowMsgQueue() = default;

    virtual ~FlowMsgQueue() = default;

    /**
     * @brief dequeue flowMsg from input queue with timeout.
     * @param flowMsg: output flowMsg
     * @param timeout: default is -1, means never timeout
     * @return FLOW_FUNC_SUCCESS: success, other:failed.
     */
    virtual int32_t Dequeue(std::shared_ptr<FlowMsg> &flowMsg, int32_t timeout = -1) = 0;

    /**
     * @brief query input queue depth(maxsize).
     * @return queue depth
     */
    virtual int32_t Depth() const = 0;

    /**
     * @brief query current item num of input queue.
     * @return item num
     */
    virtual int32_t Size() const = 0;
};
}

#endif // FLOW_FUNC_FLOW_MSG_QUEUE_H
