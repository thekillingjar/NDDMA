/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_FUNC_WRAPPER_H
#define FLOW_FUNC_FUNC_WRAPPER_H

#include <memory>
#include "flow_func/meta_params.h"
#include "flow_func/meta_run_context.h"
#include "flow_func/flow_msg.h"
#include "flow_func/flow_msg_queue.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FuncWrapper {
public:
    FuncWrapper() = default;

    virtual ~FuncWrapper() = default;

    virtual int32_t Init(
        const std::shared_ptr<MetaParams> &meta_param, const std::shared_ptr<MetaRunContext> &run_context) = 0;

    virtual int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) = 0;

    virtual int32_t ResetFlowFuncState() = 0;

    virtual int32_t Proc(const std::vector<std::shared_ptr<FlowMsgQueue>> &input_queues) = 0;
};
}  // namespace FlowFunc

#endif  // FLOW_FUNC_FUNC_WRAPPER_H
