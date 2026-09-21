/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_MODEL_H
#define FLOW_MODEL_H

#include <vector>
#include <memory>
#include "flow_func/flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowModel {
public:
    FlowModel() = default;

    virtual ~FlowModel() = default;

    virtual int32_t Init() = 0;

    virtual int32_t Run(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
        std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) = 0;

    virtual void AddExceptionTransId(uint64_t trans_id) = 0;

    virtual void DeleteExceptionTransId(uint64_t trans_id) = 0;
};
}
#endif // FLOW_MODEL_H
