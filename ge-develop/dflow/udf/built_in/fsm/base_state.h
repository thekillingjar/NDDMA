/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_BUILT_IN_FSM_BASE_STATE_H
#define UDF_BUILT_IN_FSM_BASE_STATE_H

#include "fsm/state_define.h"

namespace FlowFunc {
class BaseState {
public:
    BaseState() noexcept = default;
    virtual ~BaseState() = default;

    virtual FsmStatus Preprocess(LlmCommEntity &entity) = 0;
    virtual FsmStatus Process(LlmCommEntity &entity) = 0;
    virtual FsmStatus Postprocess(LlmCommEntity &entity) = 0;

    BaseState(const BaseState &) = delete;
    BaseState(const BaseState &&) = delete;
    BaseState &operator=(const BaseState &) = delete;
    BaseState &operator=(const BaseState &&) = delete;
};
}  // namespace FlowFunc

#endif  // UDF_BUILT_IN_FSM_BASE_STATE_H
