/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_BUILT_IN_FSM_STATE_MANAGER_H
#define UDF_BUILT_IN_FSM_STATE_MANAGER_H
#include <string>
#include "fsm/base_state.h"
#include "fsm/state_define.h"

namespace FlowFunc {
class StateManager {
public:
    static StateManager &GetInstance();
    virtual ~StateManager() = default;

    void ResigterState(FsmState id, BaseState *state, const char *id_desc);
    BaseState *GetState(FsmState id) const;
    const std::string &GetStateDesc(FsmState id) const;
    StateManager(const StateManager &) = delete;
    StateManager(const StateManager &&) = delete;
    StateManager &operator=(const StateManager &) = delete;
    StateManager &operator=(const StateManager &&) = delete;

private:
    StateManager();

    BaseState *state_[static_cast<size_t>(FsmState::kFsmInvalidState)] = {nullptr};
    std::string state_desc_[static_cast<size_t>(FsmState::kFsmInvalidState)] = {""};
};
}  // namespace FlowFunc
#endif  // UDF_BUILT_IN_FSM_STATE_MANAGER_H
