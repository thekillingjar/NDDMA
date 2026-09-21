/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/state_manager.h"
#include "error_state.h"
#include "idle_state.h"
#include "probe_state.h"
#include "receive_state.h"
#include "send_state.h"
#include "unlink_state.h"
#include "link_state.h"
#include "init_state.h"
#include "destroy_state.h"
#include "receive_transfer_req_state.h"
#include "receive_transfer_cache_state.h"
#include "receive_check_state.h"

namespace FlowFunc {
StateManager &StateManager::GetInstance() {
    static StateManager manager;
    return manager;
}

StateManager::StateManager() {
    static InitState init_state;
    ResigterState(FsmState::kFsmInitState, &init_state, "kFsmInitState");
    static LinkState link_state;
    ResigterState(FsmState::kFsmLinkState, &link_state, "kFsmLinkState");
    static IdleState idle_state;
    ResigterState(FsmState::kFsmIdleState, &idle_state, "kFsmIdleState");
    static ProbeState probe_state;
    ResigterState(FsmState::kFsmProbeState, &probe_state, "kFsmProbeState");
    static ReceiveState receive_state;
    ResigterState(FsmState::kFsmReceiveState, &receive_state, "kFsmReceiveState");
    static SendState send_state;
    ResigterState(FsmState::kFsmSendState, &send_state, "kFsmSendState");
    static UnlinkState unlink_state;
    ResigterState(FsmState::kFsmUnlinkState, &unlink_state, "kFsmUnlinkState");
    static ErrorState error_state;
    ResigterState(FsmState::kFsmErrorState, &error_state, "kFsmErrorState");
    static DestroyState destroy_state;
    ResigterState(FsmState::kFsmDestroyState, &destroy_state, "kFsmDestroyState");
    static ReceiveTransferCacheState receive_transfer_cache_state;
    ResigterState(FsmState::kFsmReceiveTransferCacheState, &receive_transfer_cache_state, "kFsmReceiveTransferCacheState");
    static ReceiveTransferReqState receive_transfer_req_state;
    ResigterState(FsmState::kFsmReceiveTransferReqState, &receive_transfer_req_state, "kFsmReceiveTransferReqState");
    static ReceiveCheckState receive_check_state;
    ResigterState(FsmState::kFsmReceiveCheckState, &receive_check_state, "kFsmReceiveCheckState");
}

void StateManager::ResigterState(FsmState id, BaseState *state, const char *id_desc) {
    state_[static_cast<int32_t>(id)] = state;
    state_desc_[static_cast<int32_t>(id)] = id_desc;
}

BaseState *StateManager::GetState(FsmState id) const {
    BaseState *state = state_[static_cast<size_t>(id)];
    if (state == nullptr) {
        UDF_LOG_ERROR("Fail to get state, id:%d.", static_cast<int32_t>(id));
    }
    return state;
}

const std::string &StateManager::GetStateDesc(FsmState id) const {
    return state_desc_[static_cast<size_t>(id)];
}
}  // namespace FlowFunc