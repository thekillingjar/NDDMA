/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_BUILT_IN_FSM_STATE_DEFINE_H
#define UDF_BUILT_IN_FSM_STATE_DEFINE_H

#include <memory>
#include "common/udf_log.h"
#include "flow_func/flow_func_log.h"
#include "hccl/base.h"

namespace FlowFunc {
constexpr const char *kMaxTimeCost = " with max time cost";
constexpr const char *kCommonTimeCost = "";
class LlmCommEntity;
using EntityPtr = std::shared_ptr<LlmCommEntity>;

enum class FsmStatus : int32_t {
    kFsmSuccess = 0,
    kFsmTimeout = 1,
    // kv not exist, sync kv/merge kv
    kFsmKvNotExist = 2,
    // repetitive remoteClusterId in update link req, link/unlink
    kFsmRepeatRequest = 3,
    // align to FSM_REQUEST_ALREADY_COMPLETED = 4,
    kFsmParamInvalid = 5,
    // align to FSM_ENGINE_FINALIZED = 6,
    // decoder not linked with prompt, sync kv/unlink
    kFsmNotLink = 7,
    // decoder already linked with prompt, link
    kFsmAlreadyLinked = 8,
    // decoder link with prompt failed, link
    kFsmLinkFailed = 9,
    // decoder unlink with prompt failed, unlink
    kFsmUnlinkFailed = 10,
    // decoder notify prompt unlink failed, unlink or link
    kFsmNotifyPromptUnlinkFailed = 11,
    // req count over limit, link/unlink
    kFsmReqOverLimit = 12,
    //  align to FSM_NOTIFY_PROMPT_CHECK_FAILED = 13,
    // func execute failed
    kFsmOutOfMemory = 14,
    // prefix has already existed
    kFsmPrefixAlreadyExist = 15,
    kFsmPrefixNotExist = 16,
    kFsmCacheIncompatible = 20,
    kFsmCacheKeyAlreadyExist = 21,
    kFsmCopyKvFailed = 22,
    kFsmCacheIdAlreadyExist = 23,
    kFsmExistLink = 24,
    // LLM_FEATURE_NOT_ENABLED
    // LLM_TIMEOUT
    kFsmLinkBusy = 27,
    kFsmFailed = 1000,
    kFsmHcclFailed = 1001,
    kFsmMbufFailed = 1002,
    kFsmDrvFailed = 1003,
    kFsmKeepState = 1004,
    kFsmEstablishLinkSuc = 1005,
    kFsmIgnore,
};

enum class FsmState : int32_t {
    kFsmInitState = 0,
    kFsmLinkState,
    kFsmIdleState,
    kFsmProbeState,
    kFsmReceiveTransferReqState,
    kFsmReceiveTransferCacheState,
    kFsmReceiveCheckState,
    kFsmReceiveState,
    kFsmSendState,
    kFsmUnlinkState,
    kFsmDestroyState,
    kFsmErrorState,
    kFsmInvalidState
};

enum class EntityType : int32_t {
    kEntityClient = 0,
    kEntityServer
};

} // namespace FlowFunc

#endif // UDF_BUILT_IN_FSM_STATE_DEFINE_H
