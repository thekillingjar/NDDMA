/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/probe_state.h"
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/llm_common.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr const int32_t kProbeNoEnvelopeFlag = 0;
constexpr const int32_t kUnlinkMsgSize = 0;
}  // namespace
FsmStatus ProbeState::Preprocess(LlmCommEntity &entity) {
    UDF_LOG_DEBUG("Enter probe state, entity:%s.", entity.GetDesc().c_str());
    return Process(entity);
}

FsmStatus ProbeState::ProbeAndGetCountAsync(LlmCommEntity &entity, int32_t &count) {
    int32_t flag = kProbeNoEnvelopeFlag;
    HcclMessage msg{};
    HcclStatus status{};
    entity.GetStatisticInfo().call_probe_total_times++;
    HcclResult ret = HcclRawImprobe(entity.GetConn(), &flag, &msg, &status);
    if (ret != HCCL_SUCCESS) {
        entity.GetStatisticInfo().call_probe_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawImprobe, entity:%s, ret:%d.", entity.GetDesc().c_str(), ret);
        return FsmStatus::kFsmHcclFailed;
    }
    if (flag == kProbeNoEnvelopeFlag) {
        return FsmStatus::kFsmKeepState;
    }
    entity.GetStatisticInfo().call_probe_success_times++;
    entity.GetProbeMsgs().emplace_back(msg);
    // get count
    ret = HcclRawGetCount(&status, HCCL_DATA_TYPE_INT8, &count);
    if (ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawGetCount, entity:%s, ret:%d.", entity.GetDesc().c_str(), ret);
        return FsmStatus::kFsmHcclFailed;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus ProbeState::Process(LlmCommEntity &entity) {
    int32_t count = -1;
    auto probe_ret = ProbeAndGetCountAsync(entity, count);
    if (probe_ret != FsmStatus::kFsmSuccess) {
        return probe_ret;
    }
    entity.GetEntityOccupied().store(true);
    entity.GetServerTickRecord().probe_req_start_tick = StatisticManager::GetInstance().GetCpuTick();
    if (entity.GetType() == EntityType::kEntityServer) {
        if (count == kUnlinkMsgSize) {
            UDF_LOG_INFO("Success to probe 0 count envelope which is unlink flag, entity:%s", entity.GetDesc().c_str());
            return entity.ChangeState(FsmState::kFsmUnlinkState);
        }
        if (static_cast<uint64_t>(count) == sizeof(uint8_t)) {
            return entity.ChangeState(FsmState::kFsmReceiveCheckState);
        }
        entity.GetSyncKvAddrInfo().req_info_count = count;
    } else {
        entity.GetTransferKvAddrInfo().req_info_count = count;
    }
    UDF_LOG_INFO("Success to probe msg, count:%d, entity:%s.", count, entity.GetDesc().c_str());
    return Postprocess(entity);
}

FsmStatus ProbeState::Postprocess(LlmCommEntity &entity) {
    UDF_LOG_DEBUG("Finish probe state, entity:%s.", entity.GetDesc().c_str());
    if (entity.GetType() == EntityType::kEntityServer) {
        return entity.ChangeState(FsmState::kFsmReceiveState);
    } else {
        return entity.ChangeState(FsmState::kFsmReceiveTransferReqState);
    }
}
}  // namespace FlowFunc
