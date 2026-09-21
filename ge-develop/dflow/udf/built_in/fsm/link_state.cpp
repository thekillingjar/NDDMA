/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/link_state.h"
#include "fsm/probe_state.h"
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
FsmStatus LinkState::Preprocess(LlmCommEntity &entity) {
    UDF_LOG_INFO("Enter link state, entity:%s.", entity.GetDesc().c_str());
    return Process(entity);
}

FsmStatus LinkState::Process(LlmCommEntity &entity) {
    if (!entity.GetLinkEstablished()) {
        FuncStatisticInfo &sum_stat_info = StatisticManager::GetInstance().GetStatisticInfo();
        FsmStatus ret = ProbeLinkClusterInfoAsync(entity);
        if (ret == FsmStatus::kFsmKeepState) {
            return ret;
        }
        if (ret != FsmStatus::kFsmSuccess) {
            sum_stat_info.link_fail_times++;
            return ret;
        }
        ret = TestRecvAsync(entity);
        if ((ret == FsmStatus::kFsmKeepState) || (ret == FsmStatus::kFsmEstablishLinkSuc)) {
            return ret;
        }
        if (ret != FsmStatus::kFsmSuccess) {
            sum_stat_info.link_fail_times++;
            return ret;
        }
    }
    return Postprocess(entity);
}

FsmStatus LinkState::ProbeLinkClusterInfoAsync(LlmCommEntity &entity) {
    if (entity.GetProbeLinkClusterInfoFlag()) {
        return FsmStatus::kFsmSuccess;
    }
    int32_t count = -1;
    auto probe_ret = ProbeState::ProbeAndGetCountAsync(entity, count);
    if (probe_ret != FsmStatus::kFsmSuccess) {
        return probe_ret;
    }
    if (count != sizeof(ClientClusterInfo)) {
        return FsmStatus::kFsmParamInvalid;
    }
    UDF_LOG_INFO("Link msg is probed");
    entity.SetProbeLinkClusterInfoFlag(true);
    HcclRequest request;
    HcclResult ret = HcclRawImrecv(&entity.GetClientClusterInfo(), sizeof(ClientClusterInfo), HCCL_DATA_TYPE_INT8,
                                   &entity.GetProbeMsgs().front(), &request);
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Call HcclRawImrecv failed, ret:%d.", ret);
        return FsmStatus::kFsmHcclFailed;
    }
    entity.GetReceiveRequests().emplace_back(request);
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkState::TestRecvAsync(LlmCommEntity &entity) {
    std::vector<HcclRequest> &requests = entity.GetReceiveRequests();
    int32_t comp_count = 0;
    std::vector<int32_t> comp_indices{0};
    std::vector<HcclStatus> comp_status{HcclStatus{}};
    HcclResult ret = HcclRawTestSome(1, requests.data(), &comp_count, comp_indices.data(), comp_status.data());
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawTestSome for link send empty resp, ret:%d.", ret);
        return FsmStatus::kFsmHcclFailed;
    }
    if (comp_count == 0) {
        return FsmStatus::kFsmKeepState;
    }
    HcclStatus &status = comp_status[0UL];
    if (status.error != 0) {
        UDF_LOG_ERROR("Get invalid test status for link send empty resp, status:%d.", status.error);
        return FsmStatus::kFsmFailed;
    }
    UDF_LOG_INFO("Establish link with cluster:%lu", entity.GetClientClusterInfo().cluster_id);
    entity.SetLinkEstablished(true);
    return FsmStatus::kFsmEstablishLinkSuc;
}

FsmStatus LinkState::Postprocess(LlmCommEntity &entity) {
    const uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - entity.GetServerTickRecord().link_start_tick;
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    bool max_tick_cost_flag = false;
    UpdateTickCost(tick_cost, stat_info.link_succ_times, stat_info.link_min_tick_cost,
                   stat_info.link_max_tick_cost, stat_info.link_total_tick_cost, max_tick_cost_flag);
    const double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    UDF_LOG_INFO("Success to link%s, time cost:%.2f us, link succ times:%lu, entity:%s.",
                 max_tick_cost_flag ? kMaxTimeCost : kCommonTimeCost,
                 time_cost, stat_info.link_succ_times.load(), entity.GetDesc().c_str());
    return entity.ChangeState(FsmState::kFsmIdleState);
}
}  // namespace FlowFunc
