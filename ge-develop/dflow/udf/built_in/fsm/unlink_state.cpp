/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/unlink_state.h"
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr const int32_t kUnlinkMsgSize = 0;
}
FsmStatus UnlinkState::Preprocess(LlmCommEntity &entity) {
    UDF_LOG_DEBUG("Enter unlink state, entity:%s.", entity.GetDesc().c_str());
    HcclRequest request;
    HcclResult ret =
        HcclRawImrecv(nullptr, kUnlinkMsgSize, HCCL_DATA_TYPE_INT8, &entity.GetProbeMsgs().front(), &request);
    if (ret != HCCL_SUCCESS) {
        (void) HcclRawClose(entity.GetConn());
        StatisticManager::GetInstance().GetStatisticInfo().unlink_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawImrecv for unlink, ret:%d, entity:%s.", ret, entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    entity.GetReceiveRequests().emplace_back(request);
    return Process(entity);
}

FsmStatus UnlinkState::Process(LlmCommEntity &entity) {
    std::vector<HcclRequest> &requests = entity.GetReceiveRequests();
    const int32_t k_test_count = 1;
    int32_t comp_count = 0;
    std::vector<HcclStatus> &comp_status = LlmCommEntityMgr::GetInstance().GetCompStatus(requests.size());
    std::vector<int32_t> &comp_indices = LlmCommEntityMgr::GetInstance().GetCompIndices(requests.size());
    HcclResult ret = HcclRawTestSome(k_test_count, requests.data(), &comp_count, comp_indices.data(), comp_status.data());
    if (ret != HCCL_SUCCESS) {
        (void) HcclRawClose(entity.GetConn());
        StatisticManager::GetInstance().GetStatisticInfo().unlink_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawTestSome, test_count:%d, entity:%s.", k_test_count, entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    if (comp_count == 0) {
        return FsmStatus::kFsmKeepState;
    }
    HcclStatus &status = comp_status[0UL];
    if (status.error != 0) {
        (void) HcclRawClose(entity.GetConn());
        StatisticManager::GetInstance().GetStatisticInfo().unlink_fail_times++;
        UDF_LOG_ERROR("Get invalid status for unlink, status:%d, entity:%s.", status.error, entity.GetDesc().c_str());
    }
    return Postprocess(entity);
}

FsmStatus UnlinkState::Postprocess(LlmCommEntity &entity) {
    entity.GetProbeMsgs().clear();
    entity.GetReceiveRequests().clear();
    HcclResult ret = HcclRawClose(entity.GetConn());
    if (ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawClose, entity:%s.", entity.GetDesc().c_str());
    }
    const uint64_t tick_cost =
        StatisticManager::GetInstance().GetCpuTick() - entity.GetServerTickRecord().probe_req_start_tick;
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    bool max_tick_cost_flag = false;
    UpdateTickCost(tick_cost, stat_info.unlink_succ_times, stat_info.unlink_min_tick_cost,
                   stat_info.unlink_max_tick_cost, stat_info.unlink_total_tick_cost, max_tick_cost_flag);
    const double time_tost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    UDF_LOG_INFO("Success to unlink%s, time_cost:%.2f us, unlink succ times:%lu, entity:%s.",
                 max_tick_cost_flag ? kMaxTimeCost : kCommonTimeCost,
                 time_tost, stat_info.unlink_succ_times.load(), entity.GetDesc().c_str());
    entity.Dump();
    return entity.ChangeState(FsmState::kFsmDestroyState);
}
}  // namespace FlowFunc
