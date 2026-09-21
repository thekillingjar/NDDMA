/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/receive_state.h"
#include <mutex>
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/llm_common.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr size_t PROBE_MSG_COUNT = 1UL;
}
FsmStatus ReceiveState::Preprocess(LlmCommEntity &entity) {
    UDF_LOG_DEBUG("Enter receive state, entity:%s.", entity.GetDesc().c_str());
    std::vector<HcclMessage> &probe_msgs = entity.GetProbeMsgs();
    if (probe_msgs.size() != PROBE_MSG_COUNT) {
        UDF_LOG_ERROR("Invalid probe msg count, size:%zu, expected count:%zu, entity:%s.", probe_msgs.size(),
                      PROBE_MSG_COUNT, entity.GetDesc().c_str());
        return FsmStatus::kFsmFailed;
    }
    LlmCommEntity::SyncKvAddrInfo &addr_info = entity.GetSyncKvAddrInfo();
    (void) entity.AllocMbuf(addr_info.sync_kv_req_mbuf,
                            addr_info.req_info_count,
                            addr_info.sync_kv_req_addr);
    if ((addr_info.sync_kv_req_mbuf == nullptr) || (addr_info.sync_kv_req_addr == nullptr)) {
        UDF_LOG_ERROR("Fail to alloc mbuf for sync kv req info, entity:%s.", entity.GetDesc().c_str());
        return FsmStatus::kFsmFailed;
    }
    // receive req info
    entity.GetServerTickRecord().send_meta_start_tick = StatisticManager::GetInstance().GetCpuTick();
    HcclRequest request;
    entity.GetStatisticInfo().call_recv_total_times++;
    HcclResult ret = HcclRawImrecv(addr_info.sync_kv_req_addr, addr_info.req_info_count, HCCL_DATA_TYPE_INT8,
                                   &probe_msgs.front(), &request);
    if (ret != HCCL_SUCCESS) {
        entity.GetStatisticInfo().call_recv_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawImrecv, data_buff:%p, count:%zu, entity:%s.", addr_info.sync_kv_req_addr,
                      sizeof(SyncKvReqInfo), entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    entity.GetStatisticInfo().call_recv_success_times++;
    // save receive mbuf and receive request to entity
    entity.GetReceiveRequests().emplace_back(request);  // only one receive request
    return Process(entity);
}

FsmStatus ReceiveState::Process(LlmCommEntity &entity) {
    std::vector<HcclRequest> &receive_requests = entity.GetReceiveRequests();
    const int32_t k_test_count = 1;
    int32_t comp_count = 0;
    std::vector<HcclStatus> &comp_status = LlmCommEntityMgr::GetInstance().GetCompStatus(receive_requests.size());
    std::vector<int32_t> &comp_indices = LlmCommEntityMgr::GetInstance().GetCompIndices(receive_requests.size());
    HcclResult ret =
        HcclRawTestSome(k_test_count, receive_requests.data(), &comp_count, comp_indices.data(), comp_status.data());
    if (ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawTestSome, test_count:%d, entity:%s.", k_test_count, entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    if (comp_count == 0) {
        UDF_LOG_INFO("Test some when receive sync kv request ret complete count is zero.");
        return FsmStatus::kFsmKeepState;
    }
    HcclStatus &status = comp_status.front();
    if (status.error != 0) {
        UDF_LOG_ERROR("Get invalid status, status:%d, entity:%s.", status.error, entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    LlmCommEntity::SyncKvAddrInfo &addr_info = entity.GetSyncKvAddrInfo();
    if (static_cast<uint64_t>(addr_info.req_info_count) < sizeof(SyncKvReqInfo)) {
        UDF_RUN_LOG_INFO(
            "Invalid req size, probably caused by transfer cache failed, count:%d, expected req len:%zu, "
            "entity:%s.", addr_info.req_info_count, sizeof(SyncKvReqInfo), entity.GetDesc().c_str());
        return entity.ChangeState(FsmState::kFsmIdleState);
    }
    EntityStatisticInfo &stat_info = entity.GetStatisticInfo();
    const uint64_t current_tick_cost =
        StatisticManager::GetInstance().GetCpuTick() - entity.GetServerTickRecord().probe_req_start_tick;
    bool max_tick_cost_flag = false;
    UpdateTickCost(current_tick_cost, stat_info.recv_req_total_times, stat_info.recv_req_min_tick_cost,
                   stat_info.recv_req_max_tick_cost, stat_info.recv_req_total_tick_cost, max_tick_cost_flag);
    stat_info.test_recv_success_times++;
    // get and record req_id
    auto *req_info = static_cast<SyncKvReqInfo *>(entity.GetSyncKvAddrInfo().sync_kv_req_addr);
    entity.SetCurCacheIdAndBatchIndex(std::make_pair(req_info->cache_id, static_cast<uint32_t>(req_info->batch_index)));
    entity.SetCurReqId(req_info->req_id);
    entity.SetCurModelId(req_info->model_id);
    entity.SetCurPrefixId(req_info->prefix_id);
    entity.SetCurIsPullBlock(static_cast<bool>(req_info->is_pull_block));
    entity.SetCurIsPullWithOffset(static_cast<bool>(req_info->is_pull_with_offset));
    const double k_time_cost = StatisticManager::GetInstance().GetTimeCost(current_tick_cost);
    if (max_tick_cost_flag) {
        entity.SetForcePrintFlag(true);
    }
    UDF_LOG_INFO("Success to receive sync kv req%s, cache_id:%ld, req_id:%lu, "
                 "time cost:%.2f us, total times:%lu, entity:%s.",
                 max_tick_cost_flag ? kMaxTimeCost : kCommonTimeCost,
                 req_info->cache_id, req_info->req_id, k_time_cost, stat_info.recv_req_total_times, entity.GetDesc().c_str());
    return Postprocess(entity);
}

FsmStatus ReceiveState::Postprocess(LlmCommEntity &entity) {
    // clear hccl resource for receive sync kv req
    entity.GetReceiveRequests().clear();
    entity.GetProbeMsgs().clear();
    UDF_LOG_DEBUG("Finish receive state, entity:%s.", entity.GetDesc().c_str());
    return entity.ChangeState(FsmState::kFsmSendState);
}
}  // namespace FlowFunc