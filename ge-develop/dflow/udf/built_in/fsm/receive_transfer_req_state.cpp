/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/receive_transfer_req_state.h"
#include <mutex>
#include "fsm/probe_state.h"
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/cache_manager.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kOneLayerCacheSize = 2U;
}

FsmStatus ReceiveTransferReqState::Preprocess(LlmCommEntity &entity) {
    std::vector<HcclMessage> &probe_msgs = entity.GetProbeMsgs();
    LlmCommEntity::TransferKvAddrInfo &addr_info = entity.GetTransferKvAddrInfo();
    // malloc dynamic length req mbuf, free in idle or error state
    (void) entity.AllocMbuf(addr_info.transfer_kv_req_mbuf, addr_info.req_info_count, addr_info.transfer_kv_req_addr);
    if ((addr_info.transfer_kv_req_mbuf == nullptr) || (addr_info.transfer_kv_req_addr == nullptr)) {
        UDF_LOG_ERROR("Fail to alloc mbuf for transfer kv req info, entity:%s.", entity.GetDesc().c_str());
        return FsmStatus::kFsmFailed;
    }
    HcclRequest request;
    HcclResult ret = HcclRawImrecv(addr_info.transfer_kv_req_addr, addr_info.req_info_count, HCCL_DATA_TYPE_INT8,
                                   &probe_msgs.front(), &request);
    entity.GetStatisticInfo().call_recv_total_times++;
    if (ret != HCCL_SUCCESS) {
        entity.GetStatisticInfo().call_recv_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawImrecv, data_buff:%p, count:%zu, entity:%s.", addr_info.transfer_kv_req_addr,
                      sizeof(TransferToRemoteReq), entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    entity.GetServerTickRecord().send_meta_start_tick = StatisticManager::GetInstance().GetCpuTick();
    entity.GetStatisticInfo().call_recv_success_times++;
    // save receive mbuf and receive request to entity
    entity.GetReceiveRequests().emplace_back(request);  // only one receive request
    return Process(entity);
}

FsmStatus ReceiveTransferReqState::TestReq(LlmCommEntity &entity) {
    LlmCommEntity::RecvTransferKvRecordInfo &record_info = entity.GetRecvTransferKvRecordInfo();
    if (record_info.recv_req_suc_flag == 1U) {
        return FsmStatus::kFsmSuccess;
    }
    std::vector<HcclRequest> &receive_requests = entity.GetReceiveRequests();
    const int32_t test_count = 1;
    int32_t comp_count = 0;
    auto ret = entity.TestCompleteAsync(receive_requests.data(), test_count, comp_count);
    if (ret != FsmStatus::kFsmSuccess) {
        return FsmStatus::kFsmHcclFailed;
    }
    if (comp_count == 0) {
        UDF_LOG_INFO("Test some when receive transfer kv request ret complete count is zero.");
        return FsmStatus::kFsmKeepState;
    }
    LlmCommEntity::TransferKvAddrInfo &addr_info = entity.GetTransferKvAddrInfo();
    if (static_cast<uint64_t>(addr_info.req_info_count) < sizeof(TransferToRemoteReq)) {
        UDF_RUN_LOG_INFO("Invalid req size, probably caused by pull cache failed, count:%d, expected req len:%zu, "
                         "entity:%s.", addr_info.req_info_count, sizeof(TransferToRemoteReq), entity.GetDesc().c_str());
        return FsmStatus::kFsmIgnore;
    }
    EntityStatisticInfo &stat_info = entity.GetStatisticInfo();
    const uint64_t current_tick_cost =
            StatisticManager::GetInstance().GetCpuTick() - entity.GetServerTickRecord().probe_req_start_tick;
    bool max_tick_cost_flag = false;
    UpdateTickCost(current_tick_cost, stat_info.recv_transfer_req_total_times, stat_info.recv_transfer_req_min_tick_cost,
                   stat_info.recv_transfer_req_max_tick_cost, stat_info.recv_transfer_req_total_tick_cost, max_tick_cost_flag);
    stat_info.test_recv_success_times++;
    // get and record reqId
    const double k_time_cost = StatisticManager::GetInstance().GetTimeCost(current_tick_cost);
    if (max_tick_cost_flag) {
        entity.SetForcePrintFlag(true);
    }
    UDF_LOG_INFO("Success to receive transfer cache req, time cost:%.2f us, total times:%lu, entity:%s.",
                 k_time_cost, stat_info.recv_req_total_times, entity.GetDesc().c_str());
    record_info.recv_req_suc_flag = 1U;
    ret = CheckAndSendMeta(entity);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus ReceiveTransferReqState::CheckAndSendMeta(LlmCommEntity &entity) {
    auto ret = FsmStatus::kFsmSuccess;
    LlmCommEntity::TransferKvAddrInfo &addr_info = entity.GetTransferKvAddrInfo();
    std::call_once(entity.GetsendMetaOnceFlag(), [&entity, &addr_info]() {
        (void) entity.AllocMbuf(addr_info.transfer_kv_resp_meta_mbuf,
                                sizeof(TransferKvMetaInfo),
                                addr_info.transfer_kv_resp_meta_addr);
    });
    if ((addr_info.transfer_kv_resp_meta_mbuf == nullptr) || (addr_info.transfer_kv_resp_meta_addr == nullptr)) {
        UDF_LOG_ERROR("Fail to alloc mbuf for send resp, entity:%s.", entity.GetDesc().c_str());
        return FsmStatus::kFsmFailed;
    }
    auto *resp_info = static_cast<TransferKvMetaInfo *>(addr_info.transfer_kv_resp_meta_addr);
    resp_info->err_code = static_cast<int32_t>(FsmStatus::kFsmSuccess);
    uint64_t buffer_info_size = static_cast<uint64_t>(addr_info.req_info_count) - sizeof(TransferToRemoteReq);
    auto *req_info = static_cast<TransferToRemoteReq *>(entity.GetTransferKvAddrInfo().transfer_kv_req_addr);
    if (req_info->total_slot_nums != (buffer_info_size / sizeof(TransferSlotInfo))) {
        UDF_RUN_LOG_INFO("Invalid req size, probably caused by pull cache failed, total_slot_nums:%lu, real count:%lu, "
                         "entity:%s.", req_info->total_slot_nums, buffer_info_size / sizeof(TransferSlotInfo),
                         entity.GetDesc().c_str());
        resp_info->err_code = static_cast<int32_t>(FsmStatus::kFsmParamInvalid);
        ret = FsmStatus::kFsmIgnore;
    } else if (req_info->dst_cache_id >= 0) {
        UDF_LOG_INFO("Push kv without addr, cache id:%ld, batch index:%lu, tensor per layer:%lu.", req_info->dst_cache_id,
                     req_info->dst_batch_index, req_info->tensor_num_per_layer);
        ret = ValidateAndSetPushAddrs(req_info, entity);
        resp_info->err_code = static_cast<int32_t>(ret);
    } else {
        if (CacheManager::GetInstance().CheckAddr(req_info->key_addr, req_info->max_size) != FsmStatus::kFsmSuccess) {
            UDF_LOG_ERROR("Invalid key addr, entity:%s.", entity.GetDesc().c_str());
            resp_info->err_code = static_cast<int32_t>(FsmStatus::kFsmParamInvalid);
            ret = FsmStatus::kFsmParamInvalid;
        } else if (CacheManager::GetInstance().CheckAddr(req_info->value_addr, req_info->max_size) != FsmStatus::kFsmSuccess) {
            UDF_LOG_ERROR("Invalid value addr, entity:%s.", entity.GetDesc().c_str());
            resp_info->err_code = static_cast<int32_t>(FsmStatus::kFsmParamInvalid);
            ret = FsmStatus::kFsmParamInvalid;
        }
        auto &dst_addrs = entity.GetPushDstAddrs();
        dst_addrs.emplace_back(req_info->key_addr);
        dst_addrs.emplace_back(req_info->value_addr);
    }
    return ret;
}

FsmStatus ReceiveTransferReqState::ValidateAndSetPushAddrs(const TransferToRemoteReq *req_info, LlmCommEntity &entity) {
    CacheEntry cache_entry;
    if (!CacheManager::GetInstance().GetCacheEntry(req_info->dst_cache_id, cache_entry)) {
        UDF_LOG_ERROR("cacheId:%ld, kv cache not found", req_info->dst_cache_id);
        return FsmStatus::kFsmKvNotExist;
    }
    if (req_info->dst_batch_index >= static_cast<uint64_t>(cache_entry.batch_size)) {
        UDF_LOG_ERROR("cacheId:%ld, batchIndex (%lu) >= batch_size(%u)",
                      req_info->dst_cache_id, req_info->dst_batch_index, cache_entry.batch_size);
        return FsmStatus::kFsmKvNotExist;
    }
    auto buffer_len = req_info->max_size;
    if (cache_entry.tensor_size < buffer_len) {
        UDF_LOG_ERROR("cacheId:%ld, kv tensor size (%lu) < required tensor size (%lu)",
                      req_info->dst_cache_id, cache_entry.tensor_size, buffer_len);
        return FsmStatus::kFsmParamInvalid;
    }
    if (cache_entry.tensors.size() % req_info->tensor_num_per_layer != 0) {
        UDF_LOG_ERROR("cacheId:%ld, tensors num:%zu is not a multiple of %lu.", req_info->dst_cache_id,
                      cache_entry.tensors.size(), req_info->tensor_num_per_layer);
        return FsmStatus::kFsmParamInvalid;
    }
    uint64_t max_layer_index = cache_entry.tensors.size() / req_info->tensor_num_per_layer;
    if (req_info->dst_layer_index >= max_layer_index) {
        UDF_LOG_ERROR("cacheId:%ld, layer index:%lu out of range [0, %lu).", req_info->dst_layer_index,
                      req_info->dst_layer_index, max_layer_index);
        return FsmStatus::kFsmParamInvalid;
    }
    auto batch_offset = req_info->dst_batch_index * cache_entry.batch_stride;
    auto tensor_start_idx = req_info->dst_layer_index * req_info->tensor_num_per_layer;
    auto &dst_addrs = entity.GetPushDstAddrs();
    dst_addrs.resize(req_info->tensor_num_per_layer);
    for (uint64_t i = 0; i < req_info->tensor_num_per_layer; i++) {
        auto &tensor = cache_entry.tensors[tensor_start_idx + i];
        auto data_addr = static_cast<uint8_t *>(tensor->GetTensor()->GetData()) + batch_offset;
        dst_addrs[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data_addr));
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus ReceiveTransferReqState::TestSendMeta(LlmCommEntity &entity) {
    LlmCommEntity::RecvTransferKvRecordInfo &record_info = entity.GetRecvTransferKvRecordInfo();
    if (record_info.call_send_meta_flag == 0U) {
        LlmCommEntity::TransferKvAddrInfo &addr_info = entity.GetTransferKvAddrInfo();
        auto send_ret = entity.SendAsync(addr_info.transfer_kv_resp_meta_addr, sizeof(TransferKvMetaInfo));
        if (send_ret != FsmStatus::kFsmSuccess) {
            return send_ret;
        }
        record_info.call_send_meta_flag = 1U;
    }
    if (record_info.send_meta_complete_flag == 1U) {
        return FsmStatus::kFsmSuccess;
    }
    std::vector<HcclRequest> &send_requests = entity.GetSendRequests();
    int32_t comp_count = 0;
    auto ret = entity.TestCompleteAsync(send_requests.data(), 1U, comp_count);
    if (ret != FsmStatus::kFsmSuccess) {
        return FsmStatus::kFsmHcclFailed;
    }
    if (comp_count == 0) {
        UDF_LOG_INFO("Send meta resp is not complete.");
        return FsmStatus::kFsmKeepState;
    }
    record_info.send_meta_complete_flag = 1U;
    return FsmStatus::kFsmSuccess;
}

FsmStatus ReceiveTransferReqState::Process(LlmCommEntity &entity) {
    auto req_ret = TestReq(entity);
    if (req_ret == FsmStatus::kFsmIgnore) {
        return entity.ChangeState(FsmState::kFsmIdleState);
    }
    // may keep state, when param is invalid, just go to send
    if ((req_ret != FsmStatus::kFsmParamInvalid) && (req_ret != FsmStatus::kFsmSuccess)) {
        return req_ret;
    }
    auto ret = TestSendMeta(entity);
    // may keep state
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    LlmCommEntity::TransferKvAddrInfo &addr_info = entity.GetTransferKvAddrInfo();
    auto *resp_info = static_cast<TransferKvMetaInfo *>(addr_info.transfer_kv_resp_meta_addr);
    if (resp_info->err_code == static_cast<int32_t>(FsmStatus::kFsmParamInvalid)) {
        return entity.ChangeState(FsmState::kFsmIdleState);
    }
    return Postprocess(entity);
}

FsmStatus ReceiveTransferReqState::Postprocess(LlmCommEntity &entity) {
    entity.GetReceiveRequests().clear();
    entity.GetProbeMsgs().clear();
    UDF_LOG_INFO("Finish receive transfer state, entity:%s.", entity.GetDesc().c_str());
    return entity.ChangeState(FsmState::kFsmReceiveTransferCacheState);
}
}  // namespace FlowFunc
