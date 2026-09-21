/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_common/cluster_manager.h"
#include "flow_func/flow_func_log.h"
#include "flow_func/meta_flow_func.h"
#include "common/scope_guard.h"
#include "llm_common/statistic_manager.h"
#include "entity/link_req_handler.h"
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/hccl_so_manager.h"

namespace FlowFunc {
namespace {
constexpr const char *kIpAttr = "ip";
constexpr const char *kPortAttr = "port";
constexpr const char *kDeviceIndexAttr = "device_index";
constexpr uint32_t kMaxLinkNum = 512;
}  // namespace

int32_t ClusterManager::Initialize(const std::shared_ptr<MetaParams> &params,
                                   int32_t device_id,
                                   uint32_t unlink_output_idx,
                                   uint32_t update_link_out_idx,
                                   bool is_prompt) {
    unlink_output_idx_ = unlink_output_idx;
    update_link_output_idx_ = update_link_out_idx;
    auto ret = params->GetAttr(kDeviceIndexAttr, device_index_);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("GetAttr %s failed.", kDeviceIndexAttr);
        return ret;
    }

    ret = HcclSoManager::GetInstance()->LoadSo();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Load hccl so failed.");
        return ret;
    }
    FsmStatus entity_ret = LlmCommEntityMgr::RegisterHcclMr(static_cast<uint32_t>(device_id), mem_addrs_);
    if (entity_ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Register hccl mr failed.");
        return FLOW_FUNC_FAILED;
    }
    if (is_prompt) {
        ret = InitServerConn(params);
    }
    UDF_LOG_INFO("Init success, unlink_output_idx = %u, update_link_out_idx = %u", unlink_output_idx, update_link_out_idx);
    return ret;
}

void ClusterManager::Finalize() {
    (void) LlmCommEntityMgr::UnRegisterHcclMr(mem_addrs_);
}

int32_t ClusterManager::InitServerConn(const std::shared_ptr<MetaParams> &params) {
    int64_t listen_ip = 0L;
    auto ret = params->GetAttr(kIpAttr, listen_ip);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("GetAttr ip failed.");
        return ret;
    }
    int64_t listen_port = 0L;
    ret = params->GetAttr(kPortAttr, listen_port);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("GetAttr port failed.");
        return ret;
    }
    auto entity_ret = LlmCommEntityMgr::GetInstance().InitServerConn(static_cast<uint32_t>(listen_ip),
                                                                    static_cast<uint16_t>(listen_port));
    if (entity_ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("InitServerConn failed. ret = %d", static_cast<int32_t>(entity_ret));
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t ClusterManager::UpdateLink(const std::shared_ptr<MetaRunContext> &run_context,
                                   const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    StatisticManager::GetInstance().GetStatisticInfo().update_link_total_times++;
    update_link_output_ = run_context->AllocTensorMsg({kUpdateLinkMaxCount}, TensorDataType::DT_INT32);
    if (update_link_output_ == nullptr) {
        UDF_LOG_ERROR("Fail to alloc update link output tensor.");
        return FLOW_FUNC_FAILED;
    }
    ScopeGuard guard([this]() { update_link_output_ = nullptr; });
    if (CheckInputTensor(input_msgs, 1UL, 0UL, sizeof(UpdateLinkReqInfo), true) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
    }
    const auto update_link_req = static_cast<UpdateLinkReqInfo *>(input_msgs.front()->GetTensor()->GetData());
    if ((update_link_req->cluster_num > kUpdateLinkMaxCount)) {
        UDF_LOG_ERROR("Cluster num:%u is over limit:%zu.", update_link_req->cluster_num, kUpdateLinkMaxCount);
        return FLOW_FUNC_FAILED;
    }
    const int64_t output_idx = update_link_output_idx_;
    UDF_LOG_INFO("Begin to update link, operator_type:%u, cluster_num:%u, device_index:%ld, output_idx:%ld.",
                 update_link_req->operator_type, update_link_req->cluster_num, device_index_, output_idx);
    auto update_link_ret = static_cast<int32_t *>(update_link_output_->GetTensor()->GetData());
    if ((update_link_req->cluster_num + LlmCommEntityMgr::GetInstance().QueryLinkNum()) > kMaxLinkNum) {
        UDF_LOG_ERROR("Link num is over limit:%u", kMaxLinkNum);
        for (size_t index = 0UL; index < static_cast<size_t>(update_link_req->cluster_num); index++) {
            update_link_ret[index] = static_cast<int32_t>(FsmStatus::kFsmParamInvalid);
        }
    } else {
        LinkReqHandler::GetInstance().SetUpdateLinkReqInfo(update_link_req, update_link_ret, device_index_);
        (void) LinkReqHandler::GetInstance().HandleReq(start_tick);
        StatisticManager::GetInstance().Dump();
    }
    int32_t ret = run_context->SetOutput(output_idx, update_link_output_);
    UDF_LOG_INFO("Finish to update link, operator_type:%u, cluster_num:%u, device_index:%ld, output_idx:%ld.",
                 update_link_req->operator_type, update_link_req->cluster_num, device_index_, output_idx);
    return ret;
}

int32_t ClusterManager::UnlinkData(const std::shared_ptr<MetaRunContext> &run_context,
                                   const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    StatisticManager::GetInstance().GetStatisticInfo().update_link_total_times++;
    unlink_output_ = run_context->AllocTensorMsg({kUpdateLinkMaxCount}, TensorDataType::DT_INT32);
    if (unlink_output_ == nullptr) {
        UDF_LOG_ERROR("Fail to alloc unlink output tensor.");
        return FLOW_FUNC_FAILED;
    }
    ScopeGuard guard([this]() { unlink_output_ = nullptr; });
    if (CheckInputTensor(input_msgs, 1UL, 0UL, sizeof(UpdateLinkReqInfo), true) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
    }
    auto output_data = static_cast<int32_t *>(unlink_output_->GetTensor()->GetData());
    const auto update_link_req = static_cast<UpdateLinkReqInfo *>(input_msgs.front()->GetTensor()->GetData());
    if (update_link_req->operator_type != static_cast<uint32_t>(LinkReqType::kServerUnlink) &&
        update_link_req->operator_type != static_cast<uint32_t>(LinkReqType::kUnlink)) {
        UDF_LOG_ERROR("Invalid param, operator type must be unlink.");
        *output_data = static_cast<int32_t>(FsmStatus::kFsmParamInvalid);
        return run_context->SetOutput(unlink_output_idx_, unlink_output_);
    }
    UDF_LOG_INFO("Begin to unlink, cluster_num:%u, output_idx:%ld.", update_link_req->cluster_num, unlink_output_idx_);
    LinkReqHandler::GetInstance().SetUpdateLinkReqInfo(
        update_link_req, static_cast<int32_t *>(unlink_output_->GetTensor()->GetData()), device_index_);
    (void) LinkReqHandler::GetInstance().HandleReq(start_tick);
    StatisticManager::GetInstance().Dump();
    int32_t ret = run_context->SetOutput(unlink_output_idx_, unlink_output_);
    UDF_LOG_INFO("Finish to unlink, cluster_num:%u, device_index:%ld, output_idx:%ld.", update_link_req->cluster_num,
                 device_index_, unlink_output_idx_);
    return ret;
}

FsmStatus ClusterManager::CheckPullReqSize(uint64_t msg_data_size, const PullKvReqInfo *req_info) {
    uint64_t block_info_byte_size = msg_data_size - sizeof(PullKvReqInfo);
    if ((req_info->block_count + req_info->prompt_block_count) > (block_info_byte_size / sizeof(uint64_t))) {
        UDF_LOG_ERROR("Invalid req size, block_count:%u, prompt_block_count:%u, real count:%lu.",
                      req_info->block_count, req_info->prompt_block_count, block_info_byte_size / sizeof(uint64_t));
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus ClusterManager::PullCacheFromPrompt(PullKvReqInfo *req_info,
                                              bool enable_paged_attention,
                                              uint64_t start_tick,
                                              const CacheEntry *cache_entry) {
    // check link status
    EntityPtr entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->prompt_cluster_id);
    if (entity == nullptr) {
        UDF_RUN_LOG_ERROR("Current cluster is not linked with remote cluster:%lu, req_id:%lu.",
                          req_info->prompt_cluster_id, req_info->req_id);
        return FsmStatus::kFsmNotLink;
    }
    if (entity->GetType() != EntityType::kEntityClient) {
        UDF_RUN_LOG_ERROR("Invalid entity type, entity type is %d, req_id:%lu, prompt_cluster_id:%lu.",
                          req_info->req_id, req_info->prompt_cluster_id, static_cast<int32_t>(entity->GetType()));
        return FsmStatus::kFsmFailed;
    }
    // sync kv from prompt
    PullOrTransferExtraInfo extra_info{enable_paged_attention, true, false, start_tick};
    FsmStatus ret = entity->PullOrTransferCache({}, *req_info, {}, cache_entry, extra_info);
    const uint64_t finish_tick = StatisticManager::GetInstance().GetCpuTick();
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR(
            "Fail to sync kv from prompt, ret:%d, %s, time_cost:%.2f us, entity:%s.",
            static_cast<int32_t>(ret),
            CacheKeyDebugString(*req_info).c_str(),
            StatisticManager::GetInstance().GetTimeCost(finish_tick - start_tick),
            entity->GetDesc().c_str());
        return ret;
    }
    bool max_tick_cost_flag = false;
    const uint64_t tick_cost = finish_tick - start_tick;
    EntityStatisticInfo &stat_info = entity->GetStatisticInfo();
    UpdateTickCost(tick_cost, stat_info.sync_kv_total_times, stat_info.sync_kv_min_tick_cost,
                   stat_info.sync_kv_max_tick_cost, stat_info.sync_kv_total_tick_cost, max_tick_cost_flag);
    const double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        entity->SetForcePrintFlag(true);
        UDF_RUN_LOG_INFO(
            "Success to pull kv from prompt with max time cost, %s, "
            "time cost:%.2f us, total_times:%lu, entity:%s.",
            CacheKeyDebugString(*req_info).c_str(), time_cost, stat_info.sync_kv_total_times,
            entity->GetDesc().c_str());
    }
    UDF_LOG_INFO(
        "Success to pull kv from prompt, %s, time_cost:%.2f us, total_times:%lu, entity:%s.",
        CacheKeyDebugString(*req_info).c_str(), time_cost, stat_info.sync_kv_total_times, entity->GetDesc().c_str());
    entity->PrintStatisticInfo();
    return FsmStatus::kFsmSuccess;
}

FsmStatus ClusterManager::TransferCache(TransferKvReqInfo *req_info,
                                        bool enable_paged_attention,
                                        uint64_t start_tick,
                                        const CacheEntry *cache_entry) {
    // check link status
    EntityPtr entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
    if (entity == nullptr) {
        UDF_RUN_LOG_ERROR("Current cluster is not linked with remote cluster:%lu.", req_info->dst_cluster_id);
        return FsmStatus::kFsmNotLink;
    }
    // sync kv from prompt
    PullOrTransferExtraInfo extra_info{enable_paged_attention, false, false, start_tick};
    FsmStatus ret = entity->PullOrTransferCache(*req_info, {}, {}, cache_entry, extra_info);
    const uint64_t finish_tick = StatisticManager::GetInstance().GetCpuTick();
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to transfer cache, ret:%d, %s, time cost:%.2f us, entity:%s.",
                      static_cast<int32_t>(ret), TransferReqDebugString(*req_info).c_str(),
                      StatisticManager::GetInstance().GetTimeCost(finish_tick - start_tick),
                      entity->GetDesc().c_str());
        return ret;
    }
    bool max_tick_cost_flag = false;
    const uint64_t tick_cost = finish_tick - start_tick;
    EntityStatisticInfo &stat_info = entity->GetStatisticInfo();
    UpdateTickCost(tick_cost, stat_info.transfer_kv_total_times, stat_info.transfer_kv_min_tick_cost,
                   stat_info.transfer_kv_max_tick_cost, stat_info.transfer_kv_total_tick_cost, max_tick_cost_flag);
    const double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        entity->SetForcePrintFlag(true);
        UDF_RUN_LOG_INFO("Success to transfer cache with max time cost, %s, time cost:%.2f us, total_times:%lu, entity:%s.",
                         TransferReqDebugString(*req_info).c_str(), time_cost, stat_info.transfer_kv_total_times,
                         entity->GetDesc().c_str());
    }
    UDF_LOG_INFO("Success to transfer cache, %s, time cost:%.2f us, total_times:%lu, entity:%s.",
                 TransferReqDebugString(*req_info).c_str(), time_cost, stat_info.transfer_kv_total_times, entity->GetDesc().c_str());
    entity->PrintPutStatisticInfo();
    return FsmStatus::kFsmSuccess;
}

int32_t ClusterManager::SyncKvData(const std::shared_ptr<MetaRunContext> &run_context,
                                   const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const
{
    (void) run_context;
    (void) input_msgs;
    LlmCommEntityMgr::GetInstance().HandleRequest(true);
    return FLOW_FUNC_SUCCESS;
}

}  // namespace FlowFunc
