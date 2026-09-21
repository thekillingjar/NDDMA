/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_service_flow_func.h"
#include "flow_func/flow_func_log.h"
#include "flow_func/meta_flow_func.h"
#include "llm_common/statistic_manager.h"
#include "llm_common/llm_common.h"
#include "flow_func/flow_func_timer.h"
#include "entity/llm_comm_entity_mgr.h"

namespace FlowFunc {
namespace {
constexpr const char *kAttrNameRole = "role";
constexpr const char *kAttrNameOutIndices = "output_indices";
constexpr const char *kRolePrompt = "Prompt";
constexpr const char *kRoleDecoder = "Decoder";
constexpr const char *kRoleMix = "Mix";

template<typename T>
FsmStatus AsReqInfo(const std::shared_ptr<FlowMsg> &input, const T *&req_info) {
    const auto &tensor = *input->GetTensor();
    if (tensor.GetDataSize() != sizeof(T)) {
        UDF_LOG_ERROR("Expect tensor size = %zu, but only %lu",
                      sizeof(T),
                      tensor.GetDataSize());
        return FsmStatus::kFsmParamInvalid;
    }
    req_info = reinterpret_cast<const T *>(tensor.GetData());
    return FsmStatus::kFsmSuccess;
}

std::string ToRole(int32_t role_id) {
  std::map<int32_t, std::string> roleIdToName {
      {0, kRolePrompt},
      {1, kRoleDecoder},
      {2, kRoleMix},
  };
  return roleIdToName[role_id];
}

template<>
FsmStatus AsReqInfo<AllocateCacheReqInfo>(const std::shared_ptr<FlowMsg> &input,
                                          const AllocateCacheReqInfo *&req_info) {
    const auto &tensor = *input->GetTensor();
    const auto tensor_size = tensor.GetDataSize();
    if (tensor_size < sizeof(AllocateCacheReqInfo)) {
        UDF_LOG_ERROR("Expect tensor size at least = %zu, but only %zu",
                      sizeof(AllocateCacheReqInfo),
                      tensor_size);
        return FsmStatus::kFsmParamInvalid;
    }

    req_info = reinterpret_cast<const AllocateCacheReqInfo *>(tensor.GetData());
    if ((req_info->num_requests > 0) &&
        (tensor_size != (sizeof(AllocateCacheReqInfo) + (req_info->num_requests * sizeof(int64_t))))) {
        UDF_LOG_ERROR("Expect tensor size = %zu, but only %zu",
                      sizeof(AllocateCacheReqInfo) + (req_info->num_requests * sizeof(int64_t)),
                      tensor_size);
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

template<>
FsmStatus AsReqInfo<CopyCacheReqInfo>(const std::shared_ptr<FlowMsg> &input, const CopyCacheReqInfo *&req_info) {
    const auto &tensor = *input->GetTensor();
    const auto tensor_size = tensor.GetDataSize();
    if (tensor_size < sizeof(CopyCacheReqInfo)) {
        return FsmStatus::kFsmParamInvalid;
    }
    req_info = reinterpret_cast<const CopyCacheReqInfo *>(tensor.GetData());
    auto expect_size = sizeof(CopyCacheReqInfo) + (req_info->copy_block_count * sizeof(BlockCopyInfo));
    if ((req_info->copy_block_count > 0U) && (tensor_size != expect_size)) {
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}
}  // namespace

LlmServiceFlowFunc::~LlmServiceFlowFunc() {
    UDF_LOG_INFO("Destruct LlmServiceFlowFunc.");
    FinalizeStatistic();
    cluster_manager_.Finalize();
}

int32_t LlmServiceFlowFunc::Init(const std::shared_ptr<MetaParams> &params) {
    SetEnvs(params);
    device_id_ = params->GetRunningDeviceId();
    if (ParseFlowFuncAttrs(params) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
    }
    const auto is_prompt = role_ == kRolePrompt;
    const auto ret = InitClusterManager(params, is_prompt);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Initialize cluster manager failed.");
        return ret;
    }
    CacheManager::GetInstance().Enable();
    ResetStatistic(role_);
    UDF_LOG_INFO("LlmServiceFlowFunc Initialized, deviceId = %d, role = %s", device_id_, role_.c_str());
    return FLOW_FUNC_SUCCESS;
}

void LlmServiceFlowFunc::FinalizeStatistic() {
    if (statistic_timer_handle_ != nullptr) {
        (void) FlowFuncTimer::Instance().StopTimer(statistic_timer_handle_);
        (void) FlowFuncTimer::Instance().DeleteTimer(statistic_timer_handle_);
        statistic_timer_handle_ = nullptr;
    }
    StatisticManager::GetInstance().GetStatisticInfo().Reset();
}

void LlmServiceFlowFunc::ResetStatistic(const std::string &role) {
    FinalizeStatistic();
    if (role != kRoleMix) {
        const auto is_prompt = role == kRolePrompt;
        StatisticManager::GetInstance().GetStatisticInfo().is_server.store(is_prompt);
        // create and start statistic timer. Period is 80s, as running log limit 50 logs per hour.
        statistic_timer_handle_ = FlowFuncTimer::Instance().CreateTimer([]() {
            StatisticManager::GetInstance().Dump();
            StatisticManager::GetInstance().DumpServerProfilingTrack();
            LlmCommEntityMgr::GetInstance().DumpServerEntities();
        });
        constexpr uint32_t statistic_timer_period = 80U * 1000U;
        (void)FlowFuncTimer::Instance().StartTimer(statistic_timer_handle_, statistic_timer_period, false);
    }
}

int32_t LlmServiceFlowFunc::ParseFlowFuncAttrs(const std::shared_ptr<MetaParams> &params) {
    std::vector<int64_t> out_indices;
    const auto ret = params->GetAttr(kAttrNameOutIndices, out_indices);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("GetAttr %s failed, ret = %d", kAttrNameOutIndices, ret);
        return FLOW_FUNC_FAILED;
    }
    for (const auto out_index : out_indices) {
        flow_func_out_indices_.emplace_back(static_cast<uint32_t>(out_index));
    }
    UDF_LOG_INFO("%s = %s", kAttrNameOutIndices, ToString(out_indices).c_str());
    flow_func_out_indices_.resize(static_cast<size_t>(FlowFuncType::kMax), -1);
    AscendString role_str;
    (void) params->GetAttr(kAttrNameRole, role_str);
    role_ = role_str.GetString();
    return FLOW_FUNC_SUCCESS;
}

int32_t LlmServiceFlowFunc::InitClusterManager(const std::shared_ptr<MetaParams> &params, bool is_prompt) {
    const auto unlink_out_index = flow_func_out_indices_[static_cast<size_t>(FlowFuncType::kUnlink)];
    const auto update_link_out_index = flow_func_out_indices_[static_cast<size_t>(FlowFuncType::kUpdateLink)];
    const auto ret = cluster_manager_.Initialize(params, device_id_, unlink_out_index, update_link_out_index, is_prompt);
    return ret;
}

int32_t LlmServiceFlowFunc::Initialize(const std::shared_ptr<MetaRunContext> &run_context,
                                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs)
{
    (void) input_msgs;
    const auto out_index = flow_func_out_indices_[static_cast<size_t>(FlowFuncType::kInitialize)];
    return SetOutput(run_context, out_index, FsmStatus::kFsmSuccess);
}

int32_t LlmServiceFlowFunc::AllocateCache(const std::shared_ptr<MetaRunContext> &run_context,
                                          const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const auto out_index = flow_func_out_indices_[static_cast<size_t>(FlowFuncType::kAllocate)];
    // response: Status(s32) | padding(s32) | tensor_addr * N
    const AllocateCacheReqInfo *req_info = nullptr;
    auto ret = AsReqInfo(input_msgs[0], req_info);
    if (ret != FsmStatus::kFsmSuccess) {
        return SetOutput(run_context, out_index, ret);
    }

    const auto output_dim = static_cast<int64_t>(req_info->num_tensors) + 1L;
    auto result_msg = run_context->AllocTensorMsg({output_dim}, TensorDataType::DT_UINT64);
    if (result_msg == nullptr) {
        UDF_LOG_ERROR("Fail to alloc output tensor");
        return FLOW_FUNC_FAILED;
    }
    auto *output_data = static_cast<uint64_t *>(result_msg->GetTensor()->GetData());
    ret = CacheManager::GetInstance().AllocateCache(run_context, *req_info, &output_data[1]);
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    if (ret == FsmStatus::kFsmSuccess) {
        stat_info.call_allocate_suc_times++;
    } else {
        stat_info.call_allocate_fail_times++;
    }
    *static_cast<int32_t *>(result_msg->GetTensor()->GetData()) = static_cast<int32_t>(ret);
    return run_context->SetOutput(out_index, result_msg);
}

int32_t LlmServiceFlowFunc::DeallocateCache(const std::shared_ptr<MetaRunContext> &run_context,
                                            const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const int64_t *cache_id = nullptr;
    auto ret = AsReqInfo(input_msgs[0], cache_id);
    if (ret == FsmStatus::kFsmSuccess) {
        ret = CacheManager::GetInstance().DeallocateCache(*cache_id);
    }
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    if (ret == FsmStatus::kFsmSuccess) {
        stat_info.call_deallocate_suc_times++;
    } else {
        stat_info.call_deallocate_fail_times++;
    }
    return SetOutput(run_context, FlowFuncType::kDeallocate, ret);
}

int32_t LlmServiceFlowFunc::RemoveCacheIndex(const std::shared_ptr<MetaRunContext> &run_context,
                                             const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const RemoveCacheIndexReqInfo *req_info = nullptr;
    auto ret = AsReqInfo(input_msgs[0], req_info);
    if (ret == FsmStatus::kFsmSuccess) {
        bool is_prefix = (req_info->prefix_id != UINT64_MAX);
        std::pair<uint64_t, uint64_t>
            cache_index = std::make_pair(is_prefix ? req_info->prefix_id : req_info->req_id, req_info->model_id);
        (void) CacheManager::GetInstance().RemoveCacheIndex(cache_index, is_prefix);
    }
    return SetOutput(run_context, FlowFuncType::kRemoveIndex, ret);
}

int32_t LlmServiceFlowFunc::CopyCache(const std::shared_ptr<MetaRunContext> &run_context,
                                      const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    const CopyCacheReqInfo *req_info = nullptr;
    FsmStatus ret = AsReqInfo(input_msgs[0], req_info);
    if (ret == FsmStatus::kFsmSuccess) {
        ret = CacheManager::GetInstance().CopyCache(*req_info);
    }
    bool max_tick_cost_flag = false;
    const auto tick_cost = StatisticManager::GetInstance().GetCpuTick() - start_tick;
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    UpdateTickCost(tick_cost, stat_info.copy_kv_total_times, stat_info.copy_kv_min_tick_cost,
                   stat_info.copy_kv_max_tick_cost, stat_info.copy_kv_total_tick_cost, max_tick_cost_flag);
    const double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        StatisticManager::GetInstance().Dump();
        UDF_RUN_LOG_INFO("Success to copy cache with max time cost, %s, time_cost:%.2f us, total_times:%lu.",
                         CopyCacheDebugString(*req_info).c_str(), time_cost, stat_info.copy_kv_total_times.load());
    }
    UDF_LOG_INFO("Success to copy cache, %s, time_cost:%.2f us, total_time:%lu.",
                 CopyCacheDebugString(*req_info).c_str(), time_cost, stat_info.copy_kv_total_times.load());
    return SetOutput(run_context, FlowFuncType::kCopy, ret);
}

int32_t LlmServiceFlowFunc::GetCache(const std::shared_ptr<MetaRunContext> &run_context,
                                     const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const auto out_index = flow_func_out_indices_[static_cast<size_t>(FlowFuncType::kGetTensorSummary)];
    const GetCacheReqInfo *req_info = nullptr;
    FsmStatus fsm_ret = AsReqInfo(input_msgs[0], req_info);
    if (fsm_ret != FsmStatus::kFsmSuccess) {
        return SetOutput(run_context, out_index, fsm_ret);
    }
    UDF_LOG_INFO("[cache_id:%ld][GetTensor] start, tensor_index = %d", req_info->cache_id, req_info->tensor_index);
    CacheEntry cache_entry;
    if (!CacheManager::GetInstance().GetCacheEntry(req_info->cache_id, cache_entry)) {
        UDF_LOG_ERROR("[cache_id:%ld][GetTensor] failed, cache not exist");
        return SetOutput(run_context, out_index, FsmStatus::kFsmKvNotExist);
    }

    const auto result_msg = run_context->AllocTensorMsg({2}, TensorDataType::DT_INT32);
    if (result_msg == nullptr) {
        UDF_LOG_ERROR("Fail to alloc output tensor");
        return FLOW_FUNC_FAILED;
    }
    auto *ret_code_and_num = static_cast<int32_t *>(result_msg->GetTensor()->GetData());
    ret_code_and_num[0] = static_cast<int32_t>(FsmStatus::kFsmSuccess);
    ret_code_and_num[1] = 1;  // will support multiple tensor later
    auto ret = run_context->SetOutput(out_index, result_msg);
    if (ret == FLOW_FUNC_SUCCESS) {
        auto tensor = cache_entry.tensors[req_info->tensor_index];
        ret = run_context->SetOutput(out_index, std::move(tensor));
        UDF_LOG_INFO("[cache_id:%ld][GetTensor] end, tensor_index = %d", req_info->cache_id, req_info->tensor_index);
    }
    return ret;
}

int32_t LlmServiceFlowFunc::TransferCache(const std::shared_ptr<MetaRunContext> &run_context,
                                          const std::vector<std::shared_ptr<FlowMsg> > &input_msgs) {
    const uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    stat_info.transfer_kv_total_times++;
    if (CheckInputTensor(input_msgs, 1UL, 0UL, sizeof(TransferKvReqInfo), true) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
    }
    auto req_info = static_cast<TransferKvReqInfo *>(input_msgs.front()->GetTensor()->GetData());
    if (ClusterManager::CheckPullAndTransferReqSize(input_msgs.front()->GetTensor()->GetDataSize(), req_info)
        != FsmStatus::kFsmSuccess) {
        return FLOW_FUNC_FAILED;
    }
    auto fsm_ret = CacheManager::GetInstance().TransferCache(req_info, cluster_manager_, start_tick);
    int32_t ret = SetOutput(run_context, FlowFuncType::kTransfer, fsm_ret);
    if ((ret != FLOW_FUNC_SUCCESS) || (fsm_ret != FsmStatus::kFsmSuccess)) {
        stat_info.sync_kv_fail_times++;
        const uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - start_tick;
        UDF_LOG_ERROR("Finish to transfer kv cache, %s, timeout:%lu us, time_cost:%.2f us, transfer kv ret:%d, set output ret:%d.",
                      TransferReqDebugString(*req_info).c_str(), req_info->timeout,
                      StatisticManager::GetInstance().GetTimeCost(tick_cost), static_cast<int32_t>(fsm_ret), ret);
        return ret;
    }
    bool max_tick_cost_flag = false;
    const auto tick_cost = StatisticManager::GetInstance().GetCpuTick() - start_tick;
    UpdateTickCost(tick_cost, stat_info.transfer_kv_succ_times, stat_info.transfer_kv_min_tick_cost,
                   stat_info.transfer_kv_max_tick_cost, stat_info.transfer_kv_total_tick_cost, max_tick_cost_flag);
    const double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        StatisticManager::GetInstance().Dump();
        UDF_RUN_LOG_INFO(
                "Success to transfer cache with max time cost, %s, timeout:%lu us, time_cost:%.2f us, total_times:%lu.",
                TransferReqDebugString(*req_info).c_str(), req_info->timeout, time_cost, stat_info.transfer_kv_succ_times.load());
    }
    UDF_LOG_INFO("Success to transfer cache, %s, timeout:%lu us, time_cost:%.2f us, total_time:%lu.",
                 TransferReqDebugString(*req_info).c_str(), req_info->timeout, time_cost,
                 stat_info.transfer_kv_succ_times.load());
    if (stat_info.sync_kv_total_times.load() > kRecordMaxLimit) {
        StatisticManager::GetInstance().ResetProfilingTrack();
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t LlmServiceFlowFunc::PullCache(const std::shared_ptr<MetaRunContext> &run_context,
                                      const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    stat_info.sync_kv_total_times++;
    if (CheckInputTensor(input_msgs, 1UL, 0UL, sizeof(PullKvReqInfo), true) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
    }
    auto req_info = static_cast<PullKvReqInfo *>(input_msgs.front()->GetTensor()->GetData());
    const auto additional_index_count = req_info->dst_tensor_index_count + req_info->src_tensor_index_count;
    const auto check_ret = ClusterManager::CheckPullAndTransferReqSize(input_msgs.front()->GetTensor()->GetDataSize(),
                                                                      req_info, additional_index_count);
    if (check_ret != FsmStatus::kFsmSuccess) {
        return FLOW_FUNC_FAILED;
    }
    auto fsm_ret = CacheManager::GetInstance().PullCache(req_info, cluster_manager_, start_tick);
    int32_t ret = SetOutput(run_context, FlowFuncType::kPull, fsm_ret);
    if ((ret != FLOW_FUNC_SUCCESS) || (fsm_ret != FsmStatus::kFsmSuccess)) {
        stat_info.sync_kv_fail_times++;
        const uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - start_tick;
        UDF_LOG_ERROR(
            "Finish to sync kv cache, %s, "
            "timeout:%lu us, time_cost:%.2f us, sync_ret:%d, set output ret:%d.",
            CacheKeyDebugString(*req_info).c_str(),
            req_info->timeout, StatisticManager::GetInstance().GetTimeCost(tick_cost), static_cast<int32_t>(fsm_ret), ret);
        return ret;
    }
    bool max_tick_cost_flag = false;
    const auto tick_cost = StatisticManager::GetInstance().GetCpuTick() - start_tick;
    UpdateTickCost(tick_cost, stat_info.sync_kv_succ_times, stat_info.sync_kv_min_tick_cost,
                   stat_info.sync_kv_max_tick_cost, stat_info.sync_kv_total_tick_cost, max_tick_cost_flag);
    const double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        StatisticManager::GetInstance().Dump();
        UDF_RUN_LOG_INFO(
            "Success to pull cache with max time cost, %s, timeout:%lu us, time_cost:%.2f us, total_times:%lu.",
            CacheKeyDebugString(*req_info).c_str(), req_info->timeout, time_cost, stat_info.sync_kv_succ_times.load());
    }
    UDF_LOG_INFO("Success to pull cache, %s, timeout:%lu us, time_cost:%.2f us, total_time:%lu.",
                 CacheKeyDebugString(*req_info).c_str(), req_info->timeout, time_cost, stat_info.sync_kv_succ_times.load());
    if (stat_info.sync_kv_total_times.load() > kRecordMaxLimit) {
        StatisticManager::GetInstance().ResetProfilingTrack();
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t LlmServiceFlowFunc::SwitchRole(const std::shared_ptr<MetaRunContext> &run_context,
                                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const SwitchRoleReqInfo *req_info = nullptr;
    FsmStatus fsm_ret = AsReqInfo(input_msgs[0], req_info);
    if (fsm_ret != FsmStatus::kFsmSuccess) {
        return SetOutput(run_context, FlowFuncType::kSwitchRole, fsm_ret);
    }
    std::string target_role = ToRole(req_info->role_id);
    UDF_LOG_INFO("[SwitchRole] [%s->%s] start", role_.c_str(), target_role.c_str());
    if ((role_ == kRolePrompt) || (role_ == kRoleDecoder)) {
        if (LlmCommEntityMgr::GetInstance().HasAnyLink()) {
            UDF_LOG_ERROR("[SwitchRole] [%s->%s] failed, existing link must be unlinked before switching",
                          role_.c_str(), target_role.c_str());
            return SetOutput(run_context, FlowFuncType::kSwitchRole, FsmStatus::kFsmExistLink);
        }
        LlmCommEntityMgr::GetInstance().FinalizeServerConn();
    }
    if (target_role == kRolePrompt) {
        UDF_LOG_INFO("start to init server conn, ip=%u, port=%u", req_info->prompt_listen_ip, req_info->prompt_listen_port);
        auto entity_ret = LlmCommEntityMgr::GetInstance().InitServerConn(req_info->prompt_listen_ip,
                                                                        req_info->prompt_listen_port);
        if (entity_ret != FsmStatus::kFsmSuccess) {
          UDF_LOG_ERROR("[SwitchRole] [%s->%s] InitServerConn failed. ret = %d",
                        role_.c_str(), target_role.c_str(), static_cast<int32_t>(entity_ret));
          return SetOutput(run_context, FlowFuncType::kSwitchRole, entity_ret);
        }
    }
    ResetStatistic(target_role);
    UDF_LOG_INFO("[SwitchRole] [%s->%s] success", role_.c_str(), target_role.c_str());
    role_ = target_role;
    return SetOutput(run_context, FlowFuncType::kSwitchRole, FsmStatus::kFsmSuccess);
}

int32_t LlmServiceFlowFunc::UpdateLink(const std::shared_ptr<MetaRunContext> &run_context,
                                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    return cluster_manager_.UpdateLink(run_context, input_msgs);
}

int32_t LlmServiceFlowFunc::UnlinkData(const std::shared_ptr<MetaRunContext> &run_context,
                                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    return cluster_manager_.UnlinkData(run_context, input_msgs);
}

// only support in Decoder currently.
int32_t LlmServiceFlowFunc::CheckLink(const std::shared_ptr<MetaRunContext> &run_context,
                                      const std::vector<std::shared_ptr<FlowMsg> > &input_msgs) {
    const uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    if (CheckInputTensor(input_msgs, 1UL, 0UL, sizeof(CheckLinkReqInfo), false) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
    }
    auto req_info = static_cast<CheckLinkReqInfo *>(input_msgs.front()->GetTensor()->GetData());
    EntityPtr entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
    auto fsm_ret = FsmStatus::kFsmNotLink;
    if (entity != nullptr) {
        PullOrTransferExtraInfo extra_info{false, false, true, start_tick};
        fsm_ret = entity->PullOrTransferCache({}, {}, *req_info, nullptr, extra_info);
    } else {
        UDF_RUN_LOG_ERROR("Current cluster is not linked with remote cluster:%lu.", req_info->dst_cluster_id);
    }
    int32_t ret = SetOutput(run_context, FlowFuncType::kCheckLink, fsm_ret);
    if ((ret != FLOW_FUNC_SUCCESS) || (fsm_ret != FsmStatus::kFsmSuccess)) {
        const uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - start_tick;
        UDF_LOG_ERROR("Check link failed, cluster:%lu, timeout:%lu us, time cost:%.2f us, check ret:%d, "
                      "set output ret:%d.", req_info->dst_cluster_id, req_info->timeout,
                      StatisticManager::GetInstance().GetTimeCost(tick_cost), static_cast<int32_t>(fsm_ret), ret);
        return ret;
    }
    const uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - start_tick;
    UDF_LOG_INFO("Check link success, cluster:%lu, timeout:%lu us, time cost:%.2f us, check ret:%d, "
                 "set output ret:%d.", req_info->dst_cluster_id, req_info->timeout,
                 StatisticManager::GetInstance().GetTimeCost(tick_cost), static_cast<int32_t>(fsm_ret), ret);
    return FLOW_FUNC_SUCCESS;
}

int32_t LlmServiceFlowFunc::SyncKvData(const std::shared_ptr<MetaRunContext> &run_context,
                                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    if (role_ == kRolePrompt) {
        return cluster_manager_.SyncKvData(run_context, input_msgs);
    } else {
        LlmCommEntityMgr::GetInstance().HandleRequest(false);
        return FLOW_FUNC_SUCCESS;
    }
}

int32_t LlmServiceFlowFunc::SetOutput(const std::shared_ptr<MetaRunContext> &run_context,
                                      uint32_t output_idx,
                                      FsmStatus status) {
    auto result_msg = run_context->AllocTensorMsg({}, TensorDataType::DT_INT32);
    if (result_msg == nullptr) {
        UDF_LOG_ERROR("Fail to alloc output tensor for outputIndex = %u", output_idx);
        return FLOW_FUNC_FAILED;
    }
    auto output_data = static_cast<int32_t *>(result_msg->GetTensor()->GetData());
    *output_data = static_cast<int32_t>(status);
    const auto ret = run_context->SetOutput(output_idx, result_msg);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Fail to set output outputIndex = %u", output_idx);
    }
    return ret;
}

int32_t LlmServiceFlowFunc::SetOutput(const std::shared_ptr<MetaRunContext> &run_context,
                                      FlowFuncType flow_func_type,
                                      FsmStatus status) {
    const auto out_index = flow_func_out_indices_[static_cast<size_t>(flow_func_type)];
    return SetOutput(run_context, out_index, status);
}

void LlmServiceFlowFunc::SetEnvs(const std::shared_ptr<MetaParams> &params) {
    const static std::string kValidEnvNamePrefix = "HCCL_";
    std::vector<AscendString> env_names;
    std::vector<AscendString> env_values;
    (void) params->GetAttr("ENV_NAMES", env_names);
    (void) params->GetAttr("ENV_VALUES", env_values);
    UDF_LOG_INFO("env_names = %s, env_values = %s", ToString(env_names).c_str(), ToString(env_values).c_str());
    if (env_names.size() == env_values.size()) {
        for (size_t i = 0U; i < env_names.size(); ++i) {
            if (std::string(env_names[i].GetString()).find(kValidEnvNamePrefix) == 0) {
                setenv(env_names[i].GetString(), env_values[i].GetString(), 1);
            }
        }
    }
}

FLOW_FUNC_REGISTRAR(LlmServiceFlowFunc)
    .RegProcFunc("_BuiltIn_initialize_func", &LlmServiceFlowFunc::Initialize)
    .RegProcFunc("_BuiltIn_cluster_update_link_func", &LlmServiceFlowFunc::UpdateLink)
    .RegProcFunc("_BuiltIn_cluster_unlink_func", &LlmServiceFlowFunc::UnlinkData)
    .RegProcFunc("_BuiltIn_check_link_func", &LlmServiceFlowFunc::CheckLink)
    .RegProcFunc("_BuiltIn_cluster_kv_data_sync_func", &LlmServiceFlowFunc::SyncKvData)
    .RegProcFunc("_BuiltIn_cache_allocate_func", &LlmServiceFlowFunc::AllocateCache)
    .RegProcFunc("_BuiltIn_cache_deallocate_func", &LlmServiceFlowFunc::DeallocateCache)
    .RegProcFunc("_BuiltIn_cache_get_func", &LlmServiceFlowFunc::GetCache)
    .RegProcFunc("_BuiltIn_cache_copy_func", &LlmServiceFlowFunc::CopyCache)
    .RegProcFunc("_BuiltIn_cache_remove_index_func", &LlmServiceFlowFunc::RemoveCacheIndex)
    .RegProcFunc("_BuiltIn_switch_role_func", &LlmServiceFlowFunc::SwitchRole)
    .RegProcFunc("_BuiltIn_cache_pull_func", &LlmServiceFlowFunc::PullCache)
    .RegProcFunc("_BuiltIn_cache_transfer_func", &LlmServiceFlowFunc::TransferCache);
}  // namespace FlowFunc

