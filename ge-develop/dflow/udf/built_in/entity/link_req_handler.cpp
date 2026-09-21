/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "link_req_handler.h"
#include <set>
#include <thread>
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/statistic_manager.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kCheckTimeoutLoopCount = 5UL;
constexpr uint64_t kUpdateLinkTimeout = 3000000UL;  // 3s
constexpr const char *kLinkStr = "link";
constexpr const char *kUnlinkStr = "unlink";
}  // namespace

LinkReqHandler &LinkReqHandler::GetInstance() {
    static LinkReqHandler handler;
    return handler;
}

LinkReqHandler::LinkReqHandler() {
    update_link_comp_flags_.reserve(kUpdateLinkMaxCount);
    update_link_states_.reserve(kUpdateLinkMaxCount);
    update_link_conns_.reserve(kUpdateLinkMaxCount);
    send_unlink_req_comp_flags_.reserve(kUpdateLinkMaxCount);
    update_link_requests_.reserve(kUpdateLinkMaxCount);
    update_link_start_ticks_.reserve(kUpdateLinkMaxCount);
    update_link_client_infos_.reserve(kUpdateLinkMaxCount);
}

void LinkReqHandler::SetUpdateLinkReqInfo(UpdateLinkReqInfo *link_req, int32_t *update_link_ret, int64_t device_index) {
    update_link_req_info_ = link_req;
    update_link_ret_ = update_link_ret;
    device_index_ = device_index;
    update_link_comp_count_ = 0U;
    update_link_comp_flags_.assign(kUpdateLinkMaxCount, false);
    update_link_conns_.assign(kUpdateLinkMaxCount, nullptr);
    send_unlink_req_comp_flags_.assign(kUpdateLinkMaxCount, false);
    update_link_start_ticks_.assign(kUpdateLinkMaxCount, 0UL);
    update_link_requests_.assign(kUpdateLinkMaxCount, nullptr);
    update_link_states_.assign(kUpdateLinkMaxCount, DoLinkState::kLinkInit);
}

FsmStatus LinkReqHandler::HandleReq(uint64_t start_tick) {
    UDF_LOG_INFO("Begin to update links, operator_type:%u, cluster_num:%u.", update_link_req_info_->operator_type,
                 update_link_req_info_->cluster_num);
    // trans update link info
    DevLinkReqInfo dev_link_req_info{};
    FsmStatus ret = TransUpdateLinkInfo(*update_link_req_info_, dev_link_req_info);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    const uint32_t cluster_num = dev_link_req_info.cluster_num;
    auto req_type = static_cast<LinkReqType>(dev_link_req_info.operator_type);
    if (req_type == LinkReqType::kServerUnlink) {
        for (size_t index = 0UL; index < static_cast<size_t>(cluster_num); index++) {
            DoServerUnlink(index, dev_link_req_info.cluster_infos.at(index));
        }
        return FsmStatus::kFsmSuccess;
    }
    if (req_type == LinkReqType::kUnlink) {
        for (size_t index = 0UL; index < static_cast<size_t>(cluster_num); index++) {
            if (update_link_comp_flags_.at(index)) {
                continue;
            }
            ClusterAddrInfo &cluster = dev_link_req_info.cluster_infos.at(index);
            auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(cluster.remote_cluster_id);
            if (entity->GetUnlinkMutex().try_lock()) {
                std::lock_guard<std::mutex> unlinkLock(entity->GetUnlinkMutex(), std::adopt_lock);
                std::lock_guard<std::mutex> lock(entity->GetMutex());
                entity->GetIsUnlinking().store(true, std::memory_order_relaxed);
                continue;
            }
            UDF_LOG_ERROR("Link is busy now.");
            SetResult(LinkReqType::kUnlink, index, FsmStatus::kFsmLinkBusy);
        }
    }
    // loop update link
    uint64_t loop_count = 0UL;
    while (update_link_comp_count_ < cluster_num) {
        loop_count++;
        for (size_t index = 0UL; index < static_cast<size_t>(cluster_num); index++) {
            ClusterAddrInfo &cluster = dev_link_req_info.cluster_infos.at(index);
            if (update_link_comp_flags_.at(index)) {
                continue;
            }
            if (req_type == LinkReqType::kLink) {
                DoLink(index, cluster);
            } else if (req_type == LinkReqType::kUnlink) {
                DoUnlink(index, cluster);
            }
        }
        if ((update_link_comp_count_ < cluster_num) && (loop_count % kCheckTimeoutLoopCount== 0UL)) {
            FsmStatus check_ret = CheckTimeout(start_tick, dev_link_req_info.timeout);
            if (check_ret != FsmStatus::kFsmSuccess) {
                SetOutput(cluster_num, FsmStatus::kFsmTimeout);
                CloseTimeoutConn(cluster_num, req_type, dev_link_req_info);
                ResetNotifyPromptFailed(cluster_num);
                UDF_LOG_ERROR("Update link timeout, operator_type:%u, cluster num:%u, comp num:%u.",
                              dev_link_req_info.operator_type, cluster_num, update_link_comp_count_);
                return FsmStatus::kFsmTimeout;
            }
        }
    }
    UDF_LOG_INFO("Finish to update links, operator_type:%u, cluster_num:%u.", dev_link_req_info.operator_type, cluster_num);
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkReqHandler::TransUpdateLinkInfo(const UpdateLinkReqInfo &req_info,
                                              LinkReqHandler::DevLinkReqInfo &dev_req_info) {
    // check and set cluster num
    if ((req_info.cluster_num > kUpdateLinkMaxCount) || (req_info.cluster_num == 0U)) {
        UDF_LOG_ERROR("Param invalid for update link req, cluster_num:%u, max cluster_num:%zu.", req_info.cluster_num,
                      kUpdateLinkMaxCount);
        SetOutput(static_cast<size_t>(kUpdateLinkMaxCount), FsmStatus::kFsmReqOverLimit);
        return FsmStatus::kFsmReqOverLimit;
    }
    dev_req_info.cluster_num = req_info.cluster_num;
    // check and set operator type
    uint32_t operator_type = req_info.operator_type;
    if ((operator_type != static_cast<uint32_t>(LinkReqType::kLink)) &&
        (operator_type != static_cast<uint32_t>(LinkReqType::kUnlink)) &&
        (operator_type != static_cast<uint32_t>(LinkReqType::kServerUnlink))) {
        UDF_LOG_ERROR("Param invalid for update link req, operator_type:%u.", operator_type);
        SetOutput(static_cast<size_t>(req_info.cluster_num), FsmStatus::kFsmParamInvalid);
        return FsmStatus::kFsmParamInvalid;
    }
    dev_req_info.operator_type = operator_type;
    // set timeout
    dev_req_info.timeout = (req_info.timeout == 0UL) ? kUpdateLinkTimeout : req_info.timeout;
    const auto clusters_info_addr = reinterpret_cast<uintptr_t>(req_info.cluster_infos);
    const size_t cluster_info_size = sizeof(ClusterInfo) + req_info.cluster_infos[0].ip_num * sizeof(IpInfo);
    // check and set cluster_infos
    std::set<uint64_t> remote_cluster_ids;
    for (size_t index = 0UL; index < static_cast<size_t>(dev_req_info.cluster_num); index++) {
        ClusterAddrInfo cluster_addr_info{};
        auto *cluster = reinterpret_cast<ClusterInfo *>(clusters_info_addr + cluster_info_size * index);
        FsmStatus ret = TransClusterInfo(*cluster, static_cast<LinkReqType>(operator_type),
                                         req_info.cluster_infos[0].ip_num, cluster_addr_info);
        if (ret != FsmStatus::kFsmSuccess) {
            update_link_ret_[index] = static_cast<int32_t>(ret);
            update_link_comp_flags_[index] = true;
            update_link_comp_count_++;
            dev_req_info.cluster_infos.emplace_back(cluster_addr_info);
            continue;
        }
        // check repetitive link/unlink req
        auto iter = remote_cluster_ids.find(cluster_addr_info.remote_cluster_id);
        if (iter != remote_cluster_ids.end()) {
            update_link_ret_[index] = static_cast<int32_t>(FsmStatus::kFsmRepeatRequest);
            update_link_comp_flags_[index] = true;
            update_link_comp_count_++;
            dev_req_info.cluster_infos.emplace_back(cluster_addr_info);
            UDF_LOG_ERROR("Repetitive update link req, operator_type:%u, cluster index:%zu, cluster info:%s.",
                          operator_type, index, ToString(cluster_addr_info).c_str());
            continue;
        }
        (void) remote_cluster_ids.emplace(cluster_addr_info.remote_cluster_id);
        update_link_start_ticks_[index] = StatisticManager::GetInstance().GetCpuTick();
        UDF_LOG_INFO("Prepare to update link, operator_type:%u, cluster index:%zu, cluster num:%u, cluster info:%s.",
                     operator_type, index, req_info.cluster_num, ToString(cluster_addr_info).c_str());
        dev_req_info.cluster_infos.emplace_back(cluster_addr_info);
    }
    if (update_link_comp_count_ == req_info.cluster_num) {
        UDF_RUN_LOG_WARN("Param invalid in all clusters of update link req, operator_type:%u, cluster num:%u.",
                         operator_type, req_info.cluster_num);
        return FsmStatus::kFsmParamInvalid;
    }
    UDF_LOG_INFO("Success to trans update link req, operator_type:%u, invalid cluster num:%u, "
                 "total cluster num:%u.", operator_type, update_link_comp_count_, req_info.cluster_num);
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkReqHandler::TransClusterInfo(const ClusterInfo &cluster_info, LinkReqType req_type, uint32_t ip_num,
                                           LinkReqHandler::ClusterAddrInfo &cluster) const {
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    cluster.local_cluster_id = cluster_info.local_cluster_id;
    cluster.remote_cluster_id = cluster_info.remote_cluster_id;
    if (cluster_info.ip_num != ip_num) {
        UDF_LOG_ERROR("Param invalid for update link req, device index:%ld, cluster ip num:%u, expected ip num:%u.",
                      device_index_, cluster_info.ip_num, ip_num);
        return FsmStatus::kFsmParamInvalid;
    }
    if (device_index_ >= cluster_info.ip_num) {
        UDF_LOG_ERROR("Param invalid for update link req, device_index:%ld is over cluster ip_num:%u, "
                      "operator_type:%u.", device_index_, cluster_info.ip_num, static_cast<uint32_t>(req_type));
        return FsmStatus::kFsmParamInvalid;
    }
    auto iter = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(cluster.remote_cluster_id);
    if ((req_type == LinkReqType::kLink) && (iter != nullptr)) {
        UDF_LOG_ERROR("Param invalid for link, link already exist, remote_cluster_id:%lu.", cluster.remote_cluster_id);
        stat_info.link_fail_times++;
        return FsmStatus::kFsmAlreadyLinked;
    }
    if (req_type == LinkReqType::kUnlink) {
        if (iter == nullptr) {
            UDF_RUN_LOG_WARN("Param invalid for unlink, link not exist, remote_cluster_id:%lu.", cluster.remote_cluster_id);
            stat_info.unlink_succ_times++;
            return FsmStatus::kFsmNotLink;
        } else if ((iter->GetRemoteIp() != cluster_info.ip_infos[device_index_].remote_ip) ||
            (iter->GetLocalIp() != cluster_info.ip_infos[device_index_].local_ip)) {
            UDF_LOG_ERROR("Param invalid for unlink, ip inconsistent, remote_cluster_id:%lu.", cluster.remote_cluster_id);
            stat_info.unlink_fail_times++;
            return FsmStatus::kFsmParamInvalid;
        }
    }
    const IpInfo &ip_info = cluster_info.ip_infos[device_index_];
    cluster.local_hccl_addr.type = HcclAddrType::HCCL_ADDR_TYPE_ROCE;
    cluster.local_hccl_addr.info.tcp.ipv4Addr = (ip_num == 0U) ? 0U : ip_info.local_ip;
    cluster.local_hccl_addr.info.tcp.port = (ip_num == 0U) ? 0U : ip_info.local_port;
    cluster.local_hccl_addr.type = HcclAddrType::HCCL_ADDR_TYPE_ROCE;
    cluster.remote_hccl_addr.info.tcp.ipv4Addr = (ip_num == 0U) ? 0U : ip_info.remote_ip;
    cluster.remote_hccl_addr.info.tcp.port = (ip_num == 0U) ? 0U : ip_info.remote_port;
    UDF_LOG_INFO("Success to trans cluster info, operator_type:%u, cluster info:%s.", static_cast<uint32_t>(req_type),
                 ToString(cluster).c_str());
    return FsmStatus::kFsmSuccess;
}

void LinkReqHandler::SetResult(LinkReqType link_req_type, size_t cluster_index, FsmStatus status) {
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    update_link_ret_[cluster_index] = static_cast<int32_t>(status);
    update_link_comp_flags_[cluster_index] = true;
    update_link_comp_count_++;
    if (status == FsmStatus::kFsmSuccess) {
        bool max_tick_cost_flag = false;
        const uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - update_link_start_ticks_[cluster_index];
        UpdateTickCost(tick_cost, stat_info.unlink_succ_times, stat_info.unlink_min_tick_cost,
                       stat_info.unlink_max_tick_cost, stat_info.unlink_total_tick_cost, max_tick_cost_flag);
        const double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
        auto type_str = (link_req_type == LinkReqType::kLink ? kLinkStr : kUnlinkStr);
        UDF_LOG_INFO("Success to %s%s, time cost:%.2f us, %s succ times:%lu.",
                     type_str, max_tick_cost_flag ? kMaxTimeCost : kCommonTimeCost, time_cost, type_str,
                     stat_info.unlink_succ_times.load());
    } else {
        link_req_type == LinkReqType::kLink ? stat_info.link_fail_times++ : stat_info.unlink_fail_times++;
    }
}

FsmStatus LinkReqHandler::InitClientConn(size_t cluster_index, ClusterAddrInfo &cluster, const std::string &cluster_desc) {
    HcclConn client_conn = nullptr;
    FsmStatus ret = LlmCommEntityMgr::InitClientConn(cluster.local_hccl_addr, client_conn);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to init client conn, remote cluster:%s, ret:%d, "
                      "Which may be result of being executed by unexpected user. "
                      "Please check details in device app log with key word:\"IsBuiltinUdfUser\".",
                      cluster_desc.c_str(), static_cast<int32_t>(ret));
        SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmLinkFailed);
        if (client_conn != nullptr) {
            CloseConn(client_conn);
        }
        return FsmStatus::kFsmFailed;
    }
    update_link_conns_[cluster_index] = client_conn;
    update_link_states_[cluster_index] = DoLinkState::kLinkTryBind;
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkReqHandler::TryBindConn(size_t cluster_index, ClusterAddrInfo &cluster,
                                      const std::string &cluster_desc, const HcclConn &hccl_conn) {
    HcclResult ret = HcclRawBind(hccl_conn, &cluster.local_hccl_addr);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        UDF_RUN_LOG_INFO("HcclRawBind need try again.");
        return FsmStatus::kFsmKeepState;
    }
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to bind local conn, local_hccl_addr:%s, ret:%d", cluster_desc.c_str(),
                      ret);
        SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmLinkFailed);
        (void) CloseConn(hccl_conn);
        return FsmStatus::kFsmFailed;
    }
    update_link_states_[cluster_index] = DoLinkState::kLinkTryConnect;
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkReqHandler::TryConnectServer(size_t cluster_index, ClusterAddrInfo &cluster,
                                           const std::string &cluster_desc, const HcclConn &hccl_conn) {
    HcclResult ret = HcclRawConnect(hccl_conn, &cluster.remote_hccl_addr);
    if (ret == HCCL_E_AGAIN) {
        UDF_LOG_INFO("Need retry to call HcclRawConnect, remote cluster:%s, ret:%d.", cluster_desc.c_str(), ret);
        return FsmStatus::kFsmFailed;
    }
    if (ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawConnect, remote cluster:%s, ret:%d.", cluster_desc.c_str(), ret);
        SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmLinkFailed);
        CloseConn(hccl_conn);
        return FsmStatus::kFsmFailed;
    }
    update_link_states_[cluster_index] = DoLinkState::kLinkTrySend;
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkReqHandler::TrySendClusterInfo(size_t cluster_index, ClusterAddrInfo &cluster,
                                             const std::string &cluster_desc, const HcclConn &hccl_conn) {
    HcclRequest send_request;
    update_link_client_infos_[cluster_index].cluster_id = cluster.local_cluster_id;
    HcclResult hccl_ret = HcclRawIsend(&update_link_client_infos_[cluster_index],
                                      static_cast<int32_t>(sizeof(ClientClusterInfo)), HCCL_DATA_TYPE_INT8, hccl_conn,
                                      &send_request);
    if (hccl_ret == HCCL_E_AGAIN) {
        UDF_RUN_LOG_INFO("HcclRawIsend need try again, cluster:%s, ret:%d.", cluster_desc.c_str(), hccl_ret);
        return FsmStatus::kFsmKeepState;
    }
    if (hccl_ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawIsend, remote cluster:%s, ret:%d.", cluster_desc.c_str(), hccl_ret);
        SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmLinkFailed);
        CloseConn(hccl_conn);
        return FsmStatus::kFsmHcclFailed;
    }
    update_link_requests_[cluster_index] = send_request;
    update_link_states_[cluster_index] = DoLinkState::kLinkTryTest;
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkReqHandler::TryTestSendClusterInfo(size_t cluster_index, ClusterAddrInfo &cluster,
                                                 const std::string &cluster_desc, const HcclConn &hccl_conn) {
    int32_t comp_count = 0;
    int32_t comp_index = 0;
    HcclStatus comp_status{};
    HcclResult ret = HcclRawTestSome(1, &update_link_requests_[cluster_index], &comp_count, &comp_index, &comp_status);
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawTestSome, remote cluster:%s, ret:%d.", cluster_desc.c_str(), ret);
        SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmLinkFailed);
        CloseConn(hccl_conn);
        return FsmStatus::kFsmFailed;
    }
    if (comp_count == 1) {
        if (comp_status.error != 0) {
            UDF_LOG_ERROR("Fail to test, unexpected comp status, remote cluster:%s.", cluster_desc.c_str());
            SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmLinkFailed);
            CloseConn(hccl_conn);
            return FsmStatus::kFsmFailed;
        }
        auto entity =
            LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, hccl_conn, cluster.local_hccl_addr,
                                                         cluster.remote_hccl_addr, cluster.remote_cluster_id);
        if (entity == nullptr) {
            UDF_LOG_ERROR("Fail to create entity, remote cluster:%s.", cluster_desc.c_str());
            SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmFailed);
            CloseConn(hccl_conn);
            return FsmStatus::kFsmFailed;
        }
        FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
        update_link_comp_flags_[cluster_index] = true;
        update_link_comp_count_++;
        update_link_ret_[cluster_index] = static_cast<int32_t>(FsmStatus::kFsmSuccess);
        bool max_tick_cost_flag = false;
        const uint64_t
            tick_cost = StatisticManager::GetInstance().GetCpuTick() - update_link_start_ticks_[cluster_index];
        entity->GetStatisticInfo().link_tick_cost = tick_cost;
        UpdateTickCost(tick_cost, stat_info.link_succ_times, stat_info.link_min_tick_cost, stat_info.link_max_tick_cost,
                       stat_info.link_total_tick_cost, max_tick_cost_flag);
        const double k_time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
        UDF_LOG_INFO("Success to link%s to remote cluster:%s, time cost:%.2f us, link succ times:%lu.",
                     max_tick_cost_flag ? kMaxTimeCost : kCommonTimeCost, cluster_desc.c_str(), k_time_cost,
                     stat_info.link_succ_times.load());
    } else if (comp_count > 1) {
        UDF_LOG_ERROR("Fail to test, unexpected result cnt, remote cluster:%s.", cluster_desc.c_str());
        SetResult(LinkReqType::kLink, cluster_index, FsmStatus::kFsmLinkFailed);
        CloseConn(hccl_conn);
        return FsmStatus::kFsmKeepState;
    }
    return FsmStatus::kFsmSuccess;
}

void LinkReqHandler::DoLink(size_t cluster_index, ClusterAddrInfo &cluster) {
    std::string cluster_desc = ToString(cluster);
    UDF_LOG_DEBUG("Begin to link, remote cluster:%s.", cluster_desc.c_str());
    if (update_link_conns_[cluster_index] == nullptr) {
        if (InitClientConn(cluster_index, cluster, cluster_desc) != FsmStatus::kFsmSuccess) {
            return;
        }
    }
    HcclConn hccl_conn = update_link_conns_[cluster_index];
    if (update_link_states_[cluster_index] == DoLinkState::kLinkTryBind) {
        if (TryBindConn(cluster_index, cluster, cluster_desc, hccl_conn) != FsmStatus::kFsmSuccess) {
            return;
        }
    }
    if (update_link_states_[cluster_index] == DoLinkState::kLinkTryConnect) {
        if (TryConnectServer(cluster_index, cluster, cluster_desc, hccl_conn) != FsmStatus::kFsmSuccess) {
            return;
        }
    }
    if (update_link_states_[cluster_index] == DoLinkState::kLinkTrySend) {
        if (TrySendClusterInfo(cluster_index, cluster, cluster_desc, hccl_conn) != FsmStatus::kFsmSuccess) {
            return;
        }
    }
    if (update_link_states_[cluster_index] == DoLinkState::kLinkTryTest) {
        if (TryTestSendClusterInfo(cluster_index, cluster, cluster_desc, hccl_conn) != FsmStatus::kFsmSuccess) {
            return;
        }
    }
}

FsmStatus LinkReqHandler::NotifyPromptUnlink(const EntityPtr &entity, size_t cluster_index, ClusterAddrInfo &cluster) {
    std::string cluster_desc = ToString(cluster);
    FuncStatisticInfo &stat_info = StatisticManager::GetInstance().GetStatisticInfo();
    if (!send_unlink_req_comp_flags_[cluster_index]) {
        HcclRequest request;
        HcclResult ret = HcclRawIsend(nullptr, 0, HcclDataType::HCCL_DATA_TYPE_INT8, update_link_conns_[cluster_index], &request);
        if (ret == HCCL_E_AGAIN) {
            stat_info.unlink_fail_times++;
            UDF_RUN_LOG_INFO("Need retry to send empty msg for unlink, remote cluster:%s.", cluster_desc.c_str());
            return FsmStatus::kFsmKeepState;
        }
        if (ret != HCCL_SUCCESS) {
            SetResult(LinkReqType::kUnlink, cluster_index, FsmStatus::kFsmNotifyPromptUnlinkFailed);
            (void) CleanEntityAndConn(entity, cluster_index, cluster.remote_cluster_id);
            UDF_LOG_ERROR("Fail to send empty msg for unlink, remote cluster:%s, ret:%d", cluster_desc.c_str(), ret);
            return FsmStatus::kFsmHcclFailed;
        }
        update_link_requests_[cluster_index] = request;
        send_unlink_req_comp_flags_[cluster_index] = true;
    }
    int32_t comp_count = 0;
    int32_t comp_index = 0;
    HcclStatus comp_status{};
    HcclResult ret = HcclRawTestSome(1, &update_link_requests_[cluster_index], &comp_count, &comp_index, &comp_status);
    if (ret != HcclResult::HCCL_SUCCESS) {
        SetResult(LinkReqType::kUnlink, cluster_index, FsmStatus::kFsmNotifyPromptUnlinkFailed);
        (void) CleanEntityAndConn(entity, cluster_index, cluster.remote_cluster_id);
        UDF_LOG_ERROR("Fail to test for unlink, remote cluster:%s, ret:%d.", cluster_desc.c_str(), ret);
        return FsmStatus::kFsmHcclFailed;
    }
    if (comp_count == 0) {
        return FsmStatus::kFsmKeepState;
    }
    if (comp_status.error != 0) {
        SetResult(LinkReqType::kUnlink, cluster_index, FsmStatus::kFsmNotifyPromptUnlinkFailed);
        (void) CleanEntityAndConn(entity, cluster_index, cluster.remote_cluster_id);
        UDF_LOG_ERROR("Get invalid test status, status:%d, remote_cluster:%s.", comp_status.error,
                      cluster_desc.c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    return FsmStatus::kFsmSuccess;
}

void LinkReqHandler::DoUnlink(size_t cluster_index, ClusterAddrInfo &cluster) {
    // no need check nullptr
    auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(cluster.remote_cluster_id);
    update_link_conns_[cluster_index] = entity->GetConn();
    if (update_link_req_info_->force_flag == 0U) {
        if (NotifyPromptUnlink(entity, cluster_index, cluster) != FsmStatus::kFsmSuccess) {
            return;
        }
    }
    FsmStatus closeRet = CleanEntityAndConn(entity, cluster_index, cluster.remote_cluster_id);
    if (closeRet != FsmStatus::kFsmSuccess) {
        SetResult(LinkReqType::kUnlink, cluster_index, FsmStatus::kFsmUnlinkFailed);
        return;
    }
    SetResult(LinkReqType::kUnlink, cluster_index, FsmStatus::kFsmSuccess);
}

void LinkReqHandler::DoServerUnlink(size_t cluster_index, ClusterAddrInfo &cluster) {
    ClusterInfo &cluster_info = update_link_req_info_->cluster_infos[cluster_index];
    const IpInfo &ip_info = cluster_info.ip_infos[device_index_];
    HcclConn conn = LlmCommEntityMgr::GetInstance().GetEntityByIp(ip_info.remote_ip);
    if (conn == nullptr) {
        return SetResult(LinkReqType::kServerUnlink, cluster_index, FsmStatus::kFsmNotLink);
    }
    update_link_conns_[cluster_index] = conn;
    auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(cluster.remote_cluster_id);
    if (entity != nullptr) {
        if (!entity->GetUnlinkMutex().try_lock()) {
            UDF_LOG_ERROR("Link is busy now.");
            SetResult(LinkReqType::kUnlink, cluster_index, FsmStatus::kFsmLinkBusy);
            return;
        }
        std::lock_guard<std::mutex> unlinkLock(entity->GetUnlinkMutex(), std::adopt_lock);
        std::lock_guard<std::mutex> lock(entity->GetMutex());
        FsmStatus closeRet = CleanEntityAndConn(entity, cluster_index, cluster.remote_cluster_id, true);
        if (closeRet != FsmStatus::kFsmSuccess) {
            SetResult(LinkReqType::kServerUnlink, cluster_index, FsmStatus::kFsmUnlinkFailed);
            return;
        }
    } else {
        return SetResult(LinkReqType::kServerUnlink, cluster_index, FsmStatus::kFsmNotLink);
    }
    SetResult(LinkReqType::kServerUnlink, cluster_index, FsmStatus::kFsmSuccess);
}

FsmStatus LinkReqHandler::CleanEntityAndConn(const EntityPtr &entity, size_t cluster_index, uint64_t remote_cluster_id,
                                             bool is_server) {
    HcclConn conn = update_link_conns_[cluster_index];
    if (entity != nullptr) {
        entity->GetStatisticInfo().unlink_tick_cost =
                StatisticManager::GetInstance().GetCpuTick() - update_link_start_ticks_[cluster_index];
        entity->Dump();
        if (is_server) {
            return entity->MarkEntityDeletedByConn();
        } else {
            (void) LlmCommEntityMgr::GetInstance().DeleteEntityByRemoteClusterId(remote_cluster_id);
            return CloseConn(conn);
        }
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LinkReqHandler::CloseConn(const HcclConn &conn) {
    HcclResult hccl_ret = HcclRawForceClose(conn);
    if (hccl_ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to close hccl conn:%p", conn);
        return FsmStatus::kFsmUnlinkFailed;
    }
    UDF_LOG_INFO("Success to close hccl conn:%p", conn);
    return FsmStatus::kFsmSuccess;
}

void LinkReqHandler::SetOutput(size_t output_count, FsmStatus status) {
    for (size_t index = 0UL; index < output_count; index++) {
        if (!update_link_comp_flags_[index]) {
            if (status == FsmStatus::kFsmTimeout) {
                UDF_RUN_LOG_INFO("Timeout, cluster index:%zu, cur state:%d",
                                 index, static_cast<int32_t>(update_link_states_[index]));
            }
            update_link_ret_[index] = static_cast<int32_t>(status);
        }
    }
}

void LinkReqHandler::ResetNotifyPromptFailed(size_t output_count) {
    for (size_t index = 0UL; index < output_count; index++) {
        if (!update_link_comp_flags_[index] && update_link_states_[index] == DoLinkState::kLinkTrySend) {
            update_link_ret_[index] = static_cast<int32_t>(FsmStatus::kFsmNotifyPromptUnlinkFailed);
        }
    }
}

void LinkReqHandler::CloseTimeoutConn(size_t cluster_num, LinkReqType req_type, const DevLinkReqInfo &dev_link_req_info) {
    for (size_t index = 0UL; index < cluster_num; index++) {
        if (update_link_ret_[index] == static_cast<int32_t>(FsmStatus::kFsmTimeout)) {
            if (req_type == LinkReqType::kUnlink) {
                const ClusterAddrInfo &cluster = dev_link_req_info.cluster_infos.at(index);
                auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(cluster.remote_cluster_id);
                (void) CleanEntityAndConn(entity, index, cluster.remote_cluster_id);
            } else {
                (void) CloseConn(update_link_conns_[index]);
            }
        }
    }
}

std::string LinkReqHandler::ToString(const ClusterAddrInfo &cluster) {
    std::string desc;
    uint32_t local_ip = cluster.local_hccl_addr.info.tcp.ipv4Addr;
    uint16_t local_port = cluster.local_hccl_addr.info.tcp.port;
    uint32_t remote_ip = cluster.remote_hccl_addr.info.tcp.ipv4Addr;
    uint16_t remote_port = cluster.remote_hccl_addr.info.tcp.port;
    desc.append("[remote_cluster_id:")
        .append(std::to_string(cluster.remote_cluster_id))
        .append(", local ip:")
        .append(ToIp(local_ip))
        .append("(")
        .append(std::to_string(local_ip))
        .append(")_")
        .append(std::to_string(local_port))
        .append(" ,remote ip:")
        .append(ToIp(remote_ip))
        .append("(")
        .append(std::to_string(remote_ip))
        .append(")_")
        .append(std::to_string(remote_port))
        .append("]");
    return desc;
}
}  // namespace FlowFunc