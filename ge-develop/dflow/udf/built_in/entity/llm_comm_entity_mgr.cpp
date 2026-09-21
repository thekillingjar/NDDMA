/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "entity/llm_comm_entity_mgr.h"
#include <memory>
#include "securec.h"
#include "ascend_hal_define.h"
#include "ascend_hal.h"
#include "fsm/state_define.h"
#include "llm_common/hccl_proxy.h"
#include "llm_common/llm_common.h"
#include "common/scope_guard.h"

namespace FlowFunc {
namespace {
constexpr size_t kDefaultMultiRequestCount = 1024UL;
constexpr uint64_t kCheckTimeoutLoopCount = 1000UL;
constexpr uint64_t kProcessTimeout = 1000000UL;         // 1s
constexpr uint64_t kResetProfilingTimeInterval = 60UL * 60UL;  // one hour
}  // namespace

LlmCommEntityMgr &LlmCommEntityMgr::GetInstance() {
    static LlmCommEntityMgr manager;
    return manager;
}

LlmCommEntityMgr::LlmCommEntityMgr() : listen_conn_(nullptr), listen_hccl_addr_({}) {
    comp_indices_.resize(kDefaultMultiRequestCount);
    comp_status_.resize(kDefaultMultiRequestCount);
}

LlmCommEntityMgr::~LlmCommEntityMgr() {
    ClearEntities();
}

EntityPtr LlmCommEntityMgr::GetEntityByConn(HcclConn conn) {
    std::lock_guard<std::mutex> lock(entity_mutex_);
    auto iter = server_entity_map_.find(conn);
    if (iter != server_entity_map_.end()) {
        return iter->second;
    }
    UDF_LOG_INFO("Not exist entity, conn:%p.", conn);
    return nullptr;
}

HcclConn LlmCommEntityMgr::GetEntityByIp(uint32_t ip) const {
    auto iter = ip_to_conns_.find(ip);
    if (iter != ip_to_conns_.end()) {
        return iter->second;
    }
    UDF_RUN_LOG_WARN("Entity is not exist for remote ip:%u.", ip);
    return nullptr;
}

size_t LlmCommEntityMgr::GetEntityMapSize() {
    mgr_need_use_mtx_.store(true, std::memory_order_relaxed);
    ScopeGuard guard([this] { mgr_need_use_mtx_.store(false, std::memory_order_relaxed); });
    std::lock_guard<std::mutex> lock(entity_mutex_);
    return server_entity_map_.size();
}

EntityPtr LlmCommEntityMgr::GetEntityByRemoteClusterId(uint64_t remote_cluster_id) {
    mgr_need_use_mtx_.store(true, std::memory_order_relaxed);
    ScopeGuard guard([this] { mgr_need_use_mtx_.store(false, std::memory_order_relaxed); });
    std::lock_guard<std::mutex> lock(entity_mutex_);
    auto iter = client_entity_map_.find(remote_cluster_id);
    if ((iter != client_entity_map_.end()) && (iter->second->GetCurState() != FsmState::kFsmDestroyState)) {
        return iter->second;
    }
    UDF_RUN_LOG_WARN("Not exist entity, remote_cluster_id:%lu.", remote_cluster_id);
    return nullptr;
}

EntityPtr LlmCommEntityMgr::CreateEntity(EntityType type, HcclConn conn, HcclAddr &local_hccl_addr,
                                         HcclAddr &remote_hccl_addr, uint64_t remote_cluster_id) {
    EntityPtr entity;
    try {
        entity = std::make_shared<LlmCommEntity>(type, conn, local_hccl_addr, remote_hccl_addr);
    } catch (const std::bad_alloc &) {
        UDF_LOG_ERROR("Make shared failed");
        return nullptr;
    }
    mgr_need_use_mtx_.store(true, std::memory_order_relaxed);
    ScopeGuard guard([this] { mgr_need_use_mtx_.store(false, std::memory_order_relaxed); });
    UDF_LOG_INFO("Set high priority flag.");
    std::lock_guard<std::mutex> lock(entity_mutex_);
    if (type == EntityType::kEntityServer) {
        server_entity_map_[conn] = entity;
        (void) ip_to_conns_.emplace(remote_hccl_addr.info.tcp.ipv4Addr, conn);
    } else {
        client_entity_map_[remote_cluster_id] = entity;
    }
    UDF_LOG_INFO("Success to create entity:%s.", entity->GetDesc().c_str());
    return entity;
}

void LlmCommEntityMgr::AddClientEntityMap(uint64_t remote_cluster_id, EntityPtr entity) {
    std::lock_guard<std::mutex> lock(entity_mutex_);
    entity->SetRemoteClusterId(remote_cluster_id);
    UDF_LOG_INFO("Add client entity for cluster:%lu", remote_cluster_id);
    (void)client_entity_map_.emplace(remote_cluster_id, entity);
}

FsmStatus LlmCommEntityMgr::DeleteEntityByRemoteClusterId(uint64_t remote_cluster_id) {
    mgr_need_use_mtx_.store(true, std::memory_order_relaxed);
    ScopeGuard guard([this] { mgr_need_use_mtx_.store(false, std::memory_order_relaxed); });
    std::lock_guard<std::mutex> lock(entity_mutex_);
    auto iter = client_entity_map_.find(remote_cluster_id);
    if (iter == client_entity_map_.end()) {
        UDF_LOG_INFO("Not exist remote_cluster_id:%lu.", remote_cluster_id);
        return FsmStatus::kFsmSuccess;
    }
    UDF_LOG_INFO("Delete entity:%s.", iter->second->GetDesc().c_str());
    (void) client_entity_map_.erase(iter);
    return FsmStatus::kFsmSuccess;
}

std::vector<int32_t> &LlmCommEntityMgr::GetCompIndices(size_t req_size) {
    if (comp_indices_.size() < req_size) {
        comp_indices_.resize(req_size);
    }
    return comp_indices_;
}

std::vector<HcclStatus> &LlmCommEntityMgr::GetCompStatus(size_t req_size) {
    if (comp_status_.size() < req_size) {
        comp_status_.resize(req_size);
    }
    return comp_status_;
}

FsmStatus LlmCommEntityMgr::InitServerConn(uint32_t ip, uint16_t port, bool need_lock) {
    server_conn_inited_ = false;
    uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    if (need_lock) {
        std::lock_guard<std::mutex> lock(switch_mutex_);
    }
    HcclResult ret = HcclRawOpen(&listen_conn_);
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Call HcclRawOpen failed, ret:%d.", ret);
        return FsmStatus::kFsmHcclFailed;
    }
    listen_hccl_addr_.type = HcclAddrType::HCCL_ADDR_TYPE_ROCE;
    listen_hccl_addr_.info.tcp.ipv4Addr = ip;
    listen_hccl_addr_.info.tcp.port = port;
    ret = HcclRawBind(listen_conn_, &listen_hccl_addr_);
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Bind server conn failed, ret:%d.", ret);
        return FsmStatus::kFsmHcclFailed;
    }
    ret = HcclRawListen(listen_conn_, 1);
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Listen server conn failed, ret:%d.", ret);
        return FsmStatus::kFsmHcclFailed;
    }
    initialized_ = true;
    UDF_LOG_INFO("Init server conn time cost:%.2f us.",
                 StatisticManager::GetInstance().GetTimeCost(StatisticManager::GetInstance().GetCpuTick()-start_tick));
    server_ip_ = ip;
    server_port_ = port;
    server_conn_inited_ = true;
    return FsmStatus::kFsmSuccess;
}

void LlmCommEntityMgr::ReopenServerConn() {
    if (listen_conn_ != nullptr) {
        auto ret = HcclRawForceClose(listen_conn_);
        if (ret != HCCL_SUCCESS) {
            UDF_LOG_ERROR("Close conn failed, ret:%d.", ret);
        }
        listen_conn_ = nullptr;
    }
    auto init_ret = InitServerConn(server_ip_, server_port_, false);
    if (init_ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Init server conn failed, ret:%d.", static_cast<int32_t>(init_ret));
    }
}

FsmStatus LlmCommEntityMgr::InitClientConn(HcclAddr &local_hccl_addr, HcclConn &hccl_conn) {
    HcclResult ret = HcclRawOpen(&hccl_conn);
    if (ret != HcclResult::HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to create local conn, local_hccl_addr:%s, ret:%d.", ToDesc(local_hccl_addr).c_str(),
                      ret);
        return FsmStatus::kFsmHcclFailed;
    }
    return FsmStatus::kFsmSuccess;
}

void LlmCommEntityMgr::PromptHandleReq() {
    HandleLinkRequest();
    // process all conn entities: no need lock
    for (auto iter = server_entity_map_.begin(); iter != server_entity_map_.end();) {
        if (iter->second->GetCurState() == FsmState::kFsmErrorState) {
            iter++;
            continue;
        }
        if (iter->second->GetReqIsUsing().load(std::memory_order_relaxed) &&
            !iter->second->GetEntityOccupied().load()) {
            iter++;
            continue;
        }
        auto &mutex = iter->second->GetMutex();
        if (mutex.try_lock()) {
            std::lock_guard<std::mutex> lock(mutex, std::adopt_lock);
            if (iter->second->GetCurState() == FsmState::kFsmDestroyState) {
                std::lock_guard<std::mutex> mapLock(entity_mutex_);
                UDF_LOG_INFO("start erase entity:%lu.", iter->second->GetRemoteClusterId());
                EraseIpToConnMap(iter->second->GetRemoteIp(), iter->second->GetConn());
                EraseClientMapByClusterId(iter->second->GetRemoteClusterId());
                iter = server_entity_map_.erase(iter);
                continue;
            }
            bool is_init_or_link = (iter->second->GetCurState() == FsmState::kFsmInitState) ||
                                (iter->second->GetCurState() == FsmState::kFsmLinkState);
            FsmStatus status = iter->second->ProcessState();
            if (is_init_or_link && status == FsmStatus::kFsmEstablishLinkSuc) {
                AddClientEntityMap(iter->second->GetClientClusterInfo().cluster_id, iter->second);
            } else if ((status != FsmStatus::kFsmSuccess) && (status != FsmStatus::kFsmKeepState)) {
                (void) iter->second->ChangeState(FsmState::kFsmErrorState);
            }
        }
        iter++;
    }
}

void LlmCommEntityMgr::DecoderHandleReq() {
    if (mgr_need_use_mtx_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> mapLock(entity_mutex_);
    for (auto iter = client_entity_map_.begin(); iter != client_entity_map_.end();) {
        if (iter->second->GetCurState() == FsmState::kFsmErrorState) {
            iter++;
            continue;
        }
        if (iter->second->GetReqIsUsing().load(std::memory_order_relaxed) &&
            !iter->second->GetEntityOccupied().load()) {
            iter++;
            continue;
        }
        auto &mutex = iter->second->GetMutex();
        if (!iter->second->GetIsUnlinking().load(std::memory_order_relaxed) && mutex.try_lock()) {
            std::lock_guard<std::mutex> lock(mutex, std::adopt_lock);
            FsmStatus status = iter->second->ProcessState();
            if ((status != FsmStatus::kFsmSuccess) && (status != FsmStatus::kFsmKeepState)) {
                (void) iter->second->ChangeState(FsmState::kFsmErrorState);
            }
        }
        iter++;
    }
    UDF_LOG_DEBUG("DecoderHandleReq free lock.");
}

void LlmCommEntityMgr::HandleRequest(bool is_prompt) {
    static uint64_t func_execute_count = 0UL;
    func_execute_count++;
    uint64_t start_tick = StatisticManager::GetInstance().GetCpuTick();
    uint32_t loop_count = 0;
    UDF_LOG_DEBUG("Enter HandleRequest.");
    while ((loop_count < kCheckTimeoutLoopCount) ||
        (CheckTimeout(start_tick, kProcessTimeout) != FsmStatus::kFsmTimeout)) {
        loop_count++;
        if (loop_count > kCheckTimeoutLoopCount) {
            loop_count = 0U;
        }
        if (is_prompt) {
            if (!initialized_.load()) {
                continue;
            }
            std::lock_guard<std::mutex> lk(switch_mutex_);
            PromptHandleReq();
        } else {
            DecoderHandleReq();
        }
    }
    // one hour check statistic info
    if (func_execute_count % kResetProfilingTimeInterval == 0UL) {
        StatisticManager::GetInstance().ResetProfilingTrack();
    }
}

void LlmCommEntityMgr::HandleLinkRequest() {
    if (!server_conn_inited_) {
        ReopenServerConn();
        return;
    }
    HcclConn hccl_conn = nullptr;
    HcclAddr remote_hccl_addr{};
    HcclResult accept_ret = HcclRawAccept(listen_conn_, &remote_hccl_addr, &hccl_conn);
    if ((accept_ret != HCCL_SUCCESS) && (accept_ret != HCCL_E_AGAIN)) {
        UDF_LOG_ERROR("Fail to call HcclRawAccept, ret:%d.", accept_ret);
        ReopenServerConn();
        return;
    }
    if (hccl_conn == nullptr) {
        return;
    }
    // accept new link
    const uint32_t remote_ip = remote_hccl_addr.info.tcp.ipv4Addr;
    auto iter = ip_to_conns_.find(remote_ip);
    if (iter != ip_to_conns_.end()) {
        auto entity = GetEntityByConn(iter->second);
        if (entity == nullptr) {
            EraseIpToConnMap(remote_ip, iter->second);
            UDF_RUN_LOG_INFO("Success to accept new link with residual data in ip_to_conns map, remote hccl addr:%s.",
                             ToDesc(remote_hccl_addr).c_str());
        } else {
            (void) HcclRawForceClose(entity->GetConn());
            entity->SetConn(hccl_conn);
            entity->SetLinkEstablished(false);
            entity->SetProbeLinkClusterInfoFlag(false);
            entity->ClearResource();
            entity->ChangeState(FsmState::kFsmLinkState);
            UDF_RUN_LOG_INFO("Success to accept new force link, remote hccl addr:%s.", ToDesc(remote_hccl_addr).c_str());
            return;
        }
    }
    EntityPtr entity = this->CreateEntity(EntityType::kEntityServer, hccl_conn, listen_hccl_addr_, remote_hccl_addr);
    if (entity == nullptr) {
        UDF_LOG_ERROR("failed to create server comm entity.");
        return;
    }
    entity->GetServerTickRecord().link_start_tick = StatisticManager::GetInstance().GetCpuTick();
    UDF_LOG_INFO("Success to accept new link, remote hccl addr:%s.", ToDesc(remote_hccl_addr).c_str());
}

FsmStatus LlmCommEntityMgr::RegisterHcclMr(uint32_t dev_id, std::vector<uint64_t> &mem_addrs) {
    GroupQueryInput drv_input;
    error_t ret = memset_s(&drv_input, sizeof(drv_input), 0, sizeof(drv_input));
    if (ret != EOK) {
        UDF_LOG_ERROR("Memset failed, ret=%d!", ret);
        return FsmStatus::kFsmFailed;
    }
    char *grp_name_ptr = drv_input.grpQueryGroupAddrPara.grpName;
    FsmStatus query_grp_ret = QueryCurMemGrp(&grp_name_ptr);
    if (query_grp_ret != FsmStatus::kFsmSuccess) {
        return query_grp_ret;
    }
    UDF_LOG_INFO("Current group name:%s", grp_name_ptr);
    drv_input.grpQueryGroupAddrPara.devId = dev_id;
    const auto k_drv_input_len = static_cast<uint32_t>(sizeof(drv_input));
    const std::unique_ptr<GroupQueryOutput> k_drv_output_ptr(new(std::nothrow) GroupQueryOutput());
    if (k_drv_output_ptr == nullptr) {
        UDF_LOG_ERROR("Malloc failed.");
        return FsmStatus::kFsmFailed;
    }
    uint32_t drv_output_len = 0U;
    GroupQueryOutput *drv_output = k_drv_output_ptr.get();
    int32_t drv_ret = halGrpQuery(GRP_QUERY_GROUP_ADDR_INFO, &drv_input, k_drv_input_len, drv_output,
                                 &drv_output_len);
    if (drv_ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        UDF_LOG_ERROR("Call halGrpQuery failed, ret=%d.", drv_ret);
        return FsmStatus::kFsmDrvFailed;
    }
    size_t output_num = (static_cast<size_t>(drv_output_len) / sizeof(GrpQueryGroupAddrInfo));
    for (size_t i = 0; i < output_num; ++i) {
        void *addr_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(drv_output->grpQueryGroupAddrInfo[i].addr));
        HcclResult hccl_ret = HcclRegisterGlobalMemory(addr_ptr, drv_output->grpQueryGroupAddrInfo[i].size);
        if (hccl_ret != HcclResult::HCCL_SUCCESS) {
            UDF_LOG_ERROR("Call HcclRegisterGlobalMemory failed, ret:%d.", hccl_ret);
            return FsmStatus::kFsmHcclFailed;
        }
        UDF_LOG_INFO("Register mr success, addr:%llu, size:%llu.", drv_output->grpQueryGroupAddrInfo[i].addr,
                     drv_output->grpQueryGroupAddrInfo[i].size);
        mem_addrs.emplace_back(drv_output->grpQueryGroupAddrInfo[i].addr);
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntityMgr::QueryCurMemGrp(char **group_name) {
    const std::unique_ptr<GroupQueryOutput> k_drv_output_ptr(new(std::nothrow) GroupQueryOutput());
    if (k_drv_output_ptr == nullptr) {
        UDF_LOG_ERROR("Malloc failed.");
        return FsmStatus::kFsmFailed;
    }
    GroupQueryOutput *drv_output = k_drv_output_ptr.get();
    uint32_t drv_output_len = 0U;
    auto cur_pid = drvDeviceGetBareTgid();
    auto drv_ret = halGrpQuery(GRP_QUERY_GROUPS_OF_PROCESS, &cur_pid, static_cast<uint32_t>(sizeof(cur_pid)),
                              drv_output,
                              &drv_output_len);
    if (drv_ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        UDF_LOG_ERROR("Call halGrpQuery failed, ret=%d.", drv_ret);
        return FsmStatus::kFsmDrvFailed;
    }
    if (drv_output_len == 0U) {
        UDF_LOG_ERROR("Query current mem group failed, size is zero.");
        return FsmStatus::kFsmFailed;
    }
    if ((drv_output_len / sizeof(drv_output->grpQueryGroupsOfProcInfo[0])) > 1) {
        UDF_LOG_WARN("Query current mem group not expected over 1, size:%u.", drv_output_len);
    }
    errno_t ret = strcpy_s(*group_name, sizeof(drv_output->grpQueryGroupsOfProcInfo[0].groupName),
                           drv_output->grpQueryGroupsOfProcInfo[0].groupName);
    if (ret != EOK) {
        UDF_LOG_ERROR("Copy group name failed, ret=%d!", ret);
        return FsmStatus::kFsmFailed;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntityMgr::UnRegisterHcclMr(std::vector<uint64_t> &mem_addrs) {
    for (const auto &k_mem_addr: mem_addrs) {
        void *addr_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(k_mem_addr));
        HcclResult ret = HcclUnregisterGlobalMemory(addr_ptr);
        if (ret != HcclResult::HCCL_SUCCESS) {
            UDF_LOG_ERROR("Unregister mr failed, ret:%d, addr:%lu", ret, k_mem_addr);
            return FsmStatus::kFsmHcclFailed;
        }
        UDF_LOG_INFO("Unregister mr success, addr:%lu", k_mem_addr);
    }
    return FsmStatus::kFsmSuccess;
}

void LlmCommEntityMgr::ClearEntities() {
    ip_to_conns_.clear();
    std::lock_guard<std::mutex> lock(entity_mutex_);
    server_entity_map_.clear();
    client_entity_map_.clear();
}

void LlmCommEntityMgr::EraseIpToConnMap(uint32_t ip, const HcclConn conn) {
    for (auto multi_iter = ip_to_conns_.find(ip); multi_iter != ip_to_conns_.end(); multi_iter++) {
        if (multi_iter->second == conn) {
            (void) ip_to_conns_.erase(multi_iter);
            return;
        }
    }
}

void LlmCommEntityMgr::EraseClientMapByClusterId(uint64_t remote_cluster_id) {
    auto iter = client_entity_map_.find(remote_cluster_id);
    if (iter != client_entity_map_.end()) {
        client_entity_map_.erase(iter);
    }
}

void LlmCommEntityMgr::DumpServerEntities() {
    std::unordered_map<HcclConn, EntityPtr> entity_map;
    {
        std::lock_guard<std::mutex> lock(entity_mutex_);
        entity_map = server_entity_map_;
    }
    for (auto &iter : entity_map) {
        iter.second->Dump();
    }
}

void LlmCommEntityMgr::DumpClientEntities() {
    std::unordered_map<uint64_t, EntityPtr> entity_map;
    {
        std::lock_guard<std::mutex> lock(entity_mutex_);
        entity_map = client_entity_map_;
    }
    for (auto &iter : entity_map) {
        iter.second->Dump();
    }
}

bool LlmCommEntityMgr::HasAnyLink() {
    mgr_need_use_mtx_.store(true, std::memory_order_relaxed);
    ScopeGuard guard([this] { mgr_need_use_mtx_.store(false, std::memory_order_relaxed); });
    std::lock_guard<std::mutex> lock(entity_mutex_);
    UDF_LOG_INFO("server_entity_map_.size=%zu, client_entity_map_.size=%zu",
                 server_entity_map_.size(), client_entity_map_.size());
    return (!server_entity_map_.empty()) || (!client_entity_map_.empty());
}

void LlmCommEntityMgr::FinalizeServerConn() {
    initialized_.store(false);
    std::lock_guard<std::mutex> lock(switch_mutex_);
    if (listen_conn_ != nullptr) {
        (void) HcclRawClose(listen_conn_);
        listen_conn_ = nullptr;
        UDF_LOG_INFO("server listen conn closed");
    }
}

size_t LlmCommEntityMgr::QueryLinkNum() const {
    return client_entity_map_.size();
}
}  // namespace FlowFunc