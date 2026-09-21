/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_ENTITY_LLM_COMM_ENTITY_MGR_H
#define BUILT_IN_ENTITY_LLM_COMM_ENTITY_MGR_H

#include <mutex>
#include <unordered_map>
#include <vector>
#include "entity/llm_comm_entity.h"
#include "fsm/state_define.h"
#include "llm_common/llm_common.h"

namespace FlowFunc {
class LlmCommEntityMgr {
public:
    static LlmCommEntityMgr &GetInstance();
    ~LlmCommEntityMgr();
    EntityPtr GetEntityByConn(HcclConn conn);
    EntityPtr GetEntityByRemoteClusterId(uint64_t remote_cluster_id);
    HcclConn GetEntityByIp(uint32_t ip) const;
    EntityPtr CreateEntity(EntityType type, HcclConn conn, HcclAddr &local_hccl_addr,
                           HcclAddr &remote_hccl_addr, uint64_t remote_cluster_id = 0);
    FsmStatus MarkEntityDeletedByConn(HcclConn conn);
    FsmStatus DeleteEntityByRemoteClusterId(uint64_t remote_cluster_id);
    std::vector<int32_t> &GetCompIndices(size_t req_size);
    std::vector<HcclStatus> &GetCompStatus(size_t req_size);
    FsmStatus InitServerConn(uint32_t ip, uint16_t port, bool need_lock = true);
    void FinalizeServerConn();
    static FsmStatus InitClientConn(HcclAddr &local_hccl_addr, HcclConn &hccl_conn);
    void HandleRequest(bool is_prompt);
    void PromptHandleReq();
    void DecoderHandleReq();
    void HandleLinkRequest();
    size_t GetEntityMapSize();
    static FsmStatus RegisterHcclMr(uint32_t dev_id, std::vector<uint64_t> &mem_addrs);
    static FsmStatus UnRegisterHcclMr(std::vector<uint64_t> &mem_addrs);
    void ClearEntities();
    void DumpServerEntities();
    void DumpClientEntities();
    bool HasAnyLink();
    void AddClientEntityMap(uint64_t remote_cluster_id, EntityPtr entity);
    size_t QueryLinkNum() const;
    LlmCommEntityMgr(const LlmCommEntityMgr &) = delete;
    LlmCommEntityMgr(const LlmCommEntityMgr &&) = delete;
    LlmCommEntityMgr &operator=(const LlmCommEntityMgr &) = delete;
    LlmCommEntityMgr &operator=(const LlmCommEntityMgr &&) = delete;
private:
    LlmCommEntityMgr();
    void EraseClientMapByClusterId(uint64_t remote_cluster_id);
    void EraseIpToConnMap(uint32_t ip, const HcclConn conn);
    static FsmStatus QueryCurMemGrp(char **group_name);
    void ReopenServerConn();
    std::unordered_map<HcclConn, EntityPtr> server_entity_map_;
    std::unordered_multimap<uint32_t, HcclConn> ip_to_conns_;  // check risk link for server
    std::unordered_map<uint64_t, EntityPtr> client_entity_map_;
    HcclConn listen_conn_;
    HcclAddr listen_hccl_addr_;
    std::vector<int32_t> comp_indices_;
    std::vector<HcclStatus> comp_status_;
    // mutex for server map and client map
    std::mutex entity_mutex_;
    std::mutex switch_mutex_;
    std::atomic_bool mgr_need_use_mtx_{false};
    std::atomic_bool initialized_ {false};
    uint32_t server_ip_ = 0U;
    uint16_t server_port_ = 0U;
    bool server_conn_inited_ = false;
};
}  // namespace FlowFunc
#endif  // BUILT_IN_ENTITY_LLM_COMM_ENTITY_MGR_H