/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_ENTITY_LINK_REQ_HANDLER_H
#define BUILT_IN_ENTITY_LINK_REQ_HANDLER_H

#include <memory>
#include <vector>
#include "flow_func/flow_msg.h"
#include "fsm/state_define.h"
#include "hccl/base.h"
#include "llm_common/llm_common.h"

namespace FlowFunc {
constexpr size_t kUpdateLinkMaxCount = 16UL;

class LinkReqHandler {
public:
    enum class DoLinkState : int32_t {
        kLinkInit = 0,
        kLinkTryBind,
        kLinkTryConnect,
        kLinkTrySend,
        kLinkTryTest
    };

    struct ClusterAddrInfo {
        uint64_t local_cluster_id;
        uint64_t remote_cluster_id;
        HcclAddr local_hccl_addr;
        HcclAddr remote_hccl_addr;
    };

    struct DevLinkReqInfo {
        uint32_t operator_type;
        uint32_t cluster_num;
        uint64_t timeout;  // timeout interval: us
        std::vector<ClusterAddrInfo> cluster_infos;
    };

    static LinkReqHandler &GetInstance();

    virtual ~LinkReqHandler() = default;

    void SetUpdateLinkReqInfo(UpdateLinkReqInfo *link_req, int32_t *update_link_ret, int64_t device_index);

    FsmStatus HandleReq(uint64_t start_tick);

    LinkReqHandler(const LinkReqHandler &) = delete;

    LinkReqHandler(const LinkReqHandler &&) = delete;

    LinkReqHandler &operator=(const LinkReqHandler &) = delete;

    LinkReqHandler &operator=(const LinkReqHandler &&) = delete;

private:
    LinkReqHandler();

    FsmStatus NotifyPromptUnlink(const EntityPtr &entity, size_t cluster_index, ClusterAddrInfo &cluster);

    FsmStatus TransUpdateLinkInfo(const UpdateLinkReqInfo &req_info, DevLinkReqInfo &dev_req_info);

    FsmStatus TransClusterInfo(const ClusterInfo &cluster_info, LinkReqType req_type, uint32_t ip_num,
                               ClusterAddrInfo &cluster) const;

    void DoLink(size_t cluster_index, ClusterAddrInfo &cluster);

    void DoUnlink(size_t cluster_index, ClusterAddrInfo &cluster);

    void DoServerUnlink(size_t cluster_index, ClusterAddrInfo &cluster);

    void SetResult(LinkReqType link_req_type, size_t cluster_index, FsmStatus status);

    FsmStatus CleanEntityAndConn(const EntityPtr &entity, size_t cluster_index, uint64_t remote_cluster_id,
                                 bool is_server = false);

    static FsmStatus CloseConn(const HcclConn &conn);

    void SetOutput(size_t output_count, FsmStatus status);

    void ResetNotifyPromptFailed(size_t output_count);

    void CloseTimeoutConn(size_t cluster_num, LinkReqType req_type, const DevLinkReqInfo &dev_link_req_info);

    static std::string ToString(const ClusterAddrInfo &cluster);

    FsmStatus InitClientConn(size_t cluster_index, ClusterAddrInfo &cluster, const std::string &cluster_desc);

    FsmStatus TryConnectServer(size_t cluster_index, ClusterAddrInfo &cluster, const std::string &cluster_desc,
                               const HcclConn &hccl_conn);

    FsmStatus TryBindConn(size_t cluster_index, ClusterAddrInfo &cluster,
                          const std::string &cluster_desc, const HcclConn &hccl_conn);

    FsmStatus TrySendClusterInfo(size_t cluster_index, ClusterAddrInfo &cluster, const std::string &cluster_desc,
                                 const HcclConn &hccl_conn);

    FsmStatus TryTestSendClusterInfo(size_t cluster_index, ClusterAddrInfo &cluster, const std::string &cluster_desc,
                                   const HcclConn &hccl_conn);

    UpdateLinkReqInfo *update_link_req_info_ = nullptr;
    int32_t *update_link_ret_ = nullptr;
    int64_t device_index_ = 0L;
    uint32_t update_link_comp_count_ = 0U;
    std::vector<bool> update_link_comp_flags_;
    std::vector<DoLinkState> update_link_states_;
    std::vector<bool> send_unlink_req_comp_flags_;
    std::vector<HcclConn> update_link_conns_;
    std::vector<HcclRequest> update_link_requests_;
    std::vector<uint64_t> update_link_start_ticks_;
    std::vector<ClientClusterInfo> update_link_client_infos_;
};
}  // namespace FlowFunc
#endif  // BUILT_IN_ENTITY_LINK_REQ_HANDLER_H
