/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_ENTITY_LLM_COMM_ENTITY_H
#define BUILT_IN_ENTITY_LLM_COMM_ENTITY_H

#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include "ascend_hal.h"
#include "flow_msg.h"
#include "fsm/state_define.h"
#include "llm_common/llm_common.h"
#include "llm_common/statistic_manager.h"

namespace FlowFunc {
class LlmCommEntity {
public:
    // mbuf info for sync kv cache, only for server (prompt)
    struct SyncKvAddrInfo {
        Mbuf *sync_kv_req_mbuf;
        void *sync_kv_req_addr;
        Mbuf *sync_kv_resp_meta_mbuf;
        void *sync_kv_resp_meta_addr;
        int32_t req_info_count;
    };
    struct TransferKvAddrInfo {
        Mbuf *transfer_kv_req_mbuf;
        void *transfer_kv_req_addr;
        Mbuf *transfer_kv_resp_meta_mbuf;
        void *transfer_kv_resp_meta_addr;
        int32_t req_info_count;
    };
    // record info for send kv cache, only for server (prompt)
    struct SendKvRecordInfo {
        std::vector<KvTensor> kv_tensors;    // kv tensors addr
        bool send_kv_meta_succ_flag;         // success flag of call HcclRawIsend to send kv meta
        uint64_t send_kv_succ_count;         // success count of call HcclRawIsend to send kv tensor
        uint64_t send_complete_cnt;          // send complete count
    };
    struct RecvTransferKvRecordInfo {
        uint32_t recv_req_suc_flag;
        uint32_t call_send_meta_flag;
        uint32_t send_meta_complete_flag;
        uint32_t recv_call_count;
        uint32_t recv_suc_count;
        uint64_t recv_slot_index;
        int32_t last_probed_count;
    };
    struct ServerTickRecord {
        uint64_t probe_req_start_tick;
        uint64_t send_meta_start_tick;
        uint64_t send_kv_start_tick;
        uint64_t link_start_tick;
    };
    struct ClientTickRecord {
        uint64_t send_req_start_tick;
        uint64_t probe_resp_start_tick;
        uint64_t probe_kv_start_tick;
        uint64_t send_transfer_req_start_tick;
        uint64_t probe_transfer_resp_start_tick;
        uint64_t transfer_kv_start_tick;
    };
    struct RecvKvRecordInfo {
        uint64_t recv_complete_cnt;          // recv complete count
    };

    LlmCommEntity(EntityType type, HcclConn conn, HcclAddr &local_hccl_addr, HcclAddr &remote_hccl_addr);

    virtual ~LlmCommEntity();

    FsmStatus ProcessState();

    FsmStatus ChangeState(FsmState next_state);

    const std::string &GetDesc() const;

    static const std::string &GetStateDesc(FsmState state);

    HcclConn &GetConn();

    void SetConn(HcclConn conn);

    uint64_t GetRemoteIp() const;

    uint64_t GetLocalIp() const;

    EntityType GetType() const;

    uint64_t GetCurReqId() const;

    uint64_t GetCurPrefixId() const;

    uint64_t GetCurModelId() const;

    bool GetCurIsPullBlock() const;

    void SetCurReqId(uint64_t req_id);

    void SetCurPrefixId(uint64_t prefix_id);

    void SetCurModelId(uint64_t model_id);

    void SetCurIsPullBlock(bool is_pull_block);

    void SetCurIsPullWithOffset(bool is_pull_with_offset);
    bool GetCurIsPullWithOffset() const;

    uint64_t GetRemoteClusterId() const;

    void SetRemoteClusterId(uint64_t remote_cluster_id);

    const std::pair<int64_t, uint32_t> &GetCurCacheIdAndBatchIndex() const;

    void SetCurCacheIdAndBatchIndex(std::pair<int64_t, uint32_t> cache_id_and_batch_index);

    std::vector<HcclRequest> &GetReceiveRequests();

    std::vector<HcclRequest> &GetSendRequests();

    std::vector<HcclMessage> &GetProbeMsgs();

    std::once_flag &GetsendMetaOnceFlag();

    std::once_flag &GetReceiveReqOnceFlag();

    SyncKvAddrInfo &GetSyncKvAddrInfo();

    TransferKvAddrInfo &GetTransferKvAddrInfo();

    SendKvRecordInfo &GetSendKvRecordInfo();

    RecvKvRecordInfo &GetRecvKvRecordInfo();

    RecvTransferKvRecordInfo &GetRecvTransferKvRecordInfo();

    bool GetProbeLinkClusterInfoFlag() const;

    void SetProbeLinkClusterInfoFlag(bool link_empty_resp_flag);

    bool GetLinkEstablished() const;

    void SetLinkEstablished(bool link_established);

    EntityStatisticInfo &GetStatisticInfo();

    ServerTickRecord &GetServerTickRecord();

    void SetForcePrintFlag(bool force_print_flag);

    void Dump();

    void PrintStatisticInfo();

    void PrintPutStatisticInfo();

    FsmStatus SendAsync(void *buff, size_t buff_size);

    FsmStatus PullOrTransferCache(const TransferKvReqInfo &req_info,
                                  const PullKvReqInfo &pull_req_info,
                                  const CheckLinkReqInfo &check_req_info,
                                  const CacheEntry *cache_entry,
                                  const PullOrTransferExtraInfo &extra_info);

    FsmStatus AllocMbuf(Mbuf *&mbuf, uint64_t count, void *&data_buff) const;

    FsmState GetCurState() const;

    void ClearResource();

    std::atomic_bool &GetEntityOccupied();

    FsmStatus TestCompleteAsync(HcclRequest requests[], uint64_t request_count, int32_t &current_comp_count);

    FsmStatus MarkEntityDeletedByConn();

    ClientClusterInfo &GetClientClusterInfo();

    std::atomic_bool &GetReqIsUsing();

    std::atomic_bool &GetIsUnlinking();

    std::mutex &GetMutex();

    std::mutex &GetUnlinkMutex();

    uint8_t &GetCheckLinkRecvData();

    FsmStatus CheckLink(const CheckLinkReqInfo &req_info);

    std::pair<uint32_t, const TransferInfo *> GetTensorNumAndIndices() const;

    std::vector<uint64_t> &GetPushDstAddrs();

    LlmCommEntity(const LlmCommEntity &) = delete;

    LlmCommEntity(const LlmCommEntity &&) = delete;

    LlmCommEntity &operator=(const LlmCommEntity &) = delete;

    LlmCommEntity &operator=(const LlmCommEntity &&) = delete;

private:
    template<typename T>
    static void GeneratePromptSendBlockInfo(const T &req_info,
                                     std::vector<std::pair<uint64_t, uint64_t>> &prompt_send_block_info);

    template<typename T>
    void AggregateKvBlocks(const T &req_info);

    FsmStatus PullKvFromPrompt(const PullKvReqInfo &req_info,
                               bool enable_paged_attention,
                               const CacheEntry *cache_entry,
                               uint64_t start_tick);

    FsmStatus TransferCache(const TransferKvReqInfo &req_info,
                            bool enable_paged_attention,
                            const CacheEntry *cache_entry,
                            uint64_t start_tick);

    FsmStatus AllocMbufForTransferKvMeta();

    void GenerateTransferSlotInfos(const TransferKvReqInfo &req_info, TransferToRemoteReq *transfer_req);

    FsmStatus SendTransferReq(const TransferKvReqInfo &req_info);

    FsmStatus SendReqToPrompt(const PullKvReqInfo &req_info);

    FsmStatus ReceiveTransferKvMeta(const TransferKvReqInfo &req_info);

    FsmStatus TransferCacheData(const TransferKvReqInfo &req_info,
                           const CacheEntry *cache_entry);

    FsmStatus ReceiveKvCache(const PullKvReqInfo &req_info,
                             const SyncKvMetaInfo &resp_info,
                             const CacheEntry *cache_entry);

    FsmStatus ReceiveSyncKvMetaInfo(const PullKvReqInfo &req_info,
                                    SyncKvMetaInfo &resp_info,
                                    const CacheEntry *cache_entry);

    FsmStatus ProbeSync(uint64_t data_aize, uint64_t timeout, bool is_receive_meta = false);

    FsmStatus TrySendWithTimeout(void *buff, size_t buff_size, uint64_t timeout);

    FsmStatus TestSingleRequestSync(HcclRequest &request, uint64_t timeout) const;

    FsmStatus TestMultiRequestsSync(HcclRequest requests[], uint64_t request_count, uint64_t &total_comp_count,
                                    uint64_t timeout) const;

    FsmStatus AllocMbufForSyncKvMetaInfo();

    FsmStatus CheckSyncKvMetaInfo(const PullKvReqInfo &req_info,
                                  SyncKvMetaInfo &resp_info,
                                  const CacheEntry *cache_entry) const;
    std::vector<uintptr_t> GetRecvAddrs(const PullKvReqInfo &req_info,
                                        const SyncKvMetaInfo &resp_info,
                                        const CacheEntry *cache_entry);
    FsmStatus ReceiveKvCacheAsync(const PullKvReqInfo &req_info,
                                  const SyncKvMetaInfo &resp_info,
                                  const std::vector<uintptr_t> &recv_addrs,
                                  uint64_t timeout);

    FsmStatus RecvSingleKvTransfer(size_t index, uintptr_t kv_addr_of_one_layer, const PullKvReqInfo &req_info,
                                   const SyncKvMetaInfo &resp_info, HcclRequest &receive_request);

    void ResetNewTransfer(size_t &transfer_index);

    void GenerateBufferInfos(const PullKvReqInfo &req_info, SyncKvReqInfo *sync_req_info);

    static void CalBuffSize(TransferToRemoteReq *transfer_req_info, uint64_t slot_index, uint64_t &buff_size);

    FsmStatus TestUncompleteCheckReq(uint64_t timeout);

private:
    void GenerateTensorIndices(const PullKvReqInfo &req_info, SyncKvReqInfo *sync_req_info) const;
    void FillSyncKvReqInfo(const PullKvReqInfo &req_info, SyncKvReqInfo *sync_req_info) const;

    std::mutex unlink_mutex_;
    EntityType type_;
    HcclConn conn_;
    FsmState cur_state_;
    std::vector<HcclRequest> receive_requests_;
    std::vector<HcclRequest> send_requests_;
    std::vector<HcclMessage> probe_msgs_;
    std::pair<int64_t, uint32_t> cur_cache_id_and_batch_index_;
    uint64_t cur_req_id_;
    uint64_t cur_prefix_id_;
    uint64_t cur_model_id_;
    SyncKvAddrInfo sync_kv_addr_info_ = {};
    TransferKvAddrInfo transfer_kv_addr_info_ = {};
    SendKvRecordInfo send_kv_record_info_ = {};
    RecvKvRecordInfo recv_kv_record_info_ = {};
    RecvTransferKvRecordInfo recv_transfer_kv_record_info_ = {};
    HcclAddr local_hccl_addr_ = {};
    HcclAddr remote_hccl_addr_ = {};
    std::string desc_;
    ClientTickRecord client_tick_record_ = {};
    ServerTickRecord server_tick_record_ = {};
    EntityStatisticInfo stat_info_ = {};
    bool force_print_flag_ = false;
    // alloc memory once flag for receive resp
    std::once_flag receive_resp_once_flag_;
    // alloc memory once flag for send resp
    std::once_flag send_meta_once_flag_;
    uint8_t check_link_data_ = UINT8_MAX;
    uint8_t check_link_recv_data_ = UINT8_MAX;
    bool probe_link_cluster_info_succ_flag_ = false;
    bool link_established_ = false;
    // paged attention
    bool enable_paged_attention_ = false;
    bool cur_is_pull_block_;
    // mark multiple transfer of sender, pair<start_block_index, block_indexes>, e.g. <0, {1,2, 4,5, 6,7}>
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> transfer_indices_;
    // mark multiple transfer of sender, e.g. {2, 2, 2} correspond to above indices, just need one scatter transfer
    std::vector<std::vector<uint64_t>> transfer_continuous_nums_;
    std::atomic_bool entity_occupied_ = {false};
    std::atomic_bool req_is_using_ = {false};
    std::mutex mutex_;
    ClientClusterInfo client_cluster_info_ = {};
    uint64_t remote_cluster_id_ = 0U;
    uint64_t pull_or_transfer_start_tick_ = 0U;
    std::atomic_bool is_unlinking_ = {false};
    bool cur_is_pull_with_offset_ = false;
    std::vector<uint64_t> push_dst_addrs_;
};
}  // namespace FlowFunc
#endif  // BUILT_IN_ENTITY_LLM_COMM_ENTITY_H
