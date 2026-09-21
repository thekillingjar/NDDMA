/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "entity/llm_comm_entity.h"
#include <cmath>
#include <memory>
#include <thread>
#include "securec.h"
#include "common/scope_guard.h"
#include "entity/llm_comm_entity_mgr.h"
#include "fsm/state_define.h"
#include "fsm/state_manager.h"
#include "llm_common/kv_cache_manager.h"
#include "llm_common/cache_manager.h"
#include "llm_common/llm_common.h"
#include "llm_common/statistic_manager.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr size_t kSingleRequestCount  = 1UL;
constexpr size_t kDefaultMultiRequestCount  = 256UL;
constexpr int32_t kTestSomeSuccessStatus  = 0;
constexpr uint64_t kSyncKvTimeout  = 1000000UL;  // 1s
constexpr int32_t kProbeNoEnvelopeFlag  = 0;
constexpr uint64_t kCheckTimeoutLoopCount  = 10UL;
constexpr size_t kDefaultBlockCount  = 1UL;
constexpr const char *kInvalidSyncCallMsg  = ", probably caused by sync call of transfer and pull";
}  // namespace
LlmCommEntity::LlmCommEntity(EntityType type, HcclConn conn, HcclAddr &local_hccl_addr, HcclAddr &remote_hccl_addr)
    : type_(type),
      conn_(conn),
      cur_state_(FsmState::kFsmInitState),
      cur_req_id_(UINT64_MAX),
      cur_prefix_id_(UINT64_MAX),
      cur_model_id_(0LU),
      sync_kv_addr_info_({nullptr, nullptr, nullptr, nullptr, 0}),
      send_kv_record_info_({{}, false, 0UL, 0UL}),
      recv_transfer_kv_record_info_({0U, 0U, 0U, 0U, 0U, 0U, -1}),
      local_hccl_addr_(local_hccl_addr),
      remote_hccl_addr_(remote_hccl_addr),
      client_tick_record_({0UL, 0UL, 0UL, 0UL, 0UL, 0UL}),
      server_tick_record_({0UL, 0UL, 0UL, 0UL}),
      force_print_flag_(false),
      cur_is_pull_block_(false) {
    if (type_ == EntityType::kEntityServer) {
        send_requests_.reserve(kDefaultMultiRequestCount );
        receive_requests_.reserve(kSingleRequestCount );
        probe_msgs_.reserve(kSingleRequestCount );
    } else {
        send_requests_.reserve(kSingleRequestCount );
        receive_requests_.reserve(kDefaultMultiRequestCount );
        probe_msgs_.reserve(kDefaultMultiRequestCount );
    }
    desc_.append("[conn:")
        .append(std::to_string(reinterpret_cast<uintptr_t>(conn_)))
        .append(", type:")
        .append(std::to_string(static_cast<int32_t>(type_)))
        .append(", local ip:")
        .append(ToIp(local_hccl_addr_.info.tcp.ipv4Addr))
        .append("(")
        .append(std::to_string(local_hccl_addr_.info.tcp.ipv4Addr))
        .append(")_")
        .append(std::to_string(local_hccl_addr_.info.tcp.port))
        .append(", remote ip:")
        .append(ToIp(remote_hccl_addr_.info.tcp.ipv4Addr))
        .append("(")
        .append(std::to_string(remote_hccl_addr_.info.tcp.ipv4Addr))
        .append(")_")
        .append(std::to_string(remote_hccl_addr_.info.tcp.port))
        .append("]");
}

LlmCommEntity::~LlmCommEntity() {
    Dump();
    // clear record resource
    ClearResource();
    // sync_kv_resp_meta_mbuf malloc only once.
    if (sync_kv_addr_info_.sync_kv_resp_meta_mbuf != nullptr) {
        (void) halMbufFree(sync_kv_addr_info_.sync_kv_resp_meta_mbuf);
        sync_kv_addr_info_.sync_kv_resp_meta_mbuf = nullptr;
        sync_kv_addr_info_.sync_kv_resp_meta_addr = nullptr;
    }
    // transfer_kv_resp_meta_mbuf malloc only once.
    if (transfer_kv_addr_info_.transfer_kv_resp_meta_mbuf != nullptr) {
        (void) halMbufFree(transfer_kv_addr_info_.transfer_kv_resp_meta_mbuf);
        transfer_kv_addr_info_.transfer_kv_resp_meta_mbuf = nullptr;
        transfer_kv_addr_info_.transfer_kv_req_addr = nullptr;
    }
}

const std::string &LlmCommEntity::GetDesc() const {
    return desc_;
}

HcclConn &LlmCommEntity::GetConn() {
    return conn_;
}

uint64_t LlmCommEntity::GetRemoteIp() const {
    return remote_hccl_addr_.info.tcp.ipv4Addr;
}

uint64_t LlmCommEntity::GetLocalIp() const {
    return local_hccl_addr_.info.tcp.ipv4Addr;
}

EntityType LlmCommEntity::GetType() const {
    return type_;
}

FsmState LlmCommEntity::GetCurState() const {
    return cur_state_;
}

uint64_t LlmCommEntity::GetCurReqId() const {
    return cur_req_id_;
}

void LlmCommEntity::SetCurReqId(uint64_t req_id) {
    cur_req_id_ = req_id;
}

uint64_t LlmCommEntity::GetCurPrefixId() const {
    return cur_prefix_id_;
}

void LlmCommEntity::SetCurPrefixId(uint64_t prefix_id) {
    cur_prefix_id_ = prefix_id;
}

bool LlmCommEntity::GetCurIsPullBlock() const {
    return cur_is_pull_block_;
}

void LlmCommEntity::SetCurIsPullBlock(bool is_pull_block) {
    cur_is_pull_block_ = is_pull_block;
}

bool LlmCommEntity::GetCurIsPullWithOffset() const {
  return cur_is_pull_with_offset_;
}

void LlmCommEntity::SetCurIsPullWithOffset(bool is_pull_with_offset) {
  cur_is_pull_with_offset_ = is_pull_with_offset;
}


uint64_t LlmCommEntity::GetCurModelId() const {
    return cur_model_id_;
}

void LlmCommEntity::SetCurModelId(uint64_t model_id) {
    cur_model_id_ = model_id;
}

const std::string &LlmCommEntity::GetStateDesc(FsmState state) {
    return StateManager::GetInstance().GetStateDesc(state);
}

FsmStatus LlmCommEntity::ProcessState() {
    auto const k_state = StateManager::GetInstance().GetState(cur_state_);
    if (k_state == nullptr) {
        UDF_LOG_ERROR("Fail to get state:%s, entity:%s.", GetStateDesc(cur_state_).c_str(), desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    return k_state->Process(*this);
}

FsmStatus LlmCommEntity::ChangeState(FsmState next_state) {
    UDF_LOG_INFO("Begin to change state from state:%s to state:%s for entity:%s.", GetStateDesc(cur_state_).c_str(),
                  GetStateDesc(next_state).c_str(), desc_.c_str());
    cur_state_ = next_state;
    const auto k_state = StateManager::GetInstance().GetState(next_state);
    if (k_state == nullptr) {
        return FsmStatus::kFsmFailed;
    }
    return k_state->Preprocess(*this);
}

std::vector<HcclRequest> &LlmCommEntity::GetReceiveRequests() {
    return receive_requests_;
}

std::vector<HcclRequest> &LlmCommEntity::GetSendRequests() {
    return send_requests_;
}

std::vector<HcclMessage> &LlmCommEntity::GetProbeMsgs() {
    return probe_msgs_;
}

LlmCommEntity::SyncKvAddrInfo &LlmCommEntity::GetSyncKvAddrInfo() {
    return sync_kv_addr_info_;
}

LlmCommEntity::TransferKvAddrInfo &LlmCommEntity::GetTransferKvAddrInfo() {
    return transfer_kv_addr_info_;
}

LlmCommEntity::SendKvRecordInfo &LlmCommEntity::GetSendKvRecordInfo() {
    return send_kv_record_info_;
}

LlmCommEntity::RecvKvRecordInfo &LlmCommEntity::GetRecvKvRecordInfo() {
    return recv_kv_record_info_;
}

LlmCommEntity::RecvTransferKvRecordInfo &LlmCommEntity::GetRecvTransferKvRecordInfo() {
    return recv_transfer_kv_record_info_;
}

EntityStatisticInfo &LlmCommEntity::GetStatisticInfo() {
    return stat_info_;
}

LlmCommEntity::ServerTickRecord &LlmCommEntity::GetServerTickRecord() {
    return server_tick_record_;
}

void LlmCommEntity::SetForcePrintFlag(bool force_print_flag) {
    force_print_flag_ = force_print_flag;
}

std::once_flag &LlmCommEntity::GetsendMetaOnceFlag() {
    return send_meta_once_flag_;
}

void LlmCommEntity::Dump() {
    // statistic info timed print need in run log
    std::string entity_type = (type_ == EntityType::kEntityServer) ? "Prompt" : "Decoder";
    // sync kv statistic for server and client
    double sync_kv_avg_time_cost =
        CalcAverageTimeCost(stat_info_.sync_kv_total_times, stat_info_.sync_kv_total_tick_cost);
    double sync_kv_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.sync_kv_max_tick_cost);
    double sync_kv_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.sync_kv_min_tick_cost);
    // link/unlink
    double link_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.link_tick_cost);
    double unlink_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.unlink_tick_cost);
    if (type_ == EntityType::kEntityServer) {
        double send_transfer_req_avg_time_cost =
                CalcAverageTimeCost(stat_info_.send_transfer_req_total_times, stat_info_.send_transfer_req_total_tick_cost);
        double send_transfer_req_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.send_transfer_req_max_tick_cost);
        double send_transfer_req_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.send_transfer_req_min_tick_cost);
        double transfer_kv_avg_time_cost =
                CalcAverageTimeCost(stat_info_.transfer_kv_total_times, stat_info_.transfer_kv_total_tick_cost);
        double transfer_kv_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.transfer_kv_max_tick_cost);
        double transfer_kv_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.transfer_kv_min_tick_cost);

        // receive req statistic for server
        double recv_req_avg_time_cost =
            CalcAverageTimeCost(stat_info_.recv_req_total_times, stat_info_.recv_req_total_tick_cost);
        double recv_req_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_req_max_tick_cost);
        double recv_req_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_req_min_tick_cost);
        // send kv statistic for server
        double send_kv_avg_time_cost =
            CalcAverageTimeCost(stat_info_.send_kv_total_times, stat_info_.send_kv_total_tick_cost);
        double send_kv_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.send_kv_max_tick_cost);
        double send_kv_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.send_kv_min_tick_cost);
        UDF_RUN_LOG_INFO(
            "%s comm entity statistic info: desc=[%s], state=[%s], "
            "HcclRawImprobe=[succ:%lu, fail:%lu, total:%lu], "
            "HcclRawImrecv=[succ:%lu, fail:%lu, total:%lu], "
            "recv HcclTestSome=[succ:%lu], "
            "HcclRawIsend=[succ:%lu, fail:%lu, full:%lu, total:%lu], "
            "send HcclTestSome=[succ:%lu], "
            "send Layer-wise req=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "send Layer-wise kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "recv req=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "send kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "sync kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us].",
            entity_type.c_str(), desc_.c_str(), GetStateDesc(cur_state_).c_str(),
            stat_info_.call_probe_success_times, stat_info_.call_probe_fail_times, stat_info_.call_probe_total_times,
            stat_info_.call_recv_success_times, stat_info_.call_recv_fail_times, stat_info_.call_recv_total_times,
            stat_info_.test_recv_success_times,
            stat_info_.call_send_success_times, stat_info_.call_send_fail_times, stat_info_.call_send_full_times,
            stat_info_.call_send_total_times,
            stat_info_.test_send_success_times,
            stat_info_.send_transfer_req_total_times, send_transfer_req_max_time_cost, send_transfer_req_min_time_cost, send_transfer_req_avg_time_cost,
            stat_info_.transfer_kv_total_times, transfer_kv_max_time_cost, transfer_kv_min_time_cost, transfer_kv_avg_time_cost,
            stat_info_.recv_req_total_times, recv_req_max_time_cost, recv_req_min_time_cost, recv_req_avg_time_cost,
            stat_info_.send_kv_total_times, send_kv_max_time_cost, send_kv_min_time_cost, send_kv_avg_time_cost,
            stat_info_.sync_kv_total_times, sync_kv_max_time_cost, sync_kv_min_time_cost, sync_kv_avg_time_cost);
    } else {
        double recv_transfer_req_avg_time_cost =
                CalcAverageTimeCost(stat_info_.recv_transfer_req_total_times, stat_info_.recv_transfer_req_total_tick_cost);
        double recv_transfer_req_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_transfer_req_max_tick_cost);
        double recv_transfer_req_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_transfer_req_min_tick_cost);
        double recv_transfer_kv_avg_time_cost =
                CalcAverageTimeCost(stat_info_.recv_transfer_kv_total_times, stat_info_.recv_transfer_kv_total_tick_cost);
        double recv_transfer_kv_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_transfer_kv_max_tick_cost);
        double recv_transfer_kv_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_transfer_kv_min_tick_cost);

        // send req statistic for client
        double send_req_avg_time_cost =
            CalcAverageTimeCost(stat_info_.send_req_total_times, stat_info_.send_req_total_tick_cost);
        double send_req_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.send_req_max_tick_cost);
        double send_req_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.send_req_min_tick_cost);
        // receive kv statistic for client
        double recv_kv_avg_time_cost =
            CalcAverageTimeCost(stat_info_.recv_kv_total_times, stat_info_.recv_kv_total_tick_cost);
        double recv_kv_max_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_kv_max_tick_cost);
        double recv_kv_min_time_cost = StatisticManager::GetInstance().GetTimeCost(stat_info_.recv_kv_min_tick_cost);
        UDF_RUN_LOG_INFO(
            "%s comm entity statistic info: desc=[%s], "
            "HcclRawIsend=[succ:%lu, fail:%lu, full:%lu, total:%lu], "
            "send HcclTestSome=[succ:%lu], "
            "HcclRawImprobe=[succ:%lu, fail:%lu, total:%lu], "
            "HcclRawImrecv=[succ:%lu, fail:%lu, total:%lu], "
            "recv HcclTestSome=[succ:%lu], "
            "recv Layer-wise req=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "recv Layer-wise kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "send req=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "recv kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "sync kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "link time cost=[%.2f us], unlink time cost=[%2.f us].",
            entity_type.c_str(), desc_.c_str(),
            stat_info_.call_send_success_times,
            stat_info_.call_send_fail_times, stat_info_.call_send_full_times, stat_info_.call_send_total_times,
            stat_info_.test_send_success_times,
            stat_info_.call_probe_success_times, stat_info_.call_probe_fail_times, stat_info_.call_probe_total_times,
            stat_info_.call_recv_success_times, stat_info_.call_recv_fail_times, stat_info_.call_recv_total_times,
            stat_info_.test_recv_success_times,
            stat_info_.recv_transfer_req_total_times, recv_transfer_req_max_time_cost, recv_transfer_req_min_time_cost, recv_transfer_req_avg_time_cost,
            stat_info_.recv_transfer_kv_total_times, recv_transfer_kv_max_time_cost, recv_transfer_kv_min_time_cost, recv_transfer_kv_avg_time_cost,
            stat_info_.send_req_total_times, send_req_max_time_cost, send_req_min_time_cost, send_req_avg_time_cost,
            stat_info_.recv_kv_total_times, recv_kv_max_time_cost, recv_kv_min_time_cost, recv_kv_avg_time_cost,
            stat_info_.sync_kv_total_times, sync_kv_max_time_cost, sync_kv_min_time_cost, sync_kv_avg_time_cost,
            link_time_cost, unlink_time_cost);
    }
}

void LlmCommEntity::PrintStatisticInfo() {
    // ignore first n record
    if (stat_info_.sync_kv_total_times <= kIgnoreFirstRecordCount) {
        return;
    }
    if (force_print_flag_ || (stat_info_.sync_kv_total_times % kPrintInterval == 1UL)) {
        Dump();
        force_print_flag_ = false;
        if (stat_info_.sync_kv_total_times > kRecordMaxLimit) {
            stat_info_.Reset();
        }
    }
}

void LlmCommEntity::PrintPutStatisticInfo() {
    // ignore first n record
    if (stat_info_.transfer_kv_total_times <= kIgnoreFirstRecordCount) {
        return;
    }
    if (force_print_flag_ || (stat_info_.transfer_kv_total_times % kPrintInterval == 1UL)) {
        Dump();
        force_print_flag_ = false;
        if (stat_info_.transfer_kv_total_times > kRecordMaxLimit) {
            stat_info_.Reset();
        }
    }
}

FsmStatus LlmCommEntity::SendAsync(void *buff, size_t buff_size) {
    HcclRequest send_request;
    stat_info_.call_send_total_times++;
    HcclResult hccl_ret = HcclRawIsend(buff, static_cast<int32_t>(buff_size), HCCL_DATA_TYPE_INT8, conn_, &send_request);
    if (hccl_ret == HCCL_E_AGAIN) {
        stat_info_.call_send_full_times++;
        UDF_RUN_LOG_INFO("HcclRawIsend need try again, buff_size:%zu, conn:%p, entity:%s, ret:%d.",
                         buff_size, conn_, desc_.c_str(), hccl_ret);
        return FsmStatus::kFsmKeepState;
    }
    if (hccl_ret != HCCL_SUCCESS) {
        stat_info_.call_send_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawIsend, buff_size:%zu, conn:%p, entity:%s, ret:%d.",
                      buff_size, conn_, desc_.c_str(), hccl_ret);
        return FsmStatus::kFsmHcclFailed;
    }
    stat_info_.call_send_success_times++;
    send_requests_.emplace_back(send_request);
    return FsmStatus::kFsmSuccess;
}

template<typename T>
void LlmCommEntity::AggregateKvBlocks(const T &req_info) {
    transfer_indices_.clear();
    transfer_continuous_nums_.clear();
    if (!enable_paged_attention_) {
        return;
    }
    std::vector<std::pair<uint64_t, uint64_t>> prompt_send_block_info;
    GeneratePromptSendBlockInfo(req_info, prompt_send_block_info);
    size_t transfer_index = 0UL;
    uint64_t prompt_send_index = 0UL;
    uint64_t prompt_send_block_num = 0UL;
    uint64_t last_block_index = UINT64_MAX - 1UL;
    transfer_continuous_nums_.emplace_back();
    transfer_indices_.emplace_back();
    transfer_indices_[transfer_index].first = prompt_send_block_info[prompt_send_index].first;
    bool reset_new_transfer = false;
    for (uint32_t i = 0U; i < req_info.block_count; ++i) {
        if (reset_new_transfer) {
            ResetNewTransfer(transfer_index);
            transfer_indices_[transfer_index].first = prompt_send_block_info[prompt_send_index].first;
            reset_new_transfer = false;
        }
        uint64_t cur_block_index = req_info.block_indices[i];
        if ((cur_block_index != (last_block_index + 1U))) {
            // when one transfer slots is full, reset transfer.
            if (transfer_continuous_nums_[transfer_index].size() == kOneAggregatedTransferBlockNums) {
                ResetNewTransfer(transfer_index);
                transfer_indices_[transfer_index].first +=
                        (transfer_indices_[transfer_index].first == UINT64_MAX) ? 0U : prompt_send_block_num;
                reset_new_transfer = false;
            }
            // append new slot
            transfer_continuous_nums_[transfer_index].emplace_back(kDefaultBlockCount );
        } else {
            transfer_continuous_nums_[transfer_index].back()++;
        }
        transfer_indices_[transfer_index].second.emplace_back(cur_block_index);
        last_block_index = cur_block_index;
        prompt_send_block_num++;
        if (prompt_send_block_num == prompt_send_block_info[prompt_send_index].second) {
            // need new transfer cause sender limit.
            reset_new_transfer = true;
            prompt_send_index++;
            prompt_send_block_num = 0UL;
            last_block_index = UINT64_MAX - 1UL;
        }
    }
    UDF_LOG_INFO("PageAttention aggregated transfer times:%zu.", transfer_indices_.size());
}

void LlmCommEntity::ResetNewTransfer(size_t &transfer_index) {
    transfer_index++;
    transfer_continuous_nums_.emplace_back();
    transfer_indices_.emplace_back();
}

void LlmCommEntity::GenerateTransferSlotInfos(const TransferKvReqInfo &req_info, TransferToRemoteReq *transfer_req) {
    if (enable_paged_attention_) {
        size_t slot_index = 0U;
        for (size_t transfer_index = 0; transfer_index < transfer_req->send_nums_per_tensor; ++transfer_index) {
            size_t slot_nums = transfer_continuous_nums_[transfer_index].size();
            size_t block_index = 0UL;
            for (size_t i = 0; i < slot_nums; ++i) {
                auto start_block_index = transfer_indices_[transfer_index].second[block_index];
                UDF_LOG_INFO("Receiver slot start block_index:%lu.", start_block_index);
                transfer_req->slot_infos[slot_index].slot_offset = start_block_index * req_info.block_len;
                auto continuous_block_nums = transfer_continuous_nums_[transfer_index][i];
                transfer_req->slot_infos[slot_index].slot_size = continuous_block_nums * req_info.block_len;
                transfer_req->slot_infos[slot_index].slot_nums_per_transfer = slot_nums;
                block_index += continuous_block_nums;
                slot_index++;
            }
        }
    } else {
        uint64_t offset = 0U;
        for (size_t transfer_index = 0; transfer_index < transfer_req->send_nums_per_tensor; ++transfer_index) {
            uint64_t buff_size;
            if (transfer_index < (transfer_req->send_nums_per_tensor - 1U)) {
                buff_size = INT32_MAX;
            } else {
                buff_size = (req_info.block_len % INT32_MAX);
            }
            transfer_req->slot_infos[transfer_index].slot_nums_per_transfer = 1U;
            transfer_req->slot_infos[transfer_index].slot_offset = offset;
            transfer_req->slot_infos[transfer_index].slot_size = buff_size;
            offset += INT32_MAX;
        }
    }
}

FsmStatus LlmCommEntity::SendTransferReq(const TransferKvReqInfo &req_info) {
    client_tick_record_.send_transfer_req_start_tick = StatisticManager::GetInstance().GetCpuTick();
    AggregateKvBlocks(req_info);
    auto transfer_num = enable_paged_attention_ ? transfer_indices_.size() :
                       static_cast<size_t>(std::ceil(static_cast<double>(req_info.block_len) / INT32_MAX));
    size_t total_slot_num = 0U;
    if (enable_paged_attention_) {
        for (size_t transfer_index = 0; transfer_index < transfer_num; ++transfer_index) {
            total_slot_num += transfer_continuous_nums_[transfer_index].size();
        }
    } else {
        total_slot_num = transfer_num;
    }
    size_t req_info_size = sizeof(TransferToRemoteReq) + total_slot_num * sizeof(TransferSlotInfo);
    (void) AllocMbuf(transfer_kv_addr_info_.transfer_kv_req_mbuf, req_info_size, transfer_kv_addr_info_.transfer_kv_req_addr);
    if ((transfer_kv_addr_info_.transfer_kv_req_mbuf == nullptr) || (transfer_kv_addr_info_.transfer_kv_req_addr == nullptr)) {
        UDF_LOG_ERROR("Fail to alloc mbuf for transfer req info, dst_cluster_id:%lu, entity:%s.",
                      req_info.dst_cluster_id, desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    auto *transfer_req_info = static_cast<TransferToRemoteReq *>(transfer_kv_addr_info_.transfer_kv_req_addr);
    transfer_req_info->key_addr = req_info.dst_key_addr;
    transfer_req_info->value_addr = req_info.dst_value_addr;
    transfer_req_info->send_nums_per_tensor = transfer_num;
    transfer_req_info->total_slot_nums = total_slot_num;
    transfer_req_info->dst_cache_id = req_info.dst_cache_id;
    transfer_req_info->dst_batch_index = req_info.dst_batch_index;
    transfer_req_info->dst_layer_index = req_info.dst_layer_index;
    transfer_req_info->tensor_num_per_layer = req_info.tensor_num_per_layer;
    GenerateTransferSlotInfos(req_info, transfer_req_info);
    if (enable_paged_attention_) {
        auto max_block_index = req_info.block_indices[0];
        for (uint32_t i = 0; i < req_info.block_count; i++) {
            if (req_info.block_indices[i] > max_block_index) {
                max_block_index = i;
            }
        }
        transfer_req_info->max_size = (max_block_index + 1) * req_info.block_len;
    } else {
        transfer_req_info->max_size = req_info.block_len;
    }
    uint64_t timeout = req_info.timeout == 0UL ? kSyncKvTimeout  : req_info.timeout;
    FsmStatus ret = TrySendWithTimeout(transfer_kv_addr_info_.transfer_kv_req_addr, req_info_size, timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to async send sync kv request, dst_cluster_id:%lu, ret:%d, entity:%s.",
                      req_info.dst_cluster_id, static_cast<int32_t>(ret), desc_.c_str());
        return ret;
    }
    // test one request
    ret = TestSingleRequestSync(send_requests_.front(), timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    stat_info_.test_send_success_times++;
    bool max_tick_cost_flag = false;
    uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - client_tick_record_.send_transfer_req_start_tick;
    UpdateTickCost(tick_cost, stat_info_.send_transfer_req_total_times, stat_info_.send_transfer_req_min_tick_cost,
                   stat_info_.send_transfer_req_max_tick_cost, stat_info_.send_transfer_req_total_tick_cost, max_tick_cost_flag);
    double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        force_print_flag_ = true;
        UDF_RUN_LOG_INFO("Success to send transfer kv request with max time cost, dst_cluster_id:%lu, "
                         "time_cost:%.2f us, totalTimes:%lu, entity:%s.", req_info.dst_cluster_id, time_cost,
                         stat_info_.send_req_total_times, desc_.c_str());
    }
    UDF_LOG_INFO("Success to send transfer cache request, %s, time_cost:%.2f us, totalTimes:%lu, entity:%s.",
                 TransferReqDebugString(req_info).c_str(), time_cost, stat_info_.send_req_total_times, desc_.c_str());
    send_requests_.clear();
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::AllocMbufForTransferKvMeta() {
    std::call_once(receive_resp_once_flag_, [this]() {
        UDF_LOG_INFO("Alloc transfer meta info mem only once.");
        FsmStatus alloc_ret = AllocMbuf(transfer_kv_addr_info_.transfer_kv_resp_meta_mbuf, sizeof(TransferKvMetaInfo),
                                       transfer_kv_addr_info_.transfer_kv_resp_meta_addr);
        if (alloc_ret == FsmStatus::kFsmSuccess) {
            auto ret = memset_s(transfer_kv_addr_info_.transfer_kv_resp_meta_addr, sizeof(TransferKvMetaInfo), 0, sizeof(TransferKvMetaInfo));
            if (ret != EOK) {
                UDF_LOG_ERROR("Fail to reset mbuf for meta.");
                (void) halMbufFree(transfer_kv_addr_info_.transfer_kv_resp_meta_mbuf);
                transfer_kv_addr_info_.transfer_kv_resp_meta_mbuf = nullptr;
                transfer_kv_addr_info_.transfer_kv_resp_meta_addr = nullptr;
            }
        }
    });
    if (transfer_kv_addr_info_.transfer_kv_resp_meta_addr == nullptr) {
        UDF_LOG_ERROR("Fail to alloc mbuf, size:%zu, entity:%s.", sizeof(TransferKvMetaInfo), desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::ReceiveTransferKvMeta(const TransferKvReqInfo &req_info) {
    client_tick_record_.probe_transfer_resp_start_tick = StatisticManager::GetInstance().GetCpuTick();
    // probe envelope
    const uint64_t timeout = (req_info.timeout == 0UL) ? kSyncKvTimeout  : req_info.timeout;
    FsmStatus ret = ProbeSync(sizeof(TransferKvMetaInfo), timeout, true);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to probe transfer kv resp, ret:%d, dst_cluster_id:%lu, entity:%s",
                      static_cast<int32_t>(ret), req_info.dst_cluster_id, desc_.c_str());
        return ret;
    }
    // alloc mbuf only once
    ret = AllocMbufForTransferKvMeta();
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to alloc mbuf for sync kv meta info, ret:%d, dst_cluster_id:%lu, entity:%s",
                      static_cast<int32_t>(ret), req_info.dst_cluster_id, desc_.c_str());
        return ret;
    }
    // receive data
    HcclRequest receive_resp_request;
    stat_info_.call_recv_total_times++;
    auto hccl_ret = HcclRawImrecv(transfer_kv_addr_info_.transfer_kv_resp_meta_addr, static_cast<int32_t>(sizeof(TransferKvMetaInfo)),
                                 HCCL_DATA_TYPE_INT8, &probe_msgs_.front(), &receive_resp_request);
    if (hccl_ret != HCCL_SUCCESS) {
        stat_info_.call_recv_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawImrecv, ret:%d, data_buff:%p, count:%zu, dst_cluster_id:%lu, entity:%s.",
                      hccl_ret, transfer_kv_addr_info_.transfer_kv_resp_meta_addr, sizeof(TransferKvMetaInfo), req_info.dst_cluster_id,
                      desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    stat_info_.call_recv_success_times++;
    receive_requests_.emplace_back(receive_resp_request);  // no need emplace back to receive_requests_
    // test one request
    ret = TestSingleRequestSync(receive_requests_.front(), timeout);
    uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - client_tick_record_.probe_transfer_resp_start_tick;
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to receive resp, dst_cluster_id:%lu, time_cost:%.2f us, entity:%s, ret:%d.",
                      req_info.dst_cluster_id, StatisticManager::GetInstance().GetTimeCost(tick_cost),
                      desc_.c_str(), static_cast<int32_t>(ret));
        return ret;
    }
    stat_info_.test_recv_success_times++;
    // check result, no need clear receive_requests_ and probe_msgs_
    auto transfer_meta_info = static_cast<TransferKvMetaInfo *>(transfer_kv_addr_info_.transfer_kv_resp_meta_addr);
    receive_requests_.clear();  // need clear receive_requests_ and probe_msgs_ for sync kv tensors
    probe_msgs_.clear();
    return static_cast<FsmStatus>(transfer_meta_info->err_code);
}

void LlmCommEntity::CalBuffSize(TransferToRemoteReq *transfer_req_info, uint64_t slot_index, uint64_t &buff_size) {
    for (uint64_t i = 0U; i < transfer_req_info->slot_infos[slot_index].slot_nums_per_transfer; i++) {
        buff_size += transfer_req_info->slot_infos[slot_index + i].slot_size;
    }
}

FsmStatus LlmCommEntity::TransferCacheData(const TransferKvReqInfo &req_info, const CacheEntry *cache_entry) {
    client_tick_record_.transfer_kv_start_tick = StatisticManager::GetInstance().GetCpuTick();
    const uint64_t timeout = (req_info.timeout == 0UL) ? kSyncKvTimeout  : req_info.timeout;
    auto *transfer_req_info = static_cast<TransferToRemoteReq *>(transfer_kv_addr_info_.transfer_kv_req_addr);
    uint64_t send_complete_cnt = 0U;
    auto tensor_start_idx = req_info.src_layer_index * req_info.tensor_num_per_layer;
    for (uint64_t i = 0; i < req_info.tensor_num_per_layer; i++) {
        // validate array index before.
        auto &tensor = cache_entry->tensors[tensor_start_idx + i];
        uint64_t send_index = 0U;
        uint64_t slot_index = 0U;
        while (send_index < transfer_req_info->send_nums_per_tensor) {
            auto tensor_addr = reinterpret_cast<uint8_t *>(tensor->GetTensor()->GetData());
            if (!enable_paged_attention_) {
                tensor_addr += req_info.src_batch_index * cache_entry->batch_stride;
                tensor_addr += transfer_req_info->slot_infos[slot_index].slot_offset;
            } else {
                UDF_LOG_INFO("Start send, start block index:%lu", transfer_indices_[send_index].first);
                tensor_addr += transfer_indices_[send_index].first * req_info.block_len;
            }
            uint64_t buff_size = 0UL;
            CalBuffSize(transfer_req_info, slot_index, buff_size);
            UDF_LOG_INFO("Start send, buff size:%lu", buff_size);
            FsmStatus ret = SendAsync(tensor_addr, buff_size);
            if (ret == FsmStatus::kFsmKeepState) {
                int32_t current_comp_count = 0;
                ret = TestCompleteAsync(send_requests_.data(), send_requests_.size(), current_comp_count);
                if (ret != FsmStatus::kFsmSuccess) {
                    return ret;
                }
                stat_info_.test_send_success_times += current_comp_count;
                send_complete_cnt += current_comp_count;
                continue;
            }
            if (ret != FsmStatus::kFsmSuccess) {
                return ret;
            }
            send_index++;
            slot_index += transfer_req_info->slot_infos[slot_index].slot_nums_per_transfer;
            UDF_LOG_INFO("Send num:%lu", send_index);
        }
    }
    UDF_LOG_INFO("Already completed count:%lu", send_complete_cnt);
    auto ret = TestMultiRequestsSync(send_requests_.data(), send_requests_.size(), send_complete_cnt, timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - client_tick_record_.transfer_kv_start_tick;
    bool max_tick_cost_flag = false;
    UpdateTickCost(tick_cost, stat_info_.transfer_kv_total_times, stat_info_.transfer_kv_min_tick_cost,
                   stat_info_.transfer_kv_max_tick_cost, stat_info_.transfer_kv_total_tick_cost, max_tick_cost_flag);
    double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        force_print_flag_ = true;
        UDF_RUN_LOG_INFO("Success to send cache with max time cost, dst_cluster_id:%lu, time_cost:%.2f us, entity:%s.",
                         req_info.dst_cluster_id, time_cost, desc_.c_str());
    }
    UDF_LOG_INFO("Success to send cache with max time cost, dst_cluster_id:%lu, time_cost:%.2f us, entity:%s.",
                     req_info.dst_cluster_id, time_cost, desc_.c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::CheckLink(const CheckLinkReqInfo &req_info) {
    auto ret = TestUncompleteCheckReq(req_info.timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Test uncompleted request failed, ret:%d.", static_cast<int32_t>(ret));
        return ret;
    }
    send_requests_.clear();
    ret = TrySendWithTimeout(&check_link_data_, sizeof(uint8_t), req_info.timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to send check link data, ret:%d, remote_cluster_id:%lu, entity:%s.",
                      static_cast<int32_t>(ret), req_info.dst_cluster_id, desc_.c_str());
        return ret;
    }
    ret = TestSingleRequestSync(send_requests_.front(), req_info.timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to send check link data, ret:%d, remote_cluster_id:%lu, entity:%s.",
                     static_cast<int32_t>(ret), req_info.dst_cluster_id, desc_.c_str());
        return ret;
    }
    ClearResource();
    UDF_LOG_INFO("Check link success, remote_cluster_id:%lu.", req_info.dst_cluster_id);
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::TestUncompleteCheckReq(uint64_t timeout) {
    if (!send_requests_.empty()) {
        // test one request, do not free send_requests_
        return TestSingleRequestSync(send_requests_.front(), timeout);
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::PullOrTransferCache(const TransferKvReqInfo &req_info,
                                             const PullKvReqInfo &pull_req_info,
                                             const CheckLinkReqInfo &check_req_info,
                                             const CacheEntry *cache_entry,
                                             const PullOrTransferExtraInfo &extra_info) {
    uint64_t start_tick = extra_info.start_tick;
    pull_or_transfer_start_tick_ = start_tick;
    if (is_unlinking_.load(std::memory_order_relaxed) || !unlink_mutex_.try_lock()) {
        UDF_LOG_ERROR("Unlink is processing.");
        return FsmStatus::kFsmLinkBusy;
    }
    std::lock_guard<std::mutex> unlinkLock(unlink_mutex_, std::adopt_lock);
    req_is_using_.store(true, std::memory_order_relaxed);
    ScopeGuard guard([this] { req_is_using_.store(false, std::memory_order_relaxed); });
    while (true) {
        // current server link is processing pull cache req
        if (entity_occupied_.load()) {
            UDF_LOG_ERROR("The current link is occupied.");
            return FsmStatus::kFsmLinkBusy;
        }
        // need lock in case of probed by main loop thread
        if (!mutex_.try_lock()) {
            FsmStatus check_ret = CheckTimeout(start_tick, req_info.timeout);
            if (check_ret != FsmStatus::kFsmSuccess) {
                return FsmStatus::kFsmLinkBusy;
            }
            continue;
        }
        std::lock_guard<std::mutex> lock(mutex_, std::adopt_lock);
        if (extra_info.is_check) {
            return CheckLink(check_req_info);
        }
        return extra_info.is_pull
                   ? PullKvFromPrompt(pull_req_info, extra_info.enable_paged_attention, cache_entry, start_tick)
                   : TransferCache(req_info, extra_info.enable_paged_attention, cache_entry, start_tick);
    }
}

FsmStatus LlmCommEntity::TransferCache(const TransferKvReqInfo &req_info,
                                       bool enable_paged_attention,
                                       const CacheEntry *cache_entry,
                                       uint64_t start_tick) {
    UDF_LOG_INFO("Begin to transfer cache, %s, timeout:%lu us, block_len:%lu, entity:%s.",
                 TransferReqDebugString(req_info).c_str(), req_info.timeout, req_info.block_len, desc_.c_str());
    ScopeGuard guard([this] { ClearResource(); });
    const auto &instance = StatisticManager::GetInstance();
    enable_paged_attention_ = enable_paged_attention;
    FsmStatus ret = SendTransferReq(req_info);
    if (ret != FsmStatus::kFsmSuccess) {
        double time_cost = instance.GetTimeCost(instance.GetCpuTick() - start_tick);
        UDF_LOG_ERROR("Fail to send transfer cache request, ret:%d, dst_cluster_id:%lu, time_cost:%.2f us, entity:%s.",
                      static_cast<int32_t>(ret), req_info.dst_cluster_id, time_cost, desc_.c_str());
        return ret;
    }
    ret = ReceiveTransferKvMeta(req_info);
    if (ret != FsmStatus::kFsmSuccess) {
        double time_cost = instance.GetTimeCost(instance.GetCpuTick() - start_tick);
        UDF_LOG_ERROR("Transfer kv meta is invalid, ret:%d, dst_cluster_id:%lu, time_cost:%.2f us, entity:%s.",
                      static_cast<int32_t>(ret), req_info.dst_cluster_id, time_cost, desc_.c_str());
        return ret;
    }
    ret = TransferCacheData(req_info, cache_entry);
    if (ret != FsmStatus::kFsmSuccess) {
        double time_cost = instance.GetTimeCost(instance.GetCpuTick() - start_tick);
        UDF_LOG_ERROR("Fail to transfer cache, ret:%d, dst_cluster_id:%lu, time cost:%.2f us, entity:%s.",
                      static_cast<int32_t>(ret), req_info.dst_cluster_id, time_cost, desc_.c_str());
        return ret;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::PullKvFromPrompt(const PullKvReqInfo &req_info,
                                          bool enable_paged_attention,
                                          const CacheEntry *cache_entry,
                                          uint64_t start_tick) {
    UDF_LOG_INFO("Begin to pull kv, %s, timeout:%lu us, block_len:%lu, entity:%s.",
                 CacheKeyDebugString(req_info).c_str(), req_info.timeout, req_info.block_len, desc_.c_str());
    const auto &instance = StatisticManager::GetInstance();
    enable_paged_attention_ = enable_paged_attention;
    ScopeGuard guard([this] { ClearResource(); });
    client_tick_record_.send_req_start_tick = start_tick;
    FsmStatus ret = SendReqToPrompt(req_info);
    if (ret != FsmStatus::kFsmSuccess) {
        double time_cost = instance.GetTimeCost(instance.GetCpuTick() - start_tick);
        UDF_LOG_ERROR("Fail to sync kv from prompt, ret:%d, req_id:%lu, prompt_cluster_id:%lu, time_cost:%.2f us, entity:%s.",
                      static_cast<int32_t>(ret), req_info.req_id, req_info.prompt_cluster_id, time_cost, desc_.c_str());
        return ret;
    }
    SyncKvMetaInfo resp_info{};
    ret = ReceiveSyncKvMetaInfo(req_info, resp_info, cache_entry);
    if (ret != FsmStatus::kFsmSuccess) {
        double time_cost = instance.GetTimeCost(instance.GetCpuTick() - start_tick);
        UDF_LOG_ERROR("Fail to sync kv from prompt, ret:%d, req_id:%lu, prompt_cluster_id:%lu, time_cost:%.2f us, entity:%s.",
                      static_cast<int32_t>(ret), req_info.req_id, req_info.prompt_cluster_id, time_cost, desc_.c_str());
        return ret;
    }
    ret = ReceiveKvCache(req_info, resp_info, cache_entry);
    if (ret != FsmStatus::kFsmSuccess) {
        double time_cost = instance.GetTimeCost(instance.GetCpuTick() - start_tick);
        UDF_LOG_ERROR("Fail to sync kv from prompt, ret:%d, req_id:%lu, prompt_cluster_id:%lu, time cost:%.2f us, entity:%s.",
                      static_cast<int32_t>(ret), req_info.req_id, req_info.prompt_cluster_id, time_cost, desc_.c_str());
        return ret;
    }
    return FsmStatus::kFsmSuccess;
}

template<typename T>
void LlmCommEntity::GeneratePromptSendBlockInfo(const T &req_info,
                                                std::vector<std::pair<uint64_t, uint64_t>> &prompt_send_block_info) {
    // prompt can be not continuous, pair represent {start block index, continues num}
    uint64_t single_transfer_max_block_nums = (INT32_MAX / req_info.block_len);
    UDF_LOG_INFO("Single transfer max block num:%lu", single_transfer_max_block_nums);
    if (req_info.prompt_block_count > 0) {
        uint64_t base_index = req_info.block_count;
        uint64_t last_block_index = req_info.block_indices[base_index];
        // default has one transfer
        prompt_send_block_info.emplace_back(last_block_index, 1);
        UDF_LOG_INFO("Need new transfer, prompt start block index:%lu.", last_block_index);
        for (uint32_t i = 1; i < req_info.prompt_block_count; ++i) {
            uint64_t cur_block_index = req_info.block_indices[base_index + i];
            if ((cur_block_index != (last_block_index + 1U)) ||
                (prompt_send_block_info.back().second == single_transfer_max_block_nums)) {
                // prompt block not continue, need new send group
                prompt_send_block_info.emplace_back(cur_block_index, 1);
                UDF_LOG_INFO("Need new transfer, prompt start block index:%lu.", cur_block_index);
            } else {
                prompt_send_block_info.back().second++;
            }
            last_block_index = cur_block_index;
        }
    } else {
        prompt_send_block_info.emplace_back(UINT64_MAX, req_info.block_count);
    }
    UDF_LOG_INFO("Prompt send info nums:%zu", prompt_send_block_info.size());
}

void LlmCommEntity::GenerateBufferInfos(const PullKvReqInfo &req_info, SyncKvReqInfo *sync_req_info) {
    if (enable_paged_attention_) {
        for (size_t i = 0; i < sync_req_info->buffer_count_per_layer; ++i) {
            sync_req_info->transfer_infos[i].buffer_info.block_index = transfer_indices_[i].first;
            sync_req_info->transfer_infos[i].buffer_info.buffer_len = transfer_indices_[i].second.size() * req_info.block_len;
            UDF_LOG_INFO("Sync transfer:%zu, start block:%lu, block num:%lu", i, transfer_indices_[i].first,
                         transfer_indices_[i].second.size());
        }
    } else {
        for (size_t i = 0; i < sync_req_info->buffer_count_per_layer; ++i) {
            sync_req_info->transfer_infos[i].buffer_info.block_index = UINT64_MAX;
            if (i < (sync_req_info->buffer_count_per_layer - 1U)) {
                sync_req_info->transfer_infos[i].buffer_info.buffer_len = INT32_MAX;
            } else {
                sync_req_info->transfer_infos[i].buffer_info.buffer_len = (req_info.block_len % INT32_MAX);
            }
            UDF_LOG_INFO("Common mode, sync transfer buffer size:%lu",
                         sync_req_info->transfer_infos[i].buffer_info.buffer_len);
        }
    }
}

void LlmCommEntity::GenerateTensorIndices(const PullKvReqInfo &req_info, SyncKvReqInfo *sync_req_info) const {
    sync_req_info->tensor_index_count = req_info.src_tensor_index_count;
    auto tensor_indices = RemoteTensorIndices(req_info, enable_paged_attention_);
    auto transfer_infos = sync_req_info->transfer_infos + sync_req_info->buffer_count_per_layer;
    for (uint32_t i = 0U; i < req_info.src_tensor_index_count; ++i) {
        transfer_infos[i].tensor_index = tensor_indices[i];
    }
    UDF_LOG_INFO("RemoteTensorIndices = %s",
                 ToString(std::vector<uint64_t>(tensor_indices, tensor_indices + sync_req_info->tensor_index_count)).c_str());
}

void LlmCommEntity::FillSyncKvReqInfo(const PullKvReqInfo &req_info, SyncKvReqInfo *sync_req_info) const {
    sync_req_info->cache_id = req_info.prompt_cache_id;
    sync_req_info->batch_index = req_info.prompt_batch_index;
    sync_req_info->req_id = req_info.req_id;
    sync_req_info->prefix_id = req_info.prefix_id;
    sync_req_info->model_id = req_info.model_id;
    sync_req_info->is_pull_block = enable_paged_attention_;
    sync_req_info->offset = req_info.src_cache_offset;
    sync_req_info->is_pull_with_offset = req_info.is_pull_with_offset;
}

FsmStatus LlmCommEntity::SendReqToPrompt(const PullKvReqInfo &req_info) {
    cur_req_id_ = req_info.req_id;
    AggregateKvBlocks(req_info);
    auto transfer_num = enable_paged_attention_ ? transfer_indices_.size() :
                       static_cast<size_t>(std::ceil(static_cast<double >(req_info.block_len) / INT32_MAX));
    size_t req_info_size = sizeof(SyncKvReqInfo) + (transfer_num + req_info.src_tensor_index_count) * sizeof(TransferInfo);
    (void) AllocMbuf(sync_kv_addr_info_.sync_kv_req_mbuf, req_info_size, sync_kv_addr_info_.sync_kv_req_addr);
    if ((sync_kv_addr_info_.sync_kv_req_mbuf == nullptr) || (sync_kv_addr_info_.sync_kv_req_addr == nullptr)) {
        UDF_LOG_ERROR("Fail to alloc mbuf for sync req info, req_id:%lu, prompt_cluster_id:%lu, entity:%s.",
                      req_info.req_id, req_info.prompt_cluster_id, desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    auto *sync_req_info = static_cast<SyncKvReqInfo *>(sync_kv_addr_info_.sync_kv_req_addr);
    FillSyncKvReqInfo(req_info, sync_req_info);
    sync_req_info->buffer_count_per_layer = transfer_num;
    GenerateBufferInfos(req_info, sync_req_info);
    GenerateTensorIndices(req_info, sync_req_info);

    // send req info
    uint64_t timeout = (req_info.timeout == 0UL) ? kSyncKvTimeout  : req_info.timeout;
    FsmStatus ret = TrySendWithTimeout(sync_kv_addr_info_.sync_kv_req_addr, req_info_size, timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to async send sync kv request, req_id:%lu, prompt_cluster_id:%lu, ret:%d, entity:%s.",
                      req_info.req_id, req_info.prompt_cluster_id, static_cast<int32_t>(ret), desc_.c_str());
        return ret;
    }
    uint64_t send_complete_cnt = 0U;
    ret = TestMultiRequestsSync(send_requests_.data(), send_requests_.size(), send_complete_cnt, timeout);
    uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - client_tick_record_.send_req_start_tick;
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR(
            "Fail to send sync kv request, req_id:%lu, prompt_cluster_id:%lu, time_cost:%.2f us, ret:%d, entity:%s.",
            req_info.req_id, req_info.prompt_cluster_id, StatisticManager::GetInstance().GetTimeCost(tick_cost),
            static_cast<int32_t>(ret), desc_.c_str());
        return ret;
    }
    stat_info_.test_send_success_times++;
    bool max_tick_cost_flag = false;
    UpdateTickCost(tick_cost, stat_info_.send_req_total_times, stat_info_.send_req_min_tick_cost,
                   stat_info_.send_req_max_tick_cost, stat_info_.send_req_total_tick_cost, max_tick_cost_flag);
    send_requests_.clear();
    double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_tick_cost_flag) {
        force_print_flag_ = true;
        UDF_RUN_LOG_INFO("Success to send sync kv request to prompt with max time cost, req_id:%lu,"
            "prompt_cluster_id:%lu, time_cost:%.2f us, totalTimes:%lu, entity:%s.",
            req_info.req_id, req_info.prompt_cluster_id, time_cost, stat_info_.send_req_total_times, desc_.c_str());
    }
    UDF_LOG_INFO("Success to send sync kv request to prompt, %s, time_cost:%.2f us, totalTimes:%lu, entity:%s.",
                 CacheKeyDebugString(req_info).c_str(), time_cost, stat_info_.send_req_total_times, desc_.c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::TrySendWithTimeout(void *buff, size_t buff_size, uint64_t timeout) {
    uint64_t start_tick = pull_or_transfer_start_tick_;
    HcclRequest send_request;
    uint64_t send_suc_count = 0UL;
    const uint64_t sendTotalCount = 1UL;
    uint64_t loop_count = 0UL;
    while (send_suc_count < sendTotalCount) {
        loop_count++;
        if (loop_count % kCheckTimeoutLoopCount  == 0UL) {
            FsmStatus ret = CheckTimeout(start_tick, timeout);
            if (ret != FsmStatus::kFsmSuccess) {
                UDF_LOG_ERROR("Call send timeout, buff_size:%lu, entity:%s.", buff_size, desc_.c_str());
                return ret;
            }
        }
        stat_info_.call_send_total_times++;
        HcclResult ret = HcclRawIsend(buff, static_cast<int32_t>(buff_size), HCCL_DATA_TYPE_INT8,
                                      conn_, &send_request);
        if (ret == HCCL_E_AGAIN) {
            stat_info_.call_send_full_times++;
            UDF_RUN_LOG_INFO("Send need retry, data_buff:%p, count:%zu, conn:%p, ret:%d, entity:%s.",
                             buff, buff_size, conn_, ret, desc_.c_str());
            continue;
        }
        if (ret != HCCL_SUCCESS) {
            stat_info_.call_send_fail_times++;
            UDF_LOG_ERROR("Fail to call send, data_buff:%p, count:%zu, conn:%p, loop_count:%lu, ret:%d, entity:%s.",
                          buff, buff_size, conn_, loop_count, ret, desc_.c_str());
            return FsmStatus::kFsmFailed;
        }
        send_suc_count++;
    }
    stat_info_.call_send_success_times++;
    send_requests_.emplace_back(send_request);
    double time_cost =
        StatisticManager::GetInstance().GetTimeCost(StatisticManager::GetInstance().GetCpuTick() - start_tick);
    UDF_LOG_INFO("Success to send, count:%zu, conn:%p, loop_count:%lu, time_cost:%.2f us, entity:%s.",
                 buff_size, conn_, loop_count, time_cost, desc_.c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::ReceiveSyncKvMetaInfo(const PullKvReqInfo &req_info,
                                               SyncKvMetaInfo &resp_info,
                                               const CacheEntry *cache_entry) {
    UDF_LOG_INFO("Begin to receive sync kv meta from prompt, %s, entity:%s.",
                 CacheKeyDebugString(req_info).c_str(), desc_.c_str());
    client_tick_record_.probe_resp_start_tick = StatisticManager::GetInstance().GetCpuTick();
    // probe envelope
    const uint64_t timeout = (req_info.timeout == 0UL) ? kSyncKvTimeout  : req_info.timeout;
    FsmStatus ret = ProbeSync(sizeof(SyncKvMetaInfo), timeout, true);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to probe sync kv resp, ret:%d, req_id:%lu, prompt_cluster_id:%lu, entity:%s",
                      static_cast<int32_t>(ret), req_info.req_id, req_info.prompt_cluster_id, desc_.c_str());
        return ret;
    }
    // alloc mbuf only once
    ret = AllocMbufForSyncKvMetaInfo();
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to alloc mbuf for sync kv meta info, ret:%d, req_id:%lu, prompt_cluster_id:%lu, entity:%s",
                      static_cast<int32_t>(ret), req_info.req_id, req_info.prompt_cluster_id, desc_.c_str());
        return ret;
    }
    // receive data
    HcclRequest receive_resp_request;
    stat_info_.call_recv_total_times++;
    auto hccl_ret = HcclRawImrecv(sync_kv_addr_info_.sync_kv_resp_meta_addr, static_cast<int32_t>(sizeof(SyncKvMetaInfo)),
                                 HCCL_DATA_TYPE_INT8, &probe_msgs_.front(), &receive_resp_request);
    if (hccl_ret != HCCL_SUCCESS) {
        stat_info_.call_recv_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawImrecv, ret:%d, data_buff:%p, count:%zu, req_id:%lu, "
                      "prompt_cluster_id:%lu, entity:%s.", hccl_ret, sync_kv_addr_info_.sync_kv_resp_meta_addr,
                      sizeof(SyncKvMetaInfo), req_info.req_id, req_info.prompt_cluster_id, desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    stat_info_.call_recv_success_times++;
    receive_requests_.emplace_back(receive_resp_request);  // no need emplace back to receive_requests_
    // test one request
    ret = TestSingleRequestSync(receive_requests_.front(), timeout);
    uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - client_tick_record_.probe_resp_start_tick;
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR(
            "Fail to receive sync kv resp from prompt, req_id:%lu, prompt_cluster_id:%lu, "
            "time_cost:%.2f us, entity:%s, ret:%d.", req_info.req_id, req_info.prompt_cluster_id,
            StatisticManager::GetInstance().GetTimeCost(tick_cost), desc_.c_str(), static_cast<int32_t>(ret));
        return ret;
    }
    stat_info_.test_recv_success_times++;
    // check result, no need clear receive_requests_ and probe_msgs_
    auto sync_kv_ret_info = static_cast<SyncKvMetaInfo *>(sync_kv_addr_info_.sync_kv_resp_meta_addr);
    resp_info = *sync_kv_ret_info;
    ret = CheckSyncKvMetaInfo(req_info, resp_info, cache_entry);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Finish to receive sync kv resp from prompt, req_id:%lu, model_id:%lu, prompt_cluster_id:%lu, "
                      "transfer_count:%u, time_cost:%.2f us, entity:%s, ret:%d.",
                      req_info.req_id, req_info.model_id, req_info.prompt_cluster_id, resp_info.transfer_count,
                      StatisticManager::GetInstance().GetTimeCost(tick_cost), desc_.c_str(), static_cast<int32_t>(ret));
        return ret;
    }
    receive_requests_.clear();  // need clear receive_requests_ and probe_msgs_ for sync kv tensors
    probe_msgs_.clear();
    UDF_LOG_INFO(
        "Success to receive sync kv meta from prompt, %s, transfer_count:%u, time_cost:%lf, entity:%s.",
        CacheKeyDebugString(req_info).c_str(), resp_info.transfer_count,
        StatisticManager::GetInstance().GetTimeCost(tick_cost), desc_.c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::ReceiveKvCache(const PullKvReqInfo &req_info,
                                        const SyncKvMetaInfo &resp_info,
                                        const CacheEntry *cache_entry) {
    UDF_LOG_INFO("Begin to receive kv cache from prompt, %s, transfer_count:%u, entity:%s.",
                 CacheKeyDebugString(req_info).c_str(), resp_info.transfer_count, desc_.c_str());
    client_tick_record_.probe_kv_start_tick = StatisticManager::GetInstance().GetCpuTick();
    uint32_t transfer_count = resp_info.transfer_count;
    const uint64_t timeout = (req_info.timeout == 0UL) ? kSyncKvTimeout  : req_info.timeout;
    // receive req info
    this->GetRecvKvRecordInfo().recv_complete_cnt = 0UL;
    FsmStatus ret = ReceiveKvCacheAsync(req_info, resp_info, GetRecvAddrs(req_info, resp_info, cache_entry), timeout);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to receive kv cache, ret:%d, transfer_count:%u, req_id:%lu, prompt_cluster_id:%lu, entity:%s.",
                      static_cast<int32_t>(ret), transfer_count, req_info.req_id, req_info.prompt_cluster_id, desc_.c_str());
        return ret;
    }
    // test requests
    uint64_t &test_success_count = this->GetRecvKvRecordInfo().recv_complete_cnt;
    ret = TestMultiRequestsSync(receive_requests_.data(), static_cast<uint64_t>(transfer_count),
                                test_success_count, timeout);
    stat_info_.test_recv_success_times += test_success_count;
    uint64_t tick_cost = StatisticManager::GetInstance().GetCpuTick() - client_tick_record_.probe_kv_start_tick;
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to test multi requests, ret:%d, transfer_count:%u, req_id:%lu, prompt_cluster_id:%lu, "
                      "time cost:%.2f us, entity:%s", static_cast<int32_t>(ret), transfer_count, req_info.req_id,
                      req_info.prompt_cluster_id, StatisticManager::GetInstance().GetTimeCost(tick_cost), desc_.c_str());
        return ret;
    }
    UDF_LOG_INFO("enable_paged_attention_ = %d, cache_id = %ld",
                 static_cast<int32_t>(enable_paged_attention_), req_info.cache_id);
    if ((!enable_paged_attention_) && (req_info.cache_id <= 0)) {
        const auto &cache = KvCacheManager::GetInstance().GetDecoderKvCache(req_info.model_id);
        (void) KvCacheManager::GetInstance().SaveDecoderKvCache({req_info.req_id, req_info.model_id}, cache);
    }
    bool max_time_cost_flag = false;
    UpdateTickCost(tick_cost, stat_info_.recv_kv_total_times, stat_info_.recv_kv_min_tick_cost,
                   stat_info_.recv_kv_max_tick_cost, stat_info_.recv_kv_total_tick_cost, max_time_cost_flag);
    double time_cost = StatisticManager::GetInstance().GetTimeCost(tick_cost);
    if (max_time_cost_flag) {
        force_print_flag_ = true;
        UDF_RUN_LOG_INFO(
            "Success to receive kv cache from prompt with max time cost, req_id:%lu, "
            "prompt_cluster_id:%lu, transfer_count:%u, time_cost:%.2f us, totalTimes:%lu, entity:%s.",
            req_info.req_id, req_info.prompt_cluster_id, transfer_count, time_cost,
            stat_info_.recv_kv_total_times, desc_.c_str());
    }
    UDF_LOG_INFO("Success to receive kv cache from prompt, %s, "
        "transfer_count:%u, time_cost:%.2f us, totalTimes:%lu, entity:%s.",
        CacheKeyDebugString(req_info).c_str(), transfer_count, time_cost, stat_info_.recv_kv_total_times, desc_.c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::ReceiveKvCacheAsync(const PullKvReqInfo &req_info,
                                             const SyncKvMetaInfo &resp_info,
                                             const std::vector<uintptr_t> &recv_addrs,
                                             uint64_t timeout) {
    uint32_t transfer_count = resp_info.transfer_count;
    size_t index = 0UL;
    bool probed = false;
    if (recv_addrs.size() != transfer_count) {
        UDF_LOG_ERROR("kv size = %zu, transfer_count = %u", recv_addrs.size(), transfer_count);
        return FsmStatus::kFsmParamInvalid;
    }
    auto *sync_req_info = static_cast<SyncKvReqInfo *>(sync_kv_addr_info_.sync_kv_req_addr);
    while (index < static_cast<size_t>(transfer_count)) {
        uint64_t transfer_index = index % sync_req_info->buffer_count_per_layer;
        size_t buff_size = sync_req_info->transfer_infos[transfer_index].buffer_info.buffer_len;
        if (!probed) {
            FsmStatus ret = ProbeSync(buff_size, timeout);
            if (ret != FsmStatus::kFsmSuccess) {
                UDF_LOG_ERROR("Fail to probe, ret:%d, index:%zu, buff_size:%lu, req_id:%lu, prompt_cluster_id:%lu,"
                              " entity:%s.", static_cast<int32_t>(ret), index, buff_size, req_info.req_id,
                              req_info.prompt_cluster_id, desc_.c_str());
                return ret;
            }
            probed = true;
        }
        uintptr_t kv_addr_of_one_layer = recv_addrs[index];
        HcclRequest receive_request;
        FsmStatus ret = RecvSingleKvTransfer(index, kv_addr_of_one_layer, req_info, resp_info, receive_request);
        if (ret == FsmStatus::kFsmKeepState) {
            continue;
        } else if (ret != FsmStatus::kFsmSuccess) {
            return ret;
        }
        stat_info_.call_recv_success_times++;
        receive_requests_.emplace_back(receive_request);
        index++;
        probed = false;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::RecvSingleKvTransfer(size_t index, uintptr_t kv_addr_of_one_layer, const PullKvReqInfo &req_info,
                                              const SyncKvMetaInfo &resp_info, HcclRequest &receive_request) {
    auto *sync_req_info = static_cast<SyncKvReqInfo *>(sync_kv_addr_info_.sync_kv_req_addr);
    uint64_t transfer_index = index % sync_req_info->buffer_count_per_layer;
    uint64_t buff_size = sync_req_info->transfer_infos[transfer_index].buffer_info.buffer_len;
    HcclResult hccl_ret;
    if (!enable_paged_attention_) {
        hccl_ret = HcclRawImrecv(reinterpret_cast<void *>(kv_addr_of_one_layer + transfer_index * INT32_MAX),
                                static_cast<int32_t>(buff_size), HCCL_DATA_TYPE_INT8, &probe_msgs_.at(index),
                                &receive_request);
    } else {
        size_t cur_slot_size = transfer_continuous_nums_[transfer_index].size();
        std::vector<void *> buffs_vec(cur_slot_size);
        std::vector<int32_t> countsVec(cur_slot_size);
        size_t block_index = 0UL;
        for (size_t i = 0; i < cur_slot_size; ++i) {
            auto start_block_index = transfer_indices_[transfer_index].second[block_index];
            buffs_vec[i] = reinterpret_cast<void *>(kv_addr_of_one_layer + start_block_index * req_info.block_len);
            // single block size count make sure < INT32_MAX, static_cast is safe
            countsVec[i] = static_cast<int32_t>(transfer_continuous_nums_[transfer_index][i] * req_info.block_len);
            block_index += transfer_continuous_nums_[transfer_index][i];
        }
        const auto buffs = buffs_vec.data();
        const auto counts = countsVec.data();
        if (cur_slot_size == 1U) {
            hccl_ret = HcclRawImrecv(buffs[0], counts[0], HCCL_DATA_TYPE_INT8, &probe_msgs_.at(index), &receive_request);
        } else {
            auto buff_count = static_cast<int32_t>(transfer_continuous_nums_[transfer_index].size());
            hccl_ret = HcclRawImrecvScatter(buffs, counts, buff_count, HCCL_DATA_TYPE_INT8, &probe_msgs_.at(index),
                                           &receive_request);
        }
    }
    if (hccl_ret == HCCL_E_AGAIN) {
        UDF_RUN_LOG_INFO("Single kv transfer need try again, ret:%d.", hccl_ret);
        int32_t current_comp_count = 0;
        auto ret = TestCompleteAsync(receive_requests_.data(), receive_requests_.size(), current_comp_count);
        if (ret != FsmStatus::kFsmSuccess) {
            UDF_LOG_ERROR("Fail to test some, ret:%d, transfer_count:%u, buff_size:%lu, req_id:%lu.",
                          static_cast<int32_t>(ret), resp_info.transfer_count, buff_size, req_info.req_id);
            return ret;
        }
        this->GetRecvKvRecordInfo().recv_complete_cnt += current_comp_count;
        return FsmStatus::kFsmKeepState;
    }
    if (hccl_ret != HCCL_SUCCESS) {
        stat_info_.call_recv_fail_times++;
        UDF_LOG_ERROR("Fail to call HcclRawImrecv, ret:%d, index:%zu, transfer_count:%u, "
                      "buff_size:%lu, req_id:%lu, model_id:%lu, prompt_cluster_id:%lu, entity:%s.",
                      hccl_ret, index, resp_info.transfer_count, buff_size,
                      resp_info.req_id, req_info.model_id, req_info.prompt_cluster_id, desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    return FsmStatus::kFsmSuccess;
}

std::vector<uintptr_t> LlmCommEntity::GetRecvAddrs(const PullKvReqInfo &req_info,
                                                   const SyncKvMetaInfo &resp_info,
                                                   const CacheEntry *cache_entry) {
    std::vector<uintptr_t> recv_addrs;
    uint32_t transfer_count = resp_info.transfer_count;
    recv_addrs.reserve(transfer_count);
    auto *sync_req_info = static_cast<SyncKvReqInfo *>(sync_kv_addr_info_.sync_kv_req_addr);
    if (cache_entry != nullptr) {
        const auto tensor_indices = LocalTensorIndices(req_info, enable_paged_attention_);
        UDF_LOG_INFO("LocalTensorIndices = %s",
                     ToString(std::vector<uint64_t>(tensor_indices, tensor_indices + req_info.dst_tensor_index_count))
                         .c_str());
        if (enable_paged_attention_) {
            for (uint32_t index = 0U; index < transfer_count; ++index) {
                auto kv_index = index / sync_req_info->buffer_count_per_layer;
                kv_index = req_info.dst_tensor_index_count > 0 ? tensor_indices[kv_index] : kv_index;
                auto kv_blocks_addr = reinterpret_cast<uintptr_t>(cache_entry->tensors[kv_index]->GetTensor()->GetData());
                recv_addrs.emplace_back(kv_blocks_addr);
            }
        } else {
            for (uint32_t index = 0U; index < transfer_count; ++index) {
                auto kv_index = index / sync_req_info->buffer_count_per_layer;
                kv_index = req_info.dst_tensor_index_count > 0 ? tensor_indices[kv_index] : kv_index;
                auto addr = static_cast<uint8_t *>(cache_entry->tensors[kv_index]->GetTensor()->GetData()) +
                    req_info.batch_index * cache_entry->batch_stride + req_info.dst_cache_offset;
                recv_addrs.emplace_back(reinterpret_cast<uintptr_t>(addr));
            }
        }
        return recv_addrs;
    }
    if (enable_paged_attention_) {
        auto &kv_blocks_tensors = KvCacheManager::GetInstance().QueryKvBlocksTensors(req_info.model_id);
        for (uint32_t index = 0U; index < transfer_count; ++index) {
            auto kv_index = index / sync_req_info->buffer_count_per_layer;
            auto kv_blocks_addr = reinterpret_cast<uintptr_t>(kv_blocks_tensors[kv_index]->GetTensor()->GetData());
            recv_addrs.emplace_back(kv_blocks_addr);
        }
    } else {
        // get kv cache: no need check kvCache nullptr
        std::shared_ptr<FlowMsg> kv_cache = KvCacheManager::GetInstance().GetDecoderKvCache(req_info.model_id);
        const auto kv_cache_addr = reinterpret_cast<uintptr_t>(kv_cache->GetTensor()->GetData());
        for (uint32_t index = 0U; index < transfer_count; ++index) {
            auto kv_index = index / sync_req_info->buffer_count_per_layer;
            recv_addrs.emplace_back(kv_cache_addr + kv_index * req_info.block_len);
        }
    }
    return recv_addrs;
}

FsmStatus LlmCommEntity::ProbeSync(uint64_t data_aize, uint64_t timeout, bool is_receive_meta) {
    uint64_t start_tick = pull_or_transfer_start_tick_;
    bool probed = false;
    uint64_t loop_count = 0UL;
    while (!probed) {
        loop_count++;
        int32_t flag = kProbeNoEnvelopeFlag ;
        HcclMessage msg{};
        HcclStatus status{};
        stat_info_.call_probe_total_times++;
        HcclResult ret = HcclRawImprobe(conn_, &flag, &msg, &status);
        if (ret != HCCL_SUCCESS) {
            stat_info_.call_probe_fail_times++;
            UDF_LOG_ERROR("Fail to call HcclRawImprobe, ret:%d, entity:%s.", ret, desc_.c_str());
            return FsmStatus::kFsmFailed;
        }
        if (flag != kProbeNoEnvelopeFlag ) {
            stat_info_.call_probe_success_times++;
            probe_msgs_.emplace_back(msg);
            probed = true;
            // get count
            int32_t count = 0;
            ret = HcclRawGetCount(&status, HCCL_DATA_TYPE_INT8, &count);
            if (ret != HCCL_SUCCESS) {
                UDF_LOG_ERROR("Fail to call HcclRawGetCount, ret:%d, entity:%s.", ret, desc_.c_str());
                return FsmStatus::kFsmFailed;
            }
            if ((count == 0) || static_cast<uint64_t>(count) > data_aize) {
                UDF_LOG_ERROR("Invalid req size%s, count:%d, expected req len:%lu, entity:%s.",
                              is_receive_meta ? kInvalidSyncCallMsg  : "", count, data_aize, desc_.c_str());
                return FsmStatus::kFsmFailed;
            }
        }
        if (!probed && (loop_count % kCheckTimeoutLoopCount  == 0UL)) {
            FsmStatus check_ret = CheckTimeout(start_tick, timeout);
            if (check_ret != FsmStatus::kFsmSuccess) {
                UDF_LOG_ERROR("Probe timeout, entity:%s.", desc_.c_str());
                return check_ret;
            }
        }
    }
    UDF_LOG_DEBUG("Success to probe envelope, data_aize:%lu, entity:%s.", data_aize, desc_.c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::TestMultiRequestsSync(HcclRequest requests[], uint64_t request_count,
                                               uint64_t &total_comp_count, uint64_t timeout) const {
    std::vector<int32_t> &comp_indices = LlmCommEntityMgr::GetInstance().GetCompIndices(request_count);
    std::vector<HcclStatus> &comp_status = LlmCommEntityMgr::GetInstance().GetCompStatus(request_count);
    uint64_t start_tick = pull_or_transfer_start_tick_;
    uint64_t loop_count = 0UL;
    auto *current_test_requests = reinterpret_cast<HcclRequest *>(reinterpret_cast<uintptr_t>(requests));
    auto current_test_count = static_cast<int32_t>(request_count);
    while (total_comp_count < request_count) {
        loop_count++;
        int32_t current_comp_count = 0;
        HcclResult ret = HcclRawTestSome(current_test_count, current_test_requests, &current_comp_count,
                                         comp_indices.data(), comp_status.data());
        if (ret != HCCL_SUCCESS) {
            UDF_LOG_ERROR("Fail to call HcclRawTestSome, current_test_count:%d, total_compCount:%lu, request_count:%lu,"
                          " entity:%s.", current_test_count, total_comp_count, request_count, desc_.c_str());
            return FsmStatus::kFsmFailed;
        }
        for (size_t index = 0UL; index < static_cast<size_t>(current_comp_count); index++) {
            HcclStatus &status = comp_status[index];
            if (status.error != kTestSomeSuccessStatus ) {
                UDF_LOG_ERROR("Get invalid test status, status:%d, current_test_count:%d, total_compCount:%lu, "
                              "request_count:%lu, entity:%s.", status.error, current_test_count, total_comp_count,
                              request_count, desc_.c_str());
                return FsmStatus::kFsmFailed;
            }
        }
        total_comp_count += static_cast<uint64_t>(current_comp_count);
        if ((total_comp_count < request_count) && (loop_count % kCheckTimeoutLoopCount  == 0UL)) {
            FsmStatus check_ret = CheckTimeout(start_tick, timeout);
            if (check_ret != FsmStatus::kFsmSuccess) {
                UDF_LOG_ERROR("Test requests timeout, total_comp_count:%lu, request_count:%lu, entity:%s.",
                              total_comp_count, request_count, desc_.c_str());
                return check_ret;
            }
        }
    }
    UDF_LOG_INFO("Success to test requests, req count:%lu, comp count:%lu, entity:%s.", request_count, total_comp_count,
                 desc_.c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::TestCompleteAsync(HcclRequest requests[], uint64_t request_count, int32_t &current_comp_count) {
    std::vector<int32_t> &comp_indices = LlmCommEntityMgr::GetInstance().GetCompIndices(request_count);
    std::vector<HcclStatus> &comp_status = LlmCommEntityMgr::GetInstance().GetCompStatus(request_count);
    auto *current_test_requests = static_cast<HcclRequest *>(requests);
    auto current_test_count = static_cast<int32_t>(request_count);
    HcclResult ret = HcclRawTestSome(current_test_count, current_test_requests, &current_comp_count,
                                     comp_indices.data(), comp_status.data());
    if (ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call test some, req count:%d, comp count:%lu, entity:%s.",
                      current_test_count, current_comp_count, desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    for (size_t index = 0UL; index < static_cast<size_t>(current_comp_count); index++) {
        HcclStatus &status = comp_status[index];
        if (status.error != kTestSomeSuccessStatus ) {
            UDF_LOG_ERROR("Get invalid test status, status:%d, req count:%d, comp count:%lu, entity:%s.",
                          status.error, current_test_count, current_comp_count, request_count, desc_.c_str());
            return FsmStatus::kFsmFailed;
        }
    }
    UDF_LOG_INFO("Success to test some, req count:%lu, complete count:%d.", request_count, current_comp_count);
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::TestSingleRequestSync(HcclRequest &request, uint64_t timeout) const {
    uint64_t test_success_count = 0UL;
    return TestMultiRequestsSync(&request, kSingleRequestCount , test_success_count, timeout);
}

FsmStatus LlmCommEntity::AllocMbuf(Mbuf *&mbuf, uint64_t count, void *&data_buff) const {
    int32_t ret = halMbufAlloc(count, &mbuf);
    if ((ret != static_cast<int32_t>(DRV_ERROR_NONE)) || (mbuf == nullptr)) {
        UDF_LOG_ERROR("Fail to call halMbufAlloc, count:%lu, ret:%d, entity:%s.", count, ret, desc_.c_str());
        return FsmStatus::kFsmMbufFailed;
    }
    ScopeGuard guard([&mbuf]() { (void) halMbufFree(mbuf); });
    // set buff len
    ret = halMbufSetDataLen(mbuf, count);
    if (ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        UDF_LOG_ERROR("Fail to call halMbufSetDataLen, mbuf:%p, count:%lu, ret:%d, entity:%s.", mbuf, count, ret,
                      desc_.c_str());
        return FsmStatus::kFsmMbufFailed;
    }
    // get buff addr
    ret = halMbufGetBuffAddr(mbuf, &data_buff);
    if ((ret != static_cast<int32_t>(DRV_ERROR_NONE)) || (data_buff == nullptr)) {
        UDF_LOG_ERROR("Fail to call halMbufGetBuffAddr, mbuf:%p, count:%lu, ret:%d, entity:%s.", mbuf, count, ret,
                      desc_.c_str());
        return FsmStatus::kFsmMbufFailed;
    }
    guard.ReleaseGuard();
    UDF_LOG_INFO("Success to alloc mbuf, count:%lu, mbuf:%p, data_buff:%p.", count, mbuf, data_buff);
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::AllocMbufForSyncKvMetaInfo() {
    std::call_once(receive_resp_once_flag_, [this]() {
        UDF_LOG_INFO("Alloc meta info mem only once.");
        FsmStatus alloc_ret = AllocMbuf(sync_kv_addr_info_.sync_kv_resp_meta_mbuf, sizeof(SyncKvMetaInfo),
                                       sync_kv_addr_info_.sync_kv_resp_meta_addr);
        if (alloc_ret == FsmStatus::kFsmSuccess) {
            auto ret = memset_s(sync_kv_addr_info_.sync_kv_resp_meta_addr, sizeof(SyncKvMetaInfo), 0, sizeof(SyncKvMetaInfo));
            if (ret != EOK) {
                UDF_LOG_ERROR("Fail to reset mbuf for meta.");
                (void) halMbufFree(sync_kv_addr_info_.sync_kv_resp_meta_mbuf);
                sync_kv_addr_info_.sync_kv_resp_meta_mbuf = nullptr;
                sync_kv_addr_info_.sync_kv_resp_meta_addr = nullptr;
            }
        }
    });
    if (sync_kv_addr_info_.sync_kv_resp_meta_addr == nullptr) {
        UDF_LOG_ERROR("Fail to alloc mbuf, size:%zu, entity:%s.", sizeof(SyncKvMetaInfo), desc_.c_str());
        return FsmStatus::kFsmFailed;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus LlmCommEntity::CheckSyncKvMetaInfo(const PullKvReqInfo &req_info,
                                             SyncKvMetaInfo &resp_info, const CacheEntry *cache_entry) const {
    auto ret = static_cast<FsmStatus>(resp_info.err_code);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR(
            "Check sync kv meta info, param invalid, req_id:%lu, model_id:%lu, prompt_cluster_id:%lu, transfer_count:%u, "
            "entity:%s, ret:%d.", req_info.req_id, req_info.model_id, req_info.prompt_cluster_id, resp_info.transfer_count,
            desc_.c_str(), static_cast<int32_t>(ret));
        return ret;
    }
    if ((resp_info.req_id != req_info.req_id) || (resp_info.model_id != req_info.model_id) || (resp_info.transfer_count == 0U)) {
        UDF_LOG_ERROR(
            "Check sync kv meta info, param invalid, req req_id:%lu, resp req_id:%lu, req model_id:%lu, resp model_id:%lu, "
            "prompt_cluster_id:%lu, transfer_count:%u, block_len:%lu, entity:%s.",
            req_info.req_id, resp_info.req_id, req_info.model_id, resp_info.model_id, req_info.prompt_cluster_id,
            resp_info.transfer_count, req_info.block_len, desc_.c_str());
        return FsmStatus::kFsmParamInvalid;
    }
    auto *sync_req_info = static_cast<SyncKvReqInfo *>(sync_kv_addr_info_.sync_kv_req_addr);
    if (cache_entry != nullptr) {
        auto num_tensors = 
            req_info.dst_tensor_index_count != 0 ? req_info.dst_tensor_index_count : static_cast<uint32_t>(cache_entry->tensors.size());
        auto src_num_tensors = resp_info.transfer_count / sync_req_info->buffer_count_per_layer;
        if ((src_num_tensors != num_tensors) || (req_info.block_len > cache_entry->batch_stride)) {
            UDF_LOG_ERROR("Invalid param, src tensor num:%u should be equal to dst tensor num:%u, "
                "block_len:%lu vs tensorSize:%zu, req_id:%lu, model_id:%lu, prompt_cluster_id:%lu, tensor_index_count:%u, entity:%s.",
                src_num_tensors, num_tensors, req_info.block_len, cache_entry->batch_stride,
                req_info.req_id, req_info.model_id, req_info.prompt_cluster_id, req_info.dst_tensor_index_count, desc_.c_str());
            return FsmStatus::kFsmParamInvalid;
        }
        return FsmStatus::kFsmSuccess;
    }
    if (!enable_paged_attention_) {
        std::shared_ptr<FlowMsg> kv_cache = KvCacheManager::GetInstance().GetDecoderKvCache(req_info.model_id);
        const uint64_t kv_cache_size = kv_cache->GetTensor()->GetDataSize();
        if (resp_info.transfer_count * req_info.block_len > kv_cache_size) {
            UDF_LOG_ERROR("Invalid param, req_id:%lu, transfer_count:%u, block_len:%lu, kv_cache_size:%lu, "
                          "model_id:%lu, entity:%s.", req_info.req_id, resp_info.transfer_count, req_info.block_len,
                          kv_cache_size, req_info.model_id, desc_.c_str());
            return FsmStatus::kFsmParamInvalid;
        }
        return FsmStatus::kFsmSuccess;
    }
    size_t send_count_per_layer = sync_req_info->buffer_count_per_layer;
    const auto &kv_block_tensors = KvCacheManager::GetInstance().QueryKvBlocksTensors(req_info.model_id);
    if (resp_info.transfer_count != (kv_block_tensors.size() * send_count_per_layer)) {
        UDF_LOG_ERROR("Invalid param, req_id:%lu, transfer_count:%u, send_count_per_layer:%u, kvSize:%zu, entity:%s.",
                      req_info.req_id, resp_info.transfer_count, send_count_per_layer, kv_block_tensors.size(), desc_.c_str());
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

void LlmCommEntity::ClearResource() {
    entity_occupied_.store(false);
    probe_msgs_.clear();
    send_requests_.clear();
    receive_requests_.clear();
    cur_req_id_ = UINT64_MAX;
    client_tick_record_ = {0UL, 0UL, 0UL, 0UL, 0UL, 0UL};
    server_tick_record_ = {0UL, 0UL, 0UL, 0UL};
    send_kv_record_info_ = {{}, false, 0UL, 0UL};
    recv_transfer_kv_record_info_ = {0U, 0U, 0U, 0U, 0U, 0U, -1};
    push_dst_addrs_.clear();
    // release mbuf
    if (sync_kv_addr_info_.sync_kv_req_mbuf != nullptr) {
        (void) halMbufFree(sync_kv_addr_info_.sync_kv_req_mbuf);
        sync_kv_addr_info_.sync_kv_req_mbuf = nullptr;
        sync_kv_addr_info_.sync_kv_req_addr = nullptr;
    }
    if (transfer_kv_addr_info_.transfer_kv_req_mbuf != nullptr) {
        (void) halMbufFree(transfer_kv_addr_info_.transfer_kv_req_mbuf);
        transfer_kv_addr_info_.transfer_kv_req_mbuf = nullptr;
        transfer_kv_addr_info_.transfer_kv_req_addr = nullptr;
    }
}

bool LlmCommEntity::GetProbeLinkClusterInfoFlag() const {
    return probe_link_cluster_info_succ_flag_;
}

void LlmCommEntity::SetProbeLinkClusterInfoFlag(bool link_empty_resp_flag) {
    probe_link_cluster_info_succ_flag_ = link_empty_resp_flag;
}

bool LlmCommEntity::GetLinkEstablished() const {
    return link_established_;
}

void LlmCommEntity::SetLinkEstablished(bool link_established) {
    link_established_ = link_established;
}

const std::pair<int64_t, uint32_t> &LlmCommEntity::GetCurCacheIdAndBatchIndex() const {
    return cur_cache_id_and_batch_index_;
}

void LlmCommEntity::SetCurCacheIdAndBatchIndex(std::pair<int64_t, uint32_t> cache_id_and_batch_index) {
    cur_cache_id_and_batch_index_ = std::move(cache_id_and_batch_index);
}

std::atomic_bool &LlmCommEntity::GetEntityOccupied() {
    return entity_occupied_;
}

FsmStatus LlmCommEntity::MarkEntityDeletedByConn() {
    UDF_LOG_INFO("Mark entity deleted:%s.", desc_.c_str());
    (void) ChangeState(FsmState::kFsmDestroyState);
    auto ret = HcclRawForceClose(GetConn()) == HCCL_SUCCESS ? FsmStatus::kFsmSuccess : FsmStatus::kFsmUnlinkFailed;
    return ret;
}

ClientClusterInfo &LlmCommEntity::GetClientClusterInfo() {
    return client_cluster_info_;
}

std::atomic_bool &LlmCommEntity::GetReqIsUsing() {
    return req_is_using_;
}

std::atomic_bool &LlmCommEntity::GetIsUnlinking() {
    return is_unlinking_;
}

std::mutex &LlmCommEntity::GetMutex() {
    return mutex_;
}

std::mutex &LlmCommEntity::GetUnlinkMutex() {
    return unlink_mutex_;
}

uint64_t LlmCommEntity::GetRemoteClusterId() const {
    return remote_cluster_id_;
}

void LlmCommEntity::SetRemoteClusterId(uint64_t remote_cluster_id) {
    remote_cluster_id_ = remote_cluster_id;
}

uint8_t &LlmCommEntity::GetCheckLinkRecvData() {
    return check_link_recv_data_;
}

void LlmCommEntity::SetConn(HcclConn conn) {
    conn_ = conn;
}

std::pair<uint32_t, const TransferInfo *> LlmCommEntity::GetTensorNumAndIndices() const {
    auto *sync_req_info = static_cast<SyncKvReqInfo *>(sync_kv_addr_info_.sync_kv_req_addr);
    return std::make_pair(sync_req_info->tensor_index_count,
                          sync_req_info->transfer_infos + sync_req_info->buffer_count_per_layer);
}

std::vector<uint64_t> &LlmCommEntity::GetPushDstAddrs() {
    return push_dst_addrs_;
}
}  // namespace FlowFunc
