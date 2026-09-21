/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/send_state.h"
#include <vector>
#include "entity/llm_comm_entity_mgr.h"
#include "fsm/state_manager.h"
#include "llm_common/kv_cache_manager.h"
#include "llm_common/cache_manager.h"
#include "llm_common/llm_common.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kMinBlockCount = 1UL;

void GetKvTensors(CacheEntry &cache_entry, uint64_t offset, size_t data_size, std::vector<KvTensor> &kv_tensors) {
    kv_tensors.reserve(cache_entry.tensors.size());
    for (const auto &kv_tensor_buffer: cache_entry.tensors) {
        KvTensor kv_tensor;
        kv_tensor.tensor_buffer = kv_tensor_buffer;
        kv_tensor.data_addr = static_cast<uint8_t *>(kv_tensor_buffer->GetTensor()->GetData()) + offset;
        kv_tensor.data_size = data_size;
        kv_tensor.block_len = data_size;
        kv_tensors.emplace_back(std::move(kv_tensor));
    }
}
}  // namespace

FsmStatus SendState::Preprocess(LlmCommEntity &entity) {
    UDF_LOG_DEBUG("Enter send state, entity:%s.", entity.GetDesc().c_str());
    std::vector<std::shared_ptr<FlowMsg>> kv_tensors;
    FsmStatus ret = GenerateSyncKvMetaInfo(entity);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    return Process(entity);
}

FsmStatus SendState::Process(LlmCommEntity &entity) {
    FsmStatus ret = SendSyncKvMetaAsync(entity);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    ret = SendKvCacheAsync(entity);
    if (ret != FsmStatus::kFsmSuccess) {
        if (ret == FsmStatus::kFsmKeepState) {
            // EAGAIN no comm resource available, need test send first to free resource.
            (void) TestSendRequestsAsync(entity);
        }
        return ret;
    }
    ret = TestSendRequestsAsync(entity);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    return Postprocess(entity);
}

FsmStatus SendState::Postprocess(LlmCommEntity &entity) {
    EntityStatisticInfo &stat_info = entity.GetStatisticInfo();
    const uint64_t current_tick_cost =
        StatisticManager::GetInstance().GetCpuTick() - entity.GetServerTickRecord().probe_req_start_tick;
    bool max_tick_cost_flag = false;
    UpdateTickCost(current_tick_cost, stat_info.sync_kv_total_times, stat_info.sync_kv_min_tick_cost,
                   stat_info.sync_kv_max_tick_cost, stat_info.sync_kv_total_tick_cost, max_tick_cost_flag);
    const double k_time_cost = StatisticManager::GetInstance().GetTimeCost(current_tick_cost);
    if (max_tick_cost_flag) {
        entity.SetForcePrintFlag(true);
    }
    UDF_LOG_INFO("Finish to send kv cache%s, cache_id:%ld, req_id:%lu, model_id:%lu, "
                 "time cost:%.2f us, total_times:%lu, entity:%s.",
                 max_tick_cost_flag ? kMaxTimeCost : kCommonTimeCost,
                 entity.GetCurCacheIdAndBatchIndex().first,
                 entity.GetCurReqId(),
                 entity.GetCurModelId(),
                 k_time_cost,
                 stat_info.sync_kv_total_times,
                 entity.GetDesc().c_str());
    // remove kv cache
    ReleaseKvCacheForPrompt(entity);
    // clear resource
    entity.GetSendRequests().clear();
    entity.SetCurReqId(UINT64_MAX);
    UDF_LOG_DEBUG("Finish send state, entity:%s.", entity.GetDesc().c_str());
    return entity.ChangeState(FsmState::kFsmIdleState);
}

FsmStatus SendState::GenerateSyncKvMetaInfo(LlmCommEntity &entity) {
    LlmCommEntity::SyncKvAddrInfo &addr_info = entity.GetSyncKvAddrInfo();
    auto *req_info = static_cast<SyncKvReqInfo *>(addr_info.sync_kv_req_addr);
    uint64_t buffer_info_size = static_cast<uint64_t>(addr_info.req_info_count) - sizeof(SyncKvReqInfo);
    auto expect_count = req_info->buffer_count_per_layer + req_info->tensor_index_count;
    if (expect_count > (buffer_info_size / sizeof(SyncBufferInfo))) {
        UDF_RUN_LOG_INFO(
            "Invalid req size, expect_count:%u, real count:%lu, entity:%s.",
            expect_count, buffer_info_size / sizeof(SyncBufferInfo),
            entity.GetDesc().c_str());
        return FsmStatus::kFsmParamInvalid;
    }
    if ((req_info->buffer_count_per_layer < kMinBlockCount)) {
        UDF_LOG_ERROR("Invalid param, buffer_count_per_layer:%u, req_id:%lu, entity:%s.",
                      req_info->buffer_count_per_layer, req_info->req_id, entity.GetDesc().c_str());
        return FsmStatus::kFsmParamInvalid;
    }
    // alloc sync kv resp addr, only once
    std::call_once(entity.GetsendMetaOnceFlag(), [&entity, &addr_info]() {
        (void) entity.AllocMbuf(addr_info.sync_kv_resp_meta_mbuf,
                                sizeof(SyncKvMetaInfo),
                                addr_info.sync_kv_resp_meta_addr);
    });
    if ((addr_info.sync_kv_resp_meta_mbuf == nullptr) || (addr_info.sync_kv_resp_meta_addr == nullptr)) {
        UDF_LOG_ERROR("Fail to alloc mbuf for send kv req info, entity:%s.", entity.GetDesc().c_str());
        return FsmStatus::kFsmFailed;
    }
    // query kv cache
    uint64_t req_id = entity.GetCurReqId();
    uint64_t model_id = entity.GetCurModelId();
    auto *resp_info = static_cast<SyncKvMetaInfo *>(addr_info.sync_kv_resp_meta_addr);
    resp_info->req_id = req_id;
    resp_info->model_id = model_id;
    std::vector<KvTensor> kv_tensors;
    FsmStatus ret = QueryPromptKvCache(req_info, kv_tensors);
    if (ret == FsmStatus::kFsmSuccess) {
        ret = CheckSyncKvReqInfo(req_info, kv_tensors, entity);
    }
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("Fail to check sync kv req, req_id:%lu, entity:%s.", req_id, entity.GetDesc().c_str());
        resp_info->err_code = static_cast<int32_t>(ret);
        resp_info->transfer_count = 0U;
    } else {
        resp_info->err_code = static_cast<int32_t>(FsmStatus::kFsmSuccess);
        resp_info->transfer_count = static_cast<uint32_t>(kv_tensors.size()) * req_info->buffer_count_per_layer;
    }
    LlmCommEntity::SendKvRecordInfo &record_info = entity.GetSendKvRecordInfo();
    record_info = {kv_tensors, false, 0UL, 0UL};
    UDF_LOG_INFO("Success to generate send kv meta info, req_id:%lu, model_id:%lu, err_code:%d, transfer_count:%u, "
                 "entity:%s.", req_id, model_id, resp_info->err_code, resp_info->transfer_count, entity.GetDesc().c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::FilterByTensorIndices(const SyncKvReqInfo &req_info, std::vector<KvTensor> &kv_tensors) {
    std::vector<KvTensor> filtered;
    filtered.reserve(req_info.tensor_index_count);
    const auto tensor_indices = req_info.transfer_infos + req_info.buffer_count_per_layer;
    for (uint32_t i = 0; i < req_info.tensor_index_count; ++i) {
        const auto tensor_index = tensor_indices[i].tensor_index;
        if (tensor_index >= kv_tensors.size()) {
            UDF_LOG_ERROR("srcTensorIndex[%u] = %lu, out of range, tensorNum = %lu", i, tensor_index, kv_tensors.size());
            return FsmStatus::kFsmParamInvalid;
        }
        filtered.emplace_back(kv_tensors[tensor_index]);
        UDF_LOG_DEBUG("KvTensor added for PP, tensor index = %lu", tensor_indices[i].tensor_index);
    }
    kv_tensors = std::move(filtered);
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::QueryPromptKvCache(const SyncKvReqInfo *req_info, std::vector<KvTensor> &kv_tensors) {
    if (CacheManager::GetInstance().IsEnabled()) {
        auto ret = (req_info->is_pull_block == 1U) ? QueryBlocksKvCache(req_info, kv_tensors) :
                   QueryKvCacheInCacheManager(req_info, kv_tensors);
        if ((ret == FsmStatus::kFsmSuccess) && (req_info->tensor_index_count > 0)) {
           ret = FilterByTensorIndices(*req_info, kv_tensors);
        }
        return ret;
    }
    kv_tensors = KvCacheManager::GetInstance().QueryPromptKvCache({req_info->req_id, req_info->model_id});
    if (!kv_tensors.empty()) {
        if (CheckKvCacheManagerReq(kv_tensors, req_info) != FsmStatus::kFsmSuccess) {
            return FsmStatus::kFsmParamInvalid;
        }
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::CheckKvCacheManagerReq(const std::vector<KvTensor> &kv_tensors, const SyncKvReqInfo *req_info) {
    uint64_t real_kv_len = kv_tensors.front().data_size;
    uint64_t req_kv_len = 0UL;
    for (uint32_t i = 0U; i < req_info->buffer_count_per_layer; ++i) {
        if (req_info->transfer_infos[i].buffer_info.block_index != UINT64_MAX) {
            UDF_LOG_ERROR("Invalid buffer info, i:%u, block_index:%lu", i, req_info->transfer_infos[i].buffer_info.block_index);
            return FsmStatus::kFsmParamInvalid;
        }
        if ((req_info->buffer_count_per_layer > 1) && (i < req_info->buffer_count_per_layer - 1U)) {
            req_kv_len += req_info->transfer_infos[i].buffer_info.buffer_len;
        }
    }
    if (req_kv_len >= real_kv_len) {
        UDF_LOG_ERROR("Invalid param, req_kv_len:%lu, real_kv_len:%lu, req_id:%lu, prefix_id:%lu, model_id:%lu.",
                      req_kv_len, real_kv_len, req_info->req_id, req_info->prefix_id, req_info->model_id);
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::QueryKvCacheInCacheManager(const SyncKvReqInfo *req_info, std::vector<KvTensor> &kv_tensors) {
    std::pair<uint64_t, uint64_t> cache_key;
    bool is_prefix = false;
    if (!GetCacheKey(*req_info, cache_key, is_prefix)) {
        // pull by cache_id, and maps to no cache key
        return QueryKvTensorByCacheId(*req_info, kv_tensors);
    }
    CacheEntry cache_entry;
    if (!CacheManager::GetInstance().GetCacheEntry(cache_key, cache_entry, is_prefix)) {
        UDF_LOG_ERROR("cache_id:%ld, req:%lu, model_id:%lu, kv cache not found",
                      req_info->cache_id, req_info->req_id, req_info->model_id);
        return FsmStatus::kFsmKvNotExist;
    }
    const auto &batch_index_and_size = cache_entry.id_to_batch_index_and_size[cache_key.first];
    uint64_t buffer_len = 0U;
    for (uint32_t i = 0U; i < req_info->buffer_count_per_layer; ++i) {
        if (req_info->transfer_infos[i].buffer_info.block_index != UINT64_MAX) {
            UDF_LOG_ERROR("Request param block_index:%lu is invalid.", req_info->transfer_infos[0].buffer_info.block_index);
            return FsmStatus::kFsmParamInvalid;
        }
        buffer_len += req_info->transfer_infos[i].buffer_info.buffer_len;
    }
    if (batch_index_and_size.second < buffer_len) {
        UDF_LOG_ERROR("cache_id:%ld, req_id:%lu, model_id:%lu, kv tensor size (%lu) < required tensor size (%lu)",
                      req_info->cache_id, req_info->req_id, req_info->model_id, batch_index_and_size.second, buffer_len);
        return FsmStatus::kFsmParamInvalid;
    }
    if ((req_info->offset + buffer_len) > batch_index_and_size.second) {
      UDF_LOG_ERROR("cache_id:%ld, req_id:%lu, model_id:%lu, srcCacheOffset(%lu) is invalid, kv tensor size (%lu), required tensor size (%lu)",
                    req_info->cache_id, req_info->req_id, req_info->model_id, req_info->offset, batch_index_and_size.second, buffer_len);
      return FsmStatus::kFsmParamInvalid;
    }
    // seq_len_dim_index == 1时, KV为BSH排布, 可以按需要的大小发送,
    // 否则需要完整发送
    const auto send_size = (cache_entry.seq_len_dim_index == 1) ? buffer_len : batch_index_and_size.second;
    UDF_LOG_INFO("cache_id:%ld, req_id:%lu, prefix_id:%lu, model_id:%lu, tensor_size:%lu, "
                 "buffer_len:%lu, seq_len_dim_index:%d, send_size=%lu",
                 req_info->cache_id, req_info->req_id, req_info->prefix_id, req_info->model_id, batch_index_and_size.second,
                 buffer_len, cache_entry.seq_len_dim_index, send_size);
    const auto batch_offset = batch_index_and_size.first * cache_entry.batch_stride + req_info->offset;
    GetKvTensors(cache_entry, batch_offset, send_size, kv_tensors);
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::QueryBlocksKvCache(const SyncKvReqInfo *req_info, std::vector<KvTensor> &kv_tensors) {
    CacheEntry cache_entry;
    if (req_info->cache_id > 0) {
      if (!CacheManager::GetInstance().GetCacheEntry(req_info->cache_id, cache_entry)) {
        UDF_LOG_ERROR("cache_id:%ld, kv cache not found", req_info->cache_id);
        return FsmStatus::kFsmKvNotExist;
      }
    } else {
      std::pair<uint64_t, uint64_t> cache_key;
      cache_key = std::make_pair(req_info->req_id, req_info->model_id);
      if (!CacheManager::GetInstance().GetCacheEntry(cache_key, cache_entry, false)) {
        UDF_LOG_ERROR("req:%lu, prefix:%lu, model_id:%lu, kv cache not found",
                      req_info->req_id, req_info->prefix_id, req_info->model_id);
        return FsmStatus::kFsmKvNotExist;
      }
    }
    auto kv_tensor_msg = cache_entry.tensors[0]; // AllocateCache make sure tensors is not empty.
    auto data_size = kv_tensor_msg->GetTensor()->GetDataSize();
    for (uint32_t i = 0U; i < req_info->buffer_count_per_layer; ++i) {
        auto &buffer_info = req_info->transfer_infos[i].buffer_info;
        if ((buffer_info.buffer_len % cache_entry.block_len) != 0) {
            UDF_LOG_ERROR("Prompt request buffer length:%lu is not valid.", buffer_info.buffer_len);
            return FsmStatus::kFsmParamInvalid;
        }
        uint64_t buffer_block_num = buffer_info.buffer_len / cache_entry.block_len;
        uint64_t max_block_num = data_size / cache_entry.block_len;
        if (buffer_info.block_index > (max_block_num - buffer_block_num)) {
            UDF_LOG_ERROR("Prompt request buffer block index:%lu, length:%lu is over limit",
                          buffer_info.block_index, buffer_info.buffer_len);
            return FsmStatus::kFsmParamInvalid;
        }
    }
    kv_tensors.reserve(cache_entry.tensors.size());
    for (const auto &kv_tensor_buffer: cache_entry.tensors) {
        KvTensor kv_tensor;
        kv_tensor.tensor_buffer = kv_tensor_buffer;
        kv_tensor.data_addr = static_cast<uint8_t *>(kv_tensor_buffer->GetTensor()->GetData());
        kv_tensor.block_len = cache_entry.block_len;
        kv_tensor.data_size = cache_entry.tensor_size;
        kv_tensors.emplace_back(std::move(kv_tensor));
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::QueryKvTensorByCacheId(const SyncKvReqInfo &req_info, std::vector<KvTensor> &kv_tensors) {
    CacheEntry cache_entry;
    if (!CacheManager::GetInstance().GetCacheEntry(req_info.cache_id, cache_entry)) {
        UDF_LOG_ERROR("cache_id:%ld, kv cache not found", req_info.cache_id);
        return FsmStatus::kFsmKvNotExist;
    }
    if (req_info.batch_index >= static_cast<uint64_t>(cache_entry.batch_size)) {
        UDF_LOG_ERROR("cache_id:%ld, batch_index (%lu) >= batch_size(%u)",
                      req_info.cache_id, req_info.batch_index, cache_entry.batch_size);
        return FsmStatus::kFsmKvNotExist;
    }
    auto buffer_len = req_info.transfer_infos[0].buffer_info.buffer_len;
    if (cache_entry.batch_stride < buffer_len) {
        UDF_LOG_ERROR("cache_id:%ld, kv tensor size (%lu) < required tensor size (%lu)",
                      req_info.cache_id, cache_entry.batch_stride, buffer_len);
        return FsmStatus::kFsmParamInvalid;
    }
    if ((req_info.offset + buffer_len) > cache_entry.batch_stride) {
      UDF_LOG_ERROR("cache_id:%ld, srcCacheOffset(%lu) is invalid, kv tensor size (%lu), required tensor size (%lu)",
                    req_info.cache_id, req_info.offset, cache_entry.batch_stride, buffer_len);
      return FsmStatus::kFsmParamInvalid;
    }
    // 否则需要完整发送
    UDF_LOG_INFO("cache_id:%ld, batch_index:%u, tensor_size:%lu, send_size:%lu,",
                 req_info.cache_id, req_info.batch_index, cache_entry.batch_stride, buffer_len);
    auto batch_offset = req_info.batch_index * cache_entry.batch_stride + req_info.offset;
    GetKvTensors(cache_entry, batch_offset, buffer_len, kv_tensors);
    return FsmStatus::kFsmSuccess;
}

void SendState::ReleaseKvCacheForPrompt(const LlmCommEntity &entity) {
    if (CacheManager::GetInstance().IsEnabled()) {
        bool is_prefix = (entity.GetCurPrefixId() != UINT64_MAX);
        // prompt pa can not release kv tensor
        if (is_prefix || entity.GetCurIsPullBlock() || entity.GetCurIsPullWithOffset()) {
            return;
        }
        const auto &tensor_num_and_indices = entity.GetTensorNumAndIndices();
        if (entity.GetCurCacheIdAndBatchIndex().first >= 0) {
            // In pull by cache_id, if (cache_id, batch_index) is mapped to any CacheIndex, remove it
            CacheIndex cache_index;
            if (CacheManager::GetInstance().GetCacheIndex(entity.GetCurCacheIdAndBatchIndex(), cache_index)) {
                CacheManager::GetInstance()
                    .RemoveCacheIndex(cache_index, false, tensor_num_and_indices.first, tensor_num_and_indices.second);
            }
        } else {
            CacheManager::GetInstance().RemoveCacheIndex(std::make_pair(entity.GetCurReqId(), entity.GetCurModelId()),
                                                         false, tensor_num_and_indices.first, tensor_num_and_indices.second);
        }
    } else {
        (void) KvCacheManager::GetInstance().ReleaseKvCacheForPrompt({entity.GetCurReqId(), entity.GetCurModelId()});
    }
}

FsmStatus SendState::SendSyncKvMetaAsync(LlmCommEntity &entity) {
    LlmCommEntity::SendKvRecordInfo &record_info = entity.GetSendKvRecordInfo();
    // send resp info: need support reentrant
    if (record_info.send_kv_meta_succ_flag) {
        return FsmStatus::kFsmSuccess;
    }
    if (entity.GetServerTickRecord().send_kv_start_tick == 0UL) {
        entity.GetServerTickRecord().send_kv_start_tick = StatisticManager::GetInstance().GetCpuTick();
    }
    FsmStatus ret = entity.SendAsync(entity.GetSyncKvAddrInfo().sync_kv_resp_meta_addr, sizeof(SyncKvMetaInfo));
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    record_info.send_kv_meta_succ_flag = true;
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::SendKvCacheAsync(LlmCommEntity &entity) {
    // send kv tensors: need support reentrant
    LlmCommEntity::SendKvRecordInfo &record_info = entity.GetSendKvRecordInfo();
    auto *resp_info = static_cast<SyncKvMetaInfo *>(entity.GetSyncKvAddrInfo().sync_kv_resp_meta_addr);
    if (record_info.send_kv_succ_count == resp_info->transfer_count) {
        return FsmStatus::kFsmSuccess;
    }
    if (entity.GetServerTickRecord().send_kv_start_tick == 0UL) {
        entity.GetServerTickRecord().send_kv_start_tick = StatisticManager::GetInstance().GetCpuTick();
    }
    uint64_t req_id = entity.GetCurReqId();
    auto *req_info = static_cast<SyncKvReqInfo *>(entity.GetSyncKvAddrInfo().sync_kv_req_addr);
    UDF_LOG_INFO("Begin to send kv, req_id:%lu, buffer_count_per_layer:%u, transfer_count:%u, sendKvSucCount:%lu, entity:%s.",
                 req_id,
                 req_info->buffer_count_per_layer,
                 resp_info->transfer_count,
                 record_info.send_kv_succ_count,
                 entity.GetDesc().c_str());
    uint32_t buffer_count_per_layer = req_info->buffer_count_per_layer;
    for (uint64_t &index = record_info.send_kv_succ_count; index < resp_info->transfer_count; index++) {
        uint64_t kv_tensor_index = index / buffer_count_per_layer;
        uint64_t kv_tensor_transfer_index = index % buffer_count_per_layer;
        auto &kv_tensor = record_info.kv_tensors[kv_tensor_index];
        auto buffer_addr = reinterpret_cast<uintptr_t>(kv_tensor.data_addr);
        size_t buff_size = req_info->transfer_infos[kv_tensor_transfer_index].buffer_info.buffer_len;
        if (req_info->transfer_infos[kv_tensor_transfer_index].buffer_info.block_index == UINT64_MAX) {
            // prompt is continuous, calculate offset by previous buffer.
            uint64_t used_size = 0LU;
            for (uint64_t i = 0U; i < kv_tensor_transfer_index; ++i) {
                buffer_addr += req_info->transfer_infos[i].buffer_info.buffer_len;
                used_size += req_info->transfer_infos[i].buffer_info.buffer_len;
            }
            buff_size = (kv_tensor.data_size < (used_size + buff_size)) ? (kv_tensor.data_size - used_size) : buff_size;
        } else {
            // prompt PA, is not continuous, calculate offset by start block index.
            buffer_addr += req_info->transfer_infos[kv_tensor_transfer_index].buffer_info.block_index * kv_tensor.block_len;
        }
        FsmStatus ret = entity.SendAsync(reinterpret_cast<void *>(buffer_addr), buff_size);
        UDF_LOG_INFO("ReqId:%lu send kvIndex:%lu transferIndex:%lu, tensor info:%s", req_id, kv_tensor_index,
                     kv_tensor_transfer_index, DataDebugString(reinterpret_cast<void *>(buffer_addr), buff_size).c_str());
        if (ret != FsmStatus::kFsmSuccess) {
            return ret;
        }
    }
    UDF_LOG_INFO("Success to call send of kv tensors, total send request count:%zu, req_id:%lu, entity:%s.",
                 entity.GetSendRequests().size(), req_id, entity.GetDesc().c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::TestSendRequestsAsync(LlmCommEntity &entity) {
    std::vector<HcclRequest> &requests = entity.GetSendRequests();
    auto current_test_count = static_cast<int32_t>(requests.size());
    int32_t current_comp_count = 0;
    std::vector<HcclStatus> &comp_status = LlmCommEntityMgr::GetInstance().GetCompStatus(requests.size());
    std::vector<int32_t> &comp_indices = LlmCommEntityMgr::GetInstance().GetCompIndices(requests.size());
    auto request_array = static_cast<HcclRequest *>(requests.data());
    HcclResult ret = HcclRawTestSome(current_test_count, request_array, &current_comp_count,
                                     comp_indices.data(), comp_status.data());
    if (ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawTestSome, count:%d, req_id:%lu, entity:%s.", current_test_count,
                      entity.GetCurReqId(), entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    if (current_comp_count == 0) {
        return FsmStatus::kFsmKeepState;
    }
    for (int32_t index = 0; index < current_comp_count; index++) {
        HcclStatus &status = comp_status[static_cast<size_t>(index)];
        if (status.error != 0) {
            UDF_LOG_ERROR("Get invalid status, status:%d, req_id:%lu, entity:%s.", status.error, entity.GetCurReqId(),
                          entity.GetDesc().c_str());
            return FsmStatus::kFsmHcclFailed;
        }
    }
    EntityStatisticInfo &stat_info = entity.GetStatisticInfo();
    stat_info.test_send_success_times += current_comp_count;
    uint64_t &send_complete_cnt = entity.GetSendKvRecordInfo().send_complete_cnt;
    // finish to send kv meta
    if (send_complete_cnt == 0UL) {
        const uint64_t current_tick_cost =
            StatisticManager::GetInstance().GetCpuTick() - entity.GetServerTickRecord().send_meta_start_tick;
        const double currentTimeCost = StatisticManager::GetInstance().GetTimeCost(current_tick_cost);
        UDF_LOG_INFO("Success to send meta info, req_id:%lu, time cost:%.2f us, entity:%s.",
                     entity.GetCurReqId(), currentTimeCost, entity.GetDesc().c_str());
    }
    send_complete_cnt += current_comp_count;
    if (send_complete_cnt < requests.size()) {
        UDF_LOG_INFO("Success to send some kv tensors, complete_count:%lu, total_count:%zu, req_id:%lu, entity:%s.",
                     send_complete_cnt, requests.size(), entity.GetCurReqId(), entity.GetDesc().c_str());
        return FsmStatus::kFsmKeepState;
    }
    const uint64_t current_tick_cost =
        StatisticManager::GetInstance().GetCpuTick() - entity.GetServerTickRecord().send_kv_start_tick;
    bool max_tick_cost_flag = false;
    UpdateTickCost(current_tick_cost, stat_info.send_kv_total_times, stat_info.send_kv_min_tick_cost,
                   stat_info.send_kv_max_tick_cost, stat_info.send_kv_total_tick_cost, max_tick_cost_flag);
    const double k_time_cost = StatisticManager::GetInstance().GetTimeCost(current_comp_count);
    if (max_tick_cost_flag) {
        entity.SetForcePrintFlag(true);
    }
    UDF_LOG_INFO(
        "Success to send all kv tensors%s, complete_count:%zu, total_count:%zu, req_id:%lu, time cost:%.2f us, "
        "total_times:%lu, entity:%s.",
        max_tick_cost_flag ? kMaxTimeCost : kCommonTimeCost, send_complete_cnt, requests.size(), entity.GetCurReqId(),
        k_time_cost, stat_info.send_kv_total_times, entity.GetDesc().c_str());
    return FsmStatus::kFsmSuccess;
}

FsmStatus SendState::CheckSyncKvReqInfo(const SyncKvReqInfo *req_info,
                                        const std::vector<KvTensor> &kv_tensors,
                                        LlmCommEntity &entity) {
    int64_t cache_id = entity.GetCurCacheIdAndBatchIndex().first;
    uint64_t req_id = entity.GetCurReqId();
    if (kv_tensors.empty()) {
        UDF_LOG_ERROR("Fail to find kv cache for cache_id:%ld, req_id:%lu, model_id:%lu, entity:%s.",
                      cache_id, req_id, req_info->model_id, entity.GetDesc().c_str());
        return FsmStatus::kFsmKvNotExist;
    }
    auto kv_size = static_cast<uint32_t>(kv_tensors.size());
    auto buffer_count_per_layer = req_info->buffer_count_per_layer;
    if (buffer_count_per_layer > (std::numeric_limits<uint32_t>::max() / kv_size)) {
        UDF_LOG_ERROR("Invalid param, transfer count overflow uint32_t, kv_size:%u, buffer_count_per_layer:%u, req_id:%lu, "
                      "entity:%s.", kv_size, buffer_count_per_layer, req_id, entity.GetDesc().c_str());
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

bool SendState::GetCacheKey(const SyncKvReqInfo &req_info, std::pair<uint64_t, uint64_t> &cache_key, bool &is_prefix) {
    bool found = true;
    if (req_info.cache_id >= 0) {
        const auto searchKey = std::make_pair(req_info.cache_id, static_cast<uint32_t>(req_info.batch_index));
        found = CacheManager::GetInstance().GetCacheIndex(searchKey, cache_key);
        if (found) {
            UDF_LOG_INFO("cache_id:%ld, batch_index:%u maps to CacheKey(req_id:%lu, model_id:%lu)",
                         searchKey.first, searchKey.second, cache_key.first, cache_key.second);
        } else {
            UDF_LOG_INFO("cache_id:%ld, batch_index:%u maps to NO CacheKey", searchKey.first, searchKey.second);
        }
    } else {
        is_prefix = (req_info.prefix_id != UINT64_MAX);
        auto realReqId = (is_prefix ? req_info.prefix_id : req_info.req_id);
        cache_key = std::make_pair(realReqId, req_info.model_id);
    }
    return found;
}
}  // namespace FlowFunc
