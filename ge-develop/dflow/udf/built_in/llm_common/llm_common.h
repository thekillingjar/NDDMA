/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_LLM_COMMON_LLM_COMMON_H
#define BUILT_IN_LLM_COMMON_LLM_COMMON_H

#include <type_traits>
#include <atomic>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include "ascend_hal.h"
#include "flow_func/flow_msg.h"
#include "fsm/state_define.h"
#include "hccl/base.h"
#include "flow_func/ascend_string.h"
#include "tensor_data_type.h"

namespace FlowFunc {
constexpr size_t kOneAggregatedTransferBlockNums = 64UL;
#pragma pack(push, 1)
struct PullOrTransferExtraInfo {
    bool enable_paged_attention;
    bool is_pull;
    bool is_check;
    uint64_t start_tick;
};
struct ClientClusterInfo {
    uint64_t cluster_id;
};
// transfer kv req info: input of udf func
struct CheckLinkReqInfo {
    uint64_t dst_cluster_id;
    uint64_t timeout;  // timeout interval: us
};
struct LLMReqInfo {
    uint64_t req_id;      // req id
    uint64_t prefix_id;   // prefix id
    uint64_t model_id;    // model id
    uint64_t prompt_length;
};
struct SyncKvMetaInfo {
    uint64_t req_id;
    uint64_t model_id;    // model id
    int32_t err_code;
    uint32_t transfer_count;  // transfer count, PA: kvSize * bufferCountPerLayer, Non-PA: KvSize
};
// pull kv req info: input of udf func
struct PullKvReqInfo {
    uint64_t req_id;
    uint64_t prefix_id;
    uint64_t model_id;    // model id
    uint64_t prompt_cluster_id;
    int64_t prompt_cache_id;
    uint64_t prompt_batch_index;
    int64_t cache_id;     // for cache engine
    uint64_t batch_index;
    uint64_t timeout;     // timeout interval: us
    uint64_t block_len;   // host no need fill, PA: blockSize * hiddenSize, Non-PA: kvCacheTensorSize_
    uint32_t block_count;     // block count per layer, Non-PA: 1
    uint32_t prompt_block_count;
    uint32_t dst_tensor_index_count;
    uint32_t src_tensor_index_count;
    uint32_t is_pull_with_offset;
    uint64_t src_cache_offset;
    uint64_t dst_cache_offset;
    uint64_t block_indices[];  // block indices and tensor indices
};

// transfer kv req info: input of udf func
struct TransferKvReqInfo {
    int64_t cache_id = -1L;
    uint64_t src_batch_index = 0UL;
    uint64_t src_layer_index = 0UL;
    uint64_t dst_cluster_id = 0UL;
    uint64_t timeout = UINT64_MAX;         // timeout interval: us
    uint64_t dst_key_addr = 0UL;
    uint64_t dst_value_addr = 0UL;
    uint64_t block_len = 0UL;        // when src is not block, dst is block, host need fill, PA: blockSize * hiddenSize, Non-PA: kvCacheTensorSize_
    uint32_t block_count = 0U;
    uint32_t prompt_block_count = 0U;
    int64_t dst_cache_id = -1L;
    uint64_t dst_batch_index = 0UL;
    uint64_t dst_layer_index = 0UL;
    uint64_t tensor_num_per_layer = 0U;
    uint64_t block_indices[];  // block indices
};
struct TransferSlotInfo {
    uint64_t slot_nums_per_transfer = 0U;
    uint64_t slot_offset = 0UL;
    uint64_t slot_size = 0UL;
};
struct TransferToRemoteReq {
    uint64_t key_addr = 0UL;
    uint64_t value_addr = 0UL;
    uint64_t max_size = 0UL;
    uint64_t send_nums_per_tensor = 0U;
    uint64_t total_slot_nums = 0U;
    int64_t dst_cache_id = -1L;
    uint64_t dst_batch_index = 0UL;
    uint64_t dst_layer_index = 0UL;
    uint64_t tensor_num_per_layer = 0U;
    TransferSlotInfo slot_infos[];
};
struct TransferKvMetaInfo {
    int32_t err_code;
};
struct SyncBufferInfo {
    uint64_t block_index;    // sync kv start block index, UINT64_MAX mean prompt is not PA
    uint64_t buffer_len;    // sync kv buffer length
};

union TransferInfo {
    SyncBufferInfo buffer_info;
    uint64_t tensor_index;
};

// sync kv req info: used for raw communication between prompt and decoder
struct SyncKvReqInfo {
    int64_t cache_id = -1L;
    uint64_t batch_index = 0L;
    uint64_t req_id = 0UL;
    uint64_t prefix_id = UINT64_MAX;
    uint64_t model_id = 0UL;  // model id
    uint32_t is_pull_block = 0U;
    uint32_t buffer_count_per_layer = 0U;  // Non-PA: 1, PA: buffer count after aggregate
    uint32_t tensor_index_count = 0U;
    uint64_t offset = 0U;  // used by pull with offset
    uint32_t is_pull_with_offset = 0U;
    TransferInfo transfer_infos[];         // Non-PA: kvCacheTensorSize_, PA: every buffer may be different
};
struct MergeKvReqInfo {
    uint64_t req_id;
    uint64_t model_id;  // model id
    uint64_t batch_index;
    uint64_t batch_id;
};
struct IpInfo {
    uint32_t local_ip;
    uint16_t local_port;
    uint32_t remote_ip;
    uint16_t remote_port;
};
struct ClusterInfo {
    uint64_t local_cluster_id;
    uint64_t remote_cluster_id;
    uint32_t ip_num;
    IpInfo ip_infos[];
};
struct UpdateLinkReqInfo {
    uint32_t operator_type;
    uint32_t cluster_num;
    uint64_t timeout; // timeout interval: us
    uint32_t force_flag;
    ClusterInfo cluster_infos[];
};

struct AllocateCacheReqInfo {
    int64_t cache_id = -1;
    bool is_prefix = false;
    bool is_allocate_blocks = false;
    uint32_t num_tensors = 0U;
    int32_t dtype = 0;
    int32_t seq_len_dim_index = -1;
    uint32_t num_dims = 0;
    int64_t dims[32]{};
    uint64_t model_id = 0;
    uint64_t num_requests = 0;
    uint64_t req_ids[0];
};

struct BlockCopyInfo {
    uint64_t src_block_index;
    uint64_t dst_block_index;
};

struct CopyCacheReqInfo {
    int64_t src_cache_id = -1;
    int64_t dst_cache_id = -1;
    uint32_t src_batch_index = 0;
    uint32_t dst_batch_index = 0;
    uint64_t offset = 0UL;
    int64_t copy_size = -1L;
    uint32_t copy_block_count = 0U;
    BlockCopyInfo block_copy_infos[];
};

struct RemoveCacheIndexReqInfo {
    uint64_t req_id = 0U;
    uint64_t prefix_id = UINT64_MAX;
    uint64_t model_id = 0U;
};

struct GetCacheReqInfo {
    int64_t cache_id = -1;
    int64_t tensor_index = -1;
};

struct SwitchRoleReqInfo {
    int32_t role_id = 0;
    uint32_t prompt_listen_ip = 0U;
    uint32_t prompt_listen_port = 0U;
};
#pragma pack(pop)

enum class LinkReqType : uint32_t {
    kUnlink = 0U,
    kLink = 1U,
    kServerUnlink = 2U
};

struct KvTensor {
    std::shared_ptr<FlowMsg> tensor_buffer;  // keeping shared_ptr valid
    void *data_addr;
    size_t data_size;
    uint64_t block_len;
};

struct CacheEntry {
    uint32_t batch_size;
    int32_t seq_len_dim_index = -1;
    uint64_t tensor_size;
    uint64_t batch_stride;
    size_t ext_ref_count;
    bool is_blocks = false;
    std::pair<uint64_t, uint64_t> blocks_cache_index;
    uint64_t block_len = 0UL;
    std::vector<std::shared_ptr<FlowMsg>> tensors;
    std::map<uint64_t, std::pair<uint32_t, uint64_t>> id_to_batch_index_and_size;
    std::map<uint64_t, std::set<uint64_t>> id_to_unpulled_tensor_indices;
};

std::ostream &operator<<(std::ostream &str, const FlowFunc::AscendString &ascend_string);
std::ostream &operator<<(std::ostream &str, const FlowFunc::TensorDataType &data_type);
/**
 * trans vector to string
 * @tparam T
 * @param vec
 * @return
 */
template<typename T>
std::string ToString(const std::vector<T> &vec)
{
    std::stringstream ss;
    bool first = true;
    ss << "[";
    for (const T &ele: vec) {
        if (first) {
            first = false;
        } else {
            ss << ",";
        }
        ss << ele;
    }
    ss << "]";
    return ss.str();
}

/**
 * trans vector to string
 * @tparam T
 * @param vec
 * @return
 */
template<typename T>
std::string ToString(const std::vector<std::vector<T>> &vec)
{
    std::stringstream ss;
    bool first = true;
    for (const std::vector<T> &ele: vec) {
        if (first) {
            first = false;
        } else {
            ss << ",";
        }
        ss << FlowFunc::ToString(ele);
    }
    return ss.str();
}

/**
 * trans little endian decimal to ip addr
 * @param little_endian_decimal
 * @return
 */
std::string ToIp(uint32_t little_endian_decimal);

/**
 * trans hccl addr to string
 * @param hccl_addr
 * @return
 */
std::string ToDesc(const HcclAddr &hccl_addr);

/**
 * update tick cost
 * @param current_tick_cost
 * @param total_times
 * @param min_tick_cost
 * @param max_tick_cost
 * @param total_tick_cost
 * @param max_tick_cost_flag
 */
void UpdateTickCost(uint64_t current_tick_cost, uint64_t &total_times,
                      uint64_t &min_tick_cost, uint64_t &max_tick_cost,
                      uint64_t &total_tick_cost, bool &max_tick_cost_flag);

/**
 * update tick cost
 * @param current_tick_cost
 * @param total_times
 * @param min_tick_cost
 * @param max_tick_cost
 * @param total_tick_cost
 * @param max_tick_cost_flag
 */
void UpdateTickCost(uint64_t current_tick_cost,
                      std::atomic<uint64_t> &total_times,
                      std::atomic<uint64_t> &min_tick_cost,
                      std::atomic<uint64_t> &max_tick_cost,
                      std::atomic<uint64_t> &total_tick_cost,
                      bool &max_tick_cost_flag);

/**
 * calculate average time cost
 * @param times
 * @param total_tick_cost
 * @return
 */
double CalcAverageTimeCost(uint32_t times, uint64_t total_tick_cost);

FsmStatus CheckTimeout(uint64_t start_tick, uint64_t timeout);

int32_t CheckInputTensor(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
                           uint64_t msgs_min_size,
                           size_t target_pos,
                           uint64_t expect_size,
                           bool is_expect_min = false);

std::string DataDebugString(const void *addr, uint64_t size);

std::string CopyCacheDebugString(const CopyCacheReqInfo &req_info);

std::string CacheKeyDebugString(const PullKvReqInfo &req_info);

std::string TransferReqDebugString(const TransferKvReqInfo &req_info);

drvError_t D2DCopy(void *dst, uint64_t dst_max, const void *src, uint64_t count);

const uint64_t *LocalTensorIndices(const PullKvReqInfo &req_info, bool enable_paged_attention);

const uint64_t *RemoteTensorIndices(const PullKvReqInfo &req_info, bool enable_paged_attention);
} // namespace FlowFunc
#endif // BUILT_IN_LLM_COMMON_LLM_COMMON_H
