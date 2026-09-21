/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iomanip>
#include "ascend_hal.h"
#include "llm_common/llm_common.h"
#include "llm_common/statistic_manager.h"

namespace FlowFunc {
namespace {
constexpr uint32_t kIpValidSize = 4U;
constexpr uint32_t kIpRangeSize = 256U;
}
std::ostream &operator<<(std::ostream &str, const FlowFunc::AscendString &ascend_string) {
    str << ascend_string.GetString();
    return str;
}
std::ostream &operator<<(std::ostream &str, const FlowFunc::TensorDataType &data_type) {
    str << static_cast<int32_t>(data_type);
    return str;
}
std::string ToIp(uint32_t little_endian_decimal) {
    std::string ip;
    for (uint32_t i = 0U; i < kIpValidSize; i++) {
        ip.append(std::to_string(little_endian_decimal % kIpRangeSize));
        little_endian_decimal /= kIpRangeSize;
        if (i < (kIpValidSize - 1)) {
            ip.append(".");
        }
    }
    return ip;
}

std::string ToDesc(const HcclAddr &hccl_addr) {
    std::string desc;
    desc.append("[ip:")
        .append(ToIp(hccl_addr.info.tcp.ipv4Addr))
        .append("_")
        .append(std::to_string(hccl_addr.info.tcp.ipv4Addr))
        .append(", port:")
        .append(std::to_string(hccl_addr.info.tcp.port))
        .append("]");
    return desc;
}

void UpdateTickCost(uint64_t current_tick_cost, uint64_t &total_times, uint64_t &min_tick_cost, uint64_t &max_tick_cost,
                    uint64_t &total_tick_cost, bool &max_tick_cost_flag) {
    total_times++;
    // not statistic first n record
    if (total_times <= kIgnoreFirstRecordCount) {
        max_tick_cost_flag = true;
        return;
    }
    // original value of min_tick_cost is UINT64_MAX
    min_tick_cost = (current_tick_cost < min_tick_cost) ? current_tick_cost : min_tick_cost;
    // original value of max_tick_cost and total_tick_cost is 0
    if (current_tick_cost > max_tick_cost) {
        max_tick_cost = current_tick_cost;
        max_tick_cost_flag = true;
    }
    total_tick_cost += current_tick_cost;
}

void UpdateTickCost(uint64_t current_tick_cost, std::atomic<uint64_t> &total_times,
                    std::atomic<uint64_t> &min_tick_cost, std::atomic<uint64_t> &max_tick_cost,
                    std::atomic<uint64_t> &total_tick_cost, bool &max_tick_cost_flag) {
    total_times++;
    // original value of min_tick_cost is UINT64_MAX
    if (current_tick_cost < min_tick_cost.load()) {
        min_tick_cost.store(current_tick_cost);
    }
    // original value of max_tick_cost and total_tick_cost is 0
    if (current_tick_cost > max_tick_cost.load()) {
        max_tick_cost.store(current_tick_cost);
        max_tick_cost_flag = true;
    }
    total_tick_cost.fetch_add(current_tick_cost);
}

double CalcAverageTimeCost(uint32_t times, uint64_t total_tick_cost) {
    return (times <= kIgnoreFirstRecordCount) ? 0.0f
                                                : StatisticManager::GetInstance().GetTimeCost(total_tick_cost) /
            static_cast<double>(times - kIgnoreFirstRecordCount);
}

FsmStatus CheckTimeout(uint64_t start_tick, uint64_t timeout) {
    uint64_t current_tick = StatisticManager::GetInstance().GetCpuTick();
    double time_cost = StatisticManager::GetInstance().GetTimeCost(current_tick - start_tick);
    return (time_cost >= static_cast<double>(timeout)) ? FsmStatus::kFsmTimeout : FsmStatus::kFsmSuccess;
}

int32_t CheckInputTensor(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
                         uint64_t msgs_min_size,
                         size_t target_pos,
                         uint64_t expect_size,
                         bool is_expect_min) {
    if (input_msgs.size() < msgs_min_size) {
        UDF_LOG_ERROR("input msg size is not valid expect min:%lu, real size:%zu.", input_msgs.size(), msgs_min_size);
        return FLOW_FUNC_FAILED;
    }
    const Tensor *req_info_tensor = input_msgs[target_pos]->GetTensor();
    if ((input_msgs[target_pos]->GetRetCode() != 0) || (req_info_tensor == nullptr)) {
        UDF_LOG_ERROR("input tensor is not valid, ret code:%d.", input_msgs[target_pos]->GetRetCode());
        return FLOW_FUNC_FAILED;
    }
    if (is_expect_min) {
        if (req_info_tensor->GetDataSize() < expect_size) {
            UDF_LOG_ERROR("input tensor size:%lu is not valid, expect min:%lu.",
                          req_info_tensor->GetDataSize(),
                          expect_size);
            return FLOW_FUNC_FAILED;
        }
    } else if (req_info_tensor->GetDataSize() != expect_size) {
        UDF_LOG_ERROR("input tensor size:%lu is not valid, expect:%lu.", req_info_tensor->GetDataSize(), expect_size);
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

inline void PrintBytesHex(const uint8_t *bytes_data, uint64_t size, std::ostringstream &ss) {
    for (uint64_t i = 0UL; i < size; ++i) {
        // byte hex is 2
        ss << std::setfill('0') << std::setw(2) << std::hex << static_cast<uint32_t>(bytes_data[i]) << ' ';
    }
}

std::string DataDebugString(const void *addr, uint64_t size) {
    std::ostringstream debug_info;
    debug_info << "dataSize:" << size;
    if (addr == nullptr) {
        debug_info << ", addr:nullptr";
    } else {
        // dump first 16 bytes and last 16 bytes
        constexpr uint64_t dump_size = 16UL;
        debug_info << ", hex addr:[";
        if (size <= dump_size * 2U) {
            PrintBytesHex(static_cast<const uint8_t *>(addr), size, debug_info);
        } else {
            PrintBytesHex(static_cast<const uint8_t *>(addr), dump_size, debug_info);
            debug_info << "... ";
            PrintBytesHex(static_cast<const uint8_t *>(addr) + (size - dump_size), dump_size, debug_info);
        }
        debug_info << ']';
    }
    return debug_info.str();
}

std::string CopyCacheDebugString(const CopyCacheReqInfo &req_info) {
    std::stringstream ss;
    ss << "src_cache_id:" << req_info.src_cache_id
       << ", dst_cache_id:" << req_info.dst_cache_id
       << ", src_batch_index:" << req_info.src_batch_index
       << ", dst_batch_index:" << req_info.dst_batch_index
       << ", copy_size:" << req_info.copy_size
       << ", copy_block_count:" << req_info.copy_block_count;
    return ss.str();
}

std::string CacheKeyDebugString(const PullKvReqInfo &req_info) {
    std::stringstream ss;
    ss << "prompt_cluster_id:" << req_info.prompt_cluster_id;
    if (req_info.prompt_cache_id != -1) {
        ss << ", prompt_cache_id:" << req_info.prompt_cache_id
           << ", prompt_batch_index:" << req_info.prompt_batch_index;
    } else {
        ss << ", req_id:" << req_info.req_id
           << ", prefix_id:" << req_info.prefix_id
           << ", model_id:" << req_info.model_id;
    }
    return ss.str();
}

std::string TransferReqDebugString(const TransferKvReqInfo &req_info) {
    std::stringstream ss;
    ss << "cache_id:" << req_info.cache_id
       << ", src_batch_index:" << req_info.src_batch_index
       << ", src_layer_index:" << req_info.src_layer_index
       << ", dst_cluster_id:" << req_info.dst_cluster_id
       << ", block_count:" << req_info.block_count
       << ", prompt_block_count:" << req_info.prompt_block_count;
    return ss.str();
}

drvError_t D2DCopy(void *dst, uint64_t dst_max, const void *src, uint64_t count) {
    constexpr uint64_t kMaxBlockSize = 4UL * 1024UL * 1024UL * 1024UL;  // 4 GB
    // actually checked before; return hal error code
    if ((dst_max < count) || (count <= kMaxBlockSize)) {
        return halSdmaCopy(reinterpret_cast<uintptr_t>(dst), dst_max, reinterpret_cast<uintptr_t>(src), count);
    }
    uint64_t offset = 0;
    uint64_t remaining = count;
    drvError_t ret = DRV_ERROR_NONE;
    while ((remaining > 0) && (ret == DRV_ERROR_NONE)) {
        uint64_t size_to_copy = (remaining <= kMaxBlockSize) ? remaining : kMaxBlockSize;
        auto dst_start = static_cast<uint8_t *>(dst) + offset;
        auto src_start = static_cast<const uint8_t *>(src) + offset;
        // checked: dst_max >= count
        ret = halSdmaCopy(reinterpret_cast<uintptr_t>(dst_start), size_to_copy,
                          reinterpret_cast<uintptr_t>(src_start), size_to_copy);
        UDF_LOG_INFO("offset = %lu, size = %lu", offset, size_to_copy);
        offset += kMaxBlockSize;
        remaining -= size_to_copy;
    }
    return ret;
}
const uint64_t *LocalTensorIndices(const PullKvReqInfo &req_info, bool enable_paged_attention) {
    return enable_paged_attention ? (req_info.block_indices + req_info.block_count + req_info.prompt_block_count) :
           req_info.block_indices;
}

const uint64_t *RemoteTensorIndices(const PullKvReqInfo &req_info, bool enable_paged_attention) {
    return LocalTensorIndices(req_info, enable_paged_attention) + req_info.dst_tensor_index_count;
}

}  // namespace FlowFunc
