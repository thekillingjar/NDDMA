/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLM_ENGINE_INC_COMMON_LLM_INNER_TYPES_H
#define LLM_ENGINE_INC_COMMON_LLM_INNER_TYPES_H

#include <vector>
#include "ge/ge_api_types.h"
#include "llm_log.h"
#include "llm_datadist/llm_engine_types.h"

namespace llm {
using char_t = ge::char_t;
using float32_t = ge::float32_t;
using float64_t = ge::float64_t;

constexpr const char kMix[] = "Mix";
constexpr const char kLlmOptionListenIp[] = "llm.ListenIp";
constexpr const char kLlmOptionListenPort[] = "llm.ListenPort";
constexpr const char kLlmOptionEnableRemoteCacheAccessible[] = "llm.EnableRemoteCacheAccessible";
constexpr uint64_t kDefaultTensorNumPerLayer = 2U;

struct IpInfo {
  uint32_t ip = 0U;
  uint16_t port = 0U;
};

struct ClusterInfo {
  uint64_t remote_cluster_id = 0U;
  int32_t remote_role_type = 0;
  std::vector<IpInfo> local_ip_infos;
  std::vector<IpInfo> remote_ip_infos;
};

#pragma pack(push, 1)
enum class LinkStatus : uint32_t { OK = 0U, FAILED = 1U };
enum class RegisterMemoryStatus : uint32_t { OK = 0U, PREPARING = 1U, FAILED = 2U };

struct InnerIpInfo {
  uint32_t local_ip = 0U;
  uint16_t local_port = 0U;
  uint32_t remote_ip = 0U;
  uint16_t remote_port = 0U;
};

struct InnerClusterInfo {
  uint64_t local_cluster_id = 0U;
  uint64_t remote_cluster_id = 0U;
  uint32_t ip_infos_num = 0U;
  InnerIpInfo ip_infos[0];
};

struct LinkInfo {
  uint32_t operate_type = 0U;
  uint32_t cluster_info_num = 0U;
  uint64_t timeout = 0UL;
  uint32_t force_flag = 0U;
  InnerClusterInfo cluster_infos[0];
};

enum class LLMMemType : uint32_t { MEM_TYPE_DEVICE = 0U, MEM_TYPE_HOST = 1U };

struct LLMMemInfo {
  LLMMemType mem_type = LLMMemType::MEM_TYPE_DEVICE;
  uint64_t addr = 0UL;
  uint64_t size = 0UL;
};

struct CacheKey {
  uint64_t prompt_cluster_id;
  int64_t prompt_cache_id = -1;
  uint64_t prompt_batch_index = 0UL;
  uint64_t req_id;
  uint64_t prefix_id = UINT64_MAX;
  uint64_t model_id;
  bool is_allocate_blocks = false;
};

enum class CachePlacement : uint32_t { HOST = 0U, DEVICE = 1U };

enum class CacheMemType : uint32_t { CACHE = 0U, BLOCKS = 1U, MIX = 2U };

struct CacheDesc {
  uint32_t num_tensors;
  ge::DataType data_type;
  int32_t seq_len_dim_index = -1;
  std::vector<int64_t> shape;
  uint32_t placement = 1U;
  CacheMemType cache_mem_type = CacheMemType::CACHE;
  bool remote_accessible = true;
};

struct Cache {
  int64_t cache_id = -1;
  std::vector<std::vector<uintptr_t>> per_device_tensor_addrs;
};

struct PullCacheParam {
  int64_t size = -1;
  uint32_t batch_index = 0U;
  std::vector<uint64_t> prompt_blocks;
  std::vector<uint64_t> decoder_blocks;
  std::vector<uint64_t> src_tensor_indices;
  std::vector<uint64_t> dst_tensor_indices;
  int64_t src_cache_offset = -1;
  int64_t dst_cache_offset = -1;
  uint64_t tensor_num_per_layer = kDefaultTensorNumPerLayer;
};

struct TransferCacheConfig {
  int64_t src_cache_id = 0U;
  uint64_t batch_index = 0U;
  uint64_t layer_index = 0U;
  std::vector<uintptr_t> dst_addrs;
  uint64_t cluster_id = 0U;
  uint64_t model_id_or_cache_id = 0U;
  int64_t dst_cache_id = 0U;
  uint64_t dst_batch_index = 0U;
  uint64_t type = 0U;
  uint64_t dst_layer_index = 0U;
  uint64_t tensor_num_per_layer = kDefaultTensorNumPerLayer;
};

struct TransferBlockConfig {
  uint64_t block_mem_size = 0U;
  std::vector<uint64_t> src_blocks;
  std::vector<uint64_t> dst_blocks;
};

struct CopyCacheParam {
  int64_t src_cache_id = -1;
  int64_t dst_cache_id = -1;
  uint32_t src_batch_index = 0U;
  uint32_t dst_batch_index = 0U;
  uint64_t offset = 0UL;
  int64_t size = -1L;
  uint64_t req_id = UINT64_MAX;  // just for logging
  bool mbuf_involved = false;
  std::vector<std::pair<uint64_t, uint64_t>> copy_block_infos;
};

struct SwapBlockParam {
  const uint64_t block_size;
  const uint32_t type;
  const std::vector<std::pair<int64_t, int64_t>> &block_mapping;
};

struct SwapCacheParam {
  uint32_t src_batch_index = 0U;
  uint32_t dst_batch_index = 0U;
  uint64_t offset = 0UL;
  int64_t size = -1L;
};

struct CopyBlockInfo {
  uint64_t src_block_index;
  uint64_t dst_block_index;
};

struct CopyReqInfo {
  int64_t src_cache_id = -1;
  int64_t dst_cache_id = -1;
  uint32_t src_batch_index = 0U;
  uint32_t dst_batch_index = 0U;
  uint64_t offset = 0UL;
  int64_t size = -1L;
  uint32_t copy_block_count;
  CopyBlockInfo copy_block_infos[];
};

struct CheckLinkReqInfo {
  uint64_t dst_cluster_id = 0UL;
  uint64_t timeout = 0UL;
};

struct PullKvReqInfo {
  uint64_t req_id = 0UL;
  uint64_t prefix_id = UINT64_MAX;
  uint64_t model_id = 0UL;
  uint64_t prompt_cluster_id = 0UL;
  int64_t prompt_cache_id = -1L;
  uint64_t prompt_batch_index = 0UL;
  int64_t cache_id = -1L;
  uint64_t batch_index = 0U;
  uint64_t timeout = 0UL;
  uint64_t block_len = 0UL;
  uint32_t block_count = 0U;
  uint32_t prompt_block_count = 0U;
  uint32_t dst_tensor_index_count = 0U;
  uint32_t src_tensor_index_count = 0U;
  uint32_t is_pull_with_offset = 0U;
  uint64_t src_cache_offset = 0U;
  uint64_t dst_cache_offset = 0U;
  uint64_t block_indices[];
};

struct TransferKvReqInfo {
  int64_t cache_id = -1L;
  uint64_t src_batch_index = 0UL;
  uint64_t src_layer_index = 0UL;
  uint64_t dst_cluster_id = 0UL;
  uint64_t timeout = UINT64_MAX;
  uint64_t dst_key_addr = 0UL;
  uint64_t dst_value_addr = 0UL;
  uint64_t block_len = 0UL;
  uint32_t block_count = 0U;
  uint32_t prompt_block_count = 0U;
  int64_t dst_cache_id = -1;
  uint64_t dst_batch_index = 0U;
  uint64_t dst_layer_index = 0U;
  uint64_t tensor_num_per_layer = 0U;
  uint64_t block_indices[];
};

struct SwitchRoleReqInfo {
  int32_t role_type = 0;
  uint32_t prompt_listen_ip = 0U;
  uint32_t prompt_listen_port = 0U;
};

struct MergeKvReqInfo {
  uint64_t req_id = 0U;
  uint64_t model_id = 0UL;
  uint64_t batch_index = 0U;
  uint64_t batch_id = 0U;
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

struct RemoveCacheIndexReqInfo {
  uint64_t req_id;
  uint64_t prefix_id;
  uint64_t model_id;
};

struct GetCacheTensorsReqInfo {
  int64_t cache_id = -1;
  int64_t tensor_index = -1;
};

#pragma pack(pop)

enum class RunState : uint32_t { TODO = 0, RUNNING, FINISHED, END_OF_SEQUENCE };
struct RequestState {
  explicit RequestState(RunState run_state, bool is_input_ready = false, uint64_t batch_index = 0)
      : run_state_(run_state), is_inputs_ready_(is_input_ready), batch_index_(batch_index) {}
  RequestState() : run_state_(RunState::TODO), is_inputs_ready_(false), batch_index_(0) {}
  RunState run_state_;
  bool is_inputs_ready_;
  uint64_t batch_index_;
};

constexpr int32_t kRecognizedMaxErrorCode = 1000;

// System ID
enum class InnSystemIdType { SYSID_GE = 8 };
// Runtime env
enum class InnLogRuntime {
  RT_HOST = 0b01,
  RT_DEVICE = 0b10,
};

// Sub model
enum class InnSubModuleId {
  COMMON_MODULE = 0,
  CLIENT_MODULE = 1,
  INIT_MODULE = 2,
  SESSION_MODULE = 3,
  GRAPH_MODULE = 4,
  ENGINE_MODULE = 5,
  OPS_MODULE = 6,
  PLUGIN_MODULE = 7,
  RUNTIME_MODULE = 8,
  EXECUTOR_MODULE = 9,
  GENERATOR_MODULE = 10,
  LLM_ENGINE_MODULE = 11,
};

// Error code type
enum class InnErrorCodeType {
  ERROR_CODE = 0b01,
  EXCEPTION_CODE = 0b10,
};

// Error level
enum class InnErrorLevel {
  COMMON_LEVEL = 0b000,
  SUGGESTION_LEVEL = 0b001,
  MINOR_LEVEL = 0b010,
  MAJOR_LEVEL = 0b011,
  CRITICAL_LEVEL = 0b100,
};

inline ge::Status ConvertRetCode(int32_t value) {
  // 0xFFFU means ge::FAILED
  if (value >= kRecognizedMaxErrorCode) {
    return (static_cast<uint32_t>(0xFFU & 0b11) << 30U) | (static_cast<uint32_t>(0xFFU & 0b11) << 28U) |
        (static_cast<uint32_t>(0xFFU & 0b111) << 25U) | (static_cast<uint32_t>(0xFFU & 0xFFU) << 17U) |
        (static_cast<uint32_t>(0xFFU & 0b11111) << 12U) |
        (static_cast<uint32_t>(0x0FFFU) & (static_cast<uint32_t>(0xFFFU)));
  }
  // 0 means ge::SUCCESS
  if (value == 0) {
    return (static_cast<uint32_t>(0xFFU & 0U) << 30U) | (static_cast<uint32_t>(0xFFU & 0U) << 28U) |
           (static_cast<uint32_t>(0xFFU & 0U) << 25U) | (static_cast<uint32_t>(0xFFU & 0U) << 17U) |
           (static_cast<uint32_t>(0xFFU & 0U) << 12U) |
           (static_cast<uint32_t>(0x0FFFU) & (static_cast<uint32_t>(value)));
  }
  return (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(llm::InnLogRuntime::RT_HOST))) << 30U) |
         (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(llm::InnErrorCodeType::ERROR_CODE))) << 28U) |
         (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(llm::InnErrorLevel::COMMON_LEVEL))) << 25U) |
         (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(llm::InnSystemIdType::SYSID_GE))) << 17U) |
         (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(llm::InnSubModuleId::LLM_ENGINE_MODULE))) << 12U) |
         (static_cast<uint32_t>(0x0FFFU) & (static_cast<uint32_t>(value)));
}

inline bool operator==(const IpInfo &lhs, const IpInfo &rhs) {
  return (lhs.ip == rhs.ip) && (lhs.port == rhs.port);
}
}  // namespace llm

#endif  // LLM_ENGINE_INC_COMMON_LLM_INNER_TYPES_H
