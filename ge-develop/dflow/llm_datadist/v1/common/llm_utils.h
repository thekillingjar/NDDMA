/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_LLM_MANAGER_UTILS_H
#define AIR_RUNTIME_LLM_ENGINE_LLM_MANAGER_UTILS_H
#include <unordered_map>
#include <utility>
#include <dirent.h>
#include "llm_datadist/llm_error_codes.h"
#include "ge/ge_api.h"
#include "ge/ge_ir_build.h"
#include "nlohmann/json.hpp"
#include "flow_graph/data_flow.h"
#include "common/flow_graph_manager.h"
#include "common/llm_inner_types.h"
#include "mem_utils.h"
#include "common/def_types.h"

namespace llm {
constexpr const char OPTION_DATA_FLOW_DEPLOY_INFO_PATH[] = "ge.experiment.data_flow_deploy_info_path";
constexpr int32_t kDefaultSyncKvWaitTime = 1000;

struct DecoderWaitTimeInfo {
  int32_t sync_kv_wait_time{kDefaultSyncKvWaitTime};
};

enum class ServerType : uint32_t { Prompt = 0U, Decoder };

class LLMUtils {
 public:
  static ge::Status ParserWaitTimeInfo(const std::map<ge::AscendString, ge::AscendString> &options,
                                       DecoderWaitTimeInfo &wait_time_info);

  template<typename T>
  static ge::Status ToNumber(const std::string &num_str, T &value) {
    std::stringstream ss(num_str);
    ss >> value;
    LLM_CHK_BOOL_RET_STATUS(!ss.fail(), ge::LLM_PARAM_INVALID, "Failed to convert [%s] to number", num_str.c_str());
    LLM_CHK_BOOL_RET_STATUS(ss.eof(), ge::LLM_PARAM_INVALID, "Failed to convert [%s] to number", num_str.c_str());
    return ge::SUCCESS;
  }

  static std::string DebugString(const CacheKey &cache_key);

  static ge::Status ParseDeviceId(const std::map<ge::AscendString, ge::AscendString> &options,
                                  std::vector<int32_t> &device_ids, const char *option);

  static ge::Status GenerateClusterInfo(uint64_t cluster_id,
                                        bool need_listen_ip,
                                        size_t device_num,
                                        std::map<ge::AscendString, ge::AscendString> &options);

  static std::string GenerateNumaConfig(std::vector<int32_t> &device_ids);

  static ge::Status ParseListenIpInfo(const std::map<ge::AscendString, ge::AscendString> &options,
                                      uint32_t &ip_int,
                                      uint32_t &port);

  static ge::Status IpToInt(const std::string &ip, uint32_t &ip_int);

  static ge::Status FindContiguousBlockIndexPair(const std::vector<std::pair<int64_t, int64_t>> &block_mapping,
                                                 std::vector<std::vector<std::pair<int64_t, int64_t>>> &result);

  static ge::Status FindContiguousBlockIndexPair(const std::vector<uint64_t> &src_blocks,
                                                 const std::vector<uint64_t> &dst_blocks,
                                                 std::vector<std::vector<std::pair<int64_t, int64_t>>> &result);
  static ge::Status ParseFlag(const std::string &option_name,
                              const std::map<ge::AscendString, ge::AscendString> &options,
                              bool &enabled);

  static bool CheckMultiplyOverflowInt64(int64_t a, int64_t b);

  static int32_t CeilDiv(int32_t a, int32_t b);

  static ge::Status CalcElementCntByDims(const std::vector<int64_t> &dims, int64_t &element_cnt);

  static bool GetDataTypeLength(const ge::DataType data_type, uint32_t &length);

  static ge::Status GetSizeInBytes(int64_t element_count, ge::DataType data_type, int64_t &mem_size);

  static ge::Status CalcTensorMemSize(const std::vector<int64_t> &dims,
                                      const ge::DataType data_type,
                                      int64_t &mem_size);

 private:
 static ge::Status ParseListenIpInfo(const std::string &option,
                                      uint32_t &ip_int,
                                      uint32_t &port);
};

/// @ingroup domi_common
/// @brief onverts Vector of a number to a string.
/// @param [in] v  Vector of a number
/// @return string
template <typename T>
std::string ToString(const std::vector<T> &v) {
  bool first = true;
  std::stringstream ss;
  ss << "[";
  for (const T &x : v) {
    if (first) {
      first = false;
      ss << x;
    } else {
      ss << ", " << x;
    }
  }
  ss << "]";
  return ss.str();
}

template <typename T>
std::string ToString(const std::vector<std::vector<T>> &v) {
  bool first = true;
  std::stringstream ss;
  ss << "[";
  for (const auto &x : v) {
    if (first) {
      first = false;
      ss << llm::ToString(x);
    } else {
      ss << ", " << llm::ToString(x);
    }
  }
  ss << "]";
  return ss.str();
}

ge::Status ConvertToInt64(const std::string &str, int64_t &val);
}  // namespace llm
#endif  // AIR_RUNTIME_LLM_ENGINE_LLM_MANAGER_UTILS_H
