/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_type_utils.h"
#include <map>

namespace ge {
namespace {
const std::map<rtMemType_t, MemoryType> kRtMemTypeToExternalMemType = {
    {RT_MEMORY_DEFAULT, MemoryType::MEMORY_TYPE_DEFAULT},
    {RT_MEMORY_HBM, MemoryType::MEMORY_TYPE_DEFAULT},
    {RT_MEMORY_P2P_DDR, MemoryType::MEMORY_TYPE_P2P}
};

const std::map<MemoryType, rtMemType_t> kExternalMemTypeToRtMemType = {
    {MemoryType::MEMORY_TYPE_DEFAULT, RT_MEMORY_HBM},
    {MemoryType::MEMORY_TYPE_P2P, RT_MEMORY_P2P_DDR}
};

const std::map<MemoryType, std::string> kExternalMemTypeToString = {
    {MemoryType::MEMORY_TYPE_DEFAULT, "MEMORY_TYPE_DEFAULT"},
    {MemoryType::MEMORY_TYPE_P2P, "MEMORY_TYPE_P2P"}
};

const std::map<rtMemType_t, std::string> kRtMemTypeToString = {
    {RT_MEMORY_DEFAULT, "RT_MEMORY_DEFAULT"},
    {RT_MEMORY_HBM, "RT_MEMORY_HBM"},
    {RT_MEMORY_RDMA_HBM, "RT_MEMORY_RDMA_HBM"},
    {RT_MEMORY_DDR, "RT_MEMORY_DDR"},
    {RT_MEMORY_SPM, "RT_MEMORY_SPM"},
    {RT_MEMORY_P2P_HBM, "RT_MEMORY_P2P_HBM"},
    {RT_MEMORY_P2P_DDR, "RT_MEMORY_P2P_DDR"},
    {RT_MEMORY_DDR_NC, "RT_MEMORY_DDR_NC"},
    {RT_MEMORY_TS, "RT_MEMORY_TS"},
    {RT_MEMORY_TS_4G, "RT_MEMORY_TS_4G"},
    {RT_MEMORY_HOST, "RT_MEMORY_HOST"},
    {RT_MEMORY_SVM, "RT_MEMORY_SVM"},
    {RT_MEMORY_HOST_SVM, "RT_MEMORY_HOST_SVM"}
};
}

graphStatus MemTypeUtils::RtMemTypeToExternalMemType(const rtMemType_t rt_mem_type, MemoryType &external_mem_type) {
  const decltype(kRtMemTypeToExternalMemType)::const_iterator iter = kRtMemTypeToExternalMemType.find(rt_mem_type);
  if (iter != kRtMemTypeToExternalMemType.cend()) {
    external_mem_type = iter->second;
    return GRAPH_SUCCESS;
  }
  // rtMemType转化成对外内存类型失败，说明外部用户无法通过aclrtMalloc申请这种类型内存
  return GRAPH_FAILED;
}

graphStatus MemTypeUtils::ExternalMemTypeToRtMemType(const MemoryType external_mem_type, rtMemType_t &rt_mem_type) {
  const decltype(kExternalMemTypeToRtMemType)::const_iterator iter =
      kExternalMemTypeToRtMemType.find(external_mem_type);
  if (iter != kExternalMemTypeToRtMemType.cend()) {
    rt_mem_type = iter->second;
    return GRAPH_SUCCESS;
  }
  // 外部传入的内存类型转化为rtMemType失败，一般要报错立即返回错误
  return GRAPH_FAILED;
}

std::string MemTypeUtils::ToString(const rtMemType_t rt_mem_type) {
  const decltype(kRtMemTypeToString)::const_iterator iter = kRtMemTypeToString.find(rt_mem_type);
  if (iter != kRtMemTypeToString.cend()) {
    return iter->second;
  }
  return std::to_string(rt_mem_type);
}

std::string MemTypeUtils::ToString(const MemoryType external_mem_type) {
  const decltype(kExternalMemTypeToString)::const_iterator iter = kExternalMemTypeToString.find(external_mem_type);
  if (iter != kExternalMemTypeToString.cend()) {
    return iter->second;
  }
  return "unknown type: " + std::to_string(static_cast<int32_t>(external_mem_type));
}

bool MemTypeUtils::IsMemoryTypeSpecial(const int64_t memory_type) {
  const auto unsigned_type = static_cast<uint64_t>(memory_type);
  if ((unsigned_type == RT_MEMORY_DEFAULT) || (unsigned_type == RT_MEMORY_HBM) || (unsigned_type == RT_MEMORY_DDR)) {
    return false;
  } else {
    return true;
  }
}
} // namespace ge
