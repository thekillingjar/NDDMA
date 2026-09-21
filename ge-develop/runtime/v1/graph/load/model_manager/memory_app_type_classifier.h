/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MEMORY_APP_TYPE_CLASSIFIER_H_
#define AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MEMORY_APP_TYPE_CLASSIFIER_H_
#include <cstdint>
#include <map>
#include <vector>
#include <set>
#include "task_info/task_info.h"
namespace ge {
enum class MemoryAppType : int32_t {
  kMemoryTypeFix,  // const and var and fix fm
  kMemoryTypeFeatureMap,
  kMemoryTypeModelIo,
  kEnd
};
const char_t *GetMemoryAppTypeStr(MemoryAppType t);
class MemoryAppTypeClassifier {
 public:
  explicit MemoryAppTypeClassifier(const std::vector<MemAllocation> &allocations, const size_t fm_start_id);
  /**
   *
   * @param params
   * @return key: <memory-type-return-by-task-info,  logical-addr-return-by-task-info>, value: memory app type
   */
  std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> ClassifyByTaskRunParams(
      const std::vector<TaskRunParam> &params) const;

 private:
  void ClassifyAddrs(const std::vector<AddrDesc> &addrs,
                     std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_memory_type) const;
  MemoryAppType ClassifyByLogicalAddr(const std::pair<uint64_t, uint64_t> &mem_type_and_logical_addr) const;

 private:
  std::vector<std::pair<uint64_t, uint64_t>> fusion_mem_allocations_;
  std::set<MemAllocation> sort_fm_allocations_;
};
}  // namespace ge

#endif  // AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MEMORY_APP_TYPE_CLASSIFIER_H_
