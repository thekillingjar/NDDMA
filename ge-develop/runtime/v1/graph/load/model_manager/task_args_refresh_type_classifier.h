/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_ARGS_REFRESH_TYPE_CLASSIFIER_H_
#define AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_ARGS_REFRESH_TYPE_CLASSIFIER_H_
#include <map>
#include <cstdint>
#include <memory>
#include "memory_app_type_classifier.h"
#include "task_node_map.h"
#include "task_info/task_info.h"
#include "common/types.h"
#include "graph/small_vector.h"

namespace ge {
class TaskArgsRefreshTypeClassifier {
 public:
  // 理论上，可以直接将logical地址返回，将这一块logical地址作为fixed地址，
  // 但是令人担心的是子block复用、padding等场景，导致fixed地址额外申请较大尺寸的内存，
  // 因此这里精确地返回fixed地址的产生位置，为后面精确申请fixed地址做准备。
  enum IndexType : int32_t { kInput, kOutput, kWorkspace, kEnd };
  static const char_t *GetIndexTypeStr(IndexType it) {
    switch (it) {
      case kInput:
        return "input";
      case kOutput:
        return "output";
      case kWorkspace:
        return "workspace";
      default:
        return "unknown";
    }
  }
  static bool IsIdentityLikeType(const string &node_type) {
    static const std::set<std::string> identity_like_node_type{IDENTITY, MEMCPYASYNC};
    return (identity_like_node_type.count(node_type) > 0U);
  }

  struct TaskFixedAddr {
    size_t task_index;
    size_t iow_index;
    IndexType iow_index_type;
  };
  using FixedAddrs = std::vector<SmallVector<TaskFixedAddr, 2UL>>;

  struct TaskRefreshType {
    uint64_t task_refresh_type;
    std::vector<uint64_t> input_refresh_types;
    std::vector<uint64_t> output_refresh_types;
    std::vector<uint64_t> workspace_refresh_types;
  };

 public:
  static constexpr uint64_t kRefreshByModelIo = 1UL << 0U;
  static constexpr uint64_t kRefreshByFm = 1UL << 1U;

 public:
  TaskArgsRefreshTypeClassifier(
      const TaskNodeMap &task_node_map,
      const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_memory_app_type,
      bool is_fm_refresh_enable);

  /**
   *
   * @param params
   * @param task_indexes_to_refresh_type
   * @param fixed_addrs 识别出的fixed_addrs，外层vector代表识别了多少个fixed addr，内存vector代表本fixed
   * addr下关联的task信息
   * @return
   */
  Status ClassifyMultiTasks(const std::vector<TaskRunParam> &params,
                            std::vector<TaskRefreshType> &task_indexes_to_refresh_type, FixedAddrs &fixed_addrs,
                            bool is_physical_memory_refreshable = false) const;
  uint64_t GetRefreshTypeByLogicalAddr(const AddrDesc &addr_desc) const;

 private:
  const TaskNodeMap &task_ndoe_map_;
  const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_memory_app_type_;
  bool is_fm_refresh_enable_;
};
}  // namespace ge

#endif  // AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_ARGS_REFRESH_TYPE_CLASSIFIER_H_
