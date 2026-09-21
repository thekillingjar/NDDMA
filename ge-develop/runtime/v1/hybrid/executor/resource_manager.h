/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_HYBRID_RESOURCE_MANAGER_H_
#define GE_HYBRID_EXECUTOR_HYBRID_RESOURCE_MANAGER_H_

#include "hybrid/executor/hybrid_data_flow.h"
#include "hybrid/model/graph_item.h"

namespace ge {
namespace hybrid {
class ResourceManager {
 public:
  ResourceManager() = default;
  explicit ResourceManager(const GraphItem *const graph_item) : graph_item_(graph_item) {}
  ~ResourceManager() = default;
  Status Init(const GraphItem *const graph_item);
  DataFlowResourcePtr GetDataFlowResource(const int64_t handle) const;
  DataFlowKernelBasePtr GetDataFlowKernel(const std::string &type) const;
  void ClearDataFlowResources();

 private:
  Status InitDataFlowResource();
  mutable std::mutex mutex_;
  const GraphItem *graph_item_ = nullptr;
  // for data flow ops
  std::unordered_map<int64_t, DataFlowResourcePtr> data_flow_resources_;
  std::unordered_map<std::string, DataFlowKernelBasePtr> data_flow_kernels_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_EXECUTOR_HYBRID_RESOURCE_MANAGER_H_
