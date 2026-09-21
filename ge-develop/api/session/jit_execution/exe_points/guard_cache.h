/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_GUARD_CACHE_H
#define CANN_GRAPH_ENGINE_GUARD_CACHE_H
#include <cstdint>
#include <string>
#include <iostream>
#include <vector>
#include <memory>

#include "compute_graph.h"
#include "ge_common/ge_api_types.h"

#include "exe_graph/runtime/tensor.h"
#include "guarded_execution_point.h"


using ComputeGraphPtr = std::shared_ptr<ge::ComputeGraph>;

namespace ge {
class ExecutionPoint;
class GuardCheckCache {
 public:
  GuardCheckCache(uint32_t max_cache_count, ExecutionPoint *owner_point)
      : max_cache_count_(max_cache_count), owner_point_(owner_point){};
  ~GuardCheckCache() {
    RemoveCompiledGraph();
  };

  /**
   * 删除cache中编译结果
   *
   * @return 流程是否正常 GRAPH_SUCCESS/GRAPH_FAIL
   */
  Status RemoveCompiledGraph();

  /**
   * 向cache中添加一份编译结果
   *
   * @param gep 需要保存的model
   * @return
   */
  Status AddCompiledCompiledGraph(GuardedExecutionPoint *gep);

  /**
   * 在cache中找编译好的model
   *
   * @param input_tensor
   * @return 返回保存的 ExecutionPointModel 指针，若为nullptr则视为没找到
   */
  GuardedExecutionPoint *FindGuardedExecutionPoint(const std::vector<gert::Tensor> &input_tensor);

  GuardedExecutionPoint *FindOrCreateGuarded(const std::vector<gert::Tensor> &inputs);

  /**
   * 获取已经保存的cache个数
   *
   * @return 获取已经保存的cache个数
   */
  uint32_t GetSavedCacheNum() const;

  std::vector<std::unique_ptr<GuardedExecutionPoint>> &GetCache() {
    return cache_models_;
  };

 private:
  uint32_t max_cache_count_;
  std::vector<std::unique_ptr<GuardedExecutionPoint>> cache_models_;
  std::vector<char> last_guard_miss_reason_;
  ExecutionPoint *owner_point_;
};
}  // namespace ge
#endif  // CANN_GRAPH_ENGINE_GUARD_CACHE_H
