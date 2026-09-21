/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLICE_RESULT_MOCKER_H
#define SLICE_RESULT_MOCKER_H

#include "jit_execution/exe_points/execution_order.h"
#include "jit_execution/cache/compiled_model_cache.h"

namespace ge {
class SliceResultMocker {
public:
  SliceResultMocker(const std::string &user_graph_key, uint32_t num_eps, const std::vector<uint32_t> &num_geps);

  static void InitGtGraph();

  static void GenOmFile(const std::string &cache_dir, const std::string &graph_key, const ComputeGraphPtr &graph);

  static ComputeGraphPtr GenGraph(const string &graph_name);

  static ComputeGraphPtr GenGraphWithGuard(const string &graph_name,
                                           std::function<void(ShapeEnvAttr &attr)> func);

  static GuardedExecutionPoint *GenGEP(ExecutionPoint &ep, CompiledModelCache &cmc, const std::string &cache_dir, std::string user_graph_key);

  static std::unique_ptr<ExecutionPoint> GenExecutionPoint(int64_t ep_idx, uint32_t num_geps, CompiledModelCache &cmc,
    const std::string &cache_dir, std::string user_graph_key, bool hasRemGraph);

  static void CheckFileGenResult(const ExecutionOrder &order, const std::string &user_graph_key,
                                  const std::string &cache_dir);

  static void CheckMemObjResult(const ExecutionOrder &order, CompiledModelCache &cmc);

  void GenExecutionOrder(ExecutionOrder &order, CompiledModelCache &cmc, const std::string &cache_dir, std::string user_graph_key) const;

  void GenSlicingResultFiles(const std::string &cache_dir, std::string user_graph_key, CompiledModelCache &cmc);

  static std::vector<ComputeGraphPtr> gt_graphs_with_pattern_;

  static int64_t instance_id_;

  static std::unordered_map<std::string, uint32_t> gep_graph_key_to_pattern_map_;

  std::string user_graph_key_;

  uint32_t num_eps_;

  std::vector<uint32_t> num_geps_;
};
}
#endif // SLICE_RESULT_MOCKER_H