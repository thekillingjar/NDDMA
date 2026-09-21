/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_ADAPTER_DVPP_GRAPH_OPTIMIZER_H_
#define DVPP_ENGINE_ADAPTER_DVPP_GRAPH_OPTIMIZER_H_

#include "common/dvpp_optimizer.h"

namespace dvpp {
class DvppGraphOptimizer : public ge::GraphOptimizer {
 public:
  /**
   * @brief Construct
   */
  DvppGraphOptimizer() = default;

  /**
   * @brief Destructor
   */
  virtual ~DvppGraphOptimizer() = default;

  // Copy constructor prohibited
  DvppGraphOptimizer(
      const DvppGraphOptimizer& dvpp_graph_optimizer) = delete;

  // Move constructor prohibited
  DvppGraphOptimizer(
      const DvppGraphOptimizer&& dvpp_graph_optimizer) = delete;

  // Copy assignment prohibited
  DvppGraphOptimizer& operator=(
      const DvppGraphOptimizer& dvpp_graph_optimizer) = delete;

  // Move assignment prohibited
  DvppGraphOptimizer& operator=(
      DvppGraphOptimizer&& dvpp_graph_optimizer) = delete;

  /**
   * @brief Initialize graph optimizer
   * @param options initial options
   * @return status whether success
   */
  ge::Status Initialize(const std::map<std::string, std::string>& options,
      ge::OptimizeUtility* const optimize_utility) override;

  /**
   * @brief close graph optimizer
   * @return status whether success
   */
  ge::Status Finalize() override;

  /**
   * @brief optimize graph prepare, before infer shape, use to set dynamic shape
   * @param graph computation graph
   * @return status whether success
   */
  ge::Status OptimizeGraphPrepare(ge::ComputeGraph& graph) override;

  /**
   * @brief optimize original graph, using in graph prepatation stage
   * @param graph computation graph
   * @return status whether success
   */
  ge::Status OptimizeOriginalGraph(ge::ComputeGraph& graph) override;

  /**
   * @brief optimize original graph
   *        using for conversion op insert in graph prepatation stage
   * @param graph computation graph
   * @return status whether success
   */
  ge::Status OptimizeOriginalGraphJudgeInsert(ge::ComputeGraph& graph) override;

  /**
   * @brief optimize fused graph
   * @param graph computation graph
   * @return status whether success
   */
  ge::Status OptimizeFusedGraph(ge::ComputeGraph& graph) override;

  /**
   * @brief optimize whole graph
   * @param graph computation graph
   * @return status whether success
   */
  ge::Status OptimizeWholeGraph(ge::ComputeGraph& graph) override;

  /**
   * @brief get attribute of graph optimizer
   * @param attrs graph optimizer attributes
   * @return status whether success
   */
  ge::Status GetAttributes(ge::GraphOptimizerAttribute& attrs) const override;

 private:
  std::shared_ptr<DvppOptimizer> dvpp_optimizer_{nullptr};
}; // class DvppGraphOptimizer
} // namespace dvpp

#endif // DVPP_ENGINE_ADAPTER_DVPP_GRAPH_OPTIMIZER_H_