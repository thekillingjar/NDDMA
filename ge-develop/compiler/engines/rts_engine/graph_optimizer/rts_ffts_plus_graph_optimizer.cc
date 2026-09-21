/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rts_ffts_plus_graph_optimizer.h"
#include "common/constant/constant.h"

using namespace ge;
namespace cce {
namespace runtime {

RtsFftsPlusGraphOptimizer::RtsFftsPlusGraphOptimizer() {}

RtsFftsPlusGraphOptimizer::~RtsFftsPlusGraphOptimizer() {}

ge::Status RtsFftsPlusGraphOptimizer::Initialize(const map<std::string, std::string> &options,
                                                 ge::OptimizeUtility *const optimizeUtility) {
  (void)options;
  (void)optimizeUtility;
  // do nothing
  return SUCCESS;
}

ge::Status RtsFftsPlusGraphOptimizer::Finalize() {
  // do nothing
  return SUCCESS;
}

ge::Status RtsFftsPlusGraphOptimizer::OptimizeGraphPrepare(ComputeGraph &graph) {
  (void)graph;
  return SUCCESS;
}

ge::Status RtsFftsPlusGraphOptimizer::OptimizeWholeGraph(ComputeGraph &graph) {
  (void)graph;
  // do nothing
  return SUCCESS;
}

ge::Status RtsFftsPlusGraphOptimizer::GetAttributes(GraphOptimizerAttribute &attrs) const {
  attrs.scope = ge::UNIT;
  attrs.engineName = RTS_FFTS_PLUS_ENGINE_NAME;
  return SUCCESS;
}

ge::Status RtsFftsPlusGraphOptimizer::OptimizeOriginalGraph(ComputeGraph &graph) {
  (void)graph;
  // do nothing
  return SUCCESS;
}

ge::Status RtsFftsPlusGraphOptimizer::OptimizeFusedGraph(ComputeGraph &graph) {
  (void)graph;
  // do nothing
  return SUCCESS;
}

ge::Status RtsFftsPlusGraphOptimizer::OptimizeGraphBeforeBuild(ComputeGraph &graph) {
  (void)graph;
  // do nothing
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
