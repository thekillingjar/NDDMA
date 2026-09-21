/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ge/fusion/pattern.h"
#include "base/common/plugin/ge_make_unique_util.h"
#include "graph/anchor.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
namespace ge {
namespace fusion {
class PatternImpl {
 public:
  explicit PatternImpl(Graph &&pattern_graph) : graph_(pattern_graph) {}

  const Graph &GetGraph() const {
    return graph_;
  }

  void CaptureTensor(const NodeIo &node_output) {
    GELOGD("[CAPTURE] %s[%s] output: %ld for pattern.", NodeAdapter::GNode2Node(node_output.node)->GetNamePtr(),
           NodeAdapter::GNode2Node(node_output.node)->GetTypePtr(), node_output.index);
    captured_node_outputs_.emplace_back(node_output);
  }

  std::vector<NodeIo> GetCapturedTensors() const {
    return captured_node_outputs_;
  }
 private:
  Graph graph_;
  std::vector<NodeIo> captured_node_outputs_;
};

Pattern::Pattern(Graph &&pattern_graph) {
  impl_ = ge::MakeUnique<PatternImpl>(std::move(pattern_graph));
}

Pattern::~Pattern() = default;

const Graph &Pattern::GetGraph() const {
  if (impl_ == nullptr) {
    static Graph invalid_graph("invalid_graph");
    return invalid_graph;
  }
  return impl_->GetGraph();
}

Pattern &Pattern::CaptureTensor(const NodeIo &node_output) {
  if (impl_ != nullptr) {
    impl_->CaptureTensor(node_output);
  }
  return *this;
}

Status Pattern::GetCapturedTensors(std::vector<NodeIo> &node_outputs) const {
  GE_ASSERT_NOTNULL(impl_);
  node_outputs = impl_->GetCapturedTensors();
  return SUCCESS;
}
}  // namespace fusion
}  // namespace ge
