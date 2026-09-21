/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_asc_graph.h"
#include <vector>
#include <queue>

#include "gtest/gtest.h"
#include "ascendc_ir_def.h"
#include "ascgraph_info_complete.h"
#include "schedule_utils.h"
namespace ge {
namespace {
void SetStrides(const std::vector<ge::Expression> &repeats, std::vector<ge::Expression> &strides) {
  ge::Expression stride = ge::sym::kSymbolOne;
  for (auto iter = repeats.rbegin(); iter != repeats.rend(); ++iter) {
    if (ge::SymbolicUtils::StaticCheckEq(*iter, ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
      strides.push_back(ge::sym::kSymbolZero);
    } else {
      strides.push_back(stride);
      stride = stride * *iter;
    }
  }
  std::reverse(strides.begin(), strides.end());
}
}  // namespace
using namespace optimize;
EaseAscGraph &EaseAscGraph::Loops(const std::vector<ge::Symbol> &loops) {
  for (size_t i = 0UL; i < loops.size(); ++i) {
    asc_graph_.CreateAxis("z" + std::to_string(i), loops[i]);
  }
  optimize::AscGraphInfoComplete::CompleteApiInfo(asc_graph_);
  return *this;
}

EaseAscGraph &EaseAscGraph::Broadcast(const std::string &name, const std::set<size_t> &brc_index) {
  std::queue<ge::Node *> node_before_brc;
  std::set<ge::Node *> visited_nodes;
  auto brc = asc_graph_.FindNode(name.c_str());
  if (brc == nullptr) {
    return *this;
  }

  node_before_brc.emplace(brc->inputs[0].anchor.GetOwnerNodeBarePtr());
  while (!node_before_brc.empty()) {
    auto top_node = node_before_brc.front();
    visited_nodes.emplace(top_node);
    node_before_brc.pop();
    size1_node_to_index_[top_node].insert(brc_index.begin(), brc_index.end());
    for (const auto &in_node : top_node->GetInDataNodes()) {
      if (visited_nodes.count(in_node.get()) == 0UL) {
        node_before_brc.emplace(in_node.get());
      }
    }
  }
  return *this;
}

void EaseAscGraph::Build() {
  auto axes = asc_graph_.GetAllAxis();
  std::vector<ge::Expression> loop_repeats;
  std::vector<ge::AxisId> axis_ids;

  for (const auto &axis : axes) {
    loop_repeats.push_back(axis->size);
    axis_ids.push_back(axis->id);
  }

  for (const auto &node : asc_graph_.GetAllNodes()) {
    node->attr.sched.axis = axis_ids;
    auto node_repeats = loop_repeats;
    auto iter = size1_node_to_index_.find(node.get());
    if (iter != size1_node_to_index_.end()) {
      for (auto index : iter->second) {
        node_repeats[index] = ge::sym::kSymbolOne;
      }
    }
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    bool is_follow_input = (node->attr.api.compute_type == ge::ComputeType::kComputeElewise ||
                            node->attr.api.compute_type == ge::ComputeType::kComputeStore);
    for (auto &output : node->outputs()) {
      output->attr.axis = axis_ids;
      output->attr.vectorized_axis = axis_ids;
      if (is_follow_input) {
        output->attr.repeats = node->inputs[0].attr.repeats;
      } else {
        output->attr.repeats = node_repeats;
      }
      SetStrides(node_repeats, output->attr.strides);
      SetStrides(node_repeats, output->attr.vectorized_strides);
    }
  }

  std::vector<ge::Expression> strides;
}

}  // namespace ge
