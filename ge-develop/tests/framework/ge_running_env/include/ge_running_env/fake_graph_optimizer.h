/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_FAKE_GRAPH_OPTIMIZER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_FAKE_GRAPH_OPTIMIZER_H_
#include <string>
#include <queue>
#include "fake_ns.h"
#include "ge_running_env/env_installer.h"
#include "utils/type_utils.h"
#include "graph/preprocess/multi_batch_copy_graph.h"

FAKE_NS_BEGIN
class FakeGraphOptimizer : public ge::GraphOptimizer {
  Status Initialize(const map<std::string, std::string> &options, OptimizeUtility *const optimize_utility) override {
    return 0;
  }
  Status Finalize() override {
    return 0;
  }
  Status OptimizeOriginalGraph(ComputeGraph &graph) override {
    return 0;
  }
  Status OptimizeFusedGraph(ComputeGraph &graph) override {
    return 0;
  }
  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    return 0;
  }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    return 0;
  }
  Status OptimizeAfterGraphNormalization(const ComputeGraphPtr& graph) override {
    return 0;
  }
};

struct FormatInfo {
  Format format;
  GeShape shape;
};

struct InferredOpFormat {
  std::vector<FormatInfo> input_formats;
  std::vector<FormatInfo> output_formats;
};

class FakeMultiDimsOptimizer : public FakeGraphOptimizer {
 public:
  Status OptimizeAfterGraphNormalization(const ComputeGraphPtr& graph) override {
    return multibatch::ProcessMultiBatch(graph, graph->GetSessionID());
  }
};

class FakeFormatsOptimizer : public FakeGraphOptimizer {
 public:
  FakeFormatsOptimizer &OpFormatByName(std::string name, InferredOpFormat format) {
    op_names_to_format_[std::move(name)] = std::move(format);
    return *this;
  }
  FakeFormatsOptimizer &OpFormatByType(std::string op_type, InferredOpFormat format) {
    op_types_to_format_[std::move(op_type)] = std::move(format);
    return *this;
  }

  Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) override {
    std::queue<NodePtr> nodes;
    std::set<NodePtr> seen_nodes;
    for (const auto &node : graph.GetAllNodes()) {
      if (node->GetInDataNodes().size() == 0) {
        nodes.emplace(node);
        seen_nodes.insert(node);
      }
    }

    while(!nodes.empty()) {
      auto node = std::move(nodes.front());
      nodes.pop();

      for (auto &src_anchor : node->GetAllOutDataAnchors()) {
        auto src_format = GetSrcFormat(node, src_anchor->GetIdx());
        node->GetOpDesc()->MutableOutputDesc(src_anchor->GetIdx())->SetFormat(src_format.format);
        node->GetOpDesc()->MutableOutputDesc(src_anchor->GetIdx())->SetShape(src_format.shape);

        for (auto &dst_anchor : src_anchor->GetPeerInDataAnchors()) {
          auto dst_node = dst_anchor->GetOwnerNode();
          if (seen_nodes.insert(dst_node).second) {
            nodes.push(dst_node);
          }

          auto dst_format = GetDstFormat(dst_node, dst_anchor->GetIdx());
          dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx())->SetFormat(dst_format.format);
          dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx())->SetShape(dst_format.shape);

          if (dst_format.format != src_format.format) {
            InsertTransdata(graph, src_anchor, src_format, dst_anchor, dst_format);
          }
        }
      }
      for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
        if (seen_nodes.insert(out_ctrl_node).second) {
          nodes.push(out_ctrl_node);
        }
      }
    }
    return SUCCESS;
  }

 /* Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &root_graph) override {
    std::queue<NodePtr> nodes;
    std::set<NodePtr> seen_nodes;
    std::vector<ComputeGraphPtr> root_and_all_sub_graphs;
    root_and_all_sub_graphs.emplace_back(root_graph.shared_from_this());
    std::vector<ComputeGraphPtr> subgraphs = root_graph.GetAllSubgraphs();
    root_and_all_sub_graphs.insert(root_and_all_sub_graphs.end(), subgraphs.cbegin(), subgraphs.cend());

    for (const auto &subgraph : subgraphs) {
      for (const auto &node : subgraph->GetDirectNode()) {
        if (node->GetInDataNodes().size() == 0) {
          nodes.emplace(node);
          seen_nodes.insert(node);
        }
      }

      while(!nodes.empty()) {
        auto node = std::move(nodes.front());
        nodes.pop();

        for (auto &src_anchor : node->GetAllOutDataAnchors()) {
          auto src_format = GetSrcFormat(node, src_anchor->GetIdx());
          node->GetOpDesc()->MutableOutputDesc(src_anchor->GetIdx())->SetFormat(src_format.format);
          node->GetOpDesc()->MutableOutputDesc(src_anchor->GetIdx())->SetShape(src_format.shape);

          for (auto &dst_anchor : src_anchor->GetPeerInDataAnchors()) {
            auto dst_node = dst_anchor->GetOwnerNode();
            if (seen_nodes.insert(dst_node).second) {
              nodes.push(dst_node);
            }

            auto dst_format = GetDstFormat(dst_node, dst_anchor->GetIdx());
            dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx())->SetFormat(dst_format.format);
            dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx())->SetShape(dst_format.shape);

            if (dst_format.format != src_format.format) {
              InsertTransdata(*subgraph, src_anchor, src_format, dst_anchor, dst_format);
            }
          }
        }
        for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
          if (seen_nodes.insert(out_ctrl_node).second) {
            nodes.push(out_ctrl_node);
          }
        }
      }
    }
    return SUCCESS;
  }*/
 private:
  FormatInfo GetSrcFormat(const NodePtr &src_node, int32_t out_index) {
    auto iter = op_names_to_format_.find(src_node->GetName());
    if (iter != op_names_to_format_.end()) {
      return iter->second.output_formats[out_index];
    }
    iter = op_types_to_format_.find(src_node->GetType());
    if (iter != op_types_to_format_.end()) {
      return iter->second.output_formats[out_index];
    }
    auto td = src_node->GetOpDesc()->GetOutputDescPtr(out_index);
    return {td->GetFormat(), td->GetShape()};
  }

  FormatInfo GetDstFormat(const NodePtr &dst_node, int32_t in_index) {
    auto iter = op_names_to_format_.find(dst_node->GetName());
    if (iter != op_names_to_format_.end()) {
      return iter->second.input_formats[in_index];
    }
    iter = op_types_to_format_.find(dst_node->GetType());
    if (iter != op_types_to_format_.end()) {
      return iter->second.input_formats[in_index];
    }
    auto td = dst_node->GetOpDesc()->GetInputDescPtr(in_index);
    return {td->GetFormat(), td->GetShape()};
  }
  void InsertTransdata(ComputeGraph &graph,
                       const OutDataAnchorPtr &src_anchor, const FormatInfo &src_format,
                       const InDataAnchorPtr &dst_anchor, const FormatInfo &dst_format) {
    std::string name = "transdata_" + std::to_string(transdata_index_++);
    auto op_desc = MakeShared<OpDesc>(name, TRANSDATA);
    op_desc->AddInputDesc("src", *src_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src_anchor->GetIdx()));
    op_desc->AddOutputDesc("dst", *dst_anchor->GetOwnerNode()->GetOpDesc()->GetInputDescPtr(dst_anchor->GetIdx()));
    op_desc->MutableInputDesc(0)->SetFormat(src_format.format);
    op_desc->MutableInputDesc(0)->SetShape(src_format.shape);
    op_desc->MutableOutputDesc(0)->SetFormat(dst_format.format);
    op_desc->MutableOutputDesc(0)->SetShape(dst_format.shape);
    AttrUtils::SetStr(op_desc, "src_format", TypeUtils::FormatToSerialString(src_format.format));
    AttrUtils::SetStr(op_desc, "dst_format", TypeUtils::FormatToSerialString(dst_format.format));

    auto node = graph.AddNode(op_desc);
    src_anchor->Unlink(dst_anchor);
    src_anchor->LinkTo(node->GetInDataAnchor(0));
    node->GetOutDataAnchor(0)->LinkTo(dst_anchor);
  }
 private:
  std::map<std::string, InferredOpFormat> op_types_to_format_;
  std::map<std::string, InferredOpFormat> op_names_to_format_;
  std::atomic<int32_t> transdata_index_{0};
};

FAKE_NS_END


#endif //AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_FAKE_GRAPH_OPTIMIZER_H_
