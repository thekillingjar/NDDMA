/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser_utils.h"
#include "ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "framework/omg/parser/parser_types.h"
#include "graph/anchor.h"
#include "common/checker.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_registry.h"
#include "base/err_msg.h"

namespace ge {
namespace {
bool HasOneNonDataNode(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  int32_t non_data_nums = 0;
  for (const auto& node : graph->GetDirectNode()) {
    if ((node->GetType() != parser::DATA) && (node->GetType() != parser::NETOUTPUT)) {
      non_data_nums++;
    }
  }
  GELOGD("Graph has non data node num is %d", non_data_nums);
  return (non_data_nums == 1);
}
}

Status ParserUtils::ExpandOneToManyGraph(const Graph &graph, OutputMapping &output_mapping) {
  GELOGD("Begin to run ParserUtils::ExpandOneToManyGraph.");
  for (const auto &ge_node : graph.GetDirectNode()) {
    NodePtr node = NodeAdapter::GNode2Node(ge_node);
    GE_CHECK_NOTNULL(node);
    std::string ori_type;
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, ori_type);
    domi::ParseOpToGraphFunc parse_op_to_graph_func =
        domi::OpRegistry::Instance()->GetParseOpToGraphFunc(node->GetType(), ori_type);
    if (parse_op_to_graph_func == nullptr) {
      GELOGD("node:%s type:%s ori type:%s has no parse_op_to_graph_func.",
             node->GetName().c_str(), node->GetType().c_str(), ori_type.c_str());
      continue;
    }
    GELOGI("node:%s type:%s ori type:%s has registered one to many parser func.",
           node->GetName().c_str(), node->GetType().c_str(), ori_type.c_str());
    Graph subgraph("one_to_many_graph");
    Operator op = OpDescUtils::CreateOperatorFromNode(node);
    Status ret = parse_op_to_graph_func(op, subgraph);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get one to many graph failed for op:%s.", GetOperatorName(op).c_str());
      GELOGE(FAILED, "[Invoke][ParseOpToGraphFunc]Get one to many graph failed for op:%s.",
             GetOperatorName(op).c_str());
      return FAILED;
    }
    ret = ExpandNodeToSubgraph(subgraph, node, graph, output_mapping);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Invoke][ExpandNodeToSubgraph]Expand one to many graph failed for op:%s.",
             GetOperatorName(op).c_str());
      return FAILED;
    }
  }
  ComputeGraphPtr compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  if (compute_graph->TopologicalSorting() != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "TopologicalSorting failed, graph:%s.", compute_graph->GetName().c_str());
    GELOGE(FAILED, "[Invoke][TopologicalSorting] failed, graph:%s.", compute_graph->GetName().c_str());
    return FAILED;
  }
  GELOGD("Run ParserUtils::ExpandOneToManyGraph success.");
  return SUCCESS;
}

Status ParserUtils::ExpandNodeToSubgraph(const Graph &subgraph, const NodePtr &node, const Graph &graph,
                                         OutputMapping &output_mapping) {
  ComputeGraphPtr sub_compute_graph = GraphUtilsEx::GetComputeGraph(subgraph);
  GE_CHECK_NOTNULL(sub_compute_graph);
  ComputeGraphPtr compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  // add subgraph node to graph.
  bool no_need_change_name = HasOneNonDataNode(sub_compute_graph);
  for (const auto &sub_node : sub_compute_graph->GetDirectNode()) {
    auto op_desc = sub_node->GetOpDesc();
    std::string new_name;
    if (!no_need_change_name) {
      static std::atomic_long new_node_index(0);
      new_name = "PartitionedCall_" + sub_node->GetName() + "_" + to_string(new_node_index++);
    } else {
      new_name = node->GetName();
    }
    op_desc->SetName(new_name);
    std::vector<std::string> node_name_vec = { node->GetName() };
    (void)ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
        std::move(node_name_vec));
  }
  GE_ASSERT_SUCCESS(GraphUtils::ExpandNodeWithGraph(node, sub_compute_graph));
  // handle output context.
  std::vector<std::pair<NodePtr, int32_t>> out_node_index = sub_compute_graph->GetGraphOutNodesInfo();
  for (size_t index = 0; index < out_node_index.size(); index++) {
    NodePtr out_node = out_node_index[index].first;
    int32_t out_index = out_node_index[index].second;
    GELOGD("Begin to handle output node: %s[%d] with index:%zu", out_node->GetName().c_str(), out_index, index);
    std::string key = GenOutputKey({node->GetName(), index});
    GE_ASSERT_NOTNULL(out_node);
    output_mapping[key] = std::make_pair(out_node->GetName(), out_index); 
  }
  return SUCCESS;
}

string ParserUtils::GenOutputKey(const OutputNodeInfo &node_info) {
  return node_info.first + ":" + std::to_string(node_info.second);
}

void ParserUtils::UpdateOutputNodeInfo(const OutputMapping &final_output_nodes, OutputNodeInfo &output_node_info) {
  std::string key = ParserUtils::GenOutputKey(output_node_info);
  auto iter = final_output_nodes.find(key);
  if (iter != final_output_nodes.end()) {
    output_node_info = iter->second;
    GELOGD("Update output node info, origin[%s], now[%s].",
           key.c_str(), ParserUtils::GenOutputKey(output_node_info).c_str());
  }
}

void ParserUtils::UpdateOutputCtx(const OutputMapping &final_output_nodes, OutputMapping &tensor_to_nodes) {
  for (auto &tensor_to_node : tensor_to_nodes) {
    std::string tensor_name = tensor_to_node.first;
    auto &output_node_info = tensor_to_node.second;
    UpdateOutputNodeInfo(final_output_nodes, output_node_info);
  }
}

std::string ParserUtils::GetOperatorName(const Operator &op) {
  AscendString name;
  (void)op.GetName(name);
  return name.GetString() == nullptr ? "" : std::string(name.GetString());
}

std::string ParserUtils::GetOperatorType(const Operator &op) {
  AscendString type;
  (void)op.GetOpType(type);
  return type.GetString() == nullptr ? "" : std::string(type.GetString());
}

std::string ParserUtils::GetGraphName(const Graph &graph) {
  AscendString name;
  (void)graph.GetName(name);
  return name.GetString() == nullptr ? "" : std::string(name.GetString());
}
}  // namespace ge
