/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/control_flow_and_stream/data_pass.h"

#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
const int32_t kDataIndexOffset = 2;
Status MappingSubgraphInput(const ComputeGraphPtr &graph, const std::function<int(int32_t data_index)> &input) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }

    int32_t index = -1;
    if (!AttrUtils::GetInt(node->GetOpDesc(), "index", index)) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from op:%s(%s) failed", "index",
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Get][Attr] index from op:%s(%s) failed",
             node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }

    int32_t parent_index = input(index);
    GELOGI("Generate subgraph input map for subgraph %s, data index %d, parent index %d",
           graph->GetName().c_str(), index, parent_index);
    if (!AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
             node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MappingSubgraphOutput(const ComputeGraphPtr &graph, const std::function<int(int32_t retval_index)> &output) {
  const auto &output_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  if (output_node == nullptr) {
    return SUCCESS;
  }

  const auto &op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (size_t index = 0; index < op_desc->GetInputsSize(); ++index) {
    int32_t parent_index = output(index);
    GELOGI("Generate subgraph output map for subgraph %s, index %zu, parent index %d",
           graph->GetName().c_str(), index, parent_index);
    if (parent_index == -1) {
      continue;
    }

    GeTensorDescPtr tensor = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor);
    if (!AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to tensor of op:%s(%s) input:%zu failed",
                        ATTR_NAME_PARENT_NODE_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                        index);
      GELOGE(FAILED, "[Set][Attr] %s to tensor of op:%s(%s) input:%zu failed",
             ATTR_NAME_PARENT_NODE_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), index);
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MappingSubgraphIndex(const ComputeGraphPtr &graph,
                            const std::function<int(int32_t data_index)> &input,
                            const std::function<int(int32_t retval_index)> &output) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(input);
  GE_CHECK_NOTNULL(output);
  if (MappingSubgraphInput(graph, input) != SUCCESS) {
    GELOGE(FAILED, "[Call][MappingSubgraphInput] for graph:%s failed", graph->GetName().c_str());
    return FAILED;
  }

  if (MappingSubgraphOutput(graph, output) != SUCCESS) {
    GELOGE(FAILED, "[Call][MappingSubgraphOutput] for graph:%s failed", graph->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status ParseSubgraphPostFnCase(const std::string &subgraph_name, const ComputeGraphPtr &graph) {
  (void)subgraph_name;
  return MappingSubgraphIndex(graph,
      [](int32_t data_index) { return data_index + 1; },
      [](int32_t retval_index) { return retval_index; });
}

Status ParseSubgraphPostFnIf(const std::string &subgraph_name, const ComputeGraphPtr &graph) {
  (void)subgraph_name;
  return MappingSubgraphIndex(graph,
      [](int32_t data_index) { return data_index + 1; },
      [](int32_t retval_index) { return retval_index; });
}

Status ParseSubgraphPostFnWhile(const std::string &subgraph_name, const ComputeGraphPtr &graph) {
  return MappingSubgraphIndex(graph,
      [](int32_t data_index) { return data_index; },
      [&subgraph_name](int32_t retval_index) { return (subgraph_name == "cond") ? -1 : retval_index; });
}

Status ParseSubgraphPostFnFor(const std::string &subgraph_name, const ComputeGraphPtr &graph) {
  (void)subgraph_name;
  return MappingSubgraphIndex(graph,
      [](int32_t data_index) { return (data_index == 0) ? 0 : data_index + kDataIndexOffset; },
      [](int32_t retval_index) { return retval_index; });
}

Status ParseSubgraphPostFnPartitionedCall(const std::string &subgraph_name, const ComputeGraphPtr &graph) {
  (void)subgraph_name;
  return MappingSubgraphIndex(graph,
      [](int32_t data_index) { return data_index; },
      [](int32_t retval_index) { return retval_index; });
}

Status ParseSubgraphPostFnFlowNode(const std::string &subgraph_name, const ComputeGraphPtr &graph) {
  (void)subgraph_name;
  (void)graph;
  return SUCCESS;
}
}

Status DataPass::PostParseSubgraph(const ComputeGraphPtr &graph, const std::string &ir_name,
                                   const NodePtr &parent_node) const {
  using ParseSubgraphFunc = std::function<Status(const std::string &subgraph_name, const ComputeGraphPtr &graph)>;
  const static std::map<std::string, ParseSubgraphFunc> subgraph_handle = {
      {FOR, ParseSubgraphPostFnFor},
      {CASE, ParseSubgraphPostFnCase},
      {STATELESSCASE, ParseSubgraphPostFnCase},
      {IF, ParseSubgraphPostFnIf},
      {_IF, ParseSubgraphPostFnIf},
      {STATELESSIF, ParseSubgraphPostFnIf},
      {WHILE, ParseSubgraphPostFnWhile},
      {_WHILE, ParseSubgraphPostFnWhile},
      {STATELESSWHILE, ParseSubgraphPostFnWhile},
      {PARTITIONEDCALL, ParseSubgraphPostFnPartitionedCall},
      {STATEFULPARTITIONEDCALL, ParseSubgraphPostFnPartitionedCall},
      {FLOWNODE, ParseSubgraphPostFnFlowNode}};

  auto post_func_it = subgraph_handle.find(parent_node->GetType());
  if (post_func_it == subgraph_handle.end()) {
    REPORT_INNER_ERR_MSG("E19999", "The subgraph post func for node %s type %s is null, check invalid",
                       parent_node->GetName().c_str(), parent_node->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] The subgraph post func for node %s type %s is null.",
           parent_node->GetName().c_str(), parent_node->GetType().c_str());
    return FAILED;
  }

  if (post_func_it->second(ir_name, graph) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Post process subgraph %s on node %s type %s failed",
                       graph->GetName().c_str(), parent_node->GetName().c_str(), parent_node->GetType().c_str());
    GELOGE(FAILED, "[Call][PostFunc] Failed to post process subgraph %s on node %s type %s",
           graph->GetName().c_str(), parent_node->GetName().c_str(), parent_node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status DataPass::Run(ComputeGraphPtr compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);
  if (compute_graph->GetParentNode() == nullptr) {      // for subgraph post process.
    return SUCCESS;
  }

  for (const NodePtr &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetType() == DATA) {
      uint32_t parent_index = 0;
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
        break;        // parent_index not set, Graph from IR.
      }

      return SUCCESS; // Graph from Parser.
    }
  }

  std::string subgraph_name;
  const auto &parent_node = compute_graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node->GetOpDesc());
  auto func_desc = parent_node->GetOpDesc();
  GE_CHK_STATUS_RET(func_desc->GetSubgraphNameByInstanceName(compute_graph->GetName(), subgraph_name),
                    "[Get][SubGraphName] for Graph:%s failed.", compute_graph->GetName().c_str());

  GELOGI("Post process for subgraph %s, Subgraph name: %s, Parent name: %s, Parent type: %s.",
         compute_graph->GetName().c_str(), subgraph_name.c_str(), parent_node->GetName().c_str(),
         parent_node->GetType().c_str());

  const auto &parent_graph = compute_graph->GetParentGraph();
  GE_CHECK_NOTNULL(parent_graph);
  for (const NodePtr &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if ((node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2) || (node->GetType() == NETOUTPUT)) {
      continue;
    }

    node->GetOpDesc()->SetName(parent_node->GetName() + "_" + compute_graph->GetName() + "/" + node->GetName());
  }

  return PostParseSubgraph(compute_graph, subgraph_name, parent_node);
}

REG_PASS_OPTION("DataPass").LEVELS(OoLevel::kO0);
}  // namespace ge
