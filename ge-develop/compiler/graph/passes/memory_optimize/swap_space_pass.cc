/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/memory_optimize/swap_space_pass.h"

#include "graph/utils/op_desc_utils.h"
#include "graph/ge_context.h"
#include "common/checker.h"

#define REQUIRE(cond, ...)                       \
  do {                                           \
    if (!(cond)) {                               \
      REPORT_INNER_ERR_MSG("E19999", __VA_ARGS__); \
      GELOGE(FAILED, "[SWAP]" __VA_ARGS__);      \
      return FAILED;                             \
    }                                            \
  } while (false)

#define REQUIRE_OR_DUMP(cond, ...)               \
  do {                                           \
    if (!(cond)) {                               \
      GE_DUMP(graph, "SwapFailed");              \
      REPORT_INNER_ERR_MSG("E19999", __VA_ARGS__); \
      GELOGE(FAILED, "[SWAP]" __VA_ARGS__);      \
      return FAILED;                             \
    }                                            \
  } while (false)

#define REQUIRE_NOT_NULL(cond, ...) REQUIRE(((cond) != nullptr), __VA_ARGS__)
#define REQUIRE_SUCCESS(cond, ...) REQUIRE(((cond) == SUCCESS), __VA_ARGS__)
#define REQUIRE_SUCCESS_OR_DUMP(cond, ...) REQUIRE_OR_DUMP(((cond) == SUCCESS), __VA_ARGS__)

namespace ge {
namespace {
const std::string OPTION_SWAP_SPACE_NODES = "ge.swapSpaceNodes";
const int32_t kOnlyAnchorIndex = 0;

void MakeCopyAndComputeStreamSame(const OpDescPtr &op_desc) {
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_FORCE_ATTACH_STREAM, true);
}

void SetCopyKind(const OpDescPtr &mem_copy_op, const rtMemcpyKind_t kind) {
  (void)AttrUtils::SetInt(mem_copy_op, ATTR_NAME_RT_MEMCPY_KIND, kind);
}

bool CheckDigitStr(const std::string &str) {
  for (const auto &c : str) {
    if (!isdigit(c)) {
      REPORT_INNER_ERR_MSG("E19999", "param str:%s is not positive integer", str.c_str());
      GELOGE(FAILED, "[Check][Param] Value[%s] is not positive integer", str.c_str());
      return false;
    }
  }
  return true;
}
// Don't go through branches, since we don't know whether they'll be executed or not.
bool CouldBeAsTrigger(const NodePtr &node) {
  std::string stream_label;
  bool get_attr = AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);
  return !get_attr || stream_label.empty();
}

bool HasSecondPath(const NodePtr &src_node, const NodePtr &dst_node) {
  std::vector<NodePtr> temp_stack;
  std::set<std::string> visited;
  temp_stack.push_back(dst_node);
  while (!temp_stack.empty()) {
    const auto &node = temp_stack.back();
    temp_stack.pop_back();
    if (!visited.insert(node->GetName()).second) {
      continue;
    }
    for (const auto &out_node : node->GetOutAllNodes()) {
      if (out_node->GetName() == src_node->GetName()) {
        return true;
      }
      temp_stack.push_back(out_node);
    }
  }
  return false;
}

NodePtr FindSwapInTrigger(const NodePtr &swap_in_node) {
  NodePtr trigger_node = nullptr;
  int64_t trigger_node_index = 0;
  for (const auto &node : swap_in_node->GetOutAllNodes()) {
    for (const auto &input_node : node->GetInAllNodes()) {
      const auto execute_index = input_node->GetOpDesc()->GetId();
      if ((input_node != swap_in_node) && (CouldBeAsTrigger(input_node)) &&
          (!HasSecondPath(input_node, swap_in_node)) && (execute_index > trigger_node_index)) {
        trigger_node = input_node;
        trigger_node_index = execute_index;
      }
    }
  }
  return trigger_node;
}
}  // namespace

Status SwapSpacePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if ((graph->GetParentGraph() != nullptr) || (graph->GetGraphUnknownFlag())) {
    GELOGI("Subgraph or unknown shape root graph is not supported for now, graph_name:[%s]", graph->GetName().c_str());
    return SUCCESS;
  }
  // Parser option to figure out what needs to be swapped;
  std::map<NodePtr, SwapInfo> swapping_candidates;
  REQUIRE_SUCCESS_OR_DUMP(GetAllSwapCandidates(graph, swapping_candidates),
                          "[SWAP][Get] swap_space_nodes from graph failed, graph_name:[%s]", graph->GetName().c_str());
  if (swapping_candidates.empty()) {
    // Nothing to do
    GELOGI("There are no swap space nodes given, just skip, graph_name:[%s]", graph->GetName().c_str());
    return SUCCESS;
  }
  GELOGI("Graph %s start to swap.", graph->GetName().c_str());
  REQUIRE_SUCCESS(graph->TopologicalSorting(), "[SWAP][TOPO] failed, graph_name:[%s]", graph->GetName().c_str());
  GE_DUMP(graph, "BeforeSwapSpace");
  REQUIRE_SUCCESS_OR_DUMP(InsertSpaceCopyNodes(graph, swapping_candidates),
                          "[SWAP][Add] swap in/out nodes for graph failed, graph_name:[%s]", graph->GetName().c_str());
  GE_DUMP(graph, "AfterSwapSpace");
  return SUCCESS;
}

Status SwapSpacePass::GetAllSwapCandidates(const ComputeGraphPtr &graph,
                                           std::map<NodePtr, SwapInfo> &swapping_candidates) const {
  std::string swap_space_nodes;
  if ((GetContext().GetOption(OPTION_SWAP_SPACE_NODES, swap_space_nodes) != GRAPH_SUCCESS) ||
      swap_space_nodes.empty()) {
    return SUCCESS;
  }
  GELOGD("Get option %s, value %s", OPTION_SWAP_SPACE_NODES.c_str(), swap_space_nodes.c_str());
  for (const auto &item : StringUtils::Split(swap_space_nodes, ';')) {
    std::vector<std::string> node_and_input_index = StringUtils::Split(item, ':');
    REQUIRE(node_and_input_index.size() == 2U,  // node_and_input_index, the size must be 2
            "[SWAP][Check] option %s is invalid, correct format is like node_name1:0;node_name1:2; node_name2:0",
            item.c_str());
    REQUIRE(CheckDigitStr(node_and_input_index.back()),
            "[SWAP][Check] option %s is invalid, correct format is like node_name1:0;node_name1:2; node_name2:0",
            item.c_str());
    const auto &node_name = node_and_input_index.front();
    auto node = graph->FindNode(node_name);
    if (node == nullptr) {
      for (const auto &subgraph : graph->GetAllSubgraphs()) {
        node = subgraph->FindNode(node_name);
        if (node != nullptr) {
          break;
        }
      }
    }
    REQUIRE_NOT_NULL(node, "[SWAP][Get] invalid swap_space_nodes option %s, which %s does not exist in graph %s",
                     swap_space_nodes.c_str(), node_name.c_str(), graph->GetName().c_str());
    REQUIRE_SUCCESS(CheckNodeCouldSwap(node), "[SWAP][Check]Node %s need assign special memory, which can not swap out",
                    node_name.c_str());
    int input_index = 0;
    REQUIRE_SUCCESS(ConvertToInt32(StringUtils::Trim(node_and_input_index.back()), input_index),
                    "[SWAP][Check]Node %s's input index %s is invalid", node_name.c_str(),
                    node_and_input_index.back().c_str());
    const auto &in_data_anchor = node->GetInDataAnchor(input_index);
    REQUIRE_NOT_NULL(in_data_anchor,
                     "[SWAP][Get] invalid swap_space_nodes option %s, which %s does not have %d input index",
                     swap_space_nodes.c_str(), node_name.c_str(), input_index);
    const auto &src_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_data_anchor);
    const auto &src_node = src_out_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    SwapInfo &swap_info = swapping_candidates[src_node];
    swap_info.node = src_node->GetName();
    std::vector<string> &consumer_nodes = swap_info.output_to_swaps[src_out_data_anchor->GetIdx()];
    consumer_nodes.push_back(node_name);
  }
  return SUCCESS;
}

/*
input graph: with swapping data edge Node0-->Node3
                +-----------------------------------------+
               |                                         v
+------+     +-------+     +-------+     +-------+     +-------+     +-----------+
| Data | --> | Node0 | --> | Node1 | --> | Node2 | --> | Node3 | --> | NetOutput |
+------+     +-------+     +-------+     +-------+     +-------+     +-----------+

output graph: Node0's out tensor is swapped out before node1's execution(with D2H==>Node1 control edge); Before
Node3's execution, tensor is swapped in, then Node2 can reuse Node0's output tensor
               +-------------------------+             +-------------------------+
               |                         v             |                         v
+------+     +-------+     +-----+     +-------+     +-------+     +-----+     +-------+     +-----------+
| Data | --> | Node0 | --> | D2H | ==> | Node1 | --> | Node2 | ==> | H2D | --> | Node3 | --> | NetOutput |
+------+     +-------+     +-----+     +-------+     +-------+     +-----+     +-------+     +-----------+
                             |                                       ^
                             +---------------------------------------+
*/
Status SwapSpacePass::InsertSpaceCopyNodes(const ComputeGraphPtr &graph,
                                           const std::map<NodePtr, SwapInfo> &swapping_candidates) {
  std::vector<NodePtr> copy_from_device_to_host_nodes;
  for (const auto &swapping_candidate : swapping_candidates) {
    REQUIRE_SUCCESS(SwapOutProcess(swapping_candidate, copy_from_device_to_host_nodes),
                    "[SWAP][Insert] d2h node failed for swap out node %s", swapping_candidate.first->GetName().c_str());
  }
  for (auto &d2h_node : copy_from_device_to_host_nodes) {
    REQUIRE_SUCCESS(SwapInProcess(graph, d2h_node), "[SWAP][Insert] h2d node failed for swap in node %s",
                    d2h_node->GetName().c_str());
  }
  return SUCCESS;
}

Status SwapSpacePass::PrepareForMemAssign(const NodePtr &mem_copy_node, const rtMemcpyKind_t rt_memcpy_kind) {
  GE_CHECK_NOTNULL(mem_copy_node);
  const auto &mem_copy_op_desc = mem_copy_node->GetOpDesc();
  GE_CHECK_NOTNULL(mem_copy_op_desc);
  std::vector<int64_t> host_memory_types{RT_MEMORY_HOST};
  if (rt_memcpy_kind == RT_MEMCPY_DEVICE_TO_HOST) {
    (void)ge::AttrUtils::SetListInt(mem_copy_op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, host_memory_types);
  } else {
    (void)ge::AttrUtils::SetListInt(mem_copy_op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, host_memory_types);
  }
  return SUCCESS;
}

Status SwapSpacePass::CheckNodeCouldSwap(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool is_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, is_continuous);
  REQUIRE(!is_continuous, "[SWAP][CHECK] node %s 's input need to be continuous, which cannot swap",
          op_desc->GetName().c_str());
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_continuous);
  REQUIRE(!is_continuous, "[SWAP][CHECK] node %s 's input need to be continuous, which cannot swap",
          op_desc->GetName().c_str());
  return SUCCESS;
}

Status SwapSpacePass::SwapOutProcess(const std::pair<NodePtr, SwapInfo> &swapping_candidate,
                                     std::vector<NodePtr> &d2h_mem_cpy_nodes) const {
  const auto &node = swapping_candidate.first;
  const auto &swap_info = swapping_candidate.second;
  for (auto &out_anchor : node->GetAllOutDataAnchors()) {
    const auto &iter = swap_info.output_to_swaps.find(out_anchor->GetIdx());
    if (iter == swap_info.output_to_swaps.cend()) {
      continue;
    }
    OpDescBuilder op_desc_builder(node->GetName() + "_out_" + std::to_string(out_anchor->GetIdx()) + "_memcpy_async",
                                  MEMCPYASYNC);
    const auto &mem_copy_op = op_desc_builder.AddInput("x", node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx()))
                                  .AddOutput("y", node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx()))
                                  .Build();
    const auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
    REQUIRE(!peer_in_anchors.empty(), "[SWAP][Insert] node failed; node %s has no output data nodes for [%d]th anchor ",
            node->GetName().c_str(), out_anchor->GetIdx());
    std::vector<InDataAnchorPtr> swap_out_consumer_anchors;
    std::vector<InDataAnchorPtr> other_anchors;
    for (const auto &in_data_anchor : out_anchor->GetPeerInDataAnchors()) {
      const std::vector<string> &swap_out_consumer_nodes = iter->second;
      if (std::find(swap_out_consumer_nodes.cbegin(), swap_out_consumer_nodes.cend(),
                    in_data_anchor->GetOwnerNode()->GetName()) != swap_out_consumer_nodes.cend()) {
        swap_out_consumer_anchors.push_back(in_data_anchor);
      } else {
        other_anchors.push_back(in_data_anchor);
      }
    }
    // add swap out edge
    const auto &mem_copy_node = GraphUtils::InsertNodeAfter(out_anchor,
                                                            swap_out_consumer_anchors,
                                                            mem_copy_op);
    GE_ASSERT_NOTNULL(mem_copy_node);
    // Make sure the tensor is swapped out quickly: look for nodes that
    // will execute after the tensor is generated and add a control
    // dependency from the swap out node to those nodes.
    for (size_t index = 0U; index < other_anchors.size(); ++index) {
      const auto &other_consumer_node = peer_in_anchors.at(index)->GetOwnerNode();
      GE_CHECK_NOTNULL(other_consumer_node);
      REQUIRE_SUCCESS(
          GraphUtils::AddEdge(mem_copy_node->GetOutControlAnchor(), other_consumer_node->GetInControlAnchor()),
          "[SWAP][AddCtrlEdge] between %s and %s failed", mem_copy_node->GetName().c_str(),
          other_consumer_node->GetName().c_str());
      // Try to ensure other consumer node has same stream with it's input node and it's peer copy node:
      // which is needed for swap out tensor's memory to be reused
      MakeCopyAndComputeStreamSame(other_consumer_node->GetOpDesc());
    }
    // Try to ensure d2h node has same stream with it's input node and it's peer nodes:
    // which is needed for swap out tensor's memory to be reused
    MakeCopyAndComputeStreamSame(mem_copy_op);
    SetCopyKind(mem_copy_op, RT_MEMCPY_DEVICE_TO_HOST);
    REQUIRE_SUCCESS(PrepareForMemAssign(mem_copy_node, RT_MEMCPY_DEVICE_TO_HOST),
                    "[SWAP][PrepareForMemAssign] failed for node %s", mem_copy_node->GetName().c_str());
    d2h_mem_cpy_nodes.push_back(mem_copy_node);
  }
  return SUCCESS;
}

Status SwapSpacePass::SwapInProcess(const ComputeGraphPtr &graph, const NodePtr &node) {
  for (auto &out_anchor : node->GetAllOutDataAnchors()) {
    OpDescBuilder op_desc_builder(node->GetName() + "_memcpy_async", MEMCPYASYNC);
    const auto &mem_copy_op = op_desc_builder.AddInput("x", node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx()))
                                  .AddOutput("y", node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx()))
                                  .Build();
    const auto &mem_copy_node = graph->InsertNode(node, mem_copy_op);
    GE_CHECK_NOTNULL(mem_copy_node);
    const auto &peer_in_data_anchors = out_anchor->GetPeerInDataAnchors();
    for (const auto &peer_in_data_anchor : peer_in_data_anchors) {
      const auto &in_data_anchor = mem_copy_node->GetInDataAnchor(kOnlyAnchorIndex);
      GE_CHECK_NOTNULL(in_data_anchor);
      if (in_data_anchor->IsLinkedWith(out_anchor)) {
        // Add edge repeatedly is not supported, so remove before insert
        REQUIRE_SUCCESS(GraphUtils::RemoveEdge(in_data_anchor, out_anchor),
                        "[SWAP][RemoveEdge] between %s and %s failed", node->GetName().c_str(),
                        mem_copy_node->GetName().c_str());
      }
      REQUIRE_SUCCESS(GraphUtils::InsertNodeBefore(peer_in_data_anchor, mem_copy_node),
                      "[SWAP][Insert] copy node failed for %s", node->GetName().c_str());
    }

    // Make sure the tensor isn't swapped back immediately: look for node that
    // will execute just before we need to swap the data back, and add a control
    // dependency from that node to the swap node.
    const auto &in_trigger_node = FindSwapInTrigger(mem_copy_node);
    if ((in_trigger_node != nullptr) &&
        (!in_trigger_node->GetOutControlAnchor()->IsLinkedWith(mem_copy_node->GetInControlAnchor()))) {
      REQUIRE_SUCCESS(GraphUtils::AddEdge(in_trigger_node->GetOutControlAnchor(), mem_copy_node->GetInControlAnchor()),
                      "[SWAP][AddCtrlEdge] between %s and %s failed", in_trigger_node->GetName().c_str(),
                      mem_copy_node->GetName().c_str());
    }
    // Ensure h2d node has same stream with it's input d2h node: which is needed for h2d out tensor's memory to be
    // reused
    MakeCopyAndComputeStreamSame(mem_copy_op);
    SetCopyKind(mem_copy_op, RT_MEMCPY_HOST_TO_DEVICE);
    REQUIRE_SUCCESS(PrepareForMemAssign(mem_copy_node, RT_MEMCPY_HOST_TO_DEVICE),
                    "[SWAP][PrepareForMemAssign] failed for node %s", mem_copy_node->GetName().c_str());
  }
  return SUCCESS;
}

REG_PASS_OPTION("SwapSpacePass").LEVELS(OoLevel::kO3);
}  // namespace ge
