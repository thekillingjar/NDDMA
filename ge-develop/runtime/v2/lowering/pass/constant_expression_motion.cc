/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "constant_expression_motion.h"

#include <queue>
#include <stack>
#include <utility>
#include <map>

#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/lowering/builtin_node_types.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"

#include "runtime/model_v2_executor.h"
#include "core/builder/node_types.h"
#include "utils/nodes_collector.h"
#include "utils/inner_data_remover.h"

namespace gert {
namespace bg {
namespace {
enum class CeStartNodeType {
  kConst,
  kInnerDataFromInit,
  kEquivalantNodeWithInit,  // like allocator...
  kNotStart
};

struct CeNode {
  CeNode() = default;
  explicit CeNode(ge::FastNode *const n) : node(n) {}
  CeNode(ge::FastNode *const n, CeStartNodeType t) : node(n), start_type(t) {}

  ge::FastNode *node = nullptr;
  CeStartNodeType start_type = CeStartNodeType::kNotStart;
};

struct CeNodes {
  std::vector<ge::FastNode *> ce_nodes;
  std::unordered_set<ge::FastNode *> all_nodes_set;

  bool Append(const CeNode &ce_node) {
    auto inserted = all_nodes_set.insert(ce_node.node).second;
    if (inserted) {
      if ((ce_node.start_type == CeStartNodeType::kConst) || (ce_node.start_type == CeStartNodeType::kNotStart)) {
        ce_nodes.emplace_back(ce_node.node);
      }
    }
    return inserted;
  }
  bool Append(ge::FastNode *const node) {
    return Append({node, CeStartNodeType::kNotStart});
  }

  const std::unordered_set<ge::FastNode *> &GetAllCollectedSet() const {
    return all_nodes_set;
  }
};

struct CeGuarderNode {
  CeGuarderNode() = default;
  explicit CeGuarderNode(ge::FastNode *const n) : node(n) {}

  ge::FastNode *node = nullptr;
};

struct CeGuarderNodes {
  std::vector<ge::FastNode *> guarder_nodes;
  std::unordered_set<ge::FastNode *> all_nodes_set;

  bool Append(const CeGuarderNode &node) {
    auto inserted = all_nodes_set.insert(node.node).second;
    if (inserted) {
      guarder_nodes.push_back(node.node);
    }
    return inserted;
  }
  bool Append(ge::FastNode *const node) {
    return Append(CeGuarderNode{node});
  }
  const std::unordered_set<ge::FastNode *> &GetAllCollectedSet() const {
    return all_nodes_set;
  }
};

struct InEdge {
  ge::EdgeDstEndpoint dst_endpoint;
  ge::FastNode *inner_data;
  ge::EdgeSrcEndpoint src_endpoint_on_init;
};

struct OutEdge {
  ge::EdgeSrcEndpoint src_endpoint;
  ge::ExecuteGraph *subgraph;
  std::vector<ge::EdgeDstEndpoint> peer_in_guarders;
  std::vector<ge::EdgeDstEndpoint> peer_in_remains;
};

bool IsStateful(const ge::FastNode *const node) {
  // todo 当前还不支持stateful机制
  (void)node;
  return false;
}

// todo IsConnectedToNetOutput不应该存在，正式方案在零拷贝需求中完成
//   对于控制算子的场景，应该在load方案中完成
bool IsConnectedToNetOutput(const ge::FastNode *const node) {
  for (const auto dst_node : node->GetOutDataNodes()) {
    if (IsOutputType(dst_node->GetTypePtr()) || IsInnerOutput(dst_node->GetTypePtr())) {
      return true;
    }
  }
  return false;
}

bool IsNodeHasSubgraph(const ge::FastNode *const node) {
  return !node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty();
}

bool IsStart(ge::FastNode *const node, size_t variant_num, ConstantExpressionMotion &cem, CeNode &ce_node) {
  const auto node_type = node->GetTypePtr();
  if (IsConstType(node_type)) {
    ce_node = {node, CeStartNodeType::kConst};
    return true;
  }
  if (IsInnerDataType(node_type)) {
    auto inner_datas_to_init_out = cem.GetSrcFromInitGraph(node, variant_num);
    if (inner_datas_to_init_out.node == nullptr) {
      return false;
    }

    ce_node = {node, CeStartNodeType::kInnerDataFromInit};
    return true;
  }
  // 判断该节点是否在init图上有等价节点
  if (cem.GetEquivalantNodesInInit(node) != nullptr) {
    ce_node = {node, CeStartNodeType::kEquivalantNodeWithInit};
    return true;
  }
  return false;
}

bool IsStop(const ge::FastNode *const node) {
  const auto node_type = node->GetTypePtr();
  return IsStateful(node) || IsOutputType(node_type) || IsInnerOutput(node_type) || IsAllocNode(node_type) ||
         IsConnectedToNetOutput(node) || IsNodeHasSubgraph(node);
}

ge::graphStatus CollectCeAndInnerData(ConstantExpressionMotion &cem, ge::ExecuteGraph *const graph, CeNodes &ce_nodes) {
  const auto parent_node = graph->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_node);
  size_t loop_variant_range = std::numeric_limits<size_t>::max();
  if (IsWhileType(parent_node->GetTypePtr())) {
    loop_variant_range = parent_node->GetDataOutNum();
  }
  /*
   * todo: ce-node带有guarder，而guarder的数据输入中，除了ce-node本身，还有其他node，此时其他的输入node有两种可能：
   *   1. 其他的输入node不是ce-node，此时guarder无法被提取到subgraph外部，
   *      因此受到guarder影响，已经被识别成ce-node也无法被提取
   *   2. 其他的输入node都是InnerData（来自于子图外部），此时guarder可以被提取到subgraph外部
   *   这里需要处理这两种场景，对于第一种要停止collect动作；对于第二种，在重连函数中，需要将guarder的输入换成父图中对应的node。
   *   当前还不支持这种场景（当前的guarder都没有额外输入），因此当前若遇到了这种场景，在guarder重连时会报错
   *   3. node is guarder of other ce node.
   */
  auto ret = CollectNodes<CeNode, CeNodes>(
      graph, ce_nodes,
      [&loop_variant_range, &cem](ge::FastNode *const node, CeNode &ce_node) {
        return IsStart(node, loop_variant_range, cem, ce_node);
      },
      [](ge::FastNode *const node) { return IsStop(node); });
  GE_ASSERT_SUCCESS(ret);
  return ge::GRAPH_SUCCESS;
}

bool IsCeGuarderStart(ge::FastNode *const node, const std::unordered_set<ge::FastNode *> &init_nodes,
                      CeGuarderNode &guarder) {
  if (init_nodes.count(node) > 0U) {
    return false;
  }
  int32_t release_index;
  if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kReleaseResourceIndex, release_index)) {
    return false;
  }
  auto in_node = ge::FastNodeUtils::GetInDataNodeByIndex(node, release_index);
  if (in_node == nullptr) {
    GELOGW("Release node %s index %d can not find the input node.", node->GetName().c_str(), release_index);
    return false;
  }
  if (init_nodes.count(in_node) == 0U) {
    return false;
  }
  if (node->GetAllOutDataEdgesSize() > 0) {
    return false;
  }

  guarder.node = node;
  return true;
}

ge::graphStatus CollectCeGuarders(const ge::ExecuteGraph *const graph, const CeNodes &ce_nodes,
                                  CeGuarderNodes &ce_guarders) {
  return CollectNodes<CeGuarderNode, CeGuarderNodes>(
      graph, ce_guarders,
      [&ce_nodes](ge::FastNode *const node, CeGuarderNode &c_node) {
        return IsCeGuarderStart(node, ce_nodes.GetAllCollectedSet(), c_node);
      },
      [](ge::FastNode *const node) {
        return IsStateful(node) || IsOutputType(node->GetTypePtr()) || IsInnerOutput(node->GetTypePtr());
      });
}

ge::graphStatus CollectInputEdges(const ConstantExpressionMotion &cem, const ge::FastNode *const ce_node,
                                  std::vector<InEdge> &in_edges) {
  for (const auto in_data_edge : ce_node->GetAllInDataEdgesRef()) {
    if (in_data_edge == nullptr) {
      continue;
    }
    auto dst_endpoint = ge::FastNodeUtils::GetDstEndpoint(in_data_edge);
    auto src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(in_data_edge);
    auto src_node = src_endpoint.node;
    GE_ASSERT_NOTNULL(src_node);

    if (IsInnerDataType(src_node->GetTypePtr())) {
      in_edges.push_back({dst_endpoint, src_node, cem.GetInitOut(src_node)});
      continue;
    }

    const auto node_from_init = cem.GetEquivalantNodesInInit(src_node);
    if (node_from_init != nullptr) {
      in_edges.push_back({dst_endpoint, src_node, {node_from_init, src_endpoint.index}});
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CollectOutputEdges(const ConstantExpressionMotion &cem, const CeNodes &ce_nodes,
                                   ge::FastNode *const ce_node, const CeGuarderNodes &ce_guarders,
                                   std::vector<OutEdge> &out_edges) {
  const auto &out_data_edges_ref = ce_node->GetAllOutDataEdgesRef();
  auto ce_owner_graph = ce_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(ce_owner_graph);
  for (int32_t out_idx = 0; static_cast<size_t>(out_idx) < out_data_edges_ref.size(); ++out_idx) {
    OutEdge out_edge = {{ce_node, out_idx}, ce_owner_graph, {}, {}};
    for (const auto edge : out_data_edges_ref.at(out_idx)) {
      if (edge == nullptr) {
        continue;
      }
      auto dst_endpoint = ge::FastNodeUtils::GetDstEndpoint(edge);
      auto dst_node = dst_endpoint.node;
      GE_ASSERT_NOTNULL(dst_node);
      if ((ce_nodes.GetAllCollectedSet().count(dst_node) > 0U) && (cem.GetEquivalantNodesInInit(dst_node) == nullptr)) {
        continue;
      }
      if (ce_guarders.GetAllCollectedSet().count(dst_node) > 0U) {
        out_edge.peer_in_guarders.emplace_back(dst_endpoint);
      } else {
        out_edge.peer_in_remains.emplace_back(dst_endpoint);
      }
    }

    if (!out_edge.peer_in_remains.empty() || !out_edge.peer_in_guarders.empty()) {
      out_edges.emplace_back(std::move(out_edge));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CollectIoEdges(const ConstantExpressionMotion &cem, const CeNodes &ce_nodes,
                               const CeGuarderNodes &ce_guarders, std::vector<InEdge> &in_edges,
                               std::vector<OutEdge> &out_edges) {
  for (const auto ce_node : ce_nodes.ce_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(CollectInputEdges(cem, ce_node, in_edges));
    GE_ASSERT_GRAPH_SUCCESS(CollectOutputEdges(cem, ce_nodes, ce_node, ce_guarders, out_edges));
  }

  return ge::GRAPH_SUCCESS;
}

ge::FastNode *CreateInnerData(ge::ExecuteGraph *const graph, int64_t index) {
  static std::atomic<uint64_t> cem_index{0U};
  std::string name;
  name.append("CEM_").append(graph->GetName()).append("_InnerData_").append(std::to_string(cem_index.fetch_add(1U)));
  auto op_desc = ge::MakeShared<ge::OpDesc>(name, kInnerData);
  if (op_desc == nullptr) {
    return nullptr;
  }
  if (!ge::AttrUtils::SetInt(op_desc, "index", index)) {
    return nullptr;
  }
  if (op_desc->AddOutputDesc("y", ge::GeTensorDesc()) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  return graph->AddNode(op_desc);
}

ge::graphStatus UnlinkInDataEdgeFromEndpoint(const ge::EdgeDstEndpoint &endpoint) {
  const auto in_data_edge = endpoint.node->GetInDataEdgeByIndex(endpoint.index);
  auto owner_graph = endpoint.node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_GRAPH_SUCCESS(owner_graph->RemoveEdge(in_data_edge));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnlinkDataEdges(const std::vector<InEdge> &in_edges, std::vector<OutEdge> &out_edges) {
  // 通过 DstEndpoint 删除对应位置输入边
  for (const auto &in_edge : in_edges) {
    auto in_data_edge = in_edge.dst_endpoint.node->GetInDataEdgeByIndex(in_edge.dst_endpoint.index);
    GE_ASSERT_NOTNULL(in_data_edge);
    GE_ASSERT_GRAPH_SUCCESS(UnlinkInDataEdgeFromEndpoint(ge::FastNodeUtils::GetDstEndpoint(in_data_edge)));
  }

  for (const auto &out_edge : out_edges) {
    for (const auto &dst_endpoint : out_edge.peer_in_remains) {
      GE_ASSERT_SUCCESS(UnlinkInDataEdgeFromEndpoint(dst_endpoint));
    }
    for (const auto &dst_endpoint : out_edge.peer_in_guarders) {
      GE_ASSERT_SUCCESS(UnlinkInDataEdgeFromEndpoint(dst_endpoint));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnlinkInCtrlEdges(const std::vector<ge::FastNode *> &nodes,
                                  const std::unordered_set<ge::FastNode *> &scope_nodes) {
  for (const auto node : nodes) {
    auto owner_graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    for (const auto edge : node->GetAllInControlEdgesRef()) {
      if (edge != nullptr) {
        auto src_node = edge->src;
        if (scope_nodes.count(src_node) > 0U) {
          continue;
        }
        GELOGW("Found expected ctrl edge from node %s to node %s, unlink it.", src_node->GetNamePtr(),
               node->GetNamePtr());
        GE_ASSERT_GRAPH_SUCCESS(owner_graph->RemoveEdge(edge));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnlinkOutCtrlEdges(const std::vector<ge::FastNode *> &nodes,
                                   const std::unordered_set<ge::FastNode *> &scope_nodes) {
  for (const auto node : nodes) {
    auto owner_graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    for (const auto edge : node->GetAllOutControlEdgesRef()) {
      if (edge != nullptr) {
        const auto dst_node = edge->dst;
        if (scope_nodes.count(dst_node) > 0U) {
          continue;
        }
        GELOGW("Found expected ctrl edge from node %s to node %s, unlink it.", node->GetNamePtr(),
               dst_node->GetNamePtr());
        GE_ASSERT_GRAPH_SUCCESS(owner_graph->RemoveEdge(edge));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoveCeToInit(ge::ExecuteGraph *const init_graph, const CeNodes &ce_nodes,
                             std::vector<InEdge> &in_edges) {
  if (!ce_nodes.ce_nodes.empty()) {
    auto owner_graph = ce_nodes.ce_nodes[0U]->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    for (const auto node : ce_nodes.ce_nodes) {
      GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(owner_graph, node));
    }
  }
  for (const auto ce_node : ce_nodes.ce_nodes) {
    GELOGD("Move CE node %s type %s to init graph", ce_node->GetNamePtr(), ce_node->GetTypePtr());
    GE_ASSERT_NOTNULL(init_graph->AddNode(ce_node));
  }
  for (const auto &in_edge : in_edges) {
    auto owner_graph = in_edge.dst_endpoint.node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    GE_ASSERT_NOTNULL(owner_graph->AddEdge(in_edge.src_endpoint_on_init.node, in_edge.src_endpoint_on_init.index,
                                           in_edge.dst_endpoint.node, in_edge.dst_endpoint.index));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoveGuardersToDeInit(ge::ExecuteGraph *const de_init_graph, const CeGuarderNodes &guarders) {
  GE_ASSERT_NOTNULL(de_init_graph);
  if (!guarders.guarder_nodes.empty()) {
    auto owner_graph = guarders.guarder_nodes[0U]->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    for (const auto node : guarders.guarder_nodes) {
      GE_ASSERT_GRAPH_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(owner_graph, node));
    }
  }
  for (const auto &guarder : guarders.guarder_nodes) {
    GELOGD("Move guarder node %s type %s to de-init graph", guarder->GetNamePtr(), guarder->GetTypePtr());
    GE_ASSERT_NOTNULL(de_init_graph->AddNode(guarder));

    // check guarders out ctrl nodes
    for (const auto &out_node : guarder->GetOutControlNodes()) {
      if (guarders.GetAllCollectedSet().count(out_node) == 0U) {
        GELOGE(ge::FAILED,
               "Failed to do CEM, the guarder %s moving to DeInit graph has out ctrl node %s which belongs to Main",
               guarder->GetName().c_str(), out_node->GetName().c_str());
        return ge::GRAPH_FAILED;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool NeedCem(const ge::ExecuteGraph *const subgraph) {
  auto parent_node = subgraph->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_node);

  auto parent_node_type = parent_node->GetTypePtr();
  // If/Case/While的第0张图是control frame图，是固定结构的，不参与本优化
  if (IsWhileType(parent_node_type) || IsIfOrCaseType(parent_node_type)) {
    if (parent_node->GetOpDescBarePtr()->GetSubgraphInstanceName(0) == subgraph->GetName()) {
      GELOGD("The subgraph %s is the ctrl frame of node %s, skip CEM", subgraph->GetName().c_str(),
             parent_node->GetNamePtr());
      return false;
    }
  }

  while (strcmp(parent_node_type, GetSubExeGraphTypeStr(kMainExeGraph)) != 0) {
    const auto parent_graph = parent_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(parent_graph);
    parent_node = parent_graph->GetParentNodeBarePtr();
    if (parent_node == nullptr) {
      GELOGD("The subgraph %s is not Main or subgraph of Main, skip CEM", subgraph->GetName().c_str());
      return false;  // 遍历到根图了，说明这个subgraph不是Main图的子图，对于Init和DeInit的子图，不做CEM
    }
    parent_node_type = parent_node->GetTypePtr();
  }

  return true;
}

ge::EdgeDstEndpoint LinkOut(ge::ExecuteGraph *subgraph, std::vector<ge::EdgeDstEndpoint> dst_endpoint) {
  // 为子图创建 InnerData 节点并为其连接输出边至 dst_endpoints
  ge::FastNode *node = nullptr;
  while ((node = subgraph->GetParentNodeBarePtr()) != nullptr) {
    auto index = node->GetDataInNum();
    // 此处 inner_data 的 owner graph 即 subgraph
    auto inner_data = CreateInnerData(subgraph, static_cast<int64_t>(index));
    GE_ASSERT_NOTNULL(inner_data);
    for (const auto &endpoint : dst_endpoint) {
      GE_ASSERT_NOTNULL(subgraph->AddEdge(inner_data, 0, endpoint.node, endpoint.index));
    }

    GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendInputEdgeInfo(node, index + 1U));

    dst_endpoint = {{node, static_cast<int32_t>(index)}};
    subgraph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(subgraph);
  }
  GE_ASSERT_EQ(dst_endpoint.size(), 1U);
  return dst_endpoint[0];
}

ge::graphStatus LinkOutEdges(ConstantExpressionMotion &cem, const std::vector<OutEdge> &out_edges) {
  auto init_node = cem.GetInitNode();
  GE_ASSERT_NOTNULL(init_node);
  auto init_netoutput = cem.GetInitNetOutput();
  GE_ASSERT_NOTNULL(init_netoutput);
  auto init_graph = cem.GetInitGraph();
  GE_ASSERT_NOTNULL(init_graph);
  auto root_graph = init_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(root_graph);
  auto index = init_netoutput->GetDataInNum();
  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendInputEdgeInfo(init_netoutput, index + out_edges.size()));
  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendOutputEdgeInfo(init_node, index + out_edges.size()));

  for (size_t i = 0U; i < out_edges.size(); ++i) {
    auto &out_edge = out_edges[i];
    // 在 init 图中连边
    GE_ASSERT_NOTNULL(
        init_graph->AddEdge(out_edge.src_endpoint.node, out_edge.src_endpoint.index, init_netoutput, index + i));
    // 在 root 图中连边
    if (!out_edge.peer_in_remains.empty()) {
      auto root_dst_endpoint = LinkOut(out_edge.subgraph, out_edge.peer_in_remains);
      GE_ASSERT_NOTNULL(root_dst_endpoint.node);
      GE_ASSERT_NOTNULL(root_graph->AddEdge(init_node, index + i, root_dst_endpoint.node, root_dst_endpoint.index));
    }
    if (!out_edge.peer_in_guarders.empty()) {
      auto root_dst_endpoint = LinkOut(cem.GetDeInitGraph(), out_edge.peer_in_guarders);
      GE_ASSERT_NOTNULL(root_dst_endpoint.node);
      GE_ASSERT_NOTNULL(root_graph->AddEdge(init_node, index + i, root_dst_endpoint.node, root_dst_endpoint.index));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CollectNoOpDstNodes(ConstantExpressionMotion &cem, const std::vector<ge::FastNode *> &no_op_src_nodes,
                                    const std::vector<ge::FastNode *> &ce_nodes,
                                    std::vector<ge::FastNode *> &no_op_dst_nodes) {
  // 源节点为空，不需要收集目的节点了
  if (no_op_src_nodes.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  // 收集dst ce nodes
  for (const auto &node : ce_nodes) {
    if (IsConstType(node->GetTypePtr())) {
      continue;
    }
    // 判断是否是no op的控制边连接的目的节点，只连接ce node的首层计算节点
    bool is_first_layer_ce_node = true;
    for (const auto &in_node : node->GetInDataNodes()) {
      const auto data_type = in_node->GetTypePtr();
      // 连接了除ce start node之外的其他节点说明不是首层节点
      if (!IsInnerDataType(data_type) && !IsConstType(data_type) &&
          (cem.GetEquivalantNodesInInit(in_node) == nullptr)) {
        is_first_layer_ce_node = false;
        break;
      }
    }
    if (is_first_layer_ce_node) {
      no_op_dst_nodes.emplace_back(node);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::FastNode *AddNoOpNodeToInitGraph(ge::ExecuteGraph *const init_graph) {
  std::stringstream name_ss;
  static std::atomic<uint64_t> index{0U};
  name_ss << "CEM_NoOp_" << index++;
  auto op_desc = ge::MakeShared<ge::OpDesc>(name_ss.str(), ge::NOOP);
  GE_ASSERT_NOTNULL(op_desc);
  return init_graph->AddNode(op_desc);
}

ge::graphStatus CollectNoOpSrcNodesFromCeNodes(std::vector<ge::FastNode *> &ce_nodes, std::vector<OutEdge> &out_edges,
                                               std::vector<ge::FastNode *> &no_op_src_nodes) {
  // 收集ce_nodes中连接inner_netoutput的节点
  for (const auto &edge : out_edges) {
    auto node = edge.src_endpoint.node;
    GE_ASSERT_NOTNULL(node);
    if (IsConstType(node->GetTypePtr())) {
      continue;
    }
    no_op_src_nodes.emplace_back(node);
  }
  // 收集ce_nodes中的叶子节点
  for (const auto &ce_node : ce_nodes) {
    if (ce_node->GetAllOutEdgesSize() == 0U) {
      no_op_src_nodes.emplace_back(ce_node);
    }
  }
  return ge::GRAPH_SUCCESS;
}

// 此处入参 no_op_node 同时作为输出
ge::graphStatus AddCtrlFromInitNodesToCeNodes(ge::ExecuteGraph *const init_graph,
                                              const std::vector<ge::FastNode *> &no_op_src_nodes,
                                              const std::vector<ge::FastNode *> &no_op_dst_nodes,
                                              ge::FastNode *&no_op_node) {
  // 如果不存在src或者dst node，则不创建NoOp
  if (no_op_src_nodes.empty() || no_op_dst_nodes.empty()) {
    return ge::GRAPH_SUCCESS;
  }

  // 在init graph中的node到NoOp之间连接控制边,一次cem中只需要连接一次
  if (no_op_node == nullptr) {
    no_op_node = AddNoOpNodeToInitGraph(init_graph);
    GE_ASSERT_NOTNULL(no_op_node);
    for (auto &src_node : no_op_src_nodes) {
      GE_ASSERT_NOTNULL(src_node);
      GE_ASSERT_NOTNULL(init_graph->AddEdge(src_node, ge::kControlEdgeIndex, no_op_node, ge::kControlEdgeIndex));
    }
  }

  // 在NoOp到满足条件的ce nodes之间连接控制边
  for (auto &dst_node : no_op_dst_nodes) {
    GE_ASSERT_NOTNULL(init_graph->AddEdge(no_op_node, ge::kControlEdgeIndex, dst_node, ge::kControlEdgeIndex));
  }
  return ge::GRAPH_SUCCESS;
}

// 获取main图对应init图中的等价节点
void GetEquivalantNodes(const LoweringGlobalData *global_data, EquivalantNodesFromMainToInit &equivalant_nodes) {
  // 获取不同placement下主流L2 allocator的等价节点
  for (int32_t i = 0; i < kTensorPlacementEnd; ++i) {
    AllocatorDesc desc{static_cast<TensorPlacement>(i), AllocatorUsage::kAllocNodeOutput};
    auto init_l2_allocator = global_data->GetInitL2Allocator(desc);
    auto main_l2_allocator = global_data->GetMainL2Allocator(kDefaultMainStreamId, desc);
    if ((init_l2_allocator != nullptr) && (main_l2_allocator != nullptr)) {
      equivalant_nodes.emplace(main_l2_allocator->GetFastNode(), init_l2_allocator->GetFastNode());
    }
  }
}
}  // namespace

ge::graphStatus ConstantExpressionMotion::Init(ge::ExecuteGraph *const root_graph) {
  cached_inner_datas_to_init_out_.clear();
  if (root_graph != root_graph_) {
    cached_init_graph_ = nullptr;
    cached_de_init_graph_ = nullptr;
    cached_netoutput_from_init_ = nullptr;
    cached_init_node_ = nullptr;
    cached_de_init_node_ = nullptr;
    root_graph_ = root_graph;
    GetEquivalantNodes(global_data_, cached_equivalant_nodes_);
    GE_ASSERT_SUCCESS(CollectNoOpSrcNodesFromInitGraph());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstantExpressionMotion::CollectNoOpSrcNodesFromInitGraph() {
  no_op_src_nodes_.clear();
  auto init_node = GetInitNode();
  GE_ASSERT_NOTNULL(init_node);
  auto init_graph = GetInitGraph(init_node);
  GE_ASSERT_NOTNULL(init_graph);

  // 查找叶子节点
  for (const auto &node : init_graph->GetDirectNode()) {
    if (IsInnerOutput(node->GetTypePtr())) {
      cached_netoutput_from_init_ = node;
      continue;
    }
    if (node->GetAllOutEdgesSize() == 0U) {
      no_op_src_nodes_.emplace_back(node);
    }
  }

  GE_ASSERT_NOTNULL(cached_netoutput_from_init_);
  // 查找连接到inner_netoutput的节点，通过输入数据边和控制边获取 src_node
  auto collect_src_nodes = [this](const ge::FastEdge *const in_edge) {
    if (in_edge != nullptr) {
      auto src_node = in_edge->src;
      if (IsConstType(src_node->GetTypePtr()) || IsConstFeedType(src_node->GetTypePtr())) {
        return;
      }
      no_op_src_nodes_.emplace_back(src_node);
    }
  };

  GE_ASSERT_NOTNULL(cached_netoutput_from_init_);
  // 查找连接到inner_netoutput的节点，通过输入数据边和控制边获取 src_node
  for (const auto in_edge : cached_netoutput_from_init_->GetAllInDataEdgesRef()) {
    collect_src_nodes(in_edge);
  }
  for (const auto in_edge : cached_netoutput_from_init_->GetAllInControlEdgesRef()) {
    collect_src_nodes(in_edge);
  }

  return ge::GRAPH_SUCCESS;
}

void ConstantExpressionMotion::UpdateNoOpSrcNodes(ge::FastNode *const no_op_node,
                                                  std::vector<ge::FastNode *> &no_op_src_nodes_next_time) {
  // cem完成后需要更新noop src node，给下一轮cem使用
  // 如果本次创建了noop，代表本次缓存的src nodes需要清空，直接使用本次缓存的no_op_src_nodes_next_time；
  // 如果本次没有创建noop，则有两种可能，第一种no op源节点为空，则也需要使用本次缓存的no_op_src_nodes_next_time，
  // 第二种是本次cem没有计算节点移到到init图，noop目的节点为空，则直接使用上次缓存的no_op_src_nodes_
  if ((no_op_node != nullptr) || no_op_src_nodes_.empty()) {
    no_op_src_nodes_ = std::move(no_op_src_nodes_next_time);
  }
}
ge::FastNode *ConstantExpressionMotion::GetInitNode() {
  if (cached_init_node_ == nullptr) {
    cached_init_node_ =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph_, GetSubExeGraphTypeStr(kInitExeGraph));
  }
  return cached_init_node_;
}

ge::FastNode *ConstantExpressionMotion::GetInitNetOutput() const {
  /** 在 Pass::Run 初始化函数 Init 中保证将 InnerNetOutput 赋值给 cached_netoutput_from_init_
   *  因此，此处无需再通过 FindFirstNodeMatchType 的方式从 init 图中尝试寻找
   */
  return cached_netoutput_from_init_;
}

ge::ExecuteGraph *ConstantExpressionMotion::GetInitGraph() {
  if (cached_init_graph_ == nullptr) {
    auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph_, GetSubExeGraphTypeStr(kInitExeGraph));
    return GetInitGraph(init_node);
  }
  return cached_init_graph_;
}

ge::ExecuteGraph *ConstantExpressionMotion::GetInitGraph(ge::FastNode *const init_node) {
  if (cached_init_graph_ == nullptr) {
    GE_ASSERT_NOTNULL(init_node);
    cached_init_graph_ = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0U);
  }
  return cached_init_graph_;
}

ge::FastNode *ConstantExpressionMotion::GetDeInitNode() {
  if (cached_de_init_node_ == nullptr) {
    cached_de_init_node_ =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph_, GetSubExeGraphTypeStr(kDeInitExeGraph));
  }
  return cached_de_init_node_;
}

ge::ExecuteGraph *ConstantExpressionMotion::GetDeInitGraph() {
  if (cached_de_init_graph_ == nullptr) {
    auto de_init_node = GetDeInitNode();
    GE_ASSERT_NOTNULL(de_init_node);
    cached_de_init_graph_ = ge::FastNodeUtils::GetSubgraphFromNode(de_init_node, 0U);
  }
  return cached_de_init_graph_;
}

ge::EdgeSrcEndpoint ConstantExpressionMotion::GetSrcFromInitGraph(ge::FastNode *const node, const size_t variant_num) {
  std::vector<ge::FastNode *> uncached_nodes;
  auto inner_data = node;
  ge::EdgeSrcEndpoint init_src_endpoint;
  int32_t parent_node_index;
  while (true) {
    const auto iter = cached_inner_datas_to_init_out_.find(inner_data);
    if (iter != cached_inner_datas_to_init_out_.end()) {
      init_src_endpoint = iter->second;
      break;
    }
    uncached_nodes.emplace_back(inner_data);

    GE_ASSERT_TRUE(ge::AttrUtils::GetInt(inner_data->GetOpDescBarePtr(), "index", parent_node_index));
    /** todo 此处强转 size_t 到 int32_t 存在截断，由于初始值为 size_t::max，非 while 子图场景强转结果为 -1，继续后续流程
     *       若是 while 子图场景，那么在 variant_num 范围内的 InnerData 都会跳过。由此逻辑推断，loop_variant_range 的初
     *       始值是否考虑置为 0。
     */
    if (parent_node_index < static_cast<int32_t>(variant_num)) {
      return {};
    }
    const auto parent_graph = inner_data->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(parent_graph);
    const auto parent_node = parent_graph->GetParentNodeBarePtr();
    if (parent_node == nullptr) {
      return {};
    }

    auto parent_in_data_edge = parent_node->GetInDataEdgeByIndex(parent_node_index);
    GE_ASSERT_NOTNULL(parent_in_data_edge);
    auto src_node = parent_in_data_edge->src;
    auto src_index = parent_in_data_edge->src_output;

    const auto src_node_type = src_node->GetTypePtr();
    if (strcmp(src_node_type, GetSubExeGraphTypeStr(kInitExeGraph)) == 0) {
      auto netoutput = GetInitNetOutput();
      GE_ASSERT_NOTNULL(netoutput);
      auto netout_in_data_edge = netoutput->GetInDataEdgeByIndex(src_index);
      GE_ASSERT_NOTNULL(netout_in_data_edge);
      init_src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(netout_in_data_edge);
      break;
    } else if (IsInnerDataType(src_node_type)) {
      inner_data = src_node;
    } else {
      // 说明原始节点来自于main graph，增加处理等价节点对应于init graph中src输出
      auto node_from_init = GetEquivalantNodesInInit(src_node);
      if (node_from_init != nullptr) {
        init_src_endpoint = {node_from_init, src_index};
      }
      break;
    }
  }

  for (const auto &uncached_node : uncached_nodes) {
    cached_inner_datas_to_init_out_[uncached_node] = init_src_endpoint;
  }
  return init_src_endpoint;
}

ge::EdgeSrcEndpoint ConstantExpressionMotion::GetInitOut(ge::FastNode *const inner_data) const {
  const auto it = cached_inner_datas_to_init_out_.find(inner_data);
  if (it != cached_inner_datas_to_init_out_.cend()) {
    return it->second;
  }
  return {};
}

ge::FastNode *ConstantExpressionMotion::GetEquivalantNodesInInit(ge::FastNode *const node) const {
  const auto it = cached_equivalant_nodes_.find(node);
  if (it != cached_equivalant_nodes_.end()) {
    return it->second;
  }
  return nullptr;
}

ge::graphStatus ConstantExpressionMotion::Run(ge::ExecuteGraph *const graph, bool &changed) {
  GE_ASSERT_SUCCESS(Init(graph));
  std::vector<ge::ExecuteGraph *> passed_graphs;
  ge::FastNode *no_op_node = nullptr;
  std::vector<ge::FastNode *> no_op_src_nodes_next_time;
  // CEM需要自顶向下的子图优化顺序，可以一次优化完全，拓扑排序后，子图排列刚好满足自顶向下的要求
  for (const auto subgraph : graph->GetAllSubgraphs()) {
    if (!NeedCem(subgraph)) {
      continue;
    }

    CeNodes ce_nodes;
    GE_ASSERT_SUCCESS(CollectCeAndInnerData(*this, subgraph, ce_nodes));
    if (ce_nodes.ce_nodes.empty()) {
      GELOGD("The subgraph %s belongs to node %s has no CE nodes, skip CEM", subgraph->GetName().c_str(),
             subgraph->GetParentNodeBarePtr()->GetNamePtr());
      continue;
    }
    passed_graphs.emplace_back(subgraph);

    CeGuarderNodes ce_guarder_nodes;
    GE_ASSERT_SUCCESS(CollectCeGuarders(subgraph, ce_nodes, ce_guarder_nodes));

    std::vector<InEdge> in_edges;
    std::vector<OutEdge> out_edges;
    GE_ASSERT_SUCCESS(CollectIoEdges(*this, ce_nodes, ce_guarder_nodes, in_edges, out_edges));

    // 生成在init图中的结点和生成在main图的结点之间无数据连边但存在控制依赖关系时，cem前由于init图和main图的执行可以保证时序，
    // cem后main图的ce node被移动到init图, 由于控制关系丢失导致两个节点的执行顺序无法保证。
    // 在原来init图中的节点和cem移到init图的节点中间增加NoOp节点，增加控制连边来保证原来init图中的节点优先执行。
    GE_ASSERT_SUCCESS(CollectNoOpSrcNodesFromCeNodes(ce_nodes.ce_nodes, out_edges, no_op_src_nodes_next_time));
    std::vector<ge::FastNode *> no_op_dst_nodes;
    GE_ASSERT_SUCCESS(CollectNoOpDstNodes(*this, no_op_src_nodes_, ce_nodes.ce_nodes, no_op_dst_nodes));

    GE_ASSERT_SUCCESS(UnlinkDataEdges(in_edges, out_edges));
    GE_ASSERT_SUCCESS(UnlinkInCtrlEdges(ce_nodes.ce_nodes, ce_nodes.GetAllCollectedSet()));
    GE_ASSERT_SUCCESS(UnlinkOutCtrlEdges(ce_nodes.ce_nodes, ce_nodes.GetAllCollectedSet()));
    GE_ASSERT_SUCCESS(UnlinkInCtrlEdges(ce_guarder_nodes.guarder_nodes, ce_guarder_nodes.GetAllCollectedSet()));

    GE_ASSERT_SUCCESS(MoveCeToInit(GetInitGraph(), ce_nodes, in_edges));
    GE_ASSERT_SUCCESS(MoveGuardersToDeInit(GetDeInitGraph(), ce_guarder_nodes));

    GE_ASSERT_SUCCESS(LinkOutEdges(*this, out_edges));
    GE_ASSERT_SUCCESS(AddCtrlFromInitNodesToCeNodes(GetInitGraph(), no_op_src_nodes_, no_op_dst_nodes, no_op_node));
  }

  if (passed_graphs.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  changed = true;

  // 本次cem完成后更新noop的源节点
  UpdateNoOpSrcNodes(no_op_node, no_op_src_nodes_next_time);

  // 最后清理无用的InnerData稍微有点浪费，因为可以在重连边的时候，复用已经废弃的InnerData，
  // 不过这样实现太复杂了，而且我认为废弃的InnerData不会太多，收益也不算高，因此最后统一清理
  // 自底向上的InnerData移除可以一次遍历移除干净，因此这里做一下反转
  std::reverse(passed_graphs.begin(), passed_graphs.end());
  auto ret = InnerDataRemoverForSubgraphs(passed_graphs).RemoveUnusedInnerDataNodes();
  if (changed) {
    ge::DumpGraph(graph, "AfterCEM");
  }
  return ret;
}
}  // namespace bg
}  // namespace gert
