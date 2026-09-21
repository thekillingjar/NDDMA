/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/same_transdata_breadth_fusion_pass.h"
#include <stack>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common/trans_op_creator.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "api/gelib/gelib.h"
#include "formats/utils/formats_trans_utils.h"

namespace ge {
namespace {
constexpr size_t kPrintNameLimit = 200U;

bool IsSingleOutputToMultiInput(const OutDataAnchorPtr &out_data_anchor) {
  const auto &peer_in_data_anchors = out_data_anchor->GetPeerInDataAnchors();
  if (peer_in_data_anchors.empty()) {
    return false;
  }
  // 单输出多引用
  if (peer_in_data_anchors.size() > 1U) {
    return true;
  }

  const auto peer_in_anchor = *peer_in_data_anchors.begin();
  const auto next_node = peer_in_anchor->GetOwnerNode();
  if ((next_node != nullptr) && (next_node->GetOpDesc() != nullptr)) {
    if (next_node->GetOpDesc()->GetSubgraphInstanceNames().size() > 1U) {
      return true;
    }
  }
  return false;
}

bool IsTransOp(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  return node->GetType() == CAST || node->GetType() == TRANSPOSE || node->GetType() == TRANSPOSED ||
         node->GetType() == RESHAPE || node->GetType() == TRANSDATA;
}

std::unordered_map<std::string, NodeType> kNodeTypeMap = {
    {NETOUTPUT, NodeType::kNetOutput},
    {DATA, NodeType::kData},
    {TRANSDATA, NodeType::kTransdata},
    {CAST, NodeType::kCast}
};

NodeType GetNodeType(const NodePtr &node) {
  const auto type = node->GetType();
  const auto iter = kNodeTypeMap.find(type);
  if (iter != kNodeTypeMap.end()) {
    return iter->second;
  }
  if (!node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty()) {
    return NodeType::kWrapperNode;
  }
  return NodeType::kOthers;
}

bool IsWrapperNode(const OpDescPtr &op_desc) {
  return !op_desc->GetSubgraphInstanceNames().empty();
}

void PrintPaths(const Paths &paths) {
  GELOGI("Paths size: %zu", paths.size());
  for (size_t i = 0U; i < paths.size(); ++i) {
    if (!paths[i].empty()) {
      std::stringstream ss;
      for (size_t j = 0U; j < paths[i].size(); ++j) {
        const auto &link_node = paths[i][j];
        const auto &pre_node = link_node.real_peer_out_anchor->GetOwnerNode();
        const auto out_index = link_node.real_peer_out_anchor->GetIdx();
        const auto in_index = link_node.in_anchor->GetIdx();
        const auto &print_name = pre_node->GetName().size() > kPrintNameLimit ?
                                 pre_node->GetName().substr(kPrintNameLimit) : pre_node->GetName();
        ss << "[" << pre_node->GetType() << "][" << print_name << "][" << out_index << "]->[" << in_index << "]";
      }
      const auto &link_node = paths[i].back();
      const auto &last_node = link_node.in_anchor->GetOwnerNode();
      const auto &print_name = last_node->GetName().size() > kPrintNameLimit ?
                               last_node->GetName().substr(kPrintNameLimit) : last_node->GetName();
      ss << "[" << last_node->GetType() << "][" << print_name << "]";
      GELOGI("paths[%zu]: %s", i, ss.str().c_str());
    }
  }
}

void PrintFusedTransdata(const std::vector<Paths> &same_transdata_paths_groups) {
  for (const auto &same_transdata_paths : same_transdata_paths_groups) {
    std::stringstream ss;
    auto iter = same_transdata_paths.begin();
    while (iter != same_transdata_paths.end()) {
      ss << iter->back().in_anchor->GetOwnerNodeBarePtr()->GetName();
      if (iter + 1 != same_transdata_paths.end()) {
        ss << ", ";
      }
      ++iter;
    }
    GELOGI("transdata node [%s] will be fused.", ss.str().c_str());
  }
}

// avoid scene 1: A->Cast->TransData while A's DataType is not supported by TransData
// avoid scene 2: A->Cast->TransData while the output format of TransData is not supported by Cast
graphStatus CheckOpSupported(const Path &path, const LinkNode &link_node, bool &is_supported) {
  if (path.empty()) {
    is_supported = true;
    return GRAPH_SUCCESS;
  }
  const auto &transdata = link_node.in_anchor->GetOwnerNode();
  const auto &first_cast = path.front().in_anchor->GetOwnerNode();
  auto input_desc = first_cast->GetOpDesc()->GetInputDescPtr(0U);
  GE_CHECK_NOTNULL(input_desc);

  auto trans_op_desc = transdata->GetOpDesc();
  auto trans_out_desc = trans_op_desc->GetOutputDescPtr(0U);
  GE_CHECK_NOTNULL(trans_out_desc);

  // Check transdata whether is supported. src: A->Cast->TransData dst:A->TransData->Cast
  const std::shared_ptr<GeTensorDesc> new_trans_out_tensor = MakeShared<GeTensorDesc>(*trans_out_desc);
  GE_CHECK_NOTNULL(new_trans_out_tensor);
  // New transdata's output datatype has to be consistent with it's input
  new_trans_out_tensor->SetDataType(input_desc->GetDataType());
  // Create a demo transdata op to test supporting of accuracy
  auto transdata_op_desc =
      TransOpCreator::CreateTransDataOp("tmp_" + trans_op_desc->GetName(), *input_desc, *new_trans_out_tensor);
  bool is_transdata_supported = false;
  bool is_cast_supported = false;
  if (transdata_op_desc != nullptr) {
    is_transdata_supported = true;
  }
  GELOGD("transdata: %s, [%s->%s], new datatype: %s, is_transdata_supported: %d.", transdata->GetNamePtr(),
         TypeUtils::FormatToSerialString(input_desc->GetFormat()).c_str(),
         TypeUtils::FormatToSerialString(trans_out_desc->GetFormat()).c_str(),
         TypeUtils::DataTypeToSerialString(input_desc->GetDataType()).c_str(), is_transdata_supported);

  // cast_output_desc
  for (const auto &cast_link_node : path) {
    const auto &cast_node = cast_link_node.in_anchor->GetOwnerNode();
    auto cast_input_desc = cast_node->GetOpDesc()->GetInputDescPtr(0U);
    GE_CHECK_NOTNULL(cast_input_desc);
    auto cast_output_desc = cast_node->GetOpDesc()->GetOutputDescPtr(0U);
    GE_CHECK_NOTNULL(cast_output_desc);
    // Check cast whether is supported. src: A(FP16, C0=16)->Cast(FP32, C0=16)->TransData(FP32, C0=8)
    // dst:A(FP16, C0=16)->TransData(FP16, C0=8)->Cast(FP32, C0=8)
    const std::shared_ptr<GeTensorDesc> new_cast_in_tensor = MakeShared<GeTensorDesc>(*cast_input_desc);
    GE_CHECK_NOTNULL(new_cast_in_tensor);
    const std::shared_ptr<GeTensorDesc> new_cast_out_tensor = MakeShared<GeTensorDesc>(*cast_output_desc);
    GE_CHECK_NOTNULL(new_cast_out_tensor);
    // The output format of new cast has to be consistent with it's input
    new_cast_in_tensor->SetFormat(trans_out_desc->GetFormat());
    new_cast_out_tensor->SetFormat(trans_out_desc->GetFormat());
    // Create a demo cast op to test supporting of accuracy
    auto cast_op_desc =
        TransOpCreator::CreateCastOp("tmp_" + cast_node->GetName(), *new_cast_in_tensor, *new_cast_out_tensor);
    if (cast_op_desc != nullptr) {
      is_cast_supported = true;
    }
    GELOGD("cast: %s, [%s->%s], new format: %s, is_cast_supported: %d.", cast_node->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(cast_input_desc->GetDataType()).c_str(),
           TypeUtils::DataTypeToSerialString(cast_output_desc->GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(trans_out_desc->GetFormat()).c_str(), is_cast_supported);
    if (!is_cast_supported) {
      break;
    }
  }

  is_supported = is_transdata_supported && is_cast_supported;
  return GRAPH_SUCCESS;
}

std::set<std::string> GetInControlIdentityNodes(const Path &path, const LinkNode &transdata_link_node) {
  std::set<std::string> in_node_names;
  const auto &transdata_node = transdata_link_node.in_anchor->GetOwnerNode();
  for (const auto &in_node : transdata_node->GetInControlNodes()) {
    in_node_names.insert(in_node->GetName());
  }
  for (const auto &link_node : path) {
    if (link_node.node_type == NodeType::kTransdata) {
      break;
    }
    const auto node = link_node.in_anchor->GetOwnerNode();
    for (const auto &in_node : node->GetInControlNodes()) {
      if (in_node->GetType() == IDENTITY) {
        in_node_names.insert(in_node->GetName());
      }
    }
  }
  return in_node_names;
}

bool IsSame(const CompareInfo &lh, const CompareInfo &rh) {
  const bool all_same = (lh.stream_label == rh.stream_label) &&
                        (lh.input_tensor_desc->GetFormat() == rh.input_tensor_desc->GetFormat()) &&
                        (lh.output_tensor_desc->GetFormat() == rh.output_tensor_desc->GetFormat()) &&
                        (lh.input_tensor_desc->GetOriginShape() == rh.input_tensor_desc->GetOriginShape()) &&
                        (lh.output_tensor_desc->GetOriginShape() == rh.output_tensor_desc->GetOriginShape()) &&
                        (lh.in_ctrl_nodes == rh.in_ctrl_nodes);
  return all_same;
}

void PrintCompareInfo(const NodePtr &node, const CompareInfo &info) {
  std::stringstream ss;
  ss << "[";
  for (auto &item : info.in_ctrl_nodes) {
    ss << item << ", ";
  }
  ss << "]";
  GELOGI("transdata[%s] compare info: [IN %s[0x%x] [%s]], [OUT %s[0x%x] [%s]], stream_label: \"%s\", in_ctrl_nodes: %s",
         node->GetNamePtr(),
         TypeUtils::FormatToSerialString(info.input_tensor_desc->GetFormat()).c_str(),
         info.input_tensor_desc->GetFormat(),
         formats::ShapeToString(info.input_tensor_desc->GetOriginShape()).c_str(),
         TypeUtils::FormatToSerialString(info.output_tensor_desc->GetFormat()).c_str(),
         info.output_tensor_desc->GetFormat(),
         formats::ShapeToString(info.output_tensor_desc->GetOriginShape()).c_str(),
         info.stream_label.c_str(), ss.str().c_str());
}

graphStatus CopyTensorDesc(const GeTensorDesc &src_desc, GeTensorDescPtr &dst_desc) {
  GE_ASSERT_NOTNULL(dst_desc);
  dst_desc->SetFormat(src_desc.GetFormat());
  dst_desc->SetOriginFormat(src_desc.GetOriginFormat());
  dst_desc->SetShape(src_desc.GetShape());
  dst_desc->SetOriginShape(src_desc.GetOriginShape());

  std::vector<std::pair<int64_t, int64_t>> shape_range;
  GE_ASSERT_SUCCESS(src_desc.GetShapeRange(shape_range));
  GE_ASSERT_SUCCESS(dst_desc->SetShapeRange(shape_range));

  std::vector<std::pair<int64_t, int64_t>> origin_shape_range;
  GE_ASSERT_SUCCESS(src_desc.GetOriginShapeRange(origin_shape_range));
  GE_ASSERT_SUCCESS(dst_desc->SetOriginShapeRange(origin_shape_range));

  uint32_t real_dim = 0;
  if (TensorUtils::GetRealDimCnt(src_desc, real_dim) == GRAPH_SUCCESS) {
    TensorUtils::SetRealDimCnt(*dst_desc, real_dim);
  }
  return GRAPH_SUCCESS;
}

void UpdateDataType(const GeTensorDescPtr &src_desc, GeTensorDescPtr &dst_desc) {
  dst_desc->SetDataType(src_desc->GetDataType());
  dst_desc->SetOriginDataType(src_desc->GetOriginDataType());
}

/*
 * head--+--cast1--transdata1
 *       |
 *       +--cast2--transdata2
 * 不能直接用transdata的dtype，比如这种场景，显然head的输出dtype和transdata的dtype是不一样的
 */
graphStatus UpdateTransdataDtype(const Node *trans, const OutDataAnchorPtr &head_out_anchor) {
  const auto head_op_desc = head_out_anchor->GetOwnerNode()->GetOpDesc();
  const auto head_tensor_desc = head_op_desc->GetOutputDesc(head_out_anchor->GetIdx());
  auto trans_op_desc = trans->GetOpDesc();
  GE_ASSERT_NOTNULL(trans_op_desc->MutableInputDesc(0));
  GE_ASSERT_NOTNULL(trans_op_desc->MutableOutputDesc(0));
  trans_op_desc->MutableInputDesc(0)->SetDataType(head_tensor_desc.GetDataType());
  trans_op_desc->MutableInputDesc(0)->SetOriginDataType(head_tensor_desc.GetOriginDataType());
  trans_op_desc->MutableOutputDesc(0)->SetDataType(head_tensor_desc.GetDataType());
  trans_op_desc->MutableOutputDesc(0)->SetOriginDataType(head_tensor_desc.GetOriginDataType());
  return GRAPH_SUCCESS;
}

NodePtr CreateDataNode(ComputeGraphPtr &sub_graph, const size_t parent_index) {
  const auto data_name = sub_graph->GetName() + "_transdata_fusion_arg_" +std::to_string(parent_index);
  OpDescPtr op_desc = MakeShared<OpDesc>(data_name, DATA);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index));
  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(data_desc));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(data_desc));
  NodePtr data_node = sub_graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(data_node);
  GELOGI("add new data[%s]", data_name.c_str());
  return data_node;
}

// 找到topoid最小的那个，否则有可能成环
size_t GetKeepTransdataPathIndex(const Paths &paths_group) {
  size_t keep_transdata_path_index = 0U;

  for (size_t i = 1U; i < paths_group.size(); ++i) {
    if (paths_group[i].back().in_anchor->GetOwnerNode()->GetOpDesc()->GetId() <
        paths_group[keep_transdata_path_index].back().in_anchor->GetOwnerNode()->GetOpDesc()->GetId()) {
      keep_transdata_path_index = i;
    }
  }
  return keep_transdata_path_index;
}

/*
 * data--partitioned_call--transdata
 *
 *    +---------------------+
 *    |    partitioned_call |
 *    |                     |
 *    |      sub_data       |
 *    |         |           |
 *    |      Netoutput      |
 *    +---------------------+
 *
 * 为什么要判断类型对端必须是data节点？
 * 因为上图中也满足transdata和data在一个子图中的条件，但是不能在这里处理，应该在AddNewInputForNetOutput中处理
 */
graphStatus ConnetToFusedAnchors(std::vector<InDataAnchorPtr> &fused_anchors, const ComputeGraph *const sub_graph,
                                 OutDataAnchorPtr &new_data_out_anchor) {
  auto iter = fused_anchors.begin();
  while (iter != fused_anchors.end()) {
    auto fused_in_anchor = *iter;
    GE_ASSERT_NOTNULL(fused_in_anchor->GetPeerOutAnchor());
    if ((fused_in_anchor->GetOwnerNode()->GetOwnerComputeGraphBarePtr() == sub_graph)
        && (fused_in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetType() == DATA)) {
      GE_ASSERT_SUCCESS(fused_in_anchor->Unlink(fused_in_anchor->GetPeerOutAnchor()));
      GE_ASSERT_NOTNULL(new_data_out_anchor);
      GE_ASSERT_SUCCESS(new_data_out_anchor->LinkTo(fused_in_anchor));
      iter = fused_anchors.erase(iter);
    } else {
      ++iter;
    }
  }
  return GRAPH_SUCCESS;
}

bool IsAllNodeInPathsWithSameTransdata(const std::set<InDataAnchorPtr> &allowed_set, std::queue<Path> &path_queue) {
  while (!path_queue.empty()) {
    const auto &path = path_queue.front();
    for (const auto &node : path) {
      if (allowed_set.find(node.in_anchor) == allowed_set.end()) {
        const auto node_ptr = node.in_anchor->GetOwnerNodeBarePtr();
        GELOGI("node: %s, type: %s is not in allowed_set", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
        return false;
      }
    }
    path_queue.pop();
  }
  return true;
}

graphStatus MoveTransdataOutDataEdge(NodePtr &trans) {
  auto trans_out = trans->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(trans_out);
  auto trans_in = trans->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(trans_in);
  auto pre_out = trans_in->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(pre_out);

  for (auto &peer_in : trans_out->GetPeerInDataAnchors()) {
    GE_ASSERT_SUCCESS(peer_in->Unlink(trans_out));
    GE_ASSERT_SUCCESS(pre_out->LinkTo(peer_in));
  }
  return GRAPH_SUCCESS;
}

graphStatus MoveTransdataInCtrlEdge(NodePtr &trans) {
  auto trans_out = trans->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(trans_out);

  for (auto &next_node : trans->GetOutDataNodes()) {
    GE_ASSERT_SUCCESS(GraphUtils::MoveInCtrlEdges(trans, next_node));
  }
  for (auto &next_node : trans->GetOutControlNodes()) {
    GE_ASSERT_SUCCESS(GraphUtils::MoveInCtrlEdges(trans, next_node));
  }
  return GRAPH_SUCCESS;
}

/*
 *      a
 *      |
 *      if      then_subgraph     then_subgraph
 *      |       +-------------+   +-------------+
 *   transdata  |   data1     |   |   data1     |
 *      |       |     |       |   |     |       |
 *      c       |  netoutput  |   |     b       |
 *              +-------------+   |     |       |
 *                                |  netoutput  |
 *                                +-------------+
 * 约束：多子图贯穿场景，path从子图外到子图内再到子图外，约束所有子图上都存在从a到达transdata的路径
 * 完整实现这个约束比较复杂，当前为简单实现，不允许贯穿具有多子图的节点
 */
bool IsHeadNodeOutOfSubgraph(const NodePtr &netoutput, const NodePtr &head_node) {
  auto sub_graph = netoutput->GetOwnerComputeGraphBarePtr();
  while ((sub_graph != nullptr) && (sub_graph->GetParentNodeBarePtr() != nullptr)) {
    auto parent_node_graph = sub_graph->GetParentNodeBarePtr()->GetOwnerComputeGraphBarePtr();
    if (parent_node_graph == head_node->GetOwnerComputeGraphBarePtr()) {
      return true;
    }
    sub_graph = parent_node_graph;
  }
  return false;
}
} // namesapce

graphStatus SameTransdataBreadthFusionPass::Run(ComputeGraphPtr graph) {
  if (root_graph_ == nullptr) {
    root_graph_ = GraphUtils::FindRootGraph(graph);
    GE_ASSERT_NOTNULL(root_graph_);
  }
  if (graph != root_graph_) {
    GELOGI("sub graph %s in, just return.", graph->GetName().c_str());
    return GRAPH_SUCCESS;
  }

  GE_ASSERT_SUCCESS(graph->TopologicalSorting());
  GE_DUMP(graph, "before_SameTransdataBreadthFusionPass");
  GELOGI("[SameTransdataBreadthFusionPass]: optimize begin, graph: %s.", graph->GetName().c_str());
  GE_ASSERT_SUCCESS(CollectAllSubgraphDataNodesMap());
  GE_ASSERT_SUCCESS(DoRun(graph));
  GELOGI("[SameTransdataBreadthFusionPass]: Optimize success.");
  GE_DUMP(graph, "after_SameTransdataBreadthFusionPass");
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::CollectAllSubgraphDataNodesMap() {
  for (const auto &sub_graph : root_graph_->GetAllSubgraphs()) {
    if ((sub_graph == nullptr) || (sub_graph->GetParentNode() == nullptr)) {
      continue;
    }
    auto &data_nodes = graph_nodes_[sub_graph];
    for (const auto &data : sub_graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(data->GetOpDesc());
      if (data->GetType() != DATA) {
        continue;
      }
      uint32_t parent_index = 0;
      GE_ASSERT_TRUE(AttrUtils::GetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index),
                     "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                     data->GetName().c_str(), data->GetType().c_str());
      data_nodes[parent_index] = data;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::DoRun(ComputeGraphPtr graph) {
  for (const auto &head_node : graph->GetAllNodes()) {
    if (IsTransOp(head_node)) {
      continue;
    }
    for (auto &head_out_anchor : head_node->GetAllOutDataAnchors()) {
      if (!IsSingleOutputToMultiInput(head_out_anchor)) {
        continue;
      }
      GE_ASSERT_SUCCESS(RunForNode(head_out_anchor));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::RunForNode(OutDataAnchorPtr &head_out_anchor) {
  Paths paths;
  GE_ASSERT_SUCCESS(GetPathsToTransdata(head_out_anchor, paths));
  if (paths.size() <= 1U) {
    return GRAPH_SUCCESS;
  }
  PrintPaths(paths);
  GE_ASSERT_SUCCESS(FuseTransdata(paths));
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetPathsToTransdata(const OutDataAnchorPtr &head_out_anchor,
                                                                Paths &paths) const {
  std::queue<Path> path_queue;
  bool is_supported = false;
  GE_ASSERT_SUCCESS(GetRealInAnchors(head_out_anchor, head_out_anchor, path_queue, Path()));
  while (!path_queue.empty()) {
    auto cur_path = path_queue.front(); // copy
    path_queue.pop();

    const auto &link_node = cur_path.back();
    const auto &owner_node = link_node.in_anchor->GetOwnerNode();
    switch (link_node.node_type) {
      case NodeType::kTransdata :
        GE_ASSERT_SUCCESS(CheckOpSupported(cur_path, link_node, is_supported));
        if ((owner_node->GetOutDataNodesSize() != 0U) && is_supported) {
          // 这里保证path的最后一个节点一定是transdata
          paths.emplace_back(std::move(cur_path));
        }
        break;
      case NodeType::kCast:
        // path中在transdata前面如果有其他节点的话，这里保证一定是cast
        for (const auto &cast_out_anchor : owner_node->GetAllOutDataAnchors()) {
          GE_ASSERT_SUCCESS(GetRealInAnchors(cast_out_anchor, cast_out_anchor, path_queue, cur_path));
        }
        break;
      case NodeType::kOthers:
        break;
      case NodeType::kData:
      case NodeType::kNetOutput:
      case NodeType::kWrapperNode:
        GELOGE(FAILED, "type: %s node name: %s should not be here.",
               owner_node->GetTypePtr(), owner_node->GetNamePtr());
        return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetRealInAnchors(const OutDataAnchorPtr &real_out_anchor,
                                                             const OutDataAnchorPtr &out_anchor,
                                                             std::queue<Path> &path_queue,
                                                             const Path &path) const {
  std::stack<OutDataAnchorPtr> out_anchor_stack;
  out_anchor_stack.push(out_anchor);
  while (!out_anchor_stack.empty()) {
    const auto cur_out_anchor = out_anchor_stack.top();
    out_anchor_stack.pop();
    for (const auto &in_anchor : cur_out_anchor->GetPeerInDataAnchors()) {
      const auto next_node = in_anchor->GetOwnerNode();
      if ((next_node != nullptr) && (next_node->GetOpDesc() != nullptr)) {
        if (IsWrapperNode(next_node->GetOpDesc())) {
          GE_ASSERT_SUCCESS(GetRealInAnchorsForWrapperNode(in_anchor, out_anchor_stack));
        } else if (next_node->GetType() == NETOUTPUT) {
          GE_ASSERT_SUCCESS(GetRealInAnchorsForNetOutput(real_out_anchor, in_anchor, path, out_anchor_stack));
        } else {
          Path new_path(path);
          new_path.emplace_back(LinkNode{in_anchor, real_out_anchor, GetNodeType(next_node)});
          path_queue.push(std::move(new_path));
        }
      }
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetRealInAnchorsForWrapperNode(
    const InDataAnchorPtr &in_anchor, std::stack<OutDataAnchorPtr> &out_anchor_stack) const {
  const auto op_desc = in_anchor->GetOwnerNode()->GetOpDesc();
  for (const auto &name : op_desc->GetSubgraphInstanceNames()) {
    const auto &sub_graph = root_graph_->GetSubgraph(name);
    GE_ASSERT_NOTNULL(sub_graph);
    OutDataAnchorPtr data_out_anchor = nullptr;
    GE_ASSERT_SUCCESS(GetSubgraphDataOutAnchor(sub_graph, in_anchor->GetIdx(), data_out_anchor));
    if (data_out_anchor == nullptr) {
      continue;
    }
    out_anchor_stack.push(data_out_anchor);
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetSubgraphDataOutAnchor(const ComputeGraphPtr &sub_graph,
                                                                     const int32_t wrapper_node_input_index,
                                                                     OutDataAnchorPtr &data_out_anchor) const {
  GE_ASSERT_NOTNULL(sub_graph);
  const auto &sub_graph_iter = graph_nodes_.find(sub_graph);
  GE_ASSERT_TRUE(sub_graph_iter != graph_nodes_.end());

  const auto &data_nodes = sub_graph_iter->second;
  const auto &data_iter = data_nodes.find(wrapper_node_input_index);
  if (data_iter != data_nodes.end()) {
    data_out_anchor = data_iter->second->GetOutDataAnchor(0);
  } else {
    data_out_anchor = nullptr;
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetRealInAnchorsForNetOutput(
    const OutDataAnchorPtr &real_out_anchor, const InDataAnchorPtr &in_anchor, const Path &path,
    std::stack<OutDataAnchorPtr> &out_anchor_stack) const {
  const auto &netoutput = in_anchor->GetOwnerNode();
  const auto &op_desc = netoutput->GetOpDesc();
  const auto in_tensor_desc = op_desc->GetInputDesc(static_cast<uint32_t>(in_anchor->GetIdx()));
  uint32_t parent_index;
  if (!AttrUtils::GetInt(in_tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return GRAPH_SUCCESS;
  }

  GE_ASSERT_NOTNULL(netoutput->GetOwnerComputeGraphBarePtr());
  const auto &wrapper_node = netoutput->GetOwnerComputeGraphBarePtr()->GetParentNode();
  GE_ASSERT_NOTNULL(wrapper_node);

  /*
   * 如果父节点有多个子图，并且head在子图内，要求path不能通过netoutput延伸到子图外。
   * 因为如果transdata提取到某一子图内，那么其他子图就不等价了
   */
  const auto &wrapper_op_desc = wrapper_node->GetOpDesc();
  GE_ASSERT_NOTNULL(wrapper_op_desc);
  if ((wrapper_op_desc->GetSubgraphInstanceNames().size() > 1U) && (real_out_anchor != nullptr)) {
    const auto &head_node = path.empty() ? real_out_anchor->GetOwnerNode() :
                            path.front().real_peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(head_node);
    if (IsHeadNodeOutOfSubgraph(netoutput, head_node)
        || (head_node->GetOwnerComputeGraphBarePtr() == netoutput->GetOwnerComputeGraphBarePtr())) {
      GELOGI("wrapper_node:%s has %zu subgraphs, head_node: %s is in subgraph[%s], stop search through netoutput: %s",
             wrapper_node->GetNamePtr(), wrapper_op_desc->GetSubgraphInstanceNames().size(), head_node->GetNamePtr(),
             head_node->GetOwnerComputeGraphBarePtr()->GetName().c_str(), netoutput->GetNamePtr());
      return GRAPH_SUCCESS;
    }
  }

  const auto &out_anchor = wrapper_node->GetOutDataAnchor(parent_index);
  GE_ASSERT_NOTNULL(out_anchor, "wrapper_node: %s, parent_index: %u", wrapper_node->GetNamePtr(), parent_index);
  out_anchor_stack.push(out_anchor);
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::FuseTransdata(Paths &paths) {
  std::vector<Paths> same_transdata_paths_groups;
  GE_ASSERT_SUCCESS(GetSameTransdataPath(paths, same_transdata_paths_groups));

  auto iter = same_transdata_paths_groups.begin();
  while (iter != same_transdata_paths_groups.end()) {
    GE_ASSERT_SUCCESS(RemoveUnSupportedPath(*iter));
    if (iter->size() <= 1U) {
      iter = same_transdata_paths_groups.erase(iter);
    } else {
      ++iter;
    }
  }
  PrintFusedTransdata(same_transdata_paths_groups);

  // 在same_transdata_paths_groups取出来的节点，可以保证anchor，NodePtr，OpDesc，输入输出tensordesc不为空
  for (auto &paths_group : same_transdata_paths_groups) {
    GE_ASSERT_SUCCESS(AddNewPathToTransdataForDiffGraph(paths_group));
    const auto keep_transdata_path_index = GetKeepTransdataPathIndex(paths_group);
    GE_ASSERT_SUCCESS(UpdateTensorDesc(paths_group, keep_transdata_path_index));
    GE_ASSERT_SUCCESS(ExtractTransdata(paths_group, keep_transdata_path_index));
    for (size_t i = 0U; i < paths_group.size();  ++i) {
      if (i != keep_transdata_path_index) {
        GE_ASSERT_SUCCESS(DeleteTransdata(paths_group[i]));
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetSameTransdataPath(Paths &paths,
                                                                 std::vector<Paths> &same_transdata_paths_groups) {
  while (paths.size() > 1U) {
    auto iter = paths.begin();
    Paths same_transdata_paths;
    same_transdata_paths.emplace_back(std::move(*iter));
    const auto &first_path = same_transdata_paths.front();

    iter = paths.erase(iter);
    LinkNode first_transdata = first_path.back();
    CompareInfo first_info;
    GE_ASSERT_SUCCESS(GetCompareInfo(first_path, first_transdata, first_info));

    while (iter != paths.end()) {
      auto &another_path = *iter;
      LinkNode another_transdata = another_path.back();
      CompareInfo another_info;
      GE_ASSERT_SUCCESS(GetCompareInfo(another_path, another_transdata, another_info));
      if (IsSame(first_info, another_info)) {
        same_transdata_paths.emplace_back(std::move(another_path));
        iter = paths.erase(iter);
      } else {
        GELOGI("%s is different from %s", another_transdata.in_anchor->GetOwnerNode()->GetNamePtr(),
               first_transdata.in_anchor->GetOwnerNode()->GetNamePtr());
        iter++;
      }
    }
    if (same_transdata_paths.size() > 1U) {
      same_transdata_paths_groups.emplace_back(std::move(same_transdata_paths));
    }
  }
  return GRAPH_SUCCESS;
}

/*
 *                                +----------------------+
 *        +---transdata1          |   partitioned_call   |
 *        |                       |                      |
 *    op--+---transdata2          |       +---add        |
 *        |                       |       |              |
 *        +---partitioned_call    | data--+---transdta3  |
 *                                +----------------------+
 *
 *                       |
 *                      \|/
 *
 *                                +----------------------+
 *        +---transdata1          |   partitioned_call   |
 *        |                       |                      |
 *    op--+---transdata2          | data---add           |
 *        |                       |                      |
 *        +---partitioned_call    | data_1---transdta3   |
 *        |      |                +----------------------+
 *        +------+
 *   add new path from op to transdata3
 */
graphStatus SameTransdataBreadthFusionPass::AddNewPathToTransdataForDiffGraph(Paths &paths_group) {
  auto head_out_anchor = paths_group[0].front().real_peer_out_anchor;
  std::set<InDataAnchorPtr> allowed_in_anchors;
  for (const auto &cur_path : paths_group) {
    allowed_in_anchors.emplace(cur_path.front().in_anchor);
  }
  // head直连wrapper或者netoutput时，并且真正的输出节点同时有可融合和不可融合算子，这个函数才会处理，否则也是什么都不做返回了
  GE_ASSERT_SUCCESS(AddNewPath(head_out_anchor, head_out_anchor, allowed_in_anchors));
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::AddNewPath(OutDataAnchorPtr &out_anchor,
                                                       OutDataAnchorPtr &new_out_anchor,
                                                       const std::set<InDataAnchorPtr> &allowed_in_anchors) {
  AnchorPairStack out_anchor_pair_stack;
  out_anchor_pair_stack.push(std::make_pair(out_anchor, new_out_anchor));
  while (!out_anchor_pair_stack.empty()) {
    const auto cur_out_anchor = out_anchor_pair_stack.top().first;
    const auto cur_new_out_anchor = out_anchor_pair_stack.top().second;
    out_anchor_pair_stack.pop();
    for (auto &peer_in_anchor : cur_out_anchor->GetPeerInDataAnchors()) {
      const auto head_next = peer_in_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(head_next->GetOpDescBarePtr());
      const auto head_next_type = GetNodeType(head_next);
      if ((head_next_type != NodeType::kWrapperNode) && (head_next_type != NodeType::kNetOutput)) {
        if (allowed_in_anchors.find(peer_in_anchor) != allowed_in_anchors.end()) {
          GE_ASSERT_SUCCESS(peer_in_anchor->Unlink(peer_in_anchor->GetPeerOutAnchor()));
          GE_ASSERT_SUCCESS(cur_new_out_anchor->LinkTo(peer_in_anchor));
        }
        continue;
      }
      std::vector<InDataAnchorPtr> fused_anchors;
      std::vector<InDataAnchorPtr> not_fused_anchors;
      GE_ASSERT_SUCCESS(CollectFusedInAnchors(peer_in_anchor, allowed_in_anchors, head_next_type,
                                              fused_anchors, not_fused_anchors));
      if (fused_anchors.empty() || not_fused_anchors.empty()) {
        /* do nothing
         * peer_in_anchor 对应的真实的节点都是不能和当前transdata融合在一起的，
         * or, peer_in_anchor 对应的真实的节点都能和当前transdata融合在一起的
         */
        continue;
      }
      /*
       * 部分可以融合，其余的不能融合；real_in_anchor对端必然是子图的Data节点。
       * 有两种：1是子图Data单输出多引用，输出节点有的可以被融合，有的不能被融合；
       * 2是多个子图，有的子图Data节点对应的real_in_anchor可以被融合，有的Data对应的real_in_anchor
       * 1和2有可能同时出现。
       * 需要在子图中新增一个Data节点连接到可被融合的real_in_anchor上
       *
       * 为每个子图新增data，即使有的子图中没有可以融合的transdata;
       * 为每个netoutput新增输入，即使它对应的实际输出节点没有可以融合的transdata
       */
      const auto input_size = head_next->GetAllInDataAnchorsSize();
      GE_ASSERT_SUCCESS(NodeUtils::AppendInputAnchor(head_next, input_size + 1U));
      GELOGI("add new input for %s[%s], new input index:%u",
             head_next->GetTypePtr(), head_next->GetNamePtr(), input_size);
      auto new_in_anchor = head_next->GetInDataAnchor(input_size);
      GE_ASSERT_SUCCESS(cur_new_out_anchor->LinkTo(new_in_anchor));

      if (head_next_type == NodeType::kWrapperNode) {
        GE_ASSERT_SUCCESS(AddNewInputForWrapper(peer_in_anchor, fused_anchors, out_anchor_pair_stack));
      } else {
        GE_ASSERT_SUCCESS(AddNewInputForNetOutput(peer_in_anchor, fused_anchors, out_anchor_pair_stack));
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::AddNewInputForWrapper(InDataAnchorPtr &wrapper_in_anchor,
                                                                  std::vector<InDataAnchorPtr> &fused_anchors,
                                                                  AnchorPairStack &out_anchor_pair_stack) {
  if (fused_anchors.empty()) {
    return GRAPH_SUCCESS;
  }
  auto owner_node = wrapper_in_anchor->GetOwnerNode(); // not nullptr
  auto op_desc = owner_node->GetOpDesc(); // not nullptr

  const auto parent_index = owner_node->GetAllInDataAnchorsSize() - 1U; // AllInDataAnchorsSize is not zero
  for (const auto &name : op_desc->GetSubgraphInstanceNames()) {
    auto sub_graph = root_graph_->GetSubgraph(name);
    GE_ASSERT_NOTNULL(sub_graph);

    auto new_data = CreateDataNode(sub_graph, parent_index);
    GE_ASSERT_NOTNULL(new_data);
    UpdateGraphNode(sub_graph, parent_index, new_data);
    auto new_data_out_anchor = new_data->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(new_data_out_anchor);

    GE_ASSERT_SUCCESS(ConnetToFusedAnchors(fused_anchors, sub_graph.get(), new_data_out_anchor));
    if (fused_anchors.empty()) {
      return GRAPH_SUCCESS;
    }

    OutDataAnchorPtr sub_graph_data_out_anchor;
    GE_ASSERT_SUCCESS(GetSubgraphDataOutAnchor(sub_graph, wrapper_in_anchor->GetIdx(), sub_graph_data_out_anchor));
    if (sub_graph_data_out_anchor == nullptr) {
      continue;
    }
    out_anchor_pair_stack.push(std::make_pair(sub_graph_data_out_anchor, new_data_out_anchor));
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::AddNewInputForNetOutput(InDataAnchorPtr &netout_in_anchor,
                                                                    std::vector<InDataAnchorPtr> &fused_anchors,
                                                                    AnchorPairStack &out_anchor_pair_stack) const {
  if (fused_anchors.empty()) {
    return GRAPH_SUCCESS;
  }
  auto netoutput = netout_in_anchor->GetOwnerNode();
  auto netoutput_op_desc = netoutput->GetOpDesc();
  GE_ASSERT_NOTNULL(netoutput_op_desc);

  auto in_tensor_desc = netoutput_op_desc->GetInputDesc(static_cast<uint32_t>(netout_in_anchor->GetIdx()));
  uint32_t parent_index;
  if (!AttrUtils::GetInt(in_tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGW("node %s(%s) %d input does not has %s attr.", netoutput_op_desc->GetNamePtr(),
           netoutput_op_desc->GetTypePtr(), netout_in_anchor->GetIdx(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return GRAPH_SUCCESS;
  }

  GE_ASSERT_NOTNULL(netoutput->GetOwnerComputeGraphBarePtr());
  auto wrapper_node = netoutput->GetOwnerComputeGraphBarePtr()->GetParentNode();
  GE_ASSERT_NOTNULL(wrapper_node);

  // add new output for wrapper node
  const auto out_size = wrapper_node->GetAllOutDataAnchorsSize();
  GE_ASSERT_SUCCESS(NodeUtils::AppendOutputAnchor(wrapper_node, out_size + 1U));
  GELOGI("add new input for %s[%s], new input index:%u",
         wrapper_node->GetTypePtr(), wrapper_node->GetNamePtr(), out_size);
  // set parent index for netoutput input tensor desc
  const auto input_index = netoutput->GetAllInDataAnchorsSize() - 1U; // input is not empty
  auto new_in_tensor_desc = netoutput_op_desc->MutableInputDesc(input_index);
  GE_ASSERT_TRUE(AttrUtils::SetInt(new_in_tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, out_size));

  auto new_out_anchor = wrapper_node->GetOutDataAnchor(out_size);
  GE_ASSERT_SUCCESS(ConnetToFusedAnchors(fused_anchors, wrapper_node->GetOwnerComputeGraphBarePtr(), new_out_anchor));
  if (fused_anchors.empty()) {
    return GRAPH_SUCCESS;
  }
  auto out_anchor = wrapper_node->GetOutDataAnchor(parent_index);
  GE_ASSERT_NOTNULL(out_anchor);
  out_anchor_pair_stack.push(std::make_pair(out_anchor, new_out_anchor));
  return GRAPH_SUCCESS;
}

void SameTransdataBreadthFusionPass::UpdateGraphNode(const ComputeGraphPtr &sub_graph,
                                                     const uint32_t parent_index, NodePtr &node) {
  graph_nodes_[sub_graph][parent_index] = node;
}

// cast的输出都是相同的transdata，如果不满足这一点，删掉这个path
graphStatus SameTransdataBreadthFusionPass::RemoveUnSupportedPath(Paths &paths_with_same_transdata) const {
  std::set<InDataAnchorPtr> allowed_set;
  for (const auto &path : paths_with_same_transdata) {
    for (const auto &node : path) {
      allowed_set.emplace(node.in_anchor);
    }
  }
  auto path_iter = paths_with_same_transdata.begin();
  while (path_iter != paths_with_same_transdata.end()) {
    std::queue<Path> path_queue;
    // path最后一个节点是transdata，前面如果有节点的话，一定都是cast节点。获取所有cast节点的real in anchor
    for (size_t i = 0U; i < path_iter->size() - 1U; ++i) {
      const auto &cast_node = path_iter->at(i);
      for (auto &out_data_anchor : cast_node.in_anchor->GetOwnerNodeBarePtr()->GetAllOutDataAnchors()) {
        GE_ASSERT_SUCCESS(GetRealInAnchors(out_data_anchor, out_data_anchor, path_queue, Path()));
      }
    }
    if (!IsAllNodeInPathsWithSameTransdata(allowed_set, path_queue)) {
      GELOGI("remove path from paths_with_same_transdata, transdata node: %s",
             path_iter->back().in_anchor->GetOwnerNodeBarePtr()->GetNamePtr());
      path_iter = paths_with_same_transdata.erase(path_iter);
    } else {
      ++path_iter;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::GetCompareInfo(const Path &path, const LinkNode &link_node,
                                                           CompareInfo &info) {
  const auto &node = link_node.in_anchor->GetOwnerNode();
  const auto &iter = node_to_info_map_.find(node);
  if (iter != node_to_info_map_.end()) {
    info = iter->second;
    return GRAPH_SUCCESS;
  }

  // 前面流程保证这里op_desc不为空
  const auto &op_desc = node->GetOpDesc();
  (void)AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, info.stream_label);
  info.in_ctrl_nodes = GetInControlIdentityNodes(path, link_node);
  info.input_tensor_desc = op_desc->GetInputDescPtr(link_node.in_anchor->GetIdx());
  GE_ASSERT_NOTNULL(info.input_tensor_desc);
  info.output_tensor_desc = op_desc->GetOutputDescPtr(0);
  GE_ASSERT_NOTNULL(info.output_tensor_desc);
  node_to_info_map_[node] = info;
  PrintCompareInfo(node, info);
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::UpdateTensorDesc(const Paths &paths_group,
                                                             size_t keep_transdata_path_index) {
  const auto trans = paths_group[keep_transdata_path_index].back().in_anchor->GetOwnerNodeBarePtr();
  const auto head_out_anchor = paths_group[keep_transdata_path_index].front().real_peer_out_anchor;
  GE_ASSERT_SUCCESS(UpdateTransdataDtype(trans, head_out_anchor));
  const auto trans_out_tensor_desc = trans->GetOpDescBarePtr()->GetOutputDesc(0U);

  for (const auto &path : paths_group) {
    for (size_t i = 0U; i < path.size() - 1U; ++i) { // path is not empty
      const auto &link_node = path[i];
      const auto node = link_node.in_anchor->GetOwnerNodeBarePtr();
      if (link_node.in_anchor->GetPeerOutAnchor() != link_node.real_peer_out_anchor) {
        GE_ASSERT_SUCCESS(UpdateTensorDescForDiffGraph(trans_out_tensor_desc, link_node));
      }

      auto node_in_tensor_desc = node->GetOpDesc()->MutableInputDesc(0U);
      auto node_out_tensor_desc = node->GetOpDesc()->MutableOutputDesc(0U);
      GE_ASSERT_SUCCESS(CopyTensorDesc(trans_out_tensor_desc, node_in_tensor_desc));
      GE_ASSERT_SUCCESS(CopyTensorDesc(trans_out_tensor_desc, node_out_tensor_desc));
    }
    const auto &link_node = path.back();
    if (link_node.in_anchor->GetPeerOutAnchor() != link_node.real_peer_out_anchor) {
      GE_ASSERT_SUCCESS(UpdateTensorDescForDiffGraph(trans_out_tensor_desc, link_node));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::UpdateTensorDescForDiffGraph(
    const GeTensorDesc &trans_out_tensor_desc, const LinkNode &link_node) {
  std::stack<LinkNode> link_node_stack;
  link_node_stack.push(link_node);
  while (!link_node_stack.empty()) {
    const auto cur_link_node = link_node_stack.top();
    link_node_stack.pop();
    const auto &pre_node = cur_link_node.in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    GE_ASSERT_NOTNULL(pre_node->GetOpDescBarePtr());
    const auto pre_node_type = GetNodeType(pre_node);
    GE_ASSERT_TRUE((pre_node_type == NodeType::kData) || (pre_node_type == NodeType::kWrapperNode),
                   "current node %s must connect to Data or Wrapper, but pre_node %s node type is %u",
                   cur_link_node.in_anchor->GetOwnerNodeBarePtr()->GetNamePtr(), pre_node->GetNamePtr(),
                   static_cast<uint32_t>(pre_node_type));

    if (pre_node_type == NodeType::kData) {
      GE_ASSERT_SUCCESS(UpdateTensorDescForConnectData(trans_out_tensor_desc, cur_link_node, link_node_stack));
    } else {
      GE_ASSERT_SUCCESS(UpdateTensorDescForConnectWrapper(trans_out_tensor_desc, cur_link_node, link_node_stack));
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::UpdateTensorDescForConnectData(
    const GeTensorDesc &trans_out_tensor_desc, const LinkNode &link_node, std::stack<LinkNode> &link_node_stack) const {
  const auto &owner_node = link_node.in_anchor->GetOwnerNode();
  const auto owner_graph = owner_node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph->GetParentNode());

  auto out_anchor = link_node.in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  const auto data_op_desc = out_anchor->GetOwnerNode()->GetOpDesc();
  GE_ASSERT_NOTNULL(data_op_desc);

  auto data_in_tensor_desc = data_op_desc->MutableInputDesc(0U);
  auto data_out_tensor_desc = data_op_desc->MutableOutputDesc(0U);
  GE_ASSERT_SUCCESS(CopyTensorDesc(trans_out_tensor_desc, data_in_tensor_desc));
  GE_ASSERT_SUCCESS(CopyTensorDesc(trans_out_tensor_desc, data_out_tensor_desc));

  uint32_t parent_index;
  GE_ASSERT_TRUE(AttrUtils::GetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index));
  const auto wrapper_op_desc = owner_graph->GetParentNode()->GetOpDesc();
  GE_ASSERT_NOTNULL(wrapper_op_desc);
  auto wrapper_in_tensor_desc = wrapper_op_desc->MutableInputDesc(parent_index);
  GE_ASSERT_NOTNULL(wrapper_in_tensor_desc);
  GE_ASSERT_SUCCESS(CopyTensorDesc(trans_out_tensor_desc, wrapper_in_tensor_desc));

  const auto wrapper_in_anchor = owner_graph->GetParentNode()->GetInDataAnchor(parent_index);
  GE_ASSERT_NOTNULL(wrapper_in_anchor);
  const auto peer_out_anchor = wrapper_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  if (peer_out_anchor != link_node.real_peer_out_anchor) {
    const LinkNode new_link_node{wrapper_in_anchor, link_node.real_peer_out_anchor, NodeType::kWrapperNode};
    link_node_stack.push(new_link_node);
  }

  // data节点，对应父节点的输入有可能是新增的，在这里更新datatype
  const auto real_peer_node = link_node.real_peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(real_peer_node);
  GE_ASSERT_NOTNULL(real_peer_node->GetOpDesc());
  const auto &real_src_tensor_desc = real_peer_node->GetOpDesc()->MutableOutputDesc(
      static_cast<uint32_t>(link_node.real_peer_out_anchor->GetIdx()));
  GE_ASSERT_NOTNULL(real_src_tensor_desc);

  UpdateDataType(real_src_tensor_desc, wrapper_in_tensor_desc);
  UpdateDataType(real_src_tensor_desc, data_in_tensor_desc);
  UpdateDataType(real_src_tensor_desc, data_out_tensor_desc);
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::UpdateTensorDescForConnectWrapper(
    const GeTensorDesc &trans_out_tensor_desc, const LinkNode &link_node, std::stack<LinkNode> &link_node_stack) {
  GE_ASSERT_NOTNULL(link_node.in_anchor->GetPeerOutAnchor());
  const auto &wrapper_node = link_node.in_anchor->GetPeerOutAnchor()->GetOwnerNode();
  const auto &wrapper_op_desc = wrapper_node->GetOpDesc();
  GE_ASSERT_NOTNULL(wrapper_op_desc);

  const auto wrapper_node_output_index = link_node.in_anchor->GetPeerOutAnchor()->GetIdx();
  auto wrapper_node_output_tensor_desc = wrapper_op_desc->MutableOutputDesc(wrapper_node_output_index);
  GE_ASSERT_SUCCESS(CopyTensorDesc(trans_out_tensor_desc, wrapper_node_output_tensor_desc));

  // Netoutput的输入，对应父节点的输出有可能是新增的，在这里更新datatype
  const auto &real_in_tensor_desc = link_node.in_anchor->GetOwnerNode()->GetOpDesc()->MutableInputDesc(
      static_cast<uint32_t>(link_node.in_anchor->GetIdx()));
  GE_ASSERT_NOTNULL(real_in_tensor_desc);
  UpdateDataType(real_in_tensor_desc, wrapper_node_output_tensor_desc);

  const auto &sub_graph_names = wrapper_op_desc->GetSubgraphInstanceNames();
  GE_ASSERT_TRUE(!sub_graph_names.empty());

  for (const auto &sub_graph_name : sub_graph_names) {
    const auto &sub_graph = root_graph_->GetSubgraph(sub_graph_name);
    GE_ASSERT_NOTNULL(sub_graph);
    const auto &netoutput = sub_graph->FindFirstNodeMatchType(NETOUTPUT);
    GE_ASSERT_NOTNULL(netoutput);
    GE_ASSERT_NOTNULL(netoutput->GetOpDesc());
    const auto &netoutput_op_desc = netoutput->GetOpDesc();
    const auto input_size = netoutput->GetAllInDataAnchorsSize();
    uint32_t netoutput_input_index = input_size;
    for (uint32_t i = 0U; i < input_size; ++i) {
      const auto input_desc = netoutput_op_desc->GetInputDesc(i);
      uint32_t parent_index = 0U;
      if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)
          || (parent_index != static_cast<uint32_t>(wrapper_node_output_index))) {
        continue;
      }
      netoutput_input_index = i;
    }
    GE_ASSERT_TRUE(netoutput_input_index != input_size,
                   "can not find correspond input. netoutput: %s input_size: %u, parent_node: %s, parent output: %u",
                   netoutput->GetNamePtr(), input_size, wrapper_node->GetNamePtr(), wrapper_node_output_index);
    auto netoutput_in_tensor_desc = netoutput_op_desc->MutableInputDesc(netoutput_input_index);
    GE_ASSERT_SUCCESS(CopyTensorDesc(trans_out_tensor_desc, netoutput_in_tensor_desc));
    UpdateDataType(real_in_tensor_desc, netoutput_in_tensor_desc);

    const auto netoutput_in_anchor = netoutput->GetInDataAnchor(netoutput_input_index);
    GE_ASSERT_NOTNULL(netoutput_in_anchor);
    const auto peer_out_anchor = netoutput_in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    if (peer_out_anchor != link_node.real_peer_out_anchor) {
      const LinkNode new_link_node{netoutput_in_anchor, link_node.real_peer_out_anchor, NodeType::kNetOutput};
      link_node_stack.push(new_link_node);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::ExtractTransdata(const Paths &paths_group,
                                                             size_t keep_transdata_path_index) const {
  const auto &path = paths_group[keep_transdata_path_index];
  auto trans_in_anchor = path.back().in_anchor;
  GELOGI("keep transdata[%s]", trans_in_anchor->GetOwnerNode()->GetNamePtr());

  const auto &head_out_anchor = path.front().real_peer_out_anchor;
  if (head_out_anchor == trans_in_anchor->GetPeerOutAnchor()) {
    GE_ASSERT_SUCCESS(LinkHeadToTransdata(paths_group, keep_transdata_path_index));
    return GRAPH_SUCCESS;
  }

  auto trans = trans_in_anchor->GetOwnerNode();
  auto trans_out_anchor = trans->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(trans_out_anchor);
  GE_ASSERT_NOTNULL(trans_in_anchor->GetPeerOutAnchor());
  auto trans_pre = trans_in_anchor->GetPeerOutAnchor()->GetOwnerNode();
  GE_ASSERT_SUCCESS(MoveTransdataInCtrlEdge(trans));
  GE_ASSERT_SUCCESS(GraphUtils::MoveOutCtrlEdges(trans, trans_pre));
  GE_ASSERT_SUCCESS(MoveTransdataOutDataEdge(trans));

  GE_ASSERT_SUCCESS(LinkHeadToTransdata(paths_group, keep_transdata_path_index));
  auto head = head_out_anchor->GetOwnerNode();
  if (head->GetOwnerComputeGraphBarePtr() != trans->GetOwnerComputeGraphBarePtr()) {
    head->GetOwnerComputeGraphBarePtr()->AddNode(trans);
    GraphUtils::RemoveJustNode(trans->GetOwnerComputeGraph(), trans);
    trans->SetOwnerComputeGraph(head->GetOwnerComputeGraph());
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::LinkHeadToTransdata(const Paths &paths_group,
                                                                size_t keep_transdata_path_index) const {
  const auto &path = paths_group[keep_transdata_path_index];
  const auto head_out_anchor = path.front().real_peer_out_anchor;

  const auto trans_in_anchor = path.back().in_anchor;
  const auto trans = trans_in_anchor->GetOwnerNode();
  auto trans_out_anchor = trans->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(trans_out_anchor);

  std::set<InDataAnchorPtr> allowed_in_anchors;
  for (const auto &cur_path : paths_group) {
    allowed_in_anchors.emplace(cur_path.front().in_anchor);
  }
  for (auto &peer_in_anchor : head_out_anchor->GetPeerInDataAnchors()) {
    const auto head_next = peer_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(head_next->GetOpDescBarePtr());
    const auto head_next_type = GetNodeType(head_next);
    if ((head_next_type != NodeType::kWrapperNode) && (head_next_type != NodeType::kNetOutput)) {
      if (allowed_in_anchors.find(peer_in_anchor) != allowed_in_anchors.end()) {
        peer_in_anchor->UnlinkAll();
        if (peer_in_anchor->GetOwnerNode() != trans) {
          GE_ASSERT_SUCCESS(trans_out_anchor->LinkTo(peer_in_anchor));
        }
      }
      continue;
    }
    std::vector<InDataAnchorPtr> fused_anchors;
    std::vector<InDataAnchorPtr> not_fused_anchors;
    GE_ASSERT_SUCCESS(CollectFusedInAnchors(peer_in_anchor, allowed_in_anchors, head_next_type,
                                            fused_anchors, not_fused_anchors));
    (void) not_fused_anchors;
    if (!fused_anchors.empty()) {
      peer_in_anchor->UnlinkAll();
      GE_ASSERT_SUCCESS(trans_out_anchor->LinkTo(peer_in_anchor));
    }
  }
  if (!head_out_anchor->IsLinkedWith(trans_in_anchor)) {
    if (trans_in_anchor->GetPeerOutAnchor() != nullptr) {
      GE_ASSERT_SUCCESS(trans_in_anchor->Unlink(trans_in_anchor->GetPeerOutAnchor()));
    }
    GE_ASSERT_SUCCESS(head_out_anchor->LinkTo(trans_in_anchor));
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::CollectFusedInAnchors(
    const InDataAnchorPtr &in_anchor, const std::set<InDataAnchorPtr> &allowed_in_anchors,
    const NodeType head_next_type, std::vector<InDataAnchorPtr> &fused_anchors,
    std::vector<InDataAnchorPtr> &not_fused_anchors) const {
  std::queue<Path> path_queue;
  std::stack<OutDataAnchorPtr> out_anchor_stack;
  if (head_next_type == NodeType::kWrapperNode) {
    GE_ASSERT_SUCCESS(GetRealInAnchorsForWrapperNode(in_anchor, out_anchor_stack));
  } else {
    GE_ASSERT_SUCCESS(GetRealInAnchorsForNetOutput(nullptr, in_anchor, Path(), out_anchor_stack));
  }
  while (!out_anchor_stack.empty()) {
    const auto cur_out_anchor = out_anchor_stack.top();
    out_anchor_stack.pop();
    for (const auto &cur_in_anchor : cur_out_anchor->GetPeerInDataAnchors()) {
      const auto next_node = cur_in_anchor->GetOwnerNode();
      if ((next_node != nullptr) && (next_node->GetOpDesc() != nullptr)) {
        if (IsWrapperNode(next_node->GetOpDesc())) {
          GE_ASSERT_SUCCESS(GetRealInAnchorsForWrapperNode(cur_in_anchor, out_anchor_stack));
          continue;
        }
        if (next_node->GetType() == NETOUTPUT) {
          GE_ASSERT_SUCCESS(GetRealInAnchorsForNetOutput(nullptr, cur_in_anchor, Path(), out_anchor_stack));
          continue;
        }
        if (allowed_in_anchors.find(cur_in_anchor) == allowed_in_anchors.end()) {
          not_fused_anchors.emplace_back(cur_in_anchor);
        } else {
          fused_anchors.emplace_back(cur_in_anchor);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SameTransdataBreadthFusionPass::DeleteTransdata(const Path &path) const {
  auto trans_in_anchor = path.back().in_anchor;
  auto trans = trans_in_anchor->GetOwnerNode();
  auto trans_out_anchor = path.back().in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(trans_out_anchor);
  auto trans_pre = trans_out_anchor->GetOwnerNode();

  GE_ASSERT_SUCCESS(MoveTransdataInCtrlEdge(trans));
  GE_ASSERT_SUCCESS(GraphUtils::MoveOutCtrlEdges(trans, trans_pre));
  GE_ASSERT_SUCCESS(MoveTransdataOutDataEdge(trans));
  GE_ASSERT_SUCCESS(trans_in_anchor->Unlink(trans_in_anchor->GetPeerOutAnchor()));
  GE_ASSERT_SUCCESS(GraphUtils::RemoveJustNode(trans->GetOwnerComputeGraph(), trans));
  return GRAPH_SUCCESS;
}

REG_PASS_OPTION("SameTransdataBreadthFusionPass").LEVELS(OoLevel::kO3);
} // namespace ge
