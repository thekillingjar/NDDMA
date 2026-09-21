/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/graph/fe_graph_utils.h"
#include "common/fe_type_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/platform_utils.h"

namespace fe {
namespace {
bool HasPeerOutNode(const ge::Node *node, const int &anchor_index,
                    ge::NodePtr &peer_out_node) {
  auto in_anchor = node->GetInDataAnchor(anchor_index);
  FE_CHECK(in_anchor == nullptr, FE_LOGW("index:%d in_anchor is nullptr",
           anchor_index), return false);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  FE_CHECK(peer_out_anchor == nullptr, FE_LOGW("index:%d peer_out_anchor is nullptr",
           anchor_index), return false);
  peer_out_node = peer_out_anchor->GetOwnerNode();
  FE_CHECK(peer_out_node == nullptr, FE_LOGW("index:%d peer_out_anchor is nullptr",
           anchor_index), return false);
  return true;
}
} // namespace

void FeGraphUtils::DumpSubGraphAndOnnx(const ge::ComputeGraph &graph, const std::string &suffix) {
  for (auto subgraph : graph.GetAllSubgraphs()) {
    DumpGraphAndOnnx(*subgraph, suffix);
  }
}

void FeGraphUtils::DumpGraphAndOnnx(const ge::ComputeGraph &graph, const std::string &suffix) {
  DumpGraph(graph, suffix);
  ge::GraphUtils::DumpGEGraphToOnnx(graph, suffix);
}

void FeGraphUtils::DumpGraph(const ge::ComputeGraph &graph, const std::string &suffix) {
  std::shared_ptr<ge::ComputeGraph> compute_graph_ptr = FeComGraphMakeShared<ge::ComputeGraph>(graph);
  ge::GraphUtils::DumpGEGraph(compute_graph_ptr, suffix);
}

bool FeGraphUtils::IsMainGraphData(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return false;
  }
  return IsRootGraphData(op_desc_ptr->GetType()) && !IsSubGraphData(op_desc_ptr);
}

bool FeGraphUtils::IsMainGraphNetOutput(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return false;
  }
  return op_desc_ptr->GetType() == NETOUTPUT && !IsSubGraphNetOutput(op_desc_ptr);
}

bool FeGraphUtils::IsSubGraphDataOrNetOutput(const ge::OpDescPtr &op_desc_ptr) {
  return IsSubGraphData(op_desc_ptr) || IsSubGraphNetOutput(op_desc_ptr);
}

bool FeGraphUtils::IsNotSubGraphDataAndNetOutput(const ge::OpDescPtr &op_desc_ptr) {
  return !IsSubGraphData(op_desc_ptr) && !IsSubGraphNetOutput(op_desc_ptr);
}

bool FeGraphUtils::IsSubGraphData(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr || op_desc_ptr->GetType() != DATA) {
    return false;
  }
  return op_desc_ptr->HasAttr(ge::ATTR_NAME_PARENT_NODE_INDEX);
}

bool FeGraphUtils::IsSubGraphNetOutput(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr || op_desc->GetType() != NETOUTPUT) {
    return false;
  }
  for (auto &tensor : op_desc->GetAllInputsDescPtr()) {
    if (ge::AttrUtils::HasAttr(tensor, ge::ATTR_NAME_PARENT_NODE_INDEX)) {
      return true;
    }
  }
  return false;
}

Status FeGraphUtils::GetPreOutAnchorOfSubData(const ge::NodePtr &data_node_ptr,
                                              ge::OutDataAnchorPtr &pre_out_data_anchor_ptr) {
  FE_CHECK_NOTNULL(data_node_ptr);
  ge::OpDescPtr data_op_desc_ptr = data_node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(data_op_desc_ptr);
  uint32_t parent_node_index = 0;
  if (!ge::AttrUtils::GetInt(data_op_desc_ptr, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][GetPreOutAncr] attr %s is missing for node %s",
                    ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), data_op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  auto owner_graph = data_node_ptr->GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);

  ge::NodePtr parent_node_ptr = owner_graph->GetParentNode();
  FE_CHECK_NOTNULL(parent_node_ptr);
  ge::InDataAnchorPtr in_data_anchor_ptr = parent_node_ptr->GetInDataAnchor(parent_node_index);
  FE_CHECK_NOTNULL(in_data_anchor_ptr);
  pre_out_data_anchor_ptr = in_data_anchor_ptr->GetPeerOutAnchor();
  return SUCCESS;
}

Status FeGraphUtils::GetPreSubNetoutputInAnchor(std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections,
                                                std::vector<ge::InDataAnchorPtr> &vec_netoutput_in_ahchor) {
  for (const auto &cell : reflections) {
    if ((cell.in_out != ge::NODE_IN) || (cell.node->GetType() != NETOUTPUT)) {
      continue;
    }

    for (auto &in_anchor : cell.node->GetAllInDataAnchors()) {
      FE_CHECK_NOTNULL(in_anchor);
      if (in_anchor->GetIdx() == cell.in_out_idx) {
        vec_netoutput_in_ahchor.push_back(in_anchor);
        break;
      }
    }
  }

  if (vec_netoutput_in_ahchor.empty()) {
    return FAILED;
  }

  return SUCCESS;
}

Status FeGraphUtils::GetNextInAnchorsOfSubNetOutput(const ge::NodePtr &net_output_node_ptr, const int &input_index,
                                                    std::vector<ge::InDataAnchorPtr> &next_in_data_anchors) {
  FE_CHECK_NOTNULL(net_output_node_ptr);
  ge::OpDescPtr op_desc_ptr = net_output_node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto input_desc = op_desc_ptr->GetInputDescPtr(input_index);
  uint32_t parent_index = -1;
  if (!ge::AttrUtils::GetInt(input_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    FE_LOGE("");
    return FAILED;
  }

  auto owner_graph = net_output_node_ptr->GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);
  ge::NodePtr parent_node_ptr = owner_graph->GetParentNode();
  FE_CHECK_NOTNULL(parent_node_ptr);
  ge::OutDataAnchorPtr out_data_anchor_ptr = parent_node_ptr->GetOutDataAnchor(parent_index);
  FE_CHECK_NOTNULL(out_data_anchor_ptr);
  for (auto it : out_data_anchor_ptr->GetPeerInDataAnchors()) {
    next_in_data_anchors.push_back(it);
  }
  return SUCCESS;
}

Status FeGraphUtils::GetNextSubDatasOutAnchors(std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections,
                                               std::vector<ge::OutDataAnchorPtr> &out_data_anchors) {
  for (const auto &cell : reflections) {
    if ((cell.in_out != ge::NODE_OUT) || (cell.node->GetType() != DATA)) {
      continue;
    }

    for (auto &out_anchor : cell.node->GetAllOutDataAnchors()) {
      FE_CHECK_NOTNULL(out_anchor);
      if (out_anchor->GetIdx() == cell.in_out_idx) {
        out_data_anchors.push_back(out_anchor);
        break;
      }
    }
  }

  if (out_data_anchors.empty()) {
    return FAILED;
  }
  return SUCCESS;
}

Status FeGraphUtils::UpdateFormatOfRelatedEdges(const std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections,
                                                const RelationUpdateInfo &relation_update_info_a) {
  FE_LOGD("relationUpdateInfo: primary_format=[%s], sub_format=[%d], shape=[%s].",
          ge::TypeUtils::FormatToSerialString(relation_update_info_a.primary_format).c_str(),
          relation_update_info_a.sub_format, GetShapeDims(relation_update_info_a.shape).c_str());

  for (const auto &cell : reflections) {
    ge::NodePtr node_ptr = cell.node;
    FE_CHECK_NOTNULL(node_ptr);
    ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);
    auto owner_graph = node_ptr->GetOwnerComputeGraph();
    FE_CHECK_NOTNULL(owner_graph);
    string graph_name = owner_graph->GetName();

    string node_name = node_ptr->GetName();
    FE_LOGD("Graph[%s]Op[type=%s,name=%s]: cell.in_out_idx=[%d], cell.in_out=[%d].", graph_name.c_str(),
            node_ptr->GetType().c_str(), node_name.c_str(), cell.in_out_idx, cell.in_out);

    // 1. get the input or output desc
    auto index = cell.in_out_idx;
    auto desc = (cell.in_out == ge::NODE_IN ? op_desc_ptr->GetInputDesc(static_cast<uint32_t>(index))
                                            : op_desc_ptr->GetOutputDesc(static_cast<uint32_t>(index)));

    // 2. set the format
    string input_or_output = cell.in_out == ge::NODE_IN ? STR_INPUT_LOWERCASE : STR_OUTPUT_LOWERCASE;
    if (relation_update_info_a.primary_format != ge::FORMAT_RESERVED) {
      ge::Format cur_format = desc.GetFormat();
      ge::GeShape cur_shape = desc.GetShape();
      int32_t c0_bit_val = GetC0BitByDataType(desc.GetDataType());
      auto new_format = static_cast<ge::Format>(
          ge::GetFormatFromSubAndC0(relation_update_info_a.primary_format, relation_update_info_a.sub_format,
                                    c0_bit_val));
      desc.SetFormat(new_format);
      desc.SetShape(relation_update_info_a.shape);

      FE_LOGD(
          "Graph[%s]Op[type=%s,name=%s]: update the %s %d desc, cur_format=[%s], cur_shape=[%s], new_format=[%s], "
          "newShape=[%s].",
          graph_name.c_str(), node_ptr->GetType().c_str(), node_name.c_str(), input_or_output.c_str(), index,
          ge::TypeUtils::FormatToSerialString(cur_format).c_str(), GetShapeDims(cur_shape).c_str(),
          ge::TypeUtils::FormatToSerialString(new_format).c_str(), GetShapeDims(relation_update_info_a.shape).c_str());
    }

    // 3. set the attribute for the tensor desc of function op
    // sub graph data and netoutput will not be set INFERFORMAT
    if (!relation_update_info_a.attr_name.empty() &&
        (op_desc_ptr->GetType() != DATA && op_desc_ptr->GetType() != NETOUTPUT)) {
      (void)ge::AttrUtils::SetInt(desc, relation_update_info_a.attr_name, relation_update_info_a.attr_value);
    }

    // 4. update the tensor desc
    if (cell.in_out == ge::NODE_IN) {
      (void)op_desc_ptr->UpdateInputDesc(static_cast<uint32_t>(index), desc);
    } else {
      (void)op_desc_ptr->UpdateOutputDesc(static_cast<uint32_t>(index), desc);
    }
  }
  return SUCCESS;
}

bool FeGraphUtils::CheckRelatedEdgesOriginShape(const std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections) {
  int init_flag = 0;
  vector<int64_t> ref_origin_shape_dims;
  for (const auto &cell : reflections) {
    ge::NodePtr node_ptr = cell.node;
    FE_CHECK_NOTNULL(node_ptr);
    ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);
    auto owner_graph = node_ptr->GetOwnerComputeGraph();
    FE_CHECK_NOTNULL(owner_graph);
    string graph_name = owner_graph->GetName();
    string node_name = node_ptr->GetName();

    string input_output = cell.in_out == ge::NODE_IN ? STR_INPUT_LOWERCASE : STR_OUTPUT_LOWERCASE;
    FE_LOGD("Relations context: the %s %d of Graph[%s]Op[%s].", input_output.c_str(),
            cell.in_out_idx, graph_name.c_str(), node_name.c_str());

    auto index = cell.in_out_idx;
    auto desc = (cell.in_out == ge::NODE_IN ? op_desc_ptr->GetInputDescPtr(static_cast<uint32_t>(index))
                                            : op_desc_ptr->GetOutputDescPtr(static_cast<uint32_t>(index)));
    if (desc == nullptr) {
      return false;
    }
    vector<int64_t> origin_shape_dims = desc->GetOriginShape().GetDims();
    if (init_flag == 0) {
      ref_origin_shape_dims = origin_shape_dims;
      init_flag = 1;
    } else {
      if (ref_origin_shape_dims != origin_shape_dims) {
        FE_LOGD("Relations: the %s %d of Graph[%s]Op[%s], shape is not equal.", input_output.c_str(), cell.in_out_idx,
                graph_name.c_str(), node_name.c_str());
        return false;
      }
    }
  }
  return true;
}

void FeGraphUtils::GetGraphIdFromAttr(const ge::ComputeGraph &graph, string &graph_id) {
  string session_graph_id = "";
  if (ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) && !session_graph_id.empty()) {
    size_t pos = session_graph_id.find('_');
    if (pos != string::npos && pos + 1 < session_graph_id.size()) {
      graph_id = session_graph_id.substr(pos + 1);
    }
  }
  FE_LOGD("Get session_graph_id=%s graph_id=%s.", session_graph_id.c_str(), graph_id.c_str());
}

bool FeGraphUtils::CheckTypeOnRootGraph(const std::unordered_set<string> &types, ge::NodePtr &parent_node) {
  ge::NodePtr really_parent_node = nullptr;
  if (ge::NodeUtils::GetInNodeCrossPartionedCallNode(parent_node, 0, really_parent_node) != SUCCESS) {
    FE_LOGW(
        "[SubGraphOpt][PreCompileOp][SetTensorConstVal] Node[%s, %s]: failed to getInNodeCrossPartionedCallNode.",
        parent_node->GetName().c_str(), parent_node->GetType().c_str());
    return false;
  }
  if (really_parent_node != nullptr) {
    std::string node_type = really_parent_node->GetType();
    FE_LOGD("Parent_node:%s type:%s really_parent_node:%s type:%s", parent_node->GetName().c_str(),
            parent_node->GetType().c_str(), really_parent_node->GetName().c_str(),
            really_parent_node->GetType().c_str());
    parent_node = really_parent_node;
    return types.count(node_type);
  } else {
    FE_LOGD("real parent node for %s is null.", parent_node->GetName().c_str());
  }
  return false;
}

void FeGraphUtils::ProcessPartitionedCall(const std::string &name, std::string &type, ge::NodePtr &parent_node,
                                          ge::NodePtr &really_parent_node, ge::NodePtr &node) {
  if (type != PARTITIONEDCALL) {
    return;
  }
  const auto &func_graph = parent_node->GetOwnerComputeGraph();
  FE_CHECK(func_graph == nullptr, FE_LOGW("GetOwnerComputeGraph Failed."), return);
  const auto &src_graph = func_graph->TryGetExtAttr(kPartSrcGraph, ge::ComputeGraphPtr());
  FE_CHECK(src_graph == nullptr, FE_LOGW("TryGetExtAttr Failed."), return);
  const auto &root_graph = ge::GraphUtils::FindRootGraph(src_graph);
  FE_CHECK(root_graph == nullptr, FE_LOGW("FindRootGraph Failed"), return);
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    if (subgraph->GetParentNode() == nullptr) {
      continue;
    }
    if (subgraph->GetParentNode()->GetName() == parent_node->GetName()) {
      const auto &net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
      FE_CHECK(net_output_node == nullptr, FE_LOGW("GetSubgraph Failed"), return);
      int32_t parent_node_anchor_index;
      if (!ge::AttrUtils::GetInt(node->GetOpDesc(), "anchorIndex", parent_node_anchor_index)) {
        FE_LOGW("Node [%s] failed to get anchorIndex.", name.c_str());
        return;
      }
      auto in_node = ge::NodeUtils::GetInDataNodeByIndex(*net_output_node, parent_node_anchor_index);
      if (in_node == nullptr) { FE_LOGW("in_node is nullptr!"); return; }
      FE_LOGD("[SubGraphOpt][IsNodeSpecType][in_node %s, type %s]", in_node->GetName().c_str(),
              in_node->GetType().c_str());
      type = in_node->GetType();
      parent_node = in_node;
      really_parent_node = parent_node;
      break;
    }
  }
}

void FeGraphUtils::IsNodeSpecificType(const std::unordered_set<string> &types,
                                      ge::NodePtr &node, bool &matched) {
  auto type = node->GetType();
  auto name = node->GetName();
  matched = types.count(type) != 0;
  if (matched) {
    return;
  }
  // if it is placeholder, get its parent node
  if (type == OP_TYPE_PLACE_HOLDER) {
    ge::NodePtr parent_node = nullptr;
    parent_node = node->GetOpDesc()->TryGetExtAttr(ATTR_NAME_PARENT_NODE, parent_node);
    if (parent_node != nullptr) {
      type = parent_node->GetType();
      FE_LOGD("The parent node of place holder[%s] is [%s, %s].", name.c_str(),
              parent_node->GetName().c_str(), parent_node->GetType().c_str());
      ge::NodePtr really_parent_node = parent_node;
      ProcessPartitionedCall(name, type, parent_node, really_parent_node, node);
      bool parent_node_invalid = (types.count(type) == 0 &&
                                  ge::NodeUtils::GetInNodeCrossPartionedCallNode(parent_node, 0,
                                                                                 really_parent_node) != SUCCESS);
      if (parent_node_invalid) {
        FE_LOGW("[SubGraphOpt][IsNodeSpecType][Op %s, type %s]: Failed to getInNodeCrossPartionedCallNode.",
                name.c_str(), type.c_str());
        return;
      }
      if (really_parent_node != nullptr) {
        node = really_parent_node;
        type = really_parent_node->GetType();
        FE_LOGD("Parent node:%s type:%s really parent node:%s type:%s.", parent_node->GetName().c_str(),
                parent_node->GetType().c_str(), really_parent_node->GetName().c_str(),
                really_parent_node->GetType().c_str());
      }
      matched = types.count(type);
    }
  } else if (FeGraphUtils::IsSubGraphData(node->GetOpDesc())) {
    matched = FeGraphUtils::CheckTypeOnRootGraph(types, node);
  } else {
    FE_LOGD("Cannot match any types for node %s and type %s.", node->GetName().c_str(), type.c_str());
  }
}

bool FeGraphUtils::IsPeerOutConst(const ge::Node *node, const int &anchor_index,
                                  ge::NodePtr &peer_out_node) {
  if (node == nullptr) {
    return false;
  }
  auto op_desc = node->GetOpDesc();
  bool has_other_node = HasPeerOutNode(node, anchor_index, peer_out_node);
  if (has_other_node) {
    bool is_const_node = false;
    IsNodeSpecificType(kConstTypes, peer_out_node, is_const_node);
    return is_const_node;
  } else {
    return false;
  }
}

bool FeGraphUtils::IsPeerOutWeight(ge::Node *node, const int &anchor_index,
                                   ge::NodePtr &peer_out_node) {
  if (node == nullptr) {
    return false;
  }
  auto op_desc = node->GetOpDesc();
  bool has_other_node = HasPeerOutNode(node, anchor_index, peer_out_node);
  if (has_other_node) {
    FE_LOGD("[IsPeerOutWeight] Peer out node is %s.", peer_out_node->GetName().c_str());
    bool is_const_node = false;
    IsNodeSpecificType(kWeightTypes, peer_out_node, is_const_node);
    return is_const_node;
  } else {
    return false;
  }
}

Status FeGraphUtils::GetAoeTypeFromRootGraph(ge::ComputeGraph& graph, std::string &aoe_type) {
  auto nodes = graph.GetDirectNode();
  ge::ComputeGraphPtr root_graph;
  const auto &func_graph = nodes.at(0)->GetOwnerComputeGraph();
  FE_CHECK(func_graph == nullptr, FE_LOGW("GetOwnerComputeGraph Failed!"), return FAILED);
  const auto &src_graph = func_graph->TryGetExtAttr(kPartSrcGraph, ge::ComputeGraphPtr());
  if (src_graph == nullptr) {
    root_graph = ge::GraphUtils::FindRootGraph(graph.shared_from_this());
  } else {
    root_graph = ge::GraphUtils::FindRootGraph(src_graph);
  }
  FE_CHECK(root_graph == nullptr, FE_LOGW("FindRootGraph Failed!"), return FAILED);
  if (!ge::AttrUtils::GetStr(*root_graph, AOE_TYPE, aoe_type)) {
    return FAILED;
  }
  return SUCCESS;
}

void FeGraphUtils::FindPeerOpType(const ge::NodePtr &node, const bool is_input, std::string &peer_op_type) {
  if (node == nullptr) {
    return;
  }
  peer_op_type = node->GetType();
  if (kGeDeleteOpType.count(peer_op_type) != 0) {
    ge::Node::Vistor<ge::NodePtr> peer_nodes = is_input ? node->GetInDataNodes() : node->GetOutDataNodes();
    if (!peer_nodes.empty()) {
      FindPeerOpType(peer_nodes.at(0), is_input, peer_op_type);
    }
  }
  if (peer_op_type == OP_TYPE_PLACE_HOLDER || peer_op_type == OP_TYPE_END) {
    ge::NodePtr peer_node = node->GetOpDesc()->TryGetExtAttr<ge::NodePtr>(ATTR_NAME_PARENT_NODE, nullptr);
    if (peer_node != nullptr) {
      FindPeerOpType(peer_node, is_input, peer_op_type);
    } else {
      (void)ge::AttrUtils::GetStr(node->GetOpDesc(), PARENT_OP_TYPE, peer_op_type);
    }
  }
}

void FeGraphUtils::GetPrecisionModeFromGraph(const ge::ComputeGraph& graph, fe::PrecisionMode &precision_mode) {
  int precision_mode_num = -1;
  (void)ge::AttrUtils::GetInt(graph, "graph_precision_mode", precision_mode_num);
  if (precision_mode_num < 0 || precision_mode_num > static_cast<int>(fe::PrecisionMode::ENUM_UNDEFINED)) {
    FEContextUtils::GetPrecisionMode(precision_mode);
    FE_LOGD("[GraphOptJdgInst][GetPrecisionModeFromGraph] The precision mode num %d from graph is invalid.",
            precision_mode_num);
    return;
  }
  precision_mode = static_cast<fe::PrecisionMode>(precision_mode_num);
  FE_LOGD("[GraphOptJdgInst][GetPrecisionModeFromGraph] Get precision mode %d from graph.", precision_mode_num);
}
}  // namespace fe
