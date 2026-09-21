/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/graph_comm.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_log.h"
#include "common/graph_comm_impl.h"
#include "common/sgt_slice_type.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/aicore_util_constants.h"
#include "framework/common/ge_types.h"

namespace fe {
constexpr char const *kOpFixpipe = "FixPipe";
constexpr char const *kOpMerge = "Merge";
constexpr char const *kOpSwitch = "Switch";
constexpr char const *kScopeNodeSet = "ScopeNodeSet";
constexpr uint8_t kMaxDeepth = 8;

GraphComm::GraphComm(const string &engine_name) : engine_name_(engine_name), function_graph_id_(0),
      graph_common_lock_ptr_(nullptr) {
  exist_fusion_src_list_.clear();
  exist_fusion_dst_list_.clear();
  fusion_input_dataflow_list_.clear();
  fusion_output_dataflow_list_.clear();
}

GraphComm::GraphComm(const string &engine_name, std::shared_ptr<std::mutex> &lock_ptr) : engine_name_(engine_name),
      function_graph_id_(0), graph_common_lock_ptr_(lock_ptr) {
  exist_fusion_src_list_.clear();
  exist_fusion_dst_list_.clear();
  fusion_input_dataflow_list_.clear();
  fusion_output_dataflow_list_.clear();
}

GraphComm::~GraphComm() {}

Status GraphComm::Initialize() {
  graph_comm_impl_ptr_ = std::unique_ptr<GraphCommImpl>(new (std::nothrow) GraphCommImpl);
  FE_CHECK_NOTNULL(graph_comm_impl_ptr_);
  return SUCCESS;
}

bool GraphComm::IsTuningMode() const {
  std::string build_mode;
  (void)ge::GetContext().GetOption("ge.buildMode", build_mode);
  FE_LOGD("Get current build mode: %s.", build_mode.c_str());
  return (build_mode == "tuning");
}

Status GraphComm::GetFusionNodeEdgeList(vector<ge::NodePtr> &fus_nodelist,
                                        const std::unordered_set<ge::NodePtr> &node_set,
                                        std::vector<FusionDataFlow> &fus_input_edge_list,
                                        std::vector<FusionDataFlow> &fus_output_edge_list) {
  fus_input_edge_list.clear();
  fus_output_edge_list.clear();
  bool is_tuning_mode = IsTuningMode();

  for (ge::NodePtr &node : fus_nodelist) {
    if (node == nullptr) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNodes] A node in the fusion_node_list is nullptr.");
      return FAILED;
    }
    FE_LOGD("GetFusionNodeEdgeList: name = %s", node->GetName().c_str());
    std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> in_edges;
    if (graph_comm_impl_ptr_->GetAllInEdgeList(node, in_edges, 0, node_set, is_tuning_mode) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNodes] GetAllInEdgeList failed");
      return FAILED;
    }
    int cnt = 0;
    for (auto &in_edge : in_edges) {
      ++cnt;
      FE_LOGD("Node name = %s, cnt = %d.", node->GetName().c_str(), cnt);
      graph_comm_impl_ptr_->PutEdgeToFusionDataFlowVec(in_edge, in_edge.second, in_edge.first,
                                                       fus_input_edge_list);
    }

    std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> out_edges;
    if (graph_comm_impl_ptr_->GetAllOutEdgeList(node, out_edges, 0, node_set) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNodes] GetAllOutEdgeList failed");
      return FAILED;
    }

    for (auto &out_edge : out_edges) {
      graph_comm_impl_ptr_->PutEdgeToFusionDataFlowVec(out_edge, out_edge.first, out_edge.second,
                                                       fus_output_edge_list);
    }
  }

  if (fus_output_edge_list.empty()) {
    std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> out_edges;
    for (ge::NodePtr &node : fus_nodelist) {
      auto out_control_anchor = node->GetOutControlAnchor();
      if (out_control_anchor != nullptr && node->GetOutDataAnchor(0) != nullptr &&
          out_control_anchor->GetPeerInControlAnchors().size() > 0) {
        out_edges.emplace_back(std::make_pair(node->GetOutDataAnchor(0), nullptr));
      }
    }
    for (auto &out_edge : out_edges) {
      graph_comm_impl_ptr_->PutEdgeToFusionDataFlowVec(out_edge, out_edge.first, out_edge.second, fus_output_edge_list);
    }
  }
  FE_LOGD("Get InputListSize: %zu, OutputListSize: %zu.", fus_input_edge_list.size(), fus_output_edge_list.size());
  return SUCCESS;
}

Status GraphComm::GetFusionNodeCtrlEdgeList(vector<ge::NodePtr> &fus_nodelist,
                                            const std::unordered_set<ge::NodePtr> &node_set,
                                            std::vector<FusionDataFlow> &fus_input_ctrl_edge_list,
                                            std::vector<FusionDataFlow> &fus_output_ctrl_edge_list) {
  fus_input_ctrl_edge_list.clear();
  fus_output_ctrl_edge_list.clear();
  bool is_tuning_mode = IsTuningMode();
  std::unordered_set<ge::NodePtr> peer_node_set;
  for (ge::NodePtr &node : fus_nodelist) {
    FE_CHECK(node == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNdCtrlList] The node in the fusion_node_list is nullptr."),
             return FAILED);

    std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> in_edges;
    if (graph_comm_impl_ptr_->GetAllInEdgeList(node, in_edges, 1, node_set, is_tuning_mode) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNdCtrlList] GetAllInEdgeList failed");
      return FAILED;
    }
    for (auto &in_edge : in_edges) {
      if (in_edge.first != nullptr) {
        if (peer_node_set.count(in_edge.first->GetOwnerNode()) != 0) {
          continue;
        }
        peer_node_set.emplace(in_edge.first->GetOwnerNode());
      }
      graph_comm_impl_ptr_->PutEdgeToFusionDataFlowVec(in_edge, in_edge.second, in_edge.first,
                                                       fus_input_ctrl_edge_list);
    }

    std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> out_edges;
    if (graph_comm_impl_ptr_->GetAllOutEdgeList(node, out_edges, 1, node_set) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNdCtrlList] GetAllOutEdgeList failed");
      return FAILED;
    }
    for (auto &out_edge : out_edges) {
      graph_comm_impl_ptr_->PutEdgeToFusionDataFlowVec(out_edge, out_edge.first, out_edge.second,
                                                       fus_output_ctrl_edge_list);
    }
  }
  FE_LOGD("Get InputListSize: %zu, OutputListSize: %zu.", fus_input_ctrl_edge_list.size(),
          fus_output_ctrl_edge_list.size());
  return SUCCESS;
}

Status GraphComm::GetscopeNodeMap(const ge::ComputeGraph &graph, ScopeNodeMap &fusion_map) {
  int64_t scope_id = 0;

  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr->HasAttr(SCOPE_ID_ATTR) && ge::AttrUtils::GetInt(op_desc_ptr, SCOPE_ID_ATTR, scope_id) &&
        scope_id >= 0) {
      graph_comm_impl_ptr_->SaveFusionNode(scope_id, node, fusion_map);
    }
  }
  return SUCCESS;
}

void GraphComm::ClearFusionSrc() {
  exist_fusion_src_list_.clear();
  fusion_input_dataflow_list_.clear();
  fusion_output_dataflow_list_.clear();
}

void GraphComm::ClearFusionDst() { exist_fusion_dst_list_.clear(); }

// ge::AnchorPtr replace index
bool GraphComm::GetFusionSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor, int32_t &fusion_src_index,
                             int32_t &fusion_dst_index) {
  for (auto &item : exist_fusion_src_list_) {
    if ((item.src_op_id == src_op_id) && (item.src_anchor == src_anchor)) {
      fusion_src_index = item.fusion_src_index;
      fusion_dst_index = item.fusion_dst_index;
      return true;
    }
  }

  return false;
}

bool GraphComm::IsFusionDstExist(const uint32_t &dst_op_id, const ge::AnchorPtr &dst_anchor) {
  for (auto &item : exist_fusion_dst_list_) {
    if ((item.dst_op_id == dst_op_id) && (item.dst_anchor == dst_anchor)) {
      return true;
    }
  }
  return false;
}

void GraphComm::AddFusionInputSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor,
                                  const int32_t &fusion_dst_index,
                                  const std::pair<string, ge::AnchorPtr> &node_dataindex_pair) {
  graph_comm_impl_ptr_->AddFusionSrc(src_op_id, src_anchor, 0, fusion_dst_index, exist_fusion_src_list_);
  std::multimap<string, ge::AnchorPtr> flowmap;
  (void)flowmap.insert(node_dataindex_pair);
  fusion_input_dataflow_list_.emplace_back(flowmap);
}

void GraphComm::AddFusionOutputSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor,
                                   const int32_t &fusion_src_index,
                                   const std::pair<string, ge::AnchorPtr> &node_dataindex_pair) {
  graph_comm_impl_ptr_->AddFusionSrc(src_op_id, src_anchor, fusion_src_index, 0, exist_fusion_src_list_);
  std::multimap<string, ge::AnchorPtr> flowmap;
  (void)flowmap.insert(node_dataindex_pair);
  fusion_output_dataflow_list_.emplace_back(flowmap);
}

void GraphComm::SaveFusionDst(const uint32_t &dst_op_id, const ge::AnchorPtr &dst_anchor) {
  FusionOpDst value;
  value.dst_op_id = dst_op_id;
  value.dst_anchor = dst_anchor;
  exist_fusion_dst_list_.emplace_back(value);
}

Status GraphComm::GetNodeDataFlowMap(
    const ge::NodePtr &fus_node, std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> &fusion_op_anchors_map,
    ge::kFusionDataFlowVec_t &fus_dataflow_list, const int &map_type) {
  // get fusion nodes
  vector<ge::NodePtr> fus_nodelist;
  fus_nodelist = fus_node->GetOpDesc()->TryGetExtAttr(kScopeNodeSet, fus_nodelist);
  if (fus_nodelist.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNdDataFlowMap] TryGetExtAttr ScopeNodeSet failed.");
    return FAILED;
  }

  ge::kFusionDataFlowVec_t fusnode_dataflow_list;
  // in type
  if (map_type == 0) {
    fus_node->GetFusionInputFlowList(fusnode_dataflow_list);
  } else {  // out type
    fus_node->GetFusionOutputFlowList(fusnode_dataflow_list);
  }

  for (ge::NodePtr &node : fus_nodelist) {
    std::map<ge::AnchorPtr, ge::AnchorPtr> tmp_anchor_map;
    (void)fusion_op_anchors_map.insert(std::make_pair(node, tmp_anchor_map));
  }

  for (size_t i = 0; i < fusnode_dataflow_list.size(); i++) {
    std::multimap<std::string, ge::AnchorPtr> data_flow_map = fusnode_dataflow_list[i];
    std::multimap<std::string, ge::AnchorPtr> flow_map;
    for (auto &node_data : data_flow_map) {
      ge::NodePtr node = graph_comm_impl_ptr_->FindNodeInFusNodeList(node_data.first, fus_nodelist);
      FE_CHECK(node == nullptr,
               REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNdDataFlowMap] Failed to find node:%s in fusnodelist.",
               node_data.first.c_str()), return FAILED);
      int anchor_idx = std::static_pointer_cast<ge::DataAnchor>(node_data.second)->GetIdx();

      ge::AnchorPtr anchor_ptr;
      if (map_type == 0) {
        anchor_ptr = node->GetInDataAnchor(anchor_idx);
        (void)fusion_op_anchors_map[node].insert(std::make_pair(anchor_ptr,
                                                                fus_node->GetInDataAnchor(static_cast<int32_t>(i))));
      } else {
        anchor_ptr = node->GetOutDataAnchor(anchor_idx);
        (void)fusion_op_anchors_map[node].insert(std::make_pair(anchor_ptr,
                                                                fus_node->GetOutDataAnchor(static_cast<int32_t>(i))));
      }

      // update old node name and anchor
      (void)flow_map.insert(std::make_pair(node_data.first, anchor_ptr));
      FE_LOGD("node: %s, idx: %d, type: %d, fusopidx: %zu.", node_data.first.c_str(), anchor_idx, map_type, i);
    }
    fus_dataflow_list.push_back(flow_map);
  }

  return SUCCESS;
}

void GraphComm::GetEdgeNode(const vector<FusionDataFlow> &fus_input_edge_list,
                            const vector<FusionDataFlow> &fus_output_edge_list,
                            vector<ge::NodePtr> &edge_nodes) const {
  for (const FusionDataFlow &dataflow : fus_input_edge_list) {
    std::pair<ge::AnchorPtr, ge::AnchorPtr> in_edge = dataflow.edge;
    // get peer node
    if (in_edge.first == nullptr) {
      continue;
    }
    ge::NodePtr peer_node = in_edge.first->GetOwnerNode();
    edge_nodes.push_back(peer_node);
  }

  for (const FusionDataFlow &dataflow : fus_output_edge_list) {
    std::pair<ge::AnchorPtr, ge::AnchorPtr> in_edge = dataflow.edge;
    // get peer node
    if (in_edge.second == nullptr) {
      continue;
    }
    ge::NodePtr peer_node = in_edge.second->GetOwnerNode();
    edge_nodes.push_back(peer_node);
  }
}

Status GraphComm::CopyFusionOpNodes(const vector<FusionDataFlow> &fus_input_edge_list,
                                    const vector<FusionDataFlow> &fus_output_edge_list,
                                    const vector<ge::NodePtr> &fus_nodelist,
                                    const ge::OpDescPtr &fusion_op_desc,
                                    const ge::ComputeGraphPtr &fusion_graph) const {
  vector<ge::NodePtr> edge_nodes;

  GetEdgeNode(fus_input_edge_list, fus_output_edge_list, edge_nodes);

  vector<ge::NodePtr> new_edge_nodelist;
  for (ge::NodePtr &peer_node : edge_nodes) {
    // find exist node
    ge::NodePtr new_peer_node = fusion_graph->FindNode(peer_node->GetName());
    // create new node
    if (new_peer_node == nullptr) {
      ge::OpDescPtr new_op_desc = ge::AttrUtils::CloneOpDesc(peer_node->GetOpDesc());
      new_peer_node = fusion_graph->AddNode(new_op_desc);
      FE_CHECK(new_peer_node == nullptr,
               REPORT_FE_ERROR("[SubGraphOpt][Merge][CpFusOpNd] nodeNew is nullptr"), return FAILED);
      new_edge_nodelist.push_back(new_peer_node);
    }
  }
  (void)fusion_op_desc->SetExtAttr("ScopeEdgeNodeSet", new_edge_nodelist);

  vector<ge::NodePtr> new_fus_nodelist;
  for (const auto &node : fus_nodelist) {
    if (node == nullptr) {
      continue;
    }
    ge::OpDescPtr new_op_desc = ge::AttrUtils::CloneOpDesc(node->GetOpDesc());
    if (new_op_desc == nullptr) {
      continue;
    }
    ge::NodePtr new_node = fusion_graph->AddNode(new_op_desc);
    if (new_node == nullptr) {
      continue;
    }
    // init offset
    size_t input_size = new_op_desc->GetInputsSize();
    size_t output_size = new_op_desc->GetOutputsSize();
    vector<int64_t> input_offset(input_size, -1);
    vector<int64_t> output_offset(output_size, -1);
    new_op_desc->SetInputOffset(input_offset);
    new_op_desc->SetOutputOffset(output_offset);
    L2FusionInfoPtr origin_l2_info = nullptr;
    origin_l2_info = node->GetOpDesc()->TryGetExtAttr(ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, origin_l2_info);
    if (origin_l2_info != nullptr) {
      (void)new_op_desc->SetExtAttr(ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, origin_l2_info);
    } else {
      FE_LOGD("node %s does not have l2_info.", node->GetName().c_str());
    }
    new_fus_nodelist.push_back(new_node);
  }
  (void)fusion_op_desc->SetExtAttr(kScopeNodeSet, new_fus_nodelist);

  return SUCCESS;
}

Status GraphComm::CopyOutDataEdges(const ge::NodePtr &src_node,
                                   const ge::NodePtr &node,
                                   const ge::ComputeGraphPtr &fusion_graph) {
  for (auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (auto &peer_in_anchor : out_anchor->GetPeerAnchors()) {
      if (peer_in_anchor) {
        ge::NodePtr dst_node = fusion_graph->FindNode(peer_in_anchor->GetOwnerNode()->GetName());
        // if not find, dst_node is other node in origraph,
        // src_node is output_edge_node, break.
        if (dst_node == nullptr) {
          FE_LOGD("dstNode not found in fusion_graph. name: %s, type: %s, peer_name: %s, peer_type: %s.",
              node->GetName().c_str(), node->GetType().c_str(), peer_in_anchor->GetOwnerNode()->GetName().c_str(),
              peer_in_anchor->GetOwnerNode()->GetType().c_str());
          break;
        }

        Status ret =
            graph_comm_impl_ptr_->AddEdge(node->GetOutDataAnchor(out_anchor->GetIdx()), dst_node, peer_in_anchor);
        if (ret != SUCCESS) {
          REPORT_FE_ERROR("[SubGraphOpt][Merge][CpOutDataEg] Add Data Edge failed. name:%s, type:%s, idx:%d",
                          node->GetName().c_str(), node->GetType().c_str(), out_anchor->GetIdx());
          return FAILED;
        }

        FE_LOGD("Add Data Edge Success. src_name:%s, dst_name:%s, idx:%d.", src_node->GetName().c_str(),
                dst_node->GetName().c_str(), out_anchor->GetIdx());
      }
    }
  }
  return SUCCESS;
}

Status GraphComm::CopyControlEdges(const ge::NodePtr &src_node,
                                   const ge::NodePtr &node,
                                   const ge::ComputeGraphPtr &fusion_graph) {
  auto out_ctrl_anchor = src_node->GetOutControlAnchor();
  FE_CHECK_NOTNULL(out_ctrl_anchor);
  for (auto &peer_in_anchor : out_ctrl_anchor->GetPeerAnchors()) {
    if (peer_in_anchor) {
      ge::NodePtr dst_node = fusion_graph->FindNode(peer_in_anchor->GetOwnerNode()->GetName());
      if (dst_node == nullptr) {
        FE_LOGI("control: dst_node not found in fusion_graph. name:%s, type:%s, peer_name:%s, peer_type:%s.",
            node->GetName().c_str(), node->GetType().c_str(), peer_in_anchor->GetOwnerNode()->GetName().c_str(),
            peer_in_anchor->GetOwnerNode()->GetType().c_str());
        break;
      }

      Status ret = graph_comm_impl_ptr_->AddEdge(node->GetOutControlAnchor(), dst_node, peer_in_anchor);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][Merge][CpCtrlEg] Failed to add control edge. Name: %s, Type: %s",
                        node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
      FE_LOGD("Add Control Edge Success. src_name:%s, dst_name:%s.", node->GetName().c_str(),
              dst_node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status GraphComm::CopyFusionOpEdges(const ge::ComputeGraph &orig_graph, const ge::ComputeGraphPtr &fusion_graph) {
  for (ge::NodePtr &node : fusion_graph->GetDirectNode()) {
    ge::NodePtr src_node = orig_graph.FindNode(node->GetName());
    FE_CHECK(src_node == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][CpFusOpEg] name:%s, type:%s not found",
             node->GetName().c_str(), node->GetType().c_str()), return FAILED);

    Status ret = CopyOutDataEdges(src_node, node, fusion_graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][CpFusOpEg] Failed to copy out data edges for node [%s].",
                      node->GetName().c_str());
      return ret;
    }


    ret = CopyControlEdges(src_node, node, fusion_graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][CpFusOpEg] Failed to copy control edges out for node %s.",
                      node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status GraphComm::MergeFusionNodeEdgeList(const ge::NodePtr &fus_node, const std::vector<ge::NodePtr> &fus_nodelist,
                                          const std::vector<FusionDataFlow> &fus_input_edge_list,
                                          const std::vector<FusionDataFlow> &fus_output_edge_list) {
  if (MergeFusionNodeInputEdgeList(fus_node, fus_nodelist, fus_input_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdEgList] Failed to MergeFusionNodeInputEdgeList.");
    return FAILED;
  }

  if (MergeFusionNodeOutputEdgeList(fus_node, fus_output_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdEgList] Failed to MergeFusionNodeOutputEdgeList.");
    return FAILED;
  }

  return SUCCESS;
}

Status GraphComm::MergeFunctionNodeEdgeList(const ge::NodePtr &fus_node, const std::vector<ge::NodePtr> &fus_nodelist,
                                            const std::vector<FusionDataFlow> &fus_input_edge_list,
                                            const std::vector<FusionDataFlow> &fus_output_edge_list) {
  if (MergeFunctionNodeInputEdgeList(fus_node, fus_nodelist, fus_input_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdEgList] Failed to MergeFusionNodeInputEdgeList.");
    return FAILED;
  }

  if (MergeFusionNodeOutputEdgeList(fus_node, fus_output_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdEgList] Failed to MergeFusionNodeOutputEdgeList.");
    return FAILED;
  }

  return SUCCESS;
}

Status GraphComm::MergeFusionNodeCtrlEdgeList(const ge::NodePtr &fus_node,
                                              const std::vector<ge::NodePtr> &fus_nodelist,
                                              const std::vector<FusionDataFlow> &fus_input_ctrl_edge_list,
                                              const std::vector<FusionDataFlow> &fus_output_ctrl_edge_list) {
  if (fus_input_ctrl_edge_list.empty() && fus_output_ctrl_edge_list.empty()) {
    FE_LOGD("No Ctrl Edge In %s, list size %zu.", fus_node->GetName().c_str(),
            fus_nodelist.size());
    return SUCCESS;
  }

  if (graph_comm_impl_ptr_->MergeFusionNodeInputCtrlEdgeList(fus_node, fus_input_ctrl_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdCtrlEgList] Failed to merge input control edge.");
    return FAILED;
  }

  if (graph_comm_impl_ptr_->MergeFusionNodeOutputCtrlEdgeList(fus_node, fus_output_ctrl_edge_list) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdCtrlEgList] Failed to merge output control edge.");
    return FAILED;
  }

  return SUCCESS;
}

Status GraphComm::MergeFusionNodeInputEdgeList(const ge::NodePtr &fus_node, const vector<ge::NodePtr> &fus_nodelist,
                                               const vector<FusionDataFlow> &fus_input_edge_list) {
  (void) fus_nodelist;
  int32_t fusion_dst_index = -1;

  ClearFusionSrc();
  FE_CHECK(fus_node == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdInEg] fusNode is nullptr."), return FAILED);
  std::vector<uint32_t> fused_tensor_indexes;
  ffts::ThreadSliceMapPtr fused_slice_info = nullptr;
  fused_slice_info = fus_node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, fused_slice_info);
  for (const FusionDataFlow &dataflow : fus_input_edge_list) {
    auto inedge = dataflow.edge;
    if (inedge.first == nullptr) {
      ++fusion_dst_index;
      continue;
    }

    const std::pair<string, ge::AnchorPtr> &node_dstindex_pair = dataflow.node_dataindex_pair;
    auto src_node = inedge.first->GetOwnerNode();
    FE_CHECK_NOTNULL(src_node);
    int64_t op_id = src_node->GetOpDesc()->GetId();
    fusion_dst_index = fusion_dst_index + 1;
    AddFusionInputSrc(op_id, inedge.first, fusion_dst_index, node_dstindex_pair);
    auto src_out_data_anchor = std::static_pointer_cast<ge::OutDataAnchor>(inedge.first);
    auto old_dst_anchor = std::static_pointer_cast<ge::InDataAnchor>(inedge.second);
    if (src_out_data_anchor->ReplacePeer(old_dst_anchor,
                                         fus_node->GetInDataAnchor(fusion_dst_index)) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdInEg] Failed to add edge between %s's output %d and %s's input %d",
                      src_node->GetName().c_str(), src_out_data_anchor->GetIdx(),
                      fus_node->GetName().c_str(), fusion_dst_index);
      return FAILED;
    }
    if ((fused_slice_info == nullptr) || (fused_slice_info->thread_mode == 0)) {
      continue;
    }
    auto dst_node = inedge.second->GetOwnerNode();
    FE_CHECK_NOTNULL(dst_node);
    std::vector<uint32_t> input_tensor_indexes;
    (void)ge::AttrUtils::GetListInt(dst_node->GetOpDesc(), ge::kInputTensorIndexs, input_tensor_indexes);
    FE_LOGD("Merge input edge src[%s] dst[%s] with index size[%zu].", src_node->GetName().c_str(),
            dst_node->GetName().c_str(), input_tensor_indexes.size());
    int32_t ori_idx = inedge.second->GetIdx();
    FE_LOGD("Dst anchor ori index:%d.", ori_idx);
    if (std::find(input_tensor_indexes.begin(), input_tensor_indexes.end(), ori_idx) != input_tensor_indexes.end()) {
      FE_LOGD("Fused emplace index: %d.", fusion_dst_index);
      fused_tensor_indexes.emplace_back(fusion_dst_index);
    }
  }
  (void)ge::AttrUtils::SetListInt(fus_node->GetOpDesc(), ge::kInputTensorIndexs, fused_tensor_indexes);
  fus_node->SetFusionInputFlowList(fusion_input_dataflow_list_);

  return SUCCESS;
}

Status GraphComm::MergeFunctionNodeInputEdgeList(const ge::NodePtr &fus_node, const vector<ge::NodePtr> &fus_nodelist,
                                                 const vector<FusionDataFlow> &fus_input_edge_list) {
  (void) fus_nodelist;
  int32_t fusion_dst_index = -1;

  ClearFusionSrc();
  FE_CHECK(fus_node == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdInEg] fusNode is nullptr."), return FAILED);

  for (const FusionDataFlow &dataflow : fus_input_edge_list) {
    auto inedge = dataflow.edge;
    if (inedge.first == nullptr) {
      continue;
    }

    const std::pair<string, ge::AnchorPtr> &node_dstindex_pair = dataflow.node_dataindex_pair;
    auto src_node = inedge.first->GetOwnerNode();
    FE_CHECK_NOTNULL(src_node);
    int64_t op_id = src_node->GetOpDesc()->GetId();

    fusion_dst_index = fusion_dst_index + 1;
    AddFusionInputSrc(op_id, inedge.first, fusion_dst_index, node_dstindex_pair);
    auto src_out_data_anchor = std::static_pointer_cast<ge::OutDataAnchor>(inedge.first);
    auto old_dst_anchor = std::static_pointer_cast<ge::InDataAnchor>(inedge.second);
    if (src_out_data_anchor->ReplacePeer(old_dst_anchor,
                                         fus_node->GetInDataAnchor(fusion_dst_index)) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdInEg] Failed to add edge between %s's output %d and %s's input %d",
                      src_node->GetName().c_str(), src_out_data_anchor->GetIdx(),
                      fus_node->GetName().c_str(), fusion_dst_index);
      return FAILED;
    }
  }

  fus_node->SetFusionInputFlowList(fusion_input_dataflow_list_);

  return SUCCESS;
}

void GraphComm::UnlinkOldEdges(const vector<FusionDataFlow> &fus_output_edge_list) {
  ClearFusionSrc();
  ClearFusionDst();
  for (const FusionDataFlow &dataflow : fus_output_edge_list) {
    auto outedge = dataflow.edge;
    if (outedge.first == nullptr || outedge.second == nullptr) {
      continue;
    }
    if (outedge.first->Unlink(outedge.second) != ge::GRAPH_SUCCESS) {
      FE_LOGD("Unlink output edge unsuccessful.");
    }
  }
}

Status GraphComm::MergeFusionNodeOutputEdgeList(const ge::NodePtr &fus_node,
                                                const vector<FusionDataFlow> &fus_output_edge_list) {
  UnlinkOldEdges(fus_output_edge_list);
  int32_t fusion_src_index = 0;
  FE_CHECK(fus_node == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdOutEgList] fusNode is nullptr."), return FAILED);
  std::vector<uint32_t> fused_tensor_indexes;
  ffts::ThreadSliceMapPtr fused_slice_info = nullptr;
  fused_slice_info = fus_node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, fused_slice_info);
  for (const FusionDataFlow &dataflow : fus_output_edge_list) {
    auto outedge = dataflow.edge;
    const std::pair<string, ge::AnchorPtr> &node_srcindex_pair = dataflow.node_dataindex_pair;
    if (outedge.second == nullptr) {
      continue;
    }
    auto peer_in_node = outedge.second->GetOwnerNode();
    FE_CHECK(peer_in_node == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdOutEgList] Peer in node is nullptr."), return FAILED);

    int64_t dst_op_id = peer_in_node->GetOpDesc()->GetId();
    ge::DataAnchorPtr out_edge_dst_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.second);

    if (IsFusionDstExist(dst_op_id, outedge.second)) {
      FE_CHECK(out_edge_dst_data_anchor_ptr == nullptr,
               REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdOutEgList] outEdgeDstDataAnchorPtr is nullptr."),
               return FAILED);
      FE_LOGI("MergeFusionNodeOutputEdgeList DstId %u, DstIndex %u.", static_cast<uint32_t>(dst_op_id),
              static_cast<uint32_t>(out_edge_dst_data_anchor_ptr->GetIdx()));
      continue;
    }
    SaveFusionDst(dst_op_id, outedge.second);

    ge::DataAnchorPtr out_edge_src_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.first);
    FE_CHECK(out_edge_src_data_anchor_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdOutEgList] outEdgeSrcDataAnchorPtr is nullptr."),
             return FAILED);
    auto node = out_edge_src_data_anchor_ptr->GetOwnerNode();
    FE_CHECK(node == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][MrgFusNdOutEgList] node is nullptr."), return FAILED);

    auto dst_in_data_anchor = std::static_pointer_cast<ge::InDataAnchor>(outedge.second);
    FE_CHECK_NOTNULL(dst_in_data_anchor);
    auto dst_node = dst_in_data_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(dst_node);
    int32_t fusion_src_exist_index;
    int32_t fusion_dst_exist_index;

    auto src_node = outedge.first->GetOwnerNode();
    FE_CHECK_NOTNULL(src_node);
    int64_t src_op_id = src_node->GetOpDesc()->GetId();
    if (!GetFusionSrc(src_op_id, outedge.first, fusion_src_exist_index, fusion_dst_exist_index)) {
      AddFusionOutputSrc(src_op_id, outedge.first, fusion_src_index, node_srcindex_pair);
      if (ge::GraphUtils::AddEdge(fus_node->GetOutDataAnchor(fusion_src_index), dst_in_data_anchor) !=
          ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][Merge][LinkOutput] Failed to add edge between %s's output %d and %s's input %d",
                        fus_node->GetName().c_str(), fusion_src_index,
                        dst_node->GetName().c_str(), dst_in_data_anchor->GetIdx());
        return FAILED;
      }
      if ((fused_slice_info == nullptr) || (fused_slice_info->thread_mode == 0)) {
        fusion_src_index++;
        continue;
      }
      std::vector<uint32_t> output_tensor_indexes;
      (void)ge::AttrUtils::GetListInt(src_node->GetOpDesc(), ge::kOutputTensorIndexs, output_tensor_indexes);
      FE_LOGD("Merge output edge src[%s] to dst[%s], index size[%zu].", src_node->GetName().c_str(),
              dst_node->GetName().c_str(), output_tensor_indexes.size());
      int32_t ori_idx = outedge.first->GetIdx();
      FE_LOGD("Dst anchor ori index:%d.", ori_idx);
      if (std::find(output_tensor_indexes.begin(), output_tensor_indexes.end(), ori_idx) !=
          output_tensor_indexes.end()) {
        FE_LOGD("Output fused emplace index:%d.", fusion_src_index);
        fused_tensor_indexes.emplace_back(fusion_src_index);
      }
      fusion_src_index++;
    } else {
      if (ge::GraphUtils::AddEdge(fus_node->GetOutDataAnchor(fusion_src_exist_index), dst_in_data_anchor) !=
          ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][Merge][LinkOutput] Failed to add edge between %s's output %d and %s's input %d",
                        fus_node->GetName().c_str(), fusion_src_exist_index,
                        dst_node->GetName().c_str(), dst_in_data_anchor->GetIdx());
        return FAILED;
      }
    }
  }
  (void)ge::AttrUtils::SetListInt(fus_node->GetOpDesc(), ge::kOutputTensorIndexs, fused_tensor_indexes);

  fus_node->SetFusionOutputFlowList(fusion_output_dataflow_list_);

  return SUCCESS;
}

string GraphComm::GetEngineName() { return engine_name_; }

Status GraphComm::AddFunctionNodeInputDesc(const ge::OpDescPtr &fus_op,
                                           vector<fe::FusionDataFlow> &fus_input_edge_list) {
  int32_t fusion_dst_index = -1;
  ClearFusionSrc();
  vector<bool> fusion_is_input_const_vector;

  for (fe::FusionDataFlow &dataflow : fus_input_edge_list) {
    auto inedge = dataflow.edge;
    std::pair<string, ge::AnchorPtr> &node_dstindex_pair = dataflow.node_dataindex_pair;
    auto out_anchor = inedge.first;
    if (out_anchor == nullptr) {
      continue;
    }
    int64_t src_op_id = out_anchor->GetOwnerNode()->GetOpDesc()->GetId();

    // only support data edge
    fusion_dst_index = fusion_dst_index + 1;
    vector<int64_t> input_vector;
    input_vector = fus_op->GetInputOffset();
    if (static_cast<uint32_t>(fusion_dst_index) < input_vector.size()) {
      return FAILED;
    }

    ge::DataAnchorPtr in_edge_dst_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(inedge.second);
    FE_CHECK(in_edge_dst_data_anchor_ptr == nullptr, FE_LOGE("inEdgeDstDataAnchorPtr is nullptr."), return FAILED);
    ge::OpDescPtr in_edge_dst_op_desc_ptr = in_edge_dst_data_anchor_ptr->GetOwnerNode()->GetOpDesc();
    vector<bool> is_input_const_vector = in_edge_dst_op_desc_ptr->GetIsInputConst();
    uint32_t dst_anchor_index = static_cast<uint32_t>(in_edge_dst_data_anchor_ptr->GetIdx());
    if (dst_anchor_index < in_edge_dst_op_desc_ptr->GetAllInputsSize()) {
      uint32_t idx = static_cast<uint32_t>(in_edge_dst_data_anchor_ptr->GetIdx());
      if (fus_op->AddInputDesc(*(in_edge_dst_op_desc_ptr->GetInputDescPtr(idx))) != ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("Add input desc failed.");
        return FAILED;
      }

      if (dst_anchor_index < is_input_const_vector.size()) {
        fusion_is_input_const_vector.push_back(is_input_const_vector[in_edge_dst_data_anchor_ptr->GetIdx()]);
      } else {
        fusion_is_input_const_vector.push_back(false);
      }
    } else {
      int32_t input_desc_size = static_cast<int32_t>(in_edge_dst_op_desc_ptr->GetAllInputsSize());
      int32_t is_input_const_size = static_cast<int32_t>(is_input_const_vector.size());
      int32_t DstIndex = in_edge_dst_data_anchor_ptr->GetIdx();
      REPORT_FE_ERROR("AddFusionNodeInput: input_desc_size=%u, is_input_const_size=%u, input DstIndex=%u.",
                      static_cast<uint32_t>(input_desc_size), static_cast<uint32_t>(is_input_const_size),
                      static_cast<uint32_t>(DstIndex));
      return FAILED;
    }
    fus_op->SetIsInputConst(fusion_is_input_const_vector);
    AddFusionInputSrc(src_op_id, inedge.first, fusion_dst_index, node_dstindex_pair);
  }

  return SUCCESS;
}

Status GraphComm::AddFunctionNodeOutputDesc(const ge::OpDescPtr &fus_op,
                                            vector<fe::FusionDataFlow> &fus_output_edge_list) {
  ClearFusionSrc();
  ClearFusionDst();

  int32_t fusion_src_index = 0;
  for (fe::FusionDataFlow &dataflow : fus_output_edge_list) {
    auto outedge = dataflow.edge;
    std::pair<string, ge::AnchorPtr> &node_srcindex_pair = dataflow.node_dataindex_pair;
    if (outedge.second == nullptr) {
      continue;
    }
    int64_t dst_op_id = outedge.second->GetOwnerNode()->GetOpDesc()->GetId();
    ge::DataAnchorPtr out_edge_dst_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.second);
    FE_CHECK(out_edge_dst_data_anchor_ptr == nullptr, FE_LOGE("outEdgeDstDataAnchorPtr is nullptr."),
    return FAILED);
    if (IsFusionDstExist(dst_op_id, outedge.second)) {
      FE_LOGD("MergeFusionNodeOutputEdgeList: Dstid %u, DstIndex %u.", static_cast<uint32_t>(dst_op_id),
              static_cast<uint32_t>(out_edge_dst_data_anchor_ptr->GetIdx()));
      continue;
    }
    SaveFusionDst(dst_op_id, outedge.second);

    int32_t fusion_src_exist_index;
    int32_t fusion_dst_exist_index;

    ge::DataAnchorPtr out_edge_src_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.first);
    FE_CHECK(out_edge_src_data_anchor_ptr == nullptr, FE_LOGE("outEdgeSrcDataAnchorPtr is nullptr."),
    return FAILED);
    ge::OpDescPtr out_edge_src_op_desc_ptr = out_edge_src_data_anchor_ptr->GetOwnerNode()->GetOpDesc();

    auto src_op_id = outedge.first->GetOwnerNode()->GetOpDesc()->GetId();
    if (!GetFusionSrc(src_op_id, outedge.first, fusion_src_exist_index, fusion_dst_exist_index)) {
      FE_CHECK(out_edge_src_op_desc_ptr == nullptr, FE_LOGE("outEdgeSrcOpDescPtr is nullptr."), return FAILED);
      if (static_cast<uint32_t>(out_edge_src_data_anchor_ptr->GetIdx()) < out_edge_src_op_desc_ptr->GetOutputsSize()) {
        uint32_t idx = static_cast<uint32_t>(out_edge_src_data_anchor_ptr->GetIdx());
        if (fus_op->AddOutputDesc(*(out_edge_src_op_desc_ptr->GetOutputDescPtr(idx))) != ge::GRAPH_SUCCESS) {
          FE_LOGD("AddOutputDesc operation did not succeed");
        }
        AddFusionOutputSrc(src_op_id, outedge.first, fusion_src_index, node_srcindex_pair);
      } else {
        int32_t output_desc_size = static_cast<int32_t>(out_edge_src_op_desc_ptr->GetOutputsSize());
        int32_t SrcIndex = out_edge_src_data_anchor_ptr->GetIdx();
        REPORT_FE_ERROR("MergeFusionNodeOutputEdgeList: Output Desc Size %u, Src Index %u.",
                        static_cast<uint32_t>(output_desc_size), static_cast<uint32_t>(SrcIndex));
        return FAILED;
      }
      fusion_src_index++;
    }
  }

  return SUCCESS;
}

Status GraphComm::EstablishConnection(const ge::NodePtr &node, const std::unordered_set<std::string> &node_name_set,
                                      ge::CompleteGraphBuilder &builder) const {
  const auto &src_node_name = node->GetName();
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    FE_CHECK_NOTNULL(out_data_anchor);
    uint32_t src_idx = static_cast<uint32_t>(out_data_anchor->GetIdx());
    for (auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      FE_CHECK_NOTNULL(in_data_anchor);
      auto dst_node = in_data_anchor->GetOwnerNode();
      FE_CHECK_NOTNULL(dst_node);
      if (node_name_set.find(dst_node->GetName()) == node_name_set.end()) {
        FE_LOGD("Node[name:%s] is out of op, skipping.", dst_node->GetName().c_str());
        continue;
      }
      builder.AddDataLink(src_node_name, src_idx, dst_node->GetName(), static_cast<uint32_t>(in_data_anchor->GetIdx()));
    }
  }
  auto out_ctrl_anchor =  node->GetOutControlAnchor();
  FE_CHECK_NOTNULL(out_ctrl_anchor);
  for (auto &in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
    FE_CHECK_NOTNULL(in_ctrl_anchor);
    auto dst_node = in_ctrl_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(dst_node);
    if (node_name_set.find(dst_node->GetName()) == node_name_set.end()) {
      FE_LOGD("Dst Node[name:%s] linked by control edge is out op, skip.", dst_node->GetName().c_str());
      continue;
    }
    builder.AddControlLink(src_node_name, dst_node->GetName());
  }
  return SUCCESS;
}

ge::OpDescPtr GraphComm::CreateFunctionOp(vector<ge::NodePtr> &node_vec) const {
  if (node_vec.empty()) {
    return nullptr;
  }
  ge::NodePtr first_node = node_vec[0];
  FE_CHECK(first_node == nullptr, FE_LOGE("CreateFusionOp encountered nullptr pointer."), return nullptr);
  ge::OpDescPtr function_opdef = nullptr;
  FE_MAKE_SHARED(function_opdef = std::make_shared<ge::OpDesc>(ge::OpDesc()), return nullptr);
  std::string fusion_node_name;

  for (ge::NodePtr &node : node_vec) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[CMSubGraphOpt][CrtFuncOp] node is nullptr."), return nullptr);
    fusion_node_name += node->GetOpDesc()->GetName();
    if (fusion_node_name.size() > kMaxOpNmaLen) {
      fusion_node_name = first_node->GetOpDesc()->GetName() + "_function_op";
      break;
    }
  }
  function_opdef->SetName(fusion_node_name);
  ge::OpDescUtilsEx::SetType(function_opdef, "PartitionedCall");

  // copy session graph id
  string session_graph_id;
  if (ge::AttrUtils::GetStr(first_node->GetOpDesc(), ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    if (!ge::AttrUtils::SetStr(function_opdef, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      REPORT_FE_ERROR("Op[%s]: failed to set the attribute %s.", fusion_node_name.c_str(),
                      ge::ATTR_NAME_SESSION_GRAPH_ID.c_str());
      return nullptr;
    }
  }

  return function_opdef;
}

uint64_t GraphComm::GetAtomicId() const {
  static std::atomic<uint64_t> global_atomic_id(1);
  return global_atomic_id.fetch_add(1, std::memory_order_relaxed);
}

Status GraphComm::ConnectionSubGraphToRootGraph(const ge::ComputeGraphPtr &sub_graph,
                                                ge::ComputeGraphPtr &root_graph,
                                                std::shared_ptr<std::mutex> &graph_common_lock_ptr) {
  if (graph_common_lock_ptr != nullptr) {
    const std::lock_guard <std::mutex> lock_guard(*graph_common_lock_ptr);
    if (root_graph->AddSubgraph(sub_graph->GetName(), sub_graph) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("Failed to add sub graph. Root graph: %s.", root_graph->GetName().c_str());
      return FAILED;
    }
    FE_LOGD("Add sub graph success. root graph: %s, subgraph: %s.",
            root_graph->GetName().c_str(), sub_graph->GetName().c_str());
  }

  return SUCCESS;
}

Status GraphComm::AddOuterDataEdgeInputs(const vector<fe::FusionDataFlow> &input_edge_list,
                                         ge::CompleteGraphBuilder &builder) const {
  FE_LOGD("input_edge_list size is %zu", input_edge_list.size());
  std::map<uint32_t, uint32_t> input_mapping;
  uint32_t index = 0;
  for (auto &input_edge : input_edge_list) {
    if (input_edge.edge.first == nullptr) {
      FE_LOGD("input_edge.first is null, which indicates a null optional input.");
      continue;
    }
    FE_CHECK(input_edge.edge.second == nullptr, FE_LOGE("input_data_edge.second is null."), return FAILED);
    std::int32_t src_idx = input_edge.edge.first->GetIdx();
    std::int32_t fusion_idx = input_edge.edge.second->GetIdx();
    ge::NodePtr src_node = input_edge.edge.first->GetOwnerNode();
    ge::NodePtr fusion_node = input_edge.edge.second->GetOwnerNode();
    FE_LOGD("srcIdx is %d, fusion_idx is %d. src_node is %s, fusion_node is %s.", src_idx, fusion_idx,
            src_node->GetName().c_str(), fusion_node->GetName().c_str());
    // add relation: input index of new graph to input index of original graph.
    std::vector<std::string> ffts_node_names{fusion_node->GetName()};
    std::vector<uint32_t> dst_indexes{static_cast<uint32_t>(fusion_idx)};
    builder.SetInput(index, ffts_node_names, dst_indexes);
    input_mapping.emplace(index, index);
    index++;
  }
  builder.SetInputMapping(input_mapping);
  return SUCCESS;
}

Status GraphComm::AddOuterCtrlEdgeOutputs(const vector<fe::FusionDataFlow> &output_ctrl_edge_list,
                                          ge::CompleteGraphBuilder &builder) const {
  FE_LOGD("ctrl edge size is %zu", output_ctrl_edge_list.size());
  std::unordered_set<ge::AnchorPtr> sub_output;
  for (auto &output_edge : output_ctrl_edge_list) {
    if (sub_output.count(output_edge.edge.first) == 0) {
      sub_output.emplace(output_edge.edge.first);
      FE_CHECK(output_edge.edge.first == nullptr, REPORT_FE_ERROR("output_ctrl_edge.first is null"), return FAILED);
      ge::NodePtr src_node = output_edge.edge.first->GetOwnerNode();
      builder.AddTarget(src_node->GetName());
    }
  }
  return SUCCESS;
}

Status GraphComm::AddOuterDataEdgeOutputs(const vector<fe::FusionDataFlow> &output_edge_list,
                                          ge::CompleteGraphBuilder &builder) const {
  FE_LOGD("Data edge size is %zu bytes.", output_edge_list.size());
  std::map<uint32_t, uint32_t> output_mapping;
  std::unordered_set<ge::AnchorPtr> sub_output;
  uint32_t output_index = 0;
  for (auto &output_edge : output_edge_list) {
    if (sub_output.count(output_edge.edge.first) == 0) {
      sub_output.emplace(output_edge.edge.first);
      FE_CHECK(output_edge.edge.first == nullptr, REPORT_FE_ERROR("output_data_edge.first is null"), return FAILED);
      if (output_edge.edge.second == nullptr) {
        continue;
      }
      std::int32_t src_idx = output_edge.edge.first->GetIdx();
      std::int32_t fusion_idx = output_edge.edge.second->GetIdx();
      ge::NodePtr src_node = output_edge.edge.first->GetOwnerNode();
      ge::NodePtr fusion_node = output_edge.edge.second->GetOwnerNode();
      FE_LOGD("out src_idx is %d, out fusion_idx is %d. src_node is %s, fusion_node is %s.",
              src_idx, fusion_idx, src_node->GetName().c_str(), fusion_node->GetName().c_str());
      // add relation: output index of new graph to output index of original graph.
      builder.AddOutput(src_node->GetName(), static_cast<uint32_t>(src_idx));
      output_mapping.emplace(output_index, output_index);
      output_index++;
    }
  }

  builder.SetOutputMapping(output_mapping);
  return SUCCESS;
}

Status GraphComm::CreateFunctionOpSubGraph(const ge::NodePtr &function_node,
                                           std::vector<ge::NodePtr> &node_vec,
                                           vector<fe::FusionDataFlow> &input_edge_list,
                                           vector<fe::FusionDataFlow> &output_edge_list,
                                           vector<fe::FusionDataFlow> &output_ctrl_edge_list) {
  if (node_vec.empty()) {
    return FAILED;
  }
  auto graph = node_vec[0]->GetOwnerComputeGraph();
  auto func_graph = function_node->GetOwnerComputeGraph();
  auto root_graph = ge::GraphUtils::FindRootGraph(graph);
  FE_CHECK_NOTNULL(root_graph);
  std::unordered_set<std::string> node_name_set;
  for (auto &node : node_vec) {
    node_name_set.emplace(node->GetName());
  }
  string sub_graph_name = node_vec[0]->GetName();
  string function_graph_suffix = "function_graph_" + std::to_string(function_graph_id_++);
  if (sub_graph_name.size() + function_graph_suffix.size() <= kMaxOpNmaLen) {
    sub_graph_name += function_graph_suffix;
  }

  FE_LOGD("Function_node: %s, subgraph name is %s.", function_node->GetName().c_str(), sub_graph_name.c_str());
  ge::CompleteGraphBuilder builder(sub_graph_name, false);

  for (auto &node : node_vec) {
    /* Adds a node to the builder. Currently, only the input parameter op_desc
       is supported, and the connection information will be lost. Therefore,
       the AddDataLink interface needs to be invoked to re-establish the
       connection. */
    builder.AddNode(node->GetOpDesc());

    FE_LOGD("Starting re-establishment of connection to node [name: %s]. Out data size is %u.",
            node->GetName().c_str(), node->GetAllOutDataAnchorsSize());
    (void)EstablishConnection(node, node_name_set, builder);
  }

  if (AddOuterDataEdgeInputs(input_edge_list, builder) != SUCCESS) {
    return FAILED;
  }

  if (AddOuterCtrlEdgeOutputs(output_ctrl_edge_list, builder) != SUCCESS ||
      AddOuterDataEdgeOutputs(output_edge_list, builder) != SUCCESS) {
    return FAILED;
  }

  builder.SetParentNode(function_node);
  ge::graphStatus status = ge::GRAPH_SUCCESS;
  string err_msg;
  auto sub_graph = builder.Build(status, err_msg);
  if (status != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("Build graph failed. Status is %u, error message is: %s", status, err_msg.c_str());
    return FAILED;
  }

  sub_graph->SetParentGraph(graph);
  sub_graph->SetParentNode(function_node);
  ge::OpDescPtr function_op_desc = function_node->GetOpDesc();
  (void)function_op_desc->AddSubgraphName(sub_graph->GetName());
  (void)function_op_desc->SetSubgraphInstanceName(0, sub_graph->GetName());

  FE_CHECK(ConnectionSubGraphToRootGraph(sub_graph, root_graph, graph_common_lock_ptr_) != SUCCESS,
           FE_LOGW("ConnectionSubGraphToRootGraph failed"), return FAILED);
  for (auto &node : node_vec) {
    std::vector<int> io_map = {0};
    if (node->GetAllOutDataAnchors().size() == 0) {
      io_map = {};
    } else if (node->GetAllInDataAnchors().size() == 0) {
      io_map = {-1};
    }
    if (ge::GraphUtils::IsolateNode(node, io_map) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("Isolate Node %s not successfully.", node->GetName().c_str());
      return FAILED;
    }
    if (ge::GraphUtils::RemoveNodeWithoutRelink(graph, node) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("Remove node: %s without relinking failed", node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

ge::NodePtr GraphComm::TransSingleSubGraph(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &node_vec) {
  vector<fe::FusionDataFlow> input_edge_list;
  vector<fe::FusionDataFlow> output_edge_list;
  vector<fe::FusionDataFlow> input_ctrl_edge_list;
  vector<fe::FusionDataFlow> output_ctrl_edge_list;
  std::unordered_set<ge::NodePtr> node_set;
  for (auto &node : node_vec) {
    node_set.emplace(node);
  }

  FE_LOGD("TransSingleSubGraph enter.");

  if (GetFusionNodeEdgeList(node_vec, node_set, input_edge_list, output_edge_list) != SUCCESS) {
    FE_LOGE("GetFusionNodeEdgeList failed.");
    return nullptr;
  }

  if (GetFusionNodeCtrlEdgeList(node_vec, node_set, input_ctrl_edge_list, output_ctrl_edge_list) != SUCCESS) {
    FE_LOGE("GetFusionNodeCtrlEdgeList failed.");
    return nullptr;
  }

  ge::OpDescPtr function_op = CreateFunctionOp(node_vec);
  if (function_op == nullptr) {
    FE_LOGE("CreateFunctionOp failed.");
    return nullptr;
  }

  if (AddFunctionNodeInputDesc(function_op, input_edge_list) != SUCCESS) {
    FE_LOGE("Failed to AddFusionNodeInputDesc.");
    return nullptr;
  }

  if (AddFunctionNodeOutputDesc(function_op, output_edge_list) != SUCCESS) {
    FE_LOGE("Failed to AddFusionNodeOutputDesc.");
    return nullptr;
  }

  std::vector<ge::NodePtr> function_node_vec = graph.FuseNodeKeepTopo(node_vec, {function_op});
  if (function_node_vec.empty()) {
    return nullptr;
  }
  ge::NodePtr function_node = function_node_vec.at(0);
  if (function_node == nullptr) {
    return nullptr;
  }

  // Merge Same scope node
  if (MergeFunctionNodeEdgeList(function_node, node_vec, input_edge_list, output_edge_list) !=
      SUCCESS) {
    FE_LOGE("MergeFusionNodeEdgeList failed!");
    return nullptr;
  }

  if (MergeFusionNodeCtrlEdgeList(function_node, node_vec, input_ctrl_edge_list, output_ctrl_edge_list) != SUCCESS) {
    FE_LOGE("MergeFusionNodeCtrlEdgeList failed!");
    return nullptr;
  }

  if (CreateFunctionOpSubGraph(function_node, node_vec, input_edge_list, output_edge_list,
                               output_ctrl_edge_list) != SUCCESS) {
    FE_LOGE("CreateFunctionOpSubGraph for function node %s failed!", function_node->GetName().c_str());
    return nullptr;
  }
  return function_node;
}

Status GraphComm::UnfoldFuncOp(ge::ComputeGraph& graph) {
  // UnfoldSubGraph
  auto nodes = graph.GetDirectNode();
  std::vector<ge::ComputeGraphPtr> sub_graphs;
  ge::ComputeGraphPtr owner_graph = nullptr;
  if (!nodes.empty()) {
    owner_graph = nodes.at(0)->GetOwnerComputeGraph();
    FE_CHECK_NOTNULL(owner_graph);
    ge::ComputeGraphPtr src_graph = owner_graph->TryGetExtAttr(kPartSrcGraph, ge::ComputeGraphPtr());
    if (src_graph == nullptr) {
      src_graph = owner_graph;
    }
    auto root_graph = ge::GraphUtils::FindRootGraph(src_graph);
    FE_CHECK_NOTNULL(root_graph);

    if (graph_common_lock_ptr_ != nullptr) {
      const std::lock_guard<std::mutex> lock_guard(*graph_common_lock_ptr_);
      sub_graphs = root_graph->GetAllSubgraphs();
    }
  }

  for (auto &node : nodes) {
    if (node->GetType() != "PartitionedCall") {
      continue;
    }
    bool in_fixpipe_sub_graph = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kInFixpipeSubGraph, in_fixpipe_sub_graph);
    if (!in_fixpipe_sub_graph) {
      continue;
    }
    const auto &filter = [](const ge::ComputeGraphPtr &graph_ptr) {
      const auto &parent = graph_ptr->GetParentNode();
      if ((parent == nullptr) || (parent->GetOpDesc() == nullptr)) {
        return false;
      }
      bool no_need_partition = false;
      ge::AttrUtils::GetBool(graph_ptr, ge::ATTR_NAME_NO_NEED_PARTITION, no_need_partition);
      if ((parent->GetType() == "PartitionedCall") &&
          (parent->GetOpDesc()->GetOpEngineName() == AI_CORE_NAME) &&
          no_need_partition) {
        return true;
      }
      return false;
    };
    for (const auto &sub_graph : sub_graphs) {
      if (sub_graph == nullptr || sub_graph->GetParentNode() == nullptr) {
        FE_LOGD("[GraphOpt][UnfoldFuncOp] Sub-graph is null or sub-graphs parent node is null.");
        continue;
      }
      if (sub_graph->GetParentNode()->GetName() != node->GetName()) {
        continue;
      }

      if (ge::GraphUtils::UnfoldGraph(sub_graph, owner_graph, node, filter) != ge::GRAPH_SUCCESS) {
        return FAILED;
      }
      ge::AttrUtils::SetBool(sub_graph, ge::ATTR_NAME_NO_NEED_MERGE, true);

      for (auto &func_node : owner_graph->GetDirectNode()) {
        if (func_node->GetOpDesc()->HasAttr(kInFixpipeSubGraph) && func_node->GetType() != "PartitionedCall") {
          (void)ge::AttrUtils::SetBool(func_node->GetOpDesc(), kInFixpipeSubGraph, false);
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphComm::ConvertFixpipePartitionCalledOp(ge::ComputeGraph& graph, const AICoreMode &ai_core_mode) {
  if (SetFixpipeToFuncOp(graph, ai_core_mode) != SUCCESS) {
    FE_LOGE("trans fixpipe to func op failed, graph: [%s]", graph.GetName().c_str());
    return FAILED;
  }
  for (auto &sub_graph : graph.GetAllSubgraphs()) {
    if (SetFixpipeToFuncOp(*sub_graph, ai_core_mode) != SUCCESS) {
      FE_LOGE("trans fixpipe to func op failed, graph: [%s]", sub_graph->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphComm::SetFixpipeToFuncOp(ge::ComputeGraph& graph, const AICoreMode &ai_core_mode) {
  // create fixpipe function op
  for (const auto &node : graph.GetDirectNode()) {
    bool is_valid_fixpipe = (node->GetType() == kOpFixpipe);
    uint32_t thread_scope_id = 0;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kThreadScopeId, thread_scope_id);
    if (ai_core_mode == FFTS_MODE_FFTS_PLUS) {
      is_valid_fixpipe = is_valid_fixpipe && (thread_scope_id == 0);
    }
    if (is_valid_fixpipe) {
      bool in_fixpipe_sub_graph = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kInFixpipeSubGraph, in_fixpipe_sub_graph);
      if (in_fixpipe_sub_graph) {
        continue;
      }
      std::vector<ge::NodePtr> fixpipe_nodes;
      CommonCollectFixpipeRelativeNodes(node, fixpipe_nodes);
      for (auto setnode : fixpipe_nodes) {
        if (setnode == nullptr || setnode->GetOpDesc() == nullptr) {
          continue;
        }
        (void)ge::AttrUtils::SetBool(setnode->GetOpDesc(), kInFixpipeSubGraph, true);
      }
      ge::NodePtr func_node = nullptr;
      func_node = TransSingleSubGraph(graph, fixpipe_nodes);
      if (func_node == nullptr) {
        FE_LOGE("TransSingleSubGraph of node[%s] failed.", node->GetName().c_str());
        return FAILED;
      }
      func_node->GetOpDesc()->SetOpEngineName(AI_CORE_NAME);
      func_node->GetOpDesc()->SetOpKernelLibName(AI_CORE_NAME);
      (void)ge::AttrUtils::SetStr(func_node->GetOpDesc(), ge::ATTR_NAME_ENGINE_NAME_FOR_LX, AI_CORE_NAME);
      (void)ge::AttrUtils::SetStr(func_node->GetOpDesc(), ge::ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, AI_CORE_NAME);
      (void)ge::AttrUtils::SetBool(func_node->GetOpDesc(), kInFixpipeSubGraph, true);
      std::vector<ge::ComputeGraphPtr> sub_graphs;
      (void)ge::NodeUtils::GetDirectSubgraphs(func_node, sub_graphs);
      for (auto &sub_graph : sub_graphs) {
        ge::AttrUtils::SetBool(sub_graph, ge::ATTR_NAME_NO_NEED_PARTITION, true);
      }
    }
  }
  return SUCCESS;
}

void GraphComm::CommonCollectFixpipeRelativeNodes(const ge::NodePtr &node,
                                                  std::vector<ge::NodePtr> &fixpipe_nodes) const {
  if (node == nullptr ||node->GetInDataNodes().size() == 0) {
    return;
  }
  ge::NodePtr cube_node = node->GetInDataNodes().at(0);
  if (cube_node == nullptr) {
    return;
  }
  CollectSwitchMergeNodes(cube_node, fixpipe_nodes);
  CollectFixpipe(cube_node, fixpipe_nodes);
}

//  AscendQuant  switch QuantBiasOptimization
//    |       \  /   |  /|
//    |        \/    | / |
//    |       / \    |/  |
//    |      /   \  /|   |
//    |     /     \/ |   |
//    |    /      /\ |   |
//  Conv2DCompress  Conv2d
//         \      /
//          Merge     A
//           \       /
//            Fixpipe
void GraphComm::CollectSwitchInMaxDeepth(const ge::NodePtr &cur_node, std::vector<ge::NodePtr> &to_fold_nodes,
                                         uint8_t cur_deepth, uint8_t &switch_deepth, bool &exist_switch) const {
  if (cur_deepth >= kMaxDeepth) {
    FE_LOGD("Current depth %u exceeds maximum depth.", cur_deepth);
    return;
  }
  if (cur_node->GetType() == kOpSwitch) {
    switch_deepth = cur_deepth;
    exist_switch = true;
    FE_LOGD("Has found switch node [%s], cur deepth is %u.", cur_node->GetName().c_str(), cur_deepth);
    return;
  }
  for (const auto &in_anchor : cur_node->GetAllInDataAnchors()) {
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr || peer_out_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    const auto peer_node = peer_out_anchor->GetOwnerNode();
    to_fold_nodes.emplace_back(peer_node);
    CollectSwitchInMaxDeepth(peer_node, to_fold_nodes, ++cur_deepth, switch_deepth, exist_switch);
    if (switch_deepth != cur_deepth) {
      to_fold_nodes.pop_back();
      cur_deepth--;
      continue;
    }
    switch_deepth--;
    cur_deepth--;
  }
}

void GraphComm::CollectSwitchMergeNodes(const ge::NodePtr &cube_node,
                                        std::vector<ge::NodePtr> &fixpipe_nodes) const {
  FE_LOGI("Start to collect switch and merge nodes.");
  std::unordered_set<ge::NodePtr> fixpipe_nodes_set;
  fixpipe_nodes.emplace_back(cube_node);
  if (cube_node->GetType() != kOpMerge) {
    return;
  }
  const auto input_nodes = cube_node->GetInDataNodes();
  if (input_nodes.empty()) {
    return;
  }
  std::vector<ge::NodePtr> to_fold_nodes;
  for (const auto &in_anchor : cube_node->GetAllInDataAnchors()) {
    uint8_t deepth = 0;
    uint8_t switch_deepth = 0;
    bool exist_switch = false;
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr || peer_out_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    const auto peer_node = peer_out_anchor->GetOwnerNode();

    to_fold_nodes.emplace_back(peer_node);
    CollectSwitchInMaxDeepth(peer_node, to_fold_nodes, deepth, switch_deepth, exist_switch);
    if (!exist_switch) {
      to_fold_nodes.pop_back();
    }
  }
  if (!to_fold_nodes.empty()) {
    for (const auto &node : to_fold_nodes) {
      fixpipe_nodes_set.emplace(node);
    }
    for (const auto &node : fixpipe_nodes_set) {
      fixpipe_nodes.emplace_back(node);
      FE_LOGD("fixpipe_nodes insert node [%s].", node->GetName().c_str());
    }
  }
  return;
}

void GraphComm::CollectFixpipe(const ge::NodePtr &cube_node, std::vector<ge::NodePtr> &fixpipe_nodes) const {
  if (cube_node == nullptr) {
    return;
  }
  for (const auto &out_anchor : cube_node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
    for (const auto &peer_in_anchor : peer_in_anchors) {
      if (peer_in_anchor == nullptr) {
        continue;
      }
      auto peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      if (peer_node->GetType() == kOpFixpipe) {
        if (peer_in_anchor->GetIdx() == 0) {
          fixpipe_nodes.emplace_back(peer_node);
        }
      } else {
        continue;
      }
    }
  }
}
}  // namespace fe
