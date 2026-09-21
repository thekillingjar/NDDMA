/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/graph_comm_impl.h"
#include "common/fe_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"

namespace fe {
GraphCommImpl::GraphCommImpl() {}
GraphCommImpl::~GraphCommImpl() {}

Status GraphCommImpl::GetAllInEdgeList(const ge::NodePtr &node,
                                       std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> &in_edge_pair,
                                       const int32_t &edge_type,
                                       const std::unordered_set<ge::NodePtr> &fus_node_set,
                                       bool is_tuning_mode) const {
  // data anchor
  if (edge_type == 0) {
    for (size_t i = 0; i < node->GetAllInDataAnchors().size(); i++) {
      auto in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
      FE_CHECK(in_data_anchor == nullptr,
               REPORT_FE_ERROR("[SubGraphOpt][Merge][GetAllInEgList] inDataAnchor is nullptr."), return FAILED);
      // In tuning mode, the fusion_op removes unconnected anchors because the aicore error problem has not been solved
      if (is_tuning_mode && in_data_anchor->GetPeerOutAnchor() == nullptr) {
        continue;
      }
      auto pre_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      if (pre_out_data_anchor != nullptr && IsInfusNodeList(pre_out_data_anchor->GetOwnerNode(), fus_node_set)) {
        continue;
      }
      in_edge_pair.push_back(make_pair(pre_out_data_anchor, in_data_anchor));
    }
    FE_LOGD("Get data anchors, size: %zu.", in_edge_pair.size());
  } else {
    ge::InControlAnchorPtr in_ctrl_anchor = node->GetInControlAnchor();
    FE_CHECK(in_ctrl_anchor == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][GetAllInEgList] inCtrlAnchor is nullptr."), return FAILED);
    for (auto &pre_out_data_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
      if (pre_out_data_anchor != nullptr && IsInfusNodeList(pre_out_data_anchor->GetOwnerNode(), fus_node_set)) {
        continue;
      }
      in_edge_pair.push_back(make_pair(pre_out_data_anchor, in_ctrl_anchor));
    }
    FE_LOGD("Get ctrl anchors, size:%zu.", in_edge_pair.size());
  }

  return SUCCESS;
}

Status GraphCommImpl::GetAllOutEdgeList(const ge::NodePtr &node,
                                        std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> &out_edge_pair,
                                        const int32_t &edge_type,
                                        const std::unordered_set<ge::NodePtr> &fus_node_set) const {
  if (edge_type == 0) {
    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      FE_CHECK(out_data_anchor == nullptr,
               REPORT_FE_ERROR("[SubGraphOpt][Merge][GetAllOutEgList] outDataAnchor is nullptr."), return FAILED);
      for (auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        if (next_in_data_anchor != nullptr && IsInfusNodeList(next_in_data_anchor->GetOwnerNode(), fus_node_set)) {
          continue;
        }
        out_edge_pair.push_back(make_pair(out_data_anchor, next_in_data_anchor));
      }
    }
    FE_LOGD("Get data anchors, size: %zu.", out_edge_pair.size());
  } else {
    ge::OutControlAnchorPtr out_ctrl_anchor = node->GetOutControlAnchor();
    FE_CHECK(out_ctrl_anchor == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Merge][GetAllOutEgList] outCtrlAnchor is nullptr."), return FAILED);
    for (auto &next_in_data_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
      if (next_in_data_anchor != nullptr && IsInfusNodeList(next_in_data_anchor->GetOwnerNode(), fus_node_set)) {
        continue;
      }
      out_edge_pair.push_back(make_pair(out_ctrl_anchor, next_in_data_anchor));
    }
    FE_LOGD("Get ctrl anchors, size:%zu.", out_edge_pair.size());
  }

  return SUCCESS;
}

void GraphCommImpl::PutEdgeToFusionDataFlowVec(const std::pair<ge::AnchorPtr, ge::AnchorPtr> &edge,
                                               const ge::AnchorPtr &fus_node_anchor,
                                               const ge::AnchorPtr &edge_node_anchor,
                                               std::vector<FusionDataFlow> &fus_edge_list) const {
  std::string dst_name = "VirtualNode";
  if (edge_node_anchor != nullptr) {
    dst_name = edge_node_anchor->GetOwnerNode()->GetName();
  }
  fus_edge_list.emplace_back(FusionDataFlow());
  // use uint32 data type so save index
  auto &flow = fus_edge_list.back();
  flow.node_dataindex_pair.first = fus_node_anchor->GetOwnerNode()->GetName();
  flow.node_dataindex_pair.second = fus_node_anchor;
  flow.edge = edge;
  FE_LOGD("Putting edge to flow vector, fus_node: %s, edge_node: %s.", flow.node_dataindex_pair.first.c_str(), dst_name.c_str());
}

void GraphCommImpl::SaveFusionNode(const uint32_t &scopeid, const ge::NodePtr &node, ScopeNodeMap &fus_node_map) const {
  ScopeNodeMap::iterator nodelist_it = fus_node_map.find(scopeid);
  if (nodelist_it == fus_node_map.end()) {
    std::vector<ge::NodePtr> node_list_new;
    node_list_new.clear();
    node_list_new.push_back(node);
    (void)fus_node_map.insert(std::pair<int64_t, std::vector<ge::NodePtr>>(static_cast<int64_t>(scopeid),
                                                                           node_list_new));
  } else {
    nodelist_it->second.push_back(node);
  }
}

void GraphCommImpl::AddFusionSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor,
                                 const int32_t &fusion_src_index, const int32_t &fusion_dst_index,
                                 std::vector<FusionOpSrc> &exist_fusion_src_list) const {
  FusionOpSrc value;
  value.src_op_id = src_op_id;
  value.src_anchor = src_anchor;
  value.fusion_src_index = fusion_src_index;
  value.fusion_dst_index = fusion_dst_index;
  exist_fusion_src_list.push_back(value);
}

Status GraphCommImpl::MergeFusionNodeInputCtrlEdgeList(const ge::NodePtr &fus_node,
                                                       const vector<FusionDataFlow> &fus_input_ctrl_edge_list) const {
  FE_CHECK(fus_node == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][Merge][LinkInCtrlEdge] fusNode is nullptr."), return FAILED);

  for (const FusionDataFlow &data_flow : fus_input_ctrl_edge_list) {
    auto in_edge = data_flow.edge;
    (void)in_edge.first->Unlink(in_edge.second);
    auto out_ctrl_anchor = std::static_pointer_cast<ge::OutControlAnchor>(in_edge.first);
    auto in_ctrl_anchor = std::static_pointer_cast<ge::InControlAnchor>(in_edge.second);
    FE_LOGD("Deleted control anchor from %s to %s.", out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
            in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
  }

  for (const FusionDataFlow &data_flow : fus_input_ctrl_edge_list) {
    auto in_edge = data_flow.edge;
    auto src_out_ctrl_anchor = std::static_pointer_cast<ge::OutControlAnchor>(in_edge.first);
    FE_CHECK_NOTNULL(src_out_ctrl_anchor);
    auto src_node = src_out_ctrl_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(src_node);
    if (ge::GraphUtils::AddEdge(src_out_ctrl_anchor, fus_node->GetInControlAnchor()) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][LinkInCtrlEdge] Failed to add edge between %s's output %d and %s's input %d",
                      src_node->GetName().c_str(), src_out_ctrl_anchor->GetIdx(),
                      fus_node->GetName().c_str(), fus_node->GetInControlAnchor()->GetIdx());
      return FAILED;
    }
    FE_LOGD("Added control anchor from %s to %s.", src_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
            fus_node->GetName().c_str());
  }

  return SUCCESS;
}

Status GraphCommImpl::MergeFusionNodeOutputCtrlEdgeList(const ge::NodePtr &fus_node,
                                                        const vector<FusionDataFlow> &fus_output_ctrl_edge_list) const
{
  FE_CHECK(fus_node == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Merge][GetNodes] fusNode is nullptr."), return FAILED);

  for (const FusionDataFlow &data_flow : fus_output_ctrl_edge_list) {
    auto out_edge = data_flow.edge;
    (void)out_edge.first->Unlink(out_edge.second);
    auto out_ctrl_anchor = std::static_pointer_cast<ge::OutControlAnchor>(out_edge.first);
    auto in_ctrl_anchor = std::static_pointer_cast<ge::InControlAnchor>(out_edge.second);
    FE_LOGD("Del out ctrl anchor from:%s to %s", out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
            in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
  }

  for (const FusionDataFlow &data_flow : fus_output_ctrl_edge_list) {
    auto out_edge = data_flow.edge;
    ge::InControlAnchorPtr in_edge_ctrl_anchor_ptr = std::static_pointer_cast<ge::InControlAnchor>(out_edge.second);
    FE_CHECK_NOTNULL(fus_node->GetOutControlAnchor());
    FE_CHECK_NOTNULL(in_edge_ctrl_anchor_ptr);
    auto dst_node = in_edge_ctrl_anchor_ptr->GetOwnerNode();
    FE_CHECK_NOTNULL(dst_node);

    if (ge::GraphUtils::AddEdge(fus_node->GetOutControlAnchor(), in_edge_ctrl_anchor_ptr) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][LinkOutCtrlEdge]Failed to add edge between %s's out %d and %s's in %d",
                      fus_node->GetName().c_str(), fus_node->GetOutControlAnchor()->GetIdx(),
                      dst_node->GetName().c_str(), in_edge_ctrl_anchor_ptr->GetIdx());
      return FAILED;
    }
    FE_LOGD("Added out control anchor from: %s to %s.", fus_node->GetName().c_str(),
            in_edge_ctrl_anchor_ptr->GetOwnerNode()->GetName().c_str());
  }

  return SUCCESS;
}

ge::NodePtr GraphCommImpl::FindNodeInFusNodeList(const string &node_name,
                                                 const vector<ge::NodePtr> &fus_nodelist) const {
  for (const ge::NodePtr &node : fus_nodelist) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Merge][FindFusNdListNd] node is null."), return nullptr);
    if (node->GetName() == node_name) {
      return node;
    }
  }
  return nullptr;
}

bool GraphCommImpl::IsInfusNodeList(const ge::NodePtr &node,
                                    const std::unordered_set<ge::NodePtr> &fus_nodelist) const {
  return (fus_nodelist.count(node) > 0);
}

Status GraphCommImpl::AddEdge(const ge::AnchorPtr &src_anchor, const ge::NodePtr &dst_node,
                              const ge::AnchorPtr &old_dst_anchor) const {
  if (src_anchor == nullptr || dst_node == nullptr || old_dst_anchor == nullptr) {
    return FAILED;
  }

  auto old_dst_data_anchor = ge::Anchor::DynamicAnchorCast<ge::InDataAnchor>(old_dst_anchor);
  auto old_dst_ctrl_anchor = ge::Anchor::DynamicAnchorCast<ge::InControlAnchor>(old_dst_anchor);
  auto src_data_anchor = ge::Anchor::DynamicAnchorCast<ge::OutDataAnchor>(src_anchor);

  if (src_data_anchor) {
    if (old_dst_data_anchor) {
      ge::graphStatus ret =
          ge::GraphUtils::AddEdge(src_data_anchor, dst_node->GetInDataAnchor(old_dst_data_anchor->GetIdx()));
      if (ret == ge::GRAPH_FAILED) {
        REPORT_FE_ERROR("[SubGraphOpt][Merge][AddEg] AddEdge failed.");
        return FAILED;
      }
    }
  }
  auto src_ctrl_anchor = ge::Anchor::DynamicAnchorCast<ge::OutControlAnchor>(src_anchor);
  if (src_ctrl_anchor && old_dst_ctrl_anchor) {
    ge::graphStatus ret = ge::GraphUtils::AddEdge(src_ctrl_anchor, dst_node->GetInControlAnchor());
    if (ret == ge::GRAPH_FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Merge][AddEg] AddEdge failed");
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace fe
