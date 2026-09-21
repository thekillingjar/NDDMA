/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/graph_replace.h"

#include <climits>
#include <sstream>
#include <stack>

#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

using std::map;
using std::stack;
using std::string;
using std::vector;

namespace fe {
namespace {
void UpdateInputDescForPeerNode(const ge::NodePtr &node, const map<ge::NodePtr, FusionRuleNodePtr> &fusion_nodes) {
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    for (const auto &peer_out_anchor : out_anchor->GetPeerInDataAnchors()) {
      auto peer_node = peer_out_anchor->GetOwnerNode();
      if (fusion_nodes.find(peer_node) != fusion_nodes.end()) {
        auto output_tensor = node->GetOpDesc()->MutableOutputDesc(out_anchor->GetIdx());
        if (output_tensor == nullptr) {
          continue;
        }
        FE_LOGD("the output desc of the node [%s]: format[%u], origin_format[%u], dtype[%u], shape[%s].",
                node->GetName().c_str(), output_tensor->GetFormat(), output_tensor->GetOriginFormat(),
                output_tensor->GetDataType(), GetShapeDims(output_tensor->MutableShape()).c_str());
        auto peer_opdesc = peer_node->GetOpDesc();
        peer_opdesc->UpdateInputDesc(peer_out_anchor->GetIdx(), *output_tensor);
      }
    }
  }
}

bool InferShape(const ge::NodePtr &node) {
  FE_LOGI("node %s: start to InferShapeAndType.", node->GetName().c_str());
  if (ge::NodeUtilsEx::InferShapeAndType(node) != ge::SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][InferShape] node[%s] InferShapeAndType failed", node->GetName().c_str());
    return false;
  }

  FE_LOGI("node %s: start to InferOriginFormat.", node->GetName().c_str());
  if (ge::NodeUtilsEx::InferOriginFormat(node) != ge::SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][InferShape] node[%s] failed to InferOriginFormat", node->GetName().c_str());
    return false;
  }
  return true;
}

bool UpdateSingleFusedNode(const ge::NodePtr &node, const map<ge::NodePtr, FusionRuleNodePtr> &fusion_nodes,
                           const map<FusionRuleAnchorPtr, set<ge::AnchorPtr>> &outer_outputs) {
  auto rule_node = fusion_nodes.at(node);
  const vector<FusionRuleAnchorPtr> &output_anchor = rule_node->GetOutputDataAnchors();
  for (const auto &out_anchor : output_anchor) {
    int output_index = out_anchor->GetAnchorIdx();
    for (const auto &peer_rule_input_anchor : out_anchor->GetPeerAnchors()) {
      if (outer_outputs.find(peer_rule_input_anchor) == outer_outputs.end()) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][InfShpDtpFmt] node[%s] does not have output anchor",
                        node->GetName().c_str());
        return false;
      }
      auto input_anchors = outer_outputs.at(peer_rule_input_anchor);
      for (auto const &input_anchor : input_anchors) {
        ge::InDataAnchorPtr dst_data_anchor = ge::Anchor::DynamicAnchorCast<ge::InDataAnchor>(input_anchor);
        if (dst_data_anchor == nullptr) {
          continue;
        }
        auto peer_node = dst_data_anchor->GetOwnerNode();
        auto input_tensor = peer_node->GetOpDesc()->GetInputDescPtr(dst_data_anchor->GetIdx());
        FE_CHECK(input_tensor == nullptr,
                 REPORT_FE_ERROR("[GraphOpt][RunFusionRule][InfShpDtpFmt] Input_tensor is null."), return false);
        if (node->GetOpDesc()->UpdateOutputDesc(output_index, *input_tensor) == ge::GRAPH_FAILED) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][InfShpDtpFmt] Failed to update output description for node [%s]",
                          node->GetName().c_str());
          return false;
        }
        break;
      }
    }
  }
  return true;
}
} // namespace

const string NEED_INFER = "isNeedInfer";
GraphReplace::GraphReplace(shared_ptr<ge::OpsKernelInfoStore> ops_kernel_info_store_ptr)
    : ops_kernel_info_store_ptr_(ops_kernel_info_store_ptr) {}
GraphReplace::~GraphReplace() {}

Status GraphReplace::ReplaceGraph(vector<GraphMatchResult> &match_results, const FusionRulePattern &fusion_rule_pattern,
                                  ge::ComputeGraph &graph) {
  size_t result_num = match_results.size();
  int32_t effect_times = 0;
  FusionInfo fusion_info(graph.GetSessionID(), to_string(graph.GetGraphID()), fusion_rule_pattern.GetRuleName(), 0, 0);
  for (size_t i = 0; i < result_num; ++i) {
    GraphMatchResult &match_result = match_results[i];
    if (!match_result.valid_flag) {
      continue;
    }

    string rule_name = fusion_rule_pattern.GetRuleName();
    UpdateMatchedOuterAnchor(match_result, rule_name);
    FusionRuleNodeMapping fusion_graph = {};
    Status ret = CreateFusionNodes(fusion_rule_pattern, match_result.origin_nodes, fusion_graph, graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][RunFusionRule][RplGph] Failed to create fusion sub_graph for fusion rule name: %s, at sub_graph number: %zu.",
          fusion_rule_pattern.GetRuleName().c_str(), (i + 1));
      return ret;
    }

    ret = UpdateAttr(match_result.origin_nodes, fusion_graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][RunFusionRule][RplGph] fusion rule name[%s] No.%zu sub_graph, update attribute value failed.",
          fusion_rule_pattern.GetRuleName().c_str(), (i + 1));
      return ret;
    }

    ret = UpdateSpecialAttr(match_result.origin_nodes, fusion_graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][RunFusionRule][RplGph] Failed to update the special attribute value for sub_graph No.%zu by rule [%s].",
          (i + 1), fusion_rule_pattern.GetRuleName().c_str());
      return ret;
    }

    if (CheckFusionNode(match_result, fusion_graph) == FAILED) {
      FE_LOGW("Fusion rule name [%s], failed to fuse for the %zu time, reason: fusion node not supported.",
              fusion_rule_pattern.GetRuleName().c_str(), (i + 1));
      if (DeleteNodes(fusion_graph, fusion_rule_pattern.GetFusionRuleNodes(), graph) == FAILED) {
        REPORT_FE_ERROR(
            "[GraphOpt][RunFusionRule][RplGph] fusion rule name[%s] No.[%zu] sub_graph, delete fusion node failed.",
            fusion_rule_pattern.GetRuleName().c_str(), (i + 1));
        return GRAPH_REPLACE_DELETE_NODE_FAILED;
      }
      return GRAPH_REPLACE_CHECKSUPPORTED_FAILED;
    }

    // Record fusion nodes
    RecordFusionNodes(fusion_graph, match_result);

    if (Replace(match_result, fusion_graph, fusion_rule_pattern, graph) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][RunFusionRule][RplGph] fusion rule name[%s] No.[%zu] sub_graph, edges or nodes replace failed.",
          fusion_rule_pattern.GetRuleName().c_str(), (i + 1));
      return FAILED;
    }

    // Post fusion process, eg. record original name, output anchor map
    PostFusion(match_result, fusion_rule_pattern.GetRuleName());

    FE_LOGD("Fusion rule name[%s] No.[%zu] time fusion success.", fusion_rule_pattern.GetRuleName().c_str(), (i + 1));
    effect_times++;
  }
  // get effect times
  fusion_info.SetEffectTimes(effect_times);
  FusionStatisticRecorder::Instance().UpdateGraphFusionEffectTimes(fusion_info);
  FE_LOGD("SessionId %lu GraphId %u fusion rule name:%s fusion Success, %d times take effect", graph.GetSessionID(),
          graph.GetGraphID(), fusion_rule_pattern.GetRuleName().c_str(), effect_times);
  return SUCCESS;
}

void GraphReplace::UpdateOuterInputs(const string &pattern_name, GraphMatchResult &match_result,
                                     std::map<FusionRuleAnchorPtr, ge::AnchorPtr> &outer_inputs) const {
  for (auto &origin_anchor_map_pair : match_result.origin_outer_inputs) {
    FusionRuleAnchorPtr rule_anchor_ptr = origin_anchor_map_pair.first;
    for (auto &peer_out_anchor : origin_anchor_map_pair.second->GetPeerAnchors()) {
      ge::NodePtr node = peer_out_anchor->GetOwnerNode();
      if (match_result.origin_nodes_set.find(node) == match_result.origin_nodes_set.end()) {
        outer_inputs.emplace(rule_anchor_ptr, peer_out_anchor);
        FE_LOGD("outerInputs rule_anchor:%s:%d, new_anchor:%s:%d, pattern_name:%s",
                rule_anchor_ptr->GetOwnerNode()->GetNodeName().c_str(), rule_anchor_ptr->GetAnchorIdx(),
                peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetIdx(), pattern_name.c_str());

        const auto iter = match_result.outer_inputs_set.find(rule_anchor_ptr);
        if (iter == match_result.outer_inputs_set.cend()) {
          std::set<ge::AnchorPtr> graph_achors = {peer_out_anchor};
          match_result.outer_inputs_set.emplace(rule_anchor_ptr, graph_achors);
          FE_LOGD("new rule_anchor:%s:%d, new_anchor:%s:%d, pattern_name:%s",
                  rule_anchor_ptr->GetOwnerNode()->GetNodeName().c_str(), rule_anchor_ptr->GetAnchorIdx(),
                  peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetIdx(), pattern_name.c_str());
        } else {
          iter->second.insert(peer_out_anchor);
          FE_LOGD("has rule_anchor:%s:%d, new_anchor:%s:%d, pattern_name:%s",
                  rule_anchor_ptr->GetOwnerNode()->GetNodeName().c_str(), rule_anchor_ptr->GetAnchorIdx(),
                  peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetIdx(), pattern_name.c_str());
        }
      }
    }
  }
}

Status GraphReplace::UpdateMatchedOuterAnchor(GraphMatchResult &match_result, string &pattern_name) const {
  if (match_result.origin_outer_inputs.empty() || match_result.origin_outer_outputs.empty()) {
    FE_LOGW("Not get origin outer input and output, pattern_name[%s]", pattern_name.c_str());
    return SUCCESS;
  }

  // update OuterInputs anchors
  std::map<FusionRuleAnchorPtr, ge::AnchorPtr> outer_inputs;
  UpdateOuterInputs(pattern_name, match_result, outer_inputs);

  // update outer_outputs anchors
  std::map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> outer_outputs;
  for (auto &origin_anchor_map : match_result.origin_outer_outputs) {
    std::set<ge::AnchorPtr> anchor_set;
    std::set<ge::AnchorPtr> origin_anchor_set = origin_anchor_map.second;
    for (auto &origin_anchor : origin_anchor_set) {
      for (auto &peer_in_anchor : origin_anchor->GetPeerAnchors()) {
        ge::NodePtr node = peer_in_anchor->GetOwnerNode();
        if (match_result.origin_nodes_set.find(node) == match_result.origin_nodes_set.end()) {
          anchor_set.insert(peer_in_anchor);
          FE_LOGD("output rule_anchor:%s:%d, new_anchor:%s:%d, pattern_name:%s",
                  origin_anchor_map.first->GetOwnerNode()->GetNodeName().c_str(),
                  origin_anchor_map.first->GetAnchorIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                  peer_in_anchor->GetIdx(), pattern_name.c_str());
        }
      }
    }
    outer_outputs.insert(make_pair(origin_anchor_map.first, anchor_set));
  }

  if (outer_outputs.size() != match_result.outer_outputs.size() ||
      outer_inputs.size() != match_result.outer_inputs.size()) {
    FE_LOGD("patternName:%s, two rules are continuous", pattern_name.c_str());
  }

  match_result.outer_inputs = outer_inputs;
  match_result.outer_outputs = outer_outputs;
  return SUCCESS;
}

Status GraphReplace::CreateFusionNodes(const FusionRulePattern &fusion_rule_pattern,
                                       const FusionRuleNodeMapping &origin_sub_graph,
                                       FusionRuleNodeMapping &fusion_graph,
                                       ge::ComputeGraph &graph) const {
  NodeMapInfoPtr node_map_info = nullptr;
  (void)GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph);
  const set<FusionRuleNodePtr> &fusion_rule_node_set = fusion_rule_pattern.GetFusionRuleNodes();
  for (const auto &fusion_rule_node : fusion_rule_node_set) {
    // The fusioned node type is the same as the pre-fusion node type;
    // create fusion node with pre-fusion node opdesc;
    // if pre-fusion node can't found, create new opdesc and create fusion node
    // with new opdesc
    ge::NodePtr node = nullptr;
    // using node name of fusion rule to find whether this fusion node is in
    // matched subgraph
    ge::NodePtr origin_node = FindSameNode(fusion_rule_node, origin_sub_graph);
    if (origin_node != nullptr) {
      ge::GeTensorDesc tensor_desc;
      ge::OpDescPtr op_desc = ge::AttrUtils::CopyOpDesc(origin_node->GetOpDesc());
      // add input_opdesc
      size_t input_opdesc_count = fusion_rule_node->GetInputDataAnchors().size();
      for (size_t i = op_desc->GetInputsSize(); i < input_opdesc_count; ++i) {
        op_desc->AddInputDesc(tensor_desc);
      }
      size_t output_opdesc_count = fusion_rule_node->GetOutputDataAnchors().size();
      for (size_t i = op_desc->GetOutputsSize(); i < output_opdesc_count; ++i) {
        op_desc->AddOutputDesc(tensor_desc);
      }
      node = graph.AddNode(op_desc);
      FE_CHECK(node == nullptr,
               REPORT_FE_ERROR("[GraphOpt][RunFusionRule][Replace] Failed to create fusion node."),
               return GRAPH_REPLACE_CREATE_FUSION_NODES_FAILED);

      fusion_graph[fusion_rule_node] = node;
      GraphPassUtil::AddNodeFromOpTypeMap(node_map_info, node);
      continue;
    }
    string node_name = CreateNodeName(origin_sub_graph, fusion_rule_pattern, fusion_rule_node->GetNodeType());
    node = CreateNode(fusion_rule_node, node_name, graph);
    FE_CHECK(node == nullptr,
             REPORT_FE_ERROR("[GraphOpt][RunFusionRule][Replace] Failed to create fusion node with new opdesc."),
             return GRAPH_REPLACE_CREATE_FUSION_NODES_FAILED);

    fusion_graph[fusion_rule_node] = node;
  }
  return SUCCESS;
}

Status GraphReplace::UpdateSpecialAttr(const FusionRuleNodeMapping &origin_sub_graph,
                                       const FusionRuleNodeMapping &fusion_sub_graph) const {
  vector<string> spec_attr = {ge::ATTR_NAME_STREAM_LABEL, ge::ATTR_NAME_OP_COMPILE_STRATEGY};
  for (auto &attr_name : spec_attr) {
    for (auto &ori_node : origin_sub_graph) {
      ge::OpDescPtr op_desc = ori_node.second->GetOpDesc();
      if (!ge::AttrUtils::HasAttr(op_desc, attr_name)) {
        FE_LOGD("node %s does not have attr %s", op_desc->GetName().c_str(), attr_name.c_str());
        continue;
      }

      ge::GeAttrValue attr_value;
      if (op_desc->GetAttr(attr_name, attr_value) == ge::GRAPH_FAILED) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][UpdSpclAttr] get attr %s from node %s error",
                        op_desc->GetName().c_str(), attr_name.c_str());
        return GRAPH_REPLACE_UPDATE_ATTR_FAILED;
      }

      for (auto &fusion_item : fusion_sub_graph) {
        ge::OpDescPtr fusion_op_desc = fusion_item.second->GetOpDesc();
        if (fusion_op_desc->SetAttr(attr_name, attr_value) == ge::GRAPH_FAILED) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][UpdSpclAttr] set attr %s to node %s error", attr_name.c_str(),
                          fusion_op_desc->GetName().c_str());
          return GRAPH_REPLACE_UPDATE_ATTR_FAILED;
        }
      }
      break;
    }
  }
  return SUCCESS;
}

Status GraphReplace::UpdateAttr(const FusionRuleNodeMapping &origin_sub_graph,
                                const FusionRuleNodeMapping &fusion_sub_graph) const {
  for (auto &fusion_item : fusion_sub_graph) {
    const map<string, FusionRuleAttrValuePtr> &attributes = fusion_item.first->GetAttributes();
    ge::OpDescPtr fusion_opdesc = fusion_item.second->GetOpDesc();

    for (auto &attribute : attributes) {
      string fusion_node_attr_name = attribute.first;
      FusionRuleAttrValuePtr attr_value_ptr = attribute.second;
      ge::GeAttrValue attr_value;
      if (attr_value_ptr->IsFusionRuleAttr()) {
        FusionRuleAttr fusion_rule_attr = attr_value_ptr->GetRuleNodeAttrValue();
        if (origin_sub_graph.find(attr_value_ptr->GetOwnerNode()) == origin_sub_graph.end()) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][UpdAttr] The node[%s] does not in origin SubGraph",
                          attr_value_ptr->GetOwnerNode()->GetNodeName().c_str());
          return GRAPH_REPLACE_UPDATE_ATTR_FAILED;
        }

        ge::NodePtr node = origin_sub_graph.at(attr_value_ptr->GetOwnerNode());
        string origin_node_attr_name = fusion_rule_attr.attr_name;
        ge::OpDescPtr op_desc = node->GetOpDesc();

        if (!ge::AttrUtils::HasAttr(op_desc, origin_node_attr_name)) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][UpdAttr] The node[%s] does not have attr[%s]",
                          node->GetName().c_str(), origin_node_attr_name.c_str());
          return GRAPH_REPLACE_UPDATE_ATTR_FAILED;
        }

        if (op_desc->GetAttr(origin_node_attr_name, attr_value) == ge::GRAPH_FAILED) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][UpdAttr] get attr[%s] from node[%s] error",
                          origin_node_attr_name.c_str(), node->GetName().c_str());
          return GRAPH_REPLACE_UPDATE_ATTR_FAILED;
        }
      } else {
        attr_value = attr_value_ptr->GetFixAttrValue();
      }

      if (fusion_opdesc->SetAttr(fusion_node_attr_name, attr_value) == ge::GRAPH_FAILED) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][UpdAttr] set attr[%s] to node[%s] error",
                        fusion_node_attr_name.c_str(), fusion_opdesc->GetName().c_str());
        return GRAPH_REPLACE_UPDATE_ATTR_FAILED;
      }

      // this attribute serves as a sign of infering at Graph Engine
      if (!ge::AttrUtils::SetBool(fusion_opdesc, NEED_INFER, true)) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][UpdAttr] set attr[%s] to node[%s] error", NEED_INFER.c_str(),
                        fusion_opdesc->GetName().c_str());
        return GRAPH_REPLACE_UPDATE_ATTR_FAILED;
      }
    }
  }
  return SUCCESS;
}

void GraphReplace::RecordFusionNodes(FusionRuleNodeMapping &fusion_graph,
                                     GraphMatchResult &match_result) const {
  FusionRuleNodeMapping::const_iterator it;
  for (it = fusion_graph.cbegin(); it != fusion_graph.cend(); it++) {
    match_result.fusion_nodes.push_back(it->second);
  }
}

Status GraphReplace::Replace(GraphMatchResult &match_result,
                             const FusionRuleNodeMapping &fusion_sub_graph,
                             const FusionRulePattern &fusion_rule_pattern, ge::ComputeGraph &graph) const {
  if (DeleteNodes(match_result.origin_nodes, fusion_rule_pattern.GetOriginRuleNodes(), graph) == FAILED) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][Replace] remove pre-fusion nodes error");
    return GRAPH_REPLACE_DELETE_NODE_FAILED;
  }

  for (auto fusion_item = fusion_sub_graph.begin(); fusion_item != fusion_sub_graph.end(); ++fusion_item) {
    FusionRuleNodePtr rule_node = fusion_item->first;
    ge::NodePtr fusion_node = fusion_item->second;

    // replace output anchors
    if (ReplaceOutputAnchors(rule_node, fusion_node, match_result.outer_outputs, fusion_sub_graph) == FAILED) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][Replace] replace output anchors error");
      return GRAPH_REPLACE_REPLACE_OUTPUT_FAILED;
    }
  }
  return SUCCESS;
}

void GraphReplace::PostFusion(const GraphMatchResult &match_result, const std::string &rule_name) const {
  SetDataDumpAttr(match_result);
  RecordOriginOpNames(match_result, rule_name);
}

void GraphReplace::RecordOriginOpNames(const GraphMatchResult &match_result, const std::string &rule_name) const {
  // Set origin op names from origin nodes
  vector<string> origin_op_names;
  vector<string> old_op_names;

  std::vector<ge::NodePtr> original_nodes;
  for (auto const &it : match_result.origin_nodes_set) {
    original_nodes.push_back(it);
  }

  for (auto node : match_result.fusion_nodes) {
    GraphPassUtil::RecordOriginalNames(original_nodes, node);
    GraphPassUtil::RecordOriginalOpAttrs(original_nodes, node->GetOpDesc(), rule_name);
    if (GraphPassUtil::StoreAndUpdataOriginFusionPassName(node->GetOpDesc(), original_nodes, rule_name) != SUCCESS) {
      FE_LOGD("Record node[%s] Origin fusion pass name unsuccessful", node->GetName().c_str());
    }
  }

  GraphPassUtil::InheritAttrFromOriNodes(original_nodes, match_result.fusion_nodes, BackWardInheritMode::kFusedNode);
  if (match_result.fusion_nodes.size() > 1) {
    bool is_multi_op = true;
    for (const ge::NodePtr &node : match_result.fusion_nodes) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, is_multi_op);
    }
  }
}

void GraphReplace::SetDataDumpAttr(const GraphMatchResult &match_result) const {
  // Get rule output anchor form match_result.outer_output
  for (auto &map_outer_output : match_result.outer_outputs) {
    FusionRuleAnchorPtr rule_anchor = map_outer_output.first;
    if (rule_anchor->GetPeerAnchors().empty()) {
      continue;
    }
    // Get output node anchor peer node output anchor idx
    int32_t rule_out_anchor_idx = rule_anchor->GetPeerAnchors().at(0)->GetAnchorIdx();
    if (rule_out_anchor_idx == -1) {
      continue;
    }
    // Get rule output node's anchor peer node
    FusionRuleNodePtr rule_node = rule_anchor->GetPeerAnchors().at(0)->GetOwnerNode();

    // Get matched graph node form match_result.origin_nodes
    auto it = match_result.origin_nodes.find(rule_node);
    if (it == match_result.origin_nodes.end()) {
      return;
    }
    ge::NodePtr origin_graph_node = it->second;

    std::set<ge::AnchorPtr> outer_output_set = map_outer_output.second;
    if (outer_output_set.empty()) {
      return;
    }

    const auto graph_input_anchor = dynamic_pointer_cast<ge::InDataAnchor>(*(outer_output_set.cbegin()));
    if (graph_input_anchor == nullptr || graph_input_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    // Get fusion node output anchor idx
    int32_t graph_out_anchor_idx = graph_input_anchor->GetPeerOutAnchor()->GetIdx();

    // Get fusion graph node
    ge::NodePtr fusion_graph_node = graph_input_anchor->GetPeerOutAnchor()->GetOwnerNode();

    // Set output desc
    (void)GraphPassUtil::SetOutputDescAttr(rule_out_anchor_idx, graph_out_anchor_idx, origin_graph_node,
                                           fusion_graph_node);
  }
}

ge::NodePtr GraphReplace::CreateNode(const FusionRuleNodePtr fusion_rule_node, const string &node_name,
                                     ge::ComputeGraph &graph) const {
  if (fusion_rule_node->GetNodeType().empty()) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][CrtNd] Node type of FusionRuleNode is empty.");
    return nullptr;
  }
  string node_type = fusion_rule_node->GetNodeType()[0];
  size_t in_anchor_num = fusion_rule_node->GetInputDataAnchors().size();
  size_t out_anchor_num = fusion_rule_node->GetOutputDataAnchors().size();
  ge::GeTensorDesc tensor_desc;
  auto node_op = ge::OperatorFactory::CreateOperator(node_name.c_str(), node_type.c_str());
  if (node_op.IsEmpty()) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][CrtNd] create fusion node[%s] error", node_type.c_str());
    return nullptr;
  }
  auto temp_opdesc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
  node_op.BreakConnect();

  ge::OpDescPtr op_desc = ge::AttrUtils::CopyOpDesc(temp_opdesc);
  for (size_t i = op_desc->GetInputsSize(); i < in_anchor_num; ++i) {
    if (op_desc->AddInputDesc(tensor_desc) != ge::SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][CrtNd] Failed to add input description for node [%s].", node_name.c_str());
      return nullptr;
    }
  }
  for (size_t i = op_desc->GetOutputsSize(); i < out_anchor_num; ++i) {
    if (op_desc->AddOutputDesc(tensor_desc) != ge::SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][CrtNd] Failed to add output description for node [%s].", node_name.c_str());
      return nullptr;
    }
  }
  ge::NodePtr node = graph.AddNode(op_desc);
  if (node == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][CrtNd] Failed to add node [%s].", node_name.c_str());
    return nullptr;
  }
  NodeMapInfoPtr node_map_info = nullptr;
  (void)GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph);
  (void)GraphPassUtil::AddNodeFromOpTypeMap(node_map_info, node);

  return node;
}

string GraphReplace::CreateNodeName(const FusionRuleNodeMapping &origin_sub_graph,
                                    const FusionRulePattern &fusion_rule_pattern, const vector<string> &types) const {
  ostringstream fusion_node_name;
  if (origin_sub_graph.empty()) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][CrtNdNm] Origin Sub Graph is empty.");
    return fusion_node_name.str();
  }

  static int fusion_node_count = 0;
  ge::NodePtr origin_node = origin_sub_graph.begin()->second;
  string node_name = origin_node->GetOpDesc()->GetName();
  vector<string> name_vec = StringUtils::Split(node_name, '/');
  for (size_t i = 0; i < name_vec.size(); ++i) {
    fusion_node_name << (name_vec[i] + "/");
  }
  if (!types.empty()) {
    fusion_node_name << fusion_rule_pattern.GetRuleName() << "/" << types[0] << fusion_node_count;
  }
  fusion_node_count = fusion_node_count % INT_MAX + 1;
  return fusion_node_name.str();
}

ge::NodePtr GraphReplace::FindSameNode(const FusionRuleNodePtr fusion_rule_node,
                                       const FusionRuleNodeMapping &origin_sub_graph) const {
  string node_name = fusion_rule_node->GetNodeName();
  for (auto const &item : origin_sub_graph) {
    if (item.first->GetNodeName() == node_name) {
      return item.second;
    }
  }
  return nullptr;
}
Status GraphReplace::DeleteNodes(const FusionRuleNodeMapping &nodes,
                                 const set<FusionRuleNodePtr> &rule_nodes, ge::ComputeGraph &graph) const {
  NodeMapInfoPtr node_map_info = nullptr;
  (void)GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph);

  for (auto item = rule_nodes.cbegin(); item != rule_nodes.cend(); ++item) {
    if (nodes.find(*item) == nodes.end()) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][DelNd] The node[%s] not in origin_sub_graph",
                      (*item)->GetNodeName().c_str());
      return FAILED;
    }

    ge::NodePtr node = nodes.at(*item);
    // delete input data anchors,
    for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
      auto in_data_anchor = node->GetInDataAnchor(i);
      FE_CHECK_NOTNULL(in_data_anchor);
      in_data_anchor->UnlinkAll();
    }
    // remove input control anchor
    auto in_control_anchor = node->GetInControlAnchor();
    FE_CHECK_NOTNULL(in_control_anchor);
    in_control_anchor->UnlinkAll();
    // remove node, RemoveNode function remove input and oupt anchor
    // in order to prevent automatic add edege when delete node, we should
    // remove input anchor firstly
    if (graph.RemoveNode(node) == ge::GRAPH_FAILED) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][DelNd] remove node[%s] error", node->GetName().c_str());
      return FAILED;
    }

    GraphNodeMapUtil::DelNodeFromOpTypeMap(node_map_info, node);
  }
  return SUCCESS;
}

Status GraphReplace::ReplaceInputAnchors(const FusionRuleNodePtr &rule_node, ge::NodePtr fusion_node,
                                         const map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> &outer_inputs,
                                         const FusionRuleNodeMapping &fusion_sub_graph) const {
  Status ret = ReplaceInputDataAnchors(rule_node, fusion_node, outer_inputs, fusion_sub_graph);
  if (ret == FAILED) {
    return ret;
  }

  ret = ReplaceInputCtrlAnchors(rule_node, fusion_node, outer_inputs, fusion_sub_graph);
  if (ret == FAILED) {
    return ret;
  }

  return SUCCESS;
}

Status GraphReplace::ReplaceInputCtrlAnchors(const FusionRuleNodePtr &rule_node, ge::NodePtr fusion_node,
                                             const map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> &outer_inputs,
                                             const FusionRuleNodeMapping &fusion_sub_graph) const {
  FE_LOGD("fused rule node:%s, graph node:%s", rule_node->GetNodeName().c_str(), fusion_node->GetName().c_str());
  FusionRuleAnchorPtr input_anchor = rule_node->GetInputCtrlAnchor();
  if (input_anchor == nullptr) {
    return SUCCESS;
  }

  std::set<ge::OutControlAnchorPtr> outer_ctrl_edges;
  ge::NodePtr peer_node = nullptr;
  ge::OutControlAnchorPtr src_anchor = nullptr;
  ge::InControlAnchorPtr dst_anchor = fusion_node->GetInControlAnchor();

  for (auto &peer_anchor : input_anchor->GetPeerAnchors()) {
    FusionRuleNodePtr peer_rule_node = peer_anchor->GetOwnerNode();
    // if the edge is between fusion node and fusion node, find peer node in
    // fusion sub graph
    if (outer_inputs.empty() && fusion_sub_graph.find(peer_rule_node) != fusion_sub_graph.end()) {
      peer_node = fusion_sub_graph.at(peer_rule_node);
      src_anchor = peer_node->GetOutControlAnchor();
      if (ge::GraphUtils::AddEdge(src_anchor, dst_anchor) == ge::GRAPH_FAILED) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplInCtrlAncr] Failed to add control edge from node [%s] to node [%s]",
                        peer_node->GetName().c_str(), fusion_node->GetName().c_str());
        return FAILED;
      }
    } else if (!outer_inputs.empty() && fusion_sub_graph.find(peer_rule_node) == fusion_sub_graph.end()) {
      // if the edge is between fusion node and outer input node, find peer
      // anchor in pre-fusion sub graph
      auto peer_anchors_pair = outer_inputs.find(peer_anchor);
      if (peer_anchors_pair == outer_inputs.end()) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplInCtrlAncr] outer input anchor[%s] not in match_result",
                        peer_anchor->GetAnchorName().c_str());
        return FAILED;
      }
      for (auto &peer_graph_anchor : peer_anchors_pair->second) {
        src_anchor = dynamic_pointer_cast<ge::OutControlAnchor>(peer_graph_anchor);
        if (src_anchor == nullptr) {
          continue;
        }
        outer_ctrl_edges.insert(src_anchor);
      }
    } else {
      // has been linked or do not need to add
      continue;
    }
  }

  for (auto &src_out_anchor : outer_ctrl_edges) {
    if (ge::GraphUtils::AddEdge(src_out_anchor, dst_anchor) == ge::GRAPH_FAILED) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplInCtrlAncr] add outer ctrl edge from node[%s] to node[%s] failed",
                      peer_node->GetName().c_str(), fusion_node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status GraphReplace::ReplaceInputDataAnchors(const FusionRuleNodePtr &rule_node, ge::NodePtr fusion_node,
                                             const map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> &outer_inputs,
                                             const FusionRuleNodeMapping &fusion_sub_graph) const {
  const vector<FusionRuleAnchorPtr> &input_anchor = rule_node->GetInputDataAnchors();
  ge::NodePtr peer_node = nullptr;
  ge::OutDataAnchorPtr src_anchor = nullptr;
  ge::InDataAnchorPtr dst_anchor = nullptr;
  vector<bool> is_input_const(input_anchor.size(), false);
  vector<bool> pre_is_input_const = fusion_node->GetOpDesc()->GetIsInputConst();
  if (!pre_is_input_const.empty()) {
    is_input_const = pre_is_input_const;
  }
  for (auto const &item : input_anchor) {
    vector<FusionRuleAnchorPtr> peer_anchors = item->GetPeerAnchors();
    int input_index = item->GetAnchorIdx();
    // peer anchor number must be 1, because the input anchor corresponds to
    // only one out anchor
    if (peer_anchors.size() != 1) {
      REPORT_FE_ERROR(
          "[GraphOpt][RunFusionRule][RplInDataAncr] FusionNode[%s]: peer anchor number is %zu, which must be 1.",
          fusion_node->GetName().c_str(), peer_anchors.size());
      return FAILED;
    }

    FusionRuleNodePtr peer_rule_node = peer_anchors[0]->GetOwnerNode();
    // if the edge is between fusion node and fusion node, find peer node in
    // fusion sub graph
    if (fusion_sub_graph.find(peer_rule_node) != fusion_sub_graph.end() && outer_inputs.empty()) {
      peer_node = fusion_sub_graph.at(peer_rule_node);
      int peer_output_index = peer_anchors[0]->GetAnchorIdx();
      src_anchor = peer_node->GetOutDataAnchor(peer_output_index);
    } else if (fusion_sub_graph.find(peer_rule_node) == fusion_sub_graph.end() &&
               !outer_inputs.empty()) {  // if the edge is between fusion node
                                         // and outer input node, find peer
                                         // anchor in pre-fusion sub graph
      auto peer_anchors_pair = outer_inputs.find(peer_anchors[0]);
      if (peer_anchors_pair == outer_inputs.end()) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplInDataAncr] outer input anchor[%s] not in match_result",
                        peer_anchors[0]->GetAnchorName().c_str());
        return FAILED;
      }
      if (!peer_anchors_pair->second.empty()) {
        ge::AnchorPtr peer_anchor = *(peer_anchors_pair->second.begin());
        src_anchor = dynamic_pointer_cast<ge::OutDataAnchor>(peer_anchor);
        if (src_anchor == nullptr) {
          continue;
        }
        peer_node = peer_anchor->GetOwnerNode();
      }

    } else {
      continue;
    }
    dst_anchor = fusion_node->GetInDataAnchor(input_index);
    // Because the output anchor can correspond to multiple inputs, we can
    // directly add edges
    FE_CHECK_NOTNULL(peer_node);
    if (ge::GraphUtils::AddEdge(src_anchor, dst_anchor) == ge::GRAPH_FAILED) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplInDataAncr] add edge from node[%s] to node[%s] error",
                      peer_node->GetName().c_str(), fusion_node->GetName().c_str());
      return FAILED;
    }

    if (peer_node->GetType() == CONSTANT) {
      is_input_const[input_index] = true;
    }
  }
  fusion_node->GetOpDesc()->SetIsInputConst(is_input_const);
  return SUCCESS;
}

Status GraphReplace::ReplaceOutputCtrlAnchors(const FusionRuleNodePtr &rule_node, ge::NodePtr fusion_node,
                                              const map<FusionRuleAnchorPtr, set<ge::AnchorPtr>> &outer_outputs,
                                              const FusionRuleNodeMapping &fusion_sub_graph) const {
  FusionRuleAnchorPtr output_anchor = rule_node->GetOutputCtrlAnchor();
  if (output_anchor == nullptr) {
    return SUCCESS;
  }

  const vector<FusionRuleAnchorPtr> &peer_anchors = output_anchor->GetPeerAnchors();
  ge::OutControlAnchorPtr src_anchor = fusion_node->GetOutControlAnchor();
  for (size_t i = 0; i < peer_anchors.size(); ++i) {
    FusionRuleAnchorPtr peer_anchor = peer_anchors[i];
    FusionRuleNodePtr peer_rule_node = peer_anchor->GetOwnerNode();
    // if the edge is between fusion node and fusion node, the edge had been
    // linked at ReplaceInputAnchors func if the edge is between fusion node
    // and origin graph node, find peer anchor in pre-fusion sub graph
    if (fusion_sub_graph.find(peer_rule_node) == fusion_sub_graph.end()) {
      if (outer_outputs.find(peer_anchor) == outer_outputs.end()) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplOutCtrlAncr] outer output anchor[%s] not in match result",
                        peer_anchor->GetAnchorName().c_str());
        return FAILED;
      }
      set<ge::AnchorPtr> outer_anchors = outer_outputs.at(peer_anchor);
      if (LinkOuterOutputEdges(src_anchor, outer_anchors) == FAILED) {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplOutCtrlAncr] link fusion node with outer node failed");
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status GraphReplace::ReplaceOutputDataAnchors(const FusionRuleNodePtr &rule_node, ge::NodePtr fusion_node,
                                              const map<FusionRuleAnchorPtr, set<ge::AnchorPtr>> &outer_outputs,
                                              const FusionRuleNodeMapping &fusion_sub_graph) const {
  const vector<FusionRuleAnchorPtr> &output_anchor = rule_node->GetOutputDataAnchors();
  ge::OutDataAnchorPtr src_anchor = nullptr;
  for (auto item = output_anchor.begin(); item != output_anchor.end(); ++item) {
    const vector<FusionRuleAnchorPtr> &peer_anchors = (*item)->GetPeerAnchors();
    int output_index = (*item)->GetAnchorIdx();
    src_anchor = fusion_node->GetOutDataAnchor(output_index);
    for (size_t i = 0; i < peer_anchors.size(); ++i) {
      FusionRuleAnchorPtr peer_anchor = peer_anchors[i];
      FusionRuleNodePtr peer_rule_node = peer_anchor->GetOwnerNode();
      // if the edge is between fusion node and fusion node, the edge had been
      // linked at ReplaceInputAnchors func if the edge is between fusion node
      // and origin graph node, find peer anchor in pre-fusion sub graph
      if (fusion_sub_graph.find(peer_rule_node) == fusion_sub_graph.end()) {
        if (outer_outputs.find(peer_anchor) == outer_outputs.end()) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplOutDataAncr] outer output anchor[%s] not in match result",
                          peer_anchor->GetAnchorName().c_str());
          return FAILED;
        }

        set<ge::AnchorPtr> outer_anchors = outer_outputs.at(peer_anchor);
        if (LinkOuterOutputEdges(src_anchor, outer_anchors) == FAILED) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RplOutDataAncr] link fusion node with outer node error");
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphReplace::ReplaceOutputAnchors(const FusionRuleNodePtr &rule_node, ge::NodePtr fusion_node,
                                          const map<FusionRuleAnchorPtr, set<ge::AnchorPtr>> &outer_outputs,
                                          const FusionRuleNodeMapping &fusion_sub_graph) const {
  Status ret = ReplaceOutputDataAnchors(rule_node, fusion_node, outer_outputs, fusion_sub_graph);
  if (ret == FAILED) {
    return ret;
  }

  ret = ReplaceOutputCtrlAnchors(rule_node, fusion_node, outer_outputs, fusion_sub_graph);
  if (ret == FAILED) {
    return ret;
  }

  return SUCCESS;
}

Status GraphReplace::LinkOuterOutputEdges(ge::AnchorPtr src_anchor, const set<ge::AnchorPtr> &outer_anchors) const {
  FE_CHECK_NOTNULL(src_anchor);
  for (ge::AnchorPtr const &outer_anchor : outer_anchors) {
    // outer_anchor maybe control input anchor or data input anchor
    if (ge::GraphUtils::AddEdge(src_anchor, outer_anchor) == ge::GRAPH_FAILED) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][LkOutOutEdge] add data edge from node[%s] to node[%s] error",
                      src_anchor->GetOwnerNode()->GetName().c_str(), outer_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphReplace::CheckFusionNode(GraphMatchResult &match_result,
                                     const FusionRuleNodeMapping &fusion_nodes) {
  vector<ge::NodePtr> sort_nodes;
  if (!LinkFusionNode(fusion_nodes)) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkFusNd] Failed to link edges between fusion nodes");
    return FAILED;
  }
  if (!TopoSortFusionNode(fusion_nodes, sort_nodes)) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkFusNd] sort fusion node failed");
    return FAILED;
  }
  map<ge::NodePtr, FusionRuleNodePtr> search_nodes;
  for (auto const &iter : fusion_nodes) {
    search_nodes[iter.second] = iter.first;
  }

  if (!LinkOuterInputsEdge(fusion_nodes, match_result.outer_inputs_set)) {
    REPORT_FE_ERROR(
        "[GraphOpt][RunFusionRule][ChkFusNd] Failed to link edges between the fusion nodes and outer nodes.");
    return FAILED;
  }
  if (!InferShapeDtypeAndFormat(sort_nodes, search_nodes, match_result.outer_outputs)) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkFusNd] infer shape, data type or origin format failed");
    return FAILED;
  }
  if (!CheckSupported(sort_nodes)) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkFusNd] check shape and data type support failed");
    return FAILED;
  }
  if (!CheckShapeAndTypeContinuous(match_result.outer_outputs, search_nodes, sort_nodes)) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkFusNd] check shape and data type support failed");
    return FAILED;
  }
  return SUCCESS;
}

bool GraphReplace::LinkFusionNode(const FusionRuleNodeMapping &fusion_nodes) const {
  for (auto &item : fusion_nodes) {
    FusionRuleNodePtr rule_node = item.first;
    ge::NodePtr fusion_node = item.second;
    FE_LOGD("link fused rule node:%s, graph node:%s", rule_node->GetNodeName().c_str(), fusion_node->GetName().c_str());
    if (ReplaceInputAnchors(rule_node, fusion_node, {}, fusion_nodes) == FAILED) {
      return false;
    }
  }
  return true;
}

bool GraphReplace::TopoSortFusionNode(const FusionRuleNodeMapping &fusion_nodes,
                                      vector<ge::NodePtr> &sort_nodes) const {
  stack<ge::NodePtr> node_stack;
  map<ge::NodePtr, int> node_inputs_map;
  // find the node without input, and compute the input numbers of other nodes
  for (auto const &item : fusion_nodes) {
    ge::NodePtr node = item.second;
    int input_size = 0;
    for (ge::InDataAnchorPtr const &anchor : node->GetAllInDataAnchors()) {
      if (anchor != nullptr) {
        if (CheckInt32AddOverflow(input_size, anchor->GetPeerAnchors().size()) == FAILED) {
          REPORT_FE_ERROR("[GraphOpt][RunFusionRule][TpsrFusNd] fusion node:%s peer anchor size is too much.",
                          node->GetName().c_str());
          return false;
        }
        input_size += anchor->GetPeerAnchors().size();
      }
    }
    if (input_size == 0) {
      node_stack.push(node);
    } else {
      node_inputs_map[node] = input_size;
    }
  }
  while (!node_stack.empty()) {
    ge::NodePtr node = node_stack.top();
    node_stack.pop();
    sort_nodes.push_back(node);
    FE_LOGD("sort nodes push back node[%s]", node->GetName().c_str());
    // the value of node_input_map is node's input number,
    // when the value is zero, representing the parent nodes of this node
    // have been visited and this node can be sorted
    for (ge::OutDataAnchorPtr const &anchor : node->GetAllOutDataAnchors()) {
      for (ge::InDataAnchorPtr const &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        auto iter = node_inputs_map.find(peer_in_anchor->GetOwnerNode());
        if (iter != node_inputs_map.end() && --(iter->second) == 0) {
          node_stack.push(peer_in_anchor->GetOwnerNode());
        }
      }
    }
  }
  if (sort_nodes.size() != fusion_nodes.size()) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][TpsrFusNd] sort nodes size not equal with fusion nodes");
    return false;
  }
  return true;
}

bool GraphReplace::LinkOuterInputsEdge(const FusionRuleNodeMapping &fusion_nodes,
                                       const map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> &outer_inputs) const {
  for (auto const &item : fusion_nodes) {
    FusionRuleNodePtr rule_node = item.first;
    ge::NodePtr fusion_node = item.second;
    FE_LOGD("linkouter fused rule node:%s, graph node:%s", rule_node->GetNodeName().c_str(),
            fusion_node->GetName().c_str());
    if (ReplaceInputAnchors(rule_node, fusion_node, outer_inputs, fusion_nodes) == FAILED) {
      return false;
    }
  }
  return true;
}

bool GraphReplace::InferShapeDtypeAndFormat(const vector<ge::NodePtr> &sort_nodes,
                                            const map<ge::NodePtr, FusionRuleNodePtr> &fusion_nodes,
                                            const map<FusionRuleAnchorPtr, set<ge::AnchorPtr>> &outer_outputs) const {
  for (auto const &node : sort_nodes) {
    for (ge::InDataAnchorPtr const &anchor : node->GetAllInDataAnchors()) {
      ge::OutDataAnchorPtr peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor != nullptr) {
        // if the input of fusion node is outer node, update output tensor to
        // input opdesc
        ge::NodePtr peer_node = peer_anchor->GetOwnerNode();
        if (fusion_nodes.find(peer_node) == fusion_nodes.end()) {
          auto output_tensor = peer_node->GetOpDesc()->GetOutputDescPtr(peer_anchor->GetIdx());
          FE_CHECK(output_tensor == nullptr,
                   REPORT_FE_ERROR("[GraphOpt][RunFusionRule][InfShpDtpFmt] Output_tensor is null."), return false);
          node->GetOpDesc()->UpdateInputDesc(anchor->GetIdx(), *output_tensor);
        }
      } else {
        REPORT_FE_ERROR("[GraphOpt][RunFusionRule][InfShpDtpFmt] node[%s] peer anchor [%d] is null",
                        node->GetName().c_str(), anchor->GetIdx());
        return false;
      }
    }
    // if there is only one fusion node update output tensor with its child
    // nodes input tensor
    if (sort_nodes.size() == 1) {
      return UpdateSingleFusedNode(node, fusion_nodes, outer_outputs);
    }
    // 3. if there are many fusion nodes: InferShapeAndType and InferFormat for
    // the output desc of the fusion node
    if (!InferShape(node)) {
      return false;
    }

    // 4. update the input desc for the peer node
    UpdateInputDescForPeerNode(node, fusion_nodes);
  }
  return true;
}

bool GraphReplace::CheckSupported(const vector<ge::NodePtr> &sort_nodes) {
  for (auto const &node : sort_nodes) {
    auto opdesc = node->GetOpDesc();
    if (IsPlaceOrEnd(opdesc->GetType())) {
      continue;
    }
    string un_supported_reason;
    if (!ops_kernel_info_store_ptr_->CheckSupported(node, un_supported_reason)) {
      REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkSpt] Node[%s] check_shape_and_type failed, reason is %s",
                      node->GetName().c_str(), un_supported_reason.c_str());
      return false;
    }
  }
  return true;
}

bool GraphReplace::CheckShapeAndTypeContinuous(const map<FusionRuleAnchorPtr, set<ge::AnchorPtr>> &outer_outputs,
                                               const map<ge::NodePtr, FusionRuleNodePtr> &fusion_nodes,
                                               const vector<ge::NodePtr> &sort_nodes) const {
  // if there is only one fusion node, the input tensor is the same as parent
  // node,
  // the output tensor is the same as child node
  if (sort_nodes.size() == 1) {
    return true;
  }
  for (auto const &node : sort_nodes) {
    auto rule_node = fusion_nodes.at(node);
    const vector<FusionRuleAnchorPtr> &output_anchor = rule_node->GetOutputDataAnchors();
    for (const auto &out_anchor : output_anchor) {
      const vector<FusionRuleAnchorPtr> &peer_rule_anchors = out_anchor->GetPeerAnchors();
      int output_index = out_anchor->GetAnchorIdx();
      auto output_data_anchor = node->GetOutDataAnchor(output_index);
      for (const auto &peer_rule_input_anchor : peer_rule_anchors) {
        if (outer_outputs.find(peer_rule_input_anchor) == outer_outputs.end()) {
          continue;
        }
        auto input_anchors = outer_outputs.at(peer_rule_input_anchor);
        for (auto const &input_anchor : input_anchors) {
          ge::InDataAnchorPtr dst_data_anchor = ge::Anchor::DynamicAnchorCast<ge::InDataAnchor>(input_anchor);
          // control anchor
          if (dst_data_anchor == nullptr) {
            continue;
          }
          // fusion node output shape and data type should be same with peer
          // outer node input shape and data type
          if (!CheckDataType(output_data_anchor, dst_data_anchor)) {
            REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkShpTypeContus] node[%s] check_data_type failed",
                            node->GetName().c_str());
            return false;
          }

          if (!CheckShape(output_data_anchor, dst_data_anchor)) {
            REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkShpTypeContus] node[%s] check_shape failed",
                            node->GetName().c_str());
            return false;
          }
        }
      }
    }
  }
  return true;
}
bool GraphReplace::CheckShape(ge::OutDataAnchorPtr out_anchor, ge::InDataAnchorPtr peer_in_anchor) const {
  FE_CHECK_NOTNULL(out_anchor);
  FE_CHECK_NOTNULL(peer_in_anchor);
  auto peer_node = peer_in_anchor->GetOwnerNode();
  auto node = out_anchor->GetOwnerNode();
  auto opdesc = node->GetOpDesc();
  auto peer_opdesc = peer_node->GetOpDesc();
  auto output_desc_ptr = opdesc->MutableOutputDesc(out_anchor->GetIdx());
  auto input_desc_ptr = peer_opdesc->MutableInputDesc(peer_in_anchor->GetIdx());
  if (output_desc_ptr == nullptr || input_desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkShp] node [%s] output or peer input is null",
                    opdesc->GetName().c_str());
    return false;
  }
  auto &output_shape = output_desc_ptr->MutableShape();
  auto &input_shape = input_desc_ptr->MutableShape();
  if (output_shape.GetDims() != input_shape.GetDims()) {
    FE_LOGW("node [%s] output[%d] should be equal with node[%s] input [%d].",
            node->GetName().c_str(), out_anchor->GetIdx(), peer_node->GetName().c_str(),
            peer_in_anchor->GetIdx());
    return false;
  }
  return true;
}
bool GraphReplace::CheckDataType(ge::OutDataAnchorPtr out_anchor, ge::InDataAnchorPtr peer_in_anchor) const {
  FE_CHECK_NOTNULL(out_anchor);
  auto peer_node = peer_in_anchor->GetOwnerNode();
  auto node = out_anchor->GetOwnerNode();
  auto opdesc = node->GetOpDesc();
  auto peer_opdesc = peer_node->GetOpDesc();
  ge::ConstGeTensorDescPtr output_desc_ptr = opdesc->GetOutputDescPtr(out_anchor->GetIdx());
  ge::ConstGeTensorDescPtr input_desc_ptr = peer_opdesc->GetInputDescPtr(peer_in_anchor->GetIdx());
  if (output_desc_ptr == nullptr || input_desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][ChkDatatype] node [%s] output or peer input is null",
                    opdesc->GetName().c_str());
    return false;
  }
  ge::DataType output_dtype = output_desc_ptr->GetDataType();
  ge::DataType input_dtype = input_desc_ptr->GetDataType();
  if (output_dtype != input_dtype) {
    FE_LOGW("node [%s] output [%d] data type [%d] should be equal with node [%s] input [%d] data type [%d]",
            node->GetName().c_str(), out_anchor->GetIdx(), output_dtype, peer_node->GetName().c_str(),
            peer_in_anchor->GetIdx(), input_dtype);
    return false;
  }
  return true;
}
}  // namespace fe
