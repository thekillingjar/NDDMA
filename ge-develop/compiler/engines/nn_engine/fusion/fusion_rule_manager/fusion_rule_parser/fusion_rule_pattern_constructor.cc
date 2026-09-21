/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_pattern_constructor.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"

using std::map;
using std::string;
using std::vector;

namespace fe {
namespace {
// 2 means that this input anchor already valid
static const int ALREADY_DONE = 2;
// 1 means that this input anchor has be visited twice
static const int VISTIED_TWICE = 1;
// -1 means that this input anchor hasn't be visit
static const int INIT_STATUS = -1;
// -2 means that this anchor index is default
static const int ANCHOR_INDEX_DEFAULT = -2;

Status DumpIns(FusionRuleNodePtr node) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpIns] Input node is null."),
           return INTERNAL_ERROR);

  vector<FusionRuleAnchorPtr> input_anchors = node->GetInputDataAnchors();
  FusionRuleAnchorPtr input_ctrl_anchor = node->GetInputCtrlAnchor();
  if (input_ctrl_anchor != nullptr) {
    input_anchors.push_back(input_ctrl_anchor);
  }

  FE_LOGD("        InputAnchors:");
  for (const auto &anchor : input_anchors) {
    FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpIns] Get anchor is null."),
             return INTERNAL_ERROR);
    for (const auto &peer_anchor : anchor->GetPeerAnchors()) {
      FE_CHECK(peer_anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpIns] Get peer_anchor is null."),
               return INTERNAL_ERROR);
      FE_LOGD("            [%d]<-node:%s[%d]", anchor->GetAnchorIdx(),
              peer_anchor->GetOwnerNode()->GetNodeName().c_str(), peer_anchor->GetAnchorIdx());
    }
  }

  return SUCCESS;
}

Status DumpOuts(FusionRuleNodePtr node) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpOuts] Input node is null."),
           return INTERNAL_ERROR);
  vector<FusionRuleAnchorPtr> output_anchors = node->GetOutputDataAnchors();
  FusionRuleAnchorPtr output_ctrl_anchor = node->GetOutputCtrlAnchor();
  if (output_ctrl_anchor != nullptr) {
    output_anchors.push_back(output_ctrl_anchor);
  }

  FE_LOGD("        OutputAnchors:");
  for (const auto &anchor : output_anchors) {
    FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpOuts] Get anchor is null."),
             return INTERNAL_ERROR);
    for (const auto &peer_anchor : anchor->GetPeerAnchors()) {
      FE_CHECK(peer_anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpOuts] Get peer_anchor is null."),
               return INTERNAL_ERROR);
      FE_LOGD("            [%d]->node:%s[%d]", anchor->GetAnchorIdx(),
              peer_anchor->GetOwnerNode()->GetNodeName().c_str(), peer_anchor->GetAnchorIdx());
    }
  }

  return SUCCESS;
}

Status DumpAttr(FusionRuleNodePtr node) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpAttr] Input node is null."),
           return INTERNAL_ERROR);
  auto attrs = node->GetAttributes();
  if (attrs.size() > 0) {
    FE_LOGD("        Attrs:");
    for (const auto &attr : attrs) {
      string attr_value;
      if (attr.second->IsFusionRuleAttr()) {
        auto fusion_rule_attr = attr.second->GetRuleNodeAttrValue();
        auto refleced_node = attr.second->GetOwnerNode();
        FE_CHECK(refleced_node == nullptr,
                  REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpAttr] Get owner node from %s failed",
                  attr.first.c_str()), return INTERNAL_ERROR);
        if (fusion_rule_attr.node_name != refleced_node->GetNodeName()) {
          REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpAttr] Attr owner node:%s not equal to it should be:%s",
                          refleced_node->GetNodeName().c_str(), fusion_rule_attr.node_name.c_str());
          return INTERNAL_ERROR;
        }
        attr_value = fusion_rule_attr.node_name + "." + fusion_rule_attr.attr_name;
      } else {
        auto fix_attr_value = attr.second->GetFixAttrValue();
        attr_value = GetStrFromAttrValue(fix_attr_value);
      }
      FE_LOGD("            %s:%s", attr.first.c_str(), attr_value.c_str());
    }
  }

  return SUCCESS;
}

/*
 *  @brief: Create dummy node for FusionRuleJsonAnchor, if anchor has "src",
 * create dummy node according to
 *          "src" info; otherwise, create dummy node with same name of
 * FusionRuleJsonAnchor
 */
Status CreateOuterInputNode(map<string, FusionRuleNodePtr> &input_nodes_map, FusionRuleJsonOuterPtr anchor,
                            FusionRuleNodePtr &input_node) {
  if (anchor->HasSrc()) {
    // if an outer input has "src" means that it has a specified input node
    if (input_nodes_map.find(anchor->GetSrcNode()) == input_nodes_map.end()) {
      // if can't find src node in map, create new one
      FE_MAKE_SHARED(input_node = make_shared<FusionRuleNode>(), return INTERNAL_ERROR);
      Status ret = FusionRuleNodeConstructor::Construct(input_node, anchor->GetSrcNode(), {});
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtOutInNd] Create dummy outer input node:%s failed.",
                        anchor->GetSrcNode().c_str());
        return ret;
      }

      input_nodes_map.emplace(make_pair(anchor->GetSrcNode(), input_node));
    } else {
      // otherwise, use the found node as it's own node
      input_node = input_nodes_map[anchor->GetSrcNode()];
    }
  } else {
    string input_name = anchor->GetName();
    // make a dummy node for isolated input anchor
    if (input_nodes_map.find(input_name) != input_nodes_map.end()) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtOutInNd] Input node:%s already define, redeclaration.",
                      input_name.c_str());
      return ILLEGAL_RULE;
    }

    FE_MAKE_SHARED(input_node = make_shared<FusionRuleNode>(), return INTERNAL_ERROR);
    Status ret = FusionRuleNodeConstructor::Construct(input_node, input_name, {});
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtOutInNd] Create dummy outer input node:%s failed.",
                      input_name.c_str());
      return ret;
    }
    input_nodes_map.emplace(make_pair(input_name, input_node));
  }

  return SUCCESS;
}

FusionRuleNodePtr FindNodeByName(const vector<FusionRuleNodePtr> &outer_info, string name) {
  for (const auto &node : outer_info) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][FdNdByNm] Outer node is null."),
             return nullptr);
    if (node->GetNodeName() == name) {
      return node;
    }
  }

  return nullptr;
}

FusionRuleAnchorPtr FindSrcAnchorByName(const vector<FusionRuleNodePtr> &input_info, string name) {
  for (const auto &node : input_info) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][FdSrcAnrByNm] Input node is null."),
             return nullptr);
    for (const auto &output_anchor : node->GetOutputDataAnchors()) {
      FE_CHECK(output_anchor == nullptr,
               REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][FdSrcAnrByNm] OutputAnchor is null."),
               return nullptr);
      if (output_anchor->GetAnchorName() == name) {
        return output_anchor;
      }
    }
  }

  return nullptr;
}

FusionRuleAnchorPtr FindDstAnchorByName(const vector<FusionRuleNodePtr> &output_info, string name) {
  for (const auto &node : output_info) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][FdDstAnrByNm] Input node is null."),
             return nullptr);
    for (const auto &input_anchor : node->GetInputDataAnchors()) {
      FE_CHECK(input_anchor == nullptr,
               REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][FdDstAnrByNm] InputAnchor is null."),
               return nullptr);
      if (input_anchor->GetAnchorName() == name) {
        return input_anchor;
      }
    }
  }

  return nullptr;
}

FusionRuleAnchorPtr FindAnchorByIndex(const vector<FusionRuleAnchorPtr> anchors, int index) {
  for (const auto &anchor : anchors) {
    FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][FdAnrByIndex] Input anchor is null."),
             return nullptr);
    if (anchor->GetAnchorIdx() == index) {
      return anchor;
    }
  }
  return nullptr;
}

FusionRuleAnchorPtr CreateNewNodeAnchor(FusionRuleNodePtr node, int anchor_idx, string name, int in_out_flag) {
  FusionRuleAnchorPtr anchor = nullptr;
  FE_MAKE_SHARED(anchor = make_shared<FusionRuleAnchor>(), return nullptr);
  if (FusionRuleAnchorConstructor::Construct(anchor, anchor_idx, name) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtNewNdAnr] Create anchor of node:%s's output:%d failed.",
                    node->GetNodeName().c_str(), anchor_idx);
    return nullptr;
  }
  if (in_out_flag == 0) {
    if (FusionRuleNodeConstructor::AddOutputAnchor(node, anchor) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtNewNdAnr] Add output anchor:%d to node:%s failed.", anchor_idx,
                      node->GetNodeName().c_str());
      return nullptr;
    }
  } else {
    if (FusionRuleNodeConstructor::AddInputAnchor(node, anchor) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtNewNdAnr] Add input anchor:%d to node:%s failed.", anchor_idx,
                      node->GetNodeName().c_str());
      return nullptr;
    }
  }

  return anchor;
}

FusionRuleAnchorPtr CreateInnerOutAnchor(FusionRuleJsonAnchorPtr src, map<string, FusionRuleNodePtr> &nodes,
                                         const vector<FusionRuleNodePtr> &input_info) {
  FE_CHECK(src == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtInnerOutAnr] Input src is null."),
           return nullptr);

  FusionRuleAnchorPtr anchor = nullptr;
  FusionRuleNodePtr node = nullptr;

  if (src->HasIndex()) {
    // has index means the src must be a inner node anchor
    string src_node = src->GetSrcNode();
    int src_index = src->GetSrcIndex();

    auto src_iter = nodes.find(src_node);
    if (src_iter == nodes.end()) {
      node = FindNodeByName(input_info, src_node);
    } else {
      node = src_iter->second;
    }
    FE_CHECK(
        node == nullptr,
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtInnerOutAnr] Can't find node:%s in graph.", src_node.c_str()),
        return nullptr);

    // first, find whether the anchor already be created
    vector<FusionRuleAnchorPtr> output_anchors = node->GetOutputDataAnchors();
    if (node->GetOutputCtrlAnchor()) {
      output_anchors.push_back(node->GetOutputCtrlAnchor());
    }
    anchor = FindAnchorByIndex(output_anchors, src_index);
    if (anchor == nullptr) {
      // otherwise create new one
      anchor = CreateNewNodeAnchor(node, src_index, "", 0);
      if (anchor == nullptr) {
        return nullptr;
      }
    }
  } else {
    // src don't have index, may be a inner node, need to set default index 0
    string anchor_name = src->GetName();
    auto src_iter = nodes.find(anchor_name);
    if (src_iter != nodes.end()) {
      node = src_iter->second;
      // firstly, find if the anchor already be created
      anchor = FindAnchorByIndex(node->GetOutputDataAnchors(), 0);
      if (anchor == nullptr) {
        anchor = CreateNewNodeAnchor(node, 0, "", 0);
        if (anchor == nullptr) {
          return nullptr;
        }
      }
    } else {
      // otherwise, the anchor should be from outer input, or return
      // ILLEGAL_RULE
      anchor = FindSrcAnchorByName(input_info, anchor_name);
      FE_CHECK(anchor == nullptr,
               REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtInnerOutAnr] Can't find src node %s.",
                               anchor_name.c_str()),
               return nullptr);
    }
  }

  return anchor;
}

FusionRuleAnchorPtr CreateInnerInAnchor(FusionRuleJsonAnchorPtr dst, map<string, FusionRuleNodePtr> &nodes,
                                        const vector<FusionRuleNodePtr> &output_info) {
  FE_CHECK(dst == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtInnerInAnr] Input dst is null."),
           return nullptr);
  FusionRuleAnchorPtr anchor = nullptr;
  FusionRuleNodePtr node = nullptr;

  if (dst->HasIndex()) {
    // has index means the dst must be a inner node anchor
    string src_node = dst->GetSrcNode();
    int src_index = dst->GetSrcIndex();
    auto src_iter = nodes.find(src_node);
    if (src_iter == nodes.end()) {
      node = FindNodeByName(output_info, src_node);
    } else {
      node = src_iter->second;
    }
    FE_CHECK(
        node == nullptr,
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtInnerInAnr] Can't find node:%s in graph.", src_node.c_str()),
        return nullptr);
    // firstly, find if the anchor already be created
    vector<FusionRuleAnchorPtr> input_anchors = node->GetInputDataAnchors();
    if (node->GetInputCtrlAnchor()) {
      input_anchors.push_back(node->GetInputCtrlAnchor());
    }
    anchor = FindAnchorByIndex(input_anchors, src_index);
    if (anchor == nullptr) {
      anchor = CreateNewNodeAnchor(node, src_index, "", 1);
      if (anchor == nullptr) {
        return nullptr;
      }
    }
  } else {
    // dst don't have index, may be a inner node, need to set default index 0
    string anchor_name = dst->GetName();
    auto src_iter = nodes.find(anchor_name);
    if (src_iter != nodes.end()) {
      node = src_iter->second;
      // firstly, find if the anchor already be created
      anchor = FindAnchorByIndex(node->GetInputDataAnchors(), 0);
      if (anchor == nullptr) {
        anchor = CreateNewNodeAnchor(node, 0, "", 1);
        if (anchor == nullptr) {
          return nullptr;
        }
      }
    } else {
      // otherwise, the anchor should be from outer input, or return
      // ILLEGAL_RULE
      anchor = FindDstAnchorByName(output_info, anchor_name);
      FE_CHECK(anchor == nullptr,
               REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][CrtInnerInAnr] Can't find dst node %s.",
                               dst->GetName().c_str()),
               return nullptr);
    }
  }

  return anchor;
}

Status AddAttrAssignmentExpression(const map<string, FusionRuleNodePtr> &nodes, FusionRuleJsonGraphPtr json_graph) {
  FE_CHECK(json_graph == nullptr,
           REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddAttrAsgnEpr] Input json_graph is null."),
           return ILLEGAL_RULE);

  for (const auto &attr_assign : json_graph->GetAttrAssigns()) {
    auto node_iter = nodes.find(attr_assign->GetAttr().node_name);
    if (node_iter == nodes.end()) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][AddAttrAsgnEpr] Can't find node:%s attr:%s in fusion graph inner nodes",
          attr_assign->GetAttr().node_name.c_str(), attr_assign->GetAttr().attr_name.c_str());
      return ILLEGAL_RULE;
    }

    FusionRuleNodePtr node = node_iter->second;
    Status ret = FusionRuleNodeConstructor::AddAttr(node, attr_assign->GetAttr().attr_name, attr_assign->GetValue());
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddAttrAsgnEpr] Add attr:%s to node:%s failed",
                      attr_assign->GetAttr().attr_name.c_str(), attr_assign->GetAttr().node_name.c_str());
      return ret;
    }
  }

  return SUCCESS;
}

Status LoadNodes(const FusionRulePatternPtr &pattern, const FusionRuleJsonGraphPtr &json_graph,
                 set<FusionRuleNodePtr> &rule_nodes, map<string, FusionRuleNodePtr> &nodes) {
  for (const auto &iter : json_graph->GetNodes()) {
    string node_name = iter->GetName();
    // check node.name must be unique both inner node and outer node
    if (nodes.find(node_name) != nodes.end()) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdNd] Inner graph node:%s already define in inner, redeclaration.",
                      node_name.c_str());
      return ILLEGAL_RULE;
    }
    if (FindSrcAnchorByName(pattern->GetInputInfo(), node_name) != nullptr) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][LdNd] Inner graph node:%s already define in outer inputs, redeclaration.",
          node_name.c_str());
      return ILLEGAL_RULE;
    }
    if (FindDstAnchorByName(pattern->GetOutputInfo(), node_name) != nullptr) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][LdNd] Inner graph node:%s already define in outer outputs, redeclaration.",
          node_name.c_str());
      return ILLEGAL_RULE;
    }

    // creat a new FusionRuleNode according to FusionRuleJsonNode info
    FusionRuleNodePtr node = nullptr;
    FE_MAKE_SHARED(node = make_shared<FusionRuleNode>(), return INTERNAL_ERROR);
    Status ret = FusionRuleNodeConstructor::Construct(node, node_name, iter->GetTypes());
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdNd] Create node:%s failed", node_name.c_str());
      return ret;
    }

    nodes.emplace(make_pair(node_name, node));
    rule_nodes.emplace(node);
  }
  return SUCCESS;
}

Status RecordAllNode(const vector<FusionRuleNodePtr> &input_info, const set<FusionRuleNodePtr> inner_nodes,
                     vector<FusionRuleNodePtr> &nodes) {
  for (const auto &node : input_info) {
    nodes.push_back(node);
  }
  for (const auto &node : inner_nodes) {
    nodes.push_back(node);
  }

  return SUCCESS;
}

Status SetMarkMap(const vector<FusionRuleNodePtr> &pre_nodes, map<FusionRuleNodePtr, map<int, int>> &mark_map) {
  for (const auto &node : pre_nodes) {
    for (const auto &anchor : node->GetInputDataAnchors()) {
      int index = (anchor->GetAnchorIdx() == ANCHOR_INDEX_DEFAULT) ? 0 : anchor->GetAnchorIdx();
      mark_map[node][index] = INIT_STATUS;
    }
  }

  return SUCCESS;
}

Status CheckCurrentNode(FusionRuleNodePtr &current_node, map<FusionRuleNodePtr, map<int, int>> &mark_map,
                        set<FusionRuleNodePtr> &buffer, bool &is_jump) {
  FE_CHECK(current_node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkCurtNd] Input node is null."),
           return ILLEGAL_RULE);

  for (const auto &iter : mark_map[current_node]) {
    if (iter.second != ALREADY_DONE) {
      // if current node's anchor has be visited twice, it should be in a loop
      if (iter.second == VISTIED_TWICE) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkCurtNd] Node:%s may in the loop.",
                        current_node->GetNodeName().c_str());
        return ILLEGAL_RULE;
      }
      // if buffer isn't null, choose head node in buffer as next search node
      if (buffer.size() != 0) {
        current_node = *(buffer.begin());
        buffer.erase(current_node);
      } else {
        // otherwise search the unsatisfied input node
        ++mark_map[current_node][iter.first];  // mark the current anchor has be visited once
        auto anchor = (current_node->GetInputDataAnchors())[iter.first];
        FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkCurtNd] Anchor is null."),
                 return ILLEGAL_RULE);

        if (anchor->GetPeerAnchors().size() == 0) {
          REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkCurtNd] Node:%s[%d] have no peer anchor.",
                          current_node->GetNodeName().c_str(), iter.first);
          return ILLEGAL_RULE;
        }
        auto peer_anchor = (anchor->GetPeerAnchors())[0];
        FE_CHECK(peer_anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkCurtNd] peerAnchor is null."),
                 return ILLEGAL_RULE);

        current_node = peer_anchor->GetOwnerNode();
        FE_CHECK(current_node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkCurtNd] currentNode is null."),
                 return ILLEGAL_RULE);
      }
      is_jump = true;
      break;
    }
  }

  return SUCCESS;
}

/*
 * @brief: If doing fusion graph topological sort, record inner node that
 * directly connect to OuterOutput to buffer
 */
Status InitBuffer(const set<FusionRuleNodePtr> inner_nodes, const vector<FusionRuleNodePtr> &output_info,
                  bool is_fusion_graph, set<FusionRuleNodePtr> &buffer) {
  if (is_fusion_graph) {
    for (const auto &node : output_info) {
      FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][InitBuf] Get node is null."),
               return ILLEGAL_RULE);
      for (const auto &in_anchor : node->GetInputDataAnchors()) {
        FE_CHECK(in_anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][InitBuf] Get input anchor is null."),
                 return ILLEGAL_RULE);
        for (const auto &peer_out_anchor : in_anchor->GetPeerAnchors()) {
          FE_CHECK(peer_out_anchor == nullptr,
                   REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][InitBuf] Get peer anchor is null."),
                   return ILLEGAL_RULE);
          FusionRuleNodePtr peer_node = peer_out_anchor->GetOwnerNode();
          FE_CHECK(peer_node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][InitBuf] Get peer node is null."),
                   return ILLEGAL_RULE);
          if (inner_nodes.find(peer_node) != inner_nodes.end()) {
            buffer.emplace(peer_node);
          }
        }
      }
    }
  }

  return SUCCESS;
}

/*
 * @brief: When a node's all input anchors are ready, means that this node is ok
 * to record to sorted_nodes,
 *         then choose the node to search next, and record remain output node to
 * buffer
 */
void ProcessReadyNode(const set<FusionRuleNodePtr> &inner_nodes, FusionRuleNodePtr &current_node,
                      vector<FusionRuleNodePtr> &sorted_nodes, map<FusionRuleNodePtr, map<int, int>> &mark_map,
                      set<FusionRuleNodePtr> &buffer, bool &selected_flag) {
  // if go to this, means current node all input has be recored
  if (inner_nodes.find(current_node) != inner_nodes.end()) {
    sorted_nodes.push_back(current_node);
  }
  // delete current node in buffer
  if (buffer.find(current_node) != buffer.end()) {
    buffer.erase(current_node);
  }
  // as for an sorted node, choose it's first output node as next search node,
  // record other nodes to buffer
  for (const auto &src_anchor : current_node->GetOutputDataAnchors()) {
    for (const auto &dst_anchor : src_anchor->GetPeerAnchors()) {
      auto node = dst_anchor->GetOwnerNode();
      // skip output node not in inner_nodes
      if (inner_nodes.find(node) == inner_nodes.end()) {
        continue;
      }
      // Set mark_map of edge's status ALREADY_DONE, which is from current node
      // to it's output node
      mark_map[node][dst_anchor->GetAnchorIdx()] = ALREADY_DONE;
      if (!selected_flag) {
        current_node = node;
        selected_flag = true;
      } else if (buffer.find(node) == buffer.end()) {
        buffer.emplace(node);
      }
    }
  }
}
}  // namespace

Status FusionRulePatternConstructor::DumpInfo(FusionRulePatternPtr pattern) {
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Input pattern is null."),
           return INTERNAL_ERROR);
  FE_LOGD("The fusion rule %s loaded is like this.", pattern->GetRuleName().c_str());
  FE_LOGD("Input info:");
  for (const auto &node : pattern->input_info_) {
    FE_LOGD("    node:%s", node->GetNodeName().c_str());
    if (DumpOuts(node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Dump node:%s output info failed.",
                      node->GetNodeName().c_str());
      return INTERNAL_ERROR;
    }
  }

  FE_LOGD("Output info:");
  for (const auto &node : pattern->output_info_) {
    FE_LOGD("    node:%s", node->GetNodeName().c_str());
    if (DumpIns(node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Dump node:%s input info failed.",
                      node->GetNodeName().c_str());
      return INTERNAL_ERROR;
    }
  }

  FE_LOGD("Origin graph:");
  for (const auto &node : pattern->origin_rule_nodes_) {
    FE_LOGD("    node:%s", node->GetNodeName().c_str());
    if (DumpIns(node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Dump node:%s input info failed.",
                      node->GetNodeName().c_str());
      return INTERNAL_ERROR;
    }
    if (DumpOuts(node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Dump node:%s output info failed.",
                      node->GetNodeName().c_str());
      return INTERNAL_ERROR;
    }
  }

  FE_LOGD("Fusion graph:");
  for (const auto &node : pattern->fusion_rule_nodes_) {
    FE_LOGD("    node:%s", node->GetNodeName().c_str());
    if (DumpIns(node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Dump node:%s input info failed.",
                      node->GetNodeName().c_str());
      return INTERNAL_ERROR;
    }
    if (DumpOuts(node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Dump node:%s output info failed.",
                      node->GetNodeName().c_str());
      return INTERNAL_ERROR;
    }
    if (DumpAttr(node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][DumpInfo] Dump node:%s attr info failed.",
                      node->GetNodeName().c_str());
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}

Status FusionRulePatternConstructor::Construct(FusionRulePatternPtr pattern,
                                               FusionRuleJsonPatternPtr fusion_rule_json_pattern) {
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Input pattern is null."),
           return ILLEGAL_RULE);
  FE_CHECK(fusion_rule_json_pattern == nullptr,
           REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Input fusion_rule_json_pattern is null."),
           return ILLEGAL_RULE);
  FE_LOGD("Start loading fusion rule %s.", fusion_rule_json_pattern->GetName().c_str());
  pattern->rule_name_ = fusion_rule_json_pattern->GetName();

  Status ret = LoadInputInfo(pattern, fusion_rule_json_pattern->GetInputInfos());
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Load fusion rule:%s failed, load input info failed.",
                    pattern->rule_name_.c_str());
    return ret;
  }

  ret = LoadOutputInfo(pattern, fusion_rule_json_pattern->GetOutputInfos());
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Load fusion rule:%s failed, load output info failed.",
                    pattern->rule_name_.c_str());
    return ret;
  }

  ret = LoadGraph(pattern, pattern->origin_rule_nodes_, fusion_rule_json_pattern->GetOriginGraph());
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Load fusion rule:%s failed, load origin graph failed.",
                    pattern->rule_name_.c_str());
    return ret;
  }

  ret = LoadGraph(pattern, pattern->fusion_rule_nodes_, fusion_rule_json_pattern->GetFusionGraph());
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Load fusion rule:%s failed, load fusion graph failed.",
                    pattern->rule_name_.c_str());
    return ret;
  }

  ret = CheckFusionRulePattern(pattern);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Check fusion rule pattern validity failed.");
    return ret;
  }

  if (DumpInfo(pattern) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Dump fusion rule pattern:%s failed",
                    pattern->rule_name_.c_str());
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status FusionRulePatternConstructor::LoadInputInfo(FusionRulePatternPtr pattern,
                                                   const vector<FusionRuleJsonOuterPtr> &outer_inputs) {
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdInputInfo] Input pattern is null."),
           return ILLEGAL_RULE);

  map<string, FusionRuleNodePtr> input_nodes_map;
  map<string, FusionRuleAnchorPtr> input_anchors_map;

  for (const auto &iter : outer_inputs) {
    string input_name = iter->GetName();
    // create or find exist input node to hold outer input anchor
    FusionRuleNodePtr input_node = nullptr;
    Status ret = CreateOuterInputNode(input_nodes_map, iter, input_node);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdInputInfo] Create outer input node for %s failed.",
                      input_name.c_str());
      return ret;
    }
    // then create outer input anchor
    if (input_anchors_map.find(input_name) != input_anchors_map.end()) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdInputInfo] Anchor:%s already define, redeclaration.",
                      input_name.c_str());
      return ILLEGAL_RULE;
    }

    int anchor_index;
    if (iter->HasSrc()) {
      anchor_index = iter->GetSrcIndex();
    } else {
      // if not define src in outer inputs, means it's a unspecified input
      // anchor
      anchor_index = DEFAULT_ANCHOR_INDEX;
    }

    FusionRuleAnchorPtr anchor = nullptr;
    FE_MAKE_SHARED(anchor = make_shared<FusionRuleAnchor>(), return INTERNAL_ERROR);
    ret = FusionRuleAnchorConstructor::Construct(anchor, anchor_index, input_name);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdInputInfo] Create outer input anchor:%s failed.",
                      input_name.c_str());
      return ret;
    }
    input_anchors_map.emplace(make_pair(input_name, anchor));
    // add link between node and anchor
    ret = FusionRuleNodeConstructor::AddOutputAnchor(input_node, anchor);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdInputInfo] Add output anchor:%d to node:%s failed",
                      anchor->GetAnchorIdx(), input_node->GetNodeName().c_str());
      return ret;
    }
  }

  for (const auto &iter : input_nodes_map) {
    pattern->input_info_.push_back(iter.second);
  }

  return SUCCESS;
}

Status FusionRulePatternConstructor::LoadOutputInfo(FusionRulePatternPtr pattern,
                                                    const vector<FusionRuleJsonOuterPtr> &outer_outputs) {
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdOutInfo] Input pattern is null."),
           return ILLEGAL_RULE);

  map<string, FusionRuleNodePtr> output_nodes_map;

  for (const auto &iter : outer_outputs) {
    string output_name = iter->GetName();
    // make a dummy node to hold outer output anchor
    if (output_nodes_map.find(output_name) != output_nodes_map.end()) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdOutInfo] Output node:%s already define, redeclaration.",
                      output_name.c_str());
      return ILLEGAL_RULE;
    }

    FusionRuleNodePtr output_node = nullptr;
    FE_MAKE_SHARED(output_node = make_shared<FusionRuleNode>(), return INTERNAL_ERROR);
    Status ret = FusionRuleNodeConstructor::Construct(output_node, output_name, {});
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdOutInfo] Create dummy outer output node:%s failed.",
                      output_name.c_str());
      return ret;
    }
    output_nodes_map.emplace(make_pair(output_name, output_node));

    // outer outputs don't have "src" attr, so it's a unspecified output anchor
    int anchor_index = DEFAULT_ANCHOR_INDEX;
    FusionRuleAnchorPtr anchor = nullptr;
    FE_MAKE_SHARED(anchor = make_shared<FusionRuleAnchor>(), return INTERNAL_ERROR);
    ret = FusionRuleAnchorConstructor::Construct(anchor, anchor_index, output_name);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdOutInfo] Create outer output anchor:%s failed.",
                      output_name.c_str());
      return ret;
    }
    // add link between node and anchor
    ret = FusionRuleNodeConstructor::AddInputAnchor(output_node, anchor);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdOutInfo] Add input anchor:%d to node:%s failed",
                      anchor->GetAnchorIdx(), output_node->GetNodeName().c_str());
      return ret;
    }
  }

  for (const auto &iter : output_nodes_map) {
    pattern->output_info_.push_back(iter.second);
  }

  return SUCCESS;
}

Status FusionRulePatternConstructor::LoadGraph(FusionRulePatternPtr pattern, set<FusionRuleNodePtr> &rule_nodes,
                                               FusionRuleJsonGraphPtr json_graph) {
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdGph] Input pattern is null."),
           return ILLEGAL_RULE);
  if (rule_nodes.size() != 0) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdGph] ruleNodes in pattern already has element, load failed.");
    return ILLEGAL_RULE;
  }
  FE_CHECK(json_graph == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdGph] Input json_graph is null."),
           return ILLEGAL_RULE);

  // firstly, create fusion graph's nodes, and add it to pattern's rule_nodes
  map<string, FusionRuleNodePtr> nodes;
  Status ret = LoadNodes(pattern, json_graph, rule_nodes, nodes);
  if (ret != SUCCESS) {
    return ret;
  }

  // then add anchor to fusion graph node, add link between nodes according to
  // edges info
  for (const auto &edge : json_graph->GetEdges()) {
    FusionRuleAnchorPtr src_anchor = CreateInnerOutAnchor(edge->GetSrc(), nodes, pattern->GetInputInfo());
    FE_CHECK(src_anchor == nullptr,
             REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdGph] Failed to create src anchor and load edge."),
             return ILLEGAL_RULE);

    FusionRuleAnchorPtr dst_anchor = CreateInnerInAnchor(edge->GetDst(), nodes, pattern->GetOutputInfo());
    FE_CHECK(dst_anchor == nullptr,
             REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdGph] Failed to create dst anchor and load edge."),
             return ILLEGAL_RULE);

    Status in_ret = FusionRuleAnchorConstructor::AddEdge(src_anchor, dst_anchor, rule_nodes);
    if (in_ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdGph] Add link between node:%s[%d] to node:%s[%d] failed.",
                      src_anchor->GetOwnerNode()->GetNodeName().c_str(), src_anchor->GetAnchorIdx(),
                      dst_anchor->GetOwnerNode()->GetNodeName().c_str(), dst_anchor->GetAnchorIdx());
      return in_ret;
    }
  }
  // at last, if graph has attr assginment expression, add it
  ret = AddAttrAssignmentExpression(nodes, json_graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdGph] Add attr to related node failed.");
    return ret;
  }

  return SUCCESS;
}

Status FusionRulePatternConstructor::CheckFusionRulePattern(FusionRulePatternPtr pattern) {
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Input pattern is null."),
           return ILLEGAL_RULE);
  FE_LOGD("Start checking fusion rule %s.", pattern->GetRuleName().c_str());
  Status ret = ILLEGAL_RULE;
  // check outer input nodes's validity
  for (const auto &node : pattern->input_info_) {
    ret = FusionRuleNodeConstructor::CheckNodeValidity(node);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Check outer input node's validity failed.");
      return ret;
    }
  }
  // check outer output nodes's validity
  set<FusionRuleAnchorPtr> record_map;
  for (const auto &node : pattern->output_info_) {
    ret = FusionRuleNodeConstructor::CheckNodeValidity(node);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Check outer output node's validity failed.");
      return ret;
    }
    ret = FusionRuleNodeConstructor::CheckOuterOutputValidity(node, pattern);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Check outer output node:%s's topo validity failed.",
                      node->GetNodeName().c_str());
      return ret;
    }
    ret = FusionRuleNodeConstructor::CheckOuterOutputUniqueInput(node, record_map);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Check outer output node:%s's unique input failed.",
                      node->GetNodeName().c_str());
      return ret;
    }
  }
  // check origin graph nodes's validity
  for (const auto &node : pattern->origin_rule_nodes_) {
    ret = FusionRuleNodeConstructor::CheckNodeValidity(node);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Check origin graph node's validity failed.");
      return ret;
    }
  }
  // check fusion graph nodes's validity and set attr value's owner node
  for (const auto &node : pattern->fusion_rule_nodes_) {
    ret = FusionRuleNodeConstructor::CheckNodeValidity(node);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Check fusion graph node's validity failed.");
      return ret;
    }

    ret = FusionRuleNodeConstructor::SetAttrOwnerNode(node, pattern->GetOriginRuleNodes());
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Add node:%s's attrs owner node failed.",
                      node->GetNodeName().c_str());
      return ret;
    }
  }

  // check fusion graph's topological structure
  ret = TopoligicalSorting(pattern->input_info_, pattern->output_info_, pattern->fusion_rule_nodes_, true);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkFusRulePtn] Doing fusion graph topological sorting failed.");
    return ret;
  }

  return SUCCESS;
}

Status FusionRulePatternConstructor::TopoligicalSorting(const vector<FusionRuleNodePtr> &input_info,
                                                        const vector<FusionRuleNodePtr> &output_info,
                                                        const set<FusionRuleNodePtr> &inner_nodes,
                                                        bool is_fusion_graph) {
  vector<FusionRuleNodePtr> pre_nodes;
  Status ret = RecordAllNode(input_info, inner_nodes, pre_nodes);

  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ToplcSort] Record nodes to pre nodes failed.");
    return ret;
  }
  if (pre_nodes.size() == 0) {
    FE_LOGW("Nodes to be sort is null.");
    return SUCCESS;
  }

  set<FusionRuleNodePtr> buffer;
  if (InitBuffer(inner_nodes, output_info, is_fusion_graph, buffer)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ToplcSort] Init buffer failed.");
    return ILLEGAL_RULE;
  }

  map<FusionRuleNodePtr, map<int, int>> mark_map;
  ret = SetMarkMap(pre_nodes, mark_map);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ToplcSort] Init mark map for topological sorting failed.");
    return ret;
  }

  vector<FusionRuleNodePtr> sorted_nodes;
  FusionRuleNodePtr current_node = pre_nodes[0];

  while (true) {
    bool is_jump = false;  // already choose next search node, go to next loop
    ret = CheckCurrentNode(current_node, mark_map, buffer, is_jump);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ToplcSort] Run check current node failed.");
      return ret;
    }
    // If is_jump == true, means next search node already be chosen,
    // go to next loop
    if (is_jump) {
      continue;
    }
    // flag of already select an current node's output node as next search node
    bool selected_flag = false;
    ProcessReadyNode(inner_nodes, current_node, sorted_nodes, mark_map, buffer, selected_flag);

    if (!selected_flag) {
      if (buffer.size() == 0) {
        //  both buffer and next search node is null, means sorting done
        if (inner_nodes.size() != sorted_nodes.size()) {
          REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ToplcSort] May exist isolated subgraph in graph");
          return ILLEGAL_RULE;
        }
        break;
      }
      current_node = *(buffer.begin());
    }
  }

  return SUCCESS;
}

}  // namespace fe
