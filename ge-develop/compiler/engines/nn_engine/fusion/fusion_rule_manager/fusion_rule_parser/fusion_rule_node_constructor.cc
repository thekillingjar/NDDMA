/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_node_constructor.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"

using std::string;

namespace fe {

namespace {
// 2 means that the anchor size is 2
const size_t ANCHOR_SIZE_TWO = 2;
// 1 means that the anchor size is 1
const size_t ANCHOR_SIZE_ONE = 1;
// -1 means that this anchor index is -1, which is not supported
const int ANCHOR_INDEX_MINUS_ONE = -1;
// -2 means that this anchor index is default
const int ANCHOR_INDEX_DEFAULT = -2;
}  // namespace

Status FusionRuleNodeConstructor::Construct(FusionRuleNodePtr node, const string &name,
                                            const std::vector<string> &types) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Input node is null."),
           return ILLEGAL_RULE);

  node->node_name_ = name;
  node->node_type_ = types;

  return SUCCESS;
}

Status FusionRuleNodeConstructor::AddInputAnchor(FusionRuleNodePtr node, FusionRuleAnchorPtr anchor) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddInAnr] Input node is null."),
           return ILLEGAL_RULE);
  FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddInAnr] Input anchor is null."),
           return ILLEGAL_RULE);

  if (anchor->GetAnchorIdx() == ANCHOR_INDEX_MINUS_ONE) {
    node->input_ctrl_anchor_ = anchor;
  } else {
    bool inserted = false;
    for (auto iter = node->input_data_anchors_.begin(); iter < node->input_data_anchors_.end(); ++iter) {
      if ((*iter)->GetAnchorIdx() == anchor->GetAnchorIdx()) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][AddInAnr] Node:%s already have input anchor:%d, add input anchor failed.",
            node->node_name_.c_str(), (*iter)->GetAnchorIdx());
        return ILLEGAL_RULE;
      }
      // insert anchor by increasing order
      if ((*iter)->GetAnchorIdx() > anchor->GetAnchorIdx()) {
        node->input_data_anchors_.insert(iter, anchor);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      node->input_data_anchors_.insert(node->input_data_anchors_.cend(), anchor);
    }
  }

  Status ret = FusionRuleAnchorConstructor::AddOwnerNode(anchor, node);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddInAnr] Add owner node of node:%s to anchor:%d failed",
                    node->GetNodeName().c_str(), anchor->GetAnchorIdx());
    return ret;
  }
  return SUCCESS;
}

Status FusionRuleNodeConstructor::AddOutputAnchor(FusionRuleNodePtr node, FusionRuleAnchorPtr anchor) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddOutAnr] Output node is null."),
           return ILLEGAL_RULE);
  FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddOutAnr] Output anchor is null."),
           return ILLEGAL_RULE);
  if (anchor->GetAnchorIdx() == ANCHOR_INDEX_MINUS_ONE) {
    node->output_ctrl_anchor_ = anchor;
  } else {
    bool inserted = false;
    for (auto iter = node->output_data_anchors_.begin(); iter < node->output_data_anchors_.end(); ++iter) {
      if ((*iter)->GetAnchorIdx() == anchor->GetAnchorIdx()) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][AddOutAnr] Node:%s already have output anchor:%d, add output anchor failed.",
            node->node_name_.c_str(), (*iter)->GetAnchorIdx());
        return ILLEGAL_RULE;
      }
      // insert anchor by increasing order
      if ((*iter)->GetAnchorIdx() > anchor->GetAnchorIdx()) {
        node->output_data_anchors_.insert(iter, anchor);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      node->output_data_anchors_.insert(node->output_data_anchors_.cend(), anchor);
    }
  }

  Status ret = FusionRuleAnchorConstructor::AddOwnerNode(anchor, node);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddOutAnr] Add owner node of node:%s to anchor:%d failed",
                    node->GetNodeName().c_str(), anchor->GetAnchorIdx());
    return ret;
  }
  return SUCCESS;
}

Status FusionRuleNodeConstructor::AddAttr(FusionRuleNodePtr node, const string &attr_name,
                                          FusionRuleAttrValuePtr attr_value) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddAttr] Input node is null."),
           return ILLEGAL_RULE);
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddAttr] Input attr_value is null."),
           return ILLEGAL_RULE);

  if (node->attributes_.find(attr_name) != node->attributes_.end()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddAttr] Node:%s already have attr:%s, redeclaration.",
                    node->GetNodeName().c_str(), attr_name.c_str());
    return ILLEGAL_RULE;
  }
  node->attributes_.emplace(make_pair(attr_name, attr_value));
  return SUCCESS;
}

Status FusionRuleNodeConstructor::CheckNodeValidity(FusionRuleNodePtr node) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkNdValid] Input node is null."),
           return ILLEGAL_RULE);

  // check node's intput anchor or output anchor is empty,
  // otherwise it's a isolated node
  if (node->input_data_anchors_.empty() && node->output_data_anchors_.empty() && node->input_ctrl_anchor_ == nullptr &&
      node->output_ctrl_anchor_ == nullptr) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][ChkNdValid] Node[%s] has no input or output anchor, it's an isolated node",
        node->GetNodeName().c_str());
    return ILLEGAL_RULE;
  }
  // doing continuity check for rule graph's inner nodes
  if (!node->node_type_.empty()) {
    // check input anchor index continuity
    for (size_t i = 0; i < node->input_data_anchors_.size(); ++i) {
      if (static_cast<int>(i) != node->input_data_anchors_[i]->GetAnchorIdx()) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][ChkNdValid] Node[%s]'s input anchor[%zu] not equal to it's index[%d], input"
            "anchor not continuous.",
            node->GetNodeName().c_str(), i, node->input_data_anchors_[i]->GetAnchorIdx());
        return ILLEGAL_RULE;
      }
    }
    // check output anchor index continuity
    for (size_t i = 0; i < node->output_data_anchors_.size(); ++i) {
      if (static_cast<int>(i) != node->output_data_anchors_[i]->GetAnchorIdx()) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][ChkNdValid] Node:%s's output anchor[%zu] not equal to it's index[%d], "
            "output anchor not continuous.",
            node->GetNodeName().c_str(), i, node->output_data_anchors_[i]->GetAnchorIdx());
        return ILLEGAL_RULE;
      }
    }
  }

  return SUCCESS;
}

Status FusionRuleNodeConstructor::CheckOuterOutputValidity(FusionRuleNodePtr node, FusionRulePatternPtr pattern) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutValid] Input node is null."),
           return ILLEGAL_RULE);
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutValid] Input pattern is null."),
           return ILLEGAL_RULE);
  // Check outer output node must have two input, one from origin graph, another
  // from fusion graph
  if (node->GetInputDataAnchors().size() != ANCHOR_SIZE_ONE) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][ChkOutOutValid] Outer output node must have 1 input anchor, actual is %zu",
        node->GetInputDataAnchors().size());
    return ILLEGAL_RULE;
  }

  if (node->input_data_anchors_.empty()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutValid] Input anchor is null.");
    return ILLEGAL_RULE;
  }
  FusionRuleAnchorPtr input_anchor = node->input_data_anchors_[0];
  FE_CHECK(input_anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutValid] Input anchor is null."),
           return ILLEGAL_RULE);
  if (input_anchor->GetPeerAnchors().size() != ANCHOR_SIZE_TWO) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][ChkOutOutValid] Outer output node must have 2 input peer anchors, actually is %zu",
        input_anchor->GetPeerAnchors().size());
    return ILLEGAL_RULE;
  }

  bool found_in_origin_graph = false;
  bool found_in_fusion_graph = false;
  for (const auto &peer_anchor : input_anchor->GetPeerAnchors()) {
    FE_CHECK(peer_anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutValid] Peer anchor is null."),
             return ILLEGAL_RULE);
    FusionRuleNodePtr peer_node = peer_anchor->GetOwnerNode();
    FE_CHECK(peer_node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutValid] Peer node is null."),
             return ILLEGAL_RULE);

    if (pattern->GetOriginRuleNodes().find(peer_node) != pattern->GetOriginRuleNodes().end()) {
      found_in_origin_graph = true;
    } else if (pattern->GetFusionRuleNodes().find(peer_node) != pattern->GetFusionRuleNodes().end()) {
      found_in_fusion_graph = true;
    }
  }

  if (!(found_in_origin_graph && found_in_fusion_graph)) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][ChkOutOutValid] Outer output node must have one input from origin graph, another "
        "from fusion graph");
    return ILLEGAL_RULE;
  }

  return SUCCESS;
}

Status FusionRuleNodeConstructor::CheckOuterOutputUniqueInput(FusionRuleNodePtr node,
                                                              set<FusionRuleAnchorPtr> &record_map) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutUnqIn] Input node is null."),
           return ILLEGAL_RULE);
  for (const auto &input_anchor : node->input_data_anchors_) {
    FE_CHECK(input_anchor == nullptr,
             REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutUnqIn] Input anchor is null."),
             return ILLEGAL_RULE);
    for (const auto &peer_output_anchor : input_anchor->GetPeerAnchors()) {
      FE_CHECK(peer_output_anchor == nullptr,
               REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkOutOutUnqIn] Peer anchor is null."),
               return ILLEGAL_RULE);

      if (record_map.find(peer_output_anchor) != record_map.end()) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][ChkOutOutUnqIn] Node:%s already have a link, link to outer output:%s failed.",
            peer_output_anchor->GetOwnerNode()->GetNodeName().c_str(), node->GetNodeName().c_str());
        return ILLEGAL_RULE;
      }
      record_map.emplace(peer_output_anchor);
    }
  }

  return SUCCESS;
}

Status FusionRuleNodeConstructor::SetAttrOwnerNode(FusionRuleNodePtr node, const set<FusionRuleNodePtr> &nodes) {
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][SetAttrOwnNd] Input node is null."),
           return ILLEGAL_RULE);

  for (const auto &attr : node->attributes_) {
    if (attr.second->IsFusionRuleAttr()) {
      FusionRuleAttr attr_ref = attr.second->GetRuleNodeAttrValue();
      bool found_flag = false;
      for (const auto &search_node : nodes) {
        if (search_node->GetNodeName() == attr_ref.node_name) {
          Status ret = attr.second->SetOwnerNode(search_node);
          if (ret != SUCCESS) {
            REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][SetAttrOwnNd] Set value to attr:%s of node:%s failed.",
                            attr.first.c_str(), node->GetNodeName().c_str());
            return ret;
          }
          found_flag = true;
          break;
        }
      }

      if (!found_flag) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][SetAttrOwnNd] Can't find node:%s's attr:%s reflected value of %s.%s "
            "from origin graph.",
            node->GetNodeName().c_str(), attr.first.c_str(), attr_ref.node_name.c_str(), attr_ref.attr_name.c_str());
        return ILLEGAL_RULE;
      }
    }
  }
  return SUCCESS;
}

Status FusionRuleAnchorConstructor::Construct(FusionRuleAnchorPtr anchor, int index, const string &name) {
  FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Input anchor is null."),
           return ILLEGAL_RULE);

  if (index < ANCHOR_INDEX_DEFAULT) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][Construct] Only support data anchor (index >= -2), %d is not supported",
                    index);
    return ILLEGAL_RULE;
  }
  anchor->anchor_name_ = name;
  anchor->anchor_idx_ = index;

  return SUCCESS;
}

Status FusionRuleAnchorConstructor::AddOwnerNode(FusionRuleAnchorPtr anchor, FusionRuleNodePtr owner_node) {
  FE_CHECK(anchor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddOwnNd] Input anchor is null."),
           return ILLEGAL_RULE);
  FE_CHECK(owner_node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddOwnNd] Input owner_node is null."),
           return ILLEGAL_RULE);

  if (anchor->owner_node_.lock() != nullptr) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][AddOwnNd] Add owner node:%s to anchor failed, anchor already have "
        "owner node:%s",
        owner_node->GetNodeName().c_str(), anchor->owner_node_.lock()->GetNodeName().c_str());
    return ILLEGAL_RULE;
  }
  anchor->owner_node_ = owner_node;

  return SUCCESS;
}

Status FusionRuleAnchorConstructor::AddEdge(FusionRuleAnchorPtr src, FusionRuleAnchorPtr dst,
                                            const set<FusionRuleNodePtr> &rule_nodes) {
  FE_CHECK(src == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddEdge] Input src is null."),
           return ILLEGAL_RULE);
  FE_CHECK(dst == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddEdge] Input dst is null."),
           return ILLEGAL_RULE);
  // ctrl doesnt need to check peer anchors
  if (src->GetAnchorIdx() == ANCHOR_INDEX_MINUS_ONE && dst->GetAnchorIdx() == ANCHOR_INDEX_MINUS_ONE) {
    src->peer_anchors_.push_back(dst);
    dst->peer_anchors_.push_back(src);
    return SUCCESS;
  }

  src->peer_anchors_.push_back(dst);
  if (!dst->peer_anchors_.empty()) {
    FusionRuleNodePtr dst_node = dst->GetOwnerNode();
    FE_CHECK(dst_node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddEdge] Dst node is null."),
             return ILLEGAL_RULE);
    if (rule_nodes.find(dst_node) == rule_nodes.end()) {
      // Check input anchor can only have one peer output anchor
      // outputs' input anchor can have two: one from OriginGraph, another one from FusionGraph
      for (const auto &peer_anchor : dst->peer_anchors_) {
        FE_CHECK(peer_anchor.lock() == nullptr,
                 REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddEdge] Geted peer anchor is null."),
                 return ILLEGAL_RULE);
        auto node = peer_anchor.lock()->GetOwnerNode();

        FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddEdge] Geted peer node is null."),
                 return ILLEGAL_RULE);
        if (rule_nodes.find(node) != rule_nodes.end()) {
          REPORT_FE_ERROR(
              "[GraphOpt][FusionRuleInit][AddEdge] Failed to add peer anchor of node %s %d.",
              dst->GetOwnerNode()->GetNodeName().c_str(), dst->GetAnchorIdx());
          return ILLEGAL_RULE;
        }
      }
      dst->peer_anchors_.push_back(src);
    } else {
      // otherwise, inner node's input can only have one peer anchor
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AddEdge] Node:%s input:%d already have peer_anchor, duplicated link.",
                      dst_node->GetNodeName().c_str(), dst->GetAnchorIdx());
      return ILLEGAL_RULE;
    }
  } else {
    dst->peer_anchors_.push_back(src);
  }

  return SUCCESS;
}

}  // namespace fe
