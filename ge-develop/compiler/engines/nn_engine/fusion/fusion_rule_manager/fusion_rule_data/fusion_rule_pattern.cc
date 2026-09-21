/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h"
#include "common/fe_utils.h"

namespace fe {
using std::pair;

std::string GetRuleTypeString(RuleType rule_type) {
  auto iter = RULE_TYPE_STRING_MAP.find(rule_type);
  if (iter == RULE_TYPE_STRING_MAP.end()) {
    return "unknown-rule-type " + std::to_string(static_cast<int>(rule_type));
  } else {
    return iter->second;
  }
}

FusionRulePattern::FusionRulePattern()
    : rule_name_(), input_info_(), output_info_(), origin_rule_nodes_(), fusion_rule_nodes_() {}

FusionRulePattern::~FusionRulePattern() {}

string FusionRulePattern::GetRuleName() const { return rule_name_; }

const vector<FusionRuleNodePtr> &FusionRulePattern::GetInputInfo() const { return input_info_; }

const vector<FusionRuleNodePtr> &FusionRulePattern::GetOutputInfo() const { return output_info_; }

const set<FusionRuleNodePtr> &FusionRulePattern::GetOriginRuleNodes() const { return origin_rule_nodes_; }

const set<FusionRuleNodePtr> &FusionRulePattern::GetFusionRuleNodes() const { return fusion_rule_nodes_; }

FusionRuleNode::FusionRuleNode()
    : node_name_(),
      node_type_(),
      input_data_anchors_(),
      output_data_anchors_(),
      input_ctrl_anchor_(nullptr),
      output_ctrl_anchor_(nullptr),
      attributes_() {}

FusionRuleNode::~FusionRuleNode() {}

string FusionRuleNode::GetNodeName() const { return node_name_; }

const vector<string> &FusionRuleNode::GetNodeType() const { return node_type_; }

const vector<FusionRuleAnchorPtr> &FusionRuleNode::GetInputDataAnchors() const { return input_data_anchors_; }

const vector<FusionRuleAnchorPtr> &FusionRuleNode::GetOutputDataAnchors() const { return output_data_anchors_; }

const FusionRuleAnchorPtr &FusionRuleNode::GetInputCtrlAnchor() const { return input_ctrl_anchor_; }

const FusionRuleAnchorPtr &FusionRuleNode::GetOutputCtrlAnchor() const { return output_ctrl_anchor_; }

const map<string, FusionRuleAttrValuePtr> &FusionRuleNode::GetAttributes() const { return attributes_; }

FusionRuleAnchor::FusionRuleAnchor() : anchor_idx_(DEFAULT_ANCHOR_INDEX), anchor_name_() {}

FusionRuleAnchor::~FusionRuleAnchor() {}

int FusionRuleAnchor::GetAnchorIdx() const { return anchor_idx_; }

string FusionRuleAnchor::GetAnchorName() const { return anchor_name_; }

FusionRuleNodePtr FusionRuleAnchor::GetOwnerNode() const { return owner_node_.lock(); }

vector<FusionRuleAnchorPtr> FusionRuleAnchor::GetPeerAnchors() const {
  vector<FusionRuleAnchorPtr> vec;
  for (const auto &anchor : peer_anchors_) {
    vec.push_back(anchor.lock());
  }
  return vec;
}

FusionRuleAttrValue::FusionRuleAttrValue() : is_fusion_rule_attr_(false), rule_node_attr_() {}

FusionRuleAttrValue::~FusionRuleAttrValue() {}

bool FusionRuleAttrValue::IsFusionRuleAttr() const { return is_fusion_rule_attr_; }

FusionRuleAttr FusionRuleAttrValue::GetRuleNodeAttrValue() const { return rule_node_attr_; }

ge::GeAttrValue FusionRuleAttrValue::GetFixAttrValue() const { return fix_value_attr_; }

Status FusionRuleAttrValue::SetAttrValue(const FusionRuleAttr &rule_node_attr) {
  rule_node_attr_ = rule_node_attr;
  is_fusion_rule_attr_ = true;

  return fe::SUCCESS;
}

Status FusionRuleAttrValue::SetOwnerNode(const FusionRuleNodePtr node) {
  FE_CHECK_NOTNULL(node);
  owner_node_ = node;
  return SUCCESS;
}

FusionRuleNodePtr FusionRuleAttrValue::GetOwnerNode() { return owner_node_.lock(); }
}  // namespace fe
