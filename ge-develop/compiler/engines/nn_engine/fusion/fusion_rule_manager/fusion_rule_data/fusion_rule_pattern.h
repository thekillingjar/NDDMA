/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_DATA_FUSION_RULE_PATTERN_H_
#define FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_DATA_FUSION_RULE_PATTERN_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "graph/ge_attr_value.h"
#include "common/fe_log.h"
namespace fe {

using std::string;
using std::map;
using std::vector;
using std::set;

/*
 * types of rule config file
 * each type represent one kind of config file
 */
enum class RuleType {
  /* the rules of graph fusion */
  BUILT_IN_GRAPH_RULE = 0,
  /* the rules of custom graph fusion */
  CUSTOM_GRAPH_RULE,
  RULE_TYPE_RESERVED,
};

const std::map<RuleType, std::string> RULE_TYPE_STRING_MAP{
  {RuleType::BUILT_IN_GRAPH_RULE, "built-in-graph-rule"}, {RuleType::CUSTOM_GRAPH_RULE, "custom-graph-rule"},
};

std::string GetRuleTypeString(RuleType rule_type);

/* default value of anchor's index */
const int32_t DEFAULT_ANCHOR_INDEX = -2;

class FusionRulePattern;
class FusionRuleAnchor;
class FusionRuleNode;
class FusionRuleAttrValue;
class FusionRulePatternConstructor;

using FusionRulePatternPtr = std::shared_ptr<FusionRulePattern>;
using FusionRuleNodePtr = std::shared_ptr<FusionRuleNode>;
using FusionRuleAnchorPtr = std::shared_ptr<FusionRuleAnchor>;
using FusionRuleAttrValuePtr = std::shared_ptr<FusionRuleAttrValue>;

/*
 * the description of fusion rule, which is read from config file
 * one FusionRulePattern represents one fusion rule
 * only provide 'Get' functions
 */
/** @brief implement the fuctions of data structure of fusion rules which are
*        read from config file.
*        Provide methods of fusion rules for the match of input graph and
*        fusion rules. */
class FusionRulePattern {
 public:
  FusionRulePattern();

  ~FusionRulePattern();

  string GetRuleName() const;

  const vector<FusionRuleNodePtr> &GetInputInfo() const;

  const vector<FusionRuleNodePtr> &GetOutputInfo() const;

  const set<FusionRuleNodePtr> &GetOriginRuleNodes() const;

  const set<FusionRuleNodePtr> &GetFusionRuleNodes() const;

 private:
  /* friend class will set data to the private member variables */
  friend class FusionRulePatternConstructor;

  FusionRulePattern(const FusionRulePattern &) = delete;

  FusionRulePattern &operator=(const FusionRulePattern &) = delete;

  string rule_name_;
  /*
   * store the rule graph's inputs, the FusionRuleNodePtr is viturl
   * actually graph's inputs are anchors, not the real nodes
   */
  vector<FusionRuleNodePtr> input_info_;
  /*
   * store the rule graph's outputs, the FusionRuleNodePtr is viturl
   * actually graph's outputs are anchors, not the real nodes
   */
  vector<FusionRuleNodePtr> output_info_;
  /* store nodes info and relationship between nodes of the rule which is before
   * fusion */
  set<FusionRuleNodePtr> origin_rule_nodes_;
  /* store nodes info and relationship between nodes of the rule which is after
   * fusion */
  set<FusionRuleNodePtr> fusion_rule_nodes_;
};

/*
 * the description of node's info in the FusionRulePattern.
 * contains info of node's self and the relationship of this node's successive
 * and successor node
 * the relationship use anchor to describe
 * only provide 'Get' functions
 */
class FusionRuleNode {
 public:
  FusionRuleNode();

  ~FusionRuleNode();

  string GetNodeName() const;

  const vector<string> &GetNodeType() const;

  const vector<FusionRuleAnchorPtr> &GetInputDataAnchors() const;

  const vector<FusionRuleAnchorPtr> &GetOutputDataAnchors() const;

  const FusionRuleAnchorPtr &GetInputCtrlAnchor() const;

  const FusionRuleAnchorPtr &GetOutputCtrlAnchor() const;

  const map<string, FusionRuleAttrValuePtr> &GetAttributes() const;

 private:
  /* friend class will set data to the private member variables */
  friend class FusionRuleNodeConstructor;

  FusionRuleNode(const FusionRuleNode &) = delete;

  FusionRuleNode &operator=(const FusionRuleNode &) = delete;

  string node_name_;

  vector<string> node_type_;
  /* store the successive nodes' output anchor info */
  vector<FusionRuleAnchorPtr> input_data_anchors_;
  /* store the successor nodes' input anchor info */
  vector<FusionRuleAnchorPtr> output_data_anchors_;
  /* store the successive nodes' output control anchor info */
  FusionRuleAnchorPtr input_ctrl_anchor_;
  /* store the successor nodes' input control anchor info */
  FusionRuleAnchorPtr output_ctrl_anchor_;

  map<string, FusionRuleAttrValuePtr> attributes_;
};

/*
 * the description of anchor's info in the FusionRuleNode.
 * store the relationship of this node's successive and successor node
 * only provide 'Get' functions
 */
class FusionRuleAnchor {
 public:
  FusionRuleAnchor(const FusionRuleAnchor &) = delete;
  FusionRuleAnchor &operator=(const FusionRuleAnchor &) = delete;

  FusionRuleAnchor();

  ~FusionRuleAnchor();

  int GetAnchorIdx() const;

  string GetAnchorName() const;

  FusionRuleNodePtr GetOwnerNode() const;

  vector<FusionRuleAnchorPtr> GetPeerAnchors() const;

 private:
  /* friend class will set data to the private member variables */
  friend class FusionRuleAnchorConstructor;

  int anchor_idx_;
  /*
   * only for the rule graph's inputs and outputs
   * the anchors of the ops nodes don't have the name,default value is empty
   */
  string anchor_name_;
  /* record the anchor's own node */
  std::weak_ptr<FusionRuleNode> owner_node_;
  /*
   * the anchor may connect one or more anchors
   * for the input anchor there is only one peer anchor
   * for the output anchor there may be more than one peer anchor
   */
  vector<std::weak_ptr<FusionRuleAnchor>> peer_anchors_;
};

struct FusionRuleAttr {
  string node_name;
  string attr_name;
};

/*
 * the description of node's attribute
 */
class FusionRuleAttrValue {
 public:
  FusionRuleAttrValue(const FusionRuleAttrValue &) = delete;

  FusionRuleAttrValue &operator=(const FusionRuleAttrValue &) = delete;

  FusionRuleAttrValue();

  ~FusionRuleAttrValue();

  bool IsFusionRuleAttr() const;

  FusionRuleAttr GetRuleNodeAttrValue() const;

  ge::GeAttrValue GetFixAttrValue() const;

  Status SetAttrValue(const FusionRuleAttr &rule_node_attr);

  template <typename T>
  Status SetAttrValue(const T &value) {
    if (fix_value_attr_.SetValue<T>(value) != ge::GRAPH_SUCCESS) {
      FE_LOGE("Set value to FusionRuleAttr.fix_value_attr_ failed");
      return fe::FAILED;
    }

    is_fusion_rule_attr_ = false;

    return fe::SUCCESS;
  }

  Status SetOwnerNode(const FusionRuleNodePtr node);

  FusionRuleNodePtr GetOwnerNode();

 private:
  /*
   * true: represent the attr's value is from the real node,
   *       the value store in rule_node_attr_,then fix_value_attr_type_ and
   * fix_value_attr_ are empty
   * false:represent the attr's value is from the fix value which is filled by
   * user,
   *       the value store in fix_value_attr_type_ and fix_value_attr_ ,then
   * rule_node_attr_ is empty
   */
  bool is_fusion_rule_attr_;

  FusionRuleAttr rule_node_attr_;

  ge::GeAttrValue fix_value_attr_;

  std::weak_ptr<FusionRuleNode> owner_node_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_DATA_FUSION_RULE_PATTERN_H_
