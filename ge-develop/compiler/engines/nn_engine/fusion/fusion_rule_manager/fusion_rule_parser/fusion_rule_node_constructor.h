/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_NODE_CONSTRUCTOR_H_
#define FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_NODE_CONSTRUCTOR_H_

#include <set>
#include <string>
#include <vector>
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_json_pattern.h"

namespace fe {

class FusionRuleNodeConstructor {
 public:
  /*
   * @brief: Construct a FusionRuleNode object according to name and types info
   * @param: node: FusionRuleNode object to construct
   */
  static Status Construct(FusionRuleNodePtr node, const std::string &name, const std::vector<string> &types);

  static Status AddInputAnchor(FusionRuleNodePtr node, FusionRuleAnchorPtr anchor);

  static Status AddOutputAnchor(FusionRuleNodePtr node, FusionRuleAnchorPtr anchor);

  static Status AddAttr(FusionRuleNodePtr node, const std::string &attr_name, FusionRuleAttrValuePtr attr_value);

  /*
   * @brief: Doing two validity check: 1. isolated node check; 2. in/out anchor
   *         index's continuity check
   */
  static Status CheckNodeValidity(FusionRuleNodePtr node);

  /*
   * @brief: Check outer output node must have two input, one from origin graph,
   *         another from fusion graph
   */
  static Status CheckOuterOutputValidity(FusionRuleNodePtr node, FusionRulePatternPtr pattern);

  /*
   * @brief: Check each outer output should come from unique inner output,
   *         different outer output cannot have same inner node as input
   */
  static Status CheckOuterOutputUniqueInput(FusionRuleNodePtr node, set<FusionRuleAnchorPtr> &record_map);
  /*
   * @brief: Record relected assgined attr's source node ptr
   */
  static Status SetAttrOwnerNode(FusionRuleNodePtr node, const std::set<FusionRuleNodePtr> &nodes);
};

class FusionRuleAnchorConstructor {
 public:
  /*
   * @brief: Construct a FusionRuleAnchor object according to name and index
   *         info
   * @param: anchor: FusionRuleAnchor object to construct
   */
  static Status Construct(FusionRuleAnchorPtr anchor, int index, const std::string &name);

  static Status AddOwnerNode(FusionRuleAnchorPtr anchor, FusionRuleNodePtr owner_node);

  static Status AddEdge(FusionRuleAnchorPtr src, FusionRuleAnchorPtr dst, const set<FusionRuleNodePtr> &rule_nodes);
};

}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_NODE_CONSTRUCTOR_H_
