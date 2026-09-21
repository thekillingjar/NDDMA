/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_PATTERN_CONSTRUCTOR_H_
#define FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_PATTERN_CONSTRUCTOR_H_

#include <set>
#include <vector>
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_node_constructor.h"

namespace fe {

/** @brief Provide constructor methods for FusionRulePattern
*        check & load fusion rules from FusionRuleJsonPattern object */
class FusionRulePatternConstructor {
 public:
  /*
   * @brief: Construct a FusionRulePattern object according to
   * FusionRuleJsonPattern info
   * @param: pattern: FusionRulePattern object to construct
   */
  static Status Construct(FusionRulePatternPtr pattern, FusionRuleJsonPatternPtr fusion_rule_json_pattern);

  /*
   * @brief: Construct OuterInput anchor and dummy node to hold it
   */
  static Status LoadInputInfo(FusionRulePatternPtr pattern, const std::vector<FusionRuleJsonOuterPtr> &outer_inputs);

  /*
   * @brief: Construct OuterOuter anchor and dummy node to hold it
   */
  static Status LoadOutputInfo(FusionRulePatternPtr pattern, const std::vector<FusionRuleJsonOuterPtr> &outer_outputs);

  /*
   * @brief: Construct all nodes in json graph, and add edges between nodes, if
   * there is "Attrs" info,
   *         add it to related node
   */
  static Status LoadGraph(FusionRulePatternPtr pattern, set<FusionRuleNodePtr> &rule_nodes,
                          FusionRuleJsonGraphPtr json_graph);

  static Status CheckFusionRulePattern(FusionRulePatternPtr pattern);
  /*
   * @brief: Custom topological sorting algorithm, check loop and isolated
   * subgraph in input graph
   */
  static Status TopoligicalSorting(const std::vector<FusionRuleNodePtr> &input_info,
                                   const std::vector<FusionRuleNodePtr> &output_info,
                                   const std::set<FusionRuleNodePtr> &inner_nodes, bool is_fusion_graph);

  static Status DumpInfo(FusionRulePatternPtr pattern);
};
}  // namespace fe

#endif  // FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_PATTERN_CONSTRUCTOR_H_
