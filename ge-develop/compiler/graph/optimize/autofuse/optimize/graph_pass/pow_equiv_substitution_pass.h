/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_GRAPH_PASS_POW_EQUIV_SUBSTITUTION_PASS_H
#define OPTIMIZE_PLATFORM_COMMON_GRAPH_PASS_POW_EQUIV_SUBSTITUTION_PASS_H

#include "optimize/graph_pass/base_graph_pass.h"
namespace optimize {
using SubstitutionFunc = std::function<ge::Status(ge::AscGraph &, const ge::AscNodePtr &)>;

enum class PatternType : uint8_t {
  kZero = 0U,  // 0.0 (pow 0 = Brc one)
  kHalf = 1U,       // 0.5 (pow 0.5 = sqrt)
  kNegHalf = 2U,    // -0.5 (pow -0.5 = 1/sqrt)
  kOne = 3U,        // 1.0 (pow 1 just remove node)
  kNegOne = 4U,     // -1.0 (pow -1 = reciprocal)
  kTwo = 5U,        // 2.0 (pow 2 = mul)
  kNegTwo = 6U,     // -2.0 (pow -2 = 1/(x*x))
  kThree = 7U,      // 3.0 (pow 3 = mul(x,x)*x)
  kFour = 8U,       // 4.0 (pow 4 = mul(x*x,x*x))
  kNone = 100U
};

class PowEquivSubstitutionPass final : public BaseGraphPass {
 public:
  PowEquivSubstitutionPass() = default;
  ~PowEquivSubstitutionPass() override = default;
  ge::Status RunPass(ge::AscGraph &graph) override;

 private:
  static std::vector<ge::AscNodePtr> FilterPowNodes(ge::AscGraph &graph);

  static const std::unordered_map<PatternType, SubstitutionFunc> &GetGlobalSubstitutionMap();

  static bool GetScalarInput(const ge::AscNodePtr &pow_node, std::string &scalar_val);

  // 检查 Pow 的 input0 是否是 Scalar，如果是则插入 Brc
  static ge::Status EnsureInputWithBrcIfNeeded(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status RelinkPowInputToNode(const ge::AscNodePtr &pow_node, const ge::AscNodePtr &target_node,
                                         const int32_t in_idx = 0);

  static ge::Status RelinkPowOutputToNode(const ge::AscNodePtr &pow_node, const ge::AscNodePtr &target_node);

  static ge::Status ReplaceWithScalarBrc(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status ReplaceWithSqrt(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status ReplaceWithInverseSqrt(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status RemoveUseLessPow(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status ReplaceWithInverseInput(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status ReplaceWithMul(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status ReplaceWithInverseMul(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status ReplaceWithCube(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);

  static ge::Status ReplaceWithFourthPower(ge::AscGraph &graph, const ge::AscNodePtr &pow_node);
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_COMMON_GRAPH_PASS_POW_EQUIV_SUBSTITUTION_PASS_H
