/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pow_equiv_substitution_pass.h"
#include <regex>
#include "ascir_ops_utils.h"
#include "ascir_ops.h"
#include "graph_utils.h"
#include "schedule_utils.h"
#include "pass_utils.h"

namespace {
const double kRelEpsilonFloat = 1e-7;
const double kRelEpsilonInt = 1e-20;
constexpr uint64_t kScalarZeroMask = 0x7FFFFFFFFFFFFFFFUL;

const double kScalarHalf = 0.5;
const double kScalarNegHalf = -0.5;
const double kScalarOne = 1.0;
const double kScalarNegOne = -1.0;
const double kScalarTwo = 2.0;
const double kScalarNegTwo = -2.0;
const double kScalarThree = 3.0;
const double kScalarFour = 4.0;

using optimize::PatternType;
struct ScalarTargetMap {
  double target_val;
  double epsilon;
  PatternType type;
};

const std::vector<ScalarTargetMap> kScalarTargetTable = {
    {kScalarHalf, kRelEpsilonFloat, PatternType::kHalf}, {kScalarNegHalf, kRelEpsilonFloat, PatternType::kNegHalf},
    {kScalarOne, kRelEpsilonInt, PatternType::kOne},     {kScalarNegOne, kRelEpsilonInt, PatternType::kNegOne},
    {kScalarTwo, kRelEpsilonInt, PatternType::kTwo},     {kScalarNegTwo, kRelEpsilonInt, PatternType::kNegTwo},
    {kScalarThree, kRelEpsilonInt, PatternType::kThree}, {kScalarFour, kRelEpsilonInt, PatternType::kFour}};

union DoubleBits {
  double d;
  uint64_t u;
};

PatternType CheckStringValue(const std::string &s) {
  DoubleBits val{0};
  std::istringstream iss(s);
  if (!(iss >> val.d)) {
    return PatternType::kNone;
  }

  if ((val.u & kScalarZeroMask) == 0UL) {
    return PatternType::kZero;
  }

  for (const auto &entry : kScalarTargetTable) {
    if (std::fabs(val.d - entry.target_val) < entry.epsilon) {
      return entry.type;
    }
  }
  return PatternType::kNone;
}

ge::AscNodePtr CreateMulNodeWithAttr(ge::AscGraph &graph, const ge::AscNodePtr &pow_node, const std::string &mul_suffix) {
    std::string mul_name = pow_node->GetName() + mul_suffix;
    ge::ascir_op::Mul mul(mul_name.c_str());
    auto mul_node = graph.AddNode(mul);
    GE_ASSERT_NOTNULL(mul_node);

    mul_node->attr.sched = pow_node->attr.sched;
    mul_node->outputs[0].attr = pow_node->outputs[0].attr;

    GELOGD("Created Mul node [%s] for Pow node [%s]", mul_node->GetNamePtr(), pow_node->GetNamePtr());
    return mul_node;
}
}  // namespace

namespace optimize {
using ge::ops::IsOps;

const std::unordered_map<PatternType, SubstitutionFunc> &PowEquivSubstitutionPass::GetGlobalSubstitutionMap() {
  static const std::unordered_map<PatternType, SubstitutionFunc> kGlobalSubstitutionMap = {
      {PatternType::kZero, &PowEquivSubstitutionPass::ReplaceWithScalarBrc},
      {PatternType::kHalf, &PowEquivSubstitutionPass::ReplaceWithSqrt},
      {PatternType::kNegHalf, &PowEquivSubstitutionPass::ReplaceWithInverseSqrt},
      {PatternType::kOne, &PowEquivSubstitutionPass::RemoveUseLessPow},
      {PatternType::kNegOne, &PowEquivSubstitutionPass::ReplaceWithInverseInput},
      {PatternType::kTwo, &PowEquivSubstitutionPass::ReplaceWithMul},
      {PatternType::kNegTwo, &PowEquivSubstitutionPass::ReplaceWithInverseMul},
      {PatternType::kThree, &PowEquivSubstitutionPass::ReplaceWithCube},
      {PatternType::kFour, &PowEquivSubstitutionPass::ReplaceWithFourthPower}};
  return kGlobalSubstitutionMap;
}

Status PowEquivSubstitutionPass::RunPass(ge::AscGraph &graph) {
  std::vector<ge::AscNodePtr> pow_nodes = FilterPowNodes(graph);
  if (pow_nodes.empty()) {
    GELOGD("No Pow nodes found in graph, skip substitution");
    return ge::SUCCESS;
  }

  bool changed = false;
  const std::unordered_map<PatternType, SubstitutionFunc> &pattern_to_substitution = GetGlobalSubstitutionMap();
  for (const auto &pow_node : pow_nodes) {
    std::string scalar_val;
    if (!GetScalarInput(pow_node, scalar_val)) {
      continue;
    }

    const PatternType type = CheckStringValue(scalar_val);
    if (type == PatternType::kNone) {
      continue;
    }

    const auto it = pattern_to_substitution.find(type);
    if (it != pattern_to_substitution.end()) {
      GELOGD("Pow [%s] has scalar inputs [%s], matched pattern %d.", pow_node->GetNamePtr(), scalar_val.c_str(),
             static_cast<int32_t>(type));
      changed = true;
      // 对每个 Pow 节点，如果 input0 是 Scalar，先插入 Brc
      GE_CHK_STATUS_RET(EnsureInputWithBrcIfNeeded(graph, pow_node), "Failed to ensure input0 with Brc for Pow [%s]",
                        pow_node->GetName().c_str());
      GE_CHK_STATUS_RET(it->second(graph, pow_node), "Failed to substitute Pow node [%s] for pattern [%d]",
                        pow_node->GetName().c_str(), static_cast<int32_t>(type));
    }
  }
  if (changed) {
    GE_ASSERT_SUCCESS(PassUtils::PruneGraph(graph));
    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return ge::SUCCESS;
}

std::vector<ge::AscNodePtr> PowEquivSubstitutionPass::FilterPowNodes(ge::AscGraph &graph) {
  std::vector<ge::AscNodePtr> pow_nodes;
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    GE_ASSERT_NOTNULL(node);
    if (IsOps<ge::ascir_op::Pow>(node)) {
      pow_nodes.push_back(node);
    }
  }
  return pow_nodes;
}

bool PowEquivSubstitutionPass::GetScalarInput(const ge::AscNodePtr &pow_node, std::string &scalar_val) {
  auto pow_in_anchor = pow_node->GetInDataAnchor(1);
  while (pow_in_anchor != nullptr && pow_in_anchor->GetPeerOutAnchor() != nullptr) {
    auto target_node = std::dynamic_pointer_cast<ge::AscNode>(pow_in_anchor->GetPeerOutAnchor()->GetOwnerNode());
    GE_ASSERT_NOTNULL(target_node);
    if (IsOps<ge::ascir_op::Scalar>(target_node)) {
      auto ir_attr = target_node->attr.ir_attr.get();
      GE_ASSERT_NOTNULL(ir_attr);
      GE_ASSERT_SUCCESS(ir_attr->GetAttrValue("value", scalar_val));
      return true;
    } else if (IsOps<ge::ascir_op::Broadcast>(target_node)) {
      pow_in_anchor = target_node->GetInDataAnchor(0);
    } else {
      return false;
    }
  }
  return false;
}

ge::Status PowEquivSubstitutionPass::EnsureInputWithBrcIfNeeded(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto src_out_anchor = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_out_anchor);

  auto owner_node = std::dynamic_pointer_cast<ge::AscNode>(src_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(owner_node);

  if (IsOps<ge::ascir_op::Scalar>(owner_node)) {
    // 找到 Scalar，需要插入 Brc
    std::string brc_name = pow_node->GetName() + "_Input0_Brc";
    ge::ascir_op::Broadcast brc(brc_name.c_str());
    auto brc_node = graph.AddNode(brc);
    GE_ASSERT_NOTNULL(brc_node);
    brc_node->attr.sched = pow_node->attr.sched;
    brc_node->outputs[0].attr = pow_node->outputs[0].attr;

    // 断开 Scalar 到 Pow 的边，连 Scalar -> Brc -> Pow
    GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(src_out_anchor, pow_in_anchor));
    GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(src_out_anchor, brc_node->GetInDataAnchor(0)));
    GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(brc_node->GetOutDataAnchor(0), pow_in_anchor));

    GELOGD("Inserted Brc node [%s] between Scalar [%s] and Pow [%s]", brc_node->GetNamePtr(), owner_node->GetNamePtr(),
           pow_node->GetNamePtr());
  }
  return ge::SUCCESS;
}

// 重连Pow输入到目标节点
ge::Status PowEquivSubstitutionPass::RelinkPowInputToNode(const ge::AscNodePtr &pow_node,
                                                          const ge::AscNodePtr &target_node, const int32_t in_idx) {
  const auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  const auto target_out = pow_in_anchor->GetPeerOutAnchor();
  return ge::GraphUtils::ReplaceEdgeDst(target_out, pow_in_anchor, target_node->GetInDataAnchor(in_idx));
}

// 重连Pow输出到目标节点
ge::Status PowEquivSubstitutionPass::RelinkPowOutputToNode(const ge::AscNodePtr &pow_node,
                                                           const ge::AscNodePtr &target_node) {
  const auto new_src = target_node->GetOutDataAnchor(0);
  const auto old_src = pow_node->GetOutDataAnchor(0);
  return PassUtils::RelinkAllOutNodeToSrc(old_src, new_src);
}

Status PowEquivSubstitutionPass::ReplaceWithScalarBrc(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto brc_node = PassUtils::CreateOneScalarBrc(graph, pow_node);
  GE_ASSERT_NOTNULL(brc_node);
  GE_ASSERT_SUCCESS(RelinkPowOutputToNode(pow_node, brc_node), "Failed to replace pow [%s] with brc [%s].",
                    pow_node->GetNamePtr(), brc_node->GetNamePtr());
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithSqrt(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  std::string sqrt_name = pow_node->GetName() + "_Sqrt";
  ge::ascir_op::Sqrt sqrt(sqrt_name.c_str());
  auto sqrt_node = graph.AddNode(sqrt);
  GE_ASSERT_NOTNULL(sqrt_node);
  sqrt_node->attr.sched = pow_node->attr.sched;
  sqrt_node->outputs[0].attr = pow_node->outputs[0].attr;

  // 重连输入
  GE_ASSERT_SUCCESS(RelinkPowInputToNode(pow_node, sqrt_node), "Failed to relink input for Pow node [%s] to Sqrt node",
                    pow_node->GetNamePtr());

  // 重连输出
  GE_ASSERT_SUCCESS(RelinkPowOutputToNode(pow_node, sqrt_node),
                    "Failed to relink output for Pow node [%s] to Sqrt node", pow_node->GetNamePtr());
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithInverseSqrt(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto brc_node = PassUtils::CreateOneScalarBrc(graph, pow_node);
  GE_ASSERT_NOTNULL(brc_node);

  std::string sqrt_name = pow_node->GetName() + "_Sqrt";
  ge::ascir_op::Sqrt sqrt(sqrt_name.c_str());
  auto sqrt_node = graph.AddNode(sqrt);
  GE_ASSERT_NOTNULL(sqrt_node);
  sqrt_node->attr.sched = pow_node->attr.sched;
  sqrt_node->outputs[0].attr = pow_node->outputs[0].attr;

  std::string div_name = pow_node->GetName() + "_Div";
  ge::ascir_op::Div div(div_name.c_str());
  auto div_node = graph.AddNode(div);
  GE_ASSERT_NOTNULL(div_node);
  div_node->attr.sched = pow_node->attr.sched;
  div_node->outputs[0].attr = pow_node->outputs[0].attr;
  div.x2 = sqrt.y;
  // div input0 连brc
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(brc_node->GetOutDataAnchor(0), div_node->GetInDataAnchor(0)));
  // 重连Sqrt输入
  GE_ASSERT_SUCCESS(RelinkPowInputToNode(pow_node, sqrt_node), "Failed to relink input for Pow node [%s] to Sqrt node",
                    pow_node->GetNamePtr());

  // 重连Div输出
  GE_ASSERT_SUCCESS(RelinkPowOutputToNode(pow_node, div_node), "Failed to relink output for Pow node [%s] to Div node",
                    pow_node->GetNamePtr());
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::RemoveUseLessPow([[maybe_unused]] ge::AscGraph &graph,
                                                  const ge::AscNodePtr &pow_node) {
  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto new_src = pow_in_anchor->GetPeerOutAnchor();
  auto old_src = pow_node->GetOutDataAnchor(0);
  GE_ASSERT_SUCCESS(PassUtils::RelinkAllOutNodeToSrc(old_src, new_src));
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithInverseInput(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto brc_node = PassUtils::CreateOneScalarBrc(graph, pow_node);
  GE_ASSERT_NOTNULL(brc_node);

  std::string div_name = pow_node->GetName() + "_Div";
  ge::ascir_op::Div div(div_name.c_str());
  auto div_node = graph.AddNode(div);
  GE_ASSERT_NOTNULL(div_node);
  div_node->attr.sched = pow_node->attr.sched;
  div_node->outputs[0].attr = pow_node->outputs[0].attr;
  // div input0 连brc
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(brc_node->GetOutDataAnchor(0), div_node->GetInDataAnchor(0)));
  // 重连div input1
  GE_ASSERT_SUCCESS(RelinkPowInputToNode(pow_node, div_node, 1), "Failed to relink input for Pow node [%s] to Div node",
                    pow_node->GetNamePtr());
  // 重连Div输出
  GE_CHK_STATUS_RET(RelinkPowOutputToNode(pow_node, div_node), "Failed to relink output for Pow node [%s] to Div node",
                    pow_node->GetNamePtr());
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithMul(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto mul_node = CreateMulNodeWithAttr(graph, pow_node, "_Mul");
  GE_ASSERT_NOTNULL(mul_node);
  // 重连 mul 输入
  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto target_out = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(target_out, pow_in_anchor));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul_node->GetInDataAnchor(1)));
  // 重连Mul输出
  GE_CHK_STATUS_RET(RelinkPowOutputToNode(pow_node, mul_node), "Failed to relink output for Pow node [%s] to Mul node",
                    pow_node->GetNamePtr());
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithInverseMul(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto brc_node = PassUtils::CreateOneScalarBrc(graph, pow_node);
  GE_ASSERT_NOTNULL(brc_node);

  auto mul_node = CreateMulNodeWithAttr(graph, pow_node, "_Mul");
  GE_ASSERT_NOTNULL(mul_node);

  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto target_out = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(target_out, pow_in_anchor));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul_node->GetInDataAnchor(1)));

  std::string div_name = pow_node->GetName() + "_Div";
  ge::ascir_op::Div div(div_name.c_str());
  auto div_node = graph.AddNode(div);
  GE_ASSERT_NOTNULL(div_node);
  div_node->attr.sched = pow_node->attr.sched;
  div_node->outputs[0].attr = pow_node->outputs[0].attr;

  // div输入连边
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(brc_node->GetOutDataAnchor(0), div_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), div_node->GetInDataAnchor(1)));

  // 重连Div输出
  GE_CHK_STATUS_RET(RelinkPowOutputToNode(pow_node, div_node), "Failed to relink output for Pow node [%s] to Div node",
                    pow_node->GetNamePtr());
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithCube(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto mul1_node = CreateMulNodeWithAttr(graph, pow_node, "_Mul1");
  GE_ASSERT_NOTNULL(mul1_node);
  auto mul2_node = CreateMulNodeWithAttr(graph, pow_node, "_Mul2");
  GE_ASSERT_NOTNULL(mul2_node);
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0)));

  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto target_out = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(target_out, pow_in_anchor));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul1_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul1_node->GetInDataAnchor(1)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul2_node->GetInDataAnchor(1)));

  // 重连Mul2输出
  GE_CHK_STATUS_RET(RelinkPowOutputToNode(pow_node, mul2_node), "Failed to relink output for Pow node [%s] to Mul node",
                    pow_node->GetNamePtr());
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithFourthPower(ge::AscGraph &graph, const ge::AscNodePtr &pow_node) {
  auto mul1_node = CreateMulNodeWithAttr(graph, pow_node, "_Mul1");
  GE_ASSERT_NOTNULL(mul1_node);
  auto mul2_node = CreateMulNodeWithAttr(graph, pow_node, "_Mul2");
  GE_ASSERT_NOTNULL(mul2_node);
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1)));

  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto target_out = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(target_out, pow_in_anchor));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul1_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul1_node->GetInDataAnchor(1)));

  // 重连Mul2输出
  GE_CHK_STATUS_RET(RelinkPowOutputToNode(pow_node, mul2_node), "Failed to relink output for Pow node [%s] to Mul node",
                    pow_node->GetNamePtr());
  return ge::SUCCESS;
}
}  // namespace optimize
