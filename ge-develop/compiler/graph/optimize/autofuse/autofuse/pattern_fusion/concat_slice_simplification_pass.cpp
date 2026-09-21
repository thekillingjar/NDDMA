/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_slice_simplification_pass.h"

#include "checker.h"
#include "compute_graph.h"
#include "node_utils.h"
#include "op_desc_utils.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "symbolizer/symbolic.h"
#include "utils/autofuse_utils.h"
#include "flatten_concat_pass.h"
#include "graph_utils.h"

namespace ge {
namespace {
constexpr auto kOpTypeSlice = "Slice";
constexpr auto kOpTypeSliceD = "SliceD";
constexpr auto kOpTypeConcat = "Concat";
constexpr auto kOpTypeConcatD = "ConcatD";
constexpr auto kOpTypeConcatV2 = "ConcatV2";
constexpr auto kOpTypeConcatV2D = "ConcatV2D";
constexpr auto kAttrNameN = "N";

bool IsSlice(const NodePtr &node) {
  return node->GetType() == kOpTypeSlice || node->GetType() == kOpTypeSliceD;
}

bool IsFromConcat(const NodePtr &node) {
  const auto src_node = NodeUtils::GetInDataNodeByIndex(*node, 0);
  GE_WARN_ASSERT(src_node != nullptr);
  const auto &src_op_type = src_node->GetType();
  static const std::set<std::string> kConcatOpTypes{kOpTypeConcat, kOpTypeConcatD, kOpTypeConcatV2, kOpTypeConcatV2D};
  return kConcatOpTypes.find(src_op_type) != kConcatOpTypes.end();
}

bool HasAnyControlEdges(const NodePtr &node) {
  return node->GetInControlNodesSize() > 0 || node->GetOutControlNodesSize() > 0;
}
}  // namespace

graphStatus ConcatSliceSimplificationPass::Run(const ComputeGraphPtr &graph, const GraphPasses &graph_passes, bool &changed) {
  for (const auto &node : graph->GetAllNodes()) {
    if (IsSlice(node) && IsFromConcat(node)) {
      (void)HandleSlice(node);
    } else {
      // do nothing
    }
  }
  if (need_prune_  && (graph_passes.prune_graph_func != nullptr)) {
      (void)graph_passes.prune_graph_func(graph);
      GE_ASSERT_GRAPH_SUCCESS(graph->TopologicalSorting());
      changed = true;
  }
  if (need_constant_folding_ && (graph_passes.constant_folding_func != nullptr)) {
    for (auto &node : graph->GetAllNodes()) {
      GE_ASSERT_GRAPH_SUCCESS(graph_passes.constant_folding_func(node));
    }
    changed = true;
  }
  return GRAPH_SUCCESS;
}

graphStatus ConcatSliceSimplificationPass::HandleSlice(const NodePtr &node) {
  GE_WARN_ASSERT(!HasAnyControlEdges(node), "Skip handle node %s, as: has control edges.", node->GetNamePtr());
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  GE_WARN_ASSERT(AutofuseUtils::GetListIntByInputOrAttr(node, offsets, "offsets", "offsets") == GRAPH_SUCCESS,
                 "Skip handle node %s, as: Failed to get attr: offsets.", node->GetNamePtr());
  GE_WARN_ASSERT(AutofuseUtils::GetListIntByInputOrAttr(node, sizes, "size", "size") == GRAPH_SUCCESS,
                 "Skip handle node %s, as: Failed to get attr: size.", node->GetNamePtr());
  GE_WARN_ASSERT(sizes.size() == offsets.size());
  const auto &concat_node = NodeUtils::GetInDataNodeByIndex(*node, 0);
  size_t concat_dim;
  GE_WARN_ASSERT(FlattenConcatPass::ResolveConcatDim(concat_node, concat_dim) == GRAPH_SUCCESS,
                 "Skip handle node %s, as: Failed to get concat_dim from src node: %s.", node->GetNamePtr(),
                 concat_node->GetNamePtr());
  GE_WARN_ASSERT(concat_dim < offsets.size());

  size_t input_index = std::numeric_limits<size_t>::max();
  if (!FindInput(concat_node, concat_dim, sizes, offsets, input_index)) {
    GELOGD("slice: %s does not from single source of concat", node->GetNamePtr());
    return GRAPH_SUCCESS;
  }
  // replace node
  const auto &[src_node, src_node_out_anchor] =
      NodeUtils::GetInDataNodeAndAnchorByIndex(*concat_node, static_cast<int32_t>(input_index));
  GELOGD("found single source input, index = %zu, src_node = %s(%s)", input_index, src_node->GetNamePtr(),
         src_node->GetTypePtr());
  GE_ASSERT_NOTNULL(src_node);
  const std::string new_name = node->GetName() + "_by_ConcatSliceSimplificationPass";
  auto new_slice_node = AddNewSliceNode(node->GetOwnerComputeGraph(), new_name, offsets, sizes);
  GE_ASSERT_NOTNULL(new_slice_node);
  GE_ASSERT_GRAPH_SUCCESS(new_slice_node->GetOpDesc()->UpdateInputDesc(0, src_node->GetOpDesc()->GetOutputDesc(src_node_out_anchor->GetIdx())));
  GE_ASSERT_GRAPH_SUCCESS(new_slice_node->GetOpDesc()->UpdateOutputDesc(0, node->GetOpDesc()->GetOutputDesc(0)));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodesOutDataAnchors({new_slice_node}, {node}, {0}));
  NodeUtils::UnlinkAll(*node);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src_node_out_anchor, new_slice_node->GetInDataAnchor(0)));
  need_prune_ = true;
  need_constant_folding_ = (need_constant_folding_ || NodeUtils::IsConst(*src_node));
  return GRAPH_SUCCESS;
}

bool ConcatSliceSimplificationPass::FindInput(const NodePtr &concat_node, size_t concat_dim, const std::vector<int64_t> &sizes,
                                              std::vector<int64_t> &offsets, size_t &input_index) {
  size_t num_inputs;
  const auto &concat_op_desc = concat_node->GetOpDesc();
  GE_ASSERT_NOTNULL(concat_op_desc);
  GE_WARN_ASSERT(AttrUtils::GetInt(concat_op_desc, kAttrNameN, num_inputs));
  GE_WARN_ASSERT(num_inputs <= concat_node->GetAllInDataAnchorsSize());
  const int64_t slice_start = offsets[concat_dim];
  const int64_t slice_end = sizes[concat_dim] == -1 ? concat_op_desc->GetOutputDesc(0).GetShape().GetDim(concat_dim)
                                                    : (slice_start + sizes[concat_dim]);
  GELOGD("slice_start = %ld, slice_end = %ld", slice_start, slice_end);
  int64_t span_start = 0;
  for (size_t i = 0UL; i < num_inputs; ++i) {
    const auto &input_desc = concat_op_desc->GetInputDesc(i);
    const auto dim_size = input_desc.GetShape().GetDim(concat_dim);
    const auto span_end = span_start + dim_size;
    GELOGD("input index = %zu, span_start = %ld, span_end = %ld", i, span_start, span_end);
    if (span_start >= slice_end) {
      // not found
      break;
    }
    if ((span_start <= slice_start) && (slice_end <= span_end)) {
      input_index = i;
      offsets[concat_dim] -= span_start;
      return true;
    }
    span_start = span_end;
  }
  return false;
}

NodePtr ConcatSliceSimplificationPass::AddNewSliceNode(const ComputeGraphPtr &graph, const std::string &name,
                                                       const std::vector<int64_t> &offsets,
                                                       const std::vector<int64_t> &sizes) {
  auto op = ge::OperatorFactory::CreateOperator(name.c_str(), kOpTypeSliceD);
  op.SetAttr("offsets", offsets);
  op.SetAttr("size", sizes);
  op.BreakConnect();
  const auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto new_node = graph->AddNode(op_desc);
  GELOGD("create new Slice node: %s", name.c_str());
  return new_node;
}
}  // namespace ge
