/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lower_concat_helper.h"

#include "backend/backend_spec.h"
#include "graph/utils/node_utils.h"
#include "can_fuse/backend/backend_utils.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "graph/debug/ge_op_types.h"
#include "symbolizer/symbolic_utils.h"

namespace ge {
namespace {
constexpr int64_t kAlignment = 32;
constexpr uint32_t kMaxInputNum = 63U;
constexpr int32_t kAlgTranspose = 0;
constexpr int32_t kAlgScatter = 1;
}  // namespace
LowerConcatHelper::LowerConcatHelper(NodePtr fused_asc_backend_node) :
    fused_asc_backend_node_(std::move(fused_asc_backend_node)) {
}

graphStatus LowerConcatHelper::LiftingPoorPerfFusedAscBackendOps(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == kFusedAscBackendType) {
      LowerConcatHelper lower_concat_helper(node);
      GE_ASSERT_GRAPH_SUCCESS(lower_concat_helper.LiftingPoorPerfFusedAscBackendOp(),
                              "Failed to process %s", node->GetNamePtr());
    }
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus LowerConcatHelper::LiftingPoorPerfFusedAscBackendOp() {
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!FindConcatAscBackendNode(), GRAPH_SUCCESS, "concat node not found");
  bool need_lifting = false;
  GE_ASSERT_SUCCESS(NeedLifting(need_lifting));
  if (need_lifting) {
    GE_ASSERT_SUCCESS(UnfoldFusedAscBackend());
  }
  return GRAPH_SUCCESS;
}

bool LowerConcatHelper::FindConcatAscBackendNode() {
  bool found = false;
  GE_ASSERT_NOTNULL(fused_asc_backend_node_->GetOpDescBarePtr());
  const auto attr = fused_asc_backend_node_->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
  if (attr != nullptr) {
    graph_ = attr->GetFuseComputeGraph();
    GE_ASSERT_NOTNULL(graph_);
    for (const auto &node : graph_->GetAllNodes()) {
      GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
      const auto sub_attr = node->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
      if ((sub_attr != nullptr) && (sub_attr->GetFuseType() == loop::FuseType::kConcat)) {
        concat_asc_backend_node_ = node;
        const auto &asc_graph = sub_attr->GetAscGraph();
        GE_ASSERT_NOTNULL(asc_graph);
        concat_node_ = FindConcatNode(*asc_graph);
        GE_ASSERT_NOTNULL(concat_node_);
        found = true;
        break;
      }
    }
  }
  return found;
}

AscNodePtr LowerConcatHelper::FindConcatNode(const AscGraph &asc_graph) {
  AscNodePtr concat_node;
  for (const auto &n : asc_graph.GetAllNodes()) {
    if (n->GetType() == "Concat") {
      concat_node = n;
      break;
    }
  }
  return concat_node;
}

graphStatus LowerConcatHelper::ParseConcatNode() {
  bool found = false;
  for (size_t i = 0U; i < concat_node_->inputs.Size(); ++i) {
    input_shapes_.emplace_back(concat_node_->inputs[i].attr.repeats);
  }
  output_shape_ = concat_node_->outputs[0].attr.repeats;
  GE_ASSERT_EQ(input_shapes_.front().size(), output_shape_.size());
  for (size_t i = 0U; i < output_shape_.size(); ++i) {
    if (input_shapes_.front()[i] != output_shape_[i]) {
      concat_dim_ = i;
      found = true;
      break;
    }
  }
  GE_ASSERT_TRUE(found,
                 "[%s] failed to find concat dim, input_shape[0] = %s, output_shape = %s",
                 fused_asc_backend_node_->GetNamePtr(),
                 ToString(input_shapes_.front()).c_str(),
                 ToString(output_shape_).c_str());
  return GRAPH_SUCCESS;
}

graphStatus LowerConcatHelper::ParseConcatCase() {
  int64_t stride = ge::GetSizeByDataType(concat_node_->GetOpDesc()->GetOutputDesc(0).GetDataType());
  for (size_t i = concat_dim_ + 1U; i < output_shape_.size(); ++i) {
    auto &dim_expr = output_shape_[i];
    if (dim_expr.IsConstExpr()) {
      int64_t dim_size = -1;
      dim_expr.GetConstValue(dim_size);
      if (dim_size >= 0) {
        stride *= dim_size;
      }
    }
  }
  size_t num_aligned = 0;
  std::set<std::pair<ge::Node *, int32_t>> unique_srcs;
  for (const auto in_anchor : concat_asc_backend_node_->GetAllInDataAnchorsPtr()) {
    if (in_anchor != nullptr) {
      GE_ASSERT_TRUE(static_cast<size_t>(in_anchor->GetIdx()) < input_shapes_.size());
      auto &input_shape = input_shapes_[in_anchor->GetIdx()];
      GE_ASSERT_EQ(input_shape.size(), output_shape_.size());
      const auto dim_size = input_shape[concat_dim_];
      GE_CHK_BOOL_RET_SPECIAL_STATUS((!dim_size.IsConstExpr()),
                                     ge::SUCCESS,
                                     "contains non-const dim: %s",
                                     SymbolicUtils::ToString(dim_size).c_str());
      int64_t dim_size_val = -1;
      (void) dim_size.GetConstValue(dim_size_val);
      GE_CHK_BOOL_RET_SPECIAL_STATUS(dim_size_val < 0, ge::SUCCESS,
                                     "input[%zu] contains %ld dim", in_anchor->GetIdx(), dim_size_val);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_anchor);
      const auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(peer_node);
      GELOGI("input[%d] connected to %s(%s)", in_anchor->GetIdx(), peer_node->GetNamePtr(), peer_node->GetTypePtr());
      if ((peer_node->GetType() == kAscBackendType) &&
          unique_srcs.emplace(peer_node, peer_out_anchor->GetIdx()).second) {
        total_fused_dim_size_ += dim_size_val;
      }
      num_aligned += static_cast<int64_t>(stride * dim_size_val % kAlignment == 0);
    }
  }
  case_ = concat_dim_ == 0 ? ConcatCase::kFirstDim :
          (num_aligned == concat_asc_backend_node_->GetInDataNodesSize() ? ConcatCase::kAllAligned
                                                                         : ConcatCase::kOther);
  return GRAPH_SUCCESS;
}
graphStatus LowerConcatHelper::NeedLifting(bool &need_lifting) {
  static const std::map<ConcatCase, std::string> kCaseToName{
      {ConcatCase::kFirstDim, "first_dim"}, {ConcatCase::kAllAligned, "aligned"}, {ConcatCase::kOther, "other"}};
  static const std::map<int32_t, std::map<ConcatCase, ge::float64_t>> kAlgToCaseToRatio{
      {kAlgTranspose,
       std::map<ConcatCase, ge::float64_t>{
           {ConcatCase::kFirstDim, 0.0},
           {ConcatCase::kAllAligned, 0.3333},
           {ConcatCase::kOther, 0.3333},
       }},
      {kAlgScatter,
       std::map<ConcatCase, ge::float64_t>{
           {ConcatCase::kFirstDim, 0.0},
           {ConcatCase::kAllAligned, 0.1666},
           {ConcatCase::kOther, 0.1666},
       }},
  };
  GE_CHK_BOOL_RET_SPECIAL_STATUS(HasBackwardFusion(), GRAPH_SUCCESS, "has backward fusion, do not lifting");
  if (!IsTile()) {
    GE_CHK_BOOL_RET_SPECIAL_STATUS(concat_asc_backend_node_->GetAllInDataAnchorsSize() > kMaxInputNum, GRAPH_SUCCESS,
                                   "num_inputs = %zu, do not lifting",
                                   concat_asc_backend_node_->GetAllInDataAnchorsSize());
    GE_CHK_BOOL_RET_SPECIAL_STATUS(concat_asc_backend_node_->GetAllInDataAnchorsSize() == 1U, GRAPH_SUCCESS,
                                   "single input, do not lifting");
  }
  GE_ASSERT_SUCCESS(ParseConcatNode());
  // 暂不处理concat_dim后为动态shape的场景
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!output_shape_[concat_dim_].IsConstExpr(),
                                 GRAPH_SUCCESS,
                                 "concat dim size is non-const");
  (void) output_shape_[concat_dim_].GetConstValue(output_dim_size_);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(output_dim_size_ <= 0,
                                 GRAPH_SUCCESS,
                                 "concat dim size is not positive");
  GE_ASSERT_SUCCESS(ParseConcatCase());
  GE_CHK_BOOL_RET_SPECIAL_STATUS(case_ == ConcatCase::kNoLifting, GRAPH_SUCCESS, "No need for lifting");
  auto buffer_ratio = static_cast<float64_t>(total_fused_dim_size_) / static_cast<float64_t>(output_dim_size_);
  auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  auto threshold = kAlgToCaseToRatio.at(backend_spec->concat_alg).at(case_);
  need_lifting = buffer_ratio < threshold;
  GELOGI("FusedAscBackend: %s, concat: %s, case = %s, ratio = %ld/%ld = %.15f, threshold = %f, need_lifting = %d",
         fused_asc_backend_node_->GetNamePtr(), concat_node_->GetNamePtr(),
         kCaseToName.at(case_).c_str(), total_fused_dim_size_, output_dim_size_,
         buffer_ratio, threshold, need_lifting);
  return GRAPH_SUCCESS;
}

graphStatus LowerConcatHelper::UnfoldFusedAscBackend() const {
  auto is_valid_graph = CheckGraph();
  GE_LOGW_IF(!is_valid_graph, "[%s] skip unfold, graph is abnormal", fused_asc_backend_node_->GetNamePtr());
  if (is_valid_graph) {
    GELOGI("FusedAscBackend node name = [%s], AscBackend node name = [%s], concat node name = [%s]",
           fused_asc_backend_node_->GetNamePtr(), concat_asc_backend_node_->GetNamePtr(), concat_node_->GetNamePtr());
    auto parent_graph = fused_asc_backend_node_->GetOwnerComputeGraph();
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::UnfoldGraph(graph_, parent_graph, fused_asc_backend_node_, nullptr),
                            "Failed to unfold graph");
    GE_ASSERT_GRAPH_SUCCESS(parent_graph->TopologicalSorting());
  }
  return GRAPH_SUCCESS;
}

bool LowerConcatHelper::CheckGraph() const {
  auto output_node = graph_->FindFirstNodeMatchType(NETOUTPUT);
  GE_WARN_ASSERT(output_node != nullptr);
  auto in_num = graph_->GetInputNodes().size();
  auto parent_in_num = static_cast<size_t>(fused_asc_backend_node_->GetAllInDataAnchorsSize());
  GE_WARN_ASSERT(in_num == parent_in_num, "[%s] input not match, in_num = %u, parent_node in_num = %u",
                 fused_asc_backend_node_->GetNamePtr(), in_num, parent_in_num);

  auto out_num = output_node->GetAllInDataAnchorsSize();
  auto parent_out_num = fused_asc_backend_node_->GetAllOutDataAnchorsSize();
  GE_WARN_ASSERT(out_num == parent_out_num, "[%s] output not match, out_num = %u, parent out_num = %u",
                 fused_asc_backend_node_->GetNamePtr(), out_num, parent_out_num);
  return true;
}

bool LowerConcatHelper::HasBackwardFusion() const {
  const auto out_data_nodes = concat_asc_backend_node_->GetOutDataNodes();
  return std::any_of(out_data_nodes.begin(),
                     out_data_nodes.end(),
                     [](const ge::NodePtr &peer_node) -> bool {
                       return peer_node->GetType() == kAscBackendType;
                     });
}

bool LowerConcatHelper::IsTile() const {
  return (concat_node_->GetInDataNodesSize() > 1UL) && (concat_asc_backend_node_->GetInDataNodesSize() == 1UL);
}
}  // namespace ge
