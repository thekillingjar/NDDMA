/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/node_utils.h"
#include "can_fuse/backend/backend_utils.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "graph/debug/ge_op_types.h"
#include "symbolizer/symbolic_utils.h"
#include "lower_split_helper.h"
#include "utils/autofuse_utils.h"

namespace ge {
namespace {
constexpr int64_t kAlignment = 32;
constexpr uint32_t kMaxOutputNum = 63U;
}  // namespace
LowerSplitHelper::LowerSplitHelper(NodePtr asc_backend_node) :
    asc_backend_node_(std::move(asc_backend_node)) {
}

graphStatus LowerSplitHelper::InitAndCheck() {
  GE_ASSERT_NOTNULL(asc_backend_node_->GetOpDescBarePtr());
  const auto attr = asc_backend_node_->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_TRUE(attr->HasFuseType(loop::FuseType::kSplit));
  const auto &asc_graph = attr->GetAscGraph();
  GE_ASSERT_NOTNULL(asc_graph);
  FindSplitNodes(*asc_graph, split_nodes_);
  GE_ASSERT_TRUE(split_nodes_.size() != 0U);
  return GRAPH_SUCCESS;
}

void LowerSplitHelper::FindSplitNodes(const AscGraph &asc_graph, std::vector<AscNodePtr> &split_nodes) {
  AscNodePtr split_node;
  for (const auto &n : asc_graph.GetAllNodes()) {
    if (AutofuseUtils::IsSplitType(n->GetType())) {
      split_nodes.emplace_back(n);
    }
  }
}

graphStatus LowerSplitHelper::ParseSplitNode(bool &found) {
  found = false;
  for (auto split_node : split_nodes_) {
    output_shapes_.emplace_back(split_node->outputs[0].attr.repeats);
  }
  GE_ASSERT_TRUE(split_nodes_.size() >= 1U);
  input_shape_ = split_nodes_[0]->inputs[0].attr.repeats;
  GE_ASSERT_EQ(input_shape_.size(), output_shapes_.front().size());
  for (size_t i = 0U; i < input_shape_.size(); ++i) {
    // 单输出的split可能输入输出形状一致
    if (output_shapes_.front()[i] != input_shape_[i]) {
      split_dim_ = i;
      found = true;
      break;
    }
  }
  if (!found) {
    GELOGI("[%s] split dim not found, input_shape = %s, output_shapes[0] = %s",
           asc_backend_node_->GetNamePtr(),
           ToString(input_shape_).c_str(),
           ToString(output_shapes_.front()).c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus LowerSplitHelper::ParseSplitCase() {
  GE_ASSERT_TRUE(split_nodes_.size() >= 1U);
  int64_t stride = ge::GetSizeByDataType(split_nodes_[0]->GetOpDesc()->GetOutputDesc(0).GetDataType());
  for (size_t i = split_dim_ + 1U; i < input_shape_.size(); ++i) {
    auto &dim_expr = input_shape_[i];
    if (dim_expr.IsConstExpr()) {
      int64_t split_dim_size = -1;
      dim_expr.GetConstValue(split_dim_size);
      if (split_dim_size >= 0) {
        stride *= split_dim_size;
      }
    }
  }
  size_t num_aligned = 0;
  GE_ASSERT_NOTNULL(asc_backend_node_);
  GELOGD("node: %s(%s), in anchor size: %d, out anchor size: %d", asc_backend_node_->GetNamePtr(),
         asc_backend_node_->GetTypePtr(), asc_backend_node_->GetAllInDataAnchorsSize(), asc_backend_node_->GetAllOutDataAnchorsSize());
  for (const auto &split_node : split_nodes_) {
    auto out_anchor = split_node->GetOutDataAnchor(0);
    if (out_anchor != nullptr) {
      GE_ASSERT_TRUE(static_cast<size_t>(out_anchor->GetIdx()) < output_shapes_.size());
      auto &output_shape = output_shapes_[out_anchor->GetIdx()];
      GE_ASSERT_EQ(output_shape.size(), input_shape_.size());
      GE_ASSERT_TRUE(split_dim_< output_shape.size());
      const auto dim_size = output_shape[split_dim_];
      GE_CHK_BOOL_RET_SPECIAL_STATUS((!dim_size.IsConstExpr()),
                                     ge::SUCCESS,
                                     "contains non-const dim: %s",
                                     SymbolicUtils::ToString(dim_size).c_str());
      int64_t split_dim_size_val = -1;
      (void) dim_size.GetConstValue(split_dim_size_val);
      GE_CHK_BOOL_RET_SPECIAL_STATUS(split_dim_size_val < 0, ge::SUCCESS,
                                     "output[%zu] contains %ld dim", out_anchor->GetIdx(), split_dim_size_val);
      GE_ASSERT_TRUE(!out_anchor->GetPeerInDataAnchors().empty());
      for (auto peer_in_anchor: out_anchor->GetPeerInDataAnchors()) {
        const auto peer_node = peer_in_anchor->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(peer_node);
        GELOGI("output[%d] connected to %s(%s)", out_anchor->GetIdx(), peer_node->GetNamePtr(), peer_node->GetTypePtr());
        if (peer_node->GetType() != kStoreType) {
          total_fused_dim_size_ += split_dim_size_val;
        }
        num_aligned += static_cast<int64_t>(stride * split_dim_size_val % kAlignment == 0);
      }
    }
  }
  case_ = split_dim_ == 0 ? SplitCase::kFirstDim :
          (num_aligned == asc_backend_node_->GetOutDataNodesSize() ? SplitCase::kAllAligned
                                                                         : SplitCase::kOther);
  return GRAPH_SUCCESS;
}

graphStatus LowerSplitHelper::NeedLifting(bool &need_lifting) {
  GE_ASSERT_GRAPH_SUCCESS(InitAndCheck());
  bool ratio_need_lifting;
  GE_ASSERT_GRAPH_SUCCESS(CheckFuseOtherNodes(need_lifting));
  // 暂时仅保留分支，待调试完放开
  GE_ASSERT_GRAPH_SUCCESS(CheckFuseRatio(ratio_need_lifting));
  return GRAPH_SUCCESS;
}

graphStatus LowerSplitHelper::CheckFuseOtherNodes(bool &need_lifting) {
  const auto desc = asc_backend_node_->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(desc);
  const auto attr = desc->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(attr);
  auto origin_nodes = attr->GetOriginNodes();
  for (auto &origin_node : origin_nodes) {
    // 如果有融合其它类型的节点，则判断不需要lifting
    if (!AutofuseUtils::IsSplitType(origin_node->GetType())) {
      need_lifting = false;
      return GRAPH_SUCCESS;
    }
  }
  // 如果存在同一个原节点lowering成的split节点被融合到其它节点，则可能会导致canfuse后的此ascbackend输出比原split节点少，此时也暂时不lifting
  GE_ASSERT_TRUE(origin_nodes.size() >= 1U);
  if (asc_backend_node_->GetAllOutDataAnchorsSize() < origin_nodes[0]->GetAllOutDataAnchorsSize()) {
    need_lifting = false;
    return GRAPH_SUCCESS;
  }
  need_lifting = true;
  return GRAPH_SUCCESS;
}

graphStatus LowerSplitHelper::CheckFuseRatio(bool &need_lifting) {
  static const std::map<SplitCase, std::string> kCaseToName
      {{SplitCase::kFirstDim, "first_dim"}, {SplitCase::kAllAligned, "aligned"}, {SplitCase::kOther, "other"}};
  static const std::map<SplitCase, ge::float64_t> kCaseToRatio{
      {SplitCase::kFirstDim, 0.0},
      {SplitCase::kAllAligned, 0.3333},
      {SplitCase::kOther, 0.3333},
  };
  GE_CHK_BOOL_RET_SPECIAL_STATUS(asc_backend_node_->GetAllOutDataAnchorsSize() > kMaxOutputNum, GRAPH_SUCCESS,
                                 "num_outputs = %zu, do not lifting",
                                 asc_backend_node_->GetAllOutDataAnchorsSize());
  bool found_split_dim = false;
  GE_ASSERT_SUCCESS(ParseSplitNode(found_split_dim));
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!found_split_dim, GRAPH_SUCCESS, "split dim not found");
  // 暂不处理split_dim后为动态shape的场景
  GE_ASSERT_TRUE(split_dim_< input_shape_.size());
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!input_shape_[split_dim_].IsConstExpr(),
                                 GRAPH_SUCCESS,
                                 "split dim size is non-const");
  (void) input_shape_[split_dim_].GetConstValue(input_dim_size_);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(input_dim_size_ <= 0,
                                 GRAPH_SUCCESS,
                                 "split dim size is not positive");
  GE_ASSERT_SUCCESS(ParseSplitCase());
  GE_CHK_BOOL_RET_SPECIAL_STATUS(case_ == SplitCase::kNoLifting, GRAPH_SUCCESS, "No need for lifting");
  auto buffer_ratio = static_cast<float64_t>(total_fused_dim_size_) / static_cast<float64_t>(input_dim_size_);
  auto threshold = kCaseToRatio.at(case_);
  need_lifting = buffer_ratio < threshold;
  return GRAPH_SUCCESS;
}
}  // namespace ge