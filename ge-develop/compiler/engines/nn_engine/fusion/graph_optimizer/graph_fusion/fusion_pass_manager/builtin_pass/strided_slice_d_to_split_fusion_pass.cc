/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/strided_slice_d_to_split_fusion_pass.h"
#include <string>
#include "common/util/op_info_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"
#include "common/string_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
namespace {
const string kPatternStridedSliceD = "strided_slice_d_pattern";
const string kAttrBegin = "begin";
const string kAttrEnd = "end";
const string kAttrStrides = "strides";
const string kAttrBeginMask = "begin_mask";
const string kAttrEndMask = "end_mask";
const string kAttrEllipsisMask = "ellipsis_mask";
const string kAttrNewAxisMask = "new_axis_mask";
const string kAttrShrinkAxisMask = "shrink_axis_mask";
const string kSplitDim = "split_dim";
const string kNumSplit = "num_split";
const int64_t kNumSplitMax = 63;
}
vector<FusionPattern *> StridedSliceDToSplitFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("StridedSliceDToSplitFusionPass");
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][StridedSliceDToSplitFus][DefPtn] Failed to create a new object."),
           return patterns);
  pattern->AddOpDesc(kPatternStridedSliceD, {STRIDEDSLICED}).SetOutput(kPatternStridedSliceD);
  patterns.push_back(pattern);
  return patterns;
}

std::vector<size_t> StridedSliceDToSplitFusionPass::GetMaskAxis(const int64_t &mask_value, const size_t &dims_size) {
  std::vector<size_t> mask_axis;
  if (mask_value == 0) {
    return mask_axis;
  }
  int64_t tmp_bit_value;
  for (size_t i = 0; i < dims_size; ++i) {
    tmp_bit_value = (abs(mask_value) >> i) & 1;
    if (tmp_bit_value == 1) {
      if (mask_value > 0) {
        mask_axis.emplace_back(dims_size -1 - i);
      } else {
        mask_axis.emplace_back(i);
      }
    }
  }
  return mask_axis;
}

Status StridedSliceDToSplitFusionPass::ModifyBeginAndEnd(const ge::OpDescPtr &strided_slice_d_desc,
                                                         const std::vector<int64_t> &ori_shape,
                                                         std::vector<int64_t> &begins,
                                                         std::vector<int64_t> &ends) {
  int64_t ellipsis_mask = 0;
  (void)ge::AttrUtils::GetInt(strided_slice_d_desc, kAttrEllipsisMask, ellipsis_mask);
  if (ellipsis_mask < 0) {
    FE_LOGW("Node[%s]: the attr ellipsis_mask is not supported by negative value.",
            strided_slice_d_desc->GetName().c_str());
    return FAILED;
  }
  std::vector<size_t> ellipsis_mask_axis = GetMaskAxis(ellipsis_mask, ori_shape.size());
  if (ellipsis_mask_axis.size() > 1) {
    FE_LOGW("Node[%s]: the attr ellipsis_mask is not supported by multiple non-zero bits.",
            strided_slice_d_desc->GetName().c_str());
    return FAILED;
  }
  int64_t begin_mask = 0;
  (void)ge::AttrUtils::GetInt(strided_slice_d_desc, kAttrBeginMask, begin_mask);
  std::vector<size_t> begin_mask_axis = GetMaskAxis(begin_mask, ori_shape.size());
  for (size_t &id : begin_mask_axis) {
    if (id < begins.size()) {
      begins[id] = 0;
    }
  }
  int64_t end_mask = 0;
  (void)ge::AttrUtils::GetInt(strided_slice_d_desc, kAttrEndMask, end_mask);
  std::vector<size_t> end_mask_axis = GetMaskAxis(end_mask, ori_shape.size());
  for (size_t &id : end_mask_axis) {
    if (id < begins.size()) {
      ends[id] = ori_shape[id];
    }
  }
  for (size_t &id : ellipsis_mask_axis) {
    if (id < begins.size()) {
      begins[id] = 0;
      ends[id] = ori_shape[id];
    }
  }
  for (size_t i = 0; i < begins.size(); ++i) {
    if (begins[i] < 0) {
      begins[i] += ori_shape[i];
    }
    if (ends[i] < 0) {
      ends[i] += ori_shape[i];
    }
    if (ends[i] > ori_shape[i]) {
      ends[i] = ori_shape[i];
    }
  }
  return SUCCESS;
}

bool StridedSliceDToSplitFusionPass::CheckEvenlySlice(const ge::OpDescPtr &strided_slice_d_desc,
                                                      const std::vector<int64_t> &ori_shape,
                                                      std::vector<int64_t> &begins,
                                                      std::vector<int64_t> &ends,
                                                      SplitInfo &split_info) {
  // modify begin and end
  FE_LOGD("Node[%s]: the begins is %s, and the ends is %s.", strided_slice_d_desc->GetName().c_str(),
          StringUtils::IntegerVecToString(begins).c_str(), StringUtils::IntegerVecToString(ends).c_str());
  if (ModifyBeginAndEnd(strided_slice_d_desc, ori_shape, begins, ends) != SUCCESS) {
    return false;
  }
  FE_LOGD("Node[%s]: after modify, the begins is %s, and the ends is %s.", strided_slice_d_desc->GetName().c_str(),
          StringUtils::IntegerVecToString(begins).c_str(), StringUtils::IntegerVecToString(ends).c_str());
  // evenly slice check
  int64_t slice_dim_cnt = 0;
  for (size_t i = 0; i < begins.size(); ++i) {
    int64_t tmp_slice_size = ends[i] - begins[i];
    if (tmp_slice_size > 0 && tmp_slice_size < ori_shape[i]) {
      if ((ori_shape[i] / tmp_slice_size) > kNumSplitMax) {
        FE_LOGD("Node[%s]: the num_split exceeds the maximum 63.", strided_slice_d_desc->GetName().c_str());
        return false;
      }
      slice_dim_cnt += 1;
      if ((ori_shape[i] % tmp_slice_size) != 0 || (begins[i] % tmp_slice_size) != 0) {
        FE_LOGD("Node [%s]: Failed to check evenly sliced.", strided_slice_d_desc->GetName().c_str());
        return false;
      }
      split_info.split_dim = static_cast<int64_t>(i);
      split_info.num_split = ori_shape[i] / tmp_slice_size;
      split_info.split_out_anchor = begins[i] / tmp_slice_size;
    } else if (tmp_slice_size == ori_shape[i]) {
      continue;
    } else {
      FE_LOGD("Node[%s]: the slice_size %ld check was unsuccessful.", strided_slice_d_desc->GetName().c_str(), tmp_slice_size);
      return false;
    }
  }
  if (slice_dim_cnt != 1) {
    FE_LOGD("Node[%s]: multi-dims slice is not supported yet for this pass.", strided_slice_d_desc->GetName().c_str());
    return false;
  }
  return true;
}

bool StridedSliceDToSplitFusionPass::CheckCommonCondition(const ge::NodePtr &strided_slice_d_node,
                                                          const ge::OpDescPtr &strided_slice_d_desc,
                                                          SplitInfo &split_info) {
  // shape check
  if (UnknownShapeUtils::IsUnknownShapeOp(*strided_slice_d_desc)) {
    FE_LOGD("Node[%s]: unknown shape is not supported yet for this pass.", strided_slice_d_desc->GetName().c_str());
    return false;
  }
  // input check
  for (const auto &in_node : strided_slice_d_node->GetInDataNodes()) {
    if (in_node->GetType() == fe::CONCATD || in_node->GetType() == fe::CONCATV2D) {
      FE_LOGD("Node[%s]: check input[%s] unsuccessful.", strided_slice_d_desc->GetName().c_str(),
              in_node->GetName().c_str());
      return false;
    }
  }
  // axis attr check
  int64_t new_axis_mask = 0;
  (void)ge::AttrUtils::GetInt(strided_slice_d_desc, kAttrNewAxisMask, new_axis_mask);
  if (new_axis_mask != 0) {
    FE_LOGD("Node[%s]: check the attr new_axis_mask %ld unsuccessful.",
            strided_slice_d_desc->GetName().c_str(), new_axis_mask);
    return false;
  }
  int64_t shrink_axis_mask = 0;
  (void)ge::AttrUtils::GetInt(strided_slice_d_desc, kAttrShrinkAxisMask, shrink_axis_mask);
  if (shrink_axis_mask != 0) {
    FE_LOGD("Node[%s]: failed to check the attribute shrink_axis_mask %ld.",
            strided_slice_d_desc->GetName().c_str(), shrink_axis_mask);
    return false;
  }
  // size check
  FE_CHECK_NOTNULL(strided_slice_d_desc->GetInputDescPtr(0));
  std::vector<int64_t> ori_shape = strided_slice_d_desc->GetInputDescPtr(0)->GetOriginShape().GetDims();
  std::vector<int64_t> begins;
  (void)ge::AttrUtils::GetListInt(strided_slice_d_desc, kAttrBegin, begins);
  std::vector<int64_t> ends;
  (void)ge::AttrUtils::GetListInt(strided_slice_d_desc, kAttrEnd, ends);
  std::vector<int64_t> strides;
  (void)ge::AttrUtils::GetListInt(strided_slice_d_desc, kAttrStrides, strides);
  if (begins.size() != ends.size() || begins.size() != strides.size() || ends.size() != strides.size() ||
      begins.size() > ori_shape.size()) {
    FE_LOGW("Node[%s]: the sizes of begin[%zu], end[%zu], strides[%zu] are not equal or exceed the size of shape[%zu].",
            strided_slice_d_desc->GetName().c_str(), begins.size(), ends.size(), strides.size(), ori_shape.size());
    return false;
  }
  // continuously slice check
  for (auto &stride : strides) {
    if (stride != 1) {
      FE_LOGD("Node[%s]: Continuous slice check failed.", strided_slice_d_desc->GetName().c_str());
      return false;
    }
  }
  if (!CheckEvenlySlice(strided_slice_d_desc, ori_shape, begins, ends, split_info)) {
    return false;
  }
  return true;
}

bool StridedSliceDToSplitFusionPass::CheckSplitDSupported(const ge::OpDescPtr &strided_slice_d_desc,
                                                          const SplitInfo &split_info) const {
  ge::OpDescPtr split_d_desc = ge::AttrUtils::CopyOpDesc(strided_slice_d_desc);
  if (split_d_desc == nullptr) {
    return false;
  }
  split_d_desc->SetName(strided_slice_d_desc->GetName() + "_to_split_d");
  split_d_desc->SetType(SPLITD);
  (void)ge::AttrUtils::SetInt(split_d_desc, kSplitDim, split_info.split_dim);
  (void)ge::AttrUtils::SetInt(split_d_desc, kNumSplit, split_info.num_split);
  if (!CheckOpSupported(split_d_desc)) {
    FE_LOGD("Node[%s]: is not supported by AI Core.", split_d_desc->GetName().c_str());
    return false;
  }
  return true;
}

Status StridedSliceDToSplitFusionPass::UpdateGraph(ge::ComputeGraph &graph, ge::NodePtr &strided_slice_d_node,
                                                   ge::NodePtr &split_d_node, const SplitInfo &split_info) const {
  FusionTurbo fusion_turbo(graph);
  ge::OpDescPtr strided_slice_d_desc = strided_slice_d_node->GetOpDesc();
  std::string split_d_name = strided_slice_d_desc->GetName() + "_to_split_d";
  split_d_node = fusion_turbo.AddNodeOnly(split_d_name, SPLITD, split_info.num_split);
  FE_CHECK_NOTNULL(split_d_node);
  ge::OpDescPtr split_d_desc = split_d_node->GetOpDesc();
  FE_CHECK_NOTNULL(split_d_desc);
  (void)ge::AttrUtils::SetInt(split_d_desc, kSplitDim, split_info.split_dim);
  (void)ge::AttrUtils::SetInt(split_d_desc, kNumSplit, split_info.num_split);
  Relations relations_input = {{0, {strided_slice_d_node, 0, PEER}}};
  if (fusion_turbo.LinkInput(relations_input, split_d_node, UPDATE_THIS) != SUCCESS) {
    FE_LOGE("Node[%s]: failed to link input.", split_d_name.c_str());
    return FAILED;
  }
  FE_LOGD("Node[%s]: the split_out_anchor is %ld.", split_d_name.c_str(), split_info.split_out_anchor);
  if (strided_slice_d_node->GetAllOutDataAnchorsSize() > 0) {
    auto out_ahcor = strided_slice_d_node->GetOutDataAnchor(0);
    FE_CHECK_NOTNULL(out_ahcor);
    auto peer_in_anchors = out_ahcor->GetPeerInDataAnchors();
    if (!peer_in_anchors.empty()) {
      Relations relations_output = {{static_cast<int32_t>(split_info.split_out_anchor),
                                     {strided_slice_d_node, 0, PEER}}};
      if (fusion_turbo.LinkOutput(relations_output, split_d_node, UPDATE_THIS) != SUCCESS) {
        FE_LOGE("Node[%s]: failed to link output.", split_d_name.c_str());
        return FAILED;
      }
    }
  }
  auto split_d_out_tensor_desc = split_d_desc->MutableOutputDesc(split_info.split_out_anchor);
  FE_CHECK_NOTNULL(split_d_out_tensor_desc);
  for (int64_t i = 0; i < split_info.num_split; ++i) {
    if (i == split_info.split_out_anchor) {
      continue;
    }
    if (split_d_desc->UpdateOutputDesc(static_cast<uint32_t>(i), *split_d_out_tensor_desc) != ge::GRAPH_SUCCESS) {
      FE_LOGE("Node[%s]: failed to update output tensor with ID %ld.", split_d_name.c_str(), i);
      return FAILED;
    }
  }
  if (fusion_turbo.RemoveNodeOnly(strided_slice_d_node) != SUCCESS) {
    FE_LOGW("Node [%s]: failed to remove node.", strided_slice_d_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status StridedSliceDToSplitFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                              vector<ge::NodePtr> &new_nodes) {
  ge::NodePtr strided_slice_d_node = GetNodeFromMapping(kPatternStridedSliceD, mapping);
  FE_CHECK_NOTNULL(strided_slice_d_node);
  FE_LOGD("Node[%s]: start to do StridedSliceDToSplitFusionPass.", strided_slice_d_node->GetName().c_str());
  ge::OpDescPtr strided_slice_d_desc = strided_slice_d_node->GetOpDesc();
  FE_CHECK_NOTNULL(strided_slice_d_desc);
  SplitInfo split_info = {-1, -1, -1};
  if (!CheckCommonCondition(strided_slice_d_node, strided_slice_d_desc, split_info)) {
    return NOT_CHANGED;
  }
  if (!CheckSplitDSupported(strided_slice_d_desc, split_info)) {
    return NOT_CHANGED;
  }
  ge::NodePtr split_d_node = nullptr;
  if (UpdateGraph(graph, strided_slice_d_node, split_d_node, split_info) != SUCCESS) {
    return FAILED;
  }
  FE_CHECK_NOTNULL(split_d_node);
  new_nodes.push_back(split_d_node);
  FE_LOGD("Node[%s]: ending StridedSliceDToSplitFusionPass.", strided_slice_d_node->GetName().c_str());
  return SUCCESS;
}
REG_PASS("StridedSliceDToSplitFusionPass", BUILT_IN_GRAPH_PASS,
         StridedSliceDToSplitFusionPass, SINGLE_SCENE_OPEN | FE_PASS);
}  // namespace fe
