/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/conv_concat_fusion_pass.h"
#include <map>
#include <string>
#include <vector>
#include "common/math_util.h"
#include "common/fe_context_utils.h"
#include "common/platform_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph/ge_context.h"

namespace fe {
namespace {
const std::unordered_set<std::string> kMaxPoolType = {"MaxPool", "MaxPoolV3", "Pooling"};

bool IsSwDataTypeQualified(ge::NodePtr &concat_node) {
  auto output0 = concat_node->GetOpDesc()->MutableOutputDesc(0);
  if (output0 == nullptr) {
    return false;
  }

  fe::PrecisionMode precision_mode;
  (void)FEContextUtils::GetPrecisionMode(precision_mode);
  if (precision_mode == fe::PrecisionMode::ENUM_FORCE_FP32 ||
      (precision_mode == fe::PrecisionMode::ENUM_MUST_KEEP_ORIGIN_DTYPE && output0->GetDataType() == ge::DT_FLOAT)) {
    return false;
  }

  return true;
}
}  // namespace

Status ConvConcatFusionPass::DoFusion(ge::ComputeGraph &graph, ge::NodePtr &node_ptr,
                                      vector<ge::NodePtr> &fusion_nodes) {
  (void)fusion_nodes;
  string node_name = node_ptr->GetName();
  FE_LOGD("Node[%s]: start to ConvConcatFusionPass.", node_name.c_str());
  // 1. match the pattern
  std::vector<ge::NodePtr> mish_nodes_vec = {};
  if (MatchPattern(node_ptr, mish_nodes_vec) != SUCCESS) {
    FE_LOGD("Node[%s]: Match not successfully, return NOT_CHANGED.", node_name.c_str());
    return NOT_CHANGED;
  }
  OpKernelInfoPtr op_kernel_info_ptr = nullptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, STRIDEDWRITE);
  if (op_kernel_info_ptr == nullptr) {
    (void)ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), kSupportStridedwriteOptimize, true);
  }

  bool has_pooling = false;
  for (auto input_node : node_ptr->GetInAllNodes()) {
    if (input_node->GetType() == POOLING) {
      has_pooling = true;
      break;
    }
  }
  vector <ge::OpDescPtr> stride_write_op_desc_ptr_vec = {};
  bool sw_dtype_qualified = IsSwDataTypeQualified(node_ptr);
  if (op_kernel_info_ptr != nullptr && sw_dtype_qualified &&
      InsertStrideWrite(graph, node_ptr, stride_write_op_desc_ptr_vec) != SUCCESS) {
    FE_LOGD("Node[%s]: Failed to insert stride write nodes.", node_name.c_str());
    return FAILED;
  }
  // if have quant after concat, move quant before concat
  // case1 : stridewrite-->concat-->quant to quant-->stridewrite-->concat;
  // case2 : concat-->quant to quant-->concat;
  Status ret = DoQuantFusion(graph, node_ptr, has_pooling);
  if (ret == FAILED) {
    REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][DoQuantFus] Failed to fuse node: %s.", node_ptr->GetName().c_str());
    return FAILED;
  }

  // if have mish before concat, move mish after concat
  // case1 : mish-->concat to concat-->mish;
  // case2 : mish-->stridewrite-->concat to stridewrite-->concat-->mish;
  if (ret == NOT_CHANGED && DoMishFusion(graph, node_ptr, mish_nodes_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][DoMishFus] fuse node:%s failed.", node_ptr->GetName().c_str());
    return FAILED;
  }

  // set the stride attribute of StrideWrite and StrideRead
  for (ge::OpDescPtr stride_write_op_desc_ptr : stride_write_op_desc_ptr_vec) {
    FE_CHECK_NOTNULL(node_ptr->GetOpDesc()->MutableOutputDesc(0));
    (void)ge::AttrUtils::SetInt(stride_write_op_desc_ptr, STRIDE_ATTR_STRIDE,
                                node_ptr->GetOpDesc()->MutableOutputDesc(0)->MutableShape().GetDims()[1]);
  }

  FE_LOGD("Node [%s]: end of ConvConcatFusionPass.", node_name.c_str());
  return SUCCESS;
}

Status ConvConcatFusionPass::DoMishFusion(ge::ComputeGraph &graph, ge::NodePtr &concat_node,
                                          std::vector<ge::NodePtr> &mish_nodes_vec) {
  if (PlatformUtils::Instance().GetCubeVecState() != CubeVecStateNew::CUBE_VEC_SPLIT) {
    FE_LOGD("This is not cube-vec-split stage. No need to do mish fusion.");
    return SUCCESS;
  }
  if (mish_nodes_vec.empty()) {
    FE_LOGD("Do not have mish node. No need to do mish fusion.");
    return SUCCESS;
  }

  // add new mish node after concat
  ge::NodePtr mish_node = mish_nodes_vec[0];
  FE_CHECK_NOTNULL(mish_node);
  ge::OpDescPtr new_mish = ge::AttrUtils::CopyOpDesc(mish_node->GetOpDesc());
  FE_CHECK_NOTNULL(new_mish);
  new_mish->SetName(new_mish->GetName() + "_new");
  auto new_mish_node = graph.AddNode(new_mish);
  FE_CHECK_NOTNULL(new_mish_node);

  ge::OutDataAnchorPtr concat_out_anchor = concat_node->GetOutDataAnchor(0);
  FE_CHECK_NOTNULL(concat_out_anchor);
  auto concat_out_peer_in_anchors = concat_out_anchor->GetPeerInDataAnchors();
  for (size_t i = 0; i < concat_out_peer_in_anchors.size(); ++i) {
    FE_CHECK_NOTNULL(concat_out_peer_in_anchors.at(i));
    if (InsertNode(concat_out_anchor, concat_out_peer_in_anchors.at(i), new_mish_node, ge::DataType::DT_UNDEFINED)
        != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][DoMishFus] Failed to add node [%s] from node [%s] to peer in anchor %zu.",
                      new_mish_node->GetType().c_str(), concat_node->GetName().c_str(), i);
      return FAILED;
    }
  }

  // remove old mish node
  for (auto &cur_mish_node : mish_nodes_vec) {
    FE_CHECK_NOTNULL(cur_mish_node);
    if (graph.RemoveNode(cur_mish_node) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][DoMishFus] Failed to remove node [%s] in graph [%s].",
                      cur_mish_node->GetName().c_str(), graph.GetName().c_str());
      return FAILED;
    }
  }
  mish_nodes_vec.clear();
  return SUCCESS;
}

Status ConvConcatFusionPass::IsQuantNodeSame(const ge::NodePtr quant_node,
                                             const ge::OutDataAnchor::Vistor<ge::InDataAnchorPtr> &in_anchors) {
  if (quant_node->GetType() != QUANT) {
    FE_LOGW("Do not match quant after concat.");
    return NOT_CHANGED;
  }
  float scale = -1;
  float offset = -1;
  ge::DataType data_type = quant_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  if (!ge::AttrUtils::GetFloat(quant_node->GetOpDesc(), ATTR_SCALE, scale)) {
    REPORT_FE_ERROR("Failed to get scale attribute for quantization node!");
    return FAILED;
  }
  if (!ge::AttrUtils::GetFloat(quant_node->GetOpDesc(), ATTR_OFFSET, offset)) {
    REPORT_FE_ERROR("Failed to get offset attribute for quantization node!");
    return FAILED;
  }
  for (size_t j = 1; j < in_anchors.size(); ++j) {
    auto in_anchor = in_anchors.at(j);
    FE_CHECK_NOTNULL(in_anchor);
    ge::NodePtr next_node = in_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(next_node);
    if ((next_node->GetType() != QUANT)) {
      FE_LOGW("Do not match quant after concat.");
      return NOT_CHANGED;
    }
    float current_scale = -1;
    float current_offset = -1;
    ge::DataType current_data_type = next_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    if (!ge::AttrUtils::GetFloat(next_node->GetOpDesc(), ATTR_SCALE, current_scale)) {
      REPORT_FE_ERROR("Failed to get scale attribute for quantization node!");
      return FAILED;
    }
    if (!ge::AttrUtils::GetFloat(next_node->GetOpDesc(), ATTR_OFFSET, current_offset)) {
      REPORT_FE_ERROR("Failed to get offset attribute for quantization node!");
      return FAILED;
    }
    if (!FloatEqual(current_scale, scale) ||
        !FloatEqual(current_offset, offset) ||
        data_type != current_data_type) {
      FE_LOGW(
          "current quant sale or offset attr is different with the "
          "other quant, do not fuse quant.");
      return NOT_CHANGED;
    }
  }
  return SUCCESS;
}


Status ConvConcatFusionPass::InsertStrideWrite(ge::ComputeGraph &graph, const ge::NodePtr &node_ptr,
                                               vector<ge::OpDescPtr> &stride_write_op_desc_ptr_vec) {
  // 1. get attr value of concat_dim
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  int64_t concat_dim_attr_value = GetDimAttrValue(op_desc_ptr, CONCAT_DIM, true);
  FE_LOGD("Node[%s]: the concat_dim_attr_value is [%ld].", node_name.c_str(), concat_dim_attr_value);

  // 2. set the attribute of Concat
  FE_LOGD("Node[%s]: set the attribute of Concat.", node_name.c_str());
  SetGeAttrForConcat(op_desc_ptr, 1);

  // 3. insert stride_write op
  FE_LOGD("Node[%s]: insert StrideWrite between the Concat and the previous op.", node_name.c_str());
  ge::Node::Vistor<ge::InDataAnchorPtr> all_in_data_anchors = node_ptr->GetAllInDataAnchors();
  for (size_t i = 0; i != all_in_data_anchors.size(); ++i) {
    ge::InDataAnchorPtr in_data_anchor_ptr = all_in_data_anchors.at(i);
    FE_CHECK_NOTNULL(in_data_anchor_ptr);
    ge::OutDataAnchorPtr peer_out_data_anchor_ptr = in_data_anchor_ptr->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(peer_out_data_anchor_ptr);
    ge::NodePtr pre_node_ptr = peer_out_data_anchor_ptr->GetOwnerNode();
    FE_CHECK_NOTNULL(pre_node_ptr);

    // 4.1 create stride write op
    ge::OpDescPtr stride_write_op_desc_ptr;
    CreateStridedWrite(pre_node_ptr, stride_write_op_desc_ptr);

    // 4.2 insert stride write op
    ge::NodePtr stride_write_node = graph.AddNode(stride_write_op_desc_ptr);
    FE_CHECK_NOTNULL(stride_write_node);
    InsertNode(peer_out_data_anchor_ptr, in_data_anchor_ptr, stride_write_node);

    // 4.3 compute the stride
    stride_write_op_desc_ptr_vec.push_back(stride_write_op_desc_ptr);

    // 4.4 update the input desc of concat
    auto input_desc_ptr = node_ptr->GetOpDesc()->MutableInputDesc(i);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    auto ori_data_type = node_ptr->GetOpDesc()->GetInputDesc(i).GetOriginDataType();
    (void)GetNC1HWC0Shape(input_desc_ptr, ori_data_type);

    // 4.5 check stride_write peer out node is maxpool and set pre_node attr
    auto stride_write_in_anchor = stride_write_node->GetInDataAnchor(0);
    FE_CHECK_NOTNULL(stride_write_in_anchor);
    auto peer_out_data_anchor = stride_write_in_anchor->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(peer_out_data_anchor);
    auto peer_out_node = peer_out_data_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(peer_out_node);
    if (kMaxPoolType.count(peer_out_node->GetType()) > 0) {
      FE_LOGD("peer out node[%s, %s] is maxpool, do not set pre_node attr.",
              peer_out_node->GetName().c_str(), peer_out_node->GetType().c_str());
    } else {
      (void)SetPreNodeAttr(stride_write_node);
    }
  }

  // 4.6 update the output desc of concat
  auto ori_data_type = node_ptr->GetOpDesc()->GetOutputDesc(0).GetOriginDataType();
  (void)GetNC1HWC0Shape(node_ptr->GetOpDesc()->MutableOutputDesc(0), ori_data_type);
  // set the stride attribute of StrideWrite and StrideRead
  int64_t stride = 0;
  for (ge::OpDescPtr &stride_write_op_desc_ptr : stride_write_op_desc_ptr_vec) {
    FE_CHECK_NOTNULL(node_ptr->GetOpDesc()->MutableOutputDesc(0));
    if (node_ptr->GetOpDesc()->MutableOutputDesc(0)->MutableShape().GetDims().size() < DIM_SIZE) {
      REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][InsStrdWrt] ConcatD output dim less than 2.");
      return FAILED;
    }
    FE_CHECK_NOTNULL(node_ptr->GetOpDesc()->MutableOutputDesc(0));
    stride = node_ptr->GetOpDesc()->MutableOutputDesc(0)->MutableShape().GetDims()[1];
    (void)ge::AttrUtils::SetInt(stride_write_op_desc_ptr, STRIDE_ATTR_STRIDE, stride);
  }
  FE_LOGD("Node[%s]: set the attribute of StrideWrite, the stride is [%ld].", node_name.c_str(), stride);
  return SUCCESS;
}

Status ConvConcatFusionPass::SetPreNodeAttr(const ge::NodePtr &node_ptr) const {
  FE_LOGD("Node[%s] start to set attr inherit dtype from predecessor.", node_ptr->GetName().c_str());
  if ((node_ptr->GetType() != CONV2D) && (node_ptr->GetType() != CONV2D_COMPRESS)) {
    if (!ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), kAttrInheritDtypeFromPredecessor, true)) {
      FE_LOGW("Failed to set the inherit dtype attribute for node [%s] from its predecessor.", node_ptr->GetName().c_str());
    } else {
      FE_LOGD("Node[%s] set attr inherit dtype from predecessor successfully.", node_ptr->GetName().c_str());
    }
    ge::Node::Vistor<ge::InDataAnchorPtr> all_in_data_anchors = node_ptr->GetAllInDataAnchors();
    if (all_in_data_anchors.empty()) {
      return SUCCESS;
    }
    ge::InDataAnchorPtr in_data_anchor_ptr = all_in_data_anchors.at(0);
    FE_CHECK_NOTNULL(in_data_anchor_ptr);
    ge::OutDataAnchorPtr peer_out_data_anchor_ptr = in_data_anchor_ptr->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(peer_out_data_anchor_ptr);
    ge::NodePtr pre_node_ptr = peer_out_data_anchor_ptr->GetOwnerNode();
    FE_CHECK_NOTNULL(pre_node_ptr);
    (void)SetPreNodeAttr(pre_node_ptr);
  }
  return SUCCESS;
}

Status ConvConcatFusionPass::DoQuantFusionByNode(ge::ComputeGraph &graph, ge::NodePtr concat_node,
                                                 ge::NodePtr quant_node) {
  // create new quant op and add to graph
  FE_CHECK(quant_node == nullptr || quant_node->GetOpDesc()== nullptr,
    FE_LOGD("Get quant op desc unsuccessful."), return FAILED);
  ge::DataType quant_data_type = quant_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  for (size_t i = 0; i < concat_node->GetAllInDataAnchors().size(); ++i) {
    auto in_anchor = concat_node->GetInDataAnchor(i);
    FE_CHECK_NOTNULL(in_anchor);
    FE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
    auto pre_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    FE_CHECK_NOTNULL(pre_node);

    ge::OpDescPtr new_quant = ge::AttrUtils::CopyOpDesc(quant_node->GetOpDesc());
    new_quant->SetName(quant_node->GetName() + "_" + std::to_string(i));
    auto new_quant_node = graph.AddNode(new_quant);
    FE_CHECK_NOTNULL(new_quant_node);

    Status ret;
    if (pre_node->GetType() != STRIDEDWRITE) {
      ret = InsertNode(in_anchor->GetPeerOutAnchor(), in_anchor, new_quant_node, quant_data_type);
    } else {
      FE_CHECK_NOTNULL(pre_node->GetInDataAnchor(0));
      ret = InsertNode(pre_node->GetInDataAnchor(0)->GetPeerOutAnchor(), pre_node->GetInDataAnchor(0),
                       new_quant_node, quant_data_type);
    }
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][DoQuantFus] Failed to add node: %s.", new_quant_node->GetName().c_str());
      return FAILED;
    }
    auto input_desc_ptr = concat_node->GetOpDesc()->MutableInputDesc(i);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    GetNC1HWC0Shape(input_desc_ptr, quant_data_type);
    JudgeOp(pre_node);
  }
  GetNC1HWC0Shape(concat_node->GetOpDesc()->MutableOutputDesc(0), quant_data_type);
  // remove old quant op
  ge::OutDataAnchorPtr out_data_anchor = concat_node->GetOutDataAnchor(0);
  FE_CHECK_NOTNULL(out_data_anchor);
  vector<ge::NodePtr> quant_nodes;
  for (size_t i = 0; i < out_data_anchor->GetPeerInDataAnchors().size(); ++i) {
    ge::NodePtr current_quan_node = out_data_anchor->GetPeerInDataAnchors().at(i)->GetOwnerNode();
    FE_CHECK_NOTNULL(current_quan_node);
    quant_nodes.push_back(current_quan_node);
  }

  for (const ge::NodePtr &peer_quant_node : quant_nodes) {
    FE_LOGD("RemoveNode [%s, %s].", peer_quant_node->GetNamePtr(), peer_quant_node->GetTypePtr());
    if (graph.RemoveNode(peer_quant_node)) {
      REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][DoQuantFus] Failed to remove node %s.",
                      peer_quant_node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status ConvConcatFusionPass::DoQuantFusion(ge::ComputeGraph &graph, ge::NodePtr concat_node, const bool &has_pooling) {
  if (concat_optimize_checker.IsDimCAlignedWithQuant(concat_node) && !has_pooling) {
    for (size_t i = 0; i < concat_node->GetAllOutDataAnchors().size(); i++) {
      // match and check quant
      ge::OutDataAnchorPtr out_data_anchor = concat_node->GetOutDataAnchor(i);
      FE_CHECK_NOTNULL(out_data_anchor);
      auto in_anchors = out_data_anchor->GetPeerInDataAnchors();
      if (in_anchors.size() < 1) {
        FE_LOGW("[GraphOpt][ConvConcatFus][DoQuantFus] node[%s]'s peer_in_anchor size less than one.",
                        concat_node->GetName().c_str());
        return NOT_CHANGED;
      }
      FE_CHECK_NOTNULL(in_anchors.at(0));
      ge::NodePtr quant_node = in_anchors.at(0)->GetOwnerNode();
      FE_CHECK_NOTNULL(quant_node);
      // check all quant node is same or not
      Status ret = IsQuantNodeSame(quant_node, in_anchors);
      if (ret != SUCCESS) {
        return NOT_CHANGED;
      }
      // fusion quant ,move quant before stridewrite
      ret = DoQuantFusionByNode(graph, concat_node, quant_node);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][ConvConcatFus][DoQuantFus] Failed to fuse node: %s.", quant_node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status ConvConcatFusionPass::MatchPattern(const ge::NodePtr &node_ptr, std::vector<ge::NodePtr> &mish_nodes_vec) {
  string node_name = node_ptr->GetName().c_str();
  // 1. check the concat node
  if (!concat_optimize_checker.Check(node_ptr)) {
    FE_LOGD("Node[%s]: failed to check the concat node.", node_name.c_str());
    return FAILED;
  };
  // 2. check the graph, dequant or not
  bool is_dequant = false;
  ge::Node::Vistor<ge::InDataAnchorPtr> all_in_data_anchors = node_ptr->GetAllInDataAnchors();
  for (size_t i = 0; i != all_in_data_anchors.size(); ++i) {
    ge::InDataAnchorPtr in_data_anchor_ptr = all_in_data_anchors.at(i);
    FE_CHECK_NOTNULL(in_data_anchor_ptr);
    ge::OutDataAnchorPtr peer_out_data_anchor_ptr = in_data_anchor_ptr->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(peer_out_data_anchor_ptr);
    ge::NodePtr pre_node_ptr = peer_out_data_anchor_ptr->GetOwnerNode();
    FE_CHECK_NOTNULL(pre_node_ptr);
    if (IsConvAndExpcetOp(pre_node_ptr, DEQUANT) == SUCCESS || IsDequantElemwise(pre_node_ptr) == SUCCESS) {
      is_dequant = true;
      break;
    }
  }  // 3. match the no-dequant graph
  if (!is_dequant) {
    return MatchForNoDequuant(node_ptr, mish_nodes_vec);
  } else {
    // 3. match the dequant graph
    return MatchForDequant(node_ptr, mish_nodes_vec);
  }
}

// all inputs: conv2d+concat
// all inputs: conv2d+relu+concat
// all inputs: conv2d+leakyrelu+concat
Status ConvConcatFusionPass::MatchForNoDequuant(const ge::NodePtr &node_ptr, std::vector<ge::NodePtr> &mish_nodes_vec) {
  ge::Node::Vistor<ge::InDataAnchorPtr> all_in_data_anchors = node_ptr->GetAllInDataAnchors();
  ge::InDataAnchorPtr in_data_anchor_ptr0 = all_in_data_anchors.at(0);
  FE_CHECK_NOTNULL(in_data_anchor_ptr0);
  ge::OutDataAnchorPtr peer_out_data_anchor_ptr0 = in_data_anchor_ptr0->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(peer_out_data_anchor_ptr0);
  ge::NodePtr pre_node_ptr0 = peer_out_data_anchor_ptr0->GetOwnerNode();
  FE_CHECK_NOTNULL(pre_node_ptr0);
  ConvConcatFusionPattern match_pattern = GetMatchPattern(pre_node_ptr0);
  if (match_pattern == UN_SUPPORTED) {
    FE_LOGD("Node[%s]: failed to check the no-dequant graph.", node_ptr->GetName().c_str());
    return FAILED;
  }
  if (match_pattern == PATTERN_CONV2D_MISH_CONCAT) {
    mish_nodes_vec.push_back(pre_node_ptr0);
  }

  for (size_t i = 1; i != all_in_data_anchors.size(); ++i) {
    ge::InDataAnchorPtr in_data_anchor_ptr = all_in_data_anchors.at(i);
    FE_CHECK_NOTNULL(in_data_anchor_ptr);
    ge::OutDataAnchorPtr peer_out_data_anchor_ptr = in_data_anchor_ptr->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(peer_out_data_anchor_ptr);
    ge::NodePtr pre_node_ptr = peer_out_data_anchor_ptr->GetOwnerNode();
    FE_CHECK_NOTNULL(pre_node_ptr);
    Status is_max_pool = IsMaxPool(pre_node_ptr);
    ConvConcatFusionPattern pre_node_match_pattern = GetMatchPattern(pre_node_ptr);
    bool fail_pattern1 =
        match_pattern == PATTERN_CONV2D_CONCAT && (IsConv(pre_node_ptr) != SUCCESS && is_max_pool != SUCCESS);
    bool fail_pattern2 = match_pattern == PATTERN_CONV2D_RELU_CONCAT &&
                         (IsConvAndExpcetOp(pre_node_ptr, RELU) != SUCCESS && is_max_pool != SUCCESS);
    bool fail_pattern3 = match_pattern == PATTERN_CONV2D_LEAKYRELU_CONCAT &&
                         (IsLeakyRelu(pre_node_ptr) != SUCCESS && is_max_pool != SUCCESS);
    bool fail_pattern4 = match_pattern == PATTERN_MAXPOOL_CONCAT && (pre_node_match_pattern == UN_SUPPORTED);
    bool fail_pattern5 =
        match_pattern == PATTERN_CONV2D_REQUANT_CONCAT && (pre_node_match_pattern != PATTERN_CONV2D_REQUANT_CONCAT);
    bool fail_pattern6 =
        match_pattern == PATTERN_CONV2D_MISH_CONCAT && pre_node_match_pattern != PATTERN_CONV2D_MISH_CONCAT;
    bool fail_flag = fail_pattern1 || fail_pattern2 || fail_pattern3 || fail_pattern4 || fail_pattern5 || fail_pattern6;
    if (fail_flag) {
      FE_LOGD("Node[%s]: failed to check the no-dequant graph.", node_ptr->GetName().c_str());
      return FAILED;
    }
    if (match_pattern == PATTERN_CONV2D_MISH_CONCAT) {
      mish_nodes_vec.push_back(pre_node_ptr);
    }
  }
  return SUCCESS;
}

// some inputs: conv2d+dequant, other inputs: conv2d/conv2d+relu/conv2d+leakyrelu/conv2d+dequant
Status ConvConcatFusionPass::MatchForDequant(const ge::NodePtr &node_ptr,
                                             std::vector<ge::NodePtr> &mish_nodes_vec) const {
  ge::Node::Vistor<ge::InDataAnchorPtr> all_in_data_anchors = node_ptr->GetAllInDataAnchors();
  for (size_t i = 0; i != all_in_data_anchors.size(); ++i) {
    ge::InDataAnchorPtr in_data_anchor_ptr = all_in_data_anchors.at(i);
    FE_CHECK_NOTNULL(in_data_anchor_ptr);
    ge::OutDataAnchorPtr peer_out_data_anchor_ptr = in_data_anchor_ptr->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(peer_out_data_anchor_ptr);
    ge::NodePtr pre_node_ptr = peer_out_data_anchor_ptr->GetOwnerNode();
    FE_CHECK_NOTNULL(pre_node_ptr);
    ConvConcatFusionPattern pattern = GetMatchPattern(pre_node_ptr);
    if (pattern == UN_SUPPORTED) {
      FE_LOGD("Node[%s]: Failed to check the dequantization graph.", node_ptr->GetName().c_str());
      return FAILED;
    }
    if (pre_node_ptr->GetType() == MISH) {
      mish_nodes_vec.push_back(pre_node_ptr);
    }
  }
  return SUCCESS;
}

ConvConcatFusionPattern ConvConcatFusionPass::GetMatchPattern(const ge::NodePtr &pre_node_ptr) const {
  if (UnknownShapeUtils::IsUnknownShapeOp(*pre_node_ptr->GetOpDesc())) {
    FE_LOGD("node[%s, %s] is dynamic is not be support", pre_node_ptr->GetNamePtr(), pre_node_ptr->GetTypePtr());
    return UN_SUPPORTED;
  } else if (IsConv(pre_node_ptr) == SUCCESS) {
    return PATTERN_CONV2D_CONCAT;
  } else if (IsConvAndExpcetOp(pre_node_ptr, RELU) == SUCCESS) {
    return PATTERN_CONV2D_RELU_CONCAT;
  } else if (IsLeakyRelu(pre_node_ptr) == SUCCESS) {
    return PATTERN_CONV2D_LEAKYRELU_CONCAT;
  } else if (IsConvAndExpcetOp(pre_node_ptr, DEQUANT) == SUCCESS) {
    return PATTERN_CONV2D_DEQUANT_CONCAT;
  } else if (IsConvAndExpcetOp(pre_node_ptr, REQUANT) == SUCCESS) {
    return PATTERN_CONV2D_REQUANT_CONCAT;
  } else if (IsMaxPool(pre_node_ptr) == SUCCESS) {
    return PATTERN_MAXPOOL_CONCAT;
  } else if (IsDequantElemwise(pre_node_ptr) == SUCCESS) {
    return PATTERN_CONV2D_DEQUANT_LEAKYRELU_CONCAT;
  } else if (IsConvAndExpcetOp(pre_node_ptr, MISH) == SUCCESS) {
    return PATTERN_CONV2D_MISH_CONCAT;
  }
  return UN_SUPPORTED;
}

Status ConvConcatFusionPass::IsConv(const ge::NodePtr &pre_node_ptr) const {
  return (pre_node_ptr->GetType() == CONV2D || pre_node_ptr->GetType() == CONV2D_COMPRESS) &&
                 is_single_out_and_ref(pre_node_ptr)
             ? SUCCESS
             : FAILED;
}

Status ConvConcatFusionPass::IsMaxPool(const ge::NodePtr &pre_node_ptr) const {
  if (pre_node_ptr->GetType() == POOLING) {
    int64_t mode = 0;
    (void)ge::AttrUtils::GetInt(pre_node_ptr->GetOpDesc(), "mode", mode);
    if (mode == 1) {
      FE_LOGD("mode is 1, not fuse.");
      return FAILED;
    }
  }
  bool is_pool = pre_node_ptr->GetType() == MAXPOOL ||
                 pre_node_ptr->GetType() == MAXPOOLV3 ||
                 pre_node_ptr->GetType() == POOLING;
  return (is_pool) && is_single_out_and_ref(pre_node_ptr) ? SUCCESS : FAILED;
}

Status ConvConcatFusionPass::IsConvAndExpcetOp(const ge::NodePtr &pre_node_ptr,
                                               const string &expect_op_type) const {
  if (pre_node_ptr->GetType() == expect_op_type && is_single_out_and_ref(pre_node_ptr)) {
    ge::NodePtr conv2d_node_ptr;
    Status status = NodeOptimizeUtils::GetPreNode(pre_node_ptr, 0, conv2d_node_ptr);
    if (status != SUCCESS) {
      FE_LOGD("Node[%s]: failed to get previous node.", pre_node_ptr->GetName().c_str());
      return FAILED;
    }
    if ((conv2d_node_ptr->GetType() == CONV2D || conv2d_node_ptr->GetType() == CONV2D_COMPRESS) &&
        is_single_out_and_ref(conv2d_node_ptr)) {
      return SUCCESS;
    }
  }
  return FAILED;
}

Status ConvConcatFusionPass::IsDequantElemwise(const ge::NodePtr &pre_node_ptr) const {
  bool support_elemwise = pre_node_ptr->GetType() == LEAKYRELU ||
                          pre_node_ptr->GetType() == RELU ||
                          (pre_node_ptr->GetType() == MISH);
  if (support_elemwise && is_single_out_and_ref(pre_node_ptr)) {
    ge::NodePtr dequant_node_ptr;
    Status status = NodeOptimizeUtils::GetPreNode(pre_node_ptr, 0, dequant_node_ptr);
    if (status != SUCCESS) {
      FE_LOGD("Node[%s]: failed to get previous node.", pre_node_ptr->GetName().c_str());
      return FAILED;
    }
    status = IsConvAndExpcetOp(dequant_node_ptr, DEQUANT);
    if (status == SUCCESS) {
      FE_LOGD("Node[%s]: The node structure matches with dequant-leakyrelu-concat.", pre_node_ptr->GetName().c_str());
      return SUCCESS;
    }
  }
  return FAILED;
}

Status ConvConcatFusionPass::IsLeakyRelu(const ge::NodePtr &pre_node_ptr) const {
  Status status = IsConvAndExpcetOp(pre_node_ptr, LEAKYRELU);
  if (status != SUCCESS) {
    FE_LOGD("Node[%s]: the node is not leakyrelu.", pre_node_ptr->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

vector<string> ConvConcatFusionPass::GetNodeTypes() {
  vector<string> result;
  result.push_back(CONCATD);
  result.push_back(CONCATV2D);
  return result;
}

string ConvConcatFusionPass::GetPatternName() { return "ConvConcatDPass"; }

REG_PASS("ConvConcatFusionPass", BUILT_IN_GRAPH_PASS,
         ConvConcatFusionPass, SINGLE_SCENE_OPEN | FE_PASS);
}  // namespace fe
