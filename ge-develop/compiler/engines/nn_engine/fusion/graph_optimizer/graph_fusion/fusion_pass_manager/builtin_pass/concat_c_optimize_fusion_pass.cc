/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/concat_c_optimize_fusion_pass.h"
#include <memory>
#include <vector>
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "common/lxfusion_json_util.h"
#include "common/fe_graph_common.h"
#include "graph/anchor.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
using ToOpStructPtr = std::shared_ptr<fe::ToOpStruct_t>;
namespace fe {
namespace {
const int kMaxDeepth = 100;
const int64_t kAlignNum = 32;
const int64_t kCAxisIndex = 1;
const std::string kAttrConcatDim = "concat_dim";
const std::string kPatternConcat = "concat";
const std::string kPatternConv = "conv";
const std::string kPatternRelu = "relu";
const std::string kPatternQuant = "quant";
const std::string kPatternReluOrQuant = "relu_or_quant";
const std::string kOutputOffsetForBufferFusion = "_output_offset_for_buffer_fusion";
const std::string kCanReusedForConcatCOptimize = "can_reused_for_concat_optimize";

bool CheckMultiRef(const ge::InDataAnchorPtr &input_anchor) {
  auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return false;
  }
  return peer_out_anchor->GetPeerInDataAnchors().size() > 1;
}
} // namespace

ConcatCOptimizeFusionPass::ConcatCOptimizeFusionPass()
    : concat_dim_(-1),
      real_concat_dim_(-1) {}

ConcatCOptimizeFusionPass::~ConcatCOptimizeFusionPass() {}

vector<FusionPattern *> ConcatCOptimizeFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  /*
   * ================================pattern1===================================
   *       Conv2D/Conv2DTranspose/Conv2DTransposeD/FixPipe
   *                          |
   *                  ConcatD/ConcatV2D
   * ===========================================================================
   */
  FusionPattern *pattern = new (std::nothrow) FusionPattern("ConcatCOptimizeFusionPass");
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatCOptFus][DfnPtn] Failed to create a new object."),
           return patterns);
  pattern->AddOpDesc(kPatternConcat, {CONCATD, CONCATV2D})
      .SetOutput(kPatternConcat);
  patterns.push_back(pattern);
  return patterns;
}

vector<FusionPattern *> ConcatCOptimizeFusionPass::DefineInnerPatterns() {
  vector<FusionPattern *> patterns;
  /*
   * ================================pattern1===================================
   *       Conv2D/Conv2DTranspose/Conv2DTransposeD/FixPipe
   * ===========================================================================
   */
  FusionPattern *pattern1 = new (std::nothrow) FusionPattern("ConcatCOptimizeFusionPass1");
  FE_CHECK(pattern1 == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatCOptFus][DfnPtn] Failed to create a new object."),
           return patterns);
  pattern1->AddOpDesc(kPatternConv, {CONV2D, kConv2DTranspose, kConv2DTransposeD, OP_FIXPIPE})
      .SetOutput(kPatternConv);
  patterns.push_back(pattern1);

  /*
   * ================================pattern2===================================
   *       Conv2D/Conv2DTranspose/Conv2DTransposeD/FixPipe
   *                          |
   *                 Relu/Dequant/Requant
   * ===========================================================================
   */
  FusionPattern *pattern2 = new (std::nothrow) FusionPattern("ConcatCOptimizeFusionPass2");
  FE_CHECK(pattern2 == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatCOptFus][DfnPtn] Failed to create a new object."),
           return patterns);
  pattern2->AddOpDesc(kPatternConv, {CONV2D, kConv2DTranspose, kConv2DTransposeD, OP_FIXPIPE})
      .AddOpDesc(kPatternReluOrQuant, {RELU, ASCEND_DEQUANT, kAscendRequant})
      .SetInputs(kPatternReluOrQuant, {kPatternConv})
      .SetOutput(kPatternReluOrQuant);
  patterns.push_back(pattern2);

  /*
   * ================================pattern3===================================
   *       Conv2D/Conv2DTranspose/Conv2DTransposeD/FixPipe
   *                          |
   *                       Dequant
   *                          |
   *                        Relu
   * ===========================================================================
   */
  FusionPattern *pattern3 = new (std::nothrow) FusionPattern("ConcatCOptimizeFusionPass3");
  FE_CHECK(pattern3 == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatCOptFus][DfnPtn] Failed to create a new object."),
           return patterns);
  pattern3->AddOpDesc(kPatternConv, {CONV2D, kConv2DTranspose, kConv2DTransposeD, OP_FIXPIPE})
      .AddOpDesc(kPatternQuant, {ASCEND_DEQUANT})
      .AddOpDesc(kPatternRelu, {RELU})
      .SetInputs(kPatternQuant, {kPatternConv})
      .SetInputs(kPatternRelu, {kPatternQuant})
      .SetOutput(kPatternRelu);
  patterns.push_back(pattern3);
  return patterns;
}

Status ConcatCOptimizeFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                          vector<ge::NodePtr> &fusion_nodes) {
  (void)fusion_nodes;
  ge::NodePtr concat_node = GetNodeFromMapping(kPatternConcat, mapping);
  FE_CHECK(concat_node == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatCOptFus][Fus] Concat node is nullptr."),
           return FAILED);
  Status ret = MatchInnerPatterns(concat_node);
  if (ret != SUCCESS) {
    return ret;
  }
  ge::OpDescPtr op_desc = concat_node->GetOpDesc();
  FE_CHECK(op_desc == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatCOptFus][Fus] Concat op desc is nullptr."),
           return FAILED);
  if (!CheckIsValidNode(concat_node)) {
    return NOT_CHANGED;
  }
  bool concat_c_optimize_flag = true;
  CheckPreNodeValid(concat_node, concat_c_optimize_flag);
  CheckIsLxSlicedOp(op_desc, concat_c_optimize_flag);
  (void)SetStrideWriteInfoForInputs(concat_node, concat_c_optimize_flag);
  CheckAndSetAttrForConcat(concat_node, op_desc, concat_c_optimize_flag);
  if (!ge::AttrUtils::HasAttr(graph, NEED_RE_PRECOMPILE) && concat_c_optimize_flag) {
    (void)ge::AttrUtils::SetBool(graph, NEED_RE_PRECOMPILE, true);
  }
  FE_LOGD("Node[%s]: concat_c_optimize_flag is %zu.", concat_node->GetName().c_str(), concat_c_optimize_flag);
  return SUCCESS;
}

Status ConcatCOptimizeFusionPass::MatchInnerPatterns(const ge::NodePtr &node) {
  if (node->GetInNodes().empty()) {
    FE_LOGD("Node[%s]: input node is empty.", node->GetName().c_str());
    return NOT_CHANGED;
  }
  const auto &inner_patterns = GetInnerPatterns();
  if (inner_patterns.empty()) {
    FE_LOGE("Node[%s]: inner pattern is empty.", node->GetName().c_str());
    return FAILED;
  }
  for (auto &in_node : node->GetInNodes()) {
    bool match_check_flag = false;
    for (auto &inner_pattern : inner_patterns) {
      bool type_check_flag = false;
      std::shared_ptr<FusionPattern::OpDesc> output_op_desc = inner_pattern->GetOutput();
      if (output_op_desc == nullptr) {
        continue;
      }
      for (auto &type : output_op_desc->types) {
        if (type == in_node->GetType()) {
          type_check_flag = true;
          break;
        }
      }
      if (!type_check_flag) {
        FE_LOGD("Node[%s]: does not meet the output of inner pattern!", in_node->GetName().c_str());
        continue;
      }
      Mapping mapping;
      if (MatchFromOutput(in_node, output_op_desc, mapping)) {
        match_check_flag = true;
        break;
      }
    }
    if (!match_check_flag) {
      FE_LOGD("Node[%s]: does not match the inner pattern!", in_node->GetName().c_str());
      return NOT_CHANGED;
    }
  }
  return SUCCESS;
}

bool ConcatCOptimizeFusionPass::CheckIsValidNode(const ge::NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  std::string op_name = node->GetName();
  if (!CheckConcatFormat(op_desc)) {
    FE_LOGD("Node[%s]: concat format is not NC1HWC0, which can not do c-axis optimize!", op_name.c_str());
    return false;
  }
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    FE_LOGD("Node[%s] is unknown shape op, which can not do c-axis optimize!", op_name.c_str());
    return false;
  }
  if (FeGraphCommon::IsNodeOfUnknownGraph(*node)) {
    FE_LOGD("Owner graph is dynamic graph, %s can not do c-axis optimize!", op_name.c_str());
    return false;
  }
  if (!CheckConcatDimAndAlignment(op_desc)) {
    FE_LOGD("Node[%s]: concat dim check is invalid, which can not do c-axis optimize!", op_name.c_str());
    return false;
  }
  if (!CheckInput(node)) {
    FE_LOGD("Node[%s]: input check is invalid, which can not do c-axis optimize.", op_name.c_str());
    return false;
  }

  if (!CheckOutput(node)) {
      FE_LOGD("Node[%s]: output check is invalid, which can not do c-axis optimize.", op_name.c_str());
    return false;
  }

  if (InvalidMemType(op_desc)) {
    FE_LOGD("Node[%s]: mem type check is invalid, which can not do c-axis optimize.", op_name.c_str());
    return false;
  }

  if (!CheckPeerNodeCanReUsed(node)) {
      FE_LOGD("Node[%s]: peer nodes cannot be reused, which can not do c-axis optimize.", op_name.c_str());
    return false;
  }
  return true;
}

bool ConcatCOptimizeFusionPass::CheckConcatFormat(const ge::OpDescPtr &op_desc) const {
  bool is_input_nc1hwc0_format = true;
  for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
    auto input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
    if (input_desc == nullptr) {
      continue;
    }
    if (ge::GetPrimaryFormat(input_desc->GetFormat()) != ge::FORMAT_NC1HWC0) {
      is_input_nc1hwc0_format = false;
      FE_LOGD("The input%zu's format is %d, which is not equal to NC1HWC0.",
              i, static_cast<int>(ge::GetPrimaryFormat(input_desc->GetFormat())));
      break;
    }
  }
  return is_input_nc1hwc0_format;
}

bool ConcatCOptimizeFusionPass::CheckConcatDimAndAlignment(const ge::OpDescPtr &op_desc) {
  if (!ge::AttrUtils::GetInt(op_desc, kAttrConcatDim, concat_dim_)) {
    FE_LOGW("Node[%s]: can not get concat dim.", op_desc->GetName().c_str());
    return false;
  }
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    ge::GeTensorDesc input_tensor = op_desc->GetInputDesc(i);
    ge::GeShape input_origin_ge_shape = input_tensor.GetOriginShape();
    ge::Format input_origin_format = input_tensor.GetOriginFormat();
    auto input_format = input_tensor.GetFormat();
    gert::Shape input_origin_shape;
    input_origin_shape.SetDimNum(input_origin_ge_shape.GetDimNum());
    for (size_t j = 0; j < input_origin_ge_shape.GetDimNum(); ++j) {
      input_origin_shape.SetDim(j, input_origin_ge_shape.GetDim(j));
    }
    int64_t reshape_type_mask = 0;
    (void)ge::AttrUtils::GetInt(input_tensor, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type_mask);
    transformer::TransferDimsInfo transfer_dims_info;
    transfer_dims_info.src_format = input_origin_format;
    transfer_dims_info.dst_format = input_format;
    transfer_dims_info.src_shape = input_origin_shape;
    transfer_dims_info.reshape_type_mask = reshape_type_mask;
    if (!CheckConcatDim(op_desc, transfer_dims_info)) {
      return false;
    }
    transformer::AlignShapeInfo align_shape_info;
    align_shape_info.src_format = input_origin_format;
    align_shape_info.dst_format = input_format;
    align_shape_info.src_shape = input_origin_shape;
    align_shape_info.data_type = input_tensor.GetDataType();
    align_shape_info.reshape_type_mask = reshape_type_mask;
    if (!CheckConcatDimAlignment(op_desc, align_shape_info)) {
      return false;
    }
  }
  return true;
}

bool ConcatCOptimizeFusionPass::CheckConcatDim(const ge::OpDescPtr &op_desc,
                                               const transformer::TransferDimsInfo &transfer_dims_info) {
  if (concat_dim_ < 0) {
    FE_LOGD("Concat_dim[%lld] is negative number, change it to positive.", concat_dim_);
    concat_dim_ = static_cast<int64_t>(transfer_dims_info.src_shape.GetDimNum()) + concat_dim_;
  }
  transformer::AxisIndexMapping axis_index_mapping;
  if (!transformer::TransferShapeUtils::TransferDims(transfer_dims_info, axis_index_mapping)) {
    FE_LOGW("Node[%s]: can not transfer dims.", op_desc->GetName().c_str());
    return false;
  }
  if (static_cast<size_t>(concat_dim_) >= axis_index_mapping.src_to_dst_transfer_dims.size() ||
    axis_index_mapping.src_to_dst_transfer_dims[concat_dim_].empty()) {
    FE_LOGW("Concat_dim %lld is out of range or its mapping is empty.", concat_dim_);
    return false;
  }
  real_concat_dim_ = axis_index_mapping.src_to_dst_transfer_dims[concat_dim_][0];
  FE_LOGD("Origin concat dim is %ld. Real concat dim is %ld.", concat_dim_, real_concat_dim_);
  return real_concat_dim_ == kCAxisIndex;
}

bool ConcatCOptimizeFusionPass::CheckConcatDimAlignment(const ge::OpDescPtr &op_desc,
                                                        const transformer::AlignShapeInfo &align_shape_info) const {
  gert::Shape aligned_shape;
  if (!transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape)) {
    FE_LOGW("Node[%s]: can not get aligned shape.", op_desc->GetName().c_str());
    return false;
  }
  if (aligned_shape.GetDimNum() != align_shape_info.src_shape.GetDimNum()) {
      FE_LOGW("Node[%s]: aligned shape size %zu is invalid.", op_desc->GetName().c_str(), aligned_shape.GetDimNum());
      return false;
  }
  if (static_cast<size_t>(concat_dim_) >= align_shape_info.src_shape.GetDimNum()) {
    FE_LOGW("Node[%s]: concat dim %lld exceeds the dimension of origin shape.", op_desc->GetName().c_str(),
            concat_dim_);
    return false;
  }
  if ((align_shape_info.src_shape[concat_dim_] % aligned_shape[concat_dim_]) != 0) {
    FE_LOGD("Node[%s]: concat dim can not meet the alignment %lld.", op_desc->GetName().c_str(),
            aligned_shape[concat_dim_]);
    return false;
  }
  return true;
}

Status ConcatCOptimizeFusionPass::CalcTensorSize(ge::GeTensorDesc &tensor_desc, int64_t &tensor_size,
                                                 int32_t &output_real_calc_flag) const {
  if (TensorComputeUtil::VerifyTensor(tensor_desc) != fe::SUCCESS) {
    FE_LOGD("Can not verify this tensor.");
    return fe::FAILED;
  }

  int64_t element_cnt;
  if (TensorComputeUtil::GetElementCountByMultiply(tensor_desc, element_cnt) != fe::SUCCESS) {
    FE_LOGD("Can not calculate tensor size.");
    return fe::FAILED;
  }
  ge::DataType data_type = tensor_desc.GetDataType();
  if (TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, tensor_size, output_real_calc_flag) !=
      fe::SUCCESS) {
    FE_LOGD("Can not get tensor size by element count and datatype.");
    return fe::FAILED;
  }
  return fe::SUCCESS;
}

bool ConcatCOptimizeFusionPass::CheckIs32Align(const ge::NodePtr &concat_node) const {
  for (size_t i = 0; i < concat_node->GetAllInDataAnchors().size(); i++) {
    int64_t tensor_size = 0;
    int32_t flag = 1;
    ge::GeTensorDesc tensor_desc = concat_node->GetOpDesc()->GetInputDesc(i);
    if (CalcTensorSize(tensor_desc, tensor_size, flag) != fe::SUCCESS) {
      FE_LOGD("Can not calculate input tensor size, %s can not optimize.",
          concat_node->GetName().c_str());
      return false;
    }
    if (tensor_size % kAlignNum != 0) {
      FE_LOGD("Input tensor size of concat can not be divided by 32, %s can not optimize.",
          concat_node->GetName().c_str());
      return false;
    }
  }
  return true;
}

void ConcatCOptimizeFusionPass::GetFirstOutAnchorNotInDeleteList(const ge::InDataAnchorPtr &input_anchor,
                                                        ge::OutDataAnchorPtr &src_anchor,
                                                        int current_deep) const {
  if (current_deep >= kMaxDeepth) {
    return;
  }
  auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
      return;
  }
  auto peer_node = peer_out_anchor->GetOwnerNode();
  if (peer_node == nullptr) {
      return;
  }
  if (kGeDeleteOpType.count(peer_node->GetType()) != 0) {
    auto in_anchor = peer_node->GetInDataAnchor(0);
    if (in_anchor == nullptr) {
      FE_LOGD("node:%s in data anchor of %d is nullptr.", peer_node->GetName().c_str(), 0);
      return;
    }
    GetFirstOutAnchorNotInDeleteList(in_anchor, src_anchor, current_deep + 1);
  } else {
    src_anchor = peer_out_anchor;
  }
  return;
}

bool ConcatCOptimizeFusionPass::CheckSameSrc(const ge::NodePtr &concat_node) const {
  set<ge::OutDataAnchorPtr> src_anchors;
  for (const ge::InDataAnchorPtr &input_anchor : concat_node->GetAllInDataAnchors()) {
    ge::OutDataAnchorPtr src_anchor = nullptr;
    GetFirstOutAnchorNotInDeleteList(input_anchor, src_anchor, 0);
    src_anchors.insert(src_anchor);
  }
  bool res = (src_anchors.size() != concat_node->GetAllInDataAnchors().size());
  FE_LOGD("Check concat has_same_input is[%d].", res);
  return res;
}

bool ConcatCOptimizeFusionPass::CheckControlEdge(const ge::NodePtr &concat_node) const {
  bool res = (concat_node->GetInControlNodes().size() != 0) || (concat_node->GetOutControlNodes().size() != 0);
  FE_LOGD("Check concat has_ctrl_edge is[%d].", res);
  return res;
}

bool ConcatCOptimizeFusionPass::CheckInput(const ge::NodePtr &concat_node) const {
  bool check32_align = CheckIs32Align(concat_node) || concat_node->GetAllInDataAnchorsSize() == 1;
  bool has_same_src = CheckSameSrc(concat_node);
  bool has_control_edge = CheckControlEdge(concat_node);

  return check32_align && !has_same_src && !has_control_edge;
}

bool ConcatCOptimizeFusionPass::CheckOutput(const ge::NodePtr &concat_node) const {
  for (auto output_anchor : concat_node->GetAllOutDataAnchors()) {
    for (size_t i = 0; i < output_anchor->GetPeerInDataAnchors().size(); i++) {
      auto peerAnchor = output_anchor->GetPeerInDataAnchors().at(i);
      FE_CHECK(peerAnchor == nullptr, FE_LOGD("Node %s in anchor is null.", concat_node->GetName().c_str()),
               return false);
      auto next_node = peerAnchor->GetOwnerNode();
      auto output_nodes = next_node->GetOutDataNodes();
      if ((next_node->GetType() == RESHAPE) && (!output_nodes.empty())) {
        next_node = output_nodes.at(0);
      }
      ge::OpDescPtr next_node_desc = next_node->GetOpDesc();
      string next_node_name = next_node_desc->GetName();
      if (next_node_desc->GetType() == NETOUTPUT) {
        FE_LOGD("Next node %s is netoutput, %s can not optimize.", next_node_name.c_str(),
                concat_node->GetName().c_str());
        return false;
      }
      if (next_node_desc->GetType() == OP_TYPE_END) {
        string parent_op_type;
        (void)ge::AttrUtils::GetStr(next_node_desc, PARENT_OP_TYPE, parent_op_type);
        if (parent_op_type == NETOUTPUT) {
          FE_LOGD("Next node %s is End(netoutput), %s can not optimize.", next_node_name.c_str(),
                  concat_node->GetName().c_str());
          return false;
        }
      }
      bool is_virtual_op = false;
      bool no_task = false;
      bool output_reuse_input = false;
      bool no_padding_continuous_input = false;
      bool is_continous_input = false;
      vector<int64_t> output_index;
      (void)ge::AttrUtils::GetListInt(next_node_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
      (void)ge::AttrUtils::GetBool(next_node_desc, ge::ATTR_NAME_NOTASK, no_task);
      (void)ge::AttrUtils::GetBool(next_node_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
      (void)ge::AttrUtils::GetBool(next_node_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                   no_padding_continuous_input);
      (void)ge::AttrUtils::GetBool(next_node_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
      is_virtual_op = no_task || output_reuse_input || no_padding_continuous_input || \
                      is_continous_input || (!output_index.empty());
      if (is_virtual_op) {
        FE_LOGD("Next node %s is invalid, %s cannot optimize.", next_node_name.c_str(),
                concat_node->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

bool ConcatCOptimizeFusionPass::CheckPeerNodeCanReUsed(const ge::NodePtr &concat_node) const {
  string node_name = concat_node->GetName();
  for (const ge::NodePtr &src_node : concat_node->GetInDataNodes()) {
    auto src_node_op_desc = src_node->GetOpDesc();
    if (src_node_op_desc->HasAttr(kCanReusedForConcatCOptimize)) {
      bool can_reused = true;
      (void)ge::AttrUtils::GetBool(src_node_op_desc, kCanReusedForConcatCOptimize, can_reused);
      if (!can_reused) {
        FE_LOGD("Concat [%s] peer node [%s] cannot reused, can not set no task flag.",
                node_name.c_str(), src_node->GetName().c_str());
        return false;
      }
    }
  }
  FE_LOGD("Concat [%s] all peer node can reused, can set no task flag.", node_name.c_str());
  return true;
}

bool ConcatCOptimizeFusionPass::IsPreNodeAttrValid(const ge::OpDescPtr &pre_node_desc, bool &fusion_virtual_op_flag,
                                          const string &node_name) const {
  string pre_node_name = pre_node_desc->GetName();
  bool is_continous_input = false;
  bool is_continous_output = false;
  bool is_ref = false;
  bool no_task = false;
  bool output_reuse_input = false;
  bool no_padding_continuous_input = false;
  vector<int64_t> output_index;
  (void)ge::AttrUtils::GetBool(pre_node_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
  (void)ge::AttrUtils::GetBool(pre_node_desc, ge::ATTR_NAME_CONTINUOUS_OUTPUT, is_continous_output);
  (void)ge::AttrUtils::GetBool(pre_node_desc, ge::ATTR_NAME_REFERENCE, is_ref);
  (void)ge::AttrUtils::GetListInt(pre_node_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
  (void)ge::AttrUtils::GetBool(pre_node_desc, ge::ATTR_NAME_NOTASK, no_task);
  (void)ge::AttrUtils::GetBool(pre_node_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
  (void)ge::AttrUtils::GetBool(pre_node_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, no_padding_continuous_input);
  if (is_continous_input) {
    FE_LOGD("Previous node %s has continuous_input attribute, %s can't optimize.", pre_node_name.c_str(),
            node_name.c_str());
    fusion_virtual_op_flag = false;
    return false;
  }
  if (is_continous_output) {
    FE_LOGD("Previous node %s has continuous_output attribute, %s can't optimize.", pre_node_name.c_str(),
            node_name.c_str());
    fusion_virtual_op_flag = false;
    return false;
  }
  if (is_ref) {
    FE_LOGD("Previous node %s has reference attribute, %s can't optimize.", pre_node_name.c_str(), node_name.c_str());
    fusion_virtual_op_flag = false;
    return false;
  }
  bool is_virtual_op = no_task || output_reuse_input || no_padding_continuous_input;
  if (is_virtual_op) {
    FE_LOGD("Previous node %s has _no_task attribute, %s can't optimize.", pre_node_name.c_str(), node_name.c_str());
    fusion_virtual_op_flag = false;
    return false;
  }
  if (!output_index.empty()) {
    FE_LOGD("Previous node %s has atomic output, %s can not optimize.", pre_node_name.c_str(), node_name.c_str());
    fusion_virtual_op_flag = false;
    return false;
  }
  return true;
}

void ConcatCOptimizeFusionPass::CheckPreNodeValid(const ge::NodePtr &node, bool &concat_c_optimize_flag) const {
  for (auto input_anchor : node->GetAllInDataAnchors()) {
    if (CheckMultiRef(input_anchor)) {
      concat_c_optimize_flag = false;
      break;
    }
    ge::OutDataAnchorPtr pre_out_anchor = nullptr;
    GetFirstOutAnchorNotInDeleteList(input_anchor, pre_out_anchor, 0);
    if (pre_out_anchor == nullptr) {
      continue;
    }
    ge::OpDescPtr pre_node_desc = pre_out_anchor->GetOwnerNode()->GetOpDesc();
    if (!IsPreNodeAttrValid(pre_node_desc, concat_c_optimize_flag, node->GetName())) {
      break;
    }
  }
}

void ConcatCOptimizeFusionPass::SetPeerNodeWhetherCanReUsed(const ge::NodePtr &concat_node) const {
  auto input_size = concat_node->GetAllInDataAnchorsSize();
  for (uint32_t index = 0; index < input_size; ++index) {
    auto input_anchor = concat_node->GetInDataAnchor(index);
    if (input_anchor == nullptr) {
      continue;
    }
    auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    // set concat_node all peer node cannot reuse
    ge::AttrUtils::SetBool(peer_out_anchor->GetOwnerNode()->GetOpDesc(), kCanReusedForConcatCOptimize, false);
  }
}

void ConcatCOptimizeFusionPass::CheckAndSetAttrForConcat(const ge::NodePtr &node, const ge::OpDescPtr &op_desc,
                                                bool &concat_c_optimize_flag) const {
  if (!concat_c_optimize_flag) {
    FE_LOGD("Cur concat node[%s] cannot do optimize, no need to check and set attr.", op_desc->GetName().c_str());
    return;
  }
  bool is_notask = false;
  bool is_continuous_input = false;
  bool is_reuse_input = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, is_notask);
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_continuous_input);
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, is_reuse_input);
  concat_c_optimize_flag = !(is_notask || is_continuous_input || is_reuse_input);
  if (concat_c_optimize_flag) {
    FE_LOGD("Node[%s] start to set concat attribute.", node->GetName().c_str());
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    SetPeerNodeWhetherCanReUsed(node);
  } else {
    FE_LOGD("Cur concat node %s has already no-task attr, can't optimize.", op_desc->GetName().c_str());
  }
}

void ConcatCOptimizeFusionPass::CheckIsLxSlicedOp(const ge::OpDescPtr &op_desc, bool &concat_c_optimize_flag) const {
  if (!concat_c_optimize_flag) {
    FE_LOGD("Cur concat node[%s] cannot do optimize, no need to check is_sliced_op.", op_desc->GetName().c_str());
    return;
  }
  ToOpStructPtr lx_info = nullptr;
  lx_info = op_desc->TryGetExtAttr(ge::ATTR_NAME_L1_FUSION_EXTEND_PTR, lx_info);
  if (lx_info == nullptr) {
    lx_info = op_desc->TryGetExtAttr(ATTR_NAME_L2_FUSION_EXTEND_PTR, lx_info);
  }
  if (lx_info == nullptr) {
    return;
  }
  FE_LOGD("Cur concat node[%s] has been sliced by lx_fusion.", op_desc->GetName().c_str());
  concat_c_optimize_flag = false;
}

bool ConcatCOptimizeFusionPass::CheckStrideWriteBlock32Align(const ge::GeTensorDescPtr &tensor_desc) const {
  size_t idx = kCAxisIndex;
  int64_t element_cnt = 1;
  int64_t tensor_size = 1;
  int32_t flag = 1;
  std::vector<int64_t> shape = tensor_desc->GetShape().GetDims();
  for (; idx < shape.size(); ++idx) {
    element_cnt *= shape[idx];
  }
  if (element_cnt == 0) {
    return false;
  }
  ge::DataType data_type = tensor_desc->GetDataType();
  TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, tensor_size, flag);
  return tensor_size % kAlignNum == 0;
}

void ConcatCOptimizeFusionPass::CalSliceOffset(const std::vector<int64_t> &output_shape,
                                      ge::DataType data_type, int64_t &output_offset_buff) const {
  int64_t shape_size = static_cast<int64_t>(output_shape.size());
  if (shape_size < real_concat_dim_ + 1) {
    FE_LOGE("Invalid output_shape size[%zu], concat_dim[%lld].", shape_size, real_concat_dim_);
    return;
  }

  int64_t element_cnt = 1;
  int64_t tmp = 0;
  for (auto i = real_concat_dim_; i < shape_size; ++i) {
    tmp = output_shape[i];
    if (tmp == 0) {
      FE_LOGE("Invalid output_shape_idx[%zu], val is 0.", i);
      return;
    }
    element_cnt *= tmp;
  }
  int32_t flag = 1;
  output_offset_buff = 1;
  TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, output_offset_buff, flag);
  return;
}

Status ConcatCOptimizeFusionPass::FeedToOpStructInfo(ge::OpDescPtr& op_desc, const size_t &idx,
                                            const std::vector<int64_t> &concat_out_shape,
                                            const bool &is_last_input) const {
  ge::GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(idx);
  FE_CHECK_NOTNULL(tensor_desc);
  if (!CheckStrideWriteBlock32Align(tensor_desc)) {
      FE_LOGD("Concat input node:[%s], strideWrite block is not 32 alignment.", op_desc->GetName().c_str());
      return fe::FAILED;
  }

  size_t out_size = op_desc->GetOutputsSize();
  std::vector<int64_t> output_shape = tensor_desc->GetShape().GetDims();
  std::vector<std::vector<int64_t>> concat_outputs_shape(out_size, std::vector<int64_t>(output_shape.size(), 0));
  std::vector<int64_t> outputs_offset(out_size, 0);
  std::vector<int64_t> output_i(out_size, 0);
  concat_outputs_shape[idx] = concat_out_shape;
  int64_t output_offset_buff = -1;
  FE_LOGD("Concat input_node[%s] output size is [%zu].", op_desc->GetName().c_str(), out_size);

  if (!is_last_input) {
    CalSliceOffset(output_shape, tensor_desc->GetDataType(), output_offset_buff);
    if (output_offset_buff == -1) {
      FE_LOGE("Concat input_node[%s] is unexpected.", op_desc->GetName().c_str());
      return fe::FAILED;
    }
  } else {
    output_offset_buff = 0;
    FE_LOGD("Concat input_node[%s] is the last input, no need to calculate output_offset_buff.",
            op_desc->GetName().c_str());
  }

  FE_LOGD("Concat input_node[%s], output_offset_buff is [%ld].", op_desc->GetName().c_str(), output_offset_buff);
  outputs_offset[idx] = output_offset_buff;

  (void)ge::AttrUtils::SetListInt(op_desc, kOutputOffsetForBufferFusion, outputs_offset);
  FE_LOGD("Op[%s] set attr _output_offet_for_buffer_fusion[%ld] successfully.",
          op_desc->GetName().c_str(), output_offset_buff);

  output_i[idx] = 0;
  op_desc->SetOutputOffset(output_i);

  ToOpStructPtr optimize_info = nullptr;
  FE_MAKE_SHARED(optimize_info = std::make_shared<ToOpStruct_t>(), return fe::FAILED);
  optimize_info->slice_output_shape = concat_outputs_shape;
  SetStridedInfoToNode(op_desc, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);
  (void)ge::AttrUtils::SetBool(op_desc, NEED_RE_PRECOMPILE, true);
  FE_LOGD("Op[%s] set ext_attr ATTR_NAME_CONCAT_C_OPTIMIZE successfully", op_desc->GetName().c_str());
  return fe::SUCCESS;
}

Status ConcatCOptimizeFusionPass::SetStrideWriteInfoForInputs(const ge::NodePtr &concat_node,
                                                              bool &concat_c_optimize_flag) const {
  std::string node_name = concat_node->GetName();
  if (!concat_c_optimize_flag) {
    FE_LOGD("Node:[%s] cannot do c-axis optimize, turn to next one", node_name.c_str());
    return fe::FAILED;
  }
  FE_LOGD("Node:[%s] begin to set slice info", node_name.c_str());

  ge::NodePtr peer_node = nullptr;
  ge::OpDescPtr peer_op_desc = nullptr;
  ge::OutDataAnchorPtr out_anchor = nullptr;
  ge::GeTensorDescPtr tensor_desc = nullptr;

  auto concat_out_tensor = concat_node->GetOpDesc()->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(concat_out_tensor);
  std::vector<int64_t> concat_out_shape = concat_out_tensor->GetShape().GetDims();
  auto in_anchors = concat_node->GetAllInDataAnchors();
  size_t in_size = in_anchors.size();
  size_t cnt = 0;
  size_t idx = 0;

  for (auto &in_anchor : in_anchors) {
    out_anchor = in_anchor->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(out_anchor);
    peer_op_desc = out_anchor->GetOwnerNode()->GetOpDesc();
    FE_CHECK_NOTNULL(peer_op_desc);
    ++cnt;
    bool is_last_input = (cnt == in_size) ? true : false;
    idx = out_anchor->GetIdx();
    if (FeedToOpStructInfo(peer_op_desc, idx, concat_out_shape, is_last_input) != fe::SUCCESS) {
      concat_c_optimize_flag = false;
      return fe::FAILED;
    }
  }
  FE_LOGD("Concat node[%s] set strideWrite info successfully", node_name.c_str());
  return fe::SUCCESS;
}
REG_PASS("ConcatCOptimize", BUILT_IN_AFTER_BUFFER_OPTIMIZE,
         ConcatCOptimizeFusionPass, FE_PASS);
}
