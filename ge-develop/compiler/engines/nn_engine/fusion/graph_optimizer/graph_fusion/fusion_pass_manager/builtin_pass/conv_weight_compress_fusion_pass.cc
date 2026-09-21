/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/conv_weight_compress_fusion_pass.h"
#include <algorithm>
#include <vector>
#include "common/configuration.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/util/op_info_util.h"
#include "platform/platform_info.h"
#include "common/op_info_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "common/weight_compress_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

/**
 *            weight                weight -> weight_compress
 *            /                          \   /
 *          conv        ->              switch
 *           /                          /    \
 *         relu                      Conv   ConvCompress
 *                                      \   /
 *                                      Merge
 *                                        /
 *                                       relu
 */
namespace fe {
namespace {
Status RelinkControlEdges(ge::NodePtr conv_node, ge::NodePtr conv_compress_node) {
  // connect in control anchor
  if (conv_node->GetInControlAnchor() != nullptr) {
    if (!conv_node->GetInControlAnchor()->GetPeerOutControlAnchors().empty() &&
        conv_compress_node->GetInControlAnchor() != nullptr) {
      for (const ge::OutControlAnchorPtr &out_ctrl_anchor_ptr :
           conv_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        if (ge::GraphUtils::AddEdge(out_ctrl_anchor_ptr, conv_compress_node->GetInControlAnchor()) !=
            ge::GRAPH_SUCCESS) {
          REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][RelkCtrlEdge] Failed to add input control edge for node [%s].",
                          conv_compress_node->GetName().c_str());
          return FAILED;
        }
      }
    }
  }
  // connect out control anchor
  if (conv_node->GetOutControlAnchor() != nullptr) {
    if (!conv_node->GetOutControlAnchor()->GetPeerInControlAnchors().empty() &&
        conv_compress_node->GetOutControlAnchor() != nullptr) {
      for (const ge::InControlAnchorPtr &in_ctrl_anchor_ptr :
           conv_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        if (ge::GraphUtils::AddEdge(conv_compress_node->GetOutControlAnchor(), in_ctrl_anchor_ptr) !=
            ge::GRAPH_SUCCESS) {
          REPORT_FE_ERROR(
              "[GraphOpt][ConvWgtCmpsFus][RelkCtrlEdge] Fail to add output control edge for fusion node[%s].",
              conv_compress_node->GetName().c_str());
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status RelinkDataEdgesOfMergeNode(ge::NodePtr conv_node, ge::NodePtr conv_compress_node, ge::NodePtr merge_node) {
  FE_CHECK(conv_node->GetOutDataAnchor(0) == nullptr,
           REPORT_FE_ERROR("conv_node->GetOutDataAnchor(0) is nullptr."),
           return FAILED);
  auto conv_out_peer_data_anchors = conv_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  conv_node->GetOutDataAnchor(0)->UnlinkAll();

  // link the input anchor of merge node
  if (ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][ConvWgtCmpsFus][RelkDataEdge] Fail to add edge between conv node[%s] and merge node[%s].",
        conv_node->GetName().c_str(), merge_node->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(conv_compress_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(1)) !=
      ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][ConvWgtCmpsFus][RelkDataEdge] Fail to add edge between conv compress node[%s] and merge node[%s].",
        conv_compress_node->GetName().c_str(), merge_node->GetName().c_str());
    return FAILED;
  }

  // link the output anchor of merge node
  for (const auto &peer_in_data_anchor : conv_out_peer_data_anchors) {
    if (ge::GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), peer_in_data_anchor) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][RelkDataEdge] Failed to add edge for the output anchor of merge node [%s].",
                      merge_node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
} // namespace

static const string PATTERN_CONV = "conv_pattern";
static const std::map<std::string, std::string> CONV_COMPRESS_OP_TYPE_MAP {
    {CONV2D, kConv2DCompress}, {kFullyConnection, kFullyConnectionCompress}, {MATMULV2OP, kMatMulV2Compress},
    {kConv2DTransposeD, kConv2DTransposeDCompress}};
static const std::string HOST_OP_TYPE = "WeightCompressHost";
static const std::string SWITCH_OP_TYPE = "Switch";
static const std::string MERGE_OP_TYPE = "Merge";
static const std::string TENSOR_NAME_FILTER_COMPRESS = "filter_compress";
static const std::string TENSOR_NAME_COMPRESS_INDEX = "compress_index";
const uint64_t kRecursiveLoopMax = 10000000;
uint64_t kFeRecursiveCnt = 0;
const float kDefaultCompressRatioThreshold = 0.8f;
vector<FusionPattern *> ConvWeightCompressFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("ConvWeightCompressFusion");
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][DefPtn] Failed to create a new object."),
           return patterns);

  // conv2d / FC / MatMulV2
  pattern->AddOpDesc(PATTERN_CONV, {CONV2D, kFullyConnection, MATMULV2OP, kConv2DTransposeD}).SetOutput(PATTERN_CONV);
  patterns.push_back(pattern);

  return patterns;
}

Status ConvWeightCompressFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                            vector<ge::NodePtr> &fusion_nodes) {
  (void)fusion_nodes;
  ge::NodePtr conv_node = GetNodeFromMapping(PATTERN_CONV, mapping);
  FE_CHECK(conv_node == nullptr, REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion] Conv node is nullptr."),
           return FAILED);

  std::string conv_name = conv_node->GetName();
  if (conv_node->GetOpDesc()->GetInputsSize() < 2) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion] The size of input desc of node[%s] is less than 2.",
                    conv_name.c_str());
    return FAILED;
  }

  // avoid the function loss of conv op
  ge::DataType weight_data_type = conv_node->GetOpDesc()->GetInputDescPtr(0)->GetDataType();
  if (weight_data_type != ge::DT_INT8 && weight_data_type != ge::DT_UINT8) {
    FE_LOGD("The weight data type of node[%s] is not int8 or uint8.", conv_name.c_str());
    return NOT_CHANGED;
  }

  // check whether this op is supported by ai core
  ge::OpDescPtr conv_op_desc = conv_node->GetOpDesc();
  if (!CheckOpSupported(conv_op_desc)) {
    FE_LOGI("Op[%s, %s] is not supported by AI Core.", conv_op_desc->GetName().c_str(),
            conv_op_desc->GetType().c_str());
    return NOT_CHANGED;
  }

  // if is unknown shape op, do not do weight compress.
  if (UnknownShapeUtils::IsUnknownShapeOp(*conv_node->GetOpDesc())) {
    FE_LOGD("The node[%s] is unknown shape op, does not need to be compressed.", conv_name.c_str());
    return NOT_CHANGED;
  }

  // check whether do compress this conv node
  uint8_t compress_flag = IsCompressWeight(conv_node);
  if (compress_flag == static_cast<uint8_t>(WEIGHCOMPRESSINNERFLAG::DISABLE_COMPRESS_FLAG)) {
    FE_LOGD("Node[%s] does not need to be compressed.", conv_name.c_str());
    return NOT_CHANGED;
  }

  // check all the node between conv and weight node can be fold
  if (!CheckWeightConstFoldNode(conv_node)) {
    FE_LOGD(
        "There is some node between Node[%s, %s] and the weight node "
        "that cannot be folded.",
        conv_name.c_str(), conv_node->GetType().c_str());
    return NOT_CHANGED;
  }

  FE_LOGD("Begin to do weight compress for node[%s, %s].", conv_node->GetName().c_str(), conv_node->GetType().c_str());

  ge::OpDescPtr conv_compress_op_desc = ge::AttrUtils::CopyOpDesc(conv_node->GetOpDesc());

  // modify op type of conv node
  auto iter = CONV_COMPRESS_OP_TYPE_MAP.find(conv_node->GetType());
  if (iter == CONV_COMPRESS_OP_TYPE_MAP.end()) {
    FE_LOGD("Can not find conv compress op type by op type[%s].", conv_node->GetType().c_str());
    return NOT_CHANGED;
  }
  ge::OpDescUtilsEx::SetType(conv_compress_op_desc, iter->second);

  if (CreateConvCompressOpDesc(conv_op_desc, conv_compress_op_desc, compress_flag) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion] Failed to create compression type operation for node [%s, %s].",
                    conv_op_desc->GetName().c_str(), conv_op_desc->GetType().c_str());
    return FAILED;
  }
  // check whether this op is supported by ai core
  if (!CheckOpSupported(conv_compress_op_desc)) {
    FE_LOGI("Op[%s, %s] is not supported by AI Core.", conv_compress_op_desc->GetName().c_str(),
            conv_compress_op_desc->GetType().c_str());
    return NOT_CHANGED;
  }

  // add node to graph
  ge::NodePtr conv_compress_node = graph.AddNode(conv_compress_op_desc);
  FE_CHECK(conv_compress_node == nullptr,
           REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion] Failed to add op[%s, %s] to graph.",
                           conv_compress_op_desc->GetName().c_str(), conv_compress_op_desc->GetType().c_str()),
           return FAILED);

  // add host op
  ge::NodePtr host_node = CreateHostNode(conv_op_desc, graph, compress_flag);
  FE_CHECK(host_node == nullptr,
           REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion] Failed to add weight compress host node to graph."),
           return FAILED);
  // add switch op
  ge::NodePtr switch_node = CreateSwitchNode(conv_op_desc, graph);
  FE_CHECK(switch_node == nullptr, REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion]Failed to add switch to graph."),
           return FAILED);
  // add merge op
  ge::NodePtr merge_node = CreateMergeNode(conv_op_desc, graph);
  FE_CHECK(merge_node == nullptr, REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion]Failed to add merge node to graph."),
           return FAILED);

  if (RelinkNodeEdges(conv_node, conv_compress_node, host_node, switch_node, merge_node) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][Fusion] Failed to link edges around node[%s, %s].",
                    conv_node->GetName().c_str(), conv_node->GetType().c_str());
    return FAILED;
  }
  FE_LOGD("Finished converting to compressed node type [%s].", conv_compress_node->GetName().c_str());
  return SUCCESS;
}

Status ConvWeightCompressFusionPass::RelinkNodeEdges(ge::NodePtr conv_node, ge::NodePtr conv_compress_node,
                                                     ge::NodePtr host_node, ge::NodePtr switch_node,
                                                     ge::NodePtr merge_node) const {
  // unlink the edge of conv's weight input
  FE_CHECK(conv_node->GetInDataAnchor(1) == nullptr,
           REPORT_FE_ERROR("conv_node->GetInDataAnchor(1) is nullptr."),
           return FAILED);
  ge::OutDataAnchorPtr conv_weight_output_anchor = conv_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  if (ge::GraphUtils::RemoveEdge(conv_weight_output_anchor, conv_node->GetInDataAnchor(1)) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Failed to remove the first input anchors edge from the conv node [%s].",
                    conv_node->GetName().c_str());
    return FAILED;
  }
  // link the input of host node
  if (ge::GraphUtils::AddEdge(conv_weight_output_anchor, host_node->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Failed to add edge for the host node [%s]'s first indata anchor.",
                    host_node->GetName().c_str());
    return FAILED;
  }

  // link the input of switch node with weight and host node
  if (ge::GraphUtils::AddEdge(conv_weight_output_anchor, switch_node->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Failed to add edge for switch node [%s]'s first indata anchor.",
                    switch_node->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(host_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(1)) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Failed to add edge between host node[%s] and switch node[%s]",
                    host_node->GetName().c_str(), switch_node->GetName().c_str());
    return FAILED;
  }

  // link the output of switch node with two conv node
  if (ge::GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1)) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Failed to add edge between switch node [%s] and conv node [%s].",
        switch_node->GetName().c_str(), conv_node->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(switch_node->GetOutDataAnchor(1),
                              conv_compress_node->GetInDataAnchor(TENSOR_INDEX_FILTER_COMPRESS)) !=
      ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Failed to add edge between SwitchNode[%s] and ConvCompressNode[%s].",
        switch_node->GetName().c_str(), conv_compress_node->GetName().c_str());
    return FAILED;
  }
  // link the feature map input for conv compress
  if (ge::GraphUtils::AddEdge(conv_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                              conv_compress_node->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Failed to add edge between SwitchNode[%s] and ConvCompressNode[%s].",
        switch_node->GetName().c_str(), conv_compress_node->GetName().c_str());
    return FAILED;
  }

  // link the anchor after filter input for conv compress node
  if (conv_node->GetAllInDataAnchorsSize() > TENSOR_INDEX_COMPRESS_INDEX) {
    for (uint32_t i = 2; i < conv_node->GetAllInDataAnchorsSize(); i++) {
      ge::InDataAnchorPtr in_data_anchor_ptr = conv_node->GetInDataAnchor(i);
      if (in_data_anchor_ptr != nullptr && in_data_anchor_ptr->GetPeerOutAnchor() != nullptr) {
        uint32_t index = i + 1;
        if (ge::GraphUtils::AddEdge(in_data_anchor_ptr->GetPeerOutAnchor(),
                                    conv_compress_node->GetInDataAnchor(index)) != ge::GRAPH_SUCCESS) {
          REPORT_FE_ERROR(
              "[GraphOpt][ConvWgtCmpsFus][RelkNdEdge] Fail to add edge for input[%u] of ConvCompressNode[%s].",
              index, conv_compress_node->GetName().c_str());
          return FAILED;
        }
      }
    }
  }

  if (RelinkDataEdgesOfMergeNode(conv_node, conv_compress_node, merge_node) != SUCCESS) {
    return FAILED;
  }

  return RelinkControlEdges(conv_node, conv_compress_node);
}

Status ConvWeightCompressFusionPass::CreateConvCompressOpDesc(ge::OpDescPtr conv_op_desc,
                                                              ge::OpDescPtr &conv_compress_op_desc,
                                                              const uint8_t &compress_flag) const {
  conv_compress_op_desc->SetName(conv_op_desc->GetName() + "_Compress");
  // remove all tensor desc except input
  for (uint32_t i = conv_compress_op_desc->GetAllInputsSize() - 1; i > 0; i--) {
    if (!ge::OpDescUtils::ClearInputDesc(conv_compress_op_desc, i)) {
      REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][CrtConvCprsOpDesc] Failed to clear input [%u] of node [%s].", i,
                      conv_compress_op_desc->GetName().c_str());
      return FAILED;
    }
  }

  // input name shall be updated
  std::map<string, uint32_t> new_conv_desc_input_name;
  new_conv_desc_input_name.emplace(conv_op_desc->GetInputNameByIndex(0), 0);
  if (!conv_compress_op_desc->UpdateInputName(new_conv_desc_input_name)) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][CrtConvCprsOpDesc] Failed to update the input name of node [%s].",
                    conv_compress_op_desc->GetName().c_str());
    return FAILED;
  }

  conv_compress_op_desc->AddInputDesc(TENSOR_NAME_FILTER_COMPRESS,
                                      conv_op_desc->GetInputDesc(TENSOR_INDEX_FILTER_COMPRESS));
  new_conv_desc_input_name.emplace(TENSOR_NAME_FILTER_COMPRESS, TENSOR_INDEX_FILTER_COMPRESS);

  // copy the weight tensor desc
  ge::GeTensorDesc compress_index_tensor_desc = conv_op_desc->GetInputDesc(TENSOR_INDEX_FILTER_COMPRESS);
  // set the format to ND and datatype to int8
  compress_index_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  compress_index_tensor_desc.SetFormat(ge::FORMAT_ND);
  compress_index_tensor_desc.SetOriginDataType(ge::DT_INT8);
  compress_index_tensor_desc.SetDataType(ge::DT_INT8);
  // clear the shape
  std::vector<int64_t> empty_dims;
  ge::GeShape index_shape(empty_dims);
  compress_index_tensor_desc.SetShape(index_shape);
  compress_index_tensor_desc.SetOriginShape(index_shape);
  conv_compress_op_desc->AddInputDesc(TENSOR_NAME_COMPRESS_INDEX, compress_index_tensor_desc);
  new_conv_desc_input_name.emplace(TENSOR_NAME_COMPRESS_INDEX, TENSOR_INDEX_COMPRESS_INDEX);
  if (conv_op_desc->GetAllInputsSize() > TENSOR_INDEX_COMPRESS_INDEX) {
    for (uint32_t i = TENSOR_INDEX_COMPRESS_INDEX; i < conv_op_desc->GetAllInputsSize(); ++i) {
      conv_compress_op_desc->AddInputDesc(conv_op_desc->GetInputNameByIndex(i), conv_op_desc->GetInputDesc(i));
      new_conv_desc_input_name.emplace(conv_op_desc->GetInputNameByIndex(i), i + 1);
    }
  }
  // update input name mapping
  if (!conv_compress_op_desc->UpdateInputName(new_conv_desc_input_name)) {
    REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][CrtConvCprsOpDesc] Failed to update the input name of node [%s].",
                    conv_compress_op_desc->GetName().c_str());
    return FAILED;
  }

  // update is input const
  vector<bool> is_input_const_vec = conv_compress_op_desc->GetIsInputConst();
  vector<bool> new_is_input_const_vec;
  for (uint32_t i = 0; i < is_input_const_vec.size(); ++i) {
    new_is_input_const_vec.push_back(is_input_const_vec[i]);
    if (i == TENSOR_INDEX_FILTER_COMPRESS) {
      new_is_input_const_vec.push_back(true);
    }
  }
  conv_compress_op_desc->SetIsInputConst(new_is_input_const_vec);

  if (compress_flag == static_cast<uint8_t>(WEIGHCOMPRESSINNERFLAG::WEIGHT_COMPRESS_FLAG)) {
    // add _weight_compress
    if (!ge::AttrUtils::SetBool(conv_compress_op_desc, ATTR_NAME_WEIGHT_COMPRESS, true)) {
      FE_LOGD("Setting _weight_compress attribute on node [%s] was unsuccessful.", conv_compress_op_desc->GetName().c_str());
      return FAILED;
    }
    return SUCCESS;
  } else {
    if (!ge::AttrUtils::SetStr(conv_compress_op_desc, ATTR_NAME_COMPRESS_TYPE_FLAG, kWeightSparseFourToTwo)) {
      FE_LOGD("Setting alg attribute on node [%s] was unsuccessful.", conv_compress_op_desc->GetName().c_str());
      return FAILED;
    }
    ge::GeTensorDescPtr compress_weight_descptr = conv_compress_op_desc->MutableInputDesc(TENSOR_NAME_FILTER_COMPRESS);
    Status status = SetSparsityCompressWeightShape(conv_op_desc, compress_weight_descptr, std::make_pair(false, true));
    if (status != SUCCESS) {
      return status;
    }
    ge::GeTensorDescPtr compress_index_descptr = conv_compress_op_desc->MutableInputDesc(TENSOR_NAME_COMPRESS_INDEX);
    return SetSparsityCompressWeightShape(conv_op_desc, compress_index_descptr, std::make_pair(true, true));
  }
}

ge::NodePtr ConvWeightCompressFusionPass::CreateHostNode(const ge::OpDescPtr &conv_op_desc,
                                                         ge::ComputeGraph &graph,
                                                         const uint8_t &compress_flag) const {
  // add host node
  std::string op_name = conv_op_desc->GetName() + "_" + HOST_OP_TYPE;
  ge::OpDescPtr host_op_desc = nullptr;
  FE_MAKE_SHARED(host_op_desc = std::make_shared<ge::OpDesc>(op_name, HOST_OP_TYPE), return nullptr);
  FE_CHECK(host_op_desc == nullptr,
           REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][CrtHostNd] Failed to make shared of host op desc."),
           return nullptr);
  ge::GeTensorDesc tensor_desc = conv_op_desc->GetInputDesc(1);

  ge::GeShape new_shape;
  ge::GeShape origin_shape = tensor_desc.GetOriginShape();
  ge::Format origin_format = tensor_desc.GetOriginFormat();
  ge::DataType data_type = tensor_desc.GetDataType();
  ge::Format new_format = ge::FORMAT_FRACTAL_Z;
  int32_t c0_bit = GetC0BitValFromC0(GetC0ValByOpDescAndDtype(conv_op_desc, data_type));
  new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit));
  ShapeAndFormat output_shape_and_format_info = {origin_shape, new_shape, origin_format, new_format, data_type};
  (void)GetShapeAccordingToFormat(output_shape_and_format_info);
  tensor_desc.SetFormat(ge::FORMAT_FRACTAL_Z);
  tensor_desc.SetShape(new_shape);
  host_op_desc->AddInputDesc("weight", tensor_desc);

  ge::GeTensorDesc output_desc;
  output_desc.SetDataType(ge::DT_BOOL);
  output_desc.SetOriginDataType(ge::DT_BOOL);
  output_desc.SetFormat(ge::FORMAT_ND);
  output_desc.SetOriginFormat(ge::FORMAT_ND);
  host_op_desc->AddOutputDesc("iscompress", output_desc);
  if (compress_flag == static_cast<uint8_t>(WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG)) {
   (void)ge::AttrUtils::SetStr(host_op_desc, ATTR_NAME_COMPRESS_TYPE_FLAG, kWeightSparseFourToTwo);
   FE_LOGD("Successfully set _compress flag attribute %d on node [%s].", compress_flag, host_op_desc->GetName().c_str());
  }
  ge::NodePtr host_node = graph.AddNode(host_op_desc);
  return host_node;
}

ge::NodePtr ConvWeightCompressFusionPass::CreateSwitchNode(const ge::OpDescPtr &conv_op_desc,
                                                           ge::ComputeGraph &graph) const {
  // add switch node
  std::string op_name = conv_op_desc->GetName() + "_" + SWITCH_OP_TYPE;
  ge::OpDescPtr switch_op_desc = nullptr;
  FE_MAKE_SHARED(switch_op_desc = std::make_shared<ge::OpDesc>(op_name, SWITCH_OP_TYPE), return nullptr);
  FE_CHECK(switch_op_desc == nullptr,
           REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][CrtSwtNd] Failed to make shared of switch op desc."),
           return nullptr);
  switch_op_desc->AddInputDesc("data", conv_op_desc->GetInputDesc(1));

  ge::GeTensorDesc input_desc;
  input_desc.SetDataType(ge::DT_BOOL);
  input_desc.SetOriginDataType(ge::DT_BOOL);
  input_desc.SetFormat(ge::FORMAT_ND);
  input_desc.SetOriginFormat(ge::FORMAT_ND);
  switch_op_desc->AddInputDesc("pred", input_desc);

  switch_op_desc->AddOutputDesc("output_false", conv_op_desc->GetInputDesc(1));
  switch_op_desc->AddOutputDesc("output_true", conv_op_desc->GetInputDesc(1));

  (void)ge::AttrUtils::SetInt(switch_op_desc, FORMAT_AGNOSTIC, 1);
  std::vector<int64_t> input_vec = {1};
  (void)ge::AttrUtils::SetListInt(switch_op_desc, INPUT_FORMAT_AGNOSTIC_EXCEPTION, input_vec);

  ge::NodePtr switch_node = graph.AddNode(switch_op_desc);
  return switch_node;
}

ge::NodePtr ConvWeightCompressFusionPass::CreateMergeNode(const ge::OpDescPtr &conv_op_desc,
                                                          ge::ComputeGraph &graph) const {
  // add merge node
  std::string op_name = conv_op_desc->GetName() + "_" + MERGE_OP_TYPE;
  ge::OpDescPtr merge_op_desc = nullptr;
  FE_MAKE_SHARED(merge_op_desc = std::make_shared<ge::OpDesc>(op_name, MERGE_OP_TYPE), return nullptr);
  FE_CHECK(merge_op_desc == nullptr,
           REPORT_FE_ERROR("[GraphOpt][ConvWgtCmpsFus][CrtMrgNd] Failed to create shared of host op descriptor."),
           return nullptr);

  merge_op_desc->AddInputDesc("x1", conv_op_desc->GetOutputDesc(0));
  merge_op_desc->AddInputDesc("x2", conv_op_desc->GetOutputDesc(0));
  merge_op_desc->AddOutputDesc("y", conv_op_desc->GetOutputDesc(0));

  ge::GeTensorDesc output_desc;
  output_desc.SetDataType(ge::DT_INT32);
  output_desc.SetOriginDataType(ge::DT_INT32);
  output_desc.SetFormat(ge::FORMAT_ND);
  output_desc.SetOriginFormat(ge::FORMAT_ND);
  merge_op_desc->AddOutputDesc("value_index", output_desc);

  (void)ge::AttrUtils::SetInt(merge_op_desc, FORMAT_AGNOSTIC, 1);

  ge::NodePtr merge_node = graph.AddNode(merge_op_desc);
  return merge_node;
}

uint32_t ConvWeightCompressFusionPass::IsCompressWeight(ge::NodePtr node_ptr) const {
  int64_t groups_val = GROUPS_DEFAULT_VALUE;
  if (ge::AttrUtils::GetInt(node_ptr->GetOpDesc(), ATTR_NAME_GROUPS, groups_val)) {
    if (groups_val != GROUPS_DEFAULT_VALUE) {
      FE_LOGI("The groups value of node[%s, %s] is [%ld] which is not supported.", node_ptr->GetName().c_str(),
              node_ptr->GetType().c_str(), groups_val);
      return static_cast<uint32_t>(WEIGHCOMPRESSINNERFLAG::DISABLE_COMPRESS_FLAG);
    }
  }
  bool enable_compress = Configuration::Instance(AI_CORE_NAME).IsEnableCompressWeight();
  FE_LOGD("EnableCompressWeight flag = %d.", enable_compress);
  if (!enable_compress) {
    WEIGHCOMPRESSINNERFLAG enable_sparsity = JudgeIsSparsityFlag();
    if (enable_sparsity == WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG) {
      return static_cast<uint32_t>(WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG);
    }
    bool is_compress = false;
    // if the node does not contain is_compress_weight, return false
    if (!ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), ge::ATTR_NAME_COMPRESS_WEIGHT, is_compress)) {
      FE_LOGD("The node[%s] does not have is_compress_weight attr.", node_ptr->GetName().c_str());
      return static_cast<uint32_t>(WEIGHCOMPRESSINNERFLAG::DISABLE_COMPRESS_FLAG);
    }
    if (is_compress) {
      return static_cast<uint32_t>(WEIGHCOMPRESSINNERFLAG::WEIGHT_COMPRESS_FLAG);
    }
    return static_cast<uint32_t>(WEIGHCOMPRESSINNERFLAG::DISABLE_COMPRESS_FLAG);
  }
  return static_cast<uint32_t>(WEIGHCOMPRESSINNERFLAG::WEIGHT_COMPRESS_FLAG);
}

bool ConvWeightCompressFusionPass::CheckWeightConstFoldNode(ge::NodePtr conv_node_ptr) const {
  ge::InDataAnchorPtr in_data_anchor_ptr = conv_node_ptr->GetInDataAnchor(1);
  if (in_data_anchor_ptr == nullptr || in_data_anchor_ptr->GetPeerOutAnchor() == nullptr ||
      in_data_anchor_ptr->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
    return false;
  }
  ge::NodePtr node_ptr = in_data_anchor_ptr->GetPeerOutAnchor()->GetOwnerNode();
  kFeRecursiveCnt = 0;
  return CheckConstFoldNode(node_ptr);
}

bool ConvWeightCompressFusionPass::CheckConstFoldNode(ge::NodePtr node_ptr) const {
  if (kFeRecursiveCnt == kRecursiveLoopMax) {
    FE_LOGD("Recursive calls reach max num(%lu).", kRecursiveLoopMax);
    return false;
  }
  kFeRecursiveCnt++;

  string op_type;
  if (ge::NodeUtils::GetConstOpType(node_ptr, op_type)) {
    return true;
  }

  if (kConstFoldingOpType.count(node_ptr->GetType()) == 0) {
    FE_LOGD("Node[%s, %s] can not be fold.", node_ptr->GetType().c_str(), node_ptr->GetType().c_str());
    return false;
  }
  if (node_ptr->GetInNodes().empty()) {
    return true;
  }
  for (const ge::NodePtr &node : node_ptr->GetInNodes()) {
    if (!CheckConstFoldNode(node)) {
      return false;
    }
  }
  return true;
}

REG_PASS("ConvWeightCompressFusionPass", BUILT_IN_GRAPH_PASS,
         ConvWeightCompressFusionPass, SINGLE_SCENE_OPEN | FE_PASS);
}  // namespace fe
