/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/split_conv_fusion_pass.h"
#include <string>
#include "graph/debug/ge_attr_define.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

static const uint32_t TWO_DIM_NUM = 2;
namespace fe {
vector<string> SplitConvFusionPass::GetNodeTypes() {
  vector<string> result;
  result.push_back(SPLITD);
  result.push_back(SPLITVD);
  return result;
}

string SplitConvFusionPass::GetPatternName() { return "SplitConvFusionPass"; }

Status SplitConvFusionPass::DoFusion(ge::ComputeGraph &graph, ge::NodePtr &split_node,
                                     vector<ge::NodePtr> &fusion_nodes) {
  (void)fusion_nodes;
  FE_LOGD("Define SplitConvFusionPass fusion begin.");
  // 1. check the Split
  if (!split_optimize_checker_.Check(split_node)) {
    FE_LOGI("Split node[%s] is not matched, not do fusion.", split_node->GetName().c_str());
    return NOT_CHANGED;
  };
  auto out_nodes = split_node->GetOutDataNodes();
  for (const auto &out_node : out_nodes) {
    if (out_node == nullptr || out_node->GetType() != CONV2D) {
      FE_LOGD("Split node[%s] has output node that is not conv2d, not do fusion.", split_node->GetName().c_str());
      return NOT_CHANGED;
    }
    auto cube_in_anchors = out_node->GetAllInDataAnchors();
    bool split_not_fm = cube_in_anchors.empty() || cube_in_anchors.at(0) == nullptr ||
                        cube_in_anchors.at(0)->GetPeerOutAnchor() == nullptr ||
                        cube_in_anchors.at(0)->GetPeerOutAnchor()->GetOwnerNode() != split_node;
    if (split_not_fm) {
      return NOT_CHANGED;
    }
    if (!CheckInOutNodesAttr(out_node)) {
      return NOT_CHANGED;
    }
  }
  auto in_nodes = split_node->GetInDataNodes();
  for (const auto &in_node : in_nodes) {
    if (!CheckInOutNodesAttr(in_node)) {
      return NOT_CHANGED;
    }
  }
  OpKernelInfoPtr op_kernel_info_ptr = nullptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, STRIDEDREAD);
  if (op_kernel_info_ptr == nullptr) {
    (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), kSupportStridedreadOptimize, true);
    return NOT_CHANGED;
  }
  FE_LOGD("Start to do fusion for split node[%s].", split_node->GetName().c_str());
  if (FusionSplit(graph, split_node) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SplitConvFus][DoFus] do split fusion failed");
    return FAILED;
  }

  for (size_t i = 0; i < split_node->GetAllOutDataAnchors().size(); ++i) {
    FE_CHECK_NOTNULL(split_node->GetOutDataAnchor(i));
    auto split_peer_in_anchors = split_node->GetOutDataAnchor(i)->GetPeerInDataAnchors();
    for (size_t j = 0; j < split_peer_in_anchors.size(); ++j) {
      ge::NodePtr strided_read_node = split_peer_in_anchors.at(j)->GetOwnerNode();
      FE_CHECK_NOTNULL(split_node->GetOpDesc()->MutableInputDesc(0));
      if (strided_read_node->GetOpDesc()->GetType() == STRIDEDREAD &&
          split_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDimNum() < TWO_DIM_NUM) {
          REPORT_FE_ERROR("[GraphOpt][SplitConvFus][DoFus] split %s's input dim number is less than 2.",
                          split_node->GetName().c_str());
          return FAILED;
      }
      FE_CHECK_NOTNULL(split_node->GetOpDesc()->MutableInputDesc(0));
      auto split_input_shape = split_node->GetOpDesc()->MutableInputDesc(0)->MutableShape();
      (void)ge::AttrUtils::SetInt(strided_read_node->GetOpDesc(), STRIDE_ATTR_STRIDE, split_input_shape.GetDim(1));
    }
  }

  // 3. set the attribute of Split
  FE_LOGD("Node[%s]: set the attribute of Split.", split_node->GetName().c_str());
  SetGeAttrForSplit(split_node->GetOpDesc(), 1);

  FE_LOGI("Define SplitConvFusionPass fusion end.");
  return SUCCESS;
}

bool SplitConvFusionPass::CheckInOutNodesAttr(const ge::NodePtr &node) const {
  if (node == nullptr) {
    return false;
  }
  ge::OpDescPtr op_desc = node->GetOpDesc();
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, no_task);
  if (no_task) {
    FE_LOGD("Next node [%s] has no_task attribute, can't optimize.", op_desc->GetName().c_str());
    return false;
  }
  std::string fusion_virtual_op;
  (void)ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_FUSION_VIRTUAL_OP, fusion_virtual_op);
  if (!fusion_virtual_op.empty()) {
    FE_LOGD("Next node [%s] has fusion_virtual_op attribute, can't optimize.", op_desc->GetName().c_str());
    return false;
  }
  bool is_continous_input = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
  if (is_continous_input) {
    FE_LOGD("Next node [%s] has continous_input attribute, can't optimize.", op_desc->GetName().c_str());
    return false;
  }
  vector<int64_t> output_index;
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
  if (!output_index.empty()) {
    FE_LOGD("Next node [%s] has atomic attribute, can't optimize.", op_desc->GetName().c_str());
    return false;
  }
  return true;
}

Status SplitConvFusionPass::FusionSplit(ge::ComputeGraph &graph, ge::NodePtr &split_node) {
  for (size_t i = 0; i < split_node->GetAllOutDataAnchors().size(); ++i) {
    auto data_type = split_node->GetOpDesc()->GetOutputDesc(i).GetOriginDataType();
    ge::NodePtr strided_read_node = nullptr;
    FE_CHECK_NOTNULL(split_node->GetOutDataAnchor(i));
    auto split_out_peer_anchors = split_node->GetOutDataAnchor(i)->GetPeerInDataAnchors();
    for (size_t j = 0; j < split_out_peer_anchors.size(); ++j) {
      ge::NodePtr conv2d_node = split_out_peer_anchors.at(j)->GetOwnerNode();
      if (conv2d_node->GetType() == CONV2D) {
        std::shared_ptr<ge::OpDesc> strided_read_opdesc;
        CreateStridedRead(conv2d_node, strided_read_opdesc);
        strided_read_node = graph.AddNode(strided_read_opdesc);
        FE_CHECK_NOTNULL(strided_read_node);
        InsertNode(split_node->GetOutDataAnchor(i), conv2d_node->GetInDataAnchor(0), strided_read_node);
      }
    }
    
    auto output_desc = split_node->GetOpDesc()->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      continue;
    }
    if (GetNC1HWC0Shape(output_desc, data_type) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][SplitConvFus][FusSplit] Failed to get shape of NC1HWC0.");
      return FAILED;
    }
    JudgeOp(strided_read_node);
  }
  FE_CHECK_NOTNULL(split_node->GetOpDesc()->GetInputDescPtr(0));
  auto in_data_type = split_node->GetOpDesc()->GetInputDescPtr(0)->GetOriginDataType();
  if (GetNC1HWC0Shape(split_node->GetOpDesc()->MutableInputDesc(0), in_data_type) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SplitConvFus][FusSplit] Failed to get shape of NC1HWC0.");
    return FAILED;
  }
  return SUCCESS;
}

REG_PASS("SplitConvFusionPass", BUILT_IN_GRAPH_PASS,
         SplitConvFusionPass, SINGLE_SCENE_OPEN | FE_PASS);
}  // namespace fe
