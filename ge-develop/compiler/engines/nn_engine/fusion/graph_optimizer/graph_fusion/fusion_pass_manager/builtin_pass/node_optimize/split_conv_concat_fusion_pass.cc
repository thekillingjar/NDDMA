/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/split_conv_concat_fusion_pass.h"
#include <string>
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
namespace {
const int MIN_SPLIT_DIM_NUM = 2;
constexpr char const *LEAKYRELU = "LeakyRelu";
constexpr char const *kOffConvConcatSplit = "OFF_CONV_CONCAT_SPLIT";

Status CheckQuantAndConvBeforeConcat(ge::InDataAnchorPtr &quant_in_data_anchor_ptr, ge::NodePtr &concat_prev_node,
                                     SubGraphStructure &sub_graph_structure) {
  sub_graph_structure.is_quant = (sub_graph_structure.fusion_type == FusionType::Reserved ||
                                  sub_graph_structure.fusion_type == FusionType::HasQuant) &&
                                  quant_in_data_anchor_ptr != nullptr;
  sub_graph_structure.is_only_conv = (sub_graph_structure.fusion_type == FusionType::Reserved ||
                                      sub_graph_structure.fusion_type == FusionType::OnlyConv) &&
                                      concat_prev_node->GetType() == CONV2D;
  sub_graph_structure.is_conv_and_dequant = (sub_graph_structure.fusion_type == FusionType::Reserved ||
                                             sub_graph_structure.fusion_type == FusionType::ConvAndDequant) &&
                                             concat_prev_node->GetType() == ASCEND_DEQUANT;
  ge::NodePtr leaky_relu_prev_node;
  if (concat_prev_node->GetType() == LEAKYRELU || concat_prev_node->GetType() == RELU) {
    FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0));
    FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
    leaky_relu_prev_node = concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
    sub_graph_structure.is_only_conv = (sub_graph_structure.fusion_type == FusionType::Reserved ||
                                        sub_graph_structure.fusion_type == FusionType::OnlyConv) &&
                                        leaky_relu_prev_node->GetType() == CONV2D;
    sub_graph_structure.is_conv_and_dequant = (sub_graph_structure.fusion_type == FusionType::Reserved ||
                                               sub_graph_structure.fusion_type == FusionType::ConvAndDequant) &&
                                               leaky_relu_prev_node->GetType() == ASCEND_DEQUANT;
  }
  if (sub_graph_structure.is_quant) {
    sub_graph_structure.fusion_type = FusionType::HasQuant;
  } else if (sub_graph_structure.is_only_conv) {
    sub_graph_structure.fusion_type = FusionType::OnlyConv;
  } else if (sub_graph_structure.is_conv_and_dequant) {
    sub_graph_structure.fusion_type = FusionType::ConvAndDequant;
  } else {
    FE_LOGD("Sub graph structure is not matched");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status ParseAndCheckSplitNode(ge::InDataAnchorPtr &quant_in_data_anchor_ptr, ge::NodePtr &concat_prev_node,
                              ge::NodePtr &split, const SubGraphStructure &sub_graph_structure, string &split_name) {
  ge::NodePtr leaky_relu_prev_node = nullptr;
  ge::NodePtr leaky_relu_prev_prev_node = nullptr;
  switch (sub_graph_structure.fusion_type) {
    case FusionType::HasQuant:
      FE_CHECK_NOTNULL(quant_in_data_anchor_ptr->GetPeerOutAnchor());
      split = quant_in_data_anchor_ptr->GetPeerOutAnchor()->GetOwnerNode();
      break;

    case FusionType::OnlyConv:
      if (concat_prev_node->GetType() == LEAKYRELU || concat_prev_node->GetType() == RELU) {
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        leaky_relu_prev_node = concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        FE_CHECK_NOTNULL(leaky_relu_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(leaky_relu_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        split = leaky_relu_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
      } else {
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        split = concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
      }
      break;

    case FusionType::ConvAndDequant:
      if (concat_prev_node->GetType() == LEAKYRELU || concat_prev_node->GetType() == RELU) {
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        leaky_relu_prev_node = concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        FE_CHECK_NOTNULL(leaky_relu_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(leaky_relu_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        leaky_relu_prev_prev_node = leaky_relu_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        FE_CHECK_NOTNULL(leaky_relu_prev_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(leaky_relu_prev_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        split = leaky_relu_prev_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
      } else {
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        ge::NodePtr concat_prev_prev_node = concat_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        FE_CHECK_NOTNULL(concat_prev_prev_node->GetInDataAnchor(0));
        FE_CHECK_NOTNULL(concat_prev_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor());
        split = concat_prev_prev_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
      }
      break;

    default:
      break;
  }

  if (split == nullptr) {
    return NOT_CHANGED;
  }

  if (split->GetType() != SPLITD && split->GetType() != SPLITVD) {
    return NOT_CHANGED;
  }
  if (split_name.empty()) {
    split_name = split->GetName();
  } else if (split_name != split->GetName()) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}
} // namespace

vector<string> SplitConvConcatFusionPass::GetNodeTypes() {
  vector<string> result;
  result.push_back(CONCATD);
  result.push_back(CONCATV2D);
  return result;
}

bool SplitConvConcatFusionPass::IsSameQuant(const vector<ge::NodePtr> &vector_quant) const {
  vector<QuantCmpAttr> quant_cmp_attr_vec;
  for (auto &node : vector_quant) {
    if (node->GetType() != ASCEND_QUANT) {
      return false;
    }
    QuantCmpAttr quant_cmp_attr;
    (void)ge::AttrUtils::GetFloat(node->GetOpDesc(), "scale", quant_cmp_attr.scale);
    (void)ge::AttrUtils::GetFloat(node->GetOpDesc(), "offset", quant_cmp_attr.offset);
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "sqrt_mode", quant_cmp_attr.sqrt_mode);
    quant_cmp_attr.data_type = node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    if (quant_cmp_attr_vec.empty()) {
      quant_cmp_attr_vec.push_back(quant_cmp_attr);
      continue;
    }
    if (!(quant_cmp_attr == quant_cmp_attr_vec[0])) {
      return false;
    }
  }
  return true;
}

string SplitConvConcatFusionPass::GetPatternName() { return "SplitConvConcatFusionPass"; }

Status SplitConvConcatFusionPass::DoFusion(ge::ComputeGraph &graph, ge::NodePtr &concat,
                                           vector<ge::NodePtr> &fusion_nodes) {
  (void)fusion_nodes;
  const char_t *env_value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_OFF_CONV_CONCAT_SPLIT, env_value);
  static const std::string off_conv_concat_split_str = (env_value != nullptr) ? std::string(env_value) : "";
  if (!off_conv_concat_split_str.empty()) {
    FE_LOGD("Env[%s] is [%s], skip the SplitConvConcatFusionPass.",
            kOffConvConcatSplit, off_conv_concat_split_str.c_str());
    return SUCCESS;
  }
  FE_LOGD("Define SplitConvConcatFusionPass fusion begin");
  // 1. check the Concat
  if (!concat_optimize_checker.CheckWithQuant(concat)) {
    FE_LOGD("Concat's condition does not match in SplitConvConcatFusionPass");
    return NOT_CHANGED;
  };
  vector<ge::NodePtr> vector_quant;
  vector<ge::NodePtr> vector_dequant;
  ge::NodePtr split_node = nullptr;
  // 2. parse quant and dequant node after split node and before concat node
  if (PatternConcatSplit(concat, split_node, vector_quant, vector_dequant) != SUCCESS) {
    FE_LOGD("Do not match the pattern in SplitConvConcatFusionPass");
    return NOT_CHANGED;
  }
  if (!IsSameQuant(vector_quant)) {
    FE_LOGD("The quant node of concat input link is different");
    return NOT_CHANGED;
  }
  if (CheckOutputSingleRef(concat) != SUCCESS) {
    FE_LOGD("The concat's output is not single ref");
    return NOT_CHANGED;
  }
  // 3. check the Split
  if (!split_optimize_checker.Check(split_node)) {
    FE_LOGI("Split node[%s] is not matched, not do fusion.", split_node->GetName().c_str());
    return NOT_CHANGED;
  };
  ge::NodePtr strided_read_next_node = nullptr;
  bool condition = !vector_quant.empty() && !vector_dequant.empty() &&
                   (split_node->GetAllOutDataAnchors().size() != vector_quant.size() ||
                    vector_quant.size() != vector_dequant.size());
  if (condition) {
    FE_LOGD("The number of quant is not equal to split node or dequant node");
    return NOT_CHANGED;
  }

  OpKernelInfoPtr op_kernel_info_ptr = nullptr;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, STRIDEDREAD);
  if (op_kernel_info_ptr == nullptr) {
    (void)ge::AttrUtils::SetBool(concat->GetOpDesc(), kSupportStridedwriteOptimize, true);
    (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), kSupportStridedreadOptimize, true);
    return NOT_CHANGED;
  }
  FE_LOGD("Start fusion for split node [%s]", split_node->GetName().c_str());
  if (FusionSplit(vector_quant, graph, split_node) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][DoFus] Quantization during split operation failed");
    return FAILED;
  }
  FE_LOGD("Starting fusion for concat node [%s].", concat->GetName().c_str());
  if (FusionConcat(vector_dequant, graph, concat) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][DoFus] Dequant Do Fusion Concat Failed");
    return FAILED;
  }

  if (SetStridedReadAttr(split_node) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][DoFus] Failed to set StridedReadAttr");
    return FAILED;
  }

  for (size_t i = 0; i < concat->GetAllInDataAnchors().size(); ++i) {
    ge::NodePtr strided_write_node = concat->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode();
    FE_CHECK_NOTNULL(strided_write_node);
    auto concat_out_dims = concat->GetOpDesc()->MutableOutputDesc(0)->MutableShape().GetDims();
    if (concat_out_dims.size() < 2) {
      REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][DoFus] ConcatD %s's output dim number is less than 2.",
                      concat->GetName().c_str());
      return FAILED;
    }
    (void)ge::AttrUtils::SetInt(strided_write_node->GetOpDesc(), STRIDE_ATTR_STRIDE, concat_out_dims[1]);
  }
  // 3. set the attribute of Split
  FE_LOGD("Node[%s]: set the attribute of Split.", split_node->GetName().c_str());
  SetGeAttrForSplit(split_node->GetOpDesc(), 1);
  // set the attribute of Concat
  FE_LOGD("Node[%s]: set the attribute of Concat.", concat->GetName().c_str());
  SetGeAttrForConcat(concat->GetOpDesc(), 1);

  FE_LOGD("Define SplitConvConcatFusionPass fusion end.");
  return SUCCESS;
}

Status SplitConvConcatFusionPass::SetStridedReadAttr(const ge::NodePtr &split_node) const {
  for (size_t i = 0; i < split_node->GetAllOutDataAnchors().size(); ++i) {
    FE_CHECK_NOTNULL(split_node->GetOutDataAnchor(i));
    for (size_t j = 0; j < split_node->GetOutDataAnchor(i)->GetPeerInDataAnchors().size(); ++j) {
      ge::NodePtr strided_read_node = split_node->GetOutDataAnchor(i)->GetPeerInDataAnchors().at(j)->GetOwnerNode();
      FE_CHECK_NOTNULL(strided_read_node);
      if (strided_read_node->GetOpDesc()->GetType() == STRIDEDREAD) {
        FE_CHECK_NOTNULL(split_node->GetOpDesc()->MutableInputDesc(0));
        auto split_input_dims = split_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDims();
        if (split_input_dims.size() < MIN_SPLIT_DIM_NUM) {
          REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][DoFus] split %s's input dim number is less than 2.",
                          split_node->GetName().c_str());
          return FAILED;
        }
        (void)ge::AttrUtils::SetInt(strided_read_node->GetOpDesc(), STRIDE_ATTR_STRIDE, split_input_dims[1]);
      }
    }
  }
  return SUCCESS;
}

inline bool IsConv2DSingleRef(const ge::NodePtr &node) {
  return node->GetType() == CONV2D && node->GetOutDataNodes().size() == 1;
}

Status SplitConvConcatFusionPass::CheckOutputSingleRef(ge::NodePtr &concat_node) {
  std::vector<std::string> op_type_before = {LEAKYRELU, RELU, DEQUANT};
  ge::NodePtr pre_node = nullptr;
  ge::NodePtr pre_pre_node = nullptr;
  size_t count_output_single = 0;
  for (auto &node : concat_node->GetInDataNodes()) {
    // meet Conv2D, return
    if (IsConv2DSingleRef(node)) {
      count_output_single++;
      continue;
    }
    if (node->GetOutDataNodes().size() != 1) {
      return NOT_CHANGED;
    }
    if (std::find(op_type_before.begin(), op_type_before.end(), node->GetType()) != op_type_before.end()) {
      pre_node = node->GetInDataNodes().at(0);
      if (IsConv2DSingleRef(pre_node)) {
        count_output_single++;
        continue;
      }
      if (pre_node->GetOutDataNodes().size() != 1) {
        return NOT_CHANGED;
      }
    }
    if (pre_node->GetType() == DEQUANT) {
      pre_pre_node = pre_node->GetInDataNodes().at(0);
      if (IsConv2DSingleRef(pre_pre_node)) {
        count_output_single++;
        continue;
      }
      if (pre_pre_node->GetOutDataNodes().size() != 1) {
        return NOT_CHANGED;
      }
    }
  }
  if (count_output_single != concat_node->GetInDataNodes().size()) {
    FE_LOGD("conv_output_single's size is %zu, conv2d's size is %zu.", count_output_single,
            concat_node->GetInDataNodes().size());
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SplitConvConcatFusionPass::FusionSplit(vector<ge::NodePtr> &vector_quant, ge::ComputeGraph &graph,
                                              ge::NodePtr &split_node) {
  if (!vector_quant.empty()) {
    ge::OpDescPtr new_move_quant = ge::AttrUtils::CopyOpDesc(vector_quant[0]->GetOpDesc());
    auto quant_data_type = vector_quant[0]->GetOpDesc()->GetOutputDesc(0).GetDataType();
    new_move_quant->SetName(new_move_quant->GetName() + "_move");
    ge::NodePtr new_move_quant_node = graph.AddNode(new_move_quant);
    FE_CHECK_NOTNULL(new_move_quant_node);
    FE_CHECK_NOTNULL(split_node->GetInDataAnchor(0));
    InsertNode(split_node->GetInDataAnchor(0)->GetPeerOutAnchor(), split_node->GetInDataAnchor(0),
               new_move_quant_node, quant_data_type);
  }

  for (size_t i = 0; i < split_node->GetAllOutDataAnchors().size(); i++) {
    auto quant_data_type = split_node->GetOpDesc()->GetOutputDesc(i).GetOriginDataType();
    ge::NodePtr strided_read_node = nullptr;
    if (!vector_quant.empty() && i < vector_quant.size()) {
      FE_CHECK(vector_quant[i] == nullptr || vector_quant[i]->GetOpDesc()== nullptr,
        FE_LOGD("Get quant op desc unsuccessful."), return FAILED);
      quant_data_type = vector_quant[i]->GetOpDesc()->GetOutputDesc(0).GetDataType();
      std::shared_ptr<ge::OpDesc> strided_read_opdesc;
      CreateStridedRead(vector_quant[i], strided_read_opdesc);
      strided_read_node = graph.AddNode(strided_read_opdesc);
      FE_CHECK_NOTNULL(strided_read_node);
      InsertNode(split_node->GetOutDataAnchor(i), vector_quant[i]->GetInDataAnchor(0), strided_read_node);
      if (graph.RemoveNode(vector_quant[i]) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][FusSplit] failed to remove %s",
                        vector_quant[i]->GetName().c_str());
        return FAILED;
      }
    } else {
      FE_CHECK_NOTNULL(split_node->GetOutDataAnchor(i));
      for (size_t j = 0; j < split_node->GetOutDataAnchor(i)->GetPeerInDataAnchors().size(); j++) {
        ge::NodePtr conv2d_node = split_node->GetOutDataAnchor(i)->GetPeerInDataAnchors().at(j)->GetOwnerNode();
        if (conv2d_node->GetType() == CONV2D) {
          std::shared_ptr<ge::OpDesc> strided_read_opdesc;
          CreateStridedRead(conv2d_node, strided_read_opdesc);
          strided_read_node = graph.AddNode(strided_read_opdesc);
          FE_CHECK_NOTNULL(strided_read_node);
          InsertNode(split_node->GetOutDataAnchor(i), conv2d_node->GetInDataAnchor(0), strided_read_node);
        }
      }
    }
    auto output_desc = split_node->GetOpDesc()->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      continue;
    }
    if (GetNC1HWC0Shape(output_desc, quant_data_type) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][FusSplit] Failed to get shape of NC1HWC0.");
      return FAILED;
    }
    JudgeOp(strided_read_node);
  }
  auto quant_data_type = split_node->GetOpDesc()->GetInputDesc(0).GetOriginDataType();
  if (!vector_quant.empty()) {
    quant_data_type = vector_quant[0]->GetOpDesc()->GetOutputDesc(0).GetDataType();
  }
  if (GetNC1HWC0Shape(split_node->GetOpDesc()->MutableInputDesc(0), quant_data_type) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][FusSplit] Failed to get shape of NC1HWC0.");
    return FAILED;
  }
  return SUCCESS;
}

Status SplitConvConcatFusionPass::FusionConcat(vector<ge::NodePtr> &vector_dequant, ge::ComputeGraph &graph,
                                               ge::NodePtr &concat) {
  FE_CHECK_NOTNULL(concat->GetOutDataAnchor(0));
  if (concat->GetOutDataAnchor(0)->GetPeerInDataAnchors().size()==0) {
    FE_LOGD("Concat's output node is null");
    return FAILED;
  }
  ge::NodePtr concat_next_node = concat->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  ge::OpDescPtr new_move_quant_after_concat;
  if (!vector_dequant.empty() && concat_next_node->GetType() == QUANT) {
    if (graph.RemoveNode(concat_next_node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][FusConcat] Failed to remove %s",
                      concat_next_node->GetName().c_str());
      return FAILED;
    }
  } else if (vector_dequant.empty() && (concat_next_node->GetType() == LEAKYRELU ||
            concat_next_node->GetType() == RELU || concat_next_node->GetType() == QUANT)) {
    if (graph.RemoveNode(concat_next_node) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][SplitConvConcatFus][FusConcat] Failed to remove %s",
                      concat_next_node->GetName().c_str());
      return FAILED;
    }
  }
  ge::DataType quant_data_type = concat_next_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  for (size_t i = 0; i < concat->GetAllInDataAnchors().size(); ++i) {
    ge::NodePtr strided_write_node = nullptr;
    if (!vector_dequant.empty() && i < vector_dequant.size()) {
      std::shared_ptr<ge::OpDesc> strided_write_opdesc;
      CreateStridedWrite(vector_dequant[i], strided_write_opdesc);
      strided_write_node = graph.AddNode(strided_write_opdesc);
      FE_CHECK_NOTNULL(strided_write_node);
      FE_CHECK_NOTNULL(concat->GetInDataAnchor(i));
      InsertNode(concat->GetInDataAnchor(i)->GetPeerOutAnchor(), concat->GetInDataAnchor(i), strided_write_node);

      if (concat_next_node->GetType() == QUANT) {
        ge::OpDescPtr move_quant = ge::AttrUtils::CopyOpDesc(concat_next_node->GetOpDesc());
        move_quant->SetName(move_quant->GetName() + "_" + std::to_string(i));
        ge::NodePtr move_quant_node = graph.AddNode(move_quant);
        FE_CHECK_NOTNULL(move_quant_node);
        FE_CHECK_NOTNULL(strided_write_node->GetInDataAnchor(0));
        InsertNode(strided_write_node->GetInDataAnchor(0)->GetPeerOutAnchor(), strided_write_node->GetInDataAnchor(0),
                   move_quant_node, quant_data_type);
      }
    } else {
      FE_CHECK_NOTNULL(concat->GetInDataAnchor(i));
      FE_CHECK_NOTNULL(concat->GetInDataAnchor(i)->GetPeerOutAnchor());
      ge::NodePtr conv2d_node = concat->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode();
      std::shared_ptr<ge::OpDesc> strided_write_opdesc;
      CreateStridedWrite(conv2d_node, strided_write_opdesc);
      strided_write_node = graph.AddNode(strided_write_opdesc);
      FE_CHECK_NOTNULL(strided_write_node);
      FE_CHECK_NOTNULL(concat->GetInDataAnchor(i));
      InsertNode(concat->GetInDataAnchor(i)->GetPeerOutAnchor(), concat->GetInDataAnchor(i), strided_write_node);

      if ((concat_next_node->GetType() == LEAKYRELU) || (concat_next_node->GetType() == RELU) ||
          (concat_next_node->GetType() == QUANT)) {
        ge::OpDescPtr move_relu = ge::AttrUtils::CopyOpDesc(concat_next_node->GetOpDesc());
        move_relu->SetName(move_relu->GetName() + "_" + std::to_string(i));
        ge::NodePtr move_relu_node = graph.AddNode(move_relu);
        FE_CHECK_NOTNULL(move_relu_node);
        FE_CHECK_NOTNULL(strided_write_node->GetInDataAnchor(0));
        InsertNode(strided_write_node->GetInDataAnchor(0)->GetPeerOutAnchor(), strided_write_node->GetInDataAnchor(0),
                   move_relu_node, quant_data_type);
      }
    }
    auto input_desc_ptr = concat->GetOpDesc()->MutableInputDesc(i);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    GetNC1HWC0Shape(input_desc_ptr, quant_data_type);
    JudgeOp(strided_write_node);
  }
  GetNC1HWC0Shape(concat->GetOpDesc()->MutableOutputDesc(0), quant_data_type);
  return SUCCESS;
}

Status SplitConvConcatFusionPass::PatternConcatSplit(ge::NodePtr &concat, ge::NodePtr &split_node,
                                                     vector<ge::NodePtr> &vector_quant,
                                                     vector<ge::NodePtr> &vector_dequant) {
  SubGraphStructure sub_graph_structure;
  string split_name;
  ge::NodePtr quant = nullptr;
  ge::NodePtr dequant = nullptr;
  ge::NodePtr split = nullptr;
  for (ge::InDataAnchorPtr &concat_in_data_anchor : concat->GetAllInDataAnchors()) {
    ge::OutDataAnchorPtr concat_peer_out_anchor = concat_in_data_anchor->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(concat_peer_out_anchor);
    ge::NodePtr concat_prev_node = concat_peer_out_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(concat_prev_node);
    ge::InDataAnchorPtr quant_in_data_anchor_ptr;
    if (concat_prev_node->GetType() == LEAKYRELU || concat_prev_node->GetType() == RELU) {
      ge::InDataAnchorPtr concat_prev_in_data_anchor = concat_prev_node->GetInDataAnchor(0);
      FE_CHECK_NOTNULL(concat_prev_in_data_anchor);
      ge::OutDataAnchorPtr concat_prev_peer_out_anchor = concat_prev_in_data_anchor->GetPeerOutAnchor();
      FE_CHECK_NOTNULL(concat_prev_peer_out_anchor);
      quant_in_data_anchor_ptr = PatternPrevConv2dWithQuant(concat_prev_peer_out_anchor, quant, dequant);
    } else {
      quant_in_data_anchor_ptr = PatternPrevConv2dWithQuant(concat_peer_out_anchor, quant, dequant);
    }
    if (quant_in_data_anchor_ptr != nullptr) {
      vector_quant.push_back(quant);
      vector_dequant.push_back(dequant);
    }
    Status ret =
        CheckQuantAndConvBeforeConcat(quant_in_data_anchor_ptr, concat_prev_node, sub_graph_structure);
    if (ret != SUCCESS) {
      FE_LOGD("The Quant or conv2D operation preceding the concat input link does not match.");
      return ret;
    }
    ret = ParseAndCheckSplitNode(quant_in_data_anchor_ptr, concat_prev_node, split, sub_graph_structure, split_name);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  size_t count = 0;
  FE_CHECK_NOTNULL(split);
  for (size_t i = 0; i < split->GetAllOutDataAnchors().size(); ++i) {
    FE_CHECK_NOTNULL(split->GetOutDataAnchor(i));
    for (size_t j = 0; j < split->GetOutDataAnchor(i)->GetPeerInDataAnchors().size(); ++j) {
      count++;
    }
  }

  if (concat->GetAllInDataAnchors().size() != count) {
    return NOT_CHANGED;
  }
  split_node = split;
  return SUCCESS;
}

ge::InDataAnchorPtr SplitConvConcatFusionPass::PatternPrevConv2dWithQuant(ge::OutDataAnchorPtr out_anchor,
                                                                          ge::NodePtr &quant, ge::NodePtr &dequant) {
  ge::NodePtr ascend_dequant = out_anchor->GetOwnerNode();
  if (ascend_dequant == nullptr || ascend_dequant->GetType() != DEQUANT) {
    FE_LOGD("Dequant is nullptr or type is not DEQUANT.");
    return nullptr;
  }

  ge::InDataAnchorPtr dequant_in_anchor = ascend_dequant->GetInDataAnchor(0);
  if (dequant_in_anchor == nullptr) {
    return nullptr;
  }
  ge::OutDataAnchorPtr dequant_peer_out_anchor = dequant_in_anchor->GetPeerOutAnchor();
  if (dequant_peer_out_anchor == nullptr) {
    return nullptr;
  }
  ge::NodePtr conv2d = dequant_peer_out_anchor->GetOwnerNode();
  if (conv2d == nullptr || conv2d->GetType() != CONV2D) {
    return nullptr;
  }

  ge::InDataAnchorPtr conv2d_in_anchor = conv2d->GetInDataAnchor(0);
  if (conv2d_in_anchor == nullptr) {
    return nullptr;
  }
  ge::OutDataAnchorPtr conv2d_peer_out_anchor = conv2d_in_anchor->GetPeerOutAnchor();
  if (conv2d_peer_out_anchor == nullptr) {
    return nullptr;
  }
  ge::NodePtr ascend_quant = conv2d_peer_out_anchor->GetOwnerNode();
  if (ascend_quant == nullptr || ascend_quant->GetType() != QUANT) {
    FE_LOGD("Quant is nullptr or type is not QUANT.");
    return nullptr;
  }
  quant = ascend_quant;
  dequant = ascend_dequant;
  return quant->GetInDataAnchor(0);
}

REG_PASS("SplitConvConcatFusionPass", BUILT_IN_GRAPH_PASS,
         SplitConvConcatFusionPass, SINGLE_SCENE_OPEN | FE_PASS);
}  // namespace fe
