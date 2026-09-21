/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/node_optimizer/stridedread_optimizer.h"
#include <memory>
#include <vector>
#include "common/aicore_util_attr_define.h"
#include "common/op_info_common.h"
#include "common/fe_inner_attr_define.h"
#include "common/lxfusion_json_util.h"
#include "graph/debug/ge_attr_define.h"
#include "param_calculate/tensor_compute_util.h"


using ToOpStructPtr = std::shared_ptr<fe::ToOpStruct_t>;
namespace fe {
namespace {
bool CheckSupportStrided(const ge::NodePtr &node) {
  for (auto &out_node : node->GetOutDataNodes()) {
    if (out_node == nullptr) {
      continue;
    }
    auto op_desc = out_node->GetOpDesc();
    string op_type = op_desc->GetType();
    if (op_type == ASCEND_QUANT) {
      auto quant_out = out_node->GetOutDataNodes();
      if (quant_out.empty()) {
        return false;
      }
      op_type = out_node->GetOutDataNodes().at(0)->GetType();
    }
    if (op_type != CONV2D) {
      return false;
    }
    auto input_tensor = op_desc->MutableInputDesc(0);
    if (input_tensor == nullptr) {
      return false;
    }
    if (ge::GetPrimaryFormat(input_tensor->GetFormat()) != ge::FORMAT_NC1HWC0) {
      return false;
    }
  }
  return true;
}

void UpdateTensorDesc(ge::GeTensorDescPtr &tensor_desc, const ge::DataType &data_type) {
  if (tensor_desc == nullptr) {
    return;
  }
  tensor_desc->SetOriginDataType(data_type);
  tensor_desc->SetDataType(data_type);
  const ge::GeShape &ori_shape = tensor_desc->GetOriginShape();
  ge::GeShape new_shape;
  ShapeAndFormat shape_and_format_info = {ori_shape,
                                          tensor_desc->MutableShape(),
                                          tensor_desc->GetOriginFormat(),
                                          static_cast<ge::Format>(ge::GetPrimaryFormat(tensor_desc->GetFormat())),
                                          tensor_desc->GetDataType(),
                                          static_cast<int64_t>(OpImplType::EN_IMPL_HW_TBE)};
  (void)GetShapeAccordingToFormat(shape_and_format_info);
}
}  // namespace

void StridedReadOptimizer::SetAttrForSplit(const ge::NodePtr &node, const ge::OpDescPtr &op_desc) const {
  FE_LOGD("Node[%s] start to set split attribute.", node->GetName().c_str());
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 1);
}

int32_t StridedReadOptimizer::GetRealSplitDim(const ge::OpDescPtr &op_desc) const {
  ge::GeTensorDesc input_tensor = op_desc->GetInputDesc(0);
  ge::GeTensorDesc output_tensor = op_desc->GetOutputDesc(0);
  std::vector<int64_t> input_shape = input_tensor.GetShape().GetDims();
  std::vector<int64_t> output_shape = output_tensor.GetShape().GetDims();
  size_t shape_size = input_shape.size();
  size_t out_shape_size = output_shape.size();
  if (shape_size != out_shape_size) {
    FE_LOGW("Cur node [%s], input_shape_size [%zu] does not match output_shape_size [%zu]",
            op_desc->GetName().c_str(), shape_size, out_shape_size);
    return -1;
  }
  for (size_t index = 0; index < shape_size; ++index) {
    if (input_shape[index] != output_shape[index]) {
      return index;
    }
  }
  return -1;
}

Status StridedReadOptimizer::SetStrideReadInfoForOutputs(const ge::NodePtr &split_node,
                                                         const int32_t &split_dim) const {
  std::string node_name = split_node->GetName();
  FE_LOGD("Node:[%s] begin to set slice info.", node_name.c_str());

  ge::NodePtr peer_node = nullptr;
  ge::InDataAnchorPtr in_anchor = nullptr;

  ge::GeTensorDescPtr split_in_tensor = split_node->GetOpDesc()->MutableInputDesc(0);
  FE_CHECK_NOTNULL(split_in_tensor);
  auto out_anchors = split_node->GetAllOutDataAnchors();
  int64_t pre_offset = 0;
  size_t idx = 0;
  for (auto &out_anchor : out_anchors) {
    auto peer_anchors = out_anchor->GetPeerInDataAnchors();
    for (auto &peer_in_anchor : peer_anchors) {
      FE_CHECK_NOTNULL(peer_in_anchor);
      peer_node = peer_in_anchor->GetOwnerNode();
      FE_CHECK_NOTNULL(peer_node);
      idx = peer_in_anchor->GetIdx();
      if (FeedToOpStructInfo(peer_node, pre_offset, idx, split_in_tensor, split_dim) != fe::SUCCESS) {
        REPORT_FE_ERROR("[SplitOptimize][FeedInfo] FeedToOpStructInfo of node [%s] failed.",
                        peer_node->GetName().c_str());
        return fe::FAILED;
      }
    }
  }
  FE_LOGD("Split node[%s] set stridedRead info successfully.", node_name.c_str());
  return fe::SUCCESS;
}

void StridedReadOptimizer::CalSliceOffset(const std::vector<int64_t> &output_shape,
                                          ge::DataType data_type, int64_t &output_offset_buff,
                                          const int32_t &concat_dim) const {
  int32_t shape_size = static_cast<int32_t>(output_shape.size());
  if (shape_size < concat_dim + 1) {
    FE_LOGE("Invalid output_shape size [%d] for concat_dim [%d].", shape_size, concat_dim);
    return;
  }

  int64_t element_cnt = 1;
  int64_t tmp = 0;
  for (int32_t i = concat_dim; i < shape_size; ++i) {
    tmp = output_shape[i];
    if (tmp == 0) {
      FE_LOGE("Invalid output_shape_idx value [%d], it is 0.", i);
      return;
    }
    element_cnt *= tmp;
  }
  int32_t flag = 1;
  output_offset_buff = 1;
  TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, output_offset_buff, flag);
  return;
}

Status StridedReadOptimizer::FeedToOpStructInfo(ge::NodePtr &node, int64_t &pre_offset, const size_t &idx,
                                                const ge::GeTensorDescPtr &split_in_tensor,
                                                const int32_t &split_dim) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (FeedToOpStructInfo(op_desc, pre_offset, idx, split_in_tensor, split_dim) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status StridedReadOptimizer::FeedToOpStructInfo(ge::OpDescPtr& op_desc, int64_t &pre_offset, const size_t &idx,
                                                const ge::GeTensorDescPtr &split_in_tensor,
                                                const int32_t &split_dim) const {
  ge::GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(idx);
  FE_CHECK_NOTNULL(tensor_desc);

  size_t in_size = op_desc->GetInputOffset().size();
  if (in_size == 0) {
    in_size = op_desc->GetAllInputsSize();
  }
  std::vector<int64_t> in_shape = tensor_desc->GetShape().GetDims();
  std::vector<std::vector<int64_t>> split_inputs_shape(in_size, std::vector<int64_t>(in_shape.size(), 0));
  std::vector<int64_t> input_i(in_size, 0);
  split_inputs_shape[idx] = in_shape;
  int64_t input_offset_buff = -1;
  FE_LOGD("Concat input_node %s offset size is %zu.", op_desc->GetName().c_str(), in_size);

  CalSliceOffset(in_shape, tensor_desc->GetDataType(), input_offset_buff, split_dim);
  if (input_offset_buff == -1) {
    FE_LOGE("Concat input_node %s is unexpected.", op_desc->GetName().c_str());
    return fe::FAILED;
  }
  FE_LOGD("Op[%s] set attribute input_i[%ld] successfully.", op_desc->GetName().c_str(), pre_offset);

  input_i[idx] = pre_offset;
  op_desc->SetInputOffset(input_i);
  pre_offset += input_offset_buff;
  tensor_desc->SetShape(split_in_tensor->GetShape());
  tensor_desc->SetOriginShape(split_in_tensor->GetOriginShape());
  ToOpStructPtr optimize_info = nullptr;
  FE_MAKE_SHARED(optimize_info = std::make_shared<ToOpStruct_t>(), return fe::FAILED);
  optimize_info->slice_input_shape = split_inputs_shape;
  SetStridedInfoToNode(op_desc, optimize_info, kSplitCOptimizeInfoPtr, kSplitCOptimizeInfoStr);
  (void)ge::AttrUtils::SetBool(op_desc, NEED_RE_PRECOMPILE, true);
  FE_LOGD("Op[%s] set ext split strided attr successfully.", op_desc->GetName().c_str());
  return fe::SUCCESS;
}

void StridedReadOptimizer::MoveQuantBeforeSplit(ge::ComputeGraph &graph, const ge::NodePtr &node) const {
  auto out_nodes = node->GetOutDataNodes();
  for (auto &out_node : out_nodes) {
    if (out_node->GetType() != ASCEND_QUANT) {
      return;
    }
  }
  if (out_nodes.empty()) {
    return;
  }
  ge::OpDescPtr new_move_quant = ge::AttrUtils::CopyOpDesc(out_nodes.at(0)->GetOpDesc());
  ge::DataType quant_data_type = out_nodes.at(0)->GetOpDesc()->GetOutputDesc(0).GetDataType();
  new_move_quant->SetName(new_move_quant->GetName() + "_move");
  ge::NodePtr new_quant_node = graph.AddNode(new_move_quant);
  auto last_node = node->GetInDataNodes().at(0);
  (void)ge::GraphUtils::InsertNodeBetweenDataAnchors(last_node->GetOutDataAnchor(0),
                                                     node->GetInDataAnchor(0), new_quant_node);
  // update shape&dtype
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if(op_desc->MutableInputDesc(0) == nullptr || new_move_quant->MutableInputDesc(0) == nullptr ||
     new_move_quant->MutableOutputDesc(0) == nullptr) {
       return;
  }
  ge::GeShape shape = op_desc->MutableInputDesc(0)->GetShape();
  ge::GeShape origin_shape = op_desc->MutableInputDesc(0)->GetOriginShape();
  new_move_quant->MutableInputDesc(0)->SetShape(shape);
  new_move_quant->MutableInputDesc(0)->SetOriginShape(origin_shape);
  auto output_desc = new_move_quant->MutableOutputDesc(0);
  output_desc->SetOriginShape(origin_shape);
  UpdateTensorDesc(output_desc, quant_data_type);
  for (auto &input_desc : op_desc->GetAllInputsDescPtr()) {
    UpdateTensorDesc(input_desc, quant_data_type);
  }
  for (auto &output_desc_ptr : op_desc->GetAllOutputsDescPtr()) {
    UpdateTensorDesc(output_desc_ptr, quant_data_type);
  }
  for (auto &quant : out_nodes) {
    (void)graph.RemoveNode(quant);
  }
}

void StridedReadOptimizer::DoOptimizeForSplit(ge::ComputeGraph &graph) const {
  FE_LOGD("Starting StridedReadOptimizer for graph [%s].", graph.GetName().c_str());
  ge::ComputeGraph::Vistor<ge::NodePtr> nodes = graph.GetAllNodes();
  for (auto &node : nodes) {
    bool support_optimize = false;
    ge::OpDescPtr op_desc = node->GetOpDesc();
    string node_name = op_desc->GetName();
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kSupportStridedreadOptimize, support_optimize);
    if (!support_optimize) {
      continue;
    }

    int32_t split_dim = GetRealSplitDim(node->GetOpDesc());
    if (split_dim == -1) {
      continue;
    }
    if (!CheckSupportStrided(node)) {
      continue;
    }
    MoveQuantBeforeSplit(graph, node);
    if (SetStrideReadInfoForOutputs(node, split_dim) != SUCCESS) {
      continue;
    }
    SetAttrForSplit(node, op_desc);
    FE_LOGD("Node:[%s] finish to optimize.", node_name.c_str());
  }
  FE_LOGD("End StridedReadOptimizer setup for graph [%s].", graph.GetName().c_str());
}
}  // namespace fe
