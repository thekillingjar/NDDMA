/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/node_optimizer/stridedwrite_optimizer.h"
#include <memory>
#include <vector>
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_attr_define.h"
#include "common/lxfusion_json_util.h"
#include "graph/debug/ge_attr_define.h"
#include "param_calculate/tensor_compute_util.h"


using ToOpStructPtr = std::shared_ptr<fe::ToOpStruct_t>;
namespace fe {
const std::string kOutputOffsetForBufferFusion = "_output_offset_for_buffer_fusion";
void StridedWriteOptimizer::SetAttrForConcat(const ge::NodePtr &node, const ge::OpDescPtr &op_desc) const {
  FE_LOGD("Node[%s] start to set concat attribute", node->GetName().c_str());
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 1);
}

int32_t StridedWriteOptimizer::GetRealConcatDim(const ge::OpDescPtr &op_desc) const {
  ge::GeTensorDesc input_tensor = op_desc->GetInputDesc(0);
  ge::GeTensorDesc output_tensor = op_desc->GetOutputDesc(0);
  std::vector<int64_t> input_shape = input_tensor.GetShape().GetDims();
  std::vector<int64_t> output_shape = output_tensor.GetShape().GetDims();
  size_t shape_size = input_shape.size();
  size_t out_shape_size = output_shape.size();
  if (shape_size != out_shape_size) {
    FE_LOGW("Cur node[%s], input_shape_size[%zu] does not match output_shape_size[%zu]",
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

Status StridedWriteOptimizer::SetStrideWriteInfoPhonyConcat(const ge::NodePtr &concat_node,
                                                            const std::vector<int64_t> &concat_out_shape) const {
  ge::NodePtr peer_node = nullptr;
  ge::OutDataAnchorPtr out_anchor = nullptr;
  size_t cnt = 0;
  size_t idx = 0;
  ge::OpDescPtr op_desc = concat_node->GetOpDesc();
  auto in_anchors = concat_node->GetAllInDataAnchors();
  int32_t phony_concat_dim = GetRealConcatDim(op_desc);
  if (phony_concat_dim == -1) {
    REPORT_FE_ERROR("[ConcatOptimize][SetSwInfo] The value of node phony_concat_dim is negative.");
    return FAILED;
  }
  int64_t offset = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kPhonyConcatOffset, offset);
  for (auto &in_anchor : in_anchors) {
    out_anchor = in_anchor->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(out_anchor);
    peer_node = out_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(peer_node);
    ++cnt;
    bool last_input = (cnt == in_anchors.size()) ? true : false;
    idx = out_anchor->GetIdx();
    if (FeedToOpStructInfo(peer_node, idx, concat_out_shape, last_input,
                           phony_concat_dim, offset, true) != fe::SUCCESS) {
      REPORT_FE_ERROR("[ConcatOptimize][SetSwInfo] FeedToOpStructInfo of node[%s] failed.",
                      peer_node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status StridedWriteOptimizer::SetStrideWriteInfoForInputs(const ge::NodePtr &concat_node,
                                                          const int32_t &concat_dim) const {
  std::string node_name = concat_node->GetName();
  FE_LOGD("node:[%s] begin to set slice info.", node_name.c_str());

  ge::NodePtr peer_node = nullptr;
  ge::OutDataAnchorPtr out_anchor = nullptr;

  auto concat_out_tensor = concat_node->GetOpDesc()->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(concat_out_tensor);
  std::vector<int64_t> concat_out_shape = concat_out_tensor->GetShape().GetDims();
  auto in_anchors = concat_node->GetAllInDataAnchors();
  size_t in_size = in_anchors.size();
  size_t cnt = 0;
  size_t idx = 0;
  int64_t offset = 0;

  for (auto &in_anchor : in_anchors) {
    out_anchor = in_anchor->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(out_anchor);
    peer_node = out_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(peer_node);
    ++cnt;
    bool is_last_input = (cnt == in_size) ? true : false;
    idx = out_anchor->GetIdx();
    if (FeedToOpStructInfo(peer_node, idx, concat_out_shape, is_last_input, concat_dim, offset) != fe::SUCCESS) {
      REPORT_FE_ERROR("[ConcatOptimize][FeedInfo] FeedToOpStructInfo of node[%s] failed.",
                      peer_node->GetName().c_str());
      return fe::FAILED;
    }
    if (peer_node->GetType() == OP_TYPE_PHONY_CONCAT) {
      if (SetStrideWriteInfoPhonyConcat(peer_node, concat_out_shape) != SUCCESS) {
        REPORT_FE_ERROR("[ConcatOptimize][SetSwInfo] SetStrideWriteInfoPhonyConcat of node[%s] failed.",
                        peer_node->GetName().c_str());
        return FAILED;
      }
    }
  }
  FE_LOGD("Concat node[%s] set strideWrite info successfully.", node_name.c_str());
  return fe::SUCCESS;
}

void StridedWriteOptimizer::CalSliceOffset(const std::vector<int64_t> &output_shape,
                                           ge::DataType data_type, int64_t &output_offset_buff,
                                           const int32_t &concat_dim) const {
  int32_t shape_size = static_cast<int32_t>(output_shape.size());
  if (shape_size < concat_dim + 1) {
    FE_LOGE("Invalid output_shape size [%d], concat_dim [%d].", shape_size, concat_dim);
    return;
  }

  int64_t element_cnt = 1;
  int64_t tmp = 0;
  for (int32_t i = concat_dim; i < shape_size; ++i) {
    tmp = output_shape[i];
    if (tmp == 0) {
      FE_LOGE("Invalid output_shape_idx[%d], value should not be 0.", i);
      return;
    }
    element_cnt *= tmp;
  }
  int32_t flag = 1;
  output_offset_buff = 1;
  TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, output_offset_buff, flag);
  return;
}

Status StridedWriteOptimizer::FeedToOpStructInfo(ge::NodePtr &node, const size_t &idx,
                                                 const std::vector<int64_t> &concat_out_shape,
                                                 const bool &is_last_input,
                                                 const int32_t &concat_dim,
                                                 int64_t &phony_concat_offset,
                                                 const bool &set_offset) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (FeedToOpStructInfo(op_desc, idx, concat_out_shape, is_last_input, concat_dim,
                         phony_concat_offset, set_offset) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status StridedWriteOptimizer::FeedToOpStructInfo(ge::OpDescPtr& op_desc, const size_t &idx,
                                                 const std::vector<int64_t> &concat_out_shape,
                                                 const bool &is_last_input,
                                                 const int32_t &concat_dim,
                                                 int64_t &phony_concat_offset,
                                                 const bool &set_offset) const {
  ge::GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(idx);
  FE_CHECK_NOTNULL(tensor_desc);

  size_t out_size = op_desc->GetOutputsSize();
  std::vector<int64_t> output_shape = tensor_desc->GetShape().GetDims();
  std::vector<std::vector<int64_t>> concat_outputs_shape(out_size, std::vector<int64_t>(output_shape.size(), 0));
  std::vector<int64_t> outputs_offset(out_size, 0);
  std::vector<int64_t> output_i(out_size, 0);
  concat_outputs_shape[idx] = concat_out_shape;
  int64_t output_offset_buff = -1;
  FE_LOGD("Concat output_node %s offset size is %zu.", op_desc->GetName().c_str(), out_size);

  CalSliceOffset(output_shape, tensor_desc->GetDataType(), output_offset_buff, concat_dim);
  if (output_offset_buff == -1) {
    FE_LOGE("Concat input_node [%s] is unexpected.", op_desc->GetName().c_str());
    return fe::FAILED;
  }
  if (set_offset) {
    output_i[idx] = phony_concat_offset;
  } else {
    output_i[idx] = 0;
    phony_concat_offset += output_offset_buff;
  }
  if (op_desc->GetType() == OP_TYPE_PHONY_CONCAT) {
    ge::AttrUtils::SetInt(op_desc, kPhonyConcatOffset, phony_concat_offset);
  }
  if (is_last_input) {
    output_offset_buff = 0;
    FE_LOGD("Concat input_node[%s] is the final input; therefore, there is no need to calculate the output_offset_buff.",
            op_desc->GetName().c_str());
  }
  FE_LOGD("Concat input_node[%s], phony_concat_offset is [%ld], and output_offset_buff is [%ld].",
          op_desc->GetName().c_str(), phony_concat_offset, output_offset_buff);
  outputs_offset[idx] = output_offset_buff;

  (void)ge::AttrUtils::SetListInt(op_desc, kOutputOffsetForBufferFusion, outputs_offset);
  FE_LOGD("Op[%s] set attr _output_offet_for_buffer_fusion[%ld] successfully.",
          op_desc->GetName().c_str(), output_offset_buff);

  op_desc->SetOutputOffset(output_i);

  ToOpStructPtr optimize_info = nullptr;
  FE_MAKE_SHARED(optimize_info = std::make_shared<ToOpStruct_t>(), return fe::FAILED);
  optimize_info->slice_output_shape = concat_outputs_shape;
  SetStridedInfoToNode(op_desc, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);
  (void)ge::AttrUtils::SetBool(op_desc, NEED_RE_PRECOMPILE, true);
  FE_LOGD("Op[%s] set the extended concatenated strided attribute successfully.", op_desc->GetName().c_str());
  return fe::SUCCESS;
}

void StridedWriteOptimizer::DoOptimizeForConcat(const ge::ComputeGraph &graph) const {
  FE_LOGD("Starting the StridedWriteOptimizer for graph[%s].", graph.GetName().c_str());
  ge::ComputeGraph::Vistor<ge::NodePtr> nodes = graph.GetAllNodes();
  for (auto &node : nodes) {
    bool support_optimize = false;
    ge::OpDescPtr op_desc = node->GetOpDesc();
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kSupportStridedwriteOptimize, support_optimize);
    if (!support_optimize) {
      continue;
    }

    int32_t concat_dim = GetRealConcatDim(node->GetOpDesc());
    if (concat_dim == -1) {
      continue;
    }
    auto input_tensor = op_desc->MutableInputDesc(0);
    if (input_tensor == nullptr) {
      continue;
    }
    if (ge::GetPrimaryFormat(input_tensor->GetFormat()) != ge::FORMAT_NC1HWC0) {
      op_desc->DelAttr(kSupportStridedwriteOptimize);
      continue;
    }
    if (SetStrideWriteInfoForInputs(node, concat_dim) != SUCCESS) {
      continue;
    }
    SetAttrForConcat(node, op_desc);
    FE_LOGD("Node:[%s] has finished optimizing.", op_desc->GetName().c_str());
  }
  FE_LOGD("Completed StridedWriteOptimizer for graph [%s].", graph.GetName().c_str());
}
}  // namespace fe
