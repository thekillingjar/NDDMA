/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/memory_optimize/concat_notask_pass.h"
#include "graph/utils/node_utils.h"
#include "runtime/mem.h"
#include "graph/utils/type_utils.h"
#include "common/memory/mem_type_utils.h"
#include "common/checker.h"
#include "graph/utils/graph_utils.h"

namespace ge {
const std::string ATTR_NAME_L2_FUSION_EXTEND_PTR = "l2_fusion_extend_content";
const char_t *const kPartSrcGraph = "part_src_graph";
const std::string kATTR_CAN_REUSED_FOR_CONCAT_OPTIMIZE = "can_reused_for_concat_optimize";
const int32_t VALID_NC_DIM_SIZE = 2;
constexpr int32_t TENSOR_ALIGN_SIZE  = 32;
Status ConcatNotaskPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (ge::GraphUtils::IsSingleOpScene(graph)) {
    GELOGI("Single op scene has no need to do concat optimize.");
    return SUCCESS;
  }
  bool is_memory_discontinuous = false;
  (void)ge::AttrUtils::GetBool(graph, ge::ATTR_NAME_MEMORY_DISCONTIGUOUS_ALLOCATION, is_memory_discontinuous);
  if (is_memory_discontinuous) {
    GELOGI("memory discontinuous scene has no need to do concat optimize.");
    return SUCCESS;
  }
  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    bool is_not_concat = op_desc->GetType() != "ConcatD" && op_desc->GetType() != "ConcatV2D";
    if (is_not_concat) {
      continue;
    }
    GELOGI("concat node [%s] start notask check.", node->GetName().c_str());
    cur_pro_node_name_ = node->GetName();

    if (IsUnknownShapeOp(op_desc)) {
      GELOGI("concat node [%s] is unknown shape op.", node->GetName().c_str());
      continue;
    }

    if (IsOwnerGraphUnknown(node)) {
      GELOGI("concat node [%s] is belong to unknown graph.", node->GetName().c_str());
      continue;
    }

    if (!InputCheck(node)) {
      GELOGI("concat node [%s] input does not meet the conditions.", node->GetName().c_str());
      continue;
    }

    if (!CheckConcatDim(op_desc)) {
      GELOGI("concat node [%s] concat dim does not meet the conditions.", node->GetName().c_str());
      continue;
    }

    if (!OutputCheck(node)) {
      GELOGI("concat node [%s] output does not meet the conditions.", node->GetName().c_str());
      continue;
    }

    if (!LxFusionCheck(node)) {
      GELOGI("concat node [%s] lxFusion does not meet the conditons.", node->GetName().c_str());
      continue;
    }

    SetAttrForConcatNotask(node);
  }

  return SUCCESS;
}

/*
* 示例
* NCHW => FORMAT_NC1HWC0, DT: DT_FLOAT16
* align_shape: {1, 16, 1, 1}
*/
bool ConcatNotaskPass::CheckConcatDimAlignment(const ge::OpDescPtr &op_desc, const gert::Shape &align_shape,
  const int64_t concat_dim, const ge::GeShape &ori_shape) const {
  GE_ASSERT_TRUE(!(ori_shape.GetDimNum() <= static_cast<size_t>(concat_dim) ||
    align_shape.GetDimNum() <= static_cast<size_t>(concat_dim) || align_shape[concat_dim] <= 0),
    "concat notask [%s] concat_dim %lld, ori shape size %zu, align shape size %zu, dim 1 value %lld.",
      op_desc->GetName().c_str(), concat_dim, ori_shape.GetDimNum(), align_shape.GetDimNum(), align_shape[concat_dim]);
  if ((ori_shape.GetDim(concat_dim) % align_shape[concat_dim]) != 0) {
    GELOGD("concat notask [%s] concat_dim %lld, ori shape %lld, align shape %lld.",
      op_desc->GetName().c_str(), concat_dim, ori_shape.GetDim(concat_dim), align_shape[concat_dim]);
    return false;
  }
  return true;
}

void ConcatNotaskPass::PrintTransferDims(const std::string name,
  const std::vector<std::vector<int32_t>> &transfer_dims) const {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < transfer_dims.size(); i++) {
    ss << "{";
    for (size_t j = 0; j < transfer_dims[i].size(); j++) {
      ss << transfer_dims[i][j];
      if (j != transfer_dims[i].size() - 1) {
        ss << ",";
      }
    }
    ss << "}";
    if (i != transfer_dims.size() - 1) {
    ss << ",";
  }
  }
  ss << "}";
  GELOGI("[%s]: %s", name.c_str(), ss.str().c_str());
}

void ConcatNotaskPass::PrintShape(const std::string name, const gert::Shape &shape) const {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < shape.GetDimNum(); i++) {
    ss << shape[i];
    if (i != shape.GetDimNum() - 1) {
      ss << ",";
    }
  }
  ss << "}";
  GELOGI("[%s]: %s", name.c_str(), ss.str().c_str());
}

bool ConcatNotaskPass::CheckSplitAxis(const std::vector<int32_t> &src_axes, const int64_t &axis_idx,
  const int32_t &from_axis, const gert::Shape &align_shape, const gert::Shape &src_shape) const {
  const auto out = src_axes[0];
  if (out == axis_idx) {
    // 如果是外轴，拆之前的轴要满足小于等于align_size,外轴值就会等于1
    return src_shape.GetDim(from_axis) <= align_shape.GetDim(from_axis);
  } else {
    // 若果是内轴，只有align_size是1，内轴才会等于1
    return align_shape.GetDim(from_axis) == 1;
  }
}

// 某一根轴的值是否为1
bool ConcatNotaskPass::IsFromAxisOne(const int64_t &axis_idx,
  const transformer::AxisIndexMapping &axis_index_mapping,
  const gert::Shape &align_shape, const gert::Shape &src_shape, const int32_t &from_axis) const {
  GE_ASSERT_TRUE(axis_index_mapping.src_to_dst_transfer_dims.size() > static_cast<size_t>(from_axis));
  if (axis_index_mapping.src_to_dst_transfer_dims[from_axis].size() > 1) {
    if (!CheckSplitAxis(axis_index_mapping.src_to_dst_transfer_dims[from_axis], axis_idx,
      from_axis, align_shape, src_shape)) {
      GELOGD("The value of from axis[%d] is %lld, align shape is %lld, [%s] not meet optimize condition.",
        from_axis, src_shape.GetDim(from_axis), align_shape.GetDim(from_axis), cur_pro_node_name_.c_str());
      return false;
    }
  } else {
    // 轴没被拆，值必须是1
    return src_shape.GetDim(from_axis) == 1;
  }

  return true;
}

// 合轴里边所有dim的值都为1
bool ConcatNotaskPass::IsMergedAxisAllOnes(const int64_t &axis_idx,
  const std::vector<int64_t> &shape) const {
  return shape[axis_idx] == 1;
}

// real_concat_dim轴在合轴里，合轴里concat_dim轴前边的轴值为1
bool ConcatNotaskPass::IsFrontDimsAllOnesInMergedAxis(const gert::Shape &align_shape, const gert::Shape &src_shape,
  const transformer::AxisIndexMapping &axis_index_mapping, const int64_t &real_concat_dim,
   const int64_t &concat_dim) const {
  const auto src_axes = axis_index_mapping.dst_to_src_transfer_dims[real_concat_dim];
  const auto merge_it = std::find(src_axes.begin(), src_axes.end(), concat_dim);
  GE_ASSERT_TRUE(merge_it != src_axes.end());
  for (auto it = src_axes.begin(); it != merge_it; it++) {
    const auto from_axis = *it;
    if (!IsFromAxisOne(real_concat_dim, axis_index_mapping, align_shape, src_shape, from_axis)) {
      GELOGD("The value of from axis[%d] is %lld, [%s] not meet optimize condition.",
        from_axis, src_shape.GetDim(from_axis), cur_pro_node_name_.c_str());
      return false;
    }
  }
  return true;
}

// real_concat_dim轴前边的轴值都为1
bool ConcatNotaskPass::IsFrontDimsAllOnes(const transformer::AxisIndexMapping &axis_index_mapping,
  const std::vector<int64_t> &shape, const int64_t &real_concat_dim) const {
  for (auto axis = 0; axis < real_concat_dim; axis++) {
    const auto src_axes = axis_index_mapping.dst_to_src_transfer_dims[axis];
    if (src_axes.size() > 1) {
      // 当前轴是合轴产生的
      if (!IsMergedAxisAllOnes(axis, shape)) {
        GELOGD("The value of Merged axis[%d] is %lld, [%s] not meet optimize condition.",
          axis, shape[axis], cur_pro_node_name_.c_str());
        return false;
      }
    } else {
      // 当前轴不是合轴产生的，值为1
      if (shape[axis] != 1) {
        GELOGD("The value of axis[%d] is %lld, [%s] not meet optimize condition.", axis, shape[axis], cur_pro_node_name_.c_str());
        return false;
      }
    }
  }

  return true;
}
/*
* 示例：NCHW => FORMAT_NC1HWC0
* 1.不需要补维
* src_to_dst_transfer_dims: {{0}, {1, 4}, {2}, {3}}
* dst_to_src_transfer_dims: {{0},{1},{2},{3},{1}}
* 2.需要补维
* CH => FORMAT_NC1HWC0
* 原始shape [16, 32], 运行时shape [1,2,32,1,16](需要补NW)
* src_to_dst_transfer_dims: {{1, 4}, {2}}
* dst_to_src_transfer_dims: {{-1},{0},{1},{-1},{0}}
* 示例：NCHW => FZ (C1,H,W),N1,N0,C0
* src_to_dst_transfer_dims: {{1, 2}, {0, 3}, {0}, {0}}
* dst_to_src_transfer_dims: {{1,2,3},{0},{0},{1}}
*/
bool ConcatNotaskPass::CheckRealConcatDim(const gert::Shape &align_shape, const gert::Shape &src_shape,
  const transformer::AxisIndexMapping &axis_index_mapping, const int64_t &concat_dim,
  const ge::GeTensorDesc &input_tensor) const {
  int64_t real_concat_dim = 0;
  
  GE_ASSERT_TRUE(axis_index_mapping.src_to_dst_transfer_dims[concat_dim].size() > 0);
  // 通过原始格式到运行时格式的对应关系，获取实际的concat dim轴
  real_concat_dim = axis_index_mapping.src_to_dst_transfer_dims[concat_dim][0];

  const auto shape = input_tensor.GetShape().GetDims();
  GE_ASSERT_TRUE((real_concat_dim >= 0) && (static_cast<size_t>(real_concat_dim) < shape.size()));
  const auto src_real_concat_dims = axis_index_mapping.dst_to_src_transfer_dims[real_concat_dim];
  if (src_real_concat_dims.size() > 1) {
    // real_concat_dim是合轴产生的，real_concat_dim轴前边的轴都为1且concat_dim所在的合轴里前边的值都为1
    return IsFrontDimsAllOnes(axis_index_mapping, shape, real_concat_dim) &&
      IsFrontDimsAllOnesInMergedAxis(align_shape, src_shape, axis_index_mapping, real_concat_dim, concat_dim);
  } else {
    // real_concat_dim不是合轴产生的，real_concat_dim轴前边的轴值都为1
    return IsFrontDimsAllOnes(axis_index_mapping, shape, real_concat_dim);
  }
}

bool ConcatNotaskPass::GetTransferDims(const ge::OpDescPtr &op_desc, const gert::Shape &src_shape,
  const int64_t &reshape_type_mask,
  const ge::GeTensorDesc &input_tensor, transformer::AxisIndexMapping &axis_index_mapping) const {
  const auto input_format = input_tensor.GetFormat();
  const ge::Format input_orinal_format = input_tensor.GetOriginFormat();
  transformer::TransferDimsInfo transfer_dims_info;
  transfer_dims_info.src_format = input_orinal_format;
  transfer_dims_info.dst_format = input_format;
  transfer_dims_info.src_shape = src_shape;
  transfer_dims_info.reshape_type_mask = reshape_type_mask;
  
  GELOGD("Concat node [%s] original_format=%d, format=%d, reshape_type_mask=%lld.", op_desc->GetName().c_str(),
    input_orinal_format, input_format, reshape_type_mask);
  // 调用FE接口，获取原始格式到运行时格式的轴对应关系及运行时格式到原始格式的轴对应关系
  if (!transformer::TransferShapeUtils::TransferDims(transfer_dims_info, axis_index_mapping)) {
    GELOGD("Concat [%s] notask transfer dims failed.", op_desc->GetName().c_str());
    return false;
  }
  PrintTransferDims("src_to_dst_transfer_dims", axis_index_mapping.src_to_dst_transfer_dims);
  PrintTransferDims("dst_to_src_transfer_dims", axis_index_mapping.dst_to_src_transfer_dims);
  GE_ASSERT_TRUE(axis_index_mapping.src_to_dst_transfer_dims.size() == src_shape.GetDimNum());
  GE_ASSERT_TRUE(axis_index_mapping.dst_to_src_transfer_dims.size() == input_tensor.GetShape().GetDimNum());

  return true;
}

bool ConcatNotaskPass::GetAlignedShape(const ge::OpDescPtr &op_desc, const gert::Shape &src_shape,
  const int64_t &reshape_type_mask,
  const ge::GeTensorDesc &input_tensor, gert::Shape &align_shape) const {
  const auto input_format = input_tensor.GetFormat();
  const ge::Format input_orinal_format = input_tensor.GetOriginFormat();

  GELOGD("Concat [%s] original_format=%d, format=%d, data_type=%d, reshape_type_mask=%lld.",
    op_desc->GetName().c_str(), input_orinal_format, input_format, input_tensor.GetDataType(), reshape_type_mask);
  transformer::AlignShapeInfo align_shape_info;
  align_shape_info.src_format = input_orinal_format;
  align_shape_info.dst_format = input_format;
  align_shape_info.src_shape = src_shape;
  align_shape_info.data_type = input_tensor.GetDataType();
  align_shape_info.reshape_type_mask = reshape_type_mask;
  if (!transformer::TransferShapeUtils::GetAlignedShape(align_shape_info, align_shape)) {
    GELOGD("Concat notask %s get align shape failed.", op_desc->GetName().c_str());
    return false;
  }
  PrintShape("align_shape", align_shape);
  GE_ASSERT_TRUE(align_shape.GetDimNum() == src_shape.GetDimNum());
  return true;
}

bool ConcatNotaskPass::CheckConcatDim(const ge::OpDescPtr &op_desc) const {
  int64_t concat_dim = 0;

  (void)ge::AttrUtils::GetInt(op_desc, "concat_dim", concat_dim);
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    ge::GeTensorDesc input_tensor = op_desc->GetInputDesc(i);
    ge::GeShape input_orinal_shape = input_tensor.GetOriginShape();
    gert::Shape src_shape;
    src_shape.SetDimNum(input_orinal_shape.GetDimNum());
    for (size_t j = 0; j < src_shape.GetDimNum(); j++) {
      src_shape[j] = input_orinal_shape.GetDim(j);
    }
    PrintShape("src_shape", src_shape);
    int64_t reshape_type_mask = 0;
    (void)ge::AttrUtils::GetInt(input_tensor, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type_mask);

    if (concat_dim < 0) {
      GELOGD("Concat_dim[%lld] is nagtive number, change it to positive.", concat_dim);
      concat_dim = static_cast<int64_t>(input_orinal_shape.GetDimNum()) + concat_dim;
    }
    GE_ASSERT_TRUE((concat_dim >= 0) && (static_cast<size_t>(concat_dim) < input_orinal_shape.GetDimNum()));
    gert::Shape align_shape;
    if (!GetAlignedShape(op_desc, src_shape, reshape_type_mask, input_tensor, align_shape)) {
      return false;
    }

    transformer::AxisIndexMapping axis_index_mapping;
    if (!GetTransferDims(op_desc, src_shape, reshape_type_mask, input_tensor, axis_index_mapping)) {
      return false;
    }

    // 1.整体判断逻辑是concat轴前面的轴都为1
    if (!CheckRealConcatDim(align_shape, src_shape, axis_index_mapping, concat_dim, input_tensor)) {
      GELOGD("Concat [%s] notask check real concat dim, concat_dim = %lld.", op_desc->GetName().c_str(),
        concat_dim);
      return false;
    }

    // 2.concat轴是否满足轴对齐条件即concat轴不带padding
    if (!CheckConcatDimAlignment(op_desc, align_shape, concat_dim, input_orinal_shape)) {
        GELOGD("Concat notask [%s] check concat dim alignment failed.", op_desc->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool ConcatNotaskPass::IsUnknownShapeOp(const ge::OpDescPtr &op_desc) const {
  for (auto &tenosr_desc_ptr : op_desc->GetAllInputsDescPtr()) {
    if ((tenosr_desc_ptr != nullptr) && (tenosr_desc_ptr->GetShape().IsUnknownShape())) {
      GELOGD("Concat notask input tensor is unknown shape.");
      return true;
    }
  }

  for (auto &tenosr_desc_ptr : op_desc->GetAllOutputsDescPtr()) {
    if ((tenosr_desc_ptr != nullptr) && (tenosr_desc_ptr->GetShape().IsUnknownShape())) {
      GELOGD("Concat notask output tensor is unknown shape.");
      return true;
    }
  }
  return false;
}

bool ConcatNotaskPass::OutputCheck(const ge::NodePtr &concat_node) const {
  for (auto &output_anchor : concat_node->GetAllOutDataAnchors()) {
    for (size_t i = 0; i < output_anchor->GetPeerInDataAnchors().size(); i++) {
      auto peerAnchor = output_anchor->GetPeerInDataAnchors().at(i);
      GE_ASSERT_TRUE(peerAnchor != nullptr);
      auto next_node = peerAnchor->GetOwnerNode();
      const auto output_nodes = next_node->GetOutDataNodes();
      if ((next_node->GetType() == RESHAPE) && (!output_nodes.empty())) {
        next_node = output_nodes.at(0);
      }
      ge::OpDescPtr next_node_desc = next_node->GetOpDesc();
      string next_node_name = next_node_desc->GetName();
      bool no_task = false;
      bool output_reuse_input = false;
      bool no_padding_continuous_input = false;
      (void)ge::AttrUtils::GetBool(next_node_desc, ge::ATTR_NAME_NOTASK, no_task);
      (void)ge::AttrUtils::GetBool(next_node_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
      (void)ge::AttrUtils::GetBool(next_node_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                   no_padding_continuous_input);
      const bool is_virtual_op = no_task || output_reuse_input || no_padding_continuous_input;
      if (is_virtual_op) {
        GELOGD("Next node %s has _no_task attribute, %s can't optimize.", next_node_name.c_str(),
                concat_node->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

bool ConcatNotaskPass::IsOwnerGraphUnknown(const ge::NodePtr &concat_node) const {
  bool is_dynamic = false;
  const auto &owner_graph = concat_node->GetOwnerComputeGraph();
  if (owner_graph != nullptr) {
    (void)AttrUtils::GetBool(owner_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic);
    is_dynamic = (is_dynamic || owner_graph->GetGraphUnknownFlag());
  }

  return is_dynamic;
}

bool ConcatNotaskPass::LxFusionCheck(const ge::NodePtr &node) const {
  const auto op_desc = node->GetOpDesc();
  return !IsLxFusionMem(op_desc) && !IsLxFusionOp(node);
}

bool ConcatNotaskPass::IsLxFusionMem(const ge::OpDescPtr &op_desc) const {
  std::vector<uint32_t> input_mem_type;
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, input_mem_type);
  std::vector<uint32_t> output_mem_type;
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_mem_type);
  for (auto mem_type : input_mem_type) {
    if ((mem_type == RT_MEMORY_L1) || (mem_type == RT_MEMORY_L2) || (mem_type == kRtMemoryUB)) {
      GELOGD("Node [%s] has lx addr input, not optimize.", op_desc->GetName().c_str());
      return true;
    }
  }
  for (auto mem_type : output_mem_type) {
    if ((mem_type == RT_MEMORY_L1) || (mem_type == RT_MEMORY_L2) || (mem_type == kRtMemoryUB)) {
      GELOGD("Node [%s] has lx addr output, not optimize.", op_desc->GetName().c_str());
      return true;
    }
  }
  return false;
}

const std::string kLxSlice = "lxslice";
bool ConcatNotaskPass::IsLxFusionOp(const ge::NodePtr &node) const {
  std::string op_name = node->GetName();
  size_t pos = op_name.find(kLxSlice);
  if (pos != std::string::npos) {
    GELOGD("Node [%s] is lxfusion op, can not optimize.", node->GetName().c_str());
    return true;
  }
  return false;
}

void ConcatNotaskPass::SetAttrForConcatNotask(const ge::NodePtr &node) const {
  const auto op_desc = node->GetOpDesc();
  GELOGI("success to set concat attribute for node [%s]", op_desc->GetName().c_str());
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  const auto input_size = node->GetAllInDataAnchorsSize();
  for (uint32_t index = 0; index < input_size; ++index) {
    auto input_anchor = node->GetInDataAnchor(index);
    if (input_anchor == nullptr) {
      continue;
    }
    auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto outupt_idx = peer_out_anchor->GetIdx();
    auto peer_node = peer_out_anchor->GetOwnerNode();
    auto output_tensor_desc = peer_node->GetOpDesc()->MutableOutputDesc(outupt_idx);
    if (output_tensor_desc != nullptr) {
      ge::AttrUtils::SetBool(output_tensor_desc, kATTR_CAN_REUSED_FOR_CONCAT_OPTIMIZE, false);
    }
  }
}

// 输入校验
const int32_t kDeepth = 100;
bool ConcatNotaskPass::InputCheck(const ge::NodePtr &node) {
  std::set<ge::OutDataAnchorPtr> src_anchors;
  std::set<int64_t> mem_types;
  for (size_t i = 0U; i < node->GetAllInDataAnchors().size(); i++) {
    const auto in_anchor = node->GetAllInDataAnchors().at(i);
    GE_CHECK_NOTNULL(in_anchor);
    const auto pre_out_anchor = in_anchor->GetPeerOutAnchor();
    if (pre_out_anchor == nullptr) {
      continue;
    }
    auto outupt_idx = pre_out_anchor->GetIdx();
    auto pre_node = pre_out_anchor->GetOwnerNode();
    auto pre_op_desc = pre_node->GetOpDesc();

    // 单Scalar输入理论上也可以支持notask,但是标量输入shape是空的,concat_dim轴判断不好归一，暂不支持
    if (IsScalarInput(node, i)) {
      GELOGD("Node [%s] has scalar input[%zu] which does not meet optimize condition.", cur_pro_node_name_.c_str(), i);
      return false;
    }

    // 校验Tensor是否满足对齐条件
    if (!CheckTensorAlign(node, i)) {
      GELOGD("concat node [%s] check tensor align failed.", node->GetName().c_str());
      return false;
    }

    // 校验是否存在多个输入来源于相同的输出, 如存在则不满足no task条件
    if (HasSameSourceAnchor(in_anchor, src_anchors)) {
      GELOGD("concat node [%s] has same source anchor.", node->GetName().c_str());
      return false;
    }

    // 校验上游节点类型是否支持no task
    if (!IsPreNodeTypeValid(in_anchor)) {
      return false;
    }

    // 校验上游节点是否包含子图,连续内存分配不能跨子图
    if (IsPreNodeWithSubgraph(in_anchor)) {
      GELOGD("Pre node [%s] has subgraph, [%s] can't optimize.", pre_node->GetName().c_str(), node->GetName().c_str());
      return false;
    }

    // 校验pre node out anchor Tensor是否可以参与no task优化
    if (!IsPreOutAnchorCanReuseForConcatOptimize(pre_out_anchor)) {
      GELOGD("concat node [%s] pre node [%s] can not reused.", node->GetName().c_str(), pre_node->GetName().c_str());
      return false;
    }

    // 校验pre node out anchor是否也是模型输出, 如果是则不满足no task条件
    if (!IsPreOutAnchorValidMultiRef(pre_out_anchor)) {
      GELOGD("Previous node [%s] connect to netoutput, [%s] can't optimize.", pre_node->GetName().c_str(),
        cur_pro_node_name_.c_str());
      return false;
    }

    // 校验输入节点属性
    if (!IsPreNodeAttrValid(pre_op_desc)) {
      return false;
    }

    if (!IsSameInputMemType(pre_op_desc, outupt_idx, mem_types)) {
      GELOGD("Input mem type is not same, [%s] can't optimize.", cur_pro_node_name_.c_str());
      return false;
    }
  }
  return true;
}

bool ConcatNotaskPass::IsScalarInput(const ge::NodePtr &node, const size_t input_index) const {
  // Scalar输入的shape的维度是0
  const auto td = node->GetOpDesc()->GetInputDesc(input_index);
  return td.GetOriginShape().GetDimNum() == 0;
}

bool ConcatNotaskPass::CheckTensorAlign(const ge::NodePtr &node, const size_t input_index) const {
  // 如果只有一个输入, 认为天然满足对齐条件
  if (node->GetAllInDataAnchorsSize() == 1) {
    return true;
  }

  // 大于1个输入的情况下, 需要判断每个输入都满足32B对齐，空tensor认为是不满足对齐条件
  const auto td = node->GetOpDesc()->GetInputDesc(input_index);
  const auto shape_size = td.GetShape().GetShapeSize();
  if (ge::GetSizeByDataType(td.GetDataType()) < 0) {
    GELOGI("Get data type[%s] size less than zero.",
           ge::TypeUtils::DataTypeToSerialString(td.GetDataType()).c_str());
    return false;
  }
  const auto tensor_size = ge::GetSizeInBytes(shape_size, td.GetDataType());
  return ((tensor_size > 0) && (tensor_size % TENSOR_ALIGN_SIZE == 0));
}

// 2. 不能存在多个输入来源于同一个输出, 即同一个输出被Concat多个输入引用
bool ConcatNotaskPass::HasSameSourceAnchor(const ge::InDataAnchorPtr &in_anchor,
                         std::set<ge::OutDataAnchorPtr> &src_anchors) const {
  ge::OutDataAnchorPtr src_anchor = nullptr;
  GetFirstOutAnchorNotInRefNode(in_anchor, src_anchor, 0);
  const bool has_same_src_anchor = (src_anchors.count(src_anchor) == 1U);
  src_anchors.insert(src_anchor);
  return has_same_src_anchor;
}

bool ConcatNotaskPass::IsPreNodeWithSubgraph(const ge::InDataAnchorPtr &in_anchor) const {
  ge::NodePtr node = nullptr;
  
  GetFirstNotRefNode(in_anchor, node);
  if (node == nullptr) {
    return false;
  }
  const auto op_desc = node->GetOpDesc();
  return (op_desc != nullptr) ? (!op_desc->GetSubgraphInstanceNames().empty()) : false;
}
// 3. 输入内存必须是可分配内存, 不能是用户内存, Const, variable
bool ConcatNotaskPass::IsPreNodeTypeValid(const ge::InDataAnchorPtr &in_anchor) {
  ge::NodePtr node = nullptr;
  
  GetFirstNotRefNode(in_anchor, node);
  if (node == nullptr) {
    return false;
  }
  const std::string op_type = node->GetType();
  // Hcom算子仅放开盘古场景用到的AllGather算子
  static std::set<std::string> not_support_type = {DATA, REFDATA, VARIABLE, CONSTANTOP, CONSTANT};
  if (not_support_type.count(op_type) != 0U) {
    GELOGD("concat node [%s] pre node [%s] opType is %s.", cur_pro_node_name_.c_str(),
      node->GetName().c_str(), op_type.c_str());
    return false;
  }

  return true;
}

// 4. 输入Tensor不能打已Concat no task属性
bool ConcatNotaskPass::IsPreOutAnchorCanReuseForConcatOptimize(const ge::OutDataAnchorPtr out_anchor) const {
  auto peer_node = out_anchor->GetOwnerNode();
  auto outupt_idx = out_anchor->GetIdx();
  auto output_tensor_desc = peer_node->GetOpDesc()->MutableOutputDesc(outupt_idx);
  if (output_tensor_desc == nullptr) {
    return false;
  }
  bool can_reuse = true;
  (void)ge::AttrUtils::GetBool(output_tensor_desc, kATTR_CAN_REUSED_FOR_CONCAT_OPTIMIZE, can_reuse);
  return can_reuse;
}

// 5. 输入不可以同时为模型输出
bool ConcatNotaskPass::IsPreOutAnchorValidMultiRef(const ge::OutDataAnchorPtr out_anchor) const {
  auto in_anchors = out_anchor->GetPeerInDataAnchors();
  if (in_anchors.size() == 1U) {
    return true;
  }

  for (const auto &anchor : in_anchors) {
    if (anchor->GetOwnerNode()->GetType() == NETOUTPUT) {
      return false;
    }
  }
  return true;
}

// 6、输入节点属性校验
bool ConcatNotaskPass::IsPreNodeAttrValid(const ge::OpDescPtr &pre_op_desc) {
 string pre_node_name = pre_op_desc->GetName();
  bool is_continous_input = false;
  bool is_continous_output = false;
  bool is_ref = false;
  bool no_task = false;
  bool output_reuse_input = false;
  bool no_padding_continuous_input = false;
  vector<int64_t> output_index;
  (void)ge::AttrUtils::GetBool(pre_op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
  (void)ge::AttrUtils::GetBool(pre_op_desc, ge::ATTR_NAME_CONTINUOUS_OUTPUT, is_continous_output);
  (void)ge::AttrUtils::GetBool(pre_op_desc, ge::ATTR_NAME_REFERENCE, is_ref);
  (void)ge::AttrUtils::GetListInt(pre_op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
  (void)ge::AttrUtils::GetBool(pre_op_desc, ge::ATTR_NAME_NOTASK, no_task);
  (void)ge::AttrUtils::GetBool(pre_op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
  (void)ge::AttrUtils::GetBool(pre_op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, no_padding_continuous_input);

  if (is_continous_input || is_continous_output || is_ref) {
    GELOGD("Previous node %s attribute: continuous_input %s, continuous_output %s,"
      " reference %s, node %s can't optimize.",
      pre_node_name.c_str(), is_continous_input ? "true" : "false", is_continous_output ? "true" : "false",
      is_ref ? "true" : "false", cur_pro_node_name_.c_str());
    return false;
  }

  bool is_virtual_op = no_task || output_reuse_input || no_padding_continuous_input;
  if (is_virtual_op) {
    GELOGD("Previous node %s has _no_task attribute, %s can't optimize.", pre_node_name.c_str(),
           cur_pro_node_name_.c_str());
    return false;
  }
  if (!output_index.empty()) {
    GELOGD("Previous node %s has atomic output, %s can not optimize.", pre_node_name.c_str(),
           cur_pro_node_name_.c_str());
    return false;
  }

  return true;
}

// 7. 输入tensor的内存类型要相同
bool ConcatNotaskPass::IsSameInputMemType(const ge::OpDescPtr &pre_op_desc, const size_t outupt_idx,
                         std::set<int64_t> &mem_types) const {
  std::vector<int64_t> output_mem_type;
  int64_t mem_type = RT_MEMORY_HBM;
  (void)ge::AttrUtils::GetListInt(pre_op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_mem_type);
  if (outupt_idx < output_mem_type.size()) {
    if (MemTypeUtils::IsMemoryTypeSpecial(output_mem_type[outupt_idx])) {
      mem_type = output_mem_type[outupt_idx];
    }
  }
  mem_types.insert(mem_type);

  return (mem_types.size() == 1);
}

void ConcatNotaskPass::GetFirstOutAnchorNotInRefNode(const ge::InDataAnchorPtr &input_anchor,
                                      ge::OutDataAnchorPtr &src_anchor,
                                      int32_t current_deep) const {
  if (current_deep >= kDeepth) {
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
  int32_t reuse_in_index = -1;
  const bool reuse_input_flag = GraphUtils::IsRefFromInput(peer_out_anchor, reuse_in_index);
  if (reuse_input_flag) {
    auto in_anchor = peer_node->GetInDataAnchor(reuse_in_index);
    if (in_anchor == nullptr) {
      return;
    }
    GetFirstOutAnchorNotInRefNode(in_anchor, src_anchor, current_deep + 1);
  } else {
    src_anchor = peer_out_anchor;
  }
  return;
}

void ConcatNotaskPass::GetFirstNotRefNode(const ge::InDataAnchorPtr &input_anchor,
                                      ge::NodePtr &node) const {
  ge::OutDataAnchorPtr src_anchor = nullptr;
  GetFirstOutAnchorNotInRefNode(input_anchor, src_anchor, 0);
  node = (src_anchor != nullptr) ? src_anchor->GetOwnerNode() : nullptr;
  return;
}
REG_PASS_OPTION("ConcatNotaskPass").LEVELS(OoLevel::kO3);
}  // namespace ge
