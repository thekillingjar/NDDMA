/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "common/checker.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "ge_local_ops_kernel_calc_op_param.h"

namespace ge {
namespace ge_local {

namespace {
const std::string kConcatDim = "concat_dim";
const std::string kConcatNum = "N";
const std::string kKeepInputOffset = "keep_input_offset";
const std::string kSplitDim = "split_dim";
const std::string kSplitNum = "num_split";
const std::string kKeepOutputOffset = "keep_output_offset";
const int64_t kAlignNum = 32;
}

Status GeLocalOpsKernelBuilderCalcOpParam::CheckDimAttrValid(const std::vector<int64_t> &dim_list,
                                                             const std::vector<int64_t> &num_list,
                                                             const uint32_t anchors_size) {
  const auto dim_list_size = dim_list.size();
  const auto num_list_size = num_list.size();
  GE_ASSERT_TRUE(dim_list_size != 0 && dim_list_size == num_list_size,
                 "dim_list or num_list number invalid, dim_list size[%u], num_list size[%u].",
                 dim_list_size, num_list_size);

  uint32_t num_list_product = 1;
  for (size_t i = 0; i < num_list_size; ++i) {
    GE_ASSERT_TRUE(num_list[i] > 0, "num_list index[%u] == 0, invalid.", i);
    num_list_product *= num_list[i];
  }

  GE_ASSERT_TRUE(num_list_product == anchors_size, "num_list product not equal to node count [%u].", anchors_size);

  return SUCCESS;
}

Status GeLocalOpsKernelBuilderCalcOpParam::CheckDimAttrMatchTensor(std::vector<int64_t> &dim_list,
                                                                   const GeTensorDescPtr &tensor_desc) {
  GE_CHECK_NOTNULL(tensor_desc);

  const auto dim_list_size = dim_list.size();
  const auto shape_size = tensor_desc->GetShape().GetDims().size();
  for (size_t i = 0; i < dim_list_size; ++i) {
    if (dim_list[i] < 0) {
      dim_list[i] += shape_size;
    }

    GE_ASSERT_TRUE(dim_list[i] >= 0 && static_cast<size_t>(dim_list[i]) < static_cast<size_t>(shape_size),
                   "dim_list index[%u] value[%d] invalid, shape_size[%u].",
                   i, dim_list[i], shape_size);
  }

  return SUCCESS;
}

Status GeLocalOpsKernelBuilderCalcOpParam::CheckPhonyConcatInputShapeSame(const Node &node) {
  const auto in_anchors_size = node.GetAllInDataAnchorsSize();
  const auto op_desc = node.GetOpDesc();
  const auto &input_desc = op_desc->MutableInputDesc(0);
  GE_ASSERT_NOTNULL(input_desc);
  const auto first_shape = input_desc->GetShape().GetDims();

  for (uint32_t i = 1U; i < in_anchors_size; ++i) {
    const auto input_desc = op_desc->MutableInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc);
    const auto tensor_shape = input_desc->GetShape().GetDims();

    GE_ASSERT_TRUE(first_shape.size() == tensor_shape.size(),
                   "Input shape size of PhonyConcat node [%s] are not the same, "
                   "first input shape size [%u], [%u]th input shape size [%u].",
                   node.GetName().c_str(), first_shape.size(), i, tensor_shape.size());
    for (size_t dim = 0U; dim < first_shape.size(); ++dim) {
      GE_ASSERT_TRUE(first_shape[dim] == tensor_shape[dim],
                     "Input shapes of PhonyConcat node [%s] are not the same, "
                     "first input [%u]th dim [%d], [%u]th input [%u]th dim [%d].",
                     node.GetName().c_str(), dim, first_shape[dim], i, dim, tensor_shape[dim]);
    }
  }
  return SUCCESS;
}

Status GeLocalOpsKernelBuilderCalcOpParam::CheckPhonySplitOutputShapeSame(const Node &node) {
  const auto out_anchors_size = node.GetAllOutDataAnchorsSize();
  const auto op_desc = node.GetOpDesc();
  const auto &output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  const auto first_shape = output_desc->GetShape().GetDims();

  for (uint32_t i = 1U; i < out_anchors_size; ++i) {
    const auto &tensor_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(tensor_desc);
    const auto tensor_shape = tensor_desc->GetShape().GetDims();

    GE_ASSERT_TRUE(first_shape.size() == tensor_shape.size(),
                   "Output shape size of PhonySplit node [%s] are not the same, "
                   "first output shape size [%u], [%u]th output shape size [%u].",
                   node.GetName().c_str(), first_shape.size(), i, tensor_shape.size());
    for (size_t dim = 0U; dim < first_shape.size(); ++dim) {
      GE_ASSERT_TRUE(first_shape[dim] == tensor_shape[dim],
                     "Output shapes of PhonySplit node [%s] are not the same, "
                     "first output [%u]th dim [%d], [%u]th output [%u]th dim [%d].",
                     node.GetName().c_str(), dim, first_shape[dim], i, dim, tensor_shape[dim]);
    }
  }
  return SUCCESS;
}

Status GeLocalOpsKernelBuilderCalcOpParam::CheckPhonyConcatInputIs32Align(const ConstOpDescPtr &op_desc) {
  const auto tensor_desc = op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(tensor_desc);

  int64_t tensor_size = 0;
  GE_ASSERT_SUCCESS(TensorUtils::CalcTensorMemSize(tensor_desc->GetShape(), tensor_desc->GetFormat(),
                                                   tensor_desc->GetDataType(), tensor_size),
                    "PhonyConcat node[%s] get input tensor size failed.", op_desc->GetName().c_str());
  GE_ASSERT_TRUE(tensor_size > 0, "PhonyConcat node[%s] get input tensor size failed, ret[%d].",
                 op_desc->GetName().c_str(), tensor_size);
  GE_ASSERT_TRUE(tensor_size % kAlignNum == 0, "PhonyConcat node[%s]'s input tensor size[%d], not 32 align.",
                 op_desc->GetName().c_str(), tensor_size);

  return SUCCESS;
}

int64_t GeLocalOpsKernelBuilderCalcOpParam::GetConcatOffset(const GeTensorDescPtr &tensor_desc, uint32_t node_idx,
    const std::vector<int64_t> &concat_dim_list, const std::vector<int64_t> &concat_num_list) {
  std::vector<int64_t> output_shape = tensor_desc->GetShape().GetDims();
  const auto shape_size = output_shape.size();
  const auto dim_list_size = concat_dim_list.size();
  ge::DataType data_type = tensor_desc->GetDataType();

  /*
   * 先计算slice_id list，其含义是在每个拼接轴上，该算子在第几个位置。
   * 如：拼接轴[3, 1]，拼接数量为[2, 3]，即先2个一组拼接3轴，再3个一组拼接1轴，一共拼接6个算子。
   * 则：六个算子的slice_id 分别为：
   *  idx=0: [0, 0]
   *  idx=1: [1, 0]
   *  idx=2: [0, 1]
   *  idx=3: [1, 1]
   *  idx=4: [0, 2]
   *  idx=5: [1, 2]
   */
  std::vector<uint32_t> slice_id(dim_list_size, 0);
  for (uint32_t i = 0; i < dim_list_size; i++) {
    // 被除数及下标合法性 在此函数外部保证
    slice_id[i] = node_idx % concat_num_list[i];
    node_idx /= concat_num_list[i];
  }

  int64_t mem_offset = 0;

  /*
   * 根据slice_id list，计算该算子的offset。
   * 按拼接轴list从前往后，累加每次拼接的offset，每次拼接后更新shape。
   */
  for (uint32_t i = 0; i < dim_list_size; i++) {
    const auto concat_dim = concat_dim_list[i];

    int64_t element_cnt = 1;
    for (size_t dim_idx = concat_dim; dim_idx < shape_size; ++dim_idx) {
      element_cnt *= output_shape[dim_idx];
    }

    auto tensor_size = ge::GetSizeInBytes(element_cnt, data_type);
    mem_offset += slice_id[i] * tensor_size;

    output_shape[concat_dim] *= concat_num_list[i];
  }

  return mem_offset;
}

graphStatus GeLocalOpsKernelBuilderCalcOpParam::HasBufferFusionOffset(const Node &node, bool &has_fusion_offset) {
  for (auto &in_data_anchor : node.GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_data_anchor);
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(peer_op_desc);

    std::vector<int64_t> offsets_of_fusion = {};
    bool lx_fusion = AttrUtils::GetListInt(peer_op_desc, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
    if (lx_fusion && !offsets_of_fusion.empty()) {
      has_fusion_offset = true;
    }
  }
  return SUCCESS;
}

graphStatus GeLocalOpsKernelBuilderCalcOpParam::CalcPhonyConcatNodeOffset(const Node &node) {
  // fe/lxfusion 已经处理，则跳过
  bool has_fusion_offset = false;
  GE_ASSERT_SUCCESS(HasBufferFusionOffset(node, has_fusion_offset));
  if (has_fusion_offset) {
    GELOGI("The node: %s have already written ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION.", node.GetName().c_str());
    return SUCCESS;
  }

  const bool is_unknown_graph = node.GetOwnerComputeGraph()->GetGraphUnknownFlag();
  if (is_unknown_graph) {
    GELOGE(FAILED, "PhonyConcat node:[%s] does not support unknown graph.", node.GetName().c_str());
    return FAILED;
  }

  GE_ASSERT_SUCCESS(CheckPhonyConcatInputShapeSame(node));

  auto node_op_desc = node.GetOpDesc();
  GE_ASSERT_SUCCESS(CheckPhonyConcatInputIs32Align(node_op_desc));

  std::vector<int64_t> concat_dim_list;
  std::vector<int64_t> concat_num_list;
  bool keep_input_offset = true;
  AttrUtils::GetListInt(node_op_desc, kConcatDim, concat_dim_list);
  AttrUtils::GetListInt(node_op_desc, kConcatNum, concat_num_list);
  AttrUtils::GetBool(node_op_desc, kKeepInputOffset, keep_input_offset);

  GE_ASSERT_SUCCESS(CheckDimAttrValid(concat_dim_list, concat_num_list, node.GetAllInDataAnchorsSize()),
                    "PhonyConcat node[%s] attr invalid.", node.GetName().c_str());

  uint32_t node_idx = 0;
  for (auto &in_data_anchor : node.GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_data_anchor);
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(peer_op_desc);
    auto out_anchor_idx = peer_out_data_anchor->GetIdx();

    std::vector<int64_t> outputs_offset(peer_op_desc->GetOutputsSize(), 0);
    int64_t mem_offset = 0;

    // !keep_offset，则不计算offset，统一设为起始地址，由用户自定义行为
    if (keep_input_offset) {
      ge::GeTensorDescPtr tensor_desc = peer_op_desc->MutableOutputDesc(out_anchor_idx);
      GE_ASSERT_SUCCESS(CheckDimAttrMatchTensor(concat_dim_list, tensor_desc),
                        "PhonyConcat node[%s] attr not match [%d]th in node shape.",
                        node.GetName().c_str(), in_data_anchor->GetIdx());

      mem_offset = GetConcatOffset(tensor_desc, node_idx, concat_dim_list, concat_num_list);
    }

    outputs_offset[out_anchor_idx] = mem_offset;
    (void)ge::AttrUtils::SetListInt(peer_op_desc, ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, outputs_offset);
    GELOGI("GeLocal PhonyConcat Op: %s %uth in_anchor node: %s %uth out_anchor set output offset %d.",
           node.GetName().c_str(), node_idx, peer_op_desc->GetName().c_str(), out_anchor_idx, mem_offset);
    node_idx++;
  }

  return SUCCESS;
}

int64_t GeLocalOpsKernelBuilderCalcOpParam::GetSplitOffset(const GeTensorDescPtr &tensor_desc, uint32_t node_idx,
    const std::vector<int64_t> &split_dim_list, const std::vector<int64_t> &split_num_list) {
  std::vector<int64_t> input_shape = tensor_desc->GetShape().GetDims();
  const auto shape_size = input_shape.size();
  const auto dim_list_size = split_dim_list.size();
  ge::DataType data_type = tensor_desc->GetDataType();

  std::vector<uint32_t> slice_id(dim_list_size, 0);

  // split多轴相当于 concat多轴的逆向行为，遍历split_dim_list与concat顺序相反
  // 被除数及下标合法性 在此函数外部保证
  for (int32_t i = dim_list_size - 1; i >= 0; i--) {
    slice_id[i] = node_idx % split_num_list[i];
    node_idx /= split_num_list[i];
  }
  int64_t mem_offset = 0;

  for (int32_t i = dim_list_size - 1; i >= 0; i--) {
    const auto split_dim = split_dim_list[i];

    int64_t element_cnt = 1;
    for (size_t dim_idx = split_dim; dim_idx < shape_size; ++dim_idx) {
      element_cnt *= input_shape[dim_idx];
    }

    const auto tensor_size = ge::GetSizeInBytes(element_cnt, data_type);
    mem_offset += slice_id[i] * tensor_size;

    input_shape[split_dim] *= split_num_list[i];
  }

  return mem_offset;
}

graphStatus GeLocalOpsKernelBuilderCalcOpParam::CalcPhonySplitNodeOffset(const Node &node) {
  const bool is_unknown_graph = node.GetOwnerComputeGraph()->GetGraphUnknownFlag();
  if (is_unknown_graph) {
    GELOGE(FAILED, "PhonySplit node:[%s] does not support unknown graph.", node.GetName().c_str());
    return FAILED;
  }

  GE_ASSERT_SUCCESS(CheckPhonySplitOutputShapeSame(node));

  std::vector<int64_t> split_dim_list;
  std::vector<int64_t> split_num_list;
  bool keep_output_offset = true;
  auto node_op_desc = node.GetOpDesc();

  AttrUtils::GetListInt(node_op_desc, kSplitDim, split_dim_list);
  AttrUtils::GetListInt(node_op_desc, kSplitNum, split_num_list);
  AttrUtils::GetBool(node_op_desc, kKeepOutputOffset, keep_output_offset);

  GE_ASSERT_SUCCESS(CheckDimAttrValid(split_dim_list, split_num_list, node.GetAllOutDataAnchorsSize()),
                    "PhonySplit node[%s] attr invalid.", node.GetName().c_str());

  uint32_t node_idx = 0;
  for (auto &out_data_anchor : node.GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_data_anchor);
      auto peer_op_desc = peer_in_data_anchor->GetOwnerNode()->GetOpDesc();
      GE_CHECK_NOTNULL(peer_op_desc);
      size_t in_size = peer_op_desc->GetInputsSize();
      std::vector<int64_t> inputs_offset(in_size, 0);

      auto in_anchor_idx = peer_in_data_anchor->GetIdx();
      int64_t mem_offset = 0;

      // !keep_offset，则不计算offset，统一设为起始地址，由用户自定义行为
      if (keep_output_offset) {
        ge::GeTensorDescPtr tensor_desc = peer_op_desc->MutableInputDesc(in_anchor_idx);
        GE_ASSERT_SUCCESS(CheckDimAttrMatchTensor(split_dim_list, tensor_desc),
                          "PhonySplit node[%s] attr not match [%d]th out node shape.",
                          node.GetName().c_str(), out_data_anchor->GetIdx());

        mem_offset = GetSplitOffset(tensor_desc, node_idx, split_dim_list, split_num_list);
      }

      inputs_offset[in_anchor_idx] = mem_offset;
      (void)ge::AttrUtils::SetListInt(peer_op_desc, ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, inputs_offset);
      GELOGI("GeLocal PhonySplit Op: %s %uth out_anchor node: %s %uth in_anchor set input offset %d.",
             node.GetName().c_str(), node_idx, peer_op_desc->GetName().c_str(), in_anchor_idx, mem_offset);
    }

    node_idx++;
  }

  return SUCCESS;
}

graphStatus GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput(const Node &node) {
  auto owner_graph = node.GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  if (!owner_graph->GetGraphUnknownFlag()) {
    auto op_desc = node.GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    const auto &output_desc = op_desc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(output_desc);
    ge::TensorUtils::SetReuseInput(*output_desc, true);
    ge::TensorUtils::SetReuseInputIndex(*output_desc, 0);
    GELOGI("GeLocal Op %s type %s set reuse input.", node.GetName().c_str(), node.GetType().c_str());
  }
  return ge::SUCCESS;
}
}  // namespace ge_local
}  // namespace ge
