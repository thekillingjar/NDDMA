/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pad_slice_optimize_pass.h"

#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "utils/autofuse_utils.h"

namespace ge {
namespace {
constexpr int32_t kNeedRemovePadPattern = 1;
// not support now
constexpr int32_t kNeedRemovePadSlicePattern = 2;
constexpr int32_t kPaddingsDim2 = 2;
const char *const kPadType = "Pad";
const char *const kSliceType = "Slice";
const char *const kPaddings = "paddings";

graphStatus GetListPaddings(const NodePtr &node, std::vector<int64_t> &value_vec, const std::string &input) {
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  Tensor val_tensor;
  GE_ASSERT_SUCCESS(op.GetInputConstData(input.c_str(), val_tensor));
  const auto dims = val_tensor.GetTensorDesc().GetShape().GetDims();
  GE_ASSERT_TRUE((dims.size() == 2U) || (dims.size() == 1U), "Input paddings tensor must be 1-D/2-D.");
  auto dims_len = dims.at(0U);
  if (dims.size() == 2U) {
    const auto dim_0 = dims.at(0U);
    const auto dim_1 = dims.at(1U);
    GE_ASSERT_TRUE(dim_0 > 0L, "Paddings dim0 must be positive.");
    GE_ASSERT_TRUE(dim_1 > 0L, "Paddings dim1 must be positive.");
    dims_len = dim_0 * dim_1;
  }
  GE_ASSERT_NOTNULL(val_tensor.GetData());
  const DataType dtype = val_tensor.GetTensorDesc().GetDataType();
  for (int32_t idx = 0; idx < dims_len; ++idx) {
    if (dtype == ge::DT_INT32) {
      int32_t tensor_val = *reinterpret_cast<const int32_t *>(val_tensor.GetData() + idx * sizeof(int32_t));
      value_vec.emplace_back(tensor_val);
    } else if (dtype == ge::DT_INT64) {
      int64_t tensor_val = *reinterpret_cast<const int64_t *>(val_tensor.GetData() + idx * sizeof(int64_t));
      value_vec.emplace_back(tensor_val);
    } else {
      GELOGW("Input paddings tensor must be int32 or int64.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

std::vector<std::vector<int64_t>> VectorTransToPair(const std::vector<int64_t> &vec) {
  std::vector<std::vector<int64_t>> result;
  result.reserve(vec.size() / kPaddingsDim2);
  for (size_t i = 0U; i < vec.size(); i += kPaddingsDim2) {
    result.push_back({vec[i], vec[i + 1]});
  }
  return result;
}

graphStatus ParsingPadNode(const NodePtr &node, std::vector<int64_t> &pad_input_shape,
                           std::vector<std::vector<int64_t>> &paddings_value) {
  GE_CHECK_NOTNULL(node);
  auto pad_op_desc = node->GetOpDesc();
  pad_input_shape = pad_op_desc->GetInputDescPtr(0U)->GetShape().GetDims();

  std::vector<int64_t> paddings = {};
  GE_ASSERT_SUCCESS(GetListPaddings(node, paddings, kPaddings));
  GE_ASSERT_TRUE((paddings.size() % kPaddingsDim2 == 0), "Paddings dim must be 2-D.");
  paddings_value = VectorTransToPair(paddings);
  return GRAPH_SUCCESS;
}

bool IsAllEqual(const std::vector<std::vector<int32_t>> &vec, int32_t target_value) {
  for (const auto &row : vec) {
    for (auto elem : row) {
      if (elem != target_value) {
        return false;
      }
    }
  }
  return !vec.empty();
}

bool IsPatternMatchRequirements(std::vector<int64_t> &offsets_value, std::vector<int64_t> &sizes_value,
                                std::vector<int64_t> &pad_input_shape,
                                std::vector<std::vector<int64_t>> &paddings_value,
                                std::vector<std::vector<int32_t>> &all_results) {
  std::vector<int32_t> results(offsets_value.size(), 0);
  for (size_t idx = 0U; idx < offsets_value.size(); ++idx) {
    if ((offsets_value.at(idx) >= paddings_value[idx][0]) &&
        (offsets_value.at(idx) < paddings_value[idx][0] + pad_input_shape.at(idx)) &&
        (sizes_value.at(idx) <= (paddings_value[idx][0] + pad_input_shape.at(idx) - (offsets_value.at(idx))))) {
      results[idx] = kNeedRemovePadPattern;
    } else if (((offsets_value.at(idx) < paddings_value[idx][0]) &&
                (sizes_value.at(idx) <= (paddings_value[idx][0] - offsets_value.at(idx)))) ||
               ((offsets_value.at(idx) >= (paddings_value[idx][0] + pad_input_shape.at(idx))) &&
                (sizes_value.at(idx) <= paddings_value[idx][1]))) {
      results[idx] = kNeedRemovePadSlicePattern;
    } else {
      GELOGI(
          "Index is %zu, begin value is %ld, size value is %ld, pad input shape value is %ld,"
          "paddings value dim0 is %ld,dim1 is %ld",
          idx, offsets_value.at(idx), sizes_value.at(idx), pad_input_shape.at(idx), paddings_value[idx][0],
          paddings_value[idx][1]);
      all_results.push_back(results);
      return false;
    }
  }
  all_results.push_back(results);
  return true;
}

graphStatus CheckSliceMatchRequirements(const NodePtr &node_after_pad, std::vector<int64_t> &pad_input_shape,
                                        std::vector<std::vector<int64_t>> &paddings_value,
                                        std::vector<std::vector<int32_t>> &all_results, bool &need_skip) {
  if (node_after_pad->GetType() == kSliceType) {
    const auto &src_anchor = node_after_pad->GetInDataAnchor(1U);
    GE_ASSERT_NOTNULL(src_anchor);
    auto peer_anchor = src_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_anchor);
    auto peer_node = peer_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_node);
    if (!OpTypeUtils::IsConstNode(peer_node->GetType())) {
      GELOGI("abnormal slice node %s, offsets input is not const, skip process", node_after_pad->GetNamePtr());
      need_skip = true;
      return GRAPH_SUCCESS;
    }
    std::vector<int64_t> offsets_value;
    std::vector<int64_t> sizes_value;
    GE_ASSERT_SUCCESS(AutofuseUtils::GetListIntByInputOrAttr(node_after_pad, offsets_value, "offsets", "offsets"));
    GE_ASSERT_SUCCESS(AutofuseUtils::GetListIntByInputOrAttr(node_after_pad, sizes_value, "size", "size"));
    GE_ASSERT_TRUE(offsets_value.size() == sizes_value.size(),
                   "begin_value size must be equal as size_value size");
    GE_ASSERT_TRUE(offsets_value.size() == pad_input_shape.size(),
                   "begin_value size must be equal as pad_input_shape size");
    GE_ASSERT_TRUE(offsets_value.size() == paddings_value.size(),
                   "begin_value size must be equal as paddings_value dim0 size");
    bool is_requirement_matched =
        IsPatternMatchRequirements(offsets_value, sizes_value, pad_input_shape, paddings_value, all_results);
    if (!is_requirement_matched) {
      GELOGI("Requirement one is not matched, skip this pad node.");
      need_skip = true;
    }
  } else {
    GELOGI("After node is not slice, skip optimize.");
    need_skip = true;
  }
  return GRAPH_SUCCESS;
}

graphStatus ExtractConstNodeDtype(const NodePtr &slice_node, DataType &dtype) {
  if (slice_node->GetType() == kSliceType) {
    const auto op = ge::OpDescUtils::CreateOperatorFromNode(slice_node);
    ge::Tensor val_tensor;
    GE_ASSERT_SUCCESS(op.GetInputConstData("offsets", val_tensor));
    const auto tensor_desc = val_tensor.GetTensorDesc();
    dtype = tensor_desc.GetDataType();
  }
  return GRAPH_SUCCESS;
}

graphStatus RefreshOffsetsValue(const NodePtr &slice_node, std::vector<int64_t> &before_paddings_value,
                                DataType dtype) {
  std::vector<int64_t> offsets_value;
  GE_ASSERT_SUCCESS(AutofuseUtils::GetListIntByInputOrAttr(slice_node, offsets_value, "offsets", "offsets"));
  GE_ASSERT_TRUE((offsets_value.size() == before_paddings_value.size()),
                 "offsets_value and before_paddings_value size is different.");
  std::vector<int64_t> new_offsets_value;
  for (size_t i = 0U; i < offsets_value.size(); i++) {
    GELOGI("value is %ld, new_offsets_value is %ld", offsets_value[i], offsets_value[i] - before_paddings_value[i]);
    new_offsets_value.emplace_back(offsets_value[i] - before_paddings_value[i]);
  }

  auto offsets_const_node = slice_node->GetInDataNodes().at(1U);
  const auto tensor_desc = offsets_const_node->GetOpDesc()->MutableOutputDesc(0U);

  GeTensorPtr tensor = nullptr;
  if (dtype == DT_INT32) {
    std::vector<int32_t> reformated_offsets(new_offsets_value.begin(), new_offsets_value.end());
    tensor = ComGraphMakeShared<GeTensor>(*tensor_desc, reinterpret_cast<uint8_t *>(reformated_offsets.data()),
                                          sizeof(int32_t) * new_offsets_value.size());
  } else if (dtype == DT_INT64) {
    tensor = ComGraphMakeShared<GeTensor>(*tensor_desc, reinterpret_cast<uint8_t *>(new_offsets_value.data()),
                                          sizeof(int64_t) * new_offsets_value.size());
  }
  GE_ASSERT_NOTNULL(tensor);
  GE_ASSERT_TRUE(AttrUtils::SetTensor(offsets_const_node->GetOpDesc(), "value", tensor));
  GELOGI("Success to refresh input offsets constant node for slice node %s", slice_node->GetNamePtr());
  return GRAPH_SUCCESS;
}
}

graphStatus PadSliceOptimizePass::Run(const ComputeGraphPtr &graph, bool &changed) {
  GE_CHECK_NOTNULL(graph);
  GELOGD("PadAndSliceOptimizePass::main func begin.");

  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kPadType) {
      auto output_size = node->GetOutDataNodesSize();
      if (!node->GetOutControlNodes().empty() || !node->GetInControlNodes().empty()) {
        return GRAPH_SUCCESS;
      }
      GELOGI("Pad node %s has %zu output nodes", node->GetNamePtr(), output_size);
      const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
      Tensor val_tensor;
      if (op.GetInputConstData(kPaddings, val_tensor) != GRAPH_SUCCESS) {
        GELOGI("Pad node op desc is abnormal, skip optimize");
        return GRAPH_SUCCESS;
      }
      std::vector<int64_t> pad_input_shape;
      std::vector<std::vector<int64_t>> paddings_value;
      GE_ASSERT_SUCCESS(ParsingPadNode(node, pad_input_shape, paddings_value));

      std::vector<std::vector<int32_t>> all_results;
      bool need_skip = false;
      for (size_t out_idx = 0U; (out_idx < output_size) && (!need_skip); ++out_idx) {
        auto node_after_pad = node->GetOutNodes().at(out_idx);
        GE_CHECK_NOTNULL(node_after_pad);
        GE_ASSERT_GRAPH_SUCCESS(
            CheckSliceMatchRequirements(node_after_pad, pad_input_shape, paddings_value, all_results, need_skip));
      }
      if (IsAllEqual(all_results, kNeedRemovePadPattern)) {
        GELOGI("Condition one is satisfied, pad node will be removed.");
        GE_ASSERT_SUCCESS(PostProcess(graph, node));
        changed = true;
      }
    }
  }
  GELOGD("PadAndSliceOptimizePass::main func end.");
  return GRAPH_SUCCESS;
}

graphStatus PadSliceOptimizePass::PostProcess(const ComputeGraphPtr &graph, const NodePtr &node) const {
  auto output_size = node->GetOutDataNodesSize();
  GE_ASSERT_TRUE(output_size >= 1U, "output_size must be greater equal than 1");
  GELOGI("Pad node %s has %zu output nodes", node->GetNamePtr(), output_size);
  std::vector<int64_t> paddings = {};
  GE_ASSERT_SUCCESS(GetListPaddings(node, paddings, kPaddings));
  GE_ASSERT_TRUE(paddings.size() >= kPaddingsDim2, "paddings size must be greater than 2");
  std::vector<int64_t> before_paddings_value = {};
  for (size_t index = 0U; index < paddings.size(); index += kPaddingsDim2) {
    GELOGI("value is %ld", paddings[index]);
    before_paddings_value.emplace_back(paddings[index]);
  }

  for (size_t out_idx = 0U; out_idx < output_size; ++out_idx) {
    auto node_after_pad = node->GetOutNodes().at(out_idx);
    GE_CHECK_NOTNULL(node_after_pad);

    DataType offsets_dtype = DT_INT32;
    GE_ASSERT_SUCCESS(ExtractConstNodeDtype(node_after_pad, offsets_dtype));
    auto op_desc = node_after_pad->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    *op_desc->MutableInputDesc(0U) = node->GetOpDesc()->GetInputDesc(0U);
    GE_ASSERT_SUCCESS(RefreshOffsetsValue(node_after_pad, before_paddings_value, offsets_dtype));
  }
  GE_ASSERT_SUCCESS(AutofuseUtils::DelOneNodeInGraph(graph, node));
  GELOGI("Success to delete pad node %s", node->GetNamePtr());
  return GRAPH_SUCCESS;
}
}