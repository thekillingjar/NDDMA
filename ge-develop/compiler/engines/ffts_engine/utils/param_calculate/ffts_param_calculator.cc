/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inc/ffts_param_calculator.h"
#include "inc/ffts_log.h"
#include "inc/ffts_utils.h"
#include "inc/memory_slice.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ffts {
namespace {
/*
 *  @ingroup ffts
 *  @brief   Calculate the size of a shape format According to the byte.
 *  @return  SUCCESS or FAILED
 */
Status CalcRangeSize(const vector<int64_t> &range, int64_t &range_size, const uint32_t &data_type_size) {
  range_size = 1;
  for (size_t i = 0U; i < range.size(); ++i) {
    const int64_t dim_range = range[i];
    FFTS_MUL_OVERFLOW(range_size, dim_range, range_size);
  }

  if (data_type_size > static_cast<uint32_t>(ge::kDataTypeSizeBitOffset)) {
    int64_t bits_on_byte = kOneByteBitNum;
    int64_t bits_on_data = data_type_size - ge::kDataTypeSizeBitOffset;
    int64_t data_nums_in_byte = bits_on_byte / bits_on_data;
    FFTS_ADD_OVERFLOW(range_size, data_nums_in_byte - 1, range_size);
    if (data_nums_in_byte == 0) {
      return FAILED;
    }
    range_size /= data_nums_in_byte;
    return SUCCESS;
  }
  FFTS_MUL_OVERFLOW(range_size, static_cast<int64_t>(data_type_size), range_size);
  return SUCCESS;
}
}

Status ParamCalculator::CalcAutoThreadInput(const ge::NodePtr &node,
                                            vector<vector<vector<int64_t>>> &tensor_slice,
                                            vector<uint64_t> &task_addr_offset,
                                            std::vector<uint32_t> &input_tensor_indexes)
{
  if (tensor_slice.empty()) {
    FE_LOGE("[FFTSPlus][CalcAutoThreadInput]Input tensor slice is empty.");
    return FAILED;
  }
  for (auto const &anchor : node->GetAllInDataAnchors()) {
    if (anchor == nullptr) {
      continue;
    }
    if (ge::AnchorUtils::GetStatus(anchor) == ge::ANCHOR_SUSPEND) {
      FFTS_LOGD("[FFTSPlus][CalcAutoThreadInput] Anchor Status is ge::ANCHOR_SUSPEND.");
      continue;
    }
    const uint32_t anchor_index = static_cast<uint32_t>(anchor->GetIdx());
    const auto iter = find(input_tensor_indexes.cbegin(), input_tensor_indexes.cend(), anchor_index);
    if (iter == input_tensor_indexes.cend()) {
      task_addr_offset.emplace_back(0);
      continue;
    }
     if (tensor_slice.size() <= 1) {
      FFTS_LOGI("[FFTSPlus][CalcAutoThreadInput]Anchor index:%u slice num:1.", anchor_index);
      task_addr_offset.emplace_back(0);
      continue;
    }
    const ge::GeTensorDescPtr &tensor_desc_ptr = node->GetOpDesc()->MutableInputDesc(anchor_index);
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    uint32_t data_type_size = 0U;
    (void)GetDataTypeSize(tensor_desc_ptr->GetDataType(), data_type_size);
    if (anchor_index >= tensor_slice[0U].size()) {
      FFTS_LOGE("[FFTSPlus][CalcAutoThreadInput] Anchor index: %u exceeds slice number: %zu.",
                anchor_index, tensor_slice[0].size());
      return FAILED;
    }

    auto &tensor_range = tensor_slice[0U][anchor_index];
    int64_t thread_offset = 0;
    if (CalcRangeSize(tensor_range, thread_offset, data_type_size) != SUCCESS) {
      FFTS_LOGE("[FFTSPlus][CalcAutoThreadInput]Calculate Thread Offset failed.");
      return FAILED;
    }
    FFTS_LOGD("Input anchorIndex: %u, threadoffset: %ld, dtype_size: %u.", anchor_index, thread_offset, data_type_size);
    task_addr_offset.emplace_back(thread_offset);
  }
  return SUCCESS;
}

Status ParamCalculator::CalcAutoThreadOutput(const ge::NodePtr &node,
                                             vector<vector<vector<int64_t>>> &tensor_slice,
                                             vector<uint64_t> &task_addr_offset,
                                             std::vector<uint32_t> &output_tensor_indexes) {
  FFTS_CHECK(tensor_slice.empty(), FE_LOGE("Output tensor slice is empty!"), return FAILED);
  if (node->GetAllOutDataAnchorsSize() > tensor_slice[0U].size()) {
    FFTS_LOGE("[FFTSPlus][CalcAutoThreadOutput] Anchor size: %u exceeds slice number: %zu.",
              node->GetAllOutDataAnchorsSize(), tensor_slice[0].size());
    return FAILED;
  }
  for (auto const &anchor : node->GetAllOutDataAnchors()) {
    const uint32_t anchor_index = static_cast<uint32_t>(anchor->GetIdx());
    const auto &iter = find(output_tensor_indexes.begin(), output_tensor_indexes.end(), anchor_index);
    if (iter == output_tensor_indexes.end()) {
      task_addr_offset.emplace_back(0);
      continue;
    }
    const ge::GeTensorDescPtr &tensor_desc_ptr = node->GetOpDesc()->MutableOutputDesc(anchor_index);
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    if (tensor_slice.size() == 1) {
      FFTS_LOGI("[FFTSPlus][CalcAutoThreadOutput]Anchor index:%u slice num:1.", anchor_index);
      task_addr_offset.emplace_back(0);
      continue;
    }
    uint32_t data_type_size = 0U;
    if (GetDataTypeSize(tensor_desc_ptr->GetDataType(), data_type_size) != SUCCESS) {
        FFTS_LOGE("[FFTSPlus][CalcAutoThreadOutput] Node:%s anchor index: %u data type: %d is not supported.",
                  node->GetName().c_str(), anchor_index, tensor_desc_ptr->GetDataType());
      return FAILED;
    }

    auto &tensor_range = tensor_slice[0U][anchor_index];
    int64_t thread_offset = 0;
    // calc nontail tensor size of thread
    if (CalcRangeSize(tensor_range, thread_offset, data_type_size) != SUCCESS) {
      FFTS_LOGE("[FFTSPlus][CalcAutoThreadOutput]Calculate Thread Offset failed.");
      return FAILED;
    }

    // tail tensor size of thread
    int64_t slice_tensor_size = 0;
    auto &tail_tensor_range = tensor_slice[tensor_slice.size() - 1U][anchor_index];
    if (CalcRangeSize(tail_tensor_range, slice_tensor_size, data_type_size) != SUCCESS) {
      FFTS_LOGE("[FFTSPlus][CalcAutoThreadOutput]Calculate Thread Offset failed.");
      return FAILED;
    }

    // max of tail and nontail. because the tiny tail tensor is merged in last notail
    slice_tensor_size = std::max(thread_offset, slice_tensor_size);

    (void)ge::AttrUtils::SetInt(tensor_desc_ptr, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_SIZE, slice_tensor_size);
    FFTS_LOGD("Output Index: %u, thread_offset: %ld, slice_tensor_size: %ld.",
              anchor_index, thread_offset, slice_tensor_size);

    task_addr_offset.emplace_back(thread_offset);
  }
  return SUCCESS;
}

void ParamCalculator::ConvertTensorSlice(const std::vector<std::vector<std::vector<DimRange>>> &in_tensor_slice,
                                         std::vector<std::vector<std::vector<int64_t>>> &out_tensor_slice) {
  for (auto &iter : in_tensor_slice) {
    std::vector<std::vector<int64_t>> anchorShape;
    for (auto &itlr : iter) {
      std::vector<int64_t> shape;
      for (const DimRange &range : itlr) {
        int64_t shapeValue = range.higher >= range.lower ? range.higher - range.lower : 0;
        shape.emplace_back(shapeValue);
      }
      anchorShape.emplace_back(shape);
    }
    out_tensor_slice.emplace_back(anchorShape);
  }
}
}  // namespace ffts
