/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "register/op_impl_registry.h"

#include "stridedslice.h"

namespace ge {
namespace {
// 如果begin是负数的时候，需要调整为正数
Status NormalizeInput(std::vector<Expression> &input_indexes, const std::vector<Expression> &input_dims) {
  GE_ASSERT_TRUE(input_indexes.size() <= input_dims.size(),
                 "input indexes size: %zu should not more than input shape size: %zu", input_indexes.size(),
                 input_dims.size());
  for (size_t i = 0UL; i < input_indexes.size(); i++) {
    const bool lt_zero = EXPECT_SYMBOL_LT(input_indexes[i], Symbol(0));
    input_indexes[i] = (lt_zero == true) ? input_indexes[i] + input_dims[i] : input_indexes[i];
  }
  return SUCCESS;
}

// 如果new_axis_mask和shrink_axis_mask的bit位与ellipsis_mask冲突，则不生效
void HandleMaskConflict(StridedSliceMaskAttr &mask_attr) {
  mask_attr.new_axis_mask = static_cast<int64_t>((static_cast<uint64_t>(mask_attr.new_axis_mask) &
    static_cast<uint64_t>(mask_attr.ellipsis_mask)) ^ static_cast<uint64_t>(mask_attr.new_axis_mask));
  mask_attr.shrink_axis_mask = static_cast<int64_t>((static_cast<uint64_t>(mask_attr.shrink_axis_mask) &
    static_cast<uint64_t>(mask_attr.ellipsis_mask)) ^ static_cast<uint64_t>(mask_attr.shrink_axis_mask));
  mask_attr.shrink_axis_mask = static_cast<int64_t>((static_cast<uint64_t>(mask_attr.shrink_axis_mask) &
    static_cast<uint64_t>(mask_attr.new_axis_mask)) ^ static_cast<uint64_t>(mask_attr.shrink_axis_mask));
  GELOGI("handle mask conflict, new_axis_mask: %lld, shrink_axis_mask: %lld", mask_attr.new_axis_mask,
         mask_attr.shrink_axis_mask);
}

int64_t CountBitNum(const int64_t num) {
  int64_t count = 0L;
  for (auto n = static_cast<size_t>(num); n > 0; n >>= 1) {
    count += static_cast<int64_t>(n & 1UL);
  }
  return count;
}

bool IsInEllipsisMaskRange(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t pos) {
  return pos >= ellipsis_mask_range.first && pos < ellipsis_mask_range.second;
}

Status AppendNewAxis(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t new_axis_mask,
                     const std::vector<Expression> &input_dims, std::vector<Expression> &input_append_axis_shape,
                     StrdedSliceIndexInputs &index_input) {
  const size_t begin_len = index_input.start_indexes.size();
  int64_t new_axis_num = 0L;
  for (size_t i = 0U; i < begin_len; ++i) {
    if ((static_cast<size_t>(new_axis_mask) & (1 << i)) > 0) {
      new_axis_num++;
    }
  }
  uint64_t mask_pos = 0U;
  for (size_t i = 0UL; i < input_dims.size();) {
    if ((static_cast<uint64_t>(new_axis_mask) & (1 << mask_pos)) > 0) {
      if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(input_append_axis_shape.size()))) {
        input_append_axis_shape.emplace_back(input_dims[i++]);
        index_input.is_new_axis.emplace_back(false);
      } else {
        new_axis_num--;
        input_append_axis_shape.emplace_back(Symbol(1));
        index_input.is_new_axis.emplace_back(true);
        mask_pos++;
      }
    } else {
      input_append_axis_shape.emplace_back(input_dims[i++]);
      index_input.is_new_axis.emplace_back(false);
      mask_pos++;
    }
  }
  if (new_axis_num > 0) {
    input_append_axis_shape.emplace_back(Symbol(1));
    index_input.is_new_axis.emplace_back(true);
  }
  return SUCCESS;
}

std::pair<int64_t, int64_t> GetEllipsisMaskRange(const StridedSliceMaskAttr &mask_attr,
                                                 const int64_t slice_dim_num, const int64_t input_size) {
  const int64_t bit_count = CountBitNum(mask_attr.new_axis_mask);
  const int64_t ellipsis_mask_num = input_size + bit_count - slice_dim_num + 1;
  int64_t pos = 0L;
  for (; pos < slice_dim_num; pos++) {
    if ((static_cast<size_t>(mask_attr.ellipsis_mask) & (1 << static_cast<size_t>(pos))) > 0) {
      break;
    }
  }
  // 左开右闭
  if (pos == slice_dim_num) {
    // 未设置ellipsis_mask
    return std::make_pair(-1, -1);
  }
  GELOGI("ellipsis_mask_range: [%lld, %lld)", pos, pos + ellipsis_mask_num);
  return std::make_pair(pos, pos + ellipsis_mask_num);
}

Status HandleEllipsisMask(const int64_t ellipsis_mask_index, StrdedSliceIndexInputs &index_input) {
  GE_ASSERT_TRUE(index_input.start_indexes.size() == index_input.strides_indexes.size(),
                 "start_index size: %zu should equal to strides_index size:%zu", index_input.start_indexes.size(),
                 index_input.strides_indexes.size());
  for (int64_t i = 0; i < static_cast<int64_t>(index_input.start_indexes.size()); i++) {
    if (i == ellipsis_mask_index) {
      index_input.start_indexes[i] = Symbol(0);
      index_input.strides_indexes[i] = Symbol(1);
      break;
    }
  }
  return SUCCESS;
}

Status FillMissionIndex(const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                        const std::vector<Expression> &input_dims, StrdedSliceIndexInputs &index_input) {
  std::vector<Expression> origin_start_indexes = index_input.start_indexes;
  std::vector<Expression> origin_strides_indexes = index_input.strides_indexes;
  const auto ori_start_size = origin_start_indexes.size();
  for (size_t i = ori_start_size; i < input_dims.size(); i++) {
    origin_start_indexes.emplace_back(Symbol(0));
    origin_strides_indexes.emplace_back(Symbol(1));
  }
  GE_ASSERT_SUCCESS(NormalizeInput(origin_start_indexes, input_dims));
  index_input.start_indexes.clear();
  index_input.strides_indexes.clear();
  int64_t start_index_pos = 0L;
  for (size_t i = 0UL; i < input_dims.size(); i++) {
    if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
      index_input.start_indexes.emplace_back(Symbol(0));
      index_input.strides_indexes.emplace_back(Symbol(1));
      if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
        start_index_pos++;
      }
      continue;
    }
    if (EXPECT_SYMBOL_LT(origin_start_indexes[start_index_pos], input_dims[i])) {
      index_input.start_indexes.emplace_back(origin_start_indexes[start_index_pos]);
    } else {
      index_input.start_indexes.emplace_back(input_dims[i]);
    }
    index_input.strides_indexes.emplace_back(origin_strides_indexes[start_index_pos]);
    start_index_pos++;
  }
  return ge::SUCCESS;
}

Status HandleBeginMask(const StridedSliceMaskAttr &strided_slice_attr, const std::vector<Expression> &input_dims,
                       const std::pair<int64_t, int64_t> &ellipsis_mask_range, StrdedSliceIndexInputs &index_input) {
  uint64_t mask_pos = 0U;
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
      if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
        mask_pos++;
      }
      continue;
    }
    int64_t strides_value = 0L;
    GE_ASSERT_TRUE(index_input.strides_indexes[i].GetConstValue(strides_value));
    if ((static_cast<uint64_t>(strided_slice_attr.begin_mask) & (1 << mask_pos)) > 0) {
      index_input.start_indexes[i] = (strides_value > 0) ? Symbol(0) : input_dims[i] - Symbol(1);
    }
    mask_pos++;
  }
  return SUCCESS;
}

Status HandleMaskAttr(const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                      const std::vector<Expression> &input_append_axis_shape,
                      const StridedSliceMaskAttr &strided_slice_attr, StrdedSliceIndexInputs &index_input) {
  // 处理ellipsis_mask
  GE_ASSERT_SUCCESS(HandleEllipsisMask(ellipsis_mask_range.first, index_input));
  // 补充缺省的index维度
  GE_ASSERT_SUCCESS(FillMissionIndex(ellipsis_mask_range, input_append_axis_shape, index_input));
  // 处理begin_mask
  HandleBeginMask(strided_slice_attr, input_append_axis_shape, ellipsis_mask_range, index_input);
  return SUCCESS;
}
}  // namespace

graphStatus InferShapeStridedSlice(const std::vector<Expression> &input_x_dims, const StridedSliceMaskAttr &mask_attr,
                                   StrdedSliceIndexInputs &index_input) {
  auto stride_slice_attr = mask_attr;
  HandleMaskConflict(stride_slice_attr);
  const std::pair<int64_t, int64_t> ellipsis_mask_range =
      GetEllipsisMaskRange(stride_slice_attr, static_cast<int64_t>(index_input.start_indexes.size()),
                           static_cast<int64_t>(input_x_dims.size()));
  std::vector<Expression> input_append_axis_shape;
  GE_ASSERT_SUCCESS(AppendNewAxis(ellipsis_mask_range, stride_slice_attr.new_axis_mask, input_x_dims,
                                  input_append_axis_shape, index_input));
  GE_ASSERT_SUCCESS(HandleMaskAttr(ellipsis_mask_range, input_append_axis_shape, stride_slice_attr, index_input));
  return SUCCESS;
}
}  // namespace ge