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
#include "common/util.h"
#include "common/checker.h"
#include "framework/common/types.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kXInputIndex = 0UL;
constexpr size_t kStartInputIndex = 1UL;
constexpr size_t kEndInputIndex = 2UL;
constexpr size_t kStridesInputIndex = 3UL;
constexpr size_t kOutputIndex = 0UL;

enum class StridedSliceAttrIndex {
  kAttrBeginMaskIndex,
  kAttrEndMaskIndex,
  kAttrEllipsisMaskIndex,
  kAttrNewAxisMaskIndex,
  kAttrShrinkAxisMaskIndex,
  kEnd
};

enum class StridedSliceDAttrIndex {
  kAttrStartInputIndex,
  kAttrEndInputIndex,
  kAttrStridesInputIndex,
  kAttrBeginMaskIndex,
  kAttrEndMaskIndex,
  kAttrEllipsisMaskIndex,
  kAttrNewAxisMaskIndex,
  kAttrShrinkAxisMaskIndex,
  kEnd
};

struct StridedSliceAttr {
  int64_t begin_mask{0};
  int64_t end_mask{0};
  int64_t ellipsis_mask{0};
  int64_t new_axis_mask{0};
  int64_t shrink_axis_mask{0};
};

struct StrdedSliceIndexInputs {
  std::vector<int64_t> start_indexes;
  std::vector<int64_t> end_indexes;
  std::vector<int64_t> strides_indexes;
  std::vector<bool> is_new_axis;
};

graphStatus GetValueFromInputConstData(const gert::InferSymbolComputeContext *context, const size_t index,
                                       std::vector<int64_t> &dims) {
  auto symbol_tensor = context->GetInputSymbolTensor(index);
  GE_UNSUPPORTED_IF_NULL(symbol_tensor);
  auto symbols = symbol_tensor->GetSymbolicValue();
  if (symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get %zu input symbolic value failed, node %s[%s].", index,
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  for (size_t i = 0UL; i < symbols->size(); i++) {
    int64_t dim = 0L;
    if (!(*symbols)[i].GetConstValue(dim)) {
      GELOGW("SymbolicKernel compute unsupported, reason: get %zu input const value failed, node %s[%s].", i,
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    dims.emplace_back(dim);
  }
  return SUCCESS;
}

// 如果begin和end是负数的时候，需要调整为正数
Status NormalizeInput(std::vector<int64_t> &input_indexes, const std::vector<int64_t> &input_dims) {
  GE_ASSERT_TRUE(input_indexes.size() <= input_dims.size(),
                 "input indexes size: %zu should not more than input shape size: %zu", input_indexes.size(),
                 input_dims.size());
  for (size_t i = 0UL; i < input_indexes.size(); i++) {
    input_indexes[i] = input_indexes[i] < 0 ? input_indexes[i] + input_dims[i] : input_indexes[i];
    GE_ASSERT_TRUE(input_indexes[i] >= 0, "input_indexes[%zu]=%lld", i, input_indexes[i]);
  }
  return SUCCESS;
}

// 如果new_axis_mask和shrink_axis_mask的bit位与ellipsis_mask冲突，则不生效
void HandleMaskConflict(StridedSliceAttr &strided_slice_attr) {
  strided_slice_attr.new_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.new_axis_mask) &
                                       static_cast<uint64_t>(strided_slice_attr.ellipsis_mask)) ^
                                      static_cast<uint64_t>(strided_slice_attr.new_axis_mask));
  strided_slice_attr.shrink_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask) &
                                          static_cast<uint64_t>(strided_slice_attr.ellipsis_mask)) ^
                                         static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask));
  strided_slice_attr.shrink_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask) &
                                          static_cast<uint64_t>(strided_slice_attr.new_axis_mask)) ^
                                         static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask));
  GELOGI("handle mask conflict, new_axis_mask: %lld, shrink_axis_mask: %lld", strided_slice_attr.new_axis_mask,
         strided_slice_attr.shrink_axis_mask);
}

int64_t CountBitNum(const int64_t num) {
  int64_t count = 0L;
  if (num <= 0) {
    return count;
  }
  for (uint64_t n = num; n > 0; n >>= 1) {
    count += (n & 1L);
  }
  return count;
}

bool IsInEllipsisMaskRange(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t pos) {
  return ((pos >= ellipsis_mask_range.first) && (pos < ellipsis_mask_range.second));
}

Status AppendNewAxis(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t new_axis_mask,
                     const std::vector<int64_t> &input_dims, std::vector<int64_t> &input_append_axis_shape,
                     StrdedSliceIndexInputs &index_input) {
  const size_t begin_len = index_input.start_indexes.size();
  int64_t new_axis_num = 0;
  for (size_t i = 0L; i < begin_len; ++i) {
    if ((static_cast<uint64_t>(new_axis_mask) & (1 << i)) > 0) {
      new_axis_num++;
    }
  }
  int64_t mask_pos = 0L;
  for (size_t i = 0L; i < input_dims.size();) {
    if ((static_cast<uint64_t>(new_axis_mask) & (1 << mask_pos)) > 0) {
      if ((IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(input_append_axis_shape.size())))) {
        input_append_axis_shape.emplace_back(input_dims[i++]);
        index_input.is_new_axis.emplace_back(false);
      } else {
        new_axis_num--;
        input_append_axis_shape.emplace_back(1L);
        index_input.is_new_axis.emplace_back(true);
        mask_pos++;
      }
    } else {
      input_append_axis_shape.emplace_back(input_dims[i++]);
      index_input.is_new_axis.emplace_back(false);
      mask_pos++;
    }
  }
  while (new_axis_num-- > 0) {
    input_append_axis_shape.emplace_back(1L);
    index_input.is_new_axis.emplace_back(true);
  }
  GELOGI("Input shape after insert new axis: %s", SymbolicInferUtil::VectorToStr(input_append_axis_shape).c_str());
  return SUCCESS;
}

std::pair<int64_t, int64_t> GetEllipsisMaskRange(const StridedSliceAttr &strided_slice_attr,
                                                 const int64_t slice_dim_num, const int64_t input_size) {
  int64_t bit_count = CountBitNum(strided_slice_attr.new_axis_mask);
  int64_t ellipsis_mask_num = input_size + bit_count - slice_dim_num + 1;
  int64_t pos = 0L;
  for (; pos < slice_dim_num; pos++) {
    if ((static_cast<uint64_t>(strided_slice_attr.ellipsis_mask) & (1 << pos)) > 0) {
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

Status HandleEllipsisMask(const int64_t ellipsis_mask_index, const std::vector<int64_t> &input_dims,
                          StrdedSliceIndexInputs &index_input) {
  GE_ASSERT_TRUE(index_input.start_indexes.size() == index_input.end_indexes.size(),
                 "start_index size: %zu should equal to end_index size:%zu", index_input.start_indexes.size(),
                 index_input.end_indexes.size());
  GE_ASSERT_TRUE(index_input.start_indexes.size() == index_input.strides_indexes.size(),
                 "start_index size: %zu should equal to strides_index size:%zu", index_input.start_indexes.size(),
                 index_input.strides_indexes.size());
  for (int64_t i = 0UL; i < static_cast<int64_t>(index_input.start_indexes.size()); i++) {
    if (i == ellipsis_mask_index) {
      index_input.start_indexes[i] = 0;
      index_input.end_indexes[i] = input_dims[i];
      index_input.strides_indexes[i] = 1;
      break;
    }
  }
  GELOGD("start index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorToStr(index_input.start_indexes).c_str());
  GELOGD("end index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorToStr(index_input.end_indexes).c_str());
  GELOGD("strides index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorToStr(index_input.strides_indexes).c_str());
  return SUCCESS;
}

void GetShrinkAxisIndex(const int64_t shrink_axis_mask, const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                        const int64_t index_size, std::set<int64_t> &shrink_axis_indexes) {
  int64_t bit_pos = 0L;
  for (int64_t i = 0UL; i < index_size; i++) {
    if ((static_cast<uint64_t>(shrink_axis_mask) & (1 << bit_pos)) > 0) {
      if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
        continue;
      }
      shrink_axis_indexes.insert(i);
    }
    bit_pos++;
  }
}

Status HandleShrinkAxisShape(const std::set<int64_t> &shrink_axis_indexes, StrdedSliceIndexInputs &index_input) {
  for (const auto &shrink_axis_id : shrink_axis_indexes) {
    GE_ASSERT_TRUE((shrink_axis_id < static_cast<int64_t>(index_input.start_indexes.size())) && (shrink_axis_id >= 0));
    GELOGI("Change strideslice index to [%lld, %lld, 1] of dim[%lld]", index_input.start_indexes[shrink_axis_id],
           index_input.end_indexes[shrink_axis_id], shrink_axis_id);
    index_input.end_indexes[shrink_axis_id] = index_input.start_indexes[shrink_axis_id] + 1;
    index_input.strides_indexes[shrink_axis_id] = 1;
  }
  return SUCCESS;
}

Status FillMissionIndex(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const std::vector<int64_t> &input_dims,
                        StrdedSliceIndexInputs &index_input) {
  std::vector<int64_t> origin_start_indexes = index_input.start_indexes;
  std::vector<int64_t> origin_end_indexes = index_input.end_indexes;
  std::vector<int64_t> origin_strides_indexes = index_input.strides_indexes;
  GELOGD("origin_start_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorToStr(origin_start_indexes).c_str());
  GELOGD("origin_end_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorToStr(origin_end_indexes).c_str());
  GELOGD("origin_strides_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorToStr(origin_strides_indexes).c_str());
  auto ori_start_size = origin_start_indexes.size();
  for (size_t i = ori_start_size; i < input_dims.size(); i++) {
    origin_start_indexes.emplace_back(0L);
    origin_end_indexes.emplace_back(input_dims[i]);
    origin_strides_indexes.emplace_back(1L);
  }
  GE_ASSERT_SUCCESS(NormalizeInput(origin_start_indexes, input_dims));
  GE_ASSERT_SUCCESS(NormalizeInput(origin_end_indexes, input_dims));
  index_input.start_indexes.clear();
  index_input.end_indexes.clear();
  index_input.strides_indexes.clear();
  int64_t start_index_pos = 0L;
  for (size_t i = 0UL; i < input_dims.size(); i++) {
    if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
      index_input.start_indexes.emplace_back(0L);
      index_input.end_indexes.emplace_back(input_dims[i]);
      index_input.strides_indexes.emplace_back(1L);
      if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
        start_index_pos++;
      }
      continue;
    }
    index_input.start_indexes.emplace_back(std::min(origin_start_indexes[start_index_pos], input_dims[i] - 1));
    index_input.end_indexes.emplace_back(std::min(origin_end_indexes[start_index_pos], input_dims[i]));
    index_input.strides_indexes.emplace_back(origin_strides_indexes[start_index_pos]);
    start_index_pos++;
  }
  GELOGD("start index after insert fill missing: %s",
         SymbolicInferUtil::VectorToStr(index_input.start_indexes).c_str());
  GELOGD("end index after insert handle fill missing: %s",
         SymbolicInferUtil::VectorToStr(index_input.end_indexes).c_str());
  GELOGD("strides index after insert handle fill missing: %s",
         SymbolicInferUtil::VectorToStr(index_input.strides_indexes).c_str());
  return SUCCESS;
}

void HandleBeginEndMask(const StridedSliceAttr &strided_slice_attr, const std::vector<int64_t> &input_dims,
                        const std::pair<int64_t, int64_t> &ellipsis_mask_range, StrdedSliceIndexInputs &index_input) {
  int64_t mask_pos = 0L;
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
      if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
        mask_pos++;
      }
      continue;
    }
    if ((static_cast<uint64_t>(strided_slice_attr.begin_mask) & (1 << mask_pos)) > 0) {
      index_input.start_indexes[i] = (index_input.strides_indexes[i] > 0) ? 0 : input_dims[i] - 1;
    }
    if ((static_cast<uint64_t>(strided_slice_attr.end_mask) & (1 << mask_pos)) > 0) {
      index_input.end_indexes[i] = (index_input.strides_indexes[i] > 0) ? input_dims[i] : 0;
    }
    mask_pos++;
  }
  GELOGI("start index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorToStr(index_input.start_indexes).c_str());
  GELOGI("end index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorToStr(index_input.end_indexes).c_str());
  GELOGI("strides index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorToStr(index_input.strides_indexes).c_str());
}

Status CalcOutputShape(const int64_t shrink_axis_mask, const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                       StrdedSliceIndexInputs &index_input, std::vector<Expression> &output_symbols_shape) {
  std::set<int64_t> shrink_axis_indexes;
  GetShrinkAxisIndex(shrink_axis_mask, ellipsis_mask_range, static_cast<int64_t>(index_input.start_indexes.size()),
                     shrink_axis_indexes);
  // 处理ShrinkAxisIndex，将shrink axis的维度设置成[start, start + 1, 1]
  GE_ASSERT_SUCCESS(HandleShrinkAxisShape(shrink_axis_indexes, index_input));
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (shrink_axis_indexes.count(i) > 0) {
      continue;
    }
    GE_ASSERT_TRUE(index_input.strides_indexes[i] != 0L, "index_input.strides_indexes[%zu]=%lld", i,
                   index_input.strides_indexes[i]);
    int64_t result_dim = std::max(
        0L,
        static_cast<int64_t>(std::ceil(static_cast<float>(index_input.end_indexes[i] - index_input.start_indexes[i]) /
                                       static_cast<float>(index_input.strides_indexes[i]))));
    auto output_dim = (index_input.is_new_axis[i] == true) ? 1L : result_dim;
    output_symbols_shape.emplace_back(Symbol(output_dim));
  }
  return SUCCESS;
}

bool IsNeedEndStridedSlice(const int64_t strides_index, const int64_t end_index, const int64_t current_index) {
  // 如果步长为正数，则当current_index大于end时，停止循环
  // 如果步长为负数，则当current_index小于end时，停止循环
  return (((strides_index > 0) && (current_index >= end_index)) ||
          ((strides_index < 0) && (current_index <= end_index)));
}

Status StridedSliceOutputSymbolsValue(const std::vector<Expression> &input_x_symbols,
                                      const std::vector<int64_t> &input_dims, const StrdedSliceIndexInputs &index_input,
                                      std::vector<Expression> &output_symbols_value) {
  std::vector<Expression> last_output_symbols = input_x_symbols;
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    output_symbols_value.clear();
    // 如果end索引比start索引小，shape为0，不需要刷数字
    if (((index_input.end_indexes[i] - index_input.start_indexes[i]) * index_input.strides_indexes[i]) <= 0) {
      GELOGW("value will become empty if end indexes: %lld less than start_indexes: %lld", index_input.end_indexes[i],
             index_input.start_indexes[i]);
      return SUCCESS;
    }
    int64_t block_size = std::accumulate(input_dims.begin() + i + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    int64_t block_num = static_cast<int64_t>(last_output_symbols.size()) / block_size / input_dims[i];
    GELOGI("block num: %lld, input_dims: %lld block size[%zu] : %lld", block_num, input_dims[i], i, block_size);
    for (int64_t j = 0L; j < block_num; j++) {
      int64_t index = index_input.start_indexes[i];
      while (!IsNeedEndStridedSlice(index_input.strides_indexes[i], index_input.end_indexes[i], index)) {
        GE_ASSERT_TRUE(index_input.strides_indexes[i] != 0L, "index_input.strides_indexes[%zu]=%lld", i,
                       index_input.strides_indexes[i]);
        GE_ASSERT_TRUE((static_cast<int64_t>(index) < input_dims[i]) && (static_cast<int64_t>(index) >= 0),
                       "index=%lld, input_dims[%zu]=%lld", index, i, input_dims[i]);
        auto begin_iter = last_output_symbols.begin() + (j * input_dims[i] + index) * block_size;
        auto end_iter = last_output_symbols.begin() + (j * input_dims[i] + index + 1) * block_size;
        output_symbols_value.insert(output_symbols_value.end(), begin_iter, end_iter);
        index += index_input.strides_indexes[i];
      }
    }
    last_output_symbols = output_symbols_value;
  }
  return SUCCESS;
}

Status GetStridedSliceIndexInput(gert::InferSymbolComputeContext *context, StrdedSliceIndexInputs &index_input) {
  auto ret = GetValueFromInputConstData(context, kStartInputIndex, index_input.start_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = GetValueFromInputConstData(context, kEndInputIndex, index_input.end_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = GetValueFromInputConstData(context, kStridesInputIndex, index_input.strides_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  return SUCCESS;
}

Status GetStridedSliceDIndexInput(const gert::InferSymbolComputeContext *context, StrdedSliceIndexInputs &index_input) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  auto attr_start_input = attrs->GetListInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrStartInputIndex));
  GE_ASSERT_NOTNULL(attr_start_input);

  auto attr_end_input = attrs->GetListInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrEndInputIndex));
  GE_ASSERT_NOTNULL(attr_end_input);

  auto attr_strides_input = attrs->GetListInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrStridesInputIndex));
  GE_ASSERT_NOTNULL(attr_strides_input);

  for (size_t i = 0; i < attr_start_input->GetSize(); i++) {
    index_input.start_indexes.push_back(attr_start_input->GetData()[i]);
  }

  for (size_t i = 0; i < attr_end_input->GetSize(); i++) {
    index_input.end_indexes.push_back(attr_end_input->GetData()[i]);
  }

  for (size_t i = 0; i < attr_strides_input->GetSize(); i++) {
    index_input.strides_indexes.push_back(attr_strides_input->GetData()[i]);
  }
  return SUCCESS;
}

Status GetStridedSliceMaskAttr(const gert::InferSymbolComputeContext *context, StridedSliceAttr &strided_slice_attr) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  auto begin_mask_ptr = (strcmp(context->GetNodeType(), STRIDEDSLICE) == 0)
                            ? attrs->GetInt(static_cast<size_t>(StridedSliceAttrIndex::kAttrBeginMaskIndex))
                            : attrs->GetInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrBeginMaskIndex));
  auto end_mask_ptr = (strcmp(context->GetNodeType(), STRIDEDSLICE) == 0)
                          ? attrs->GetInt(static_cast<size_t>(StridedSliceAttrIndex::kAttrEndMaskIndex))
                          : attrs->GetInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrEndMaskIndex));
  auto ellipsis_mask_ptr = (strcmp(context->GetNodeType(), STRIDEDSLICE) == 0)
                               ? attrs->GetInt(static_cast<size_t>(StridedSliceAttrIndex::kAttrEllipsisMaskIndex))
                               : attrs->GetInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrEllipsisMaskIndex));
  auto new_axis_mask_ptr = (strcmp(context->GetNodeType(), STRIDEDSLICE) == 0)
                               ? attrs->GetInt(static_cast<size_t>(StridedSliceAttrIndex::kAttrNewAxisMaskIndex))
                               : attrs->GetInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrNewAxisMaskIndex));
  auto shrink_axis_mask_ptr =
      (strcmp(context->GetNodeType(), STRIDEDSLICE) == 0)
          ? attrs->GetInt(static_cast<size_t>(StridedSliceAttrIndex::kAttrShrinkAxisMaskIndex))
          : attrs->GetInt(static_cast<size_t>(StridedSliceDAttrIndex::kAttrShrinkAxisMaskIndex));

  GE_ASSERT_NOTNULL(begin_mask_ptr);
  GE_ASSERT_NOTNULL(end_mask_ptr);
  GE_ASSERT_NOTNULL(ellipsis_mask_ptr);
  GE_ASSERT_NOTNULL(new_axis_mask_ptr);
  GE_ASSERT_NOTNULL(shrink_axis_mask_ptr);

  strided_slice_attr.begin_mask = *begin_mask_ptr;
  strided_slice_attr.end_mask = *end_mask_ptr;
  strided_slice_attr.ellipsis_mask = *ellipsis_mask_ptr;
  strided_slice_attr.new_axis_mask = *new_axis_mask_ptr;
  strided_slice_attr.shrink_axis_mask = *shrink_axis_mask_ptr;
  return SUCCESS;
}

Status HandleMaskAttr(const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                      const std::vector<int64_t> &input_append_axis_shape, const StridedSliceAttr &strided_slice_attr,
                      StrdedSliceIndexInputs &index_input) {
  // 处理ellipsis_mask
  GE_ASSERT_SUCCESS(HandleEllipsisMask(ellipsis_mask_range.first, input_append_axis_shape, index_input));
  // 补充缺省的index维度
  GE_ASSERT_SUCCESS(FillMissionIndex(ellipsis_mask_range, input_append_axis_shape, index_input));
  // 处理begin_mask和end_mask
  HandleBeginEndMask(strided_slice_attr, input_append_axis_shape, ellipsis_mask_range, index_input);
  return SUCCESS;
}
}  // namespace

static graphStatus StridedSliceSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("StridedSlice Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  StrdedSliceIndexInputs index_input;
  Status ret = PARAM_INVALID;
  if (strcmp(context->GetNodeType(), STRIDEDSLICE) == 0) {
    ret = GetStridedSliceIndexInput(context, index_input);
  } else if (strcmp(context->GetNodeType(), STRIDEDSLICED) == 0) {
    ret = GetStridedSliceDIndexInput(context, index_input);
  } else {
    GELOGW("Node type: %s is not StridedSlice or StridedSliceD.", context->GetNodeType());
  }
  if (ret != SUCCESS) {
    return ret;
  }
  std::vector<int64_t> input_x_dims;
  if (!context->GetConstInputDims(kXInputIndex, input_x_dims)) {
    return UNSUPPORTED;
  }
  auto input_x_symbols = context->GetInputSymbolTensor(kXInputIndex)->GetSymbolicValue();
  if (input_x_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  StridedSliceAttr strided_slice_attr;
  GetStridedSliceMaskAttr(context, strided_slice_attr);
  HandleMaskConflict(strided_slice_attr);
  std::pair<int64_t, int64_t> ellipsis_mask_range =
      GetEllipsisMaskRange(strided_slice_attr, static_cast<int64_t>(index_input.start_indexes.size()),
                           static_cast<int64_t>(input_x_dims.size()));
  std::vector<int64_t> input_append_axis_shape;
  GE_ASSERT_SUCCESS(AppendNewAxis(ellipsis_mask_range, strided_slice_attr.new_axis_mask, input_x_dims,
                                  input_append_axis_shape, index_input));
  GE_ASSERT_SUCCESS(HandleMaskAttr(ellipsis_mask_range, input_append_axis_shape, strided_slice_attr, index_input));

  auto out_symbols_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(out_symbols_tensor);
  std::vector<Expression> output_symbols_shape;
  GE_ASSERT_SUCCESS(
      CalcOutputShape(strided_slice_attr.shrink_axis_mask, ellipsis_mask_range, index_input, output_symbols_shape));
  GE_ASSERT_NOTNULL(out_symbols_tensor->MutableSymbolicValue());
  out_symbols_tensor->MutableOriginSymbolShape().MutableDims() = output_symbols_shape;
  auto output_symbols_value = out_symbols_tensor->MutableSymbolicValue();
  GE_ASSERT_SUCCESS(
      StridedSliceOutputSymbolsValue(*input_x_symbols, input_append_axis_shape, index_input, *output_symbols_value));
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*out_symbols_tensor).c_str());
  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(StridedSlice, StridedSliceSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(StridedSliceD, StridedSliceSymbolicKernelCompute);
}  // namespace ge
