/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
namespace ge {
namespace {
/**
 * Slice算子符号化推导的核心逻辑
 * @return 是否成功
 */
graphStatus CheckAndInfer4Slice(const gert::SymbolShape *in_shape, const gert::SymbolTensor *offsets_tensor,
                                const gert::SymbolTensor *size_tensor, gert::SymbolShape *out_shape) {
  const auto &dims = in_shape->GetDims();

  /* 校验参数：
   * 1. 每个轴的 起始位置(offsets) ∈ [0, dim]，其中offsets[i]dim，表示整个轴都不要。
   * 2. 每个轴的 切片大小(size) ∈ [-1, dim]。其中size[i]为-1时，表示整个轴都要。
   * 3. 每个轴的 起始位置+切片大小 <= dim
   */
  for (size_t i = 0U; i < dims.size(); ++i) {
    // todo 增加guard校验
    ASSERT_SYMBOL_GE(offsets_tensor->GetSymbolicValue()->at(i), Symbol(0));
    ASSERT_SYMBOL_LE(offsets_tensor->GetSymbolicValue()->at(i), dims[i]);
  }
  /* 计算输出shape，逻辑为：
   * 若 size[i] = -1, 表示取i轴从起点开始剩下的全部元素，则out_shape[i] = in_shape[i] - offsets[i]
   * 若 size[i] > -1, 表示取size[i]个元素，则out_shape[i] = size[i]
   */
  out_shape->Clear();
  for (size_t i = 0U; i < dims.size(); ++i) {
    if (EXPECT_SYMBOL_EQ(size_tensor->GetSymbolicValue()->at(i), Symbol(-1))) {
      out_shape->AppendDim(in_shape->GetDim(i) - offsets_tensor->GetSymbolicValue()->at(i));
    } else {
      out_shape->AppendDim(size_tensor->GetSymbolicValue()->at(i));
    }
  }
  return GRAPH_SUCCESS;
}

/**
 * Slice 算子的符号化shape推导
 * 【算子功能】按照指定的起点(offsets)，从输入中切分指定大小(size)的数据
 * 【算子约束】
 *      1. 起点(offsets)和切分大小(size)的轴数(dimNum)与输入一致；
 *      2. 起点(offsets)和切分大小(size)的每个维度在输入shape的有效范围内；
 *      3. 起点数据不能包含-1，切分大小可以包含-1（表示本轴从起点开始，剩下的数据都要保留）
 *      举例： 输入shape=(10,10)，offsets=[2,2]，size=[7,7]，输出shape=(7,7)。
 *      注意：offsets和size需要取到具体数值，因此需要设置为数据依赖(data depend)。
 * 【推导逻辑】
 *      1. 按照算子约束校验各个输入shape
 *      2. 默认将size的数值设置为输出shape，因为这个参数的明确含义就是切片大小
 *      3. 若size中包含-1，则对应轴的输出dim等于对应轴的输入dim - 起点，也就是上述所说的从起点开始，剩下的数据都要
 */
graphStatus InferShape4Slice(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto offsets_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(offsets_tensor);
  if (offsets_tensor->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get offset symbolic value is nullptr, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }

  auto size_tensor = context->GetInputSymbolTensor(2);
  GE_UNSUPPORTED_IF_NULL(size_tensor);
  if (size_tensor->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get size symbolic value is nullptr, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }

  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  // 校验offsets、size的轴数应该与输入一致
  GE_ASSERT_EQ(in_shape->GetDimNum(), offsets_tensor->GetSymbolicValue()->size());
  GE_ASSERT_EQ(in_shape->GetDimNum(), size_tensor->GetSymbolicValue()->size());
  // 校验offsets、size的数据类型
  auto offsets_desc = context->GetInputDesc(1);
  GE_ASSERT_NOTNULL(offsets_desc);
  auto offsets_dtype = offsets_desc->GetDataType();
  GE_ASSERT(offsets_dtype == DT_INT32 || offsets_dtype == DT_INT64, "offsets data type must in (int32, int64)");
  auto size_desc = context->GetInputDesc(2);
  GE_ASSERT_NOTNULL(size_desc);
  auto size_dtype = size_desc->GetDataType();
  GE_ASSERT(size_dtype == DT_INT32 || size_dtype == DT_INT64, "size data type must in (int32, int64)");
  return CheckAndInfer4Slice(in_shape, offsets_tensor, size_tensor, out_shape);
}

/**
 * SliceD算子符号化推导的核心逻辑
 * @return 是否成功
 */
graphStatus CheckAndInfer4SliceD(const gert::SymbolShape *in_shape,
                                 const gert::TypedContinuousVector<int64_t> *offsets_list,
                                 const gert::TypedContinuousVector<int64_t> *size_list, gert::SymbolShape *out_shape) {
  const auto &dims = in_shape->GetDims();
  /* 校验参数：
   * 1. 每个轴的 起始位置(offsets) ∈ [0, dim]，其中offsets[i]dim，表示整个轴都不要。
   * 2. 每个轴的 切片大小(size) ∈ [-1, dim]。其中size[i]为-1时，表示整个轴都要。
   * 3. 每个轴的 起始位置+切片大小 <= dim
   */
  for (size_t i = 0U; i < dims.size(); ++i) {
    ASSERT_SYMBOL_GE(Symbol(offsets_list->GetData()[i]), Symbol(0));
    ASSERT_SYMBOL_LE(Symbol(offsets_list->GetData()[i]), dims[i]);
  }
  /* 计算输出shape，逻辑为：
   * 若 size[i] = -1, 表示取i轴从起点开始剩下的全部元素，则out_shape[i] = in_shape[i] - offsets[i]
   * 若 size[i] > -1, 表示取size[i]个元素，则out_shape[i] = size[i]
   */
  out_shape->Clear();
  for (size_t i = 0U; i < dims.size(); ++i) {
    const int64_t offset = offsets_list->GetData()[i];
    const int64_t size = size_list->GetData()[i];
    if (size == -1) {
      ASSERT_SYMBOL_GE(dims[i] - Symbol(offset), Symbol(0));
      out_shape->AppendDim(dims[i] - Symbol(offset));
    } else {
      out_shape->AppendDim(Symbol(size));
    }
  }
  return GRAPH_SUCCESS;
}

/**
 * SliceD 算子的符号化shape推导
 * 【算子功能】按照指定的起点(offsets)，从输入中切分指定大小(size)的数据
 * 【算子约束】
 *      1. 起点(offsets)和切分大小(size)的轴数(dimNum)与输入一致；
 *      2. 起点(offsets)和切分大小(size)的每个维度在输入shape的有效范围内；
 *      3. 起点数据不能包含-1，切分大小可以包含-1（表示本轴从起点开始，剩下的数据都要保留）
 *      举例： 输入shape=(10,10)，offsets=[2,2]，size=[7,7]，输出shape=(7,7)。
 *      注意：offsets和size需要取到具体数值，因此需要设置为数据依赖(data depend)。
 * 【推导逻辑】
 *      1. 按照算子约束校验各个输入shape
 *      2. 默认将size的数值设置为输出shape，因为这个参数的明确含义就是切片大小
 *      3. 若size中包含-1，则对应轴的输出dim等于对应轴的输入dim - 起点，也就是上述所说的从起点开始，剩下的数据都要
 */
graphStatus InferShape4SliceD(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto offsets_list = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(offsets_list);
  GE_ASSERT_NOTNULL(offsets_list->GetData());
  auto size_list = attrs->GetListInt(1);
  GE_ASSERT_NOTNULL(size_list);
  GE_ASSERT_NOTNULL(size_list->GetData());
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  // 校验offsets、size的轴数应该与输入一致
  GE_ASSERT_EQ(in_shape->GetDimNum(), offsets_list->GetSize());
  GE_ASSERT_EQ(in_shape->GetDimNum(), size_list->GetSize());
  return CheckAndInfer4SliceD(in_shape, offsets_list, size_list, out_shape);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Slice).InferSymbolShape(InferShape4Slice);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SliceD).InferSymbolShape(InferShape4SliceD);
}  // namespace
}  // namespace ge