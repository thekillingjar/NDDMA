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
#include <cstdint>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
namespace ge {
namespace {
/**
 * Split算子符号化推导，将一个张量按照某一维度等分成m份
 * data：一个输入张量
 * num_split：要分割成几份，该参数要>=0
 * split_dim：要分割的维度，整型可以为负数，如果为负表示从后面数第几个维度，例如-1代表倒数第一个维度，不能超过输入张量的维度，
 *            该维度的长度要能被num_split整除
 * 例如
 * data: {10, 2, 4}
 * num_split: 2
 * split_dim: 0
 * 表示将data按照0轴分割成两个
 * output:
 * {5, 2, 4}
 * {5, 2, 4}
 */
template <typename T>
graphStatus GetInputSymbolTensorValue(gert::InferSymbolShapeContext *context, const size_t tensor_index,
                               const size_t expr_index, T &value) {
  auto tensor = context->GetInputSymbolTensor(tensor_index);
  if (tensor == nullptr) {
    GELOGW("Symbol Infer unsupported, get tensor index[%zu] is nullptr, node %s[%s]", tensor_index,
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  if (tensor->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get split_dim symbolic value is nullptr, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  auto exprs = *tensor->GetSymbolicValue();
  if (exprs.size() <= expr_index) {
    GELOGW("Symbol Infer unsupported, get symbolic value failed at expression index %zu, the vector expression size is %zu, node %s[%s]", expr_index, exprs.size(),
       context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  if (exprs[expr_index].GetConstValue<T>(value) == false) {
    GELOGW("Symbol Infer unsupported, get symbolic value failed at expr_index %zu, the value is not const value, node %s[%s]", expr_index, context->GetNodeName(),
       context->GetNodeType());
    return UNSUPPORTED;
  }
  return SUCCESS;
}

graphStatus InferShape4Split(gert::InferSymbolShapeContext *context) {
  // 获取split_dim，存储在input0中
  int32_t split_dim = 0;
  if (GetInputSymbolTensorValue(context, 0, 0, split_dim) == UNSUPPORTED) {
    GELOGW("Symbol Infer unsupported, get symbolic value failed, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  auto in_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto *numSplit = attrs->GetAttrPointer<int64_t>(0);
  GE_ASSERT_NOTNULL(numSplit);
  int64_t num_split = *numSplit;
  GE_ASSERT_TRUE(num_split > 0);
  int64_t dim_num = in_shape->GetDimNum();
  split_dim = split_dim >= 0 ? split_dim : split_dim + dim_num;
  GE_ASSERT_TRUE(split_dim >= 0 && split_dim < dim_num);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  Symbol num_split_sym(num_split);
  // todo 添加guard output_dim_size是整数, 依赖Mod运算
  auto output_dim_size = in_shape->GetDim(split_dim) / num_split_sym;
  *out_shape = *in_shape;
  out_shape->MutableDims()[split_dim] = output_dim_size;
  for (int64_t i = 1; i < num_split; i++) {
    auto out_shape_dynamic = context->GetOutputSymbolShape(i);
    GE_ASSERT_NOTNULL(out_shape_dynamic);
    *out_shape_dynamic = *out_shape;
  }
  return ge::GRAPH_SUCCESS;
}

/**
 * SplitV算子符号化推导，将输入data按照某一个轴分割成m份，但是不均分而是按照一个列表配置分割
 * data：输入张量
 * num_split：分割的份数，整型要求>=0
 * split_dim：分割的维度，整型可以为负数，如果为负表示从后面数第几个维度，例如-1代表倒数第一个维度，不能超过输入张量的维度
 * size_splits：一个列表每一份元素的梳理，整型列表，要求>=0(?),可以为-1，但是只能有一个-1，代表剩下的部分都分割到这份
 * 例如
 * data: {10, 3, 5}
 * num_split: 3
 * split_dim: 0
 * size_splits: [3, -1, 5]
 * 表示将data按照维度0分割成3份，第一份为3，第三份为5，剩下的部分分给第二份
 * output:
 * {3, 3, 5}
 * {2(10-3-5), 3, 5}
 * {5, 3, 5}
 */
graphStatus GetSplitVInput(gert::InferSymbolShapeContext *context, int64_t &num_split, int64_t &split_dim,
                           size_t dim_num, std::vector<int64_t> &size_splits) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto numSplit = attrs->GetAttrPointer<int64_t>(0);
  GE_ASSERT_NOTNULL(numSplit);
  num_split = *numSplit;
  GE_ASSERT_TRUE(num_split > 0);

  constexpr size_t TENSOR_IDX = 2;
  constexpr size_t EXPR_IDX = 0;
  if (GetInputSymbolTensorValue(context, TENSOR_IDX, EXPR_IDX, split_dim) == UNSUPPORTED) {
    GELOGW("Symbol Infer unsupported, get symbolic value failed, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  split_dim = split_dim >= 0 ? split_dim : split_dim + static_cast<int64_t>(dim_num);
  // 检查split_dim是否在有效范围内
  GE_ASSERT_TRUE(split_dim >= 0 && split_dim < static_cast<int64_t>(dim_num), "split_dim wrong");

  auto in_tensor1 = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(in_tensor1);
  if (in_tensor1->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  auto in_tensor1_exprs = *in_tensor1->GetSymbolicValue();
  GE_ASSERT_TRUE(static_cast<int64_t>(in_tensor1_exprs.size()) == num_split, "tensorSize");
  for (auto &expr : in_tensor1_exprs) {
    int64_t expr_value = 0;
    if (expr.GetConstValue(expr_value) == false) {
      GELOGW("Symbol Infer unsupported, get symbolic value failed, the value is not const value, node %s[%s]",
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    size_splits.push_back(expr_value);
  }
  return SUCCESS;
}
graphStatus InferShape4SplitV(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  int64_t num_split = 0L;
  int64_t split_dim = 0L;
  std::vector<int64_t> size_splits;
  auto ret = GetSplitVInput(context, num_split, split_dim, in_shape->GetDimNum(), size_splits);
  if (ret != SUCCESS) {
    return ret;
  }
  // guard保证size_splits之和小于等于dim
  int64_t sum_split = 0L;
  for (auto val_split : size_splits) {
    if (val_split != -1) {
      GE_ASSERT_TRUE(!AddOverflow(sum_split, val_split, sum_split),
                     "sum_split is overflow max value,sum_split[%lld], val_split[%lld]", sum_split, val_split);
    }
  }
  ASSERT_SYMBOL_LE(Symbol(sum_split), in_shape->GetDim(split_dim));
  int64_t dynamic_value_idx = -1L;
  int64_t split_size_value_sum = 0L;
  int64_t dynamic_value_num = 0L;
  for (int32_t i = 0; i < num_split; i++) {
    if (size_splits[i] == -1) {
      dynamic_value_num++;
      GE_ASSERT_TRUE(dynamic_value_num <= 1);
      dynamic_value_idx = i;
    } else {
      split_size_value_sum += size_splits[i];
    }
  }
  for (int64_t i = 0; i < num_split; i++) {
    auto out_shape = context->GetOutputSymbolShape(i);
    GE_ASSERT_NOTNULL(out_shape);
    *out_shape = *in_shape;
    out_shape->MutableDims()[split_dim] = Symbol(size_splits[i]);
  }
  if (dynamic_value_idx != -1L) {
    auto out_shape = context->GetOutputSymbolShape(dynamic_value_idx);
    GE_ASSERT_NOTNULL(out_shape);
    out_shape->MutableDims()[split_dim] = in_shape->GetDim(split_dim) - Symbol(split_size_value_sum);
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4SplitD(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);

  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  const auto *numSplit = attrs->GetAttrPointer<int64_t>(1);
  GE_ASSERT_NOTNULL(numSplit);
  int64_t num_split = *numSplit;
  GE_ASSERT_TRUE(num_split > 0);

  const auto *splitDim = attrs->GetAttrPointer<int64_t>(0);
  GE_ASSERT_NOTNULL(splitDim);
  int64_t split_dim = *splitDim;
  int64_t dim_num = in_shape->GetDimNum();
  // 处理为负数的情况
  split_dim = split_dim >= 0 ? split_dim : split_dim + dim_num;
  GE_ASSERT_TRUE(split_dim >= 0 && split_dim < dim_num);

  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  Symbol num_split_sym(num_split);
  // todo 添加guard output_dim_size是整数
  auto output_dim_size = in_shape->GetDim(split_dim) / num_split_sym;
  *out_shape = *in_shape;
  out_shape->MutableDims()[split_dim] = output_dim_size;
  for (int64_t i = 1; i < num_split; i++) {
    auto out_shape_dynamic = context->GetOutputSymbolShape(i);
    GE_ASSERT_NOTNULL(out_shape_dynamic);
    *out_shape_dynamic = *out_shape;
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4SplitVD(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto numSplit = attrs->GetAttrPointer<int64_t>(2);
  GE_ASSERT_NOTNULL(numSplit);
  int64_t num_split = *numSplit;
  GE_ASSERT_TRUE(num_split > 0);
  // 获取split_dim
  const auto *splitDim = attrs->GetAttrPointer<int64_t>(1);
  GE_ASSERT_NOTNULL(splitDim);
  int64_t split_dim = *splitDim;
  int64_t indim_num = in_shape->GetDimNum();
  // 处理为负数的情况
  split_dim = split_dim >= 0 ? split_dim : split_dim + indim_num;
  // 检查split_dim是否在有效范围内
  GE_ASSERT_TRUE(split_dim >= 0 && split_dim < static_cast<int64_t>(indim_num), "split_dim wrong");
  // 获取size_splits
  const auto cv = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(cv);
  GE_ASSERT_TRUE(static_cast<int64_t>(cv->GetSize()) == num_split, "size_splits should equal num_split");
  std::vector<int64_t> size_splits;
  for (int64_t i = 0; i < num_split; i++) {
    int64_t expr_value = 0;
    expr_value = cv->GetData()[i];
    size_splits.push_back(expr_value);
  }
  int64_t dynamic_value_idx = -1L;
  int64_t split_size_value_sum = 0;
  int64_t dynamic_value_num = 0;
  for (int32_t i = 0; i < num_split; i++) {
    if (size_splits[i] == -1) {
      dynamic_value_num++;
      GE_ASSERT_TRUE(dynamic_value_num <= 1);
      dynamic_value_idx = i;
    } else {
      split_size_value_sum += size_splits[i];
    }
  }
  for (int64_t i = 0; i < num_split; i++) {
    auto out_shape = context->GetOutputSymbolShape(i);
    GE_ASSERT_NOTNULL(out_shape);
    *out_shape = *in_shape;
    out_shape->MutableDims()[split_dim] = Symbol(size_splits[i]);
  }
  if (dynamic_value_idx != -1L) {
    auto out_shape = context->GetOutputSymbolShape(dynamic_value_idx);
    GE_ASSERT_NOTNULL(out_shape);
    out_shape->MutableDims()[split_dim] = in_shape->GetDim(split_dim) - Symbol(split_size_value_sum);
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Split).InferSymbolShape(InferShape4Split);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SplitV).InferSymbolShape(InferShape4SplitV);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SplitD).InferSymbolShape(InferShape4SplitD);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SplitVD).InferSymbolShape(InferShape4SplitVD);
}  // namespace
}  // namespace ge