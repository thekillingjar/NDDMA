/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arg_list_manager.h"
#include "common/checker.h"

namespace att {
const int32_t kContainerSizeSearch = -1;
ge::Status ArgListManager::LoadArgList(const TuningSpacePtr &tuning_space) {
  // 每次加载清除历史信息
  arg_list_map_.clear();
  replace_container_.clear();
  // 添加轴size表达式
  for (const auto &sub_axis : tuning_space->sub_axes) {
    GE_ASSERT_SUCCESS(AddArgExpr(sub_axis->name, sub_axis->repeat), "Add axis[%s] expr failed.",
                      sub_axis->name.c_str());
  }
  // 添加临时空间表达式
  for (const auto &pair : tuning_space->tmp_buffer) {
    string arg_name = kArgsNameTmpBuffer;
    if (pair.first != -1) {
      arg_name = GetTmpBufferName(pair.first);
    }
    SetArgExpr(arg_name, pair.second);
  }
  SetArgExpr(kArgsNameBuiltInTmpBuffer, tuning_space->builtin_tmp_buffer);
  // 添加queue、buf buffer num表达式
  for (const auto &container : tuning_space->containers) {
    if (container->GetBufferNum() == kContainerSizeSearch) {
      GE_ASSERT_SUCCESS(SetArgExpr(container->name, CreateExpr(container->name.c_str())), "Add axis[%s] expr failed.",
                        container->name.c_str());
    } else {
      GE_ASSERT_SUCCESS(SetArgExpr(container->name, CreateExpr(container->GetBufferNum())), "Add axis[%s] expr failed.",
                        container->name.c_str());
    }
    GE_ASSERT_SUCCESS(SetTensorSizeExpr(container->allocated_tensors), "Set [%s]'s tensor expr failed.",
                      container->name.c_str());
  }
  for (const auto &container : tuning_space->global_containers) {
    GE_ASSERT_SUCCESS(SetTensorSizeExpr(container->allocated_tensors), "Set [%s]'s tensor expr failed.",
                      container->name.c_str());
  }
  return ge::SUCCESS;
}

ge::Status ArgListManager::SetTensorSizeExpr(const std::vector<TensorPtr> &allocated_tensors) {
  for (const auto &tensor : allocated_tensors) {
    auto axis_size = tensor->dim_info.size();
    auto axis_repeats = tensor->repeat.size();
    auto axis_strides = tensor->stride.size();
    GE_ASSERT_TRUE(axis_size == axis_repeats, "Tensor[%s]'s axis size is not equal to repeats.", tensor->name.c_str());
    GE_ASSERT_TRUE(axis_size == axis_strides, "Tensor[%s]'s axis size is not equal to strides.", tensor->name.c_str());
    auto tensor_occupy_expr = CreateExpr(1U);
    // case1(非连续场景):         repeats = [a, b, c], strides = [b * c2, c2, 1], tensor_size = a * b * c2
    // case2:(非连续场景,一个B轴): repeats = [1, b, c], strides = [0, c2, 1], tensor_size = b * c2
    // case3:(非连续场景,均是B轴): repeats = [1, 1, 1], strides = [0, 0, 0], tensor_size = 1
    bool all_stride_zero = true;
    for (size_t i = 0u; i < axis_size; ++i) {
      if (tensor->stride[i] == 0) {
        continue;
      }
      tensor_occupy_expr = tensor->stride[i] * tensor->repeat[i];
      all_stride_zero = false;
      break;
    }
    tensor_occupy_expr = all_stride_zero ? CreateExpr(1) : tensor_occupy_expr.Simplify();
    tensor_occupy_expr = CreateExpr(tensor->data_type_size) * tensor_occupy_expr;
    Expr tensor_expr = SetTensorInfo(tensor->name, tensor_occupy_expr);
    GE_ASSERT_SUCCESS(SetArgExpr(tensor->name, tensor_expr), "Add tensor[%s] expr failed.", tensor->name.c_str());
    GELOGD("Get tensor[%s] size [%s].", tensor->name.c_str(), tensor_occupy_expr.Str().get());
  }
  return ge::SUCCESS;
}
}  // namespace att