/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_INFER_UTIL_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_INFER_UTIL_H_
#include "ge_common/ge_api_types.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "op_impl_infer_symbol_shape.h"
#include "framework/common/util.h"
#include "graph/node.h"

namespace ge {
const Expression kSymbolZero{Symbol(0)};
const Expression kSymbolOne{Symbol(1)};
const Expression kSymbolTwo{Symbol(2)};

#define GE_UNSUPPORTED_IF_NULL(exp, ...)                                                                               \
  do {                                                                                                                 \
    if (exp == nullptr) {                                                                                              \
      auto msg = CreateErrorMsg(__VA_ARGS__);                                                                          \
      GELOGW("unsupported assert failed: %s", (msg.empty() ? #exp : msg.data()));                                      \
      return ge::UNSUPPORTED;                                                                                          \
    }                                                                                                                  \
  } while (false)

class SymbolicInferUtil {
 public:
  static graphStatus GetConstInt(const gert::SymbolTensor *tensor, DataType dt, int64_t &value);

  template <typename T1, typename T2>
  static bool IsDimValid(const T1 shape_size, const T2 dim_value) {
    int64_t minimum_num = static_cast<int64_t>(shape_size) * static_cast<int64_t>(-1);
    int64_t maximum_num = static_cast<int64_t>(shape_size) - 1;
    return static_cast<int64_t>(dim_value) >= minimum_num && static_cast<int64_t>(dim_value) <= maximum_num;
  }

  template <typename T>
  static graphStatus ReduceDimsWithoutKeepDims(const gert::SymbolShape *x_shape, const std::vector<T> &axes_dims,
                                        const size_t axes_size, gert::SymbolShape *output_shape) {
    T dim_num = x_shape->GetDimNum();
    output_shape->MutableDims().clear();
    for (T j = 0; j < dim_num; ++j) {
      bool reduce_flag = false;
      for (size_t i = 0; i < axes_size; ++i) {
        if (!IsDimValid(dim_num, axes_dims[i])) {
          GELOGE(PARAM_INVALID, "axes_dims[%d] is invalid", i);
          return PARAM_INVALID;
        }
        T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
        if (dim == j) {
          reduce_flag = true;
          break;
        }
      }
      if (!reduce_flag) {
        output_shape->AppendDim(x_shape->GetDim(j));
      }
    }
    GELOGD("ReduceDimsWithoutKeepDims is SUCCESS");
    return GRAPH_SUCCESS;
  }

  template <typename T>
  static graphStatus ReduceDimsWithKeepDims(const gert::SymbolShape *x_shape, const std::vector<T> &axes_dims,
                                     const size_t axes_size, gert::SymbolShape *output_shape) {
    T dim_num = x_shape->GetDimNum();
    const bool is_scalar = dim_num == 0;
    *output_shape = *x_shape;
    if (is_scalar) {
      // no need to update output shape, when input is scalar
      return GRAPH_SUCCESS;
    }
    for (size_t i = 0; i < axes_size; ++i) {
      if (!IsDimValid(dim_num, axes_dims[i])) {
        GELOGE(PARAM_INVALID, "axes_dims[%d] is invalid", i);
        return PARAM_INVALID;
      }
      T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
      output_shape->MutableDims().at(dim) = Symbol(1);
    }
    GELOGD("ReduceDimsWithKeepDims is SUCCESS");
    return GRAPH_SUCCESS;
  }

  template <typename T>
  static graphStatus ReduceDims(const gert::SymbolShape *x_shape, const gert::SymbolTensor *axes_tensor, size_t axes_size,
                         const bool keep_dims, gert::SymbolShape *output_shape) {
    auto exps = *axes_tensor->GetSymbolicValue();
    std::vector<T> axes_dims_const;
    axes_dims_const.reserve(axes_size);
    for (size_t i = 0; i < axes_size; ++i) {
      if (exps[i].GetConstValue<T>(axes_dims_const[i]) == false) {
        return UNSUPPORTED;
      }
    }
    if (keep_dims) {
      return ReduceDimsWithKeepDims<T>(x_shape, axes_dims_const, axes_size, output_shape);
    }
    return ReduceDimsWithoutKeepDims<T>(x_shape, axes_dims_const, axes_size, output_shape);
  }
  static std::string VectorExpressionToStr(const std::vector<ge::Expression> &vec) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      result += (vec[i].Str().get());
      if (i < vec.size() - 1) {
        result += ", ";
      }
    }
    result += "]";
    return result;
  }

  static std::string VectorToStr(const std::vector<int64_t> &vec) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      result += std::to_string(vec[i]);
      if (i < vec.size() - 1) {
        result += ", ";
      }
    }
    result += "]";
    return result;
  }
  /**
  * 广播`shapes`，生成`b_shape`，并标记`need_broadcast`。
  * @param shapes 输入的shape，每个元素是一个shape，每个shape是一个维度数组
  * @param b_shape 广播后输出的shape，如果不需要广播，那么`b_shape`为空
  * @return SUCCESS为成功
  */
  static Status Broadcast(const std::vector<std::vector<Expression>> &shapes,
                          std::vector<Expression> &b_shape);
  static std::string DumpSymbolTensor(const gert::SymbolTensor &symbolic_tensor);
  static bool IsSupportCondNode(const NodePtr &node);
  static NodePtr GetCondInput(const NodePtr &node);
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_INFER_UTIL_H_
