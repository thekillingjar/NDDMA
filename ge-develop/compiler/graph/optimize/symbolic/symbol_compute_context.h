/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_SYMBOL_COMPUTE_CONTEXT_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_SYMBOL_COMPUTE_CONTEXT_H_

#include <type_traits>
#include "common/checker.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/symbolic_tensor.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"

namespace gert {
class InferSymbolComputeContext : public InferSymbolShapeContext {
 public:
  /**
  * 根据输出index，获取输出符号化Symbolshape指针，该接口仅在编译态使用；
  * @param index 输出index；
  * @return 输出符号化Symbolshape指针，index非法时，返回空指针。
  */
  SymbolTensor *GetOutputSymbolTensor(const size_t index) {
    return GetOutputPointer<SymbolTensor>(index);
  }

  bool GetConstInputDims(const size_t index, std::vector<int64_t> &dims) const {
    GE_ASSERT_NOTNULL(GetNodeName());
    GE_ASSERT_NOTNULL(GetNodeType());
    dims.clear();
    auto shape = this->GetInputSymbolShape(index);
    if (shape == nullptr) {
      GELOGW("shape is null, index %u, node %s[%s].", index, GetNodeName(), GetNodeType());
      return false;
    }
    for (const auto &s : shape->GetDims()) {
      int64_t dim = 0;
      if (!s.GetConstValue(dim)) {
        dims.clear();
        GELOGW("GetConstValue failed, expr type(%u) not ConstExpr or impl is null, index %zu, node %s[%s].",
               s.GetExprType(), index, GetNodeName(), GetNodeType());
        return false;
      }
      dims.push_back(dim);
    }
    return true;
  }
};

static_assert(std::is_standard_layout<InferSymbolComputeContext>::value,
              "The class InferSymbolComputeContext must be a POD");
}  // namespace gert
#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_SYMBOL_COMPUTE_CONTEXT_H_
