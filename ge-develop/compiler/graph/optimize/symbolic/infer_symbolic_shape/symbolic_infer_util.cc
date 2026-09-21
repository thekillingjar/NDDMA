/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "symbolic_infer_util.h"

#include "graph/utils/node_utils.h"

#include <op_type_utils.h>

namespace ge {
constexpr static size_t kByteBitCount = 8UL;


graphStatus SymbolicInferUtil::GetConstInt(const gert::SymbolTensor *tensor, DataType dt, int64_t &value) {
  if (dt == DT_INT32) {
    int32_t tmp_value = 0;
    GE_ASSERT_TRUE(tensor->GetSymbolicValue()->at(0).GetConstValue<int32_t>(tmp_value),"error info GetConstValue failed");
    value = static_cast<int64_t>(tmp_value);
  } else if (dt == DT_INT64) {
    GE_ASSERT_TRUE(tensor->GetSymbolicValue()->at(0).GetConstValue<int64_t>(value),"error info GetConstValue failed");
  } else {
    GELOGE(PARAM_INVALID, "dt must in [int32, int64]");
    return ge::PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}
Status SymbolicInferUtil::Broadcast(const std::vector<std::vector<Expression>> &shapes,
                                    std::vector<Expression> &b_shape) {
  if (shapes.empty()) {
    b_shape.clear();
    return SUCCESS;
  }
  // 1. 挑选rank最大值
  size_t max_rank = 0;
  for (const auto &shape : shapes) {
    max_rank = std::max(max_rank, shape.size());
  }
  // 2. 初始化输出结果和广播标志
  b_shape.clear();
  // 3. 右对齐方式计算broadcast，比如[4, 2, 3]和[2, 3]，输出为[4, 2, 3]
  for (size_t dim = 0U; dim < max_rank; ++dim) {
    bool found = false;
    for (const auto &shape : shapes) {
      int32_t true_idx = dim + shape.size() - max_rank;
      if (true_idx >= 0) {
        const Expression &s = shape[true_idx];
        if (!found) {
          b_shape.emplace_back(s);
          found = true;
          continue;
        }
        if (EXPECT_SYMBOL_EQ(b_shape[dim], s)) {
          if (s.IsConstExpr()) {
            b_shape[dim] = s;
          }
          GELOGI("Symbol input0[%zu]:%s is equal to input1[%zu]:%s, no need broadcast.",
                 true_idx, b_shape[dim].Serialize().get(), true_idx, s.Serialize().get());
        }  else if (EXPECT_SYMBOL_EQ(s, Symbol(1))) {
          GELOGI("Symbol input1[%zu]:%s is equal to symbol(1), should broadcast to input0[%zu]:%s.",
                 true_idx, s.Serialize().get(), true_idx, b_shape[dim].Serialize().get());
        } else if (EXPECT_SYMBOL_EQ(b_shape[dim], Symbol(1))) {
          GELOGI("Symbol input0[%zu]:%s is equal to symbol(1), should broadcast to input1[%zu]:%s.",
                 true_idx, b_shape[dim].Serialize().get(), true_idx, s.Serialize().get());
          b_shape[dim] = s;
        } else {
          GELOGE(ge::FAILED, "Symbol input0[%zu]:%s is not equal to input1[%zu]:%s which cannot broadcast.",
                 true_idx, b_shape[dim].Serialize().get(), true_idx, s.Serialize().get());
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

std::string SymbolicInferUtil::DumpSymbolTensor(const gert::SymbolTensor &symbolic_tensor) {
  std::string debug_msg = "origin symbol shape: ";
  debug_msg += SymbolicInferUtil::VectorExpressionToStr(symbolic_tensor.GetOriginSymbolShape().GetDims());
  debug_msg += ", symbolic value: ";
  if (symbolic_tensor.GetSymbolicValue() != nullptr) {
    debug_msg +=  SymbolicInferUtil::VectorExpressionToStr(*symbolic_tensor.GetSymbolicValue());
  } else {
    debug_msg += "None";
  }
  return debug_msg;
}

bool SymbolicInferUtil::IsSupportCondNode(const ge::NodePtr &node) {
  GE_WARN_ASSERT(node != nullptr);
  std::string node_type = node->GetType();
  return (kIfOpTypes.find(node_type) != kIfOpTypes.end()) ||
         (kCaseOpTypes.find(node_type) != kCaseOpTypes.end());
}

NodePtr SymbolicInferUtil::GetCondInput(const NodePtr &node) {
  GE_WARN_ASSERT(IsSupportCondNode(node));
  auto cond_input = NodeUtils::GetInDataNodeByIndex(*node, 0);
  GELOGD("Get cond node[%s] input[%s] success.", node->GetNamePtr(), cond_input->GetNamePtr());
  GE_WARN_ASSERT(cond_input != nullptr);
  // 如果node节点的输入时cast，size，stringlength需要再往上找
  const std::set<string> vaild_types = {"Cast", "Size", "StringLength"};
  if (vaild_types.find(cond_input->GetType()) != vaild_types.end()) {
    cond_input = NodeUtils::GetInDataNodeByIndex(*cond_input, 0);
    GELOGD("Cond node[%s] input is cast/size/stringlength, get input[%s] success.",
      node->GetNamePtr(), cond_input->GetNamePtr());
    GE_ASSERT_NOTNULL(cond_input);
  }
  // 如果不是data节点, 直接返回
  if (!OpTypeUtils::IsDataNode(cond_input->GetType())) {
    return cond_input;
  }
  // 如果是data节点找根图的节点
  auto parent_input = NodeUtils::GetParentInput(*cond_input);
  return parent_input == nullptr ? cond_input : parent_input;
}

}  // namespace ge