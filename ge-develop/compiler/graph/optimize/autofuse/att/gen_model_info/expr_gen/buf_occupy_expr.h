/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPR_GEN_BUF_OCCUPY_EXPR_H_
#define EXPR_GEN_BUF_OCCUPY_EXPR_H_

#include <unordered_map>
#include "parser/tuning_space.h"
#include "base/base_types.h"

namespace att {
class BufOccupyExpr {
public:
  explicit BufOccupyExpr(const TuningSpacePtr &tuning_space) : tuning_space_(tuning_space) {}
  ~BufOccupyExpr() = default;
  // 按照hardware类型获取需要内存值总和，MAX(tensor_size) * buff_num
  ge::Status GetTotalBufferOccup(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                             std::map<std::string, Expr> &container_exprs);
  // GetTotalGlobalOccup具体实现
  ge::Status GetTotalGlobalOccup(Expr &global_occup_expr);
private:
  // 按照scope汇聚buffer size
  void SummaryBufferOccup(std::unordered_map<HardwareDef, Expr> &current_occup,
                          const HardwareDef scope, Expr &new_occup) const;

  // 共存tensor的size
  ge::Status GetCoTensorSizeExpr(const std::vector<std::vector<TensorPtr>> &co_tensors, Expr &expr,
                                                const Expr &align) const;

  // 获取container的占用size信息
  ge::Status GetOccupInContainer(ContainerPtr &container, Expr &occup_per_tensor, Expr &occup_total) const;

  // GetTotalBufferOccup具体实现
  ge::Status GetBufferOccupInContainer(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                                   std::map<std::string, Expr> &container_exprs);

  TuningSpacePtr tuning_space_;
};
using BufOccupEvaluatorExprPtr = std::shared_ptr<BufOccupyExpr>;
} // namespace att

#endif // EXPR_GEN_BUF_OCCUPY_EXPR_H_