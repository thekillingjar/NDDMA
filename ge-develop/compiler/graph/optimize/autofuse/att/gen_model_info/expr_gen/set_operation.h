/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPR_GEN_SET_OPERATION_H_
#define EXPR_GEN_SET_OPERATION_H_

#include <vector>
#include "base/base_types.h"

namespace att {
// 表示取值范围的集合
struct DimRange {
  Expr upper_bound;
  Expr lower_bound;
  bool operator== (const DimRange &v) const {
    return (this->upper_bound == v.upper_bound) &&
           (this->lower_bound == v.lower_bound);
  }
};
using Coordinates = std::vector<DimRange>;
using TensorRange = std::vector<Coordinates>;

class SetOperation {
public:
  // 计算dim差集
  static std::vector<DimRange> Diff(DimRange &range1, DimRange &range2);

  // 计算tensor差集
  static TensorRange Diff(TensorRange &range1, TensorRange &range2);

  // 计算dim交集
  static DimRange Intersection(DimRange &range1, DimRange &range2);

  // 计算tensor交集
  static TensorRange Intersection(TensorRange &range1, TensorRange &range2);

  // 计算并集
  static void ProductImplement(std::vector<std::vector<uint32_t>> &seq,
                               std::vector<std::vector<uint32_t>> &res, uint32_t layer,
                               std::vector<uint32_t> &tmp);

  // 集合范围计算
  static Expr SetComputation(TensorRange &range);
};
} // namespace att

#endif // EXPR_GEN_SET_OPERATION_H_