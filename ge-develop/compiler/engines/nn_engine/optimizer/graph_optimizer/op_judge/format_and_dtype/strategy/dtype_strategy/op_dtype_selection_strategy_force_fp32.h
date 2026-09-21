/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_FORCE_FP32_H_
#define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_FORCE_FP32_H_

#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_rise_matcher.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_reduce_matcher.h"
#include "op_dtype_selection_strategy_base.h"

namespace fe {
using OpDtypeRiseMatcherPtr = std::shared_ptr<OpDtypeRiseMatcher>;
using OpDtypeReduceMatcherPtr = std::shared_ptr<OpDtypeReduceMatcher>;

class OpDtypeSelectionStrategyForceFp32 : public OpDtypeSeletionStrategyBase {
 public:
  explicit OpDtypeSelectionStrategyForceFp32(FormatDtypeQuerierPtr format_dtype_querier_ptr,
                                             OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr,
                                             OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr);

  ~OpDtypeSelectionStrategyForceFp32() override;

  /* In this mode we will match the dtype fp32 first. If the */
  Status Run(FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype) override;

  bool CubeNodeDtypeCheck(FormatDtypeSelectionBasicInfo &basic_info) const;
 private:
  OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr_;
  OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr_;
};
}
#endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_FORCE_FP32_H_
