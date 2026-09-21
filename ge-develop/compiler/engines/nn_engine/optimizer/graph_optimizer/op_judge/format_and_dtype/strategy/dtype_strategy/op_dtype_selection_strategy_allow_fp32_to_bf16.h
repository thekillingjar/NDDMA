/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_ALLOW_FP32_TO_BF16_H_
#define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_ALLOW_FP32_TO_BF16_H_

#include "op_dtype_selection_strategy_base.h"
#include "op_dtype_selection_strategy_allow_fp32_to_fp16.h"

namespace fe {
using OpDtypeRiseMatcherPtr = std::shared_ptr<OpDtypeRiseMatcher>;
using OpDtypeReduceMatcherPtr = std::shared_ptr<OpDtypeReduceMatcher>;
using AllowFp32ToFp16Selector = std::unique_ptr<OpDtypeSelectionStrategyAllowFp32ToFp16>;
class OpDtypeSelectionStrategyAllowFp32ToBf16 : public OpDtypeSeletionStrategyBase {
 public:
  explicit OpDtypeSelectionStrategyAllowFp32ToBf16(FormatDtypeQuerierPtr format_dtype_querier_ptr,
      OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr, OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr);
  ~OpDtypeSelectionStrategyAllowFp32ToBf16() override;

  void ProcessReduceMatch(const vector<ge::DataType> &op_kernel_dtype_vec,
                          const ge::DataType &origin_dtype,
                          const SelectionBasicInfo &basic_info,
                          ForbiddenDtype forbidden_dtype,
                          Status &match_origin_dtype_res);
  /* In this mode we will match the dtype bf16 first. If the */
  Status Run(SelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype) override;

 private:
  OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr_;
  OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr_;
};
}  // namespace fe
#endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_ALLOW_FP32_TO_FP16_H_
