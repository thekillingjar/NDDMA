/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_ALLOW_MIX_PRECISION_BASE_H_
#define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_ALLOW_MIX_PRECISION_BASE_H_

#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_mix_precision_matcher.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_reduce_matcher.h"
#include "op_dtype_selection_strategy_allow_fp32_to_bf16.h"
#include "op_dtype_selection_strategy_base.h"
#include "op_dtype_selection_strategy_default_mode.h"
#include "op_dtype_selection_strategy_force_fp16.h"
namespace fe {
using DefaultSelector = std::unique_ptr<OpDtypeSelectionStrategyDefaultMode>;
using AllowFp32ToBf16Selector = std::unique_ptr<OpDtypeSelectionStrategyAllowFp32ToBf16>;
using AllowFp32ToFp16Selector = std::unique_ptr<OpDtypeSelectionStrategyAllowFp32ToFp16>;

using OpDtypeMixPrecisionMatcherPtr = std::shared_ptr<OpDtypeMixPrecisionMatcher>;
using OpDtypeRiseMatcherPtr = std::shared_ptr<OpDtypeRiseMatcher>;
using OpDtypeReduceMatcherPtr = std::shared_ptr<OpDtypeReduceMatcher>;
class OpDtypeSelectionStrategyAllowMixPrecisionBase : public OpDtypeSeletionStrategyBase {
 public:
  OpDtypeSelectionStrategyAllowMixPrecisionBase(
      const std::string& engine_name,
      FormatDtypeQuerierPtr format_dtype_querier_ptr,
      OpDtypeMixPrecisionMatcherPtr op_dtype_mixed_precision_matcher_ptr,
      OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr,
      OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr,
      const string mix_precision_type,
      const ge::DataType data_type);

  ~OpDtypeSelectionStrategyAllowMixPrecisionBase() override;

  Status RunForOpInWhiteList(FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype);

  Status RunForOpInGrayList(FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype);

  /* Black list op must use their original data types. */
  Status RunForOpInBlackList(FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype) const;
  Status GetOpPrecisionPolicy(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                              PrecisionPolicy &precision_policy) const;
 private:
  void MatchForGray(const string &cur_op_desc_type, const string &cur_op_desc_name,
       const vector<ge::DataType> &op_kernel_dtype_vec, ge::DataType fahter_output_dtype,
       const FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype);
  bool IsOpFp16Bf16Fp32Cast(const ge::NodePtr &cur_node_ptr, const uint32_t& father_out_anchor_index);
  Status QueryPrecisionPolicy(const ge::NodePtr &node_ptr, PrecisionPolicy &precision_policy);

  std::string engine_name_;
  OpDtypeMixPrecisionMatcherPtr op_dtype_mixed_precision_matcher_ptr_;
  OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr_;
  OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr_;
  const string mix_precision_type_;
  const ge::DataType data_type_;
};
}  // namespace fe
#endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_ALLOW_MIX_PRECISION_H_
