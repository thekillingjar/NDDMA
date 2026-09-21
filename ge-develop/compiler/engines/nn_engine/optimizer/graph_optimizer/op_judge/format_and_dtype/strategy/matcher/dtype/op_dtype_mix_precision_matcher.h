/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_DTYPE_OP_DTYPE_MIX_PRECISION_MATCHER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_DTYPE_OP_DTYPE_MIX_PRECISION_MATCHER_H_

#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_matcher_base.h"

namespace fe {
/** @brief match the expected data type and the op_kernel data type
*        when is_auto_mix_precision switch is on. */
class OpDtypeMixPrecisionMatcher : public OpDtypeMatcherBase {
 public:
  OpDtypeMixPrecisionMatcher();
  virtual ~OpDtypeMixPrecisionMatcher() override;
  Status FindSuitableDtype(const vector<ge::DataType> &op_kernel_dtype_vec, const ge::DataType &expected_dtype,
                           vector<uint32_t> &matched_index_vec, const ge::DataType &forbidden_dtype) override;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_DTYPE_OP_DTYPE_MIX_PRECISION_MATCHER_H_
