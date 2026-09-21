/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_DTYPE_OP_DTYPE_REDUCE_MATCHER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_DTYPE_OP_DTYPE_REDUCE_MATCHER_H_
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_matcher_base.h"

namespace fe {
/** @brief match the expected data type and the op_kernel data type,
* the precision can only be reduced in the same type
* for example (fp16->fp16, fp32->fp16, double->fp16) */
class OpDtypeReduceMatcher : public OpDtypeMatcherBase {
 public:
  OpDtypeReduceMatcher();
  ~OpDtypeReduceMatcher() override;
  Status FindSuitableDtype(const vector<ge::DataType> &op_kernel_dtype_vec, const ge::DataType &expected_dtype,
                           vector<uint32_t> &matched_index_vec, const ge::DataType &forbidden_dtype) override;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_DTYPE_OP_DTYPE_REDUCE_MATCHER_H_
