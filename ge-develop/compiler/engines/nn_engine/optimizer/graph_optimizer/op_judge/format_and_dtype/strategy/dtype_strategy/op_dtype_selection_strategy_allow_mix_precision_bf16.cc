/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_dtype_selection_strategy_allow_mix_precision_bf16.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"

namespace fe {
const string KMixPrecisionType = "Bf16";
const ge::DataType KDataType = ge::DT_BF16;
OpDtypeSelectionStrategyAllowMixPrecisionBf16::OpDtypeSelectionStrategyAllowMixPrecisionBf16(
    const std::string &engine_name, FormatDtypeQuerierPtr format_dtype_querier_ptr,
    OpDtypeMixPrecisionMatcherPtr op_dtype_mixed_precision_matcher_ptr, OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr,
    OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr)
    : OpDtypeSelectionStrategyAllowMixPrecisionBase(engine_name, format_dtype_querier_ptr,
                                                    op_dtype_mixed_precision_matcher_ptr, op_dtype_rise_matcher_ptr,
                                                    op_dtype_reduce_matcher_ptr, KMixPrecisionType, KDataType) {}

OpDtypeSelectionStrategyAllowMixPrecisionBf16::~OpDtypeSelectionStrategyAllowMixPrecisionBf16() {}

/* In this mode we will match the dtype fp16 first. If the */
Status OpDtypeSelectionStrategyAllowMixPrecisionBf16::Run(FormatDtypeSelectionBasicInfo &basic_info,
                                                          ForbiddenDtype forbidden_dtype) {
  FE_CHECK_NOTNULL(basic_info.node);
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(cur_op_desc_ptr);
  PrecisionPolicy precision_policy;

  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  if (GetOpPrecisionPolicy(basic_info.node, basic_info.op_kernel_info_ptr, precision_policy) != SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: Failed to get precision policy in mix_bf16.", cur_op_desc_name.c_str(),
            cur_op_desc_type.c_str());
    return FAILED;
  }
  FE_LOGD("Op[name=%s,type=%s]: [mix_bf16]precision policy is %u.", cur_op_desc_name.c_str(), cur_op_desc_type.c_str(),
          precision_policy);
  if (precision_policy == BLACK) {
    /* If the ops is in black list, we must use its original data type */
    return RunForOpInBlackList(basic_info, forbidden_dtype);
  } else if (precision_policy == WHITE) {
    return RunForOpInWhiteList(basic_info, forbidden_dtype);
  } else {
    return RunForOpInGrayList(basic_info, forbidden_dtype);
  }
}
}  // namespace fe
