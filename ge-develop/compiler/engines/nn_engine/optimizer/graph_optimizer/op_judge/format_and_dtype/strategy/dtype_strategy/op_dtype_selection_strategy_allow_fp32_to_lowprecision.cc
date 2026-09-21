/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_dtype_selection_strategy_allow_fp32_to_lowprecision.h"

namespace fe {
OpDtypeSelectionStrategyAllowFp32ToLowPrecision::OpDtypeSelectionStrategyAllowFp32ToLowPrecision(
    FormatDtypeQuerierPtr format_dtype_querier_ptr, OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr,
    OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr)
    : OpDtypeSeletionStrategyBase(format_dtype_querier_ptr),
      op_dtype_rise_matcher_ptr_(op_dtype_rise_matcher_ptr),
      op_dtype_reduce_matcher_ptr_(op_dtype_reduce_matcher_ptr) {}

OpDtypeSelectionStrategyAllowFp32ToLowPrecision::~OpDtypeSelectionStrategyAllowFp32ToLowPrecision() {}

Status OpDtypeSelectionStrategyAllowFp32ToLowPrecision::Run(SelectionBasicInfo &basic_info,
                                                            ForbiddenDtype forbidden_dtype) {
  AllowFp32ToFp16Selector allow_fp32_to_lowerprecision_mode(new(std::nothrow) OpDtypeSelectionStrategyAllowFp32ToFp16(
                          format_dtype_querier_ptr_, op_dtype_rise_matcher_ptr_, op_dtype_reduce_matcher_ptr_));
  if (allow_fp32_to_lowerprecision_mode == nullptr) {
      return FAILED;
  }
  return allow_fp32_to_lowerprecision_mode->Run(basic_info, forbidden_dtype);
}
}  // namespace fe
