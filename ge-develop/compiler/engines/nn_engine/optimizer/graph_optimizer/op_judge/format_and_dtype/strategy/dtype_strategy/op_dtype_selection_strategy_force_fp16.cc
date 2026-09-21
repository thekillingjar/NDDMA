/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_dtype_selection_strategy_force_fp16.h"

namespace fe {
OpDtypeSelectionStrategyForceFp16::OpDtypeSelectionStrategyForceFp16(
    FormatDtypeQuerierPtr format_dtype_querier_ptr, OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr)
    : OpDtypeSeletionStrategyBase(format_dtype_querier_ptr),
      op_dtype_rise_matcher_ptr_(op_dtype_rise_matcher_ptr) {}

OpDtypeSelectionStrategyForceFp16::~OpDtypeSelectionStrategyForceFp16() {}

/* In this mode we will match the dtype fp16 first. If the */
Status OpDtypeSelectionStrategyForceFp16::Run(FormatDtypeSelectionBasicInfo& basic_info,
                                              ForbiddenDtype forbidden_dtype) {
  FE_CHECK_NOTNULL(basic_info.node);
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(cur_op_desc_ptr);
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  FE_LOGD("Op %s: matching dtype for tensor %u in forced fp16 mode.", cur_op_desc_name.c_str(), basic_info.index);

  // 2. Match fp16 if original Dtype is fp32
  FE_CHECK_NOTNULL(basic_info.cur_tensor_desc);
  ge::DataType origin_dtype = basic_info.cur_tensor_desc->GetDataType();
  if (origin_dtype == ge::DT_FLOAT || origin_dtype == ge::DT_BF16) {
    origin_dtype = ge::DT_FLOAT16;
  }
  vector<ge::DataType> input_or_output_dtype_vec;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                     basic_info.node, input_or_output_dtype_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][DtypeJdg][ForceFp16] Failed to get the support data_types.");
    return FAILED;
  }

  FE_LOGD("Op %s: matches the dtype in force_fp16 mode; the expected dtype is %u.", cur_op_desc_name.c_str(),
          origin_dtype);
  Status match_origin_dtype_res =
      op_dtype_rise_matcher_ptr_->Match(input_or_output_dtype_vec, origin_dtype,
                                        basic_info.matched_index_vec, forbidden_dtype);
  /* if force fp16 match failed, we will use the data type that it supports and
   * will not return FAILED. */
  if (match_origin_dtype_res != SUCCESS) {
    FE_LOGD("Cannot find any matched data type for tensor %u %s of dtype %u.", basic_info.index,
            cur_op_desc_name.c_str(), origin_dtype);
  }
  return SUCCESS;
}
}
