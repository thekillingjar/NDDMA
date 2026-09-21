/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_format_selection_strategy_default_mode.h"

namespace fe {
OpFormatSelectionStrategyDefaultMode::OpFormatSelectionStrategyDefaultMode(
    FormatDtypeQuerierPtr format_dtype_querier_ptr)
    : format_dtype_querier_ptr_(format_dtype_querier_ptr) {}

OpFormatSelectionStrategyDefaultMode::~OpFormatSelectionStrategyDefaultMode() {}

Status OpFormatSelectionStrategyDefaultMode::Initialize() {
  FE_MAKE_SHARED(format_matcher_ = std::make_shared<OpFormatMatcher>(), return FAILED);
  return SUCCESS;
}

Status OpFormatSelectionStrategyDefaultMode::Run(FormatDtypeSelectionBasicInfo &basic_info) {
  FE_CHECK_NOTNULL(basic_info.node);
  ge::OpDescPtr cur_op_desc_ptr = basic_info.node->GetOpDesc();
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  if (!basic_info.cur_tensor_desc) {
    return FAILED;
  }
  ge::Format cur_origin_format = GetCurOpOriginFormat(*(basic_info.cur_tensor_desc));
  FE_LOGD("Op[name=%s,type=%s]: match the origin format and dtype for tensor %d %u.", cur_op_desc_name.c_str(),
          cur_op_desc_type.c_str(), basic_info.is_input, basic_info.index);
  // 1. Get all supported format
  vector<ge::Format> input_or_output_format;
  if (format_dtype_querier_ptr_->GetSupportFormats(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                   basic_info.node, input_or_output_format) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][FmtJdg][Run] Failed to retrieve supported formats, status: FAILED.");
    return FAILED;
  }

  format_matcher_->FilterFormatIndex(basic_info.node, basic_info.op_kernel_info_ptr,
                                     input_or_output_format, basic_info.matched_index_vec);
  FE_LOGD("Op[name=%s,type=%s]: match origin format, the expected format is %s.", cur_op_desc_name.c_str(),
          cur_op_desc_type.c_str(), ge::TypeUtils::FormatToSerialString(cur_origin_format).c_str());
  Status match_origin_format_res = format_matcher_->Match(input_or_output_format, cur_origin_format, cur_origin_format,
                                                          basic_info.matched_index_vec);
  if (match_origin_format_res == SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: match the origin format success, some matched formats in op kernel have been found.",
            cur_op_desc_name.c_str(), cur_op_desc_type.c_str());
  } else {
    FE_LOGD("Op[name=%s,type=%s]: match origin format unsuccessful.", cur_op_desc_name.c_str(), cur_op_desc_type.c_str());
  }
  return SUCCESS;
}
}  // namespace fe
