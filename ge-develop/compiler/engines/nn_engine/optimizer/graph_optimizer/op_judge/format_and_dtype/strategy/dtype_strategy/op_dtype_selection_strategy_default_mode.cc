/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_dtype_selection_strategy_default_mode.h"
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "common/graph/fe_graph_utils.h"

namespace fe {
OpDtypeSelectionStrategyDefaultMode::OpDtypeSelectionStrategyDefaultMode(
    FormatDtypeQuerierPtr format_dtype_querier_ptr, OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr)
    : OpDtypeSeletionStrategyBase(format_dtype_querier_ptr),
      op_dtype_rise_matcher_ptr_(op_dtype_rise_matcher_ptr) {}

OpDtypeSelectionStrategyDefaultMode::~OpDtypeSelectionStrategyDefaultMode() {}

Status OpDtypeSelectionStrategyDefaultMode::Run(FormatDtypeSelectionBasicInfo& basic_info,
                                                ForbiddenDtype forbidden_dtype) {
  FE_CHECK_NOTNULL(basic_info.node);
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(cur_op_desc_ptr);
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  FE_LOGD("Op %s: match dtype for tensor %u in default mode.", cur_op_desc_name.c_str(), basic_info.index);

  // 2. Match Original Dtype
  ge::DataType origin_dtype = basic_info.cur_tensor_desc->GetDataType();
  vector<ge::DataType> input_or_output_dtype_vec;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                     basic_info.node, input_or_output_dtype_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][DtypeJdg][DefaultMode][Op %s type %s] Failed to get the support data_types.",
                    cur_op_desc_name.c_str(), cur_op_desc_type.c_str());
    return FAILED;
  }
  fe::PrecisionMode precision_mode = fe::PrecisionMode::ENUM_UNDEFINED;
  FeGraphUtils::GetPrecisionModeFromGraph(*(basic_info.node->GetOwnerComputeGraph()), precision_mode);
  FE_LOGD("Op %s: match the origin dtype, the expected dtype is %u, precision mode is %d.",
          cur_op_desc_name.c_str(), origin_dtype, precision_mode);
  Status match_origin_dtype_res = op_dtype_rise_matcher_ptr_->Match(input_or_output_dtype_vec, origin_dtype,
                                                                    basic_info.matched_index_vec, forbidden_dtype);
  FE_LOGD("After matching, size of vec is %zu.", basic_info.matched_index_vec.size());
  if (match_origin_dtype_res != SUCCESS) {
    FE_LOGD("[GraphOpt][DtypeJdg][DefaultMode][%s, %s] Precision loss is not allowed. Origin data type is [%s].",
            cur_op_desc_name.c_str(), cur_op_desc_type.c_str(),
            ge::TypeUtils::DataTypeToSerialString(origin_dtype).c_str());
    return SUCCESS;
  }
  return SUCCESS;
}
}
