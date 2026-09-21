/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_dtype_selection_strategy_allow_fp32_to_fp16.h"
#include "common/configuration.h"

namespace fe {
OpDtypeSelectionStrategyAllowFp32ToFp16::OpDtypeSelectionStrategyAllowFp32ToFp16(
    FormatDtypeQuerierPtr format_dtype_querier_ptr, OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr,
    OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr)
    : OpDtypeSeletionStrategyBase(format_dtype_querier_ptr),
      op_dtype_rise_matcher_ptr_(op_dtype_rise_matcher_ptr),
      op_dtype_reduce_matcher_ptr_(op_dtype_reduce_matcher_ptr) {}

OpDtypeSelectionStrategyAllowFp32ToFp16::~OpDtypeSelectionStrategyAllowFp32ToFp16() {}

void OpDtypeSelectionStrategyAllowFp32ToFp16::ProcessReduceMatch(const vector<ge::DataType> &op_kernel_dtype_vec,
                                                                 const ge::DataType &origin_dtype,
                                                                 const SelectionBasicInfo &basic_info,
                                                                 ForbiddenDtype forbidden_dtype,
                                                                 Status &match_origin_dtype_res) {
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();

  if (origin_dtype == ge::DT_FLOAT) {
    match_origin_dtype_res =
        op_dtype_reduce_matcher_ptr_->Match(op_kernel_dtype_vec, origin_dtype, basic_info.matched_index_vec,
                                            forbidden_dtype);
    if (match_origin_dtype_res == SUCCESS) {
      FE_LOGD("Op[name=%s,type=%s]: set the attr %s in AllowFp32ToFp16 mode.", cur_op_desc_name.c_str(),
              cur_op_desc_type.c_str(), kForceFp32ToFp16.c_str());
      (void)ge::AttrUtils::SetBool(cur_op_desc_ptr, kForceFp32ToFp16, true);
    }
  } else if (origin_dtype == ge::DT_BF16) {
    match_origin_dtype_res = op_dtype_reduce_matcher_ptr_->Match(op_kernel_dtype_vec, origin_dtype,
                                                                 basic_info.matched_index_vec, forbidden_dtype);
  }
  return;
}

Status OpDtypeSelectionStrategyAllowFp32ToFp16::Run(SelectionBasicInfo &basic_info, ForbiddenDtype forbidden_dtype) {
  FE_CHECK_NOTNULL(basic_info.node);
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(cur_op_desc_ptr);
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  FE_LOGD("Op[name=%s,type=%s]: match dtype for tensor %u in AllowFp32ToFp16, forbidden_type is %u.",
          cur_op_desc_name.c_str(), cur_op_desc_type.c_str(), basic_info.index, static_cast<uint32_t>(forbidden_dtype));

  FE_CHECK_NOTNULL(basic_info.cur_tensor_desc);
  ge::DataType origin_dtype = basic_info.cur_tensor_desc->GetDataType();
  vector<ge::DataType> input_or_output_dtype_vec;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                     basic_info.node, input_or_output_dtype_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][DtypeJdg][AllowFp32ToFp16] Failed to get the supported data_types for Op %s with type %s.",
                    cur_op_desc_name.c_str(), cur_op_desc_type.c_str());
    return FAILED;
  }
  std::string op_type = cur_op_desc_ptr->GetType();
  std::unordered_set<string> fp16_op_type_list = Configuration::Instance(AI_CORE_NAME).GetFp16OpTypeList();
  int64_t keep_dtype = 0;
  (void)ge::AttrUtils::GetInt(cur_op_desc_ptr, KEEP_DTYPE, keep_dtype);
  if ((origin_dtype == ge::DT_FLOAT) && (fp16_op_type_list.count(op_type) != 0) && keep_dtype == 0) {
    origin_dtype = ge::DT_FLOAT16;
  }
  FE_LOGD("Op[name=%s,type=%s]: match the origin dtype, the expected dtype is %u.", cur_op_desc_name.c_str(),
          cur_op_desc_type.c_str(), origin_dtype);
  // 1.match datatype with origin datatype using increasing mode, in this mode
  // we will first ensure the precision will not decrease.
  Status match_origin_dtype_res = op_dtype_rise_matcher_ptr_->Match(input_or_output_dtype_vec, origin_dtype,
                                                                    basic_info.matched_index_vec, forbidden_dtype);
  if (match_origin_dtype_res != SUCCESS) {
    // 1.match datatype with origin datatype using reducing mode, in this mode
    // we will allow the precision reduce from fp32 to fp16 or bf16 to fp16
    FE_LOGD("Precision loss is allowed, try to match low precision dtype.");
    ProcessReduceMatch(input_or_output_dtype_vec, origin_dtype, basic_info, forbidden_dtype,
                       match_origin_dtype_res);
  }

  if (match_origin_dtype_res == SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: match the origin dtype success, some matched dtype in op kernel have been found.",
            cur_op_desc_name.c_str(), cur_op_desc_type.c_str());
    FE_LOGD("The size of input_or_output_dtype_vec is %zu, the size of matchedIndexVec is %zu.",
            input_or_output_dtype_vec.size(), basic_info.matched_index_vec.size());
  } else {
    FE_LOGD("Op[name=%s,type=%s]: no matched the origin dtype, matchedIndexVec remain the same.",
            cur_op_desc_name.c_str(), cur_op_desc_type.c_str());
  }
  return SUCCESS;
}
}  // namespace fe
