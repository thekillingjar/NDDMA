/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_dtype_selection_strategy_force_fp32.h"
#include "common/configuration.h"
#include "common/fe_inner_attr_define.h"

namespace fe {
OpDtypeSelectionStrategyForceFp32::OpDtypeSelectionStrategyForceFp32(
    FormatDtypeQuerierPtr format_dtype_querier_ptr, OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr,
    OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr)
    : OpDtypeSeletionStrategyBase(format_dtype_querier_ptr),
      op_dtype_rise_matcher_ptr_(op_dtype_rise_matcher_ptr),
      op_dtype_reduce_matcher_ptr_(op_dtype_reduce_matcher_ptr) {}

OpDtypeSelectionStrategyForceFp32::~OpDtypeSelectionStrategyForceFp32() {}

/* In this mode we will match the dtype fp32 first. If the */
Status OpDtypeSelectionStrategyForceFp32::Run(FormatDtypeSelectionBasicInfo &basic_info,
                                              ForbiddenDtype forbidden_dtype) {
  FE_CHECK_NOTNULL(basic_info.node);
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(cur_op_desc_ptr);
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  FE_LOGD("Op %s: match dtype for tensor %u in force fp32 mode.", cur_op_desc_name.c_str(), basic_info.index);

  // 2. Match fp32 if original Dtype is fp16 or bf16
  // If current node is Fp16OpTypeList, we modify dtype as fp16 in,fp32 out.
  FE_CHECK_NOTNULL(basic_info.cur_tensor_desc);
  ge::DataType origin_dtype = basic_info.cur_tensor_desc->GetDataType();
  FE_CHECK_NOTNULL(basic_info.tensor_kernel_info_ptr);
  bool is_first_input = basic_info.tensor_kernel_info_ptr->GetIsInput() && (basic_info.index == 0);
  std::unordered_set<string> fp16_op_type_list = Configuration::Instance(AI_CORE_NAME).GetFp16OpTypeList();
  bool is_fp16_op_type = fp16_op_type_list.count(cur_op_desc_type) != 0;
  bool expected_to_fp16 = (origin_dtype == ge::DT_FLOAT) || (origin_dtype == ge::DT_BF16) ||
                          (origin_dtype == ge::DT_FLOAT16);
  bool cube_node_input_flag = is_first_input && is_fp16_op_type && expected_to_fp16;
  if (cube_node_input_flag && CubeNodeDtypeCheck(basic_info)) {
    origin_dtype = ge::DT_FLOAT16;
  } else {
    if (origin_dtype == ge::DT_FLOAT16 || origin_dtype == ge::DT_BF16) {
      origin_dtype = ge::DT_FLOAT;
    }
  }
  vector<ge::DataType> input_or_output_dtype_vec;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                     basic_info.node, input_or_output_dtype_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][DtypeJdg][ForceFp32] Failed to obtain supported data_types.");
    return FAILED;
  }

  FE_LOGD("Op %s: match the dtype in force_fp32 mode, the expected dtype is %u.", cur_op_desc_name.c_str(),
          origin_dtype);
  Status match_origin_dtype_res = op_dtype_rise_matcher_ptr_->Match(input_or_output_dtype_vec, origin_dtype,
                                                                    basic_info.matched_index_vec, forbidden_dtype);
  /* if force fp32 match failed, we will use the data type that it supports and
   * will not return FAILED. */
  if (match_origin_dtype_res != SUCCESS) {
    vector<uint32_t> bak_matched_index_vec = basic_info.matched_index_vec;
    vector<ge::DataType> classify_types;
    FE_LOGD("Allowing precision loss, attempt to match lower precision data type.");
    match_origin_dtype_res = op_dtype_reduce_matcher_ptr_->Match(input_or_output_dtype_vec, origin_dtype,
                                                                 basic_info.matched_index_vec, forbidden_dtype);
    // force_fp32 reducing, we should use fp16
  }
  if (match_origin_dtype_res != SUCCESS) {
    FE_LOGD("Cannot find any matched data type for tensor %u %s of dtype %u.", basic_info.index,
            cur_op_desc_name.c_str(), origin_dtype);
  }
  return SUCCESS;
}

/* check whether cube node op_kernel_info support fp16 in and fp32 out. */
bool OpDtypeSelectionStrategyForceFp32::CubeNodeDtypeCheck(FormatDtypeSelectionBasicInfo &basic_info) const {
  ge::OpDescPtr op_desc_ptr = basic_info.node->GetOpDesc();
  if (op_desc_ptr->HasAttr(kAttrSupported_16In_32Out)) {
    bool supported_16in_32out = false;
    (void) ge::AttrUtils::GetBool(op_desc_ptr, kAttrSupported_16In_32Out, supported_16in_32out);
    return supported_16in_32out;
  }
  FE_LOGW("Node[type=%s,name=%s]: not found attr supported_16in_32out.",
          op_desc_ptr->GetType().c_str(), op_desc_ptr->GetName().c_str());
  return false;
}
}  // namespace fe
