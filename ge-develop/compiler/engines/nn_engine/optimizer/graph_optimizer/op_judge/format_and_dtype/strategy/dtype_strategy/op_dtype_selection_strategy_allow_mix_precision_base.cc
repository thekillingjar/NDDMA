/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_dtype_selection_strategy_allow_mix_precision_base.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"

namespace fe {
OpDtypeSelectionStrategyAllowMixPrecisionBase::OpDtypeSelectionStrategyAllowMixPrecisionBase(
    const std::string &engine_name, FormatDtypeQuerierPtr format_dtype_querier_ptr,
    OpDtypeMixPrecisionMatcherPtr op_dtype_mixed_precision_matcher_ptr, OpDtypeRiseMatcherPtr op_dtype_rise_matcher_ptr,
    OpDtypeReduceMatcherPtr op_dtype_reduce_matcher_ptr, const string mix_precision_type, const ge::DataType data_type)
    : OpDtypeSeletionStrategyBase(format_dtype_querier_ptr),
      engine_name_(engine_name),
      op_dtype_mixed_precision_matcher_ptr_(op_dtype_mixed_precision_matcher_ptr),
      op_dtype_rise_matcher_ptr_(op_dtype_rise_matcher_ptr),
      op_dtype_reduce_matcher_ptr_(op_dtype_reduce_matcher_ptr),
      mix_precision_type_(mix_precision_type),
      data_type_(data_type) {}

OpDtypeSelectionStrategyAllowMixPrecisionBase::~OpDtypeSelectionStrategyAllowMixPrecisionBase() {}

bool OpDtypeSelectionStrategyAllowMixPrecisionBase::IsOpFp16Bf16Fp32Cast(const ge::NodePtr &cur_node_ptr,
                                                                         const uint32_t &father_out_anchor_index) {
  ge::OpDescPtr cur_op_desc_ptr = cur_node_ptr->GetOpDesc();
  string op_type = cur_op_desc_ptr->GetType();
  if (op_type == CAST) {
    /* If Cast is in Black list, we need to check whether it's  */
    PrecisionPolicy precision_policy = GRAY;
    Status ret = QueryPrecisionPolicy(cur_node_ptr, precision_policy);
    /* If Cast is in black list, we cannot jump over it. So we return false to
     * consider it as normal Cast. */
    if (ret == SUCCESS && precision_policy != BLACK) {
      auto father_output_desc = cur_op_desc_ptr->GetOutputDescPtr(father_out_anchor_index);
      auto father_input_desc = cur_op_desc_ptr->GetInputDescPtr(0);
      if(father_input_desc == nullptr || father_output_desc == nullptr) {
        return false;
      }
      ge::DataType fahter_out_dtype = father_output_desc->GetDataType();
      ge::DataType fahter_in_dtype = father_input_desc->GetDataType();
      bool check_father_io_dtype = false;
      if (data_type_ == ge::DT_BF16) {
        check_father_io_dtype = (fahter_out_dtype == ge::DT_FLOAT && fahter_in_dtype == ge::DT_BF16) ||
                                (fahter_out_dtype == ge::DT_FLOAT16 && fahter_in_dtype == ge::DT_BF16);
      } else {
        check_father_io_dtype = (fahter_out_dtype == ge::DT_FLOAT && fahter_in_dtype == ge::DT_FLOAT16) ||
                                (fahter_out_dtype == ge::DT_BF16 && fahter_in_dtype == ge::DT_FLOAT16);
      }
      if (check_father_io_dtype) {
        FE_LOGD("Father of %s is %u Cast", cur_op_desc_ptr->GetName().c_str(), precision_policy);
        return true;
      }
    }
    FE_LOGD("Father of %s is BLACK Cast.", cur_op_desc_ptr->GetName().c_str());
  }
  return false;
}

Status OpDtypeSelectionStrategyAllowMixPrecisionBase::QueryPrecisionPolicy(const ge::NodePtr &node_ptr,
                                                                           PrecisionPolicy &precision_policy) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto op_kernel_info_ptr =
      OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpType(EN_IMPL_HW_TBE, op_desc_ptr->GetType());
  if (op_kernel_info_ptr == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][DtypeJdg][QryPrecisPolicy] op %s is not found in tbe built-in store.",
                    op_desc_ptr->GetType().c_str());
    return FAILED;
  }

  return GetOpPrecisionPolicy(node_ptr, op_kernel_info_ptr, precision_policy);
}

Status OpDtypeSelectionStrategyAllowMixPrecisionBase::GetOpPrecisionPolicy(const ge::NodePtr &node,
                                                                           const OpKernelInfoPtr &op_kernel_info_ptr,
                                                                           PrecisionPolicy &precision_policy) const {
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  FE_CHECK_NOTNULL(node);
  precision_policy = op_kernel_info_ptr->GetPrecisionPolicy();
  precision_policy = Configuration::Instance(engine_name_).GetPrecisionPolicy(op_kernel_info_ptr->GetOpType(),
                                                                              precision_policy);
  return SUCCESS;
}

Status OpDtypeSelectionStrategyAllowMixPrecisionBase::RunForOpInWhiteList(FormatDtypeSelectionBasicInfo &basic_info,
                                                                          ForbiddenDtype forbidden_dtype) {
  FE_CHECK_NOTNULL(basic_info.node);
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(cur_op_desc_ptr);
  ge::DataType origin_dtype = basic_info.cur_tensor_desc->GetDataType();
  bool dtype_float_flag =
      (origin_dtype == ge::DT_FLOAT || origin_dtype == ge::DT_FLOAT16 || origin_dtype == ge::DT_BF16);
  /* If the op is in white list but its original data type is not
   * fp32 or fp16, we will use allow fp32_to_fp16 mode to select data type. */
  if (!dtype_float_flag) {
    FE_LOGD("The data type of tensor %u of op %s is not fp32 bf16 or fp16!", basic_info.index,
            cur_op_desc_ptr->GetName().c_str());
    FE_LOGD("Try to match original data type %u.", origin_dtype);
    DefaultSelector default_select_mode(
        new (std::nothrow) OpDtypeSelectionStrategyDefaultMode(format_dtype_querier_ptr_,
                                                               op_dtype_rise_matcher_ptr_));
    if (default_select_mode == nullptr) {
      return FAILED;
    }
    return default_select_mode->Run(basic_info, ForbiddenDtype::FORBIDDEN_NONE);
  } else {
    /* Only pick fp16 or bf16 as its dtype, if it does not support bf16/fp16, we will
     * pick the higher precision version. */
    vector<ge::DataType> input_or_output_dtype_vec;
    if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                       basic_info.node,
                                                       input_or_output_dtype_vec) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][DtypeJdg][MixedPrecision%s][Op %s type %s] Failed to obtain supported data_types.",
                      mix_precision_type_.c_str(), cur_op_desc_ptr->GetName().c_str(),
                      cur_op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    Status ret = op_dtype_mixed_precision_matcher_ptr_->Match(input_or_output_dtype_vec, origin_dtype,
                                                              basic_info.matched_index_vec, forbidden_dtype);
    if (ret != SUCCESS) {
      /* We allow the node in white list using fp32, so here we just report
       * a warning log and return success. */
      FE_LOGW("[GraphOpt][DtypeJdg][MixedPrecision%s][Op %s type %s] is in white list but it doesn't support %s!",
              mix_precision_type_.c_str(), cur_op_desc_ptr->GetName().c_str(), cur_op_desc_ptr->GetType().c_str(),
              mix_precision_type_.c_str());
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status OpDtypeSelectionStrategyAllowMixPrecisionBase::RunForOpInBlackList(FormatDtypeSelectionBasicInfo &basic_info,
                                                                          ForbiddenDtype forbidden_dtype) const {
  DefaultSelector default_select_mode(
      new (std::nothrow) OpDtypeSelectionStrategyDefaultMode(format_dtype_querier_ptr_,
                                                             op_dtype_rise_matcher_ptr_));
  if (default_select_mode == nullptr) {
    return FAILED;
  }
  Status ret = default_select_mode->Run(basic_info, forbidden_dtype);
  string node_name = basic_info.node->GetName();
  FE_CHECK_NOTNULL(basic_info.cur_tensor_desc);
  auto dtype = basic_info.cur_tensor_desc->GetDataType();
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][DtypeJdg][RunBlackListOp] Op %s is in blacklist but doesn't support it's original dtype %s",
        node_name.c_str(), ge::TypeUtils::DataTypeToSerialString(dtype).c_str());
    if (dtype == ge::DT_FLOAT || dtype == data_type_) {
      REPORT_FE_ERROR("[GraphOpt][DtypeJdg][RunBlackListOp] Op %s should not be configured as blacklist op",
                      node_name.c_str());
    }
  }
  return ret;
}

void OpDtypeSelectionStrategyAllowMixPrecisionBase::MatchForGray(const string &cur_op_desc_type,
    const string &cur_op_desc_name, const vector<ge::DataType> &op_kernel_dtype_vec,
    ge::DataType father_output_dtype, const FormatDtypeSelectionBasicInfo& basic_info,
    ForbiddenDtype forbidden_dtype) {
  FE_LOGD("Op[name=%s,type=%s]: match father dtype, the expected dtype is [%u].", cur_op_desc_name.c_str(),
          cur_op_desc_type.c_str(), father_output_dtype);
  Status match_father_dtype_res = op_dtype_rise_matcher_ptr_->Match(op_kernel_dtype_vec, father_output_dtype,
                                                                    basic_info.matched_index_vec, forbidden_dtype);
  if (match_father_dtype_res != SUCCESS) {
    FE_LOGD("Precision loss is allowed, try to match low precision dtype.");
    match_father_dtype_res = op_dtype_reduce_matcher_ptr_->Match(op_kernel_dtype_vec, father_output_dtype,
                                                                 basic_info.matched_index_vec, forbidden_dtype);
    // if father is fp32 ,but op is not supported. if father is bf16, but op is also unsuppor float or bf16;
    // so we have no choosen, need todo nothing
  }
  if (match_father_dtype_res == SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: match the father dtype success, some matched dtypes in op kernel have been found.",
            cur_op_desc_name.c_str(), cur_op_desc_type.c_str());
    FE_LOGD("The size of dtype is [%zu], the size of matched index is [%zu].", op_kernel_dtype_vec.size(),
            basic_info.matched_index_vec.size());
  } else {
    FE_LOGD("Op[name=%s,type=%s]: No match found for dtype %u, matchedIndexVec remains unchanged.", cur_op_desc_name.c_str(),
            cur_op_desc_type.c_str(), father_output_dtype);
  }
}

Status OpDtypeSelectionStrategyAllowMixPrecisionBase::RunForOpInGrayList(FormatDtypeSelectionBasicInfo &basic_info,
                                                                         ForbiddenDtype forbidden_dtype) {
  auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(cur_op_desc_ptr);
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  ge::InDataAnchorPtr in_data_anchor;
  bool has_no_father = false;
  CheckHasNoFather(basic_info.is_input, static_cast<int32_t>(basic_info.index), basic_info.node, in_data_anchor,
                   has_no_father);

  /* 1. If the node is Gray list does not have predecessors, we just match the
   * dtype with its original dtype. */
  if (has_no_father) {
    FE_LOGD("Node[%s, %s]: Operation does not have a parent node on input[%u] by mix[%s]. Matched with its original data type.",
            cur_op_desc_name.c_str(), cur_op_desc_type.c_str(), basic_info.index, mix_precision_type_.c_str());
    if (data_type_ == ge::DT_BF16) {
      AllowFp32ToBf16Selector allow_fp32_to_bf16_selector(new (std::nothrow) OpDtypeSelectionStrategyAllowFp32ToBf16(
          format_dtype_querier_ptr_, op_dtype_rise_matcher_ptr_, op_dtype_reduce_matcher_ptr_));
      FE_CHECK_NOTNULL(allow_fp32_to_bf16_selector);
      return allow_fp32_to_bf16_selector->Run(basic_info, forbidden_dtype);
    } else {
      AllowFp32ToFp16Selector allow_fp32_to_fp16_selector(new (std::nothrow) OpDtypeSelectionStrategyAllowFp32ToFp16(
          format_dtype_querier_ptr_, op_dtype_rise_matcher_ptr_, op_dtype_reduce_matcher_ptr_));
      FE_CHECK_NOTNULL(allow_fp32_to_fp16_selector);
      return allow_fp32_to_fp16_selector->Run(basic_info, forbidden_dtype);
    }
  }
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(peer_out_anchor);
  uint32_t father_out_anchor_index = static_cast<uint32_t>(peer_out_anchor->GetIdx());
  ge::OpDescPtr father_op_desc = peer_out_anchor->GetOwnerNode()->GetOpDesc();
  ge::NodePtr father_node = peer_out_anchor->GetOwnerNode();
  FE_CHECK_NOTNULL(father_node);
  FE_LOGD("Op[name=%s,type=%s]:match format and dtype for the input %u between father Op[name=%s,type=%s] and this op.",
          cur_op_desc_name.c_str(), cur_op_desc_type.c_str(), basic_info.index, father_op_desc->GetName().c_str(),
          father_op_desc->GetType().c_str());

  /* 1.1 If the father node is Cast, and the data type of output of Cast is fp32,
   * we will try to skip this cast and match the dtype in front of cast */
  if (IsOpFp16Bf16Fp32Cast(father_node, father_out_anchor_index)) {
    in_data_anchor = father_node->GetInDataAnchor(0);
    FE_CHECK_NOTNULL(in_data_anchor);
    auto father_out_anchor = in_data_anchor->GetPeerOutAnchor();
    FE_CHECK_NOTNULL(father_out_anchor);
    father_node = father_out_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(father_node);
    father_op_desc = father_node->GetOpDesc();
    FE_CHECK_NOTNULL(father_op_desc);
    father_out_anchor_index = father_out_anchor->GetIdx();
  }

  /* 2. Match all supported data type with father's output data type. */
  auto father_output_desc = father_op_desc->GetOutputDescPtr(father_out_anchor_index);
  FE_CHECK_NOTNULL(father_output_desc);
  ge::DataType father_output_dtype = father_output_desc->GetDataType();
  vector<ge::DataType> op_kernel_dtype_vec;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                     basic_info.node, op_kernel_dtype_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][DtypeJdg][RunGrayListOp] Failed to get the support data_types, return FAILED.");
    return FAILED;
  }
  MatchForGray(cur_op_desc_type, cur_op_desc_name, op_kernel_dtype_vec, father_output_dtype, basic_info,
               forbidden_dtype);
  return SUCCESS;
}
}  // namespace fe
