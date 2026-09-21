/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/update_desc/op_format_dtype_update_desc_base.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/unknown_shape_util.h"
#include "common/util/op_info_util.h"
#include "param_calculate/tensorsize_calculator.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "expand_dimension.h"
#include "common/platform_utils.h"
#include "common/range_format_transfer/transfer_range_according_to_format.h"

namespace fe {
namespace {
Status CheckMatchedIndexValid(const std::vector<ge::Format>& op_kernel_format_vec,
                              const std::vector<ge::DataType>& op_kernel_data_type_vec, const uint32_t& matched_index,
                              const ge::OpDescPtr& op_desc_ptr, const uint32_t& index, const bool& is_input) {
  FE_CHECK_NOTNULL(op_desc_ptr);
  uint32_t op_kernel_format_vec_size = op_kernel_format_vec.size();
  uint32_t op_kernel_data_type_vec_size = op_kernel_data_type_vec.size();
  if (op_kernel_format_vec.empty() || matched_index >= op_kernel_format_vec_size) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][ChkMtcIdxValid] Op[%s,type=%s]: update the %s [%u], "
                    "the matched index [%u] is larger than or equals to the size of op_kernel_format_vec [%u].",
                    op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), IS_INPUT_TO_STRING(is_input), index,
                    matched_index, op_kernel_format_vec_size);
    return FAILED;
  }

  if (op_kernel_data_type_vec.empty() || matched_index >= op_kernel_data_type_vec_size) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][ChkMtcIdxValid] Op[%s,type=%s]: update the %s [%u], the matched "
                    "index [%u] is larger than or equals to the size of opKernelDataTypeVecSize [%u].",
                    op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), IS_INPUT_TO_STRING(is_input), index,
                    matched_index, op_kernel_data_type_vec_size);
    return FAILED;
  }
  return SUCCESS;
}
} // namespace

OpFormatDtypeUpdateDescBase::OpFormatDtypeUpdateDescBase(const FormatDtypeQuerierPtr format_dtype_querier_ptr)
    : format_dtype_querier_ptr_(format_dtype_querier_ptr) {}
OpFormatDtypeUpdateDescBase::~OpFormatDtypeUpdateDescBase() {}

Status OpFormatDtypeUpdateDescBase::GetAndCheckSupportedDtype(const UpdateInfo& update_info,
                                                              ge::DataType& op_kernel_data_type) const {
  ge::OpDescPtr op_desc_ptr = update_info.node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  string op_type = op_desc_ptr->GetType();
  string op_name = op_desc_ptr->GetName();
  std::vector<ge::DataType> op_kernel_data_type_vec;
  FE_CHECK_NOTNULL(format_dtype_querier_ptr_);
  if (format_dtype_querier_ptr_->GetSupportDataTypes(update_info.op_kernel_info_ptr,
                                                     update_info.input_or_output_info_ptr, update_info.node_ptr,
                                                     op_kernel_data_type_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][GetChkSptDtype] Failed to get the support data_types for %s.",
                    op_name.c_str());
    return FAILED;
  }

  uint32_t op_kernel_data_type_vec_size = op_kernel_data_type_vec.size();
  if (op_kernel_data_type_vec.empty() || update_info.matched_index >= op_kernel_data_type_vec_size) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][UpdFmtAndDtype][GetChkSptDtype] "
        "Op[name=%s,type=%s]: update the op_input_or_output_desc %u, the matched "
        "index %u is larger than or equals to the size of "
        "opKernelDataTypeVec %u.",
        op_name.c_str(), op_type.c_str(), update_info.index, update_info.matched_index, op_kernel_data_type_vec_size);
    return FAILED;
  }
  op_kernel_data_type = op_kernel_data_type_vec[update_info.matched_index];
  return SUCCESS;
}

Status OpFormatDtypeUpdateDescBase::UpdateDtype(const UpdateInfo& update_info) const {
  ge::OpDescPtr op_desc_ptr = update_info.node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  const string &op_type = op_desc_ptr->GetType();
  const string &op_name = op_desc_ptr->GetName();

  // 1. check the matched_index and op_kernel_data_type_vec_size
  ge::DataType op_kernel_data_type;
  if (GetAndCheckSupportedDtype(update_info, op_kernel_data_type) != SUCCESS) {
    return FAILED;
  }
  // 2. if the data type in op kernel is not equal to the original data type,
  // then update op_input_or_output_desc
  ge::DataType op_input_or_output_desc_origin_dtype = update_info.op_input_or_output_desc.GetDataType();
  if (op_kernel_data_type != op_input_or_output_desc_origin_dtype) {
    update_info.op_input_or_output_desc.SetDataType(op_kernel_data_type);

    FE_LOGD("Op[name=%s,type=%s]: update the %s tensor desc %u, new data type [%s], origin data type [%s].",
            op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index,
            DTypeToStr(op_kernel_data_type).c_str(), DTypeToStr(op_input_or_output_desc_origin_dtype).c_str());
  }
  return SUCCESS;
}

Status OpFormatDtypeUpdateDescBase::GetFormatAndDtypeVec(const OpKernelInfoPtr& op_kernel_info_ptr,
                                                         const InputOrOutputInfoPtr& input_or_output_info_ptr,
                                                         const ge::NodePtr &node,
                                                         std::vector<ge::Format>& op_kernel_format_vec,
                                                         std::vector<ge::DataType>& op_kernel_data_type_vec) const {
  if (format_dtype_querier_ptr_->GetSupportFormats(op_kernel_info_ptr, input_or_output_info_ptr, node,
                                                   op_kernel_format_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][GetFmtDtypeVec] Failed to get the support formats.");
    return FAILED;
  }

  if (format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, input_or_output_info_ptr, node,
                                                     op_kernel_data_type_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][GetFmtDtypeVec] Failed to get the support data_types.");
    return FAILED;
  }
  return SUCCESS;
}

Status OpFormatDtypeUpdateDescBase::UpdateFormat(const UpdateInfo& update_info) const {
  ge::OpDescPtr op_desc_ptr = update_info.node_ptr->GetOpDesc();
  const string &op_name = op_desc_ptr->GetName();
  std::vector<ge::Format> op_kernel_format_vec;
  std::vector<ge::DataType> op_kernel_data_type_vec;
  /* 1. Get all supported format and dtype */
  FE_LOGD("Update format for %s tensor %u of op %s.", IS_INPUT_TO_STRING(update_info.is_input), update_info.index,
          update_info.node_ptr->GetName().c_str());
  Status ret = GetFormatAndDtypeVec(update_info.op_kernel_info_ptr, update_info.input_or_output_info_ptr,
                                    update_info.node_ptr, op_kernel_format_vec, op_kernel_data_type_vec);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdFmt] Node[%s]: failed to obtain supported formats for %s %u",
                    op_name.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index);
    return ret;
  }
  /* 2. Check whether the matched index is valid for this op kernel */
  ret = CheckMatchedIndexValid(op_kernel_format_vec, op_kernel_data_type_vec, update_info.matched_index, op_desc_ptr,
                               update_info.index, update_info.is_input);
  /* Log is enough in func CheckMatchedIndexValid so the following check omits
   * the log. */
  if (ret != SUCCESS) {
    return ret;
  }

  ge::Format op_kernel_format = op_kernel_format_vec[update_info.matched_index];
  ge::DataType op_kernel_dtype = op_kernel_data_type_vec[update_info.matched_index];
  if (FE_ORIGIN_FORMAT_SET.count(op_kernel_format) == 0) {
    int64_t c0_val = GetC0ValByDataType(op_kernel_dtype);
    op_kernel_format = static_cast<ge::Format>(ge::GetFormatFromC0(op_kernel_format, GetC0BitValFromC0(c0_val)));
  }
  /* 3. update the tensor current format & dtype & shape according to the
   * selected format and dtype */
  ret = CalcNewShapeAndUpdate(update_info, op_kernel_format, op_kernel_dtype);

  return ret;
}
}  // namespace fe
