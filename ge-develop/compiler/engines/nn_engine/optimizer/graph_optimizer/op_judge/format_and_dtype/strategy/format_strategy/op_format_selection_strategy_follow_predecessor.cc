/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_format_selection_strategy_follow_predecessor.h"

namespace fe {
namespace {
/* In Pytorch: The Data op's output format will be heavy format and there are
 * only Data->Op->NetOutput(three ops) in each build graph. So we can not
 * distribute heavy format from heavy op. The only way to reduce the number of
 * TransData is using concecutive principle. The Op's input format will
 * follow the output format out Data. */
bool IsHeavyData(const ge::Format& father_output_format, const string& father_op_type) {
  if (!IsRootGraphData(father_op_type) && !IsHeavyDataOp(father_op_type)) {
    return false;
  }
  bool is_heavy_format = false;
  for (const auto& ele : FE_HEAVY_FORMAT_VECTOR) {
    if (ele == father_output_format) {
      is_heavy_format = true;
    }
  }

  if (is_heavy_format) {
    return true;
  } else {
    return false;
  }
}
} // namespace

OpFormatSelectionStrategyFollowPredecessor::OpFormatSelectionStrategyFollowPredecessor(
    FormatDtypeQuerierPtr format_dtype_querier_ptr)
    : format_dtype_querier_ptr_(format_dtype_querier_ptr) {}

Status OpFormatSelectionStrategyFollowPredecessor::Initialize() {
  FE_MAKE_SHARED(format_matcher_ = std::make_shared<OpFormatMatcher>(), return FAILED);
  FE_CHECK_NOTNULL(format_matcher_);
  return SUCCESS;
}

OpFormatSelectionStrategyFollowPredecessor::~OpFormatSelectionStrategyFollowPredecessor() {}

Status OpFormatSelectionStrategyFollowPredecessor::Run(FormatDtypeSelectionBasicInfo& basic_info) {
  ge::OpDescPtr cur_op_desc_ptr = basic_info.node->GetOpDesc();
  std::string cur_op_desc_name = cur_op_desc_ptr->GetName();
  std::string cur_op_desc_type = cur_op_desc_ptr->GetType();
  FE_LOGD("Op[name=%s,type=%s]: match the format with predecessor for tensor %u.", cur_op_desc_name.c_str(),
          cur_op_desc_type.c_str(), basic_info.index);
  /* 1. Get father node */
  ge::InDataAnchorPtr in_data_anchor;
  bool has_no_father = false;
  CheckHasNoFather(basic_info.is_input, static_cast<int32_t>(basic_info.index), basic_info.node,
                   in_data_anchor, has_no_father);
  /* 1. If the node is Gray list does not have predecessors, we just match the
   * dtype with its original dtype. */
  if (has_no_father) {
    FE_LOGD("Op[name=%s,type=%s]: does not have a father node on the input [%u]. Match with its original format.",
        cur_op_desc_name.c_str(), cur_op_desc_type.c_str(), basic_info.index);
    return FAILED;
  }

  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(peer_out_anchor);
  uint32_t father_out_anchor_index = static_cast<uint32_t>(peer_out_anchor->GetIdx());
  ge::OpDescPtr father_op_desc = peer_out_anchor->GetOwnerNode()->GetOpDesc();

  /* 2. Match all supported data type with father's output data type. */
  ge::GeTensorDesc father_output_desc = father_op_desc->GetOutputDesc(father_out_anchor_index);
  ge::Format father_output_format = static_cast<ge::Format>(
          ge::GetPrimaryFormat(father_output_desc.GetFormat()));
  /* Now, if the father's type is not Data or father format is not heavy format,
   * we just return FAILED. Because in major scenarios, we will not use
   * concecutive principle. */
  if (!IsHeavyData(father_output_format, father_op_desc->GetType())) {
    return FAILED;
  }

  FE_LOGD("Op[name=%s,type=%s]: match format for the input %u between father op[name=%s,type=%s] and the current op.",
          cur_op_desc_name.c_str(), cur_op_desc_type.c_str(), basic_info.index, father_op_desc->GetName().c_str(),
          father_op_desc->GetType().c_str());

  // 3. Get all supported format
  vector<ge::Format> input_or_output_format;
  if (format_dtype_querier_ptr_->GetSupportFormats(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                   basic_info.node, input_or_output_format) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][FmtJdg][Run] Failed to get the support formats, return FAILED.");
    return FAILED;
  }

  format_matcher_->FilterFormatIndex(basic_info.node, basic_info.op_kernel_info_ptr,
                                     input_or_output_format, basic_info.matched_index_vec);
  FE_CHECK_NOTNULL(basic_info.cur_tensor_desc);
  auto cur_origin_format = GetCurOpOriginFormat(*(basic_info.cur_tensor_desc));
  /* Get expect format to conform shape is known */
  auto cur_origin_shape = GetCurOpOriginShape(*(basic_info.cur_tensor_desc));

  /* 4. match the format with father output format. */
  Status match_origin_format_res = format_matcher_->Match(input_or_output_format, father_output_format,
                                                          cur_origin_format, basic_info.matched_index_vec);
  if (match_origin_format_res == SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: match the predecessor's format successfully.", cur_op_desc_name.c_str(),
            cur_op_desc_type.c_str());
    return SUCCESS;
  } else {
    FE_LOGD("Op[name=%s,type=%s]: match predecessor's format unsuccessful.", cur_op_desc_name.c_str(),
            cur_op_desc_type.c_str());
    return match_origin_format_res;
  }
}
}  // namespace fe
