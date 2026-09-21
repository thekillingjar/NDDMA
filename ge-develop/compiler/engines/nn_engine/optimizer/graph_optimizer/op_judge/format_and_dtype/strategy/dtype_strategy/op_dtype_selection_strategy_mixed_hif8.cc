/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "op_dtype_selection_strategy_mixed_hif8.h"
 #include "common/configuration.h"
 #include "ops_store/ops_kernel_manager.h"
 
 namespace fe {
 OpDtypeSelectionStrategyMixedHif8::OpDtypeSelectionStrategyMixedHif8(
     FormatDtypeQuerierPtr format_dtype_querier_ptr,
     OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr) :
     OpDtypeSelectionStrategyTableSelectBase(format_dtype_querier_ptr, op_dtype_precise_matcher_ptr) {}
 
 Status OpDtypeSelectionStrategyMixedHif8::GetFatherOutputDtype(const ge::InDataAnchorPtr &in_data_anchor,
     ge::DataType &father_output_dtype) const {
   auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
   FE_CHECK_NOTNULL(peer_out_anchor);
   uint32_t father_out_anchor_index = static_cast<uint32_t>(peer_out_anchor->GetIdx());
   ge::OpDescPtr father_op_desc = peer_out_anchor->GetOwnerNode()->GetOpDesc();
   FE_CHECK_NOTNULL(father_op_desc);
   ge::NodePtr father_node = peer_out_anchor->GetOwnerNode();
   FE_CHECK_NOTNULL(father_node);
   auto father_output_desc = father_op_desc->GetOutputDescPtr(father_out_anchor_index);
   FE_CHECK_NOTNULL(father_output_desc);
   father_output_dtype = father_output_desc->GetDataType();
   FE_LOGD("[GraphOpt][DtypeJdg][MixedHif8] Father Op[name=%s,type=%s] output dtype is %d.",
           father_op_desc->GetNamePtr(), father_op_desc->GetTypePtr(), father_output_dtype);
   return SUCCESS;
 }
 
 Status OpDtypeSelectionStrategyMixedHif8::RunWithDtypeMapForGrayList(
     const std::map<ge::DataType,std::vector<ge::DataType>> &dtype_match_map, const ForbiddenDtype &forbidden_dtype,
     const ge::InDataAnchorPtr &in_data_anchor, FormatDtypeSelectionBasicInfo &basic_info) {
     vector<ge::DataType> input_or_output_dtype_vec;
   if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                      basic_info.node, input_or_output_dtype_vec) != SUCCESS) {
     REPORT_FE_ERROR("[GraphOpt][DtypeJdg][MixedHif8] Op[name=%s,type=%s]: Fail to get the support data_types.",
                     basic_info.node->GetNamePtr(), basic_info.node->GetTypePtr());
     return FAILED;
   }
 
   ge::DataType father_output_dtype = ge::DT_FLOAT;
   GetFatherOutputDtype(in_data_anchor, father_output_dtype);
   FE_CHECK_NOTNULL(basic_info.cur_tensor_desc);
   ge::DataType origin_dtype = basic_info.cur_tensor_desc->GetDataType();
 
   vector<ge::DataType> match_dtype_vec;
   GetDtypeMatchListFromMap(dtype_match_map, father_output_dtype, match_dtype_vec);
   // If father_output_dtype=hif8 and origin_dtype=bf16, replace [hif8,fp16,fp32] with [hif8,bf16,fp32]
   if (father_output_dtype == ge::DT_HIFLOAT8 && origin_dtype == ge::DT_BF16) {
     std::replace(match_dtype_vec.begin(), match_dtype_vec.end(), ge::DT_FLOAT16, ge::DT_BF16);
   }
   Status ret = MatchDtypeFromList(input_or_output_dtype_vec, match_dtype_vec, forbidden_dtype,
                                   basic_info.matched_index_vec);
   if (ret != SUCCESS) {
     FE_LOGW("[GraphOpt][DtypeJdg][MixedHif8] Op[name=%s,type=%s]: Fail to find any matched data type for "
             "tensor[name=%s,idx=%u], father_output_dtype %d, origin_dtype %d.", basic_info.node->GetNamePtr(),
             basic_info.node->GetTypePtr(), basic_info.cur_tensor_desc->GetName().c_str(), basic_info.index,
             father_output_dtype, origin_dtype);
     DumpDtypeMatchMapAndList(dtype_match_map, match_dtype_vec);
   }
   return ret;
 }
 
 Status OpDtypeSelectionStrategyMixedHif8::Run(FormatDtypeSelectionBasicInfo &basic_info,
     ForbiddenDtype forbidden_dtype) {
   FE_CHECK_NOTNULL(basic_info.node);
   auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
   FE_CHECK_NOTNULL(cur_op_desc_ptr);
 
   PrecisionPolicy precision_policy = GRAY;
   if (GetOpPrecisionPolicy(basic_info.op_kernel_info_ptr, precision_policy) != SUCCESS) {
     FE_LOGW("[GraphOpt][DtypeJdg][MixedHif8] Op[name=%s,type=%s]: Failed to get precision policy.",
             cur_op_desc_ptr->GetNamePtr(), cur_op_desc_ptr->GetTypePtr());
     return FAILED;
   }
   FE_LOGD("[GraphOpt][DtypeJdg][MixedHif8] Op[name=%s,type=%s]: Start match dtype for tensor[%u] precision policy[%u]",
           cur_op_desc_ptr->GetNamePtr(), cur_op_desc_ptr->GetTypePtr(), basic_info.index, precision_policy);
 
   if (precision_policy == BLACK) {
     RunWithDtypeMap(dtype_match_map_black_hif8, forbidden_dtype, basic_info);
   } else if (precision_policy == WHITE) {
     RunWithDtypeMap(dtype_match_map_white_hif8, forbidden_dtype, basic_info);
   } else {
     ge::InDataAnchorPtr in_data_anchor;
     bool has_no_father = false;
     CheckHasNoFather(basic_info.is_input, static_cast<int32_t>(basic_info.index), basic_info.node, in_data_anchor,
                     has_no_father);
     if (has_no_father) {
       RunWithDtypeMap(dtype_match_map_black_hif8, forbidden_dtype, basic_info);
     } else {
       RunWithDtypeMapForGrayList(dtype_match_map_gray_hif8, forbidden_dtype, in_data_anchor, basic_info);
     }
   }
 
   FE_LOGD("[GraphOpt][DtypeJdg][MixedHif8] Op[name=%s,type=%s]: End match dtype for tensor[%u]",
           cur_op_desc_ptr->GetNamePtr(), cur_op_desc_ptr->GetTypePtr(), basic_info.index);
   return SUCCESS;
 }
 }  // namespace fe