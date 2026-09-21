/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "op_dtype_selection_strategy_table_select_base.h"
 #include "common/configuration.h"
 #include "ops_store/ops_kernel_manager.h"
 
 namespace fe {
 OpDtypeSelectionStrategyTableSelectBase::OpDtypeSelectionStrategyTableSelectBase(
     FormatDtypeQuerierPtr format_dtype_querier_ptr,
     OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr) :
     OpDtypeSeletionStrategyBase(format_dtype_querier_ptr),
     op_dtype_precise_matcher_ptr_(op_dtype_precise_matcher_ptr) {}
 
 Status OpDtypeSelectionStrategyTableSelectBase::GetOpPrecisionPolicy(const OpKernelInfoPtr &op_kernel_info_ptr,
     PrecisionPolicy &precision_policy) const {
   FE_CHECK_NOTNULL(op_kernel_info_ptr);
   // Get form op kernel info config
   precision_policy = op_kernel_info_ptr->GetPrecisionPolicy();
   // Update from mix_list config
   precision_policy = Configuration::Instance(AI_CORE_NAME).GetPrecisionPolicy(op_kernel_info_ptr->GetOpType(),
                                                                               precision_policy);
   return SUCCESS;
 }
 
 void OpDtypeSelectionStrategyTableSelectBase::GetDtypeMatchListFromMap(
     const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
     const ge::DataType &key_dtype, vector<ge::DataType> &match_dtype_vec) const {
   auto iter = dtype_match_map.find(key_dtype);
   if (iter == dtype_match_map.end()) {
     match_dtype_vec.emplace_back(key_dtype);
   } else {
     match_dtype_vec = iter->second;
   }
 }
 
 Status OpDtypeSelectionStrategyTableSelectBase::MatchDtypeFromList(
     const vector<ge::DataType> &input_or_output_dtype_vec, const vector<ge::DataType> &match_dtype_vec,
     const ForbiddenDtype &forbidden_dtype, vector<uint32_t> &matched_index_vec) {
   for (auto dtype : match_dtype_vec) {
     Status res = op_dtype_precise_matcher_ptr_->Match(input_or_output_dtype_vec, dtype,
                                                       matched_index_vec, forbidden_dtype);
     if (res == SUCCESS) {
       FE_LOGD("[GraphOpt][DtypeJdg][TableSelect] Success matched data_type %d.", dtype);
       return SUCCESS;
     }
   }
   return FAILED;
 }
 
 void OpDtypeSelectionStrategyTableSelectBase::DumpDtypeMatchMapAndList(
     const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
     const vector<ge::DataType> &match_dtype_vec) const {
   for (const auto &item : dtype_match_map) {
     FE_LOGD("dtype_match_map[%d], match_dtype_vec size %zu", item.first, item.second.size());
     std::string match_dtype_vec_str = "";
     for (const auto &dtype : item.second) {
       match_dtype_vec_str += std::to_string(dtype) + ",";
     }
     FE_LOGD("dtype_match_map[%d], match_dtype_vec: %s", item.first, match_dtype_vec_str.c_str());
   }
 
   FE_LOGD("final match_dtype_vec size %zu", match_dtype_vec.size());
   std::string final_match_dtype_vec_str = "";
   for (const auto &dtype : match_dtype_vec) {
     final_match_dtype_vec_str += std::to_string(dtype) + ",";
   }
   FE_LOGD("final match_dtype_vec: %s", final_match_dtype_vec_str.c_str());
 }
 
 Status OpDtypeSelectionStrategyTableSelectBase::RunWithDtypeMap(
     const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
     const ForbiddenDtype &forbidden_dtype, FormatDtypeSelectionBasicInfo &basic_info) {
   vector<ge::DataType> input_or_output_dtype_vec;
   if (format_dtype_querier_ptr_->GetSupportDataTypes(basic_info.op_kernel_info_ptr, basic_info.tensor_kernel_info_ptr,
                                                      basic_info.node, input_or_output_dtype_vec) != SUCCESS) {
     REPORT_FE_ERROR("[GraphOpt][DtypeJdg][TableSelect] Op[name=%s,type=%s] fail to get the support data_types.",
                     basic_info.node->GetNamePtr(), basic_info.node->GetTypePtr());
     return FAILED;
   }
 
   FE_CHECK_NOTNULL(basic_info.cur_tensor_desc);
   ge::DataType origin_dtype = basic_info.cur_tensor_desc->GetDataType();
   vector<ge::DataType> match_dtype_vec;
   GetDtypeMatchListFromMap(dtype_match_map, origin_dtype, match_dtype_vec);
   Status ret = MatchDtypeFromList(input_or_output_dtype_vec, match_dtype_vec, forbidden_dtype,
                                   basic_info.matched_index_vec);
   if (ret != SUCCESS) {
     FE_LOGW("[GraphOpt][DtypeJdg][TableSelect] Op[name=%s,type=%s] fail to find any matched data type for "
             "tensor[name=%s,idx=%u], origin_dtype %d.", basic_info.node->GetNamePtr(), basic_info.node->GetTypePtr(),
             basic_info.cur_tensor_desc->GetName().c_str(), basic_info.index, origin_dtype);
     DumpDtypeMatchMapAndList(dtype_match_map, match_dtype_vec);
   }
   return ret;
 }
 }  // namespace fe