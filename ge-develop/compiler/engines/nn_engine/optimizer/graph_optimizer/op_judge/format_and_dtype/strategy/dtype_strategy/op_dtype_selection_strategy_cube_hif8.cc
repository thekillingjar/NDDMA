/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "op_dtype_selection_strategy_cube_hif8.h"
 #include "common/configuration.h"
 #include "ops_store/ops_kernel_manager.h"
 
 namespace fe {
 OpDtypeSelectionStrategyCubeHif8::OpDtypeSelectionStrategyCubeHif8(
     FormatDtypeQuerierPtr format_dtype_querier_ptr,
     OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr) :
     OpDtypeSelectionStrategyTableSelectBase(format_dtype_querier_ptr, op_dtype_precise_matcher_ptr) {}
 
 Status OpDtypeSelectionStrategyCubeHif8::Run(FormatDtypeSelectionBasicInfo& basic_info,
     ForbiddenDtype forbidden_dtype) {
   FE_CHECK_NOTNULL(basic_info.node);
   auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
   FE_CHECK_NOTNULL(cur_op_desc_ptr);
   FE_LOGD("[GraphOpt][DtypeJdg][CubeHif8] Op[name=%s,type=%s]: Start match dtype for tensor[%u]",
           cur_op_desc_ptr->GetNamePtr(), cur_op_desc_ptr->GetTypePtr(), basic_info.index);
 
   const std::unordered_set<string> &fp16_op_type_list = Configuration::Instance(AI_CORE_NAME).GetFp16OpTypeList();
   bool is_cube_op = fp16_op_type_list.count(cur_op_desc_ptr->GetTypePtr()) > 0;
   if (is_cube_op) {
     RunWithDtypeMap(dtype_match_map_white_hif8, forbidden_dtype, basic_info);
   } else {
     RunWithDtypeMap(dtype_match_map_black_hif8, forbidden_dtype, basic_info);
   }
 
   FE_LOGD("[GraphOpt][DtypeJdg][CubeHif8] Op[name=%s,type=%s]: End match dtype for tensor[%u]",
           cur_op_desc_ptr->GetNamePtr(), cur_op_desc_ptr->GetTypePtr(), basic_info.index);
   return SUCCESS;
 }
 }  // namespace fe