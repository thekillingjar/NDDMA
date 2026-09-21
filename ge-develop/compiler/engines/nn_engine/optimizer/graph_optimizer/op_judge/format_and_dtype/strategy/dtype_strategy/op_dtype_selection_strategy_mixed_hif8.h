/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_MIXED_HIF8_H_
 #define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_MIXED_HIF8_H_
 
 #include "op_dtype_selection_strategy_table_select_base.h"
 
 namespace fe {
 class OpDtypeSelectionStrategyMixedHif8 : public OpDtypeSelectionStrategyTableSelectBase {
  public:
   OpDtypeSelectionStrategyMixedHif8(
       FormatDtypeQuerierPtr format_dtype_querier_ptr,
       OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr);
 
   ~OpDtypeSelectionStrategyMixedHif8() override = default;
 
   Status Run(FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype) override;
 
   Status RunWithDtypeMapForGrayList(const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
     const ForbiddenDtype &forbidden_dtype, const ge::InDataAnchorPtr &in_data_anchor,
     FormatDtypeSelectionBasicInfo &basic_info);
 
  private:
   OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr_;
 
   Status GetFatherOutputDtype(const ge::InDataAnchorPtr &in_data_anchor, ge::DataType &father_output_dtype) const;
 };
 }  // namespace fe
 #endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_MIXED_HIF8_H_