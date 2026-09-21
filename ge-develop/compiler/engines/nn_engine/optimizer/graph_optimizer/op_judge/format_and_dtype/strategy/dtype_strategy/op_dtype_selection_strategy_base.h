/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_BASE_H_
#define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_BASE_H_

#include "common/fe_inner_error_codes.h"
#include "common/fe_op_info_common.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "graph/compute_graph.h"

namespace fe {
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;
/** @param op_kernel_info_ptr: ops kernel information
   * @param tensor_kernel_info_ptr: the input_info or output_info of op kernel
   * @param node: current node
   * @param index: the index of the input or output desc
   * @param cur_tensor_desc: the input or output desc of the cur_node
   * @param matched_index_vec: output parameter, all supported dtype or format index
  **/
struct FormatDtypeSelectionBasicInfo {
  const OpKernelInfoPtr& op_kernel_info_ptr;
  const InputOrOutputInfoPtr& tensor_kernel_info_ptr;
  const ge::NodePtr& node;
  const uint32_t& index;  // tensor index
  ge::ConstGeTensorDescPtr& cur_tensor_desc;
  const bool& is_input;
  vector<uint32_t>& matched_index_vec;
};

class OpDtypeSeletionStrategyBase {
 public:
  using SelectionBasicInfo = struct FormatDtypeSelectionBasicInfo;

  using OpDtypeSeletionStrategyBasePtr = std::shared_ptr<OpDtypeSeletionStrategyBase>;

  explicit OpDtypeSeletionStrategyBase(FormatDtypeQuerierPtr format_dtype_querier_ptr);
  virtual ~OpDtypeSeletionStrategyBase();

  /**
   * run the current strategy
   * @return SUCCESS or FAIL
   */
  virtual Status Run(FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype) = 0;

 protected:
  // next strategy
  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
};
}  // namespace fe

#endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_BASE_H_
