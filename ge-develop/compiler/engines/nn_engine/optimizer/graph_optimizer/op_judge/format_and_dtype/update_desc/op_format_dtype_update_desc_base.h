/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_OP_FORMAT_DTYPE_UPDATE_DESC_BASE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_OP_FORMAT_DTYPE_UPDATE_DESC_BASE_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/fe_op_info_common.h"
#include "common/update_tensor_desc.h"
#include "common/util/op_info_util.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "graph/compute_graph.h"
#include "ops_store/op_kernel_info.h"

namespace fe {
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;

class OpFormatDtypeUpdateDescBase {
 public:
  explicit OpFormatDtypeUpdateDescBase(const FormatDtypeQuerierPtr format_dtype_querier_ptr);
  ~OpFormatDtypeUpdateDescBase();

  /**
   * update the data type of the input or output desc of the node
   * @return SUCCESS or FAIL
   */
  Status UpdateDtype(const UpdateInfo &update_info) const;

  /**
   * update the format and data type of the output desc of the node
   * @return SUCCESS or FAIL
   */
  Status UpdateFormat(const UpdateInfo &update_info) const;

 private:
  /**
   * update the format and data type of the output desc of the node
   * @return SUCCESS or FAIL
   */
  Status GetFormatAndDtypeVec(const OpKernelInfoPtr &op_kernel_info_ptr,
                              const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                              std::vector<ge::Format> &op_kernel_format_vec,
                              std::vector<ge::DataType> &op_kernel_data_type_vec) const;
  /**
   * Get all supported data type and check whether the data types are valid
   * @return SUCCESS or FAIL
   */
  Status GetAndCheckSupportedDtype(const UpdateInfo &update_info, ge::DataType &op_kernel_data_type) const;

  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_OP_FORMAT_DTYPE_UPDATE_DESC_BASE_H_
