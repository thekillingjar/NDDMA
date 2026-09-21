/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_OP_FORMAT_DTYPE_UPDATE_DESC_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_OP_FORMAT_DTYPE_UPDATE_DESC_H_

#include "graph_optimizer/op_judge/format_and_dtype/update_desc/op_format_dtype_update_desc_base.h"
#include "ops_store/op_kernel_info.h"

namespace fe {

using OpFormatDtypeUpdateTensorDescBasePtr = std::shared_ptr<OpFormatDtypeUpdateDescBase>;

class OpFormatDtypeUpdateDesc {
 public:
  explicit OpFormatDtypeUpdateDesc(FormatDtypeQuerierPtr format_dtype_querier_ptr);

  virtual ~OpFormatDtypeUpdateDesc();

  Status Initialize();
  /**
   * update the format and data type of the input desc of the node
   * @param op_kernel_info_ptr op kernel information
   * @param matched_index: matched index of the op kernel information
   * @param tensor_index_name_map: the map of tensor index and its name
   * in the op kernel
   * @param node_ptr: current node, we will update it's tensor desc according
   * to the op kernel
   * @return SUCCESS or FAIL
   */
  Status UpdateTensorDtypeInfo(const OpKernelInfoPtr& op_kernel_info_ptr, const uint32_t& matched_index,
                              const IndexNameMap& tensor_index_name_map, const bool& is_input, ge::NodePtr& node_ptr);
  Status UpdateTensorFormatInfo(const OpKernelInfoPtr& op_kernel_info_ptr, const uint32_t& matched_index,
                              const IndexNameMap& tensor_index_name_map, const bool& is_input, ge::NodePtr& node_ptr);
  
 private:
  Status UpdateTensorDescInfo(const OpKernelInfoPtr& op_kernel_info_ptr, const uint32_t& matched_index,
                         const IndexNameMap& tensor_index_name_map, const bool& is_input, ge::NodePtr& node_ptr,
                         const bool is_dtype = 1);

  static const std::string DtypeFormatToString(bool is_dtype);

  OpFormatDtypeUpdateTensorDescBasePtr op_format_dtype_update_tensor_desc_ptr_;

  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_OP_FORMAT_DTYPE_UPDATE_DESC_H_
