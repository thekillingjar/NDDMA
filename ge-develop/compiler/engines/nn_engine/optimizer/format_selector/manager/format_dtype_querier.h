/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_QUERIER_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_QUERIER_H_

#include "format_selector/manager/format_dtype_manager_base.h"

namespace fe {
class FormatDtypeQuerier : public FormatDtypeManagerBase {
 public:
  explicit FormatDtypeQuerier(const std::string &engine_name);
  ~FormatDtypeQuerier() override;

  /**
   * Get the support formats and dtyps from the op_desc, if failed, get the
   * dynamic formats and dtyps. Used for CheckSubStoreSupported.
   * @param op_kernel_info_ptr op kernel info
   * @param node node ptr
   * @param format_map formats result
   * @param data_type_map dtypes result
   * @return SUCCESS or FAILED
   */
  Status GetSupportFormatAndDtype(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                                  const bool &is_dynamic_check, FormatDtypeInfo &format_dtype_info) const;

  /**
   * Get the support formats from the op_desc by the input_or_output_info.
   * Used for OpFormatDtypeJudge and HeavyFormatDistribution.
   * @param input_or_output_info_ptr input_or_output_info
   * @param node NodePtr
   * @param result formats
   * @return SUCCESS or FAILED
   */
  Status GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                           const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                           vector<ge::Format> &result) const;

  /**
   * Get the support subformats from the op_desc by the input_or_output_info.
   * Used for OpFormatDtypeJudge and HeavyFormatDistribution.
   * @param input_or_output_info_ptr input_or_output_info
   * @param node NodePtr
   * @param result subformats
   * @return SUCCESS or FAILED
   */
  Status GetSupportFormatSubFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                   const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                                   vector<ge::Format> &format_res, vector<uint32_t> &sub_format_res,
                                   uint32_t sub_format = 1) const;
  /**
   * Get the support dtypes from the op_desc by the input_or_output_info.
   * Used for OpFormatDtypeJudge and HeavyFormatDistribution.
   * @param input_or_output_info_ptr input_or_output_info
   * @param node nodePtr
   * @param result d_types
   * @return SUCCESS or FAILED
   */
  Status GetSupportDataTypes(const OpKernelInfoPtr &op_kernel_info_ptr,
                             const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                             vector<ge::DataType> &result) const;

  Status GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                             map<string, vector<ge::Format>> &format_map) const;

  Status GetFallbackFlags(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                        std::vector<bool> &fallback_flags) const;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_QUERIER_H_
