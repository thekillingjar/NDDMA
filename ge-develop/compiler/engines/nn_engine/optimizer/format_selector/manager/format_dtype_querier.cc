/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/manager/format_dtype_querier.h"
namespace fe {
FormatDtypeQuerier::FormatDtypeQuerier(const std::string &engine_name) : FormatDtypeManagerBase(engine_name) {}

FormatDtypeQuerier::~FormatDtypeQuerier() {}

Status FormatDtypeQuerier::GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                                             const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                             const ge::NodePtr &node, vector<ge::Format> &result) const {
  bool is_dynamic_check = IsOpDynamicImpl(node->GetOpDesc());
  FormatDtypeSelectorBasePtr selector = GetSelector(op_kernel_info_ptr, is_dynamic_check);
  FE_CHECK_NOTNULL(selector);
  return selector->GetSupportFormats(op_kernel_info_ptr, input_or_output_info_ptr, node, result);
}

Status FormatDtypeQuerier::GetSupportFormatSubFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                     const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                     const ge::NodePtr &node, vector<ge::Format> &format_res,
                                                     vector<uint32_t> &sub_format_res, uint32_t sub_format) const {
  bool is_dynamic_check = IsOpDynamicImpl(node->GetOpDesc());
  FormatDtypeSelectorBasePtr selector = GetSelector(op_kernel_info_ptr, is_dynamic_check);
  FE_CHECK_NOTNULL(selector);
  return selector->GetSupportFormatSubFormat(op_kernel_info_ptr, input_or_output_info_ptr, node,
                                             format_res, sub_format_res, sub_format);
}

Status FormatDtypeQuerier::GetSupportDataTypes(const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                               const ge::NodePtr &node, vector<ge::DataType> &result) const {
  bool is_dynamic_check = IsOpDynamicImpl(node->GetOpDesc());
  FormatDtypeSelectorBasePtr selector = GetSelector(op_kernel_info_ptr, is_dynamic_check);
  FE_CHECK_NOTNULL(selector);
  return selector->GetSupportDataTypes(op_kernel_info_ptr, input_or_output_info_ptr, node, result);
}

Status FormatDtypeQuerier::GetSupportFormatAndDtype(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                    const ge::NodePtr &node,
                                                    const bool &is_dynamic_check,
                                                    FormatDtypeInfo &format_dtype_info) const {
  FormatDtypeSelectorBasePtr selector = GetSelector(op_kernel_info_ptr, is_dynamic_check);
  FE_CHECK_NOTNULL(selector);
  return selector->GetSupportFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, format_dtype_info);
}

Status FormatDtypeQuerier::GetFallbackFlags(const OpKernelInfoPtr &op_kernel_info_ptr,
                                            const ge::NodePtr &node,
                                            std::vector<bool> &fallback_flags) const {
  bool is_dynamic_check = IsOpDynamicImpl(node->GetOpDesc());
  FormatDtypeSelectorBasePtr selector = GetSelector(op_kernel_info_ptr, is_dynamic_check);
  FE_CHECK_NOTNULL(selector);
  return selector->GetFallbackFlags(op_kernel_info_ptr, node, is_dynamic_check, fallback_flags);
}

Status FormatDtypeQuerier::GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const ge::NodePtr &node,
                                               map<string, vector<ge::Format>> &format_map) const {
  bool is_dynamic_check = IsOpDynamicImpl(node->GetOpDesc());
  FormatDtypeSelectorBasePtr selector = GetSelector(op_kernel_info_ptr, is_dynamic_check);
  FE_CHECK_NOTNULL(selector);
  return selector->GetAllSupportFormat(op_kernel_info_ptr, node, is_dynamic_check, format_map);
}
}  // namespace fe
