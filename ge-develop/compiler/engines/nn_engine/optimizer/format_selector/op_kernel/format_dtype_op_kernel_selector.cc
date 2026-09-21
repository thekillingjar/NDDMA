/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/op_kernel/format_dtype_op_kernel_selector.h"
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/util/json_util.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "common/configuration.h"

namespace fe {
FormatDtypeOpKernelSelector::FormatDtypeOpKernelSelector() : FormatDtypeSelectorBase() {}
FormatDtypeOpKernelSelector::~FormatDtypeOpKernelSelector() {}

Status FormatDtypeOpKernelSelector::GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const ge::NodePtr &node,
                                               const bool &is_dynamic_check,
                                               map<string, vector<ge::Format>> &format_map) {
  (void)node;
  for (InputOrOutputInfoPtr input_or_output_info_ptr : op_kernel_info_ptr->GetAllInputInfo()) {
    string unique_name = input_or_output_info_ptr->GetUniqueName();
    if (is_dynamic_check) {
      format_map[unique_name] = input_or_output_info_ptr->GetUnknownShapeFormat();
    } else {
      format_map[unique_name] = input_or_output_info_ptr->GetFormat();
    }
  }
 
  for (InputOrOutputInfoPtr input_or_output_info_ptr : op_kernel_info_ptr->GetAllOutputInfo()) {
    string unique_name = input_or_output_info_ptr->GetUniqueName();
    if (is_dynamic_check) {
      format_map[unique_name] = input_or_output_info_ptr->GetUnknownShapeFormat();
    } else {
      format_map[unique_name] = input_or_output_info_ptr->GetFormat();
    }
  }
  return SUCCESS;
}

Status FormatDtypeOpKernelSelector::GetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                          const ge::NodePtr &node,
                                                          const bool &is_dynamic_check,
                                                          FormatDtypeInfo &format_dtype_info) {
  (void)node;
  for (InputOrOutputInfoPtr input_or_output_info_ptr : op_kernel_info_ptr->GetAllInputInfo()) {
    string unique_name = input_or_output_info_ptr->GetUniqueName();
    if (is_dynamic_check) {
      format_dtype_info.format_map[unique_name] = input_or_output_info_ptr->GetUnknownShapeFormat();
      format_dtype_info.data_type_map[unique_name] = input_or_output_info_ptr->GetUnknownShapeDataType();
    } else {
      format_dtype_info.format_map[unique_name] = input_or_output_info_ptr->GetFormat();
      format_dtype_info.data_type_map[unique_name] = input_or_output_info_ptr->GetDataType();
    }
    format_dtype_info.sub_format_map[unique_name] = input_or_output_info_ptr->GetSubformat();
  }

  for (InputOrOutputInfoPtr input_or_output_info_ptr : op_kernel_info_ptr->GetAllOutputInfo()) {
    string unique_name = input_or_output_info_ptr->GetUniqueName();
    if (is_dynamic_check) {
      format_dtype_info.format_map[unique_name] = input_or_output_info_ptr->GetUnknownShapeFormat();
      format_dtype_info.data_type_map[unique_name] = input_or_output_info_ptr->GetUnknownShapeDataType();
    } else {
      format_dtype_info.format_map[unique_name] = input_or_output_info_ptr->GetFormat();
      format_dtype_info.data_type_map[unique_name] = input_or_output_info_ptr->GetDataType();
    }
    format_dtype_info.sub_format_map[unique_name] = input_or_output_info_ptr->GetSubformat();
  }
  return SUCCESS;
}

Status FormatDtypeOpKernelSelector::GetFallbackFlags(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                     const ge::NodePtr &node, bool is_dynamic_check,
                                                     std::vector<bool> &fallback_res) {
  (void)node;
  const FallbackFlags &fallback_flags = op_kernel_info_ptr->GetFallbackFlags();
  if (fallback_flags.fallbacks.empty() && fallback_flags.unknown_shape_fallbacks.empty()) {
    return SUCCESS;
  }
  if (is_dynamic_check && !fallback_flags.unknown_shape_fallbacks.empty()) {
    fallback_res = std::move(fallback_flags.unknown_shape_fallbacks);
  } else {
    fallback_res = std::move(fallback_flags.fallbacks);
  }
  return SUCCESS;
}

Status FormatDtypeOpKernelSelector::GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                      const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                      const ge::NodePtr &node, vector<ge::Format> &result) {
  (void)op_kernel_info_ptr;
  auto op_desc = node->GetOpDesc();
  if (IsOpDynamicImpl(op_desc)) {
    result = input_or_output_info_ptr->GetUnknownShapeFormat();
  } else {
    result = input_or_output_info_ptr->GetFormat();
  }
  return SUCCESS;
}

Status FormatDtypeOpKernelSelector::GetSupportFormatSubFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                              const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                              const ge::NodePtr &node, vector<ge::Format> &format_res,
                                                              vector<uint32_t> &sub_format_res,
                                                              uint32_t sub_format) {
  (void)op_kernel_info_ptr;
  (void)sub_format;
  auto op_desc = node->GetOpDesc();
  if (IsOpDynamicImpl(op_desc)) {
    format_res = input_or_output_info_ptr->GetUnknownShapeFormat();
  } else {
    format_res = input_or_output_info_ptr->GetFormat();
  }
  sub_format_res = input_or_output_info_ptr->GetSubformat();
  return SUCCESS;
}

Status FormatDtypeOpKernelSelector::GetSupportDataTypes(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                        const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                        const ge::NodePtr &node, vector<ge::DataType> &result) {
  (void)op_kernel_info_ptr;
  auto op_desc = node->GetOpDesc();
  if (IsOpDynamicImpl(op_desc)) {
    result = input_or_output_info_ptr->GetUnknownShapeDataType();
  } else {
    result = input_or_output_info_ptr->GetDataType();
  }
  return SUCCESS;
}

Status FormatDtypeOpKernelSelector::GetDynamicFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                          const ge::NodePtr &node,
                                                          const bool &is_dynamic_check,
                                                          const HeavyFormatInfo &heavy_format_info,
                                                          FormatDtypeInfo &format_dtype_info,
                                                          uint32_t sub_format) {
  (void)op_kernel_info_ptr;
  (void)node;
  (void)is_dynamic_check;
  (void)heavy_format_info;
  (void)format_dtype_info;
  (void)sub_format;
  return SUCCESS;
}
}  // namespace fe
