/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/manager/format_dtype_manager_base.h"

namespace fe {
FormatDtypeManagerBase::FormatDtypeManagerBase(const std::string &engine_name) : engine_name_(engine_name) {
  FE_MAKE_SHARED(format_dtype_op_kernel_selector_ptr_ =
      std::make_shared<FormatDtypeOpKernelSelector>(), return);
  FE_MAKE_SHARED(format_dtype_op_customize_selector_ptr_ =
      std::make_shared<FormatDtypeOpCustomizeSelector>(engine_name), return);
  FE_MAKE_SHARED(format_dtype_op_builtin_selector_ptr_ =
      std::make_shared<FormatDtypeOpBuiltinSelector>(), return);
}

FormatDtypeManagerBase::~FormatDtypeManagerBase() {}

FormatDtypeSelectorBasePtr FormatDtypeManagerBase::GetSelector(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                               const bool &is_dynamic_check) const {
  if (IsOpKernelEnable(op_kernel_info_ptr, is_dynamic_check)) {
    return format_dtype_op_kernel_selector_ptr_;
  }
  OpPattern pattern = op_kernel_info_ptr->GetOpPattern();
  if (pattern == OP_PATTERN_OP_CUSTOMIZE) {
    return format_dtype_op_customize_selector_ptr_;
  } else if (pattern == OP_PATTERN_OP_KERNEL || pattern == OP_PATTERN_FORMAT_AGNOSTIC) {
    return format_dtype_op_kernel_selector_ptr_;
  }
  return format_dtype_op_builtin_selector_ptr_;
}

const std::string& FormatDtypeManagerBase::GetEngineName() const {
  return engine_name_;
}

bool FormatDtypeManagerBase::IsOpKernelEnable(const OpKernelInfoPtr &op_kernel_info_ptr,
                                              const bool &is_dynamic_check) const {
  for (const auto &input_info_ptr : op_kernel_info_ptr->GetAllInputInfo()) {
    if (is_dynamic_check && !input_info_ptr->GetUnknownShapeFormat().empty()) {
      return true;
    }
    if (!is_dynamic_check && !input_info_ptr->GetFormat().empty()) {
      return true;
    }
  }
  return false;
}

bool FormatDtypeManagerBase::IsSelectFormat(const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_dynamic_check) {
  return GetSelector(op_kernel_info_ptr, is_dynamic_check) == format_dtype_op_customize_selector_ptr_;
}

bool FormatDtypeManagerBase::IsOpPatternBroadcast(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                  const bool &is_dynamic_check) {
  OpPattern pattern = op_kernel_info_ptr->GetOpPattern();
  return GetSelector(op_kernel_info_ptr, is_dynamic_check) == format_dtype_op_builtin_selector_ptr_ &&
          (pattern == OP_PATTERN_BROADCAST || pattern == OP_PATTERN_BROADCAST_ENHANCED);
}
}  // namespace fe
