/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_MANAGER_BASE_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_MANAGER_BASE_H_

#include "common/fe_utils.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "common/string_utils.h"
#include "format_selector/builtin/format_dtype_op_builtin_selector.h"
#include "format_selector/op_customize/format_dtype_op_customize_selector.h"
#include "format_selector/op_kernel/format_dtype_op_kernel_selector.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"

namespace fe {
using FormatDtypeSelectorBasePtr = std::shared_ptr<FormatDtypeSelectorBase>;
using FormatDtypeOpKernelSelectorPtr = std::shared_ptr<FormatDtypeOpKernelSelector>;
using FormatDtypeOpCustomizeSelectorPtr = std::shared_ptr<FormatDtypeOpCustomizeSelector>;
using FormatDtypeOpBuiltinSelectorPtr = std::shared_ptr<FormatDtypeOpBuiltinSelector>;

class FormatDtypeManagerBase {
 public:
  explicit FormatDtypeManagerBase(const std::string &engine_name);
  virtual ~FormatDtypeManagerBase();
  FormatDtypeSelectorBasePtr GetSelector(const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_dynamic_check) const;
  bool IsSelectFormat(const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_dynamic_check);
  bool IsOpPatternBroadcast(const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_dynamic_check);

 protected:
  const std::string& GetEngineName() const;

 private:
  std::string engine_name_;
  FormatDtypeOpKernelSelectorPtr format_dtype_op_kernel_selector_ptr_;
  FormatDtypeOpCustomizeSelectorPtr format_dtype_op_customize_selector_ptr_;
  FormatDtypeOpBuiltinSelectorPtr format_dtype_op_builtin_selector_ptr_;
  /* To check weather format or unknownshape_format is set in op kernel info
   * file, if it is set, then use op kernel info, else, use it by op.pattern
   * value. */
  bool IsOpKernelEnable(const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_dynamic_check) const;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_MANAGER_BASE_H_
