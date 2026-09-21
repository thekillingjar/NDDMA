/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_REGISTER_KERNEL_REGISTER_DATA_H_
#define METADEF_CXX_REGISTER_KERNEL_REGISTER_DATA_H_
#include <string>
#include "register/kernel_registry.h"
namespace gert {
class KernelRegisterData {
 public:
  explicit KernelRegisterData(const ge::char_t *kernel_type);

  KernelRegistry::KernelFuncs &GetFuncs() {
    return funcs_;
  }

  const std::string &GetKernelType() const {
    return kernel_type_;
  }

  std::string &GetCriticalSection() {
    return critical_section_;
  }

 private:
  std::string critical_section_;
  std::string kernel_type_;
  KernelRegistry::KernelFuncs funcs_;
};
}  // namespace gert

#endif  // METADEF_CXX_REGISTER_KERNEL_REGISTER_DATA_H_
