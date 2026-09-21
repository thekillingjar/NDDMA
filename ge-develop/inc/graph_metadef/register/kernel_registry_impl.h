/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_KERNEL_REGISTER_IMPL_H_
#define INC_EXTERNAL_REGISTER_KERNEL_REGISTER_IMPL_H_
#include <unordered_map>
#include <string>

#include "kernel_registry.h"

namespace gert {
class KernelRegistryImpl : public KernelRegistry {
 public:
  static KernelRegistryImpl &GetInstance();
  void RegisterKernel(std::string kernel_type, KernelInfo kernel_infos) override;
  const KernelFuncs *FindKernelFuncs(const std::string &kernel_type) const override;
  const KernelInfo *FindKernelInfo(const std::string &kernel_type) const override;

  const std::unordered_map<std::string, KernelInfo> &GetAll() const;

 private:
  std::unordered_map<std::string, KernelInfo> kernel_infos_;
};
}

#endif // INC_EXTERNAL_REGISTER_KERNEL_REGISTER_IMPL_H_
