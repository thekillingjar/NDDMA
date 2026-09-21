/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A921B1BB102444BCB310056B63147F7F_H
#define A921B1BB102444BCB310056B63147F7F_H

#include "register/kernel_registry_impl.h"
#include <string>
#include <vector>
#include <set>
#include "optional_backup.h"
namespace gert {
struct StubKernelRegistry : KernelRegistryImpl {
  explicit StubKernelRegistry(KernelRegistryImpl &registry) : KernelRegistryImpl(registry) {}
  const KernelFuncs *FindKernelFuncs(const std::string &kernel_type) const override {
    static KernelRegistry::KernelFuncs always_success_kernels{SuccessKernelFunc, SuccessCreateOutput};
    if (always_return_success_kernel && always_return_success_kernel_excluded.count(kernel_type) == 0U) {
      return &always_success_kernels;
    } else {
      return KernelRegistryImpl::FindKernelFuncs(kernel_type);
    }
  }

  static ge::graphStatus SuccessKernelFunc(KernelContext *run_context) {
    return ge::GRAPH_SUCCESS;
  }

  static ge::graphStatus SuccessCreateOutput(const ge::FastNode *node, KernelContext *run_context) {
    return ge::GRAPH_SUCCESS;
  }

  bool always_return_success_kernel = false;
  std::set<std::string> always_return_success_kernel_excluded;
};
class KernelStub {
 public:
  void AllKernelRegisteredAndSuccess(const std::set<std::string>& exclude_types = {});
  bool SetUp(const std::string &kernel_name, KernelRegistry::KernelFunc);
  KernelStub &SetUp(const std::string &kernel_name, const KernelRegistry::KernelFuncs &funcs);
  void Clear();
  KernelStub &StubTiling();
  KernelStub &StubGenerateSqeAndLaunchTask();
  KernelStub &StubTilingWithCustomTiling(KernelRegistry::KernelFunc stub_run_func);
  KernelStub &StubCalcDvppWorkSpaceSize();
  KernelStub &StubAllocDvppWorkSpaceMem();
  ~KernelStub();

 private:
  void StubWhenNeed();

 private:
  std::shared_ptr<StubKernelRegistry> stub_registry_ = nullptr;
};
}  // namespace gert

#endif
