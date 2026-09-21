/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_kernel_registry.h"
#include <mutex>
#include <map>
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/ge_common/ge_inner_error_codes.h"

namespace ge {
class OpKernelRegistry::OpKernelRegistryImpl {
 public:
  void RegisterHostCpuOp(const std::string &op_type, const OpKernelRegistry::CreateFn create_fn) {
    const std::lock_guard<std::mutex> lock(mu_);
    create_fns_[op_type] = create_fn;
  }

  OpKernelRegistry::CreateFn GetCreateFn(const std::string &op_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    const auto it = create_fns_.find(op_type);
    if (it == create_fns_.end()) {
      return nullptr;
    }

    return it->second;
  }

 private:
  std::mutex mu_;
  std::map<std::string, OpKernelRegistry::CreateFn> create_fns_;
};

OpKernelRegistry::OpKernelRegistry() {
  impl_ = ge::ComGraphMakeUnique<OpKernelRegistryImpl>();
}

OpKernelRegistry::~OpKernelRegistry() = default;

OpKernelRegistry& OpKernelRegistry::GetInstance() {
  static OpKernelRegistry instance;
  return instance;
}

bool OpKernelRegistry::IsRegistered(const std::string &op_type) const {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED,
           "[Check][Param:impl_]Failed to invoke IsRegistered %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return false;
  }

  return impl_->GetCreateFn(op_type) != nullptr;
}

void OpKernelRegistry::RegisterHostCpuOp(const std::string &op_type, const CreateFn create_fn) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED,
           "[Check][Param:impl_]Failed to register %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return;
  }

  impl_->RegisterHostCpuOp(op_type, create_fn);
}
std::unique_ptr<HostCpuOp> OpKernelRegistry::CreateHostCpuOp(const std::string &op_type) const {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED,
           "[Check][Param:impl_]Failed to create op for %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return nullptr;
  }

  const auto create_fn = impl_->GetCreateFn(op_type);
  if (create_fn == nullptr) {
    GELOGD("Host Cpu op is not registered. op type = %s", op_type.c_str());
    return nullptr;
  }

  return std::unique_ptr<HostCpuOp>(create_fn());
}

HostCpuOpRegistrar::HostCpuOpRegistrar(const char_t *const op_type, HostCpuOp *(*const create_fn)()) {
  if (op_type == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param:op_type]is null,Failed to register host cpu op");
    return;
  }

  OpKernelRegistry::GetInstance().RegisterHostCpuOp(op_type, create_fn);
}
} // namespace ge
