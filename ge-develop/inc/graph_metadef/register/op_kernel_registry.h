/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OP_KERNEL_REGISTRY_H_
#define INC_REGISTER_OP_KERNEL_REGISTRY_H_
#include <memory>
#include <string>
#include "register/register_types.h"
#include "graph_metadef/register/register.h"

namespace ge {
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpKernelRegistry {
 public:
  using CreateFn = HostCpuOp* (*)();
  ~OpKernelRegistry();

  static OpKernelRegistry& GetInstance();

  bool IsRegistered(const std::string &op_type) const;

  void RegisterHostCpuOp(const std::string &op_type, const CreateFn create_fn);

  std::unique_ptr<HostCpuOp> CreateHostCpuOp(const std::string &op_type) const;

 private:
  OpKernelRegistry();
  class OpKernelRegistryImpl;
  /*lint -e148*/
  std::unique_ptr<OpKernelRegistryImpl> impl_;
};
} // namespace ge

#endif // INC_REGISTER_OP_KERNEL_REGISTRY_H_
