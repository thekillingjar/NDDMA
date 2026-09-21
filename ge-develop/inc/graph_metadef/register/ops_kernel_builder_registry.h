/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H
#define INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H

#include <memory>
#include "register/register_types.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace ge {
using OpsKernelBuilderPtr = std::shared_ptr<OpsKernelBuilder>;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpsKernelBuilderRegistry {
 public:
  ~OpsKernelBuilderRegistry() noexcept;
  static OpsKernelBuilderRegistry &GetInstance();

  void Register(const std::string &lib_name, const OpsKernelBuilderPtr &instance);

  void Unregister(const std::string &lib_name);

  void UnregisterAll();

  const std::map<std::string, OpsKernelBuilderPtr> &GetAll() const;

 private:
  OpsKernelBuilderRegistry() = default;
  std::map<std::string, OpsKernelBuilderPtr> kernel_builders_;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpsKernelBuilderRegistrar {
 public:
  using CreateFn = OpsKernelBuilder *(*)();
  OpsKernelBuilderRegistrar(const std::string &kernel_lib_name, const CreateFn fn);
  ~OpsKernelBuilderRegistrar() noexcept;

private:
  std::string kernel_lib_name_;
};
}  // namespace ge

#define REGISTER_OPS_KERNEL_BUILDER(kernel_lib_name, builder) \
    REGISTER_OPS_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_lib_name, builder)

#define REGISTER_OPS_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_lib_name, builder) \
    REGISTER_OPS_KERNEL_BUILDER_UNIQ(ctr, kernel_lib_name, builder)

#define REGISTER_OPS_KERNEL_BUILDER_UNIQ(ctr, kernel_lib_name, builder)                         \
  static ::ge::OpsKernelBuilderRegistrar register_op_kernel_builder_##ctr                       \
      __attribute__((unused)) =                                                                 \
          ::ge::OpsKernelBuilderRegistrar((kernel_lib_name), []()->::ge::OpsKernelBuilder* {    \
            return new (std::nothrow) (builder)();                                              \
          })

#endif // INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H
