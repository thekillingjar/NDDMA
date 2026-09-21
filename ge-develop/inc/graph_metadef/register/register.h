/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_REGISTRY_H_
#define INC_REGISTER_REGISTRY_H_

#include "include/register/register.h"
#include "graph/types.h"

namespace ge {
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY HostCpuOp {
 public:
  HostCpuOp() = default;
  HostCpuOp(HostCpuOp &&) = delete;
  HostCpuOp &operator=(HostCpuOp &&) & = delete;
  virtual ~HostCpuOp() = default;
  virtual graphStatus Compute(Operator &op,
                              const std::map<std::string, const ge::Tensor> &inputs,
                              std::map<std::string, ge::Tensor> &outputs) = 0;

 private:
  HostCpuOp(const HostCpuOp &) = delete;
  HostCpuOp &operator=(const HostCpuOp &) & = delete;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY HostCpuOpRegistrar {
 public:
  HostCpuOpRegistrar(const char_t *const op_type, HostCpuOp *(*const create_fn)());
  ~HostCpuOpRegistrar() = default;
};
} // namespace ge

#define REGISTER_HOST_CPU_OP_BUILDER(name, op) \
    REGISTER_HOST_CPU_OP_BUILDER_UNIQ_HELPER(__COUNTER__, name, op)

#define REGISTER_HOST_CPU_OP_BUILDER_UNIQ_HELPER(ctr, name, op) \
    REGISTER_HOST_CPU_OP_BUILDER_UNIQ(ctr, name, op)

#define REGISTER_HOST_CPU_OP_BUILDER_UNIQ(ctr, name, op)              \
  static ::ge::HostCpuOpRegistrar register_host_cpu_op##ctr           \
      __attribute__((unused)) =                                       \
          ::ge::HostCpuOpRegistrar((name), []()->::ge::HostCpuOp* {   \
            return new (std::nothrow) (op)();                         \
          })

#endif // INC_REGISTER_REGISTRY_H_
