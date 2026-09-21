/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MODEL_KERBEL_HANDLES_MANAGER_H
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MODEL_KERBEL_HANDLES_MANAGER_H

#include <memory>
#include <unordered_map>
#include "common/kernel_handles_manager/aicpu_kernel_handles_manager.h"
#include "common/kernel_handles_manager/aicore_kernel_handles_manager.h"
#include "common/kernel_handles_manager/cust_aicpu_kernel_handles_manager.h"
#include "common/util/mem_utils.h"

namespace ge {
enum class KernelHandleType : int32_t {
  kAicore,
  kAicpu,
  kCustAicpu
};
      
using KernelHandlesManagerPtr = std::shared_ptr<KernelHandlesManager>;
class ModelKernelHandlesManager {
 public:
 ModelKernelHandlesManager() = default;
  ~ModelKernelHandlesManager() = default;
  KernelHandlesManagerPtr GetKernelHandle(KernelHandleType kernel_type) const;
  graphStatus ClearAllHandle();
 private:
  std::unordered_map<KernelHandleType, KernelHandlesManagerPtr> kernel_handles_{
      {KernelHandleType::kAicore, MakeShared<AicoreKernelHandlesManager>()},
      {KernelHandleType::kCustAicpu, MakeShared<CustAicpuKernelHandlesManager>()},
      {KernelHandleType::kAicpu, MakeShared<AicpuKernelHandlesManager>()}};
};
}

#endif // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MODEL_KERBEL_HANDLES_MANAGER_H