/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_kernel_handles_manager.h"
#include "common/checker.h"

namespace ge {
KernelHandlesManagerPtr ModelKernelHandlesManager::GetKernelHandle(KernelHandleType kernel_type) const {
  auto iter = kernel_handles_.find(kernel_type);
  GE_ASSERT_TRUE(iter != kernel_handles_.end(), "Invalid kernel type: %d", kernel_type);
  return iter->second;
}

graphStatus ModelKernelHandlesManager::ClearAllHandle() {
  for (auto &kernel_handle_iter : kernel_handles_) {
    GE_ASSERT_NOTNULL(kernel_handle_iter.second);
    GE_ASSERT_SUCCESS(kernel_handle_iter.second->ClearKernel());
  }
  return SUCCESS;
}
}