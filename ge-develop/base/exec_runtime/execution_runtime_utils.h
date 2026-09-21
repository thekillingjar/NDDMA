/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_RUNTIME_EXECUTION_RUNTIME_H_
#define BASE_RUNTIME_EXECUTION_RUNTIME_H_

#include <string>

namespace ge {
class ExecutionRuntimeUtils {
 public:
  static bool IsHeterogeneous();

  static void EnableInHeterogeneousExecutor();

  static void EnableGlobalInHeterogeneousExecutor();

  static bool IsInHeterogeneousExecutor();
 private:
  static void UpdateGraphOptions(const std::string &key, const std::string &value);
  thread_local static bool in_heterogeneous_executor_;
  static bool global_in_heterogeneous_executor_;
  };
}  // namespace ge

#endif  // BASE_RUNTIME_EXECUTION_RUNTIME_H_
