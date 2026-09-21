/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fake_execution_data.h"
#include "core/executor_error_code.h"

namespace gert {
std::mutex KernelSpy::mutex;
std::vector<NodeIdentity> KernelSpy::nodes;

KernelSpy &KernelSpy::GetInstance() {
  static KernelSpy spy;
  return spy;
}

UINT32 KernelSpy::KernelStub(KernelRunContext *context) {
  std::lock_guard lock(KernelSpy::mutex);
  KernelSpy::GetInstance().nodes.push_back(context->input_size);
  return kStatusSuccess;
}

UINT32 KernelSpy::KernelStubFailed(KernelRunContext *context) {
  std::lock_guard lock(KernelSpy::mutex);
  return kStatusFailed;
}

UINT32 KernelSpy::KernelStubEndOfSequence(KernelRunContext *context) {
  std::lock_guard lock(KernelSpy::mutex);
  return ge::END_OF_SEQUENCE;
}

void KernelSpy::Clear() {
  std::lock_guard lock(KernelSpy::mutex);
  KernelSpy::GetInstance().nodes.clear();
}

void TaskRun(TaskPackage &package) {
  for (auto &task : package) {
    task.Execute();
  }
}

}  // namespace gert
