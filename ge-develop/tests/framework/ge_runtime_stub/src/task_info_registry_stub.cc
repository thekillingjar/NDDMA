/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/task_info_registry_stub.h"
namespace ge {
void TaskInfoRegistryStub::StubRegistryIfNeed() {
  if (registry_ == nullptr) {
    registry_ = std::shared_ptr<TaskInfoFactory>(new TaskInfoFactory(), TaskInfoRegistryStub::TaskInfoFactoryDeleter);
    TaskInfoFactory::Replace(registry_);
  }
}
void TaskInfoRegistryStub::TaskInfoFactoryDeleter(TaskInfoFactory *ins) {
  delete ins;
}
TaskInfoRegistryStub::~TaskInfoRegistryStub() {
  TaskInfoFactory::Replace(nullptr);
}
TaskInfoRegistryStub &TaskInfoRegistryStub::StubTaskInfo(ModelTaskType type,
                                                         TaskInfoFactory::TaskInfoCreatorFun creator) {
  StubRegistryIfNeed();
  registry_->RegisterCreator(type, creator);
  return *this;
}
TaskInfoRegistryStub &TaskInfoRegistryStub::StubTaskInfo(ModelTaskType type,
                                                         std::shared_ptr<TaskInfo> task_info_ins) {
  return StubTaskInfo(type, [ins = std::move(task_info_ins)]() { return ins; });
}
}  // namespace ge
