/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_v2_simple_executor.h"

#include "runtime/model_v2_executor.h"
#include "lowering/model_converter.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"

namespace gert {
std::unique_ptr<RtV2SimpleExecutor> RtV2SimpleExecutor::Create(const ge::GeRootModelPtr &model, RtSession *session) {
  return Create(model, nullptr, nullptr, nullptr, session);
}

std::unique_ptr<RtV2SimpleExecutor> RtV2SimpleExecutor::Create(const ge::GeRootModelPtr &model,
                                                               ge::DevResourceAllocator &allocator,
                                                               RtSession *session) {
  return Create(model, &allocator.stream_allocator, &allocator.event_allocator, &allocator.notify_allocator, session);
}

std::unique_ptr<RtV2SimpleExecutor> RtV2SimpleExecutor::Create(const ge::GeRootModelPtr &model,
                                                               StreamAllocator *const stream_allocator,
                                                               EventAllocator *const event_allocator,
                                                               NotifyAllocator *const notify_allocator,
                                                               RtSession *session) {
  ModelConverter::Args args{{}, stream_allocator, event_allocator, notify_allocator, nullptr};
  auto graph =
      ModelConverter().ConvertGeModelToExecuteGraph(model, args);
  GE_ASSERT_NOTNULL(graph, "Failed convert ge model of graph %s", model->GetRootGraph()->GetName().c_str());

  const char_t *max_runtime_core_num = nullptr;
  MM_SYS_GET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, max_runtime_core_num);
  int32_t max_core_num;
  if (max_runtime_core_num != nullptr) {
    GE_ASSERT_SUCCESS(ge::ConvertToInt32(std::string(max_runtime_core_num), max_core_num));
  } else {
    max_core_num = 1;
  }
  std::unique_ptr<ModelV2Executor> executor = nullptr;
  GELOGD("max runtime core number is %d", max_core_num);
  if (max_core_num < static_cast<int32_t>(kLeastThreadNumber)) {
    executor = gert::ModelV2Executor::Create(graph, model, session);
  } else {
    MultiThreadExecutorOption option(ExecutorType::kTopologicalMultiThread, max_core_num);
    executor = gert::ModelV2Executor::Create(graph, option, model, session);
  }
  GE_ASSERT_NOTNULL(executor, "Failed create rt2 executor for exec graph %s", graph->GetName().c_str());
  return std::unique_ptr<RtV2SimpleExecutor>(new (std::nothrow)
                                                 RtV2SimpleExecutor(graph->GetName(), std::move(executor)));
}
}  // namespace gert
