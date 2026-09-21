/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_EXECUTOR_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_EXECUTOR_H_
#include "graph/ge_error_codes.h"

#include "common/ge_visibility.h"
#include "exe_graph_resource_guard.h"
#include "subscriber/executor_subscriber_c.h"
namespace gert {
enum SubExeGraphType { kInitExeGraph, kMainExeGraph, kDeInitExeGraph, kSubExeGraphTypeEnd };
using ResourceGuardPtr = std::unique_ptr<ResourceGuard>;
class VISIBILITY_EXPORT ExeGraphExecutor {
 public:
  using ExecuteFunc = UINT32 (*)(void *);
  using ExecuteWithCallbackFunc = UINT32 (*)(int32_t, void *, ExecutorSubscriber *);
  ge::graphStatus Load() const {
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus UnLoad() const {
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 设置图执行的输入/输出，需要注意的是，使用者需要自己保证inputs/outputs刷新完全！！！
   */
  ge::graphStatus SpecifyInputs(void *const *inputs, size_t start, size_t num) const;
  ge::graphStatus SpecifyInput(void *input, size_t start) const {
    return SpecifyInputs(&input, start, 1U);
  }
  ge::graphStatus SpecifyOutputs(void *const *outputs, size_t num) const;
  ge::graphStatus Execute() const;
  ge::graphStatus Execute(SubExeGraphType sub_graph_type, ExecutorSubscriber *callback) const;

  const void *GetExecutionData() const {
    return execution_data_;
  }
  void SetExecutionData(void *execution_data, ResourceGuardPtr resource_guard);

  void SetExecuteFunc(ExecuteFunc execute_func, ExecuteWithCallbackFunc callback_func);

 private:
  friend class ModelV2ExecutorTestHelper;

  void *execution_data_{nullptr};
  ExecuteFunc execute_func_{nullptr};
  ExecuteWithCallbackFunc execute_with_callback_func_{nullptr};
  ResourceGuardPtr resource_guard_;
};
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_EXECUTOR_H_
