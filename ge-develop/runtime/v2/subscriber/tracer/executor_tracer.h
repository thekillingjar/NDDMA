/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_EXECUTOR_TRACER_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_EXECUTOR_TRACER_H_

#include <kernel/kernel_log.h>
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "exe_graph/runtime/context_extend.h"
#include "core/execution_data.h"
#include "register/kernel_registry.h"

namespace gert {
class ExecutorTracer {
 public:
  explicit ExecutorTracer(const std::shared_ptr<const SubscriberExtendInfo> &extend_info);
  static void OnExecuteEvent(int32_t sub_exe_graph_type, const ExecutorTracer * const tracer, ExecutorEvent event,
                             const void *node, KernelStatus result);
  static std::string GetHeaderMsg(const KernelContext *const context);
 private:
  KernelRegistry::TracePrinter GetTracePrinter(const Node *const node) const;
};
} // namespace gert
#endif // AIR_CXX_RUNTIME_V2_SUBSCRIBER_EXECUTOR_TRACER_H_
