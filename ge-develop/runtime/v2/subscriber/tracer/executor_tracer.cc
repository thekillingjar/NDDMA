/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_tracer.h"
#include "core/execution_data.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "core/builder/node_types.h"
#include "core/executor_error_code.h"
#include "runtime/model_v2_executor.h"
#include "graph/def_types.h"
#include "runtime/dev.h"
#include "register/kernel_registry.h"
#include "framework/runtime/subscriber/global_tracer.h"

namespace gert {
namespace {
constexpr size_t kConstLogHeaderLen = 150U;

#define LOG_BY_RESULT(result, fmt, ...)                                  \
  do {                                                                   \
    if ((result == kStatusSuccess) || (result == ge::END_OF_SEQUENCE)) { \
      GELOGI(fmt, ##__VA_ARGS__);                                        \
    } else {                                                             \
      GELOGE(ge::FAILED, fmt, ##__VA_ARGS__);                            \
    }                                                                    \
  } while (false)
}  // namespace

ExecutorTracer::ExecutorTracer(const std::shared_ptr<const SubscriberExtendInfo> &extend_info) {
  (void)extend_info;
}

void ExecutorTracer::OnExecuteEvent(int32_t sub_exe_graph_type, const ExecutorTracer * const tracer,
                                    ExecutorEvent event, const void *node, KernelStatus result) {
  (void)sub_exe_graph_type;
  if ((GlobalTracer::GetInstance()->GetEnableFlags() == 0U) || (tracer == nullptr)) {
    return;
  }
  if (event == kModelStart) {
    GELOGI("[KernelTrace] Start model tracing.");
    return;
  }
  if (event == kModelEnd) {
    GELOGI("[KernelTrace] End model tracing.");
    return;
  }
  if (event != kExecuteEnd || node == nullptr) {
    return;
  }
  auto func = tracer->GetTracePrinter(reinterpret_cast<const Node *>(node));
  auto ct = reinterpret_cast<const KernelContext *>(&reinterpret_cast<const Node *>(node)->context);
  if (ct == nullptr) {
    GELOGE(ge::FAILED, "context is nullptr");
    return;
  }
  if (func == nullptr) {
    LOG_BY_RESULT(result, "%s", tracer->GetHeaderMsg(ct).c_str());
  } else {
    auto msgs = func(ct);
    const auto header_len = tracer->GetHeaderMsg(ct).size() + kConstLogHeaderLen;
    size_t max_log_len = MSG_LENGTH - header_len;
    for (const auto &msg : msgs) {
      auto msg_length = msg.size();
      size_t current_index = 0U;
      while (current_index < msg_length) {
        auto sub_str = msg.substr(current_index, max_log_len);
        LOG_BY_RESULT(result, "%s%s", tracer->GetHeaderMsg(ct).c_str(), sub_str.c_str());
        current_index += max_log_len;
      }
    }
  }
}

KernelRegistry::TracePrinter ExecutorTracer::GetTracePrinter(const Node *const node) const {
  const auto kernel_extend_info = reinterpret_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel_extend_info);
  auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel_extend_info->GetKernelType());
  if (kernel_funcs == nullptr) {
    return nullptr;
  }
  return kernel_funcs->trace_printer;
}

std::string ExecutorTracer::GetHeaderMsg(const KernelContext *const context) {
  std::stringstream ss;
  ss << "[KernelTrace][" << reinterpret_cast<const ExtendedKernelContext *>(context)->GetKernelName() << "]";
  return ss.str();
}
} // namespace gert
