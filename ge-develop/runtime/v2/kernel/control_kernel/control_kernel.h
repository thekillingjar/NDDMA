/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_CONTROL_KERNEL_CONTROL_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_CONTROL_KERNEL_CONTROL_KERNEL_H_
#include "exe_graph/runtime/kernel_context.h"
#include "graph/node.h"
#include "graph/ge_error_codes.h"
#include "core/execution_data.h"

namespace gert {
namespace kernel {
struct NotifySignal {
  // node_id+1 and watch_num-1 for skip target receiver
  explicit NotifySignal(NodeIdentity &target, NodeIdentity *watchers, size_t watcher_num)
      : target_receiver(target), receivers(watchers), receiver_num(watcher_num) {}
  NodeIdentity &target_receiver;
  NodeIdentity *receivers;
  size_t receiver_num;
};

ge::graphStatus GenIndexForIf(KernelContext *context);
ge::graphStatus GenIndexForCase(KernelContext *context);
ge::graphStatus SwitchNotifyKernel(KernelContext *context);
ge::graphStatus CreateSwitchNotifyOutputs(const ge::Node *node, KernelContext *context);
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_CONTROL_KERNEL_CONTROL_KERNEL_H_
