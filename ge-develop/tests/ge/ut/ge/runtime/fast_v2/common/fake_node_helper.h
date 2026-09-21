/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_FAKE_NODE_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_FAKE_NODE_H_
#include "core/execution_data.h"
#include "faker/kernel_run_context_facker.h"
namespace gert {
struct NodeHolder {
  KernelRunContextHolder context;
  Node node;
};
class FakeNodeHelper {
 public:
  static NodeHolder FakeNode(std::string node_name, const std::string &kernel_type, const uint64_t node_id = 1UL) {
    NodeHolder holder;
    holder.context =
        KernelRunContextFaker().NodeName(std::move(node_name)).KernelType(kernel_type).KernelName(kernel_type).Build();
    memcpy(&holder.node.context, holder.context.context, sizeof(*holder.context.context));
    holder.node.node_id = node_id;
    holder.node.func = nullptr;
    return holder;
  }
};
}  // namespace gert
#endif
