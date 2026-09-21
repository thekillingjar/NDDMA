/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_SUBSCRIBER_UTILS_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_SUBSCRIBER_UTILS_H_

#include <limits>
#include "graph/fast_graph/fast_node.h"
#include "utils/execute_graph_utils.h"

namespace gert {
constexpr uint32_t kInvalidProfKernelType = std::numeric_limits<uint32_t>::max();
class SubscriberUtils {
 public:
  static ge::FastNode *FindNodeFromExeGraph(ge::ExecuteGraph *const exe_graph, const std::string &node_name) {
    return ge::ExecuteGraphUtils::FindNodeFromAllNodes(exe_graph, node_name.c_str());
  }

  static bool IsNodeNeedProf(const char *kernel_type);

  static uint32_t GetProfKernelType(const char *kernel_type, bool is_for_additional_info);
};
}
#endif  // AIR_CXX_RUNTIME_V2_SUBSCRIBER_SUBSCRIBER_UTILS_H_
