/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gen_api_tiling.h"
#include <set>
#include "graph/node.h"
#include "common/checker.h"
#include "base/base_types.h"
#include "api_tiling_gen/api_tiling_gen_register.h"

namespace att {
ge::Status GetApiTilingInfo(const uint32_t tiling_case_id, const ApiTilingParams &params,
                            std::map<std::string, NodeApiTilingCode> &node_name_to_api_code) {
  // 遍历图上的节点，判断节点是否需要tiling
  for (const auto &node : params.graph.GetAllNodes()) {
    // 判断节点是否需要tiling
    const auto node_type = node->GetType();
    GE_ASSERT_NOTNULL(node, "Get graph node failed.");
    // 优先注册自动融合场景的高阶API
    if (ApiTilingGenRegistry::Instance().IsApiTilingRegistered(node_type)) {
      AutofuseApiTilingGenerator generator(params.graph, node, params.tiling_data_type, tiling_case_id);
      GE_ASSERT_SUCCESS(generator.Generate(),
                        "Generate api tiling code failed, graph[%s], node[%s] tiling data type[%s]",
                        params.graph.GetName().c_str(), node->GetName().c_str(), params.tiling_data_type.c_str());
      node_name_to_api_code[node->GetName()].function_invoke = generator.GetFuncInvoke();
      node_name_to_api_code[node->GetName()].function_impl = generator.GetFuncImpl();
      node_name_to_api_code[node->GetName()].head_files = generator.GetHeadFiles();
      continue;
    }
  }
  return ge::SUCCESS;
}
}  // namespace att
