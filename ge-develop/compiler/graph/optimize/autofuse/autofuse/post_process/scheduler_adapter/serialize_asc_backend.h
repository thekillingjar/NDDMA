/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_SERIALIZE_ASC_BACKEND_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_SERIALIZE_ASC_BACKEND_H_
#include "common/checker.h"
#include "utils/autofuse_utils.h"
#include "graph/utils/hash_utils.h"
#include "autofuser.h"
#include "can_fuse/backend/backend_utils.h"

namespace ge {
namespace asc_adapt {
inline Status SerilizeAscBackendNode(const ComputeGraphPtr &graph) {
  ComputeGraphPtr copy_graph;
  int32_t index = -1;
  for (const auto &node : graph->GetAllNodes()) {
    index++;
    if (!BackendUtils::IsBackendFuseNode(node)) {
      continue;
    }
    if (copy_graph == nullptr) {
      // 存在需要序列化节点时再拷贝子图
      GE_ASSERT_SUCCESS(AutofuseUtils::CreateComputeGraphWithGraphID(graph, "HashCopyGraph", copy_graph));
      // hash表示需要先拷贝图替换掉名字
      GE_ASSERT_SUCCESS(AutofuseUtils::CopyGraphAndRenameNode(graph, copy_graph, nullptr));
    }
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_ptr = node.get();
    GE_ASSERT_TRUE(op_desc->SetExtAttr(
        "_extra_param_builder", std::function<std::string()>([node_ptr]() -> std::string {
          std::string output;
          if (AutofuseUtils::SerilizeAscBackend(node_ptr, output) != SUCCESS) {
            GELOGE(FAILED, "node:%s(%s) serilize failed.", node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
            return std::string("");
          }
          return output;
        })));
    GE_ASSERT_TRUE(static_cast<size_t>(index) < copy_graph->GetAllNodes().size(), "index %d is over nodes size %zu",
                   index, copy_graph->GetAllNodes().size());
    // 当前改名前和改名后的映射关系通过ComputeGraph.GetAllNodes()保证顺序
    auto copy_node = copy_graph->GetAllNodes().at(index);
    auto copy_node_ptr = copy_node.get();
    std::string output;
    if (AutofuseUtils::SerilizeAscBackend(copy_node_ptr, output, true) != SUCCESS) {
      GELOGE(FAILED, "node:%s(%s) hash failed.", copy_node_ptr->GetName().c_str(), copy_node_ptr->GetType().c_str());
      output = "";
    }
    GE_ASSERT_TRUE(op_desc->SetExtAttr("_hashed_extra_param_builder", output));
    GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, "hashed_extra_param", output));
  }
  if (copy_graph != nullptr) {
    AutofuseUtils::DumpGraphToOnnx(*graph, kPostProcessDir, graph->GetName() + "_serialize_hash");
  }
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_SERIALIZE_ASC_BACKEND_H_