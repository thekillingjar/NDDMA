/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_TORCH_ADAPTION_FALLBACK_LOAD_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_TORCH_ADAPTION_FALLBACK_LOAD_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "post_process/post_process_util.h"
#include "adaption_fallback_load.h"

namespace ge {
namespace asc_adapt {
// 根据反推结果进行节点插入是按照info元素反向的顺序进行的，即每一次插入节点都插入在load节点后面，预期靠后的节点先插入
inline Status InsertViewOpNodes(AscGraph &asc_graph, const NodePtr &node, ViewOpAttrInfo &attr_info) {
  GE_ASSERT_SUCCESS(asc_adapt::InsertViewOpBroadcast(asc_graph, nullptr, node, attr_info));
  GE_ASSERT_SUCCESS(asc_adapt::InsertViewOpTranspose(asc_graph, nullptr, node, attr_info));
  return SUCCESS;
}

inline Status FallbackPro(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  if (autofuse_attr->HasFuseType(loop::FuseType::kConcat)) {
    GELOGI("graph %s fuse type is concat, don't fallback.", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  if (autofuse_attr->HasFuseType(loop::FuseType::kCube)) {
    GELOGI("graph %s fuse type is kCube, don't fallback.", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(AscGraphUtils::GetComputeGraph(asc_graph), asc_node));
  for (const auto &node : asc_graph.GetAllNodes()) {
    if ((node->GetType() != kLoadType) && (node->GetType() != kStoreType) && (node->GetType() != kGatherType)) {
      continue;
    }
    // pytorch只有load没有data,canfuse会垂直融合产生带有view算子信息的load,通过load和graph的axis id对比获取view算子信息
    ViewOpAttrInfo attr_info;
    GE_ASSERT_SUCCESS(BackendUtils::PostProBackSteppingViewOp(
        asc_graph, node, attr_info, (node->GetType() == kLoadType) || (node->GetType() == kGatherType)));
    GE_ASSERT_SUCCESS(InsertViewOpNodes(asc_graph, node, attr_info));
  }
  // 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

// torch流程和tf流程后处理的反推统一入口，统一调用FallbackPro，原本的反推改为GeFallbackPro，在tf流程在canfuse前调用
inline Status Fallback(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, FallbackPro, "post_fallback"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_TORCH_ADAPTION_FALLBACK_LOAD_H
