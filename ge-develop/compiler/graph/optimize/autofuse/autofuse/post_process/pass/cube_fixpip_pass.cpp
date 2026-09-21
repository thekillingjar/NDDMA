/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_fixpip_pass.h"
#include <string>
#include <set>
#include "ge_common/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "utils/autofuse_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/utils/graph_utils.h"
#include "post_process/post_process_util.h"
#include "utils/autofuse_utils.h"

namespace ge {
namespace {
Status SetReluAttr(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  if (node->GetType() == kMatMul) {
    auto mm_attr = dynamic_cast<ascir_op::MatMul::AscMatMulIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else if (node->GetType() == kMatMulBias) {
    auto mm_attr = dynamic_cast<ascir_op::MatMulBias::AscMatMulBiasIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else if (node->GetType() == kMatMulOffset) {
    auto mm_attr = dynamic_cast<ascir_op::MatMulOffset::AscMatMulOffsetIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else if (node->GetType() == kMatMulOffsetBias) {
    auto mm_attr = dynamic_cast<ascir_op::MatMulOffsetBias::AscMatMulOffsetBiasIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else if (node->GetType() == kBatchMatMul) {
    auto mm_attr = dynamic_cast<ascir_op::BatchMatMul::AscBatchMatMulIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else if (node->GetType() == kBatchMatMulBias) {
    auto mm_attr = dynamic_cast<ascir_op::BatchMatMulBias::AscBatchMatMulBiasIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else if (node->GetType() == kBatchMatMulOffset) {
    auto mm_attr = dynamic_cast<ascir_op::BatchMatMulOffset::AscBatchMatMulOffsetIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else if (node->GetType() == kBatchMatMulOffsetBias) {
    auto mm_attr = dynamic_cast<ascir_op::BatchMatMulOffsetBias::AscBatchMatMulOffsetBiasIrAttrDef *>(attr->ir_attr.get());
    GE_ASSERT_NOTNULL(mm_attr);
    GE_ASSERT_SUCCESS(mm_attr->SetHas_relu(1));
  } else {
    // nop
  }
  GELOGD("Set relu for node(%s %s) succ.", node->GetName().c_str(), node->GetType().c_str());
  return SUCCESS;
}

Status CubeFixpip(AscGraph &graph, const NodePtr &asc_node) {
  if (!BackendUtils::IsCubeAscNode(asc_node)) {
    GELOGI("node %s(%s) not cube type, fixpip skip.", asc_node->GetNamePtr(), asc_node->GetType().c_str());
    return SUCCESS;
  }

  bool is_relu_only = false;
  NodePtr cube_node = nullptr;
  for (const auto &node : graph.GetAllNodes()) {
    if (AutofuseUtils::IsCubeNodeType(node)) {
      std::vector<NodePtr> peer_in_nodes;
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerInNodes(node, peer_in_nodes, 0));

      if (peer_in_nodes.size() != 1U) {
        GELOGI("node:%s(%s) has %zu peer out nodes, skip.", node->GetName().c_str(), node->GetType().c_str(),
               peer_in_nodes.size());
        return SUCCESS;
      }
      auto peer_in_node = peer_in_nodes.at(0);
      if (peer_in_node->GetType() != kReluType) {
        GELOGI("node:%s(%s) peer node not relu, skip.", node->GetName().c_str(), node->GetType().c_str());
        return SUCCESS;
      }
      // 当前不需要删除relu节点，以防遇到不支持fixpip的tiling key需要走ub模板 GE_ASSERT_SUCCESS(asc_adapt::DelNode(graph, peer_in_node));
      // 后续需要删除relu节点时在此处设置relu标记，不在下方is_relu_only分支设置GE_ASSERT_SUCCESS(SetReluAttr(node));
      is_relu_only = true;
      cube_node = node;
    } else if (node->GetType() != kDataType && node->GetType() != kLoadType && node->GetType() != kStoreType &&
               node->GetType() != kOutputType && node->GetType() != kReluType) {
      GELOGI("node:%s(%s) is not relu or input/output node, skip.", node->GetName().c_str(), node->GetType().c_str());
      return SUCCESS;
    }
  }
  if (is_relu_only) {
    GE_ASSERT_SUCCESS(SetReluAttr(cube_node));
  }

  // 没有删除节点不需要做topo排序asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(graph));
  return SUCCESS;
}
}

Status CubeFixpipPass::Run(const ComputeGraphPtr &graph) const {
  // 代码预埋，act支持所有模板的fixpip能力后，开启此功能,后面要fixpip能力要注意act是否支持fixpip+后融合elemetnwise混合
  // 当前只支持纯matmul+relu无别的节点且matmul不输出多引用的场景标记relu，不删除relu节点，后端直接选relu的matmul模板，不走ub模板
  GE_ASSERT_SUCCESS(asc_adapt::ProcessAscBackendNodes(graph, CubeFixpip, "cube_fixpip_pass"));
  GELOGI("Graph %s completed CubeFixpipPass successfully.", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
