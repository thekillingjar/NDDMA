/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_ADAPTION_COMPLETE_ASC_IO_INDEX_H
#define CANN_GRAPH_ENGINE_ADAPTION_COMPLETE_ASC_IO_INDEX_H

#include "common/checker.h"
#include "post_process/post_process_util.h"
#include "ascir_ops.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace asc_adapt {
inline Status SetDataIndex(const NodePtr &data_node, int64_t index) {
  GE_ASSERT_NOTNULL(data_node);
  GE_ASSERT_NOTNULL(data_node->GetOpDesc());
  const auto &attr = data_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  auto *data_attr = dynamic_cast<AscDataIrAttrDef *>(attr->ir_attr.get());
  GE_ASSERT_NOTNULL(data_attr);
  GE_ASSERT_SUCCESS(data_attr->SetIndex(index));
  GELOGD("node: %s(%s) set data index=%ld.", data_node->GetNamePtr(), data_node->GetType().c_str(), index);
  return SUCCESS;
}

inline Status SetOutputIndex(const NodePtr &output_node, int64_t index) {
  GE_ASSERT_NOTNULL(output_node);
  GE_ASSERT_NOTNULL(output_node->GetOpDesc());
  const auto &attr = output_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  auto *output_attr = dynamic_cast<ascir_op::Output::AscOutputIrAttrDef *>(attr->ir_attr.get());
  GE_ASSERT_NOTNULL(output_attr);
  GE_ASSERT_SUCCESS(output_attr->SetIndex(index));
  GELOGD("node: %s(%s) set output index=%ld.", output_node->GetNamePtr(), output_node->GetType().c_str(), index);
  return SUCCESS;
}

inline Status CompleteAscIoIndex(AscGraph &asc_graph, const NodePtr &asc_node) {
  // asc graph输入输出index值对应设置
  int64_t index = -1;
  auto input_nodes = asc_graph.GetInputNodes();
  for (const auto &input_node : input_nodes) {
    if (input_node->GetType() != kDataType) {
      continue;
    }
    // data添加index
    GE_ASSERT_SUCCESS(SetDataIndex(input_node, ++index));
  }
  auto &inter_attrs = GetInterAttrs(BackendUtils::GetNodeAutoFuseAttr(asc_node));
  auto output_nodes = inter_attrs.fused_subgraph_outputs;

  // Eliminate Duplicates
  RemoveDuplicates(output_nodes);

  index = -1;
  for (const auto &output_node : output_nodes) {
    if (output_node->GetType() != kOutputType) {
      continue;
    }
    // output添加index
    GE_ASSERT_SUCCESS(SetOutputIndex(output_node, ++index));
  }
  return SUCCESS;
}

inline Status CompleteIndexAttrForAscGraph(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, CompleteAscIoIndex, "complete_asc_io_index"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // CANN_GRAPH_ENGINE_ADAPTION_COMPLETE_ASC_IO_INDEX_H
