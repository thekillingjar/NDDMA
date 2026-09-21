/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_host_inputs_fuse_pass.h"

#include <queue>
#include <stack>
#include "common/checker.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "common/util/mem_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "runtime/model_v2_executor.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "engine/aicpu/kernel/aicpu_args_handler.h"
#include "engine/aicpu/kernel/aicpu_fuse_host_inputs.h"
#include "engine/aicpu/kernel/aicpu_resource_manager.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace bg {
namespace {
bool IsSingleMakeTensorDevice(ge::FastNode *const node, ge::FastNode *const dst_node) {
  if (node->GetType() != kernel::kMakeSureTensorAtDevice) {
    return false;
  }

  for (const auto &out_data_edges : node->GetAllOutDataEdgesRef()) {
    for (const auto out_edge : out_data_edges) {
      if (out_edge != nullptr) {
        const auto out_node = out_edge->dst;
        if ((out_node->GetName() != dst_node->GetName()) && (out_node->GetType() != "FreeMemory")) {
          GELOGD("Node[%s] has mulit dst nodes, such as %s", node->GetName().c_str(), dst_node->GetName().c_str());
          return false;
        }
      }
    }
  }

  return true;
}

ge::FastNode *CreateConstNode(ge::ExecuteGraph *const graph, const std::string &node_name, const void *data,
                              size_t size, bool is_string) {
  auto const_op_desc = ge::MakeShared<ge::OpDesc>(node_name, "Const");
  GE_ASSERT_NOTNULL(const_op_desc);
  GE_ASSERT_SUCCESS(const_op_desc->AddOutputDesc(ge::GeTensorDesc()));
  auto node = graph->AddNode(const_op_desc);
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_SUCCESS(node->GetOpDescBarePtr()->SetAttr("is_string", ge::AnyValue::CreateFrom(is_string)));
  GE_ASSERT_TRUE(ge::AttrUtils::SetZeroCopyBytes(node->GetOpDescBarePtr(), kConstValue,
                                                 ge::Buffer::CopyFrom(ge::PtrToPtr<void, uint8_t>(data), size)));
  return node;
}

ge::graphStatus AddAicpuArgsHandlerInput(ge::FastNode *const fuse_node, ge::FastNode *const dst_node) {
  auto owner_graph = dst_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  auto args_handler_edge = dst_node->GetInDataEdgeByIndex(0);
  GE_CHECK_NOTNULL(args_handler_edge);
  auto src_node = args_handler_edge->src;
  auto src_index = args_handler_edge->src_output;
  GE_ASSERT_NOTNULL(owner_graph->AddEdge(src_node, src_index, fuse_node, 0));
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus AicpuHostInputsFusePass::ReplaceFuseHostInputsNode(ge::FastNode *const fuse_node) {
  int32_t input_anchor_index = static_cast<int32_t>(kernel::AicpuFuseHostInputs::kStream);
  int32_t output_data_index = 0;
  auto graph = fuse_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);

  for (const auto proc_node : host_inputs_proc_nodes_) {
    uint32_t proc_node_anchor_index = 0;
    for (const auto in_data_edge : proc_node->GetAllInDataEdgesRef()) {
      if (in_data_edge == nullptr) {
        continue;
      }
      auto in_data_node = in_data_edge->src;
      auto in_data_node_index = in_data_edge->src_output;
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(in_data_edge));
      GE_ASSERT_TRUE(input_anchor_index < static_cast<int32_t>(fuse_node->GetDataInNum()));

      if ((input_anchor_index < static_cast<int32_t>(kernel::AicpuFuseHostInputs::kAddrAndLengthStart)) ||
          (proc_node_anchor_index >= static_cast<int32_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart))) {
        GE_ASSERT_NOTNULL(graph->AddEdge(in_data_node, in_data_node_index, fuse_node, input_anchor_index++));
      }
      ++proc_node_anchor_index;
    }

    for (const auto &out_data_edges : proc_node->GetAllOutDataEdgesRef()) {
      for (const auto out_edge : out_data_edges) {
        if (out_edge != nullptr) {
          auto out_data_node = out_edge->dst;
          auto out_data_node_index = out_edge->dst_input;
          GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(out_edge));
          GE_ASSERT_TRUE(output_data_index < static_cast<int32_t>(fuse_node->GetDataOutNum()));
          GE_ASSERT_NOTNULL(graph->AddEdge(fuse_node, output_data_index, out_data_node, out_data_node_index));
        }
      }
      ++output_data_index;
    }

    const auto &proc_node_name = proc_node->GetName();
    if (ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph, proc_node_name.c_str()) != nullptr) {
      GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::MoveInCtrlEdges(proc_node, fuse_node));
      GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::MoveOutCtrlEdges(proc_node, fuse_node));
      GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(graph, proc_node));
    }
  }
  GE_ASSERT_TRUE(host_inputs_addr_index_.size() == fuse_node->GetDataOutNum());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuHostInputsFusePass::AddFuseIndexesInput(ge::FastNode *const fuse_node) const {
  const auto graph = fuse_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);

  std::string input_index_name = "Const_" + fuse_node->GetName() + "_Index";
  const size_t input_num = host_inputs_addr_index_.size();
  auto input_index_node =
      CreateConstNode(graph, input_index_name, host_inputs_addr_index_.data(), input_num * sizeof(int32_t), true);
  GE_CHECK_NOTNULL(input_index_node);
  GE_ASSERT_NOTNULL(graph->AddEdge(input_index_node, 0, fuse_node,
                                       static_cast<int32_t>(kernel::AicpuFuseHostInputs::kInputsIndex)));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuHostInputsFusePass::AddFuseOpOutputDesc(const ge::OpDescPtr &fuse_desc) const {
  // each proc node output tensor_data
  for (auto &proc_node : host_inputs_proc_nodes_) {
    auto proc_op_desc = proc_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(proc_op_desc);
    const uint32_t proc_node_output_num = proc_node->GetDataOutNum();
    for (size_t i = 0; i < proc_node_output_num; ++i) {  // may has multi output
      GE_ASSERT_SUCCESS(fuse_desc->AddOutputDesc(proc_op_desc->GetOutputDesc(i)));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuHostInputsFusePass::AddFuseOpInputDesc(const ge::OpDescPtr &fuse_desc,
                                                            const ge::OpDescPtr &dst_desc) const {
  GE_ASSERT_SUCCESS(fuse_desc->AddInputDesc(dst_desc->GetInputDesc(0)));  // AicpuArgsHandler
  GE_ASSERT_SUCCESS(fuse_desc->AddInputDesc(ge::GeTensorDesc()));         // InputsIndex

  // Stream & Allocator
  // Get from first MakeSureTensorAtDevice node, size has checked not 0.
  auto first_src_op = host_inputs_proc_nodes_[0]->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(first_src_op);
  size_t proc_node_addr_start = static_cast<size_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart);
  for (size_t i = 0U; i < proc_node_addr_start; ++i) {
    GE_ASSERT_SUCCESS(fuse_desc->AddInputDesc(first_src_op->GetInputDesc(i)));
  }

  // each proc node has tensor_data, tensor_size, storage_shape, data_type as input
  for (const auto proc_node : host_inputs_proc_nodes_) {
    auto proc_op_desc = proc_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(proc_op_desc);
    const uint32_t proc_node_input_num = proc_node->GetDataInNum();
    for (size_t i = proc_node_addr_start; i < proc_node_input_num; ++i) {
      GE_ASSERT_SUCCESS(fuse_desc->AddInputDesc(proc_op_desc->GetInputDesc(i)));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::FastNode *AicpuHostInputsFusePass::CreateFuseHostInputsNode(ge::FastNode *const node) const {
  std::string fused_node_name = node->GetName() + "_FuseHostInputs";
  auto fused_op_desc = ge::MakeShared<ge::OpDesc>(fused_node_name, "AicpuFuseHost");
  GE_ASSERT_NOTNULL(fused_op_desc);
  GE_ASSERT_SUCCESS(AddFuseOpInputDesc(fused_op_desc, node->GetOpDescPtr()));
  GE_ASSERT_SUCCESS(AddFuseOpOutputDesc(fused_op_desc));

  auto graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
  auto fuse_node = graph->AddNode(fused_op_desc);
  GE_ASSERT_SUCCESS(AddAicpuArgsHandlerInput(fuse_node, node));
  GE_ASSERT_SUCCESS(AddFuseIndexesInput(fuse_node));
  return fuse_node;
}

ge::graphStatus AicpuHostInputsFusePass::FuseHostInputsProcNodes(ge::FastNode *const node, bool &changed) {
  if (host_inputs_proc_nodes_.empty() || host_inputs_addr_index_.empty()) {
    GELOGD("Node %s no need fuse host inputs node.", node->GetName().c_str());
    return ge::GRAPH_SUCCESS;
  }

  GELOGD("Start to create Node[%s] fuse host inputs node.  ", node->GetName().c_str());
  auto fuse_node = CreateFuseHostInputsNode(node);
  GE_CHECK_NOTNULL(fuse_node);
  GE_ASSERT_SUCCESS(ReplaceFuseHostInputsNode(fuse_node));
  changed = true;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuHostInputsFusePass::SetHostInputsNodesAndIndex(ge::FastNode *const node) {
  GELOGD("Find update addr node name[%s] host input.", node->GetName().c_str());
  host_inputs_proc_nodes_.clear();
  host_inputs_addr_index_.clear();

  for (const auto src_edge : node->GetAllInDataEdgesRef()) {
    if (src_edge == nullptr) {
      continue;
    }
    auto src_node = src_edge->src;

    if (IsSingleMakeTensorDevice(src_node, node)) {
      host_inputs_proc_nodes_.emplace_back(src_node);
      for (const auto peer_edge : src_node->GetAllOutDataEdges()) {
        // src_node may has multi output
        if (peer_edge->dst->GetName() == node->GetName()) {
          auto anchor_index = peer_edge->dst_input;
          GE_ASSERT_TRUE(anchor_index >= 1);  // skip input of args_handle
          host_inputs_addr_index_.emplace_back(anchor_index - 1);
          GELOGD("Find input index[%d] src node[%s].", anchor_index - 1, src_node->GetName().c_str());
        }
      }
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuHostInputsFusePass::Run(ge::ExecuteGraph *const graph, bool &changed) {
  if (!AicpuResourceManager::GetInstance().IsSingleOp()) {
    GELOGI("[AicpuHostInputsFusePass]: is not single op scene!");
    return ge::GRAPH_SUCCESS;
  }

  auto update_addr_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph, "UpdateAicpuIoAddr");
  for (const auto node : update_addr_nodes) {
    GE_ASSERT_SUCCESS(SetHostInputsNodesAndIndex(node));
    GE_ASSERT_SUCCESS(FuseHostInputsProcNodes(node, changed));
  }
  if (changed) {
    ge::DumpGraph(graph, "AfterAicpuFuseHostInputs");
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
