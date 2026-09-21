/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_out_tensor_zero_copy.h"

#include <queue>
#include <stack>

#include "common/checker.h"
#include "graph/ge_context.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/util/mem_utils.h"

#include "utils/pruner.h"
#include "utils/resource_guarder.h"
#include "lowering/pass_changed_kernels_info.h"
#include "core/builder/node_types.h"
#include "exe_graph/lowering/builtin_node_types.h"
#include "kernel/common_kernel_impl/build_tensor.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "kernel/memory/memory_kernel.h"
#include "kernel/outputs/model_outputs.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"

namespace gert {
namespace bg {
namespace {
bool IsOutputData(const ge::FastNode *const node) {
  return (strcmp(node->GetTypePtr(), "OutputData") == 0);
}
std::pair<ge::FastNode *, int32_t> GetPeerOutNodeAndIndex(const ge::FastNode *const node, int32_t input_index) {
  auto in_data_edge = node->GetInDataEdgeByIndex(input_index);
  GE_ASSERT_NOTNULL(in_data_edge);
  auto src_node = in_data_edge->src;
  return {src_node, in_data_edge->src_output};
}

ge::graphStatus FindAllocNodesInSubgraph(const std::pair<ge::FastNode *, int32_t> &alloc_node_and_index,
                                         std::vector<ge::FastNode *> &inner_alloc_nodes) {
  std::stack<std::pair<ge::FastNode *, int32_t>> s;
  s.push(alloc_node_and_index);
  while (!s.empty()) {
    const auto cond_node_and_index = s.top();
    s.pop();
    const auto cond_node = cond_node_and_index.first;
    const auto op_desc = cond_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    GE_ASSERT_TRUE(!subgraph_names.empty());
    const size_t subgraph_num = subgraph_names.size() - 1U;
    for (size_t i = 0U; i < subgraph_num; ++i) {
      auto subgraph = ge::FastNodeUtils::GetSubgraphFromNode(cond_node, i + 1U);
      GE_ASSERT_NOTNULL(subgraph);
      auto inner_netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(subgraph, "InnerNetOutput");
      GE_ASSERT_NOTNULL(inner_netoutput);
      auto peer_out_node_and_index = GetPeerOutNodeAndIndex(inner_netoutput, cond_node_and_index.second);
      auto alloc_node = peer_out_node_and_index.first;
      GE_ASSERT_NOTNULL(alloc_node);
      if (alloc_node->GetType() == "IdentityAddr") {
        peer_out_node_and_index = GetPeerOutNodeAndIndex(alloc_node, peer_out_node_and_index.second);
        alloc_node = peer_out_node_and_index.first;
        GE_ASSERT_NOTNULL(alloc_node);
      }
      if (IsModelOutZeroCopyEnabledAllocNode(alloc_node->GetTypePtr())) {
        inner_alloc_nodes.emplace_back(alloc_node);
        continue;
      }
      if (IsIfOrCaseType(alloc_node->GetTypePtr())) {
        s.push(peer_out_node_and_index);
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAllAllocNodes(const ge::FastNode *copy_node, std::vector<ge::FastNode *> &alloc_nodes) {
  auto peer_out_node_and_index =
      GetPeerOutNodeAndIndex(copy_node, static_cast<int32_t>(kernel::BuildTensorInputs::kTensorData));
  auto alloc_node = peer_out_node_and_index.first;
  GE_ASSERT_NOTNULL(alloc_node);
  if (alloc_node->GetType() == "IdentityAddr") {
    peer_out_node_and_index = GetPeerOutNodeAndIndex(alloc_node, peer_out_node_and_index.second);
    alloc_node = peer_out_node_and_index.first;
    GE_ASSERT_NOTNULL(alloc_node);
  }

  // 当前只对AllocMemory、AllocMemHbm、AllocMemHost在条件算子子图内的场景做了零拷贝优化
  if (IsIfOrCaseType(alloc_node->GetTypePtr())) {
    GE_ASSERT_SUCCESS(FindAllocNodesInSubgraph(peer_out_node_and_index, alloc_nodes));
    return ge::GRAPH_SUCCESS;
  }
  if (IsModelOutZeroCopyEnabledAllocNode(alloc_node->GetTypePtr())) {
    alloc_nodes.emplace_back(alloc_node);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AppendInputForAllocNode(ge::FastNode *alloc_node, int32_t output_index) {
  auto graph = alloc_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  auto alloc_input_num = alloc_node->GetDataInNum();
  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendInputEdgeInfo(alloc_node, alloc_input_num + 1U));
  std::string inner_data_name;
  static std::atomic<uint64_t> id{0U};
  inner_data_name.append("OutputData_").append(std::to_string(output_index)).append("_").append(graph->GetName())
                 .append("_InnerData_").append(std::to_string(id.fetch_add(1U)));
  GELOGI("AppendInputForAllocNode create InnerData:%s", inner_data_name.c_str());
  auto op_desc = ge::MakeShared<ge::OpDesc>(inner_data_name, kInnerData);
  GE_ASSERT_NOTNULL(op_desc);
  const auto parent_node = graph->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_node);
  auto parent_input_num = parent_node->GetDataInNum();
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "index", static_cast<int64_t>(parent_input_num)));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc("y", ge::GeTensorDesc()));
  auto inner_data = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(inner_data);
  GE_ASSERT_NOTNULL(graph->AddEdge(inner_data, 0, alloc_node, static_cast<int32_t>(alloc_input_num)));
  return ge::GRAPH_SUCCESS;
}

/**
 *
 *                                          NetOutput                                                   NetOutput
 *                      c                    /c     \                               c                    /c    |
 *       FreeMemory <---------  EnsureAtUserMemory  |                FreeMemory <---------  EnsureAtUserMemory |
 *       ^      ^              /  |   |     |    \  |      =>        ^      ^              /  |   |     |    \ |
 *      c|      |           shape |  attrs stream output            c|      |           shape |  attrs stream \|
 *       |      |                 |0                                 |      |                 |0               +
 * SomeNodes <--+----------- AllocMemory                       SomeNodes <--+----------- AllocModelOutTensor   |
 *                    0      /          \                                         0      /          |       \  |
 *                       allocator     size                                          allocator     size    output
 */
ge::graphStatus TransformAllocNodesToZeroCopyNodes(const ge::FastNode *copy_node,
                                                   const std::vector<ge::FastNode *> &alloc_nodes) {
  auto copy_in_data_edge =
      copy_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::EnsureTensorAtOutMemoryInputs::kOutputData));
  GE_ASSERT_NOTNULL(copy_in_data_edge);
  auto output_data_node = copy_in_data_edge->src;
  auto output_data_src_index = copy_in_data_edge->src_output;
  for (const auto alloc_node : alloc_nodes) {
    auto op_desc = alloc_node->GetOpDescPtr();
    GE_ASSERT_NOTNULL(op_desc);
    ge::OpDescUtilsEx::SetType(op_desc, kernel::kAllocModelOut);
    auto cur_node = alloc_node;
    auto curr_owner_graph = cur_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(curr_owner_graph);
    auto copy_owner_graph = copy_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(copy_owner_graph);
    while (curr_owner_graph != copy_owner_graph) {
      GE_ASSERT_SUCCESS(AppendInputForAllocNode(cur_node, output_data_src_index));
      auto parent_node = curr_owner_graph->GetParentNodeBarePtr();
      GE_ASSERT_NOTNULL(parent_node);
      cur_node = parent_node;
      curr_owner_graph = cur_node->GetExtendInfo()->GetOwnerGraphBarePtr();
      GE_ASSERT_NOTNULL(curr_owner_graph);
    }
    auto input_num = cur_node->GetDataInNum();
    GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendInputEdgeInfo(cur_node, static_cast<size_t>(input_num + 1U)));
    GE_ASSERT_NOTNULL(
        copy_owner_graph->AddEdge(output_data_node, output_data_src_index, cur_node, static_cast<int32_t>(input_num)));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RemoveInput(ge::FastNode *node, int32_t input_index, ge::EdgeSrcEndpoint &input_node_and_src_index) {
  auto in_data_edge = node->GetInDataEdgeByIndex(input_index);
  GE_ASSERT_NOTNULL(in_data_edge);
  auto src_node = in_data_edge->src;
  auto src_index = in_data_edge->src_output;
  auto owner_graph = src_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_SUCCESS(owner_graph->RemoveEdge(in_data_edge));
  input_node_and_src_index = {src_node, src_index};
  return ge::GRAPH_SUCCESS;
}
/**
 *
 *                                          NetOutput                                                  NetOutput
 *                      c                    /c    |                                                    /c    |
 *       FreeMemory <---------  EnsureAtUserMemory |                                       EnsureAtUserMemory |
 *       ^      ^              /  |   |     |    \ |                                      /  |   |     |    \ |
 *      c|      |           shape |  attrs stream \|      =>                           shape |  attrs stream \|
 *       |      |                 |0               +                                         |1               +
 * SomeNodes <--+----------- AllocModelOutTensor   |             SomeNodes <------  SplitTensorForOutputData /
 *                    0      /          |       \  |                          1         2/     1|    0\     /
 *                       allocator     size    output                                 size  allocator  output
 */
ge::graphStatus AlwaysZeroCopy(ge::FastNode *alloc_node) {
  auto op_desc = alloc_node->GetOpDescPtr();
  ge::OpDescUtilsEx::SetType(op_desc, kernel::kSplitTensorForOutputData);
  // unlink inputs
  ge::EdgeSrcEndpoint allocator_src_endpoint;
  ge::EdgeSrcEndpoint size_src_endpoint;
  ge::EdgeSrcEndpoint output_src_endpoint;
  GE_ASSERT_SUCCESS(RemoveInput(alloc_node, static_cast<int32_t>(kernel::AllocModelOutTensorInputs::kAllocator),
                                allocator_src_endpoint));

  GE_ASSERT_SUCCESS(
      RemoveInput(alloc_node, static_cast<int32_t>(kernel::AllocModelOutTensorInputs::kAllocSize), size_src_endpoint));

  GE_ASSERT_SUCCESS(RemoveInput(alloc_node, static_cast<int32_t>(kernel::AllocModelOutTensorInputs::kOutputData),
                                output_src_endpoint));

  // reconnect inputs
  auto owner_graph = alloc_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_SUCCESS(ge::FastNodeUtils::RemoveInputEdgeInfo(
      alloc_node, static_cast<uint32_t>(kernel::AllocModelOutTensorInputs::kInputNum)));
  GE_ASSERT_NOTNULL(owner_graph->AddEdge(output_src_endpoint.node, output_src_endpoint.index, alloc_node,
                                         static_cast<int32_t>(kernel::SplitTensorForOutputDataInputs::kTensor)));
  GE_ASSERT_NOTNULL(owner_graph->AddEdge(allocator_src_endpoint.node, allocator_src_endpoint.index, alloc_node,
                                         static_cast<int32_t>(kernel::SplitTensorForOutputDataInputs::kAllocator)));
  GE_ASSERT_NOTNULL(owner_graph->AddEdge(size_src_endpoint.node, size_src_endpoint.index, alloc_node,
                                         static_cast<int32_t>(kernel::SplitTensorForOutputDataInputs::kSize)));

  // reconnect outputs
  GE_ASSERT_SUCCESS(
      ge::FastNodeUtils::AppendOutputEdgeInfo(alloc_node, static_cast<uint32_t>(kernel::SplitTensorOutputs::kNum)));
  ge::EdgeSrcEndpoint td_src_endpoint = {alloc_node, static_cast<int32_t>(kernel::SplitTensorOutputs::kTensorData)};
  const auto &shape_out_edges =
      alloc_node->GetOutEdgesRefByIndex(static_cast<int32_t>(kernel::SplitTensorOutputs::kShape));
  for (const auto out_edge : shape_out_edges) {
    if (out_edge != nullptr) {
      auto dst_node = out_edge->dst;
      if (IsGuarderOf(alloc_node, dst_node)) {
        GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(dst_node, {}));
        GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(owner_graph, dst_node));
      } else {
        GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::ReplaceEdgeSrc(out_edge, td_src_endpoint));
      }
    }
  }

  bool changed;
  std::vector<ge::FastNode *> start_nodes = {allocator_src_endpoint.node};
  GE_ASSERT_SUCCESS(Pruner().PruneFromNodes(start_nodes, changed));

  alloc_node->GetOpDescBarePtr()->SetExtAttr(kPassChangedInfo,
                                             PassChangedKernels{{std::pair<KernelNameAndIdx, KernelNameAndIdx>(
                                                 {alloc_node->GetName(), 0}, {alloc_node->GetName(), 1})}});

  return ge::GRAPH_SUCCESS;
}
}  // namespace
ge::graphStatus ModelOutTensorZeroCopy::Run(ge::ExecuteGraph *const graph, bool &changed) {
  const auto output_data_nodes = graph->GetAllNodes(IsOutputData);
  GE_ASSERT_TRUE(!output_data_nodes.empty());
  auto output_data_node = output_data_nodes.at(0);
  GE_ASSERT_NOTNULL(output_data_node);

  for (const auto copy_node : output_data_node->GetOutDataNodes()) {
    if (copy_node->GetType() != kernel::kEnsureTensorAtOutMemory) {
      continue;
    }
    std::vector<ge::FastNode *> alloc_nodes;
    GE_ASSERT_SUCCESS(GetAllAllocNodes(copy_node, alloc_nodes));

    if (alloc_nodes.empty()) {
      GELOGD("Can not enable zero copy because the alloc node can not be found.");
      continue;
    }
    changed = true;
    GELOGD("Model out tensor zero copy is enabled, graph:%s", graph->GetName().c_str());
    GE_ASSERT_SUCCESS(TransformAllocNodesToZeroCopyNodes(copy_node, alloc_nodes));

    // 当前AlwaysZeroCopy只支持alloc_node在main图上的场景，还未支持在子图内场景
    auto copy_node_owner_graph = copy_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(copy_node_owner_graph);
    if (GetLoweringOption().always_zero_copy) {
      for (const auto alloc_node : alloc_nodes) {
        auto alloc_node_owner_graph = alloc_node->GetExtendInfo()->GetOwnerGraphBarePtr();
        GE_ASSERT_NOTNULL(alloc_node_owner_graph);
        if (alloc_node_owner_graph == copy_node_owner_graph) {
          GE_ASSERT_SUCCESS(AlwaysZeroCopy(alloc_node));
          break;
        }
      }
    }
  }
  if (changed) {
    ge::DumpGraph(graph, "AfterZeroCopy");
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert