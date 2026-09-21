/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sequential_execution_data_builder.h"
#include <map>
#include <queue>
#include "core/builder/node_types.h"
#include "core/builder/graph_node.h"
#include "register/kernel_registry.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace {
void PushToReadyQueue(ge::FastNode *const node, std::map<int64_t,
                      std::queue<ge::FastNode *>> &priorities_to_ready_nodes,
                      std::unordered_set<ge::FastNode *> &seen_nodes) {
  if (seen_nodes.insert(node).second) {
    int64_t priority = std::numeric_limits<int64_t>::max();
    (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "priority", priority);
    priorities_to_ready_nodes[priority].push(node);
  }
}
ge::FastNode *PopFromReadyQueue(std::map<int64_t, std::queue<ge::FastNode *>> &priorities_to_ready_nodes) {
  auto iter = priorities_to_ready_nodes.begin();
  auto node = iter->second.front();
  iter->second.pop();
  if (iter->second.empty()) {
    priorities_to_ready_nodes.erase(iter);
  }
  return node;
}
ge::graphStatus InitAllKernelFunctions(const std::vector<std::pair<ge::FastNode *, Node *>> &graph_nodes_to_executor_node) {
  for (const auto &graph_node_to_exe_node : graph_nodes_to_executor_node) {
    auto &org_node = graph_node_to_exe_node.first;
    auto &exe_node = graph_node_to_exe_node.second;
    auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(org_node->GetType());
    if (kernel_funcs == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "Failed to find exec node [%s] kernel funcs", org_node->GetType().c_str());
      return ge::GRAPH_FAILED;
    }
    exe_node->func = reinterpret_cast<KernelFunc>(kernel_funcs->run_func);
  }
  return ge::GRAPH_SUCCESS;
}
AsyncAnyValue **CreateInputValues(GraphNode &graph_nodes, TopologicalResourceGuard *resource_guard) {
  auto total_size = sizeof(AsyncAnyValue *) * graph_nodes.indexes_to_feed_input.size();
  std::unique_ptr<uint8_t[]> input_values_guard = ge::MakeUnique<uint8_t[]>(total_size);
  if (input_values_guard == nullptr) {
    return nullptr;
  }
  auto input_values = reinterpret_cast<AsyncAnyValue **>(input_values_guard.get());
  resource_guard->ResetInputsArray(std::move(input_values_guard));
  int32_t index = 0;
  std::set<int32_t> append_indexes;
  for (const auto &index_to_feed_input : graph_nodes.indexes_to_feed_input) {
    if (index_to_feed_input.first >= 0) {
      if (index != index_to_feed_input.first) {
        GELOGE(ge::PARAM_INVALID, "Failed to load exe graph, input data index %d, expect %d", index_to_feed_input.first,
               index);
        return nullptr;
      }
      input_values[index++] = index_to_feed_input.second;
    } else {
      if (!append_indexes.insert(index_to_feed_input.first).second) {
        GELOGE(ge::PARAM_INVALID, "Failed to load exe graph, duplicated input data index %d",
               index_to_feed_input.first);
        return nullptr;
      }
    }
  }
  int32_t feed_size = index;
  for (auto iter = append_indexes.rbegin(); iter != append_indexes.rend(); ++iter) {
    auto append_index = (*iter * -1) - 1;
    if (append_index + feed_size != index) {
      GELOGE(ge::PARAM_INVALID, "Failed to load exe graph, append input data index %d, expect %d", append_index, index);
      return nullptr;
    }
    input_values[index++] = graph_nodes.indexes_to_feed_input[*iter];
  }
  return input_values;
}
}  // namespace

SequentialExecutionDataBuilder::SequentialExecutionDataBuilder(GraphExecutorBuilder &executor_builder)
    : ExecutionDataBuilder(executor_builder) {}

ResourceGuardPtr SequentialExecutionDataBuilder::Build() {
  auto resource_guard = ge::MakeUnique<TopologicalResourceGuard>();
  GE_ASSERT_NOTNULL(resource_guard);
  auto execution_data = ge::MakeUnique<uint8_t[]>(sizeof(SequentialExecutionData));
  GE_ASSERT_NOTNULL(execution_data);

  GraphNode graph_nodes;
  GE_ASSERT_SUCCESS(Build(reinterpret_cast<SequentialExecutionData *>(execution_data.get()), resource_guard.get(),
                          graph_nodes));

  resource_guard->ResetExecutionData(std::move(execution_data));
  GE_ASSERT_SUCCESS(CreateKernelOutputs(GetOrderedGraphToExeNodes()));
  return resource_guard;
}

ge::graphStatus SequentialExecutionDataBuilder::Build(SequentialExecutionData *execution_data,
                                                      ResourceGuard *resource_guard, GraphNode &graph_nodes) {
  GE_ASSERT_GRAPH_SUCCESS(
      AllocGraphAsyncValues(GetExecutorBuilder().GetExeGraph()->GetAllNodes(), graph_async_value_));
  const auto topo_resource_guard = reinterpret_cast<TopologicalResourceGuard *>(resource_guard);
  topo_resource_guard->ResetAnyValue(std::move(graph_async_value_.values_guarder), graph_async_value_.total_num);

  if (re_order_by_priority_) {
    GELOGI("Read in graph nodes with priority re-order, graph name is %s",
           GetExecutorBuilder().GetExeGraph()->GetName().c_str());
    GE_ASSERT_SUCCESS(ReadInNodesByPriority(graph_nodes, resource_guard));
  } else {
    GELOGI("Read in graph nodes without priority re-order, graph name is %s",
           GetExecutorBuilder().GetExeGraph()->GetName().c_str());
    GE_ASSERT_SUCCESS(ReadInNodes(graph_nodes, resource_guard));
  }

  GE_ASSERT_SUCCESS(InitAllKernelFunctions(ordered_graph_to_exe_nodes_));
  GE_ASSERT_SUCCESS(InitAllKernelExtendInfos(ordered_graph_to_exe_nodes_));
  return CreateExecutionData(graph_nodes, execution_data, resource_guard);
}
SequentialExecutionDataBuilder &SequentialExecutionDataBuilder::ReOrderByPriority(bool flag) {
  re_order_by_priority_ = flag;
  return *this;
}
ge::graphStatus SequentialExecutionDataBuilder::ReadInOneNode(ge::FastNode *const node, GraphNode &graph_nodes,
                                                              ResourceGuard *resource_guard) {
  const auto node_type = node->GetTypePtr();
  // 处理3类和4类节点
  if (IsMemTransferNode(node_type) || IsUsrOutputNode(node_type)) {
    return ge::GRAPH_SUCCESS;
  }
  // 处理1类节点：图输入
  if (IsGraphInputNode(node_type)) {
    return graph_nodes.ReadInFeed(node, graph_async_value_);
  }
  // 处理2类节点：图输出
  if (IsGraphOutputNode(node_type)) {
    return graph_nodes.ReadInNetOutput(node, graph_async_value_);
  }
  // 处理5类节点：含子图节点
  if (IsHasSubGraphNode(node_type)) {
    return graph_nodes.ReadInNodeHasSubgraph(node);
  }
  // 6类和7类节点在图上有执行节点，需要创建执行node
  Node *exe_node = graph_nodes.ReadInNode(node, graph_async_value_);
  GE_ASSERT_NOTNULL(exe_node);

  const auto topo_resource_guard = reinterpret_cast<TopologicalResourceGuard *>(resource_guard);
  topo_resource_guard->PushNode(exe_node);
  ordered_graph_to_exe_nodes_.emplace_back(node, exe_node);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus SequentialExecutionDataBuilder::ReadInNodes(GraphNode &graph_nodes, ResourceGuard *resource_guard) {
  auto graph = GetExecutorBuilder().GetExeGraph();
  for (const auto node : graph->GetAllNodes()) {
    GE_ASSERT_SUCCESS(ReadInOneNode(node, graph_nodes, resource_guard));
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus SequentialExecutionDataBuilder::ReadInNodesByPriority(GraphNode &graph_nodes,
                                                                      ResourceGuard *resource_guard) {
  auto graph = GetExecutorBuilder().GetExeGraph();
  std::map<int64_t, std::queue<ge::FastNode *>> priorities_to_ready_nodes;
  std::unordered_set<ge::FastNode *> seen_nodes;
  std::unordered_map<ge::FastNode *, size_t> indegree_map;
  const auto collect_nodes = [&priorities_to_ready_nodes, &seen_nodes,
                              &indegree_map](const ge::FastEdge *const edge) -> ge::graphStatus {
    if (edge == nullptr) {
      return ge::GRAPH_SUCCESS;
    }
    const auto out_node = edge->dst;
    const auto it = indegree_map.find(out_node);
    GE_ASSERT_TRUE(it != indegree_map.end(), "can not find node %s", out_node->GetName().c_str());
    --it->second;
    if (it->second > 0U) {
      return ge::GRAPH_SUCCESS;
    }
    PushToReadyQueue(out_node, priorities_to_ready_nodes, seen_nodes);
    return ge::GRAPH_SUCCESS;
  };

  // sequential 仅支持平铺图，所以DirectNode就可以了
  for (const auto node : graph->GetDirectNode()) {
    const auto in_nodes_size = node->GetAllInEdgeSize();
    indegree_map[node] = in_nodes_size;
    if (in_nodes_size == 0UL) {
      PushToReadyQueue(node, priorities_to_ready_nodes, seen_nodes);
    }
  }

  while (!priorities_to_ready_nodes.empty()) {
    auto node = PopFromReadyQueue(priorities_to_ready_nodes);
    GE_ASSERT_SUCCESS(ReadInOneNode(node, graph_nodes, resource_guard));
    for (const auto &out_data_edges : node->GetAllOutDataEdgesRef()) {
      for (const auto out_data_edge : out_data_edges) {
        GE_ASSERT_GRAPH_SUCCESS(collect_nodes(out_data_edge));
      }
    }
    for (const auto out_ctl_edge : node->GetAllOutControlEdgesRef()) {
      GE_ASSERT_GRAPH_SUCCESS(collect_nodes(out_ctl_edge));
    }
  }

  return ge::GRAPH_SUCCESS;
}
ge::graphStatus SequentialExecutionDataBuilder::InitAllKernelExtendInfos(
    const vector<std::pair<ge::FastNode *, Node *>> &graph_nodes_to_executor_node) const {
  for (const auto &graph_node_to_exe_node : graph_nodes_to_executor_node) {
    auto &org_node = graph_node_to_exe_node.first;
    auto &exe_node = graph_node_to_exe_node.second;
    const auto op_desc = org_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    int64_t compute_node_index;
    if (ge::AttrUtils::GetInt(op_desc, kComputeNodeIndex, compute_node_index)) {
      auto node_info = GetExecutorBuilder().GetModelLevelData().compute_nodes_info->Get<void>(compute_node_index);
      if (node_info == nullptr) {
        GELOGE(ge::FAILED,
               "Failed to get compute node info by index %" PRId64
               ", "
               "node name %s, type %s",
               compute_node_index, org_node->GetName().c_str(), org_node->GetType().c_str());
        return ge::GRAPH_FAILED;
      }
      exe_node->context.compute_node_info = node_info;
    }

    int64_t kernel_extend_index;
    if (ge::AttrUtils::GetInt(op_desc, kKernelExtendIndex, kernel_extend_index)) {
      auto ke_info = GetExecutorBuilder().GetModelLevelData().kernel_extend_info->Get<void>(kernel_extend_index);
      if (ke_info == nullptr) {
        GELOGE(ge::FAILED,
               "Failed to get kernel extend info by index %" PRId64
               ", node name %s, "
               "type %s",
               kernel_extend_index, org_node->GetName().c_str(), org_node->GetType().c_str());
        return ge::GRAPH_FAILED;
      }
      exe_node->context.kernel_extend_info = ke_info;
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus SequentialExecutionDataBuilder::CreateExecutionData(GraphNode &graph_node,
                                                                    SequentialExecutionData *execution_data,
                                                                    ResourceGuard *resource_guard) const{
  const auto topo_resource_guard = static_cast<TopologicalResourceGuard *>(resource_guard);
  execution_data->node_num = graph_node.nodes.size();
  execution_data->nodes = static_cast<Node **>(topo_resource_guard->ResetNodesArray(CreateCArray(graph_node.nodes)));
  if (execution_data->node_num > 0U) {
    GE_ASSERT_NOTNULL(execution_data->nodes);
  }

  execution_data->input_num = graph_node.indexes_to_feed_input.size();
  execution_data->input_values = CreateInputValues(graph_node, topo_resource_guard);
  if (execution_data->input_num > 0U) {
    GE_ASSERT_NOTNULL(execution_data->input_values);
  }

  execution_data->output_num = graph_node.indexes_to_output.size();
  execution_data->output_values =
      static_cast<AsyncAnyValue **>(topo_resource_guard->ResetOutputsArray(CreateCArray(graph_node.indexes_to_output)));
  if (execution_data->output_num > 0U) {
    GE_ASSERT_NOTNULL(execution_data->output_values);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SequentialExecutionDataBuilder::AllocGraphAsyncValues(
    const std::vector<ge::FastNode *> &graph_nodes, GraphAsyncValue &graph_async_value) {
  // todo
  //
  // alloc节点在Init图中，其输出被main图中的节点引用，而alloc节点与被其引用的节点之间没有数据边连接，所以在root图上看不到这个chain
  // 以前认为只有子图能看到父图的chain，平级子图之间互相看不到chain，这个规则最好还是不要打破，后面再想一下办法
  auto symbols_to_value = GetExecutorBuilder().GetSymbolsToValue();

  std::vector<uint64_t> ordered_symbols;
  std::unordered_map<uint64_t, std::vector<AsyncAnyValue **>> symbols_to_chain;
  for (const auto node : graph_nodes) {
    auto &output_values = graph_async_value.nodes_to_output_values[node];
    auto &input_values = graph_async_value.nodes_to_input_values[node];
    output_values.resize(node->GetDataOutNum());
    input_values.resize(node->GetDataInNum());
    std::vector<std::pair<uint64_t, AsyncAnyValue **>> symbol_av_addrs;
    for (size_t i = 0U; i < node->GetDataOutNum(); ++i) {
      auto symbol_id = node->GetExtendInfo()->GetOutputSymbol(i);
      GE_ASSERT_TRUE(symbol_id != ge::kInvalidSymbol);
      symbol_av_addrs.emplace_back(symbol_id, &output_values[i]);
    }
    // 执行图已经topo排序，遍历input的时候，可以保证输出的symbol已经被标记
    for (size_t i = 0U; i < node->GetDataInNum(); ++i) {
      auto symbol_id = node->GetExtendInfo()->GetInputSymbol(i);
      GE_ASSERT_TRUE(symbol_id != ge::kInvalidSymbol);
      symbol_av_addrs.emplace_back(symbol_id, &input_values[i]);
    }
    for (const auto &symbol_av_addr : symbol_av_addrs) {
      auto symbol = symbol_av_addr.first;
      auto value_iter = symbols_to_value->find(symbol);
      if (value_iter == symbols_to_value->end()) {
        auto &chain_values = symbols_to_chain[symbol];
        if (chain_values.empty()) {
          ordered_symbols.push_back(symbol);
        }
        chain_values.push_back(symbol_av_addr.second);
      } else {
        *symbol_av_addr.second = value_iter->second;
      }
    }
  }

  graph_async_value.total_num = ordered_symbols.size();
  graph_async_value.values_guarder = ge::MakeUnique<uint8_t[]>(graph_async_value.total_num * sizeof(AsyncAnyValue));
  graph_async_value.values = reinterpret_cast<AsyncAnyValue *>(graph_async_value.values_guarder.get());
  GE_CHECK_NOTNULL(graph_async_value.values);

  AsyncAnyValue *current_symbol_value = graph_async_value.values;
  for (auto &symbol : ordered_symbols) {
    (*symbols_to_value)[symbol] = current_symbol_value;
    for (auto &value_addr : symbols_to_chain[symbol]) {
      *value_addr = current_symbol_value;
    }
    current_symbol_value++;
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
