/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <benchmark/benchmark.h>
#include <memory>
#include <string>
#include <vector>
#include <graph_utils.h>
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "fast_graph/fast_graph_impl.h"
#include "fast_node_utils.h"
#include "node_utils.h"
#include "graph/op_desc.h"

using namespace ge;

#define GRAPH_CHECK_RET true

class NodeBuilder {
 public:
  NodeBuilder(string name, string type, const std::shared_ptr<ExecuteGraph> &owner_graph)
      : name_(std::move(name)), type_(std::move(type)), owner_graph_(owner_graph) {}
  NodeBuilder &InputNum(int64_t num) {
    input_num_ = num;
    return *this;
  }
  NodeBuilder &OutputNum(int64_t num) {
    output_num_ = num;
    return *this;
  }

  NodeBuilder &IoNum(int64_t input_num, int64_t output_num) {
    return InputNum(input_num).OutputNum(output_num);
  }

  FastNode *Build() const {
    auto op_desc = std::make_shared<OpDesc>(name_, type_);
    auto td = GeTensorDesc();
    for (int64_t i = 0; i < input_num_; ++i) {
      op_desc->AddInputDesc(td);
    }
    for (int64_t i = 0; i < output_num_; ++i) {
      op_desc->AddOutputDesc(td);
    }
    return owner_graph_->AddNode(op_desc);
  }

 private:
  std::string name_;
  std::string type_;
  std::shared_ptr<ExecuteGraph> owner_graph_;
  int64_t input_num_ = 0;
  int64_t output_num_ = 0;
};

void OpDescCreate(int64_t node_num, std::shared_ptr<OpDesc> *op_desc, int64_t io_num) {
  for (int64_t j = 0; j < node_num; j++) {
    op_desc[j] = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();

    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (int64_t i = 0; i < io_num; ++i) {
      op_desc[j]->AddOutputDesc(td);
    }
  }
}

static void OLD_Graph_Creation(benchmark::State &state) {
  for (auto _ : state) {
    auto compute_graph = std::make_shared<ge::ComputeGraph>("graph");
    benchmark::DoNotOptimize(compute_graph);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_Creation);

static void NEW_Graph_Creation(benchmark::State &state) {
  for (auto _ : state) {
    auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
    benchmark::DoNotOptimize(compute_graph);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(NEW_Graph_Creation);

static void TEST_HASH_TIME(benchmark::State &state) {
  std::string test = "hello, world.";
  int loop_num = 1000;
  for (int i = 0; i < loop_num; ++i) {
    test += "a";
  }
  for (auto _ : state) {
    auto size = std::hash<std::string>{}(test);
    benchmark::DoNotOptimize(size);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(TEST_HASH_TIME);

static void OLD_Graph_AddAndRemoveSingleNode(benchmark::State &state) {
  auto compute_graph = std::make_shared<ComputeGraph>("graph0");
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  for (int j = 0; j < node_num; j++) {
    if (j != 0) compute_graph->AddNode(op_desc[j]);
  }

  ge::NodePtr node[node_num];
  for (auto _ : state) {
    node[0] = compute_graph->AddNode(op_desc[0]);
#if GRAPH_CHECK_RET
    if (node[0] == nullptr) {
      std::cout << "Graph_AddNode Error" << std::endl;
      return;
    }
#endif
    GraphUtils::RemoveJustNode(compute_graph, node[0]);
    benchmark::DoNotOptimize(node[0]);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_AddAndRemoveSingleNode)->Args({20, 10})->Args({20, 100})->Args({20, 1000})->Args({20, 10000});

static void NEW_Graph_AddAndRemoveSingleNode(benchmark::State &state) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph1");
  int64_t node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};

  int64_t io_num = state.range(0);
  OpDescCreate(node_num, op_desc, io_num);
  for (int j = 0; j < node_num; j++) {
    if (j != 0) compute_graph->AddNode(op_desc[j]);
  }

  FastNode *node[node_num] = {};
  for (auto _ : state) {
    node[0] = compute_graph->AddNode(op_desc[0]);
#if GRAPH_CHECK_RET
    if (node[0] == nullptr) {
      std::cout << "Graph_AddNode Error" << std::endl;
      return;
    }
#endif
    compute_graph->RemoveJustNode(node[0]);

    benchmark::DoNotOptimize(node[0]);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(NEW_Graph_AddAndRemoveSingleNode)->Args({20, 10})->Args({20, 100})->Args({20, 1000})->Args({20, 10000});

static void NEW_Graph_AddAndRemoveMultiNode(benchmark::State &state) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph2");
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);

  FastNode *node[node_num] = {};
  for (auto _ : state) {
    for (int64_t j = 0; j < node_num; j++) {
      node[j] = compute_graph->AddNode(op_desc[j]);
    }

    for (int64_t j = 0; j < node_num; j++) {
      compute_graph->RemoveJustNode(node[j]);
    }
    benchmark::DoNotOptimize(node[0]);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(NEW_Graph_AddAndRemoveMultiNode)->Args({20, 10000})->Args({20, 100000});

static void OLD_Graph_AddAndRemoveMultiNode(benchmark::State &state) {
  auto compute_graph = std::make_shared<ComputeGraph>("graph01");
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);

  ge::NodePtr node[node_num];
  for (auto _ : state) {
    for (int64_t j = 0; j < node_num; j++) {
      node[j] = compute_graph->AddNode(op_desc[j]);
    }

    for (int64_t j = 0; j < node_num; j++) {
      GraphUtils::RemoveJustNode(compute_graph, node[j]);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_AddAndRemoveMultiNode)->Args({20, 10000})->Args({20, 100000});

static void OLD_Graph_ADD_NODE_WITH_NODE(benchmark::State &state) {
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  auto root_graph2 = std::make_shared<ComputeGraph>("root_graph2");
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);

  NodePtr node[node_num] = {};
  for (auto _ : state) {
    for (int i = 0; i < node_num; i++) {
      node[i] = root_graph2->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
      if (node[i] == nullptr) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE AddNode Error" << std::endl;
        return;
      }
#endif
      auto ret = GraphUtils::RemoveJustNode(root_graph2, node[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE RemoveJustNode Error" << std::endl;
        return;
      }
#endif

      node[i] = root_graph->AddNode(node[i]);
#if GRAPH_CHECK_RET
      if (node[i] == nullptr) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE AddNode Error" << std::endl;
        return;
      }
#endif

      ret = GraphUtils::RemoveJustNode(root_graph, node[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE RemoveJustNode Error" << std::endl;
        return;
      }
#endif
    }
  }
}
BENCHMARK(OLD_Graph_ADD_NODE_WITH_NODE)->Args({20, 100});
BENCHMARK(OLD_Graph_ADD_NODE_WITH_NODE)->Args({20, 1000});
BENCHMARK(OLD_Graph_ADD_NODE_WITH_NODE)->Args({20, 10000});
BENCHMARK(OLD_Graph_ADD_NODE_WITH_NODE)->Args({20, 50000});

static void NEW_Graph_ADD_NODE_WITH_NODE(benchmark::State &state) {
  auto new_root_graph2 = std::make_shared<ExecuteGraph>("new_graph2");
  auto new_root_graph = std::make_shared<ExecuteGraph>("new_graph");
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  FastNode *node[node_num] = {};
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);

  for (auto _ : state) {
    for (int i = 0; i < node_num; i++) {
      node[i] = new_root_graph2->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
      if (node[i] == nullptr) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE Add Node Error" << std::endl;
        return;
      }
#endif
      auto ret = new_root_graph2->RemoveJustNode(node[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE Remove Node Error" << std::endl;
        return;
      }
#endif

      node[i] = new_root_graph->AddNode(node[i]);
#if GRAPH_CHECK_RET
      if (node[i] == nullptr) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE Add Node Error" << std::endl;
        return;
      }
#endif

      ret = new_root_graph->RemoveJustNode(node[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "NEW_Graph_ADD_NODE_WITH_NODE Remove Node Error" << std::endl;
        return;
      }
#endif
    }
  }
}
BENCHMARK(NEW_Graph_ADD_NODE_WITH_NODE)->Args({20, 100});
BENCHMARK(NEW_Graph_ADD_NODE_WITH_NODE)->Args({20, 1000});
BENCHMARK(NEW_Graph_ADD_NODE_WITH_NODE)->Args({20, 10000});
BENCHMARK(NEW_Graph_ADD_NODE_WITH_NODE)->Args({20, 50000});

static void OLD_Graph_GetDirectNode(benchmark::State &state) {
  auto compute_graph = std::make_shared<ComputeGraph>("graph");
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);

  for (int j = 0; j < node_num; j++) {
    compute_graph->AddNode(op_desc[j]);
  }

  for (auto _ : state) {
    auto ret = compute_graph->GetDirectNode();
    if (ret.size() == 0) {
      std::cout << "OLD GetDirectNode Error " << std::endl;
      return;
    }

    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_GetDirectNode)->Args({20, 10})->Args({20, 100})->Args({20, 1000})->Args({20, 10000})->Iterations(1);

static void New_Graph_GetDirectNode(benchmark::State &state) {
  auto compute_graph = std::make_shared<ExecuteGraph>(std::string("graph"));
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  for (int j = 0; j < node_num; j++) {
    (void)compute_graph->AddNode(op_desc[j]);
  }

  for (auto _ : state) {
    auto ret = compute_graph->GetDirectNode();
    if (ret.size() == 0) {
      std::cout << "OLD GetDirectNode Error " << std::endl;
      return;
    }

    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(New_Graph_GetDirectNode)->Args({20, 10})->Args({20, 100})->Args({20, 1000})->Args({20, 10000})->Iterations(1);

static void Graph_AddAndRemoveEdge(benchmark::State &state) {
  auto compute_graph = std::make_shared<ExecuteGraph>("graph");
  int num = state.range(1);
  int vec_size = state.range(0);
  std::vector<FastNode *> vec;
  vec.resize(vec_size);

  for (int i = 0; i < vec_size; i++) {
    vec[i] = NodeBuilder("Node" + std::to_string(i), "Node", compute_graph).IoNum(num, num).Build();
#if GRAPH_CHECK_RET
    if (vec[i] == nullptr) {
      std::cout << "Graph_AddEdge vec[i] Error " << i << std::endl;
      return;
    }
#endif
  }

  FastEdge *edge[num] = {};
  for (auto _ : state) {
    for (int i = 0; i < num; i++) {
      edge[i] = compute_graph->AddEdge(vec[1], i, vec[0], i);
#if GRAPH_CHECK_RET
      if (edge[i] == nullptr) {
        std::cout << "Graph_AddEdge Error " << std::endl;
        return;
      }
#endif
    }

    for (int i = 0; i < num; i++) {
      auto ret = compute_graph->RemoveEdge(edge[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "OLD_Graph_AddAndRemoveEdge RemoveEdge Error " << i << std::endl;
        return;
      }
#endif
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(Graph_AddAndRemoveEdge)->Args({20, 20})->Args({20, 100})->Args({20, 1000})->Args({20, 10000})->Iterations(1);

static void OLD_Graph_AddAndRemoveEdge(benchmark::State &state) {
  auto compute_graph = std::make_shared<ComputeGraph>("graph");
  int64_t io_num = state.range(0);
  int64_t node_num = state.range(1);
  std::vector<Node *> nodes;
  nodes.resize(node_num);

  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  for (int j = 0; j < node_num; j++) {
    auto tmp = compute_graph->AddNode(op_desc[j]);
    nodes[j] = tmp.get();
  }

  for (auto _ : state) {
    for (int i = 1; i < node_num; i++) {
      auto ret = GraphUtils::AddEdge(nodes[i]->GetOutDataAnchor(0), nodes[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "OLD_Graph_AddAndRemoveEdge AddEdge Error " << i << std::endl;
        return;
      }
#endif
    }

    for (int i = 1; i < node_num; i++) {
      auto ret = GraphUtils::RemoveEdge(nodes[i]->GetOutDataAnchor(0), nodes[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "OLD_Graph_AddAndRemoveEdge RemoveEdge Error " << i << std::endl;
        return;
      }
#endif
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_AddAndRemoveEdge)
    ->Args({20, 20})
    ->Args({20, 100})
    ->Args({20, 1000})
    ->Args({20, 10000})
    ->Iterations(1);

static void Graph_GetAllEdge(benchmark::State &state) {
  auto compute_graph = std::make_shared<ExecuteGraph>("graph");
  std::vector<FastNode *> vec;
  int node_num = 2;
  vec.resize(node_num);
  int edge_num = state.range(0);

  for (int i = 0; i < node_num; i++) {
    vec[i] = NodeBuilder("Node" + std::to_string(i), "Node", compute_graph).IoNum(edge_num, edge_num).Build();
  }

  FastEdge *edge[edge_num] = {};
  for (int i = 0; i < edge_num; i++) {
    edge[i] = compute_graph->AddEdge(vec[1], i, vec[0], i);
  }

  for (auto _ : state) {
    auto ret = compute_graph->GetAllEdges();
#if GRAPH_CHECK_RET
    if (ret.size() == 0) {
      std::cout << "Graph_GetAllOutEdge Error " << std::endl;
      return;
    }
#endif

    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(Graph_GetAllEdge)->Arg(20)->Arg(100)->Arg(1000)->Arg(10000);

static void Graph_AddAndRemoveSubgraph(benchmark::State &state) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  auto subgraph_num = state.range();
  int edge_num = 5;
  FastNode *node[subgraph_num] = {};
  std::shared_ptr<OpDesc> op_desc[subgraph_num] = {nullptr};
  OpDescCreate(subgraph_num, op_desc, edge_num);

  for (int i = 0; i < subgraph_num; ++i) {
    node[i] = root_graph->AddNode(op_desc[i]);
  }

  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    sub_graph[i]->SetParentGraph(root_graph.get());
    sub_graph[i]->SetParentNode(node[i]);
  }

  for (int i = 0; i < subgraph_num - 1; i++) {
    auto ret = root_graph->AddSubGraph(sub_graph[i]);
#if GRAPH_CHECK_RET
    if (ret == nullptr) {
      std::cout << "Graph_RemoveHeadEdge Error" << std::endl;
      return;
    }
#endif
  }

  for (auto _ : state) {
    auto ret = root_graph->AddSubGraph(sub_graph[subgraph_num - 1]);
#if GRAPH_CHECK_RET
    if (ret == nullptr) {
      std::cout << "Graph_RemoveHeadEdge Error" << std::endl;
      return;
    }
#endif
    root_graph->RemoveSubGraph(ret);

    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(Graph_AddAndRemoveSubgraph)->Arg(10)->Arg(100)->Arg(1000)->Arg(10000);

static void OLD_Graph_AddAndRemoveSubgraph(benchmark::State &state) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  auto subgraph_num = state.range();
  int edge_num = 5;
  NodePtr node[subgraph_num] = {};
  std::shared_ptr<OpDesc> op_desc[subgraph_num] = {nullptr};
  ComputeGraphPtr sub_graph[subgraph_num] = {nullptr};

  OpDescCreate(subgraph_num, op_desc, edge_num);
  for (int i = 0; i < subgraph_num; ++i) {
    node[i] = root_graph->AddNode(op_desc[i]);
  }
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ComputeGraph>("subgraph_" + std::to_string(i));
    sub_graph[i]->SetParentGraph(root_graph);
    sub_graph[i]->SetParentNode(node[i]);
  }
  for (int i = 0; i < subgraph_num - 1; i++) {
    auto ret = root_graph->AddSubgraph(sub_graph[i]);
#if GRAPH_CHECK_RET
    if (ret != GRAPH_SUCCESS) {
      std::cout << "OLD_Graph_AddAndRemoveSubgraph 0 Error" << std::endl;
      return;
    }
#endif
  }

  for (auto _ : state) {
    auto ret = root_graph->AddSubgraph(sub_graph[subgraph_num - 1]);
#if GRAPH_CHECK_RET
    if (ret != GRAPH_SUCCESS) {
      std::cout << "OLD_Graph_AddAndRemoveSubgraph 1 Error" << std::endl;
      return;
    }
#endif
    root_graph->RemoveSubGraph(sub_graph[subgraph_num - 1]);

    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_AddAndRemoveSubgraph)->Arg(10)->Arg(100)->Arg(1000)->Arg(10000);

static void Graph_Sort(benchmark::State &state) {
  auto compute_graph = std::make_shared<ExecuteGraph>("graph");
  std::vector<FastNode *> vec;
  int io_num = state.range(0);
  int node_num = state.range(1);
  vec.resize(node_num);

  for (int i = 0; i < node_num; i++) {
    vec[i] = NodeBuilder("Node" + std::to_string(i), "Node", compute_graph).IoNum(io_num, io_num).Build();
#if GRAPH_CHECK_RET
    if (vec[i] == nullptr) {
      std::cout << "Graph_Sort Error." << std::endl;
      return;
    }
#endif
  }

  FastEdge *edge[node_num] = {};
  for (int j = 1; j < node_num; j++) {
    edge[j] = compute_graph->AddEdge(vec[j - 1], 1, vec[j], 0);
#if GRAPH_CHECK_RET
    if (edge[j] == nullptr) {
      std::cout << "Graph_Sort Error." << std::endl;
      return;
    }
#endif
  }

  for (auto _ : state) {
    auto ret = compute_graph->TopologicalSortingGraph(compute_graph.get(), true);
#if GRAPH_CHECK_RET
    if (ret != GRAPH_SUCCESS) {
      std::cout << "Graph_Sort Error: " << ret << std::endl;
      return;
    }
#endif
    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(Graph_Sort)->Args({20, 10000})->Args({20, 50000})->Args({20, 100000})->Args({20, 200000})->Iterations(1);

static void OLD_Graph_Sort(benchmark::State &state) {
  auto compute_graph = std::make_shared<ComputeGraph>("graph");
  int io_num = state.range(0);
  int node_num = state.range(1);
  std::vector<NodePtr> vec;
  vec.resize(node_num);

  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  for (int i = 0; i < node_num; i++) {
    vec[i] = compute_graph->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
    if (vec[i] == nullptr) {
      std::cout << "OLD_Graph_Sort Error: 0" << std::endl;
      return;
    }
#endif
  }

  for (int j = 1; j < node_num; j++) {
    auto ret = GraphUtils::AddEdge(vec[j - 1]->GetOutDataAnchor(1), vec[j]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
    if (ret != GRAPH_SUCCESS) {
      std::cout << "OLD_Graph_Sort Error: 1" << std::endl;
      return;
    }
#endif
  }

  for (auto _ : state) {
    auto ret = compute_graph->TopologicalSortingGraph(true);
#if GRAPH_CHECK_RET
    if (ret != GRAPH_SUCCESS) {
      std::cout << "OLD_Graph_Sort Error: 2" << std::endl;
      return;
    }
#endif

    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_Sort)->Args({20, 10000})->Args({20, 50000})->Args({20, 100000})->Args({20, 200000})->Iterations(1);

static void Graph_ALL_RUN(benchmark::State &state) {
  int node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  int edge_num = state.range(0);
  OpDescCreate(node_num, op_desc, edge_num);

  auto subgraph_num = state.range(2);
  auto subgraph_node_num = state.range(3);
  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};

  FastNode *node[node_num] = {};
  FastEdge *edge[node_num] = {};
  ExecuteGraph *quick_graph[subgraph_num] = {nullptr};
  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  for (int i = 0; i < subgraph_num; i++) {
    OpDescCreate(subgraph_node_num, sub_op_desc[i], edge_num);
  }

  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  for (auto _ : state) {
    for (int i = 0; i < node_num; i++) {
      node[i] = root_graph->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
      if (node[i] == nullptr) {
        std::cout << "0 Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
    }

    for (int i = 1; i < node_num; i++) {
      edge[i] = root_graph->AddEdge(node[i], 1, node[i - 1], 0);
#if GRAPH_CHECK_RET
      if (edge[i] == nullptr) {
        std::cout << "1 Graph_ALL_RUN Add Edge Error " << i << std::endl;
        return;
      }
#endif
    }

    FastNode *sub_node[subgraph_num][subgraph_node_num] = {};
    for (int i = 0; i < subgraph_num; i++) {
      sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
      for (int j = 0; j < subgraph_node_num; j++) {
        sub_node[i][j] = sub_graph[i]->AddNode(sub_op_desc[i][j]);
#if GRAPH_CHECK_RET
        if (sub_node[i][j] == nullptr) {
          std::cout << "Graph_ALL_RUN add subgraph node error." << std::endl;
          return;
        }
#endif
      }
    }

    for (int i = 0; i < subgraph_num; i++) {
      for (int j = 1; j < subgraph_node_num; j++) {
        auto ret = sub_graph[i]->AddEdge(sub_node[i][j], 1, sub_node[i][j - 1], 0);
#if GRAPH_CHECK_RET
        if (ret == nullptr) {
          std::cout << "1 Graph_ALL_RUN sub graph edge Error " << j << std::endl;
          return;
        }
#endif
      }
    }

    for (int i = 0; i < subgraph_num; ++i) {
      quick_graph[i] = root_graph->AddSubGraph(sub_graph[i]);
#if GRAPH_CHECK_RET
      if (quick_graph[i] == nullptr) {
        std::cout << "2 Graph_ALL_RUN add subgraph Error" << std::endl;
        return;
      }
#endif
    }

    root_graph->TopologicalSortingGraph(root_graph.get(), true);

    for (int i = 1; i < node_num; i++) {
      root_graph->RemoveEdge(edge[i]);
    }

    for (int i = 0; i < node_num; i++) {
      root_graph->RemoveJustNode(node[i]);
    }

    for (int i = 0; i < subgraph_num; ++i) {
      root_graph->RemoveSubGraph(quick_graph[i]);
    }

    benchmark::ClobberMemory();
  }
}
BENCHMARK(Graph_ALL_RUN)->Args({20, 2000, 1000, 10})->Iterations(1);
BENCHMARK(Graph_ALL_RUN)->Args({20, 4000, 1000, 10})->Iterations(1);
BENCHMARK(Graph_ALL_RUN)->Args({20, 6000, 1000, 10})->Iterations(1);
BENCHMARK(Graph_ALL_RUN)->Args({20, 8000, 1000, 10})->Iterations(1);

static void OLD_Graph_ALL_RUN(benchmark::State &state) {
  auto subgraph_num = state.range(2);
  int node_num = state.range(1);
  int edge_num = state.range(0);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, edge_num);

  auto subgraph_node_num = state.range(3);
  ComputeGraphPtr sub_graph[subgraph_num] = {nullptr};
  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  for (int i = 0; i < subgraph_num; i++) {
    OpDescCreate(subgraph_node_num, sub_op_desc[i], edge_num);
  }

  NodePtr node[node_num] = {};
  ComputeGraphPtr quick_graph[subgraph_num] = {nullptr};
  auto old_root_graph = std::make_shared<ComputeGraph>("root_graph");
  for (auto _ : state) {
    for (int i = 0; i < node_num; i++) {
      node[i] = old_root_graph->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
      if (node[i] == nullptr) {
        std::cout << "OLD_Graph_ALL_RUN add node error." << std::endl;
        return;
      }
#endif
    }

    for (int i = 1; i < node_num; i++) {
      auto ret = GraphUtils::AddEdge(node[i]->GetOutDataAnchor(1), node[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "OLD_Graph_ALL_RUN add edge error" << std::endl;
        return;
      }
#endif
    }

    NodePtr sub_graph_node[subgraph_num][subgraph_node_num] = {};
    for (int i = 0; i < subgraph_num; i++) {
      sub_graph[i] = std::make_shared<ComputeGraph>("subgraph_" + std::to_string(i));
      for (int j = 0; j < subgraph_node_num; j++) {
        sub_graph_node[i][j] = sub_graph[i]->AddNode(sub_op_desc[i][j]);
#if GRAPH_CHECK_RET
        if (sub_graph_node[i][j] == nullptr) {
          std::cout << "OLD_Graph_ALL_RUN add node error." << std::endl;
          return;
        }
#endif
      }
    }

    for (int i = 0; i < subgraph_num; i++) {
      for (int j = 1; j < subgraph_node_num; j++) {
        GraphUtils::AddEdge(sub_graph_node[i][j]->GetOutDataAnchor(1), sub_graph_node[i][j - 1]->GetInDataAnchor(0));
      }
    }

    for (int64_t i = 0; i < subgraph_num; ++i) {
      quick_graph[i] = old_root_graph->AddSubGraph(sub_graph[i]);
#if GRAPH_CHECK_RET
      if (quick_graph[i] == nullptr) {
        std::cout << "2 OLD_Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
    }

    old_root_graph->TopologicalSortingGraph(true);

    for (int i = 1; i < node_num; i++) {
      auto ret = GraphUtils::RemoveEdge(node[i]->GetOutDataAnchor(1), node[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "0 OLD_Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
    }

    for (int i = 0; i < node_num; i++) {
      auto ret = GraphUtils::RemoveJustNode(old_root_graph, node[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "OLD_Graph_ALL_RUN remove node error." << std::endl;
        return;
      }
#endif
    }

    for (int64_t i = 0; i < subgraph_num; ++i) {
      old_root_graph->RemoveSubGraph(quick_graph[i]);
    }

    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_ALL_RUN)->Args({20, 2000, 1000, 10})->Iterations(1);
BENCHMARK(OLD_Graph_ALL_RUN)->Args({20, 4000, 1000, 10})->Iterations(1);
BENCHMARK(OLD_Graph_ALL_RUN)->Args({20, 6000, 1000, 10})->Iterations(1);
BENCHMARK(OLD_Graph_ALL_RUN)->Args({20, 8000, 1000, 10})->Iterations(1);

static void Graph_AddAndRemoveSubgraph_Multi(benchmark::State &state) {
  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  auto subgraph_num = state.range();

  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
  }

  int edge_num = 5;
  FastNode *node[subgraph_num] = {};
  std::shared_ptr<OpDesc> op_desc[subgraph_num] = {nullptr};
  OpDescCreate(subgraph_num, op_desc, edge_num);
  for (int i = 0; i < subgraph_num; ++i) {
    node[i] = root_graph->AddNode(op_desc[i]);
  }

  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    sub_graph[i]->SetParentGraph(root_graph.get());
    sub_graph[i]->SetParentNode(node[i]);
  }

  ExecuteGraph *new_sub_graph[subgraph_num] = {nullptr};
  for (auto _ : state) {
    for (int64_t i = 0; i < subgraph_num; ++i) {
      new_sub_graph[i] = root_graph->AddSubGraph(sub_graph[i]);
#if GRAPH_CHECK_RET
      if (new_sub_graph[i] == nullptr) {
        std::cout << "Graph_RemoveHeadEdge Error" << std::endl;
        return;
      }
#endif
    }
    for (int64_t i = 0; i < subgraph_num; ++i) {
      root_graph->RemoveSubGraph(new_sub_graph[i]);
    }

    benchmark::ClobberMemory();
  }
}
BENCHMARK(Graph_AddAndRemoveSubgraph_Multi)->Arg(10)->Arg(100)->Arg(1000)->Arg(10000);

static void Graph_AddAndRemoveSubgraph_Multi_OLD(benchmark::State &state) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  auto subgraph_num = state.range();
  int edge_num = 5;
  NodePtr node[subgraph_num] = {};
  std::shared_ptr<OpDesc> op_desc[subgraph_num] = {nullptr};
  OpDescCreate(subgraph_num, op_desc, edge_num);
  for (int i = 0; i < subgraph_num; ++i) {
    node[i] = root_graph->AddNode(op_desc[i]);
  }

  ComputeGraphPtr sub_graph[subgraph_num] = {nullptr};
  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ComputeGraph>("subgraph_" + std::to_string(i));
    sub_graph[i]->SetParentGraph(root_graph);
    sub_graph[i]->SetParentNode(node[i]);
  }

  for (auto _ : state) {
    for (int64_t i = 0; i < subgraph_num; ++i) {
      auto ret = root_graph->AddSubGraph(sub_graph[i]);
#if GRAPH_CHECK_RET
      if (ret == nullptr) {
        std::cout << "Graph_RemoveHeadEdge Error" << std::endl;
        return;
      }
#endif
    }
    for (int64_t i = 0; i < subgraph_num; ++i) {
      root_graph->RemoveSubGraph(sub_graph[i]);
    }

    benchmark::ClobberMemory();
  }
}
BENCHMARK(Graph_AddAndRemoveSubgraph_Multi_OLD)->Arg(10)->Arg(100)->Arg(1000)->Arg(10000);

static void TEST_ANCHOR(benchmark::State &state) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  auto edge_num = state.range();
  int node_num = 2;
  NodePtr node[node_num] = {};

  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, edge_num);
  for (int i = 0; i < node_num; ++i) {
    node[i] = root_graph->AddNode(op_desc[i]);
  }

  InDataAnchorPtr ptr[edge_num] = {};
  OutDataAnchorPtr out_ptr[edge_num] = {};

  for (auto _ : state) {
    node[0]->GetAllInAnchors();
    node[0]->GetAllOutAnchors();
    benchmark::ClobberMemory();
  }
}
BENCHMARK(TEST_ANCHOR)->Arg(10)->Arg(100)->Arg(1000)->Iterations(1);

static void TEST_ANCHOR_PEER_GET(benchmark::State &state) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  auto anchor_num = state.range();
  int node_num = 2;
  NodePtr node[node_num] = {};
  InDataAnchorPtr ptr[anchor_num] = {};
  OutDataAnchorPtr out_ptr[anchor_num] = {};
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};

  OpDescCreate(node_num, op_desc, anchor_num);
  for (int i = 0; i < node_num; ++i) {
    node[i] = root_graph->AddNode(op_desc[i]);
  }

  for (int j = 0; j < anchor_num; j++) {
    GraphUtils::AddEdge(node[0]->GetOutDataAnchor(0), node[1]->GetInDataAnchor(j));
  }

  for (auto _ : state) {
    auto ret = node[0]->GetOutDataAnchor(0)->GetPeerAnchors();
    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(TEST_ANCHOR_PEER_GET)->Arg(10)->Arg(100)->Arg(1000);

static void TEST_GET_IN_NODES(benchmark::State &state) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  auto anchor_num = state.range();
  int node_num = 1001;
  NodePtr node[node_num] = {};

  for (int j = 0; j < node_num; j++) {
    OpDescPtr op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int64_t i = 0; i < anchor_num; ++i) {
      op_desc->AddInputDesc(td);
    }
    for (int64_t i = 0; i < anchor_num; ++i) {
      op_desc->AddOutputDesc(td);
    }
    node[j] = root_graph->AddNode(op_desc);
  }

  for (int j = 0; j < anchor_num; j++) {
    GraphUtils::AddEdge(node[0]->GetOutDataAnchor(j), node[j]->GetInDataAnchor(j));
  }

  InDataAnchorPtr ptr[anchor_num] = {};
  OutDataAnchorPtr out_ptr[anchor_num] = {};

  for (auto _ : state) {
    auto ret = node[0]->GetOutNodes();
    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(TEST_GET_IN_NODES)->Arg(10)->Arg(100)->Arg(1000);

static void TEST_OLD_Graph_ALL_RUN(benchmark::State &state) {
  int edge_num = state.range(0);
  int node_num = state.range(1);
  auto subgraph_num = state.range(2);
  auto subgraph_node_num = state.range(3);

  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  ComputeGraphPtr sub_graph[subgraph_num] = {nullptr};
  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  NodePtr node[node_num] = {};
  ComputeGraphPtr quick_graph[subgraph_num] = {nullptr};

  OpDescCreate(node_num, op_desc, edge_num);
  for (int i = 0; i < subgraph_num; i++) {
    OpDescCreate(subgraph_num, sub_op_desc[i], edge_num);
  }

  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  for (auto _ : state) {
    for (int i = 0; i < node_num; i++) {
      node[i] = root_graph->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
      if (node[i] == nullptr) {
        std::cout << "0 OLD_Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
    }

    for (int i = 1; i < node_num; i++) {
      auto ret = GraphUtils::AddEdge(node[i]->GetOutDataAnchor(0), node[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "0 OLD_Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
      ret = GraphUtils::AddEdge(node[i]->GetOutDataAnchor(1), node[i - 1]->GetInDataAnchor(1));
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "0 OLD_Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
    }

    root_graph->TopologicalSortingGraph(true);

    for (int i = 1; i < node_num; i++) {
      auto ret = GraphUtils::RemoveEdge(node[i]->GetOutDataAnchor(0), node[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "0 OLD_Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
      ret = GraphUtils::RemoveEdge(node[i]->GetOutDataAnchor(1), node[i - 1]->GetInDataAnchor(1));
    }

    for (int i = 0; i < node_num; i++) {
      auto ret = GraphUtils::RemoveJustNode(root_graph, node[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "TEST_OLD_Graph_ALL_RUN remove node error." << std::endl;
        return;
      }
#endif
    }

    benchmark::ClobberMemory();
  }
}
BENCHMARK(TEST_OLD_Graph_ALL_RUN)->Args({20, 500, 1, 0})->Iterations(100);

static void OLD_GRAPH_DEEPCOPY(benchmark::State &state) {
  auto subgraph_num = state.range(2);
  int node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  for (int j = 0; j < node_num; j++) {
    op_desc[j] = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int64_t i = 0; i < state.range(0); ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (int64_t i = 0; i < state.range(0); ++i) {
      op_desc[j]->AddOutputDesc(td);
    }
  }

  auto subgraph_node_num = state.range(3);
  ComputeGraphPtr sub_graph[subgraph_num] = {nullptr};

  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  for (int i = 0; i < subgraph_num; i++) {
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_op_desc[i][j] = std::make_shared<OpDesc>("op", "op");
      auto td = GeTensorDesc();

      for (int64_t x = 0; x < state.range(0); ++x) {
        sub_op_desc[i][j]->AddInputDesc(td);
      }
      for (int64_t x = 0; x < state.range(0); ++x) {
        sub_op_desc[i][j]->AddOutputDesc(td);
      }
    }
  }

  NodePtr node[node_num] = {};
  ComputeGraphPtr quick_graph[subgraph_num] = {nullptr};
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");

  for (int i = 0; i < node_num; i++) {
    node[i] = root_graph->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
    if (node[i] == nullptr) {
      std::cout << "0 OLD_Graph_ALL_RUN Error" << std::endl;
      return;
    }
#endif
  }

  for (int i = 1; i < node_num; i++) {
    auto ret = GraphUtils::AddEdge(node[i]->GetOutDataAnchor(1), node[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
    if (ret != GRAPH_SUCCESS) {
      std::cout << "0 OLD_Graph_ALL_RUN Error" << std::endl;
      return;
    }
#endif
  }

  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ComputeGraph>("subgraph_" + std::to_string(i));
    NodePtr sub_graph_node[subgraph_node_num] = {};
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_graph_node[j] = sub_graph[i]->AddNode(sub_op_desc[i][j]);
#if GRAPH_CHECK_RET
      if (sub_graph_node[j] == nullptr) {
        std::cout << "1111 Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
    }
    for (int j = 1; j < subgraph_node_num; j++) {
      GraphUtils::AddEdge(sub_graph_node[j]->GetOutDataAnchor(1), sub_graph_node[j - 1]->GetInDataAnchor(0));
    }
  }

  for (int64_t i = 0; i < subgraph_num; ++i) {
    quick_graph[i] = root_graph->AddSubGraph(sub_graph[i]);
#if GRAPH_CHECK_RET
    if (quick_graph[i] == nullptr) {
      std::cout << "2 OLD_Graph_ALL_RUN Error" << std::endl;
      return;
    }
#endif
  }

  auto test_graph = std::make_shared<ComputeGraph>("test");

  for (auto _ : state) {
    GraphUtils::CopyComputeGraph(root_graph, test_graph);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_GRAPH_DEEPCOPY)->Args({20, 10000, 1, 10})->Iterations(1);
BENCHMARK(OLD_GRAPH_DEEPCOPY)->Args({20, 50000, 10, 10})->Iterations(1);
BENCHMARK(OLD_GRAPH_DEEPCOPY)->Args({20, 50000, 100, 10})->Iterations(1);
BENCHMARK(OLD_GRAPH_DEEPCOPY)->Args({20, 50000, 1000, 10})->Iterations(1);

static void NEW_GRAPH_DEEPCOPY(benchmark::State &state) {
  int node_num = state.range(1);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  for (int j = 0; j < node_num; j++) {
    op_desc[j] = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();

    for (int64_t i = 0; i < state.range(0); ++i) {
      op_desc[j]->AddInputDesc(td);
    }
    for (int64_t i = 0; i < state.range(0); ++i) {
      op_desc[j]->AddOutputDesc(td);
    }
  }

  auto subgraph_num = state.range(2);
  auto subgraph_node_num = state.range(3);
  std::shared_ptr<ExecuteGraph> sub_graph[subgraph_num] = {nullptr};

  FastNode *node[node_num] = {};
  FastEdge *edge[node_num] = {};
  ExecuteGraph *quick_graph[subgraph_num] = {nullptr};
  std::shared_ptr<OpDesc> sub_op_desc[subgraph_num][subgraph_node_num] = {};
  for (int i = 0; i < subgraph_num; i++) {
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_op_desc[i][j] = std::make_shared<OpDesc>("op", "op");
      auto td = GeTensorDesc();

      for (int64_t x = 0; x < state.range(0); ++x) {
        sub_op_desc[i][j]->AddInputDesc(td);
      }
      for (int64_t x = 0; x < state.range(0); ++x) {
        sub_op_desc[i][j]->AddOutputDesc(td);
      }
    }
  }

  auto root_graph = std::make_shared<ExecuteGraph>("root_graph");
  for (int i = 0; i < node_num; i++) {
    node[i] = root_graph->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
    if (node[i] == nullptr) {
      std::cout << "0 Graph_ALL_RUN Error" << std::endl;
      return;
    }
#endif
  }

  for (int i = 1; i < node_num; i++) {
    edge[i] = root_graph->AddEdge(node[i], 1, node[i - 1], 0);
#if GRAPH_CHECK_RET
    if (edge[i] == nullptr) {
      std::cout << "1 Graph_ALL_RUN Add Edge Error " << i << std::endl;
      return;
    }
#endif
  }

  for (int i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ExecuteGraph>("subgraph_" + std::to_string(i));
    FastNode *sub_graph_node[subgraph_node_num] = {};
    for (int j = 0; j < subgraph_node_num; j++) {
      sub_graph_node[j] = sub_graph[i]->AddNode(sub_op_desc[i][j]);
#if GRAPH_CHECK_RET
      if (sub_graph_node[j] == nullptr) {
        std::cout << "1111 Graph_ALL_RUN Error" << std::endl;
        return;
      }
#endif
    }
    for (int j = 1; j < subgraph_node_num; j++) {
      auto ret = sub_graph[i]->AddEdge(sub_graph_node[j], 1, sub_graph_node[j - 1], 0);
#if GRAPH_CHECK_RET
      if (ret == nullptr) {
        std::cout << "1 Graph_ALL_RUN sub graph Error" << j << std::endl;
        return;
      }
#endif
    }
  }

  for (int i = 0; i < subgraph_num; ++i) {
    quick_graph[i] = root_graph->AddSubGraph(sub_graph[i]);
#if GRAPH_CHECK_RET
    if (quick_graph[i] == nullptr) {
      std::cout << "2 Graph_ALL_RUN Error" << std::endl;
      return;
    }
#endif
  }

  auto test1_graph = std::make_shared<ExecuteGraph>("root_graph");
  for (auto _ : state) {
    test1_graph->CompleteCopy(*(root_graph.get()));
    benchmark::ClobberMemory();
  }
}
BENCHMARK(NEW_GRAPH_DEEPCOPY)->Args({20, 10000, 1, 10})->Iterations(1);
BENCHMARK(NEW_GRAPH_DEEPCOPY)->Args({20, 50000, 10, 10})->Iterations(1);
BENCHMARK(NEW_GRAPH_DEEPCOPY)->Args({20, 50000, 100, 10})->Iterations(1);
BENCHMARK(NEW_GRAPH_DEEPCOPY)->Args({20, 50000, 1000, 10})->Iterations(1);

static void TEST_REMOVE_NODE(benchmark::State &state) {
  int node_num = state.range(1);
  int edge_num = state.range(0);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, edge_num);

  NodePtr node[node_num] = {};
  auto old_root_graph = std::make_shared<ComputeGraph>("root_graph");

  for (int i = 0; i < node_num; i++) {
    node[i] = old_root_graph->AddNode(op_desc[i]);
#if GRAPH_CHECK_RET
    if (node[i] == nullptr) {
      std::cout << "OLD_Graph_ALL_RUN add node error." << std::endl;
      return;
    }
#endif
  }

  for (int i = 1; i < node_num; i++) {
    auto ret = GraphUtils::AddEdge(node[i]->GetOutDataAnchor(1), node[i - 1]->GetInDataAnchor(0));
#if GRAPH_CHECK_RET
    if (ret != GRAPH_SUCCESS) {
      std::cout << "OLD_Graph_ALL_RUN add edge error" << std::endl;
      return;
    }
#endif
  }

  for (auto _ : state) {
    for (int i = 0; i < node_num; i++) {
      auto ret = old_root_graph->RemoveNode(node[i]);
#if GRAPH_CHECK_RET
      if (ret != GRAPH_SUCCESS) {
        std::cout << "OLD_Graph_ALL_RUN remove node error." << std::endl;
        return;
      }
#endif
    }

    benchmark::ClobberMemory();
  }
}
BENCHMARK(TEST_REMOVE_NODE)->Args({20, 10000, 2000, 10})->Iterations(1);
BENCHMARK(TEST_REMOVE_NODE)->Args({20, 10000, 4000, 10})->Iterations(1);
BENCHMARK(TEST_REMOVE_NODE)->Args({20, 10000, 6000, 10})->Iterations(1);
BENCHMARK(TEST_REMOVE_NODE)->Args({20, 10000, 10000, 10})->Iterations(1);

static void TEST_GetSubGraph(benchmark::State &state) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  int node_num = state.range(0);
  size_t subgraph_num = state.range(1);
  int edge_num = 5;
  NodePtr node[node_num] = {};
  for (int i = 0; i < node_num; ++i) {
    auto op_desc = std::make_shared<OpDesc>("op", "op");
    auto td = GeTensorDesc();
    for (int j = 0; j < edge_num; ++j) {
      op_desc->AddInputDesc(td);
      op_desc->AddOutputDesc(td);
    }
    node[i] = root_graph->AddNode(op_desc);
  }

  std::shared_ptr<ComputeGraph> sub_graph[subgraph_num] = {nullptr};
  for (size_t i = 0; i < subgraph_num; i++) {
    sub_graph[i] = std::make_shared<ComputeGraph>("subgraph_" + std::to_string(i));
    sub_graph[i]->SetParentGraph(root_graph);
    sub_graph[i]->SetParentNode(node[i]);
  }

  for (size_t i = 0; i < subgraph_num; i++) {
    std::string name = "subgraph_" + std::to_string(i);
    root_graph->AddSubgraph(name, sub_graph[i]);
  }

  for (auto _ : state) {
    auto subgraphs = root_graph->GetAllSubgraphs();
#if GRAPH_CHECK_RET
    if (subgraphs.size() != subgraph_num) {
      std::cout << "0 TEST_GetSubGraph Error" << std::endl;
      exit(1);
    }
#endif
    benchmark::DoNotOptimize(subgraphs);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(TEST_GetSubGraph)->Args({10000, 2000});
BENCHMARK(TEST_GetSubGraph)->Args({10000, 4000});
BENCHMARK(TEST_GetSubGraph)->Args({10000, 6000});
BENCHMARK(TEST_GetSubGraph)->Args({10000, 8000});

static void New_Graph_AddNodeAndUpdateIo(benchmark::State &state) {
  auto compute_graph = std::make_shared<ExecuteGraph>("graph0");
  int node_num = 1;
  int io_num = 10;
  int new_num = state.range(0);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  FastNode *node  = compute_graph->AddNode(op_desc[0]);

  for (auto _ : state) {
    auto ret = FastNodeUtils::AppendInputEdgeInfo(node, new_num);
    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(New_Graph_AddNodeAndUpdateIo)->Arg(11)->Arg(21)->Arg(110)->Arg(1010)->Iterations(1);

static void New_Graph_AddNodeAndUpdateOutput_Step(benchmark::State &state) {
  auto compute_graph = std::make_shared<ExecuteGraph>("graph0");
  int node_num = 1;
  int io_num = 10;
  int new_num = state.range(0);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  FastNode *node  = compute_graph->AddNode(op_desc[0]);

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  for (int i = op_desc[0]->GetOutputsSize(); i < new_num; ++i) {
    op_desc[0]->AddOutputDesc(data_desc);
  }

  for (auto _ : state) {
    for (int i = io_num; i < new_num; ++i) {
      node->UpdateDataOutNum(i);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(New_Graph_AddNodeAndUpdateOutput_Step)->Arg(20)->Arg(110)->Arg(1010)->Iterations(1);

static void New_Graph_AddNodeAndUpdateInput_Step(benchmark::State &state) {
  auto compute_graph = std::make_shared<ExecuteGraph>("graph0");
  int node_num = 1;
  int io_num = 10;
  int new_num = state.range(0);
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  FastNode *node  = compute_graph->AddNode(op_desc[0]);

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  for (int i = op_desc[0]->GetOutputsSize(); i < new_num; ++i) {
    op_desc[0]->AddOutputDesc(data_desc);
  }

  for (auto _ : state) {
    for (int i = io_num; i < new_num; ++i) {
      node->UpdateDataInNum(i);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(New_Graph_AddNodeAndUpdateInput_Step)->Arg(20)->Arg(110)->Arg(1010)->Iterations(1);

static void OLD_Graph_AddNodeAndUpdateIo(benchmark::State &state) {
  auto compute_graph = std::make_shared<ComputeGraph>("graph0");
  int node_num = 1;
  int io_num = 10;
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, io_num);
  NodePtr node  = compute_graph->AddNode(op_desc[0]);
  int new_num = state.range(0);

  for (auto _ : state) {
    auto ret = NodeUtils::AppendInputAnchor(node, new_num);
    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_Graph_AddNodeAndUpdateIo)->Arg(11)->Arg(21)->Arg(110)->Arg(1010)->Iterations(1);

static void OLD_ChangeEdgeAndNodeOwner(benchmark::State &state) {
  auto compute_graph = std::make_shared<ge::ComputeGraph>("graph");
  int node_num = state.range(0);
  int edge_num = 5;
  NodePtr node[node_num] = {};
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, edge_num);

  for (int i = 0; i < node_num; ++i) {
    node[i] = compute_graph->AddNode(op_desc[i]);
    if (node[i] == nullptr) {
      return;
    }
  }

  for (int i = 1; i < node_num; ++i) {
    GraphUtils::AddEdge(node[i - 1]->GetOutDataAnchor(0), node[i]->GetInDataAnchor(0));
    GraphUtils::AddEdge(node[i - 1]->GetOutControlAnchor(), node[i]->GetInControlAnchor());
  }

  auto graph1 = std::make_shared<ge::ComputeGraph>("graph1");
  for (auto _ : state) {
    for (int i = 0; i < node_num; ++i) {
      graph1->AddNode(node[i]);
      GraphUtils::RemoveJustNode(compute_graph, node[i]);
    }

    for (int i = 0; i < node_num; ++i) {
      compute_graph->AddNode(node[i]);
      GraphUtils::RemoveJustNode(graph1, node[i]);
    }

    benchmark::ClobberMemory();
  }
}
BENCHMARK(OLD_ChangeEdgeAndNodeOwner)->Arg(10)->Arg(100)->Arg(1000);

static void NEW_ChangeEdgeAndNodeOwner(benchmark::State &state) {
  auto compute_graph = std::make_shared<ge::ExecuteGraph>("graph");
  int node_num = state.range(0);
  int edge_num = 5;
  FastNode *node[node_num] = {};
  std::shared_ptr<OpDesc> op_desc[node_num] = {nullptr};
  OpDescCreate(node_num, op_desc, edge_num);

  for (int i = 0; i < node_num; ++i) {
    node[i] = compute_graph->AddNode(op_desc[i]);
    if (node[i] == nullptr) {
      return;
    }
  }

  FastEdge *edge[node_num] = {};
  FastEdge *ctrl_edge[node_num] = {};
  for (int i = 1; i < node_num; ++i) {
    edge[i] = compute_graph->AddEdge(node[i - 1], 0, node[i], 0);
    ctrl_edge[i] = compute_graph->AddEdge(node[i - 1], kControlEdgeIndex, node[i], kControlEdgeIndex);
  }

  auto graph1 = std::make_shared<ge::ExecuteGraph>("graph1");
  for (auto _ : state) {
    for (int i = 0; i < node_num; ++i) {
      graph1->AddNode(node[i]);
    }

    for (int i = 0; i < node_num; ++i) {
      auto &edges = node[i]->GetAllInDataEdgesRef();
      for (auto edge : edges) {
        if (edge != nullptr) {
          graph1->MoveEdgeToGraph(edge);
        }
      }
    }

    for (int i = 0; i < node_num; ++i) {
      compute_graph->AddNode(node[i]);
    }

    for (int i = 0; i < node_num; ++i) {
      auto &edges = node[i]->GetAllInDataEdgesRef();
      for (auto edge : edges) {
        if (edge != nullptr) {
          compute_graph->MoveEdgeToGraph(edge);
        }
      }
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(NEW_ChangeEdgeAndNodeOwner)->Arg(10)->Arg(100)->Arg(1000);

BENCHMARK_MAIN();
