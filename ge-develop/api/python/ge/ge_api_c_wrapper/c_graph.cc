/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/graph.h"
#include "utils/graph_utils.h"
#include "utils/graph_utils_ex.h"
#include "common/checker.h"
#include "ge_api_c_wrapper_utils.h"

#include <iostream>
#include <sstream>

#ifdef __cplusplus

using namespace ge;
using namespace ge::c_wrapper;
extern "C" {
#endif

Graph *GeApiWrapper_Graph_CreateGraph(const char *name) {
  auto graph = new (std::nothrow) ge::Graph(name);
  GE_ASSERT_NOTNULL(graph);
  (void)graph->SetValid();
  return graph;
}
void GeApiWrapper_Graph_DestroyGraph(const Graph *graph) {
  delete graph;
}

void GeApiWrapper_Graph_FreeGraphArray(Graph **graphs) {
  delete[] graphs;
}

GNode **GeApiWrapper_Graph_GetAllNodes(const Graph *graph, size_t *node_num) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(node_num);
  return VecGNodesToArray(graph->GetAllNodes(), node_num);
}

const char *GeApiWrapper_Graph_GetName(const Graph *graph) {
  GE_ASSERT_NOTNULL(graph);
  AscendString name;
  GE_ASSERT_GRAPH_SUCCESS(graph->GetName(name));
  return AscendStringToChar(name);
}

graphStatus GeApiWrapper_Graph_GetAttr(const Graph *graph, const char *key, void *attr_value) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(key);
  GE_ASSERT_NOTNULL(attr_value);
  auto *av = static_cast<AttrValue *>(attr_value);
  AscendString attr_name(key);
  return graph->GetAttr(attr_name, *av);
}

graphStatus GeApiWrapper_Graph_SetAttr(Graph *graph, const char *key, const void *attr_value) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(key);
  GE_ASSERT_NOTNULL(attr_value);
  auto *av = static_cast<const AttrValue *>(attr_value);
  AscendString attr_name(key);
  return graph->SetAttr(attr_name, *av);
}

graphStatus GeApiWrapper_Graph_Dump_To_Onnx(Graph *graph, const char *path, const char *suffix) {
  GE_ASSERT_NOTNULL(graph);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  GE_ASSERT_NOTNULL(compute_graph);
  ge::GraphUtils::DumpGrphToOnnx(*compute_graph, path, suffix);
  return ge::GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_Graph_Dump_To_File(const Graph *graph, int32_t format, const char *suffix) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(suffix);
  AscendString suffix_str(suffix);
  return graph->DumpToFile(static_cast<ge::Graph::DumpFormat>(format), suffix_str);
}

const char *GeApiWrapper_Graph_Dump_To_Stream(const Graph *graph, int32_t format) {
  GE_ASSERT_NOTNULL(graph);
  std::ostringstream oss;
  GE_ASSERT_GRAPH_SUCCESS(graph->Dump(static_cast<ge::Graph::DumpFormat>(format), oss));
  std::string dump_result = oss.str();
  return ge::c_wrapper::MallocCopyString(dump_result.c_str());
}

graphStatus GeApiWrapper_Graph_SaveToAir(const Graph *graph, const char_t *file_name) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(file_name);
  return graph->SaveToFile(file_name);
}

graphStatus GeApiWrapper_Graph_LoadFromAir(Graph *graph, const char_t *file_name) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(file_name);
  return graph->LoadFromFile(file_name);
}

GNode **GeApiWrapper_Graph_GetDirectNode(const Graph *graph, size_t *node_num) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(node_num);
  return VecGNodesToArray(graph->GetDirectNode(), node_num);
}

graphStatus GeApiWrapper_Graph_RemoveEdge(Graph *graph, GNode &src_node, const int32_t src_port_index,
                                          GNode &dst_node, const int32_t dst_port_index) {
  GE_ASSERT_NOTNULL(graph);
  return graph->RemoveEdge(src_node, src_port_index, dst_node, dst_port_index);
}

graphStatus GeApiWrapper_Graph_AddDataEdge(Graph *graph, GNode &src_node, const int32_t src_port_index,
                                           GNode &dst_node, const int32_t dst_port_index) {
  GE_ASSERT_NOTNULL(graph);
  return graph->AddDataEdge(src_node, src_port_index, dst_node, dst_port_index);
}

graphStatus GeApiWrapper_Graph_AddControlEdge(Graph *graph, GNode &src_node, GNode &dst_node) {
  GE_ASSERT_NOTNULL(graph);
  return graph->AddControlEdge(src_node, dst_node);
}

graphStatus GeApiWrapper_Graph_RemoveNode(Graph *graph, GNode &node) {
  GE_ASSERT_NOTNULL(graph);
  return graph->RemoveNode(node);
}

graphStatus GeApiWrapper_Graph_FindNodeByName(const Graph *graph, const char *name, GNode **node) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(name);
  AscendString node_name(name);
  auto node_found_ptr = graph->FindNodeByName(node_name);
  GE_ASSERT_NOTNULL(node_found_ptr);
  *node = new (std::nothrow) GNode(*node_found_ptr);
  return ge::GRAPH_SUCCESS;
}

Graph **GeApiWrapper_Graph_GetAllSubgraphs(const Graph *graph, size_t *subgraph_num) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(subgraph_num);
  auto subgraphs = graph->GetAllSubgraphs();
  if (subgraphs.empty()) {
    return nullptr;
  }
  *subgraph_num = subgraphs.size();
  auto c_graphs = new (std::nothrow) Graph *[*subgraph_num];
  GE_ASSERT_NOTNULL(c_graphs);
  for (size_t i = 0U; i < *subgraph_num; ++i) {
    c_graphs[i] = new (std::nothrow) Graph(*subgraphs.at(i));
    GE_ASSERT_NOTNULL(c_graphs[i]);
  }
  return c_graphs;
}

Graph *GeApiWrapper_Graph_GetSubGraph(const Graph *graph, const char *name) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(name);
  auto subgraph_ptr = graph->GetSubGraph(name);
  if (subgraph_ptr == nullptr) {
    return nullptr;
  }
  return new (std::nothrow) Graph(*subgraph_ptr);
}

graphStatus GeApiWrapper_Graph_AddSubGraph(Graph *graph, const Graph *subgraph) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(subgraph);
  return graph->AddSubGraph(*subgraph);
}

graphStatus GeApiWrapper_Graph_RemoveSubgraph(Graph *graph, const char *name) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(name);
  return graph->RemoveSubgraph(name);
}

#ifdef __cplusplus
}
#endif
