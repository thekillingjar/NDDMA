/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_showcase.h"
#include "es_Assign.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <iostream>
#include "ge/ge_api.h"
using namespace ge;
using namespace ge::es;
namespace es_showcase {

int RunGraph(ge::Graph &graph, const std::vector<ge::Tensor> &inputs,
             const std::string &output_prefix) {
  ge::Utils::PrintTensorsToFile(inputs, "input");
  std::map<ge::AscendString, ge::AscendString> options;
  auto *s = new (std::nothrow) ge::Session(options);
  if (s == nullptr) {
    std::cout << "Global session not ready" << std::endl;
    return -1;
  }
  static uint32_t next =0;
  const uint32_t graph_id = next++;
  auto ret = s->AddGraph(graph_id, graph);
  if (ret != ge::SUCCESS) {
    std::cout << "AddGraph failed" << std::endl;
    delete s;
    return -1;
  }
  std::vector<ge::Tensor> outputs;
  ret = s->RunGraph(graph_id, inputs, outputs);
  if (ret != ge::SUCCESS) {
    std::cout << "RunGraph failed" << std::endl;
    (void)s->RemoveGraph(graph_id);
    delete s;
    return -1;
  }
  (void)s->RemoveGraph(graph_id);
  ge::Utils::PrintTensorsToFile(outputs, output_prefix);
  delete s;
  return 0;
}

std::unique_ptr<ge::Graph> MakeControlEdgeGraphByEs() {
  // 初始化图构建器实例，用于提供构图所需的上下文、工作空间及构建相关方法
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeControlEdgeGraph");
  // 创建两个输入节点，起始节点指无输入依赖的节点，通常包括图的输入（如 Data 节点）和权重常量（如 Const 节点）
  auto input1 = graph_builder->CreateInput(0, "input1", ge::DT_FLOAT, ge::FORMAT_ND, {2});
  auto input2 = graph_builder->CreateInput(1, "input2", ge::DT_FLOAT, ge::FORMAT_ND, {2});
  // 创建一个shape为2的可变输入节点
  auto variable = graph_builder->CreateVariable(0, "var").SetShape({2});
  // 将input2的值赋值给variable assign_result为中间节点
  auto assign_result = Assign(variable, input2);
  // add_result 为输出节点，作为计算结果的终点
  auto add_result = variable + input1;
  // 为add_result添加控制边依赖（assign_result计算需先执行完成，再执行add_result计算）
  add_result.AddControlEdge({assign_result});
  return graph_builder->BuildAndReset({add_result});
}
void MakeControlEdgeGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeControlEdgeGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_control_edge_graph"));
}

int MakeControlEdgeGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeControlEdgeGraphByEs();
  std::vector<ge::Tensor> inputs;
  std::vector<float> input1_data = {1.0f, 2.0f};
  std::vector<float> input2_data = {3.0f, 4.0f};
  inputs.push_back(*ge::Utils::StubTensor<float>(input1_data, {2}));
  inputs.push_back(*ge::Utils::StubTensor<float>(input2_data, {2}));
  return RunGraph(*graph, inputs, "ControlEdge");
}
}