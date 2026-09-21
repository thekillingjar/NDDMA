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
#include "es_If.h"
#include "utils.h"
#include <memory>
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
    delete s;
    return -1;
  }
  static uint32_t next =0;
  const uint32_t graph_id = next++;
  auto ret = s->AddGraph(graph_id, graph);
  if (ret != ge::SUCCESS) {
    std::cout << "AddGraph failed" << std::endl;
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

std::unique_ptr<ge::Graph> MakeIfGraphByEs() {
    /*
    If算子原型注释：
    REG_OP(If)
      .INPUT(cond, TensorType::ALL())
      .DYNAMIC_INPUT(input, TensorType::ALL())
      .DYNAMIC_OUTPUT(output, TensorType::ALL())
      .GRAPH(then_branch)
      .GRAPH(else_branch)
      .OP_END_FACTORY_REG(If)
    */
    // 初始化图构建器实例，用于提供构图所需的上下文、工作空间及构建相关方法
    auto graph_builder = std::make_unique<EsGraphBuilder>("MakeIfGraph");
    // MakeIfGraph实例创建输入节点
    auto input = graph_builder->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_ND, {2});
    // 创建权重常量节
    auto cond = graph_builder->CreateInput(1, "cond", ge::DT_INT32, ge::FORMAT_ND, {});
    // 初始化图构建器实例，用于构建子图then_branch
    auto then_graph_builder = std::make_unique<EsGraphBuilder>("then_branch");
    auto then_input = then_graph_builder->CreateInput(0, "then_input", ge::DT_FLOAT, ge::FORMAT_ND, {2});
    auto then_result = then_input * then_graph_builder->CreateScalar(2.0f);
    // 将 then_result设置为then_graph_builder的输出节点
    (void) then_graph_builder->SetOutput(then_result, 0);
    // 构建 then_branch子图
    auto then_branch = then_graph_builder->BuildAndReset();
    auto else_graph_builder = std::make_unique<EsGraphBuilder>("else_branch");
    auto else_input = else_graph_builder->CreateInput(0, "else_input", ge::DT_FLOAT, ge::FORMAT_ND, {2});
    auto else_result = else_input * else_graph_builder->CreateScalar(0.0f);
    (void) else_graph_builder->SetOutput(else_result, 0);
    auto else_branch = else_graph_builder->BuildAndReset();
    std::vector<EsTensorHolder> if_inputs;
    if_inputs.push_back(input);
    auto if_result = If(cond, if_inputs, 1, std::move(then_branch), std::move(else_branch));
    return graph_builder->BuildAndReset({if_result[0]});
}
void MakeIfGraphByEsAndDump() {
    std::unique_ptr<ge::Graph> graph = MakeIfGraphByEs();
    graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_if_graph"));
}
int MakeIfGraphByEsAndRun(int32_t cond_value) {
    std::unique_ptr<ge::Graph> graph = MakeIfGraphByEs();
    std::vector<ge::Tensor> inputs;
    std::vector<float> input_data = {1.0f, 2.0f};
    inputs.push_back(*ge::Utils::StubTensor<float>(input_data, {2}));
    inputs.push_back(*ge::Utils::StubTensor<int32_t>({cond_value}, {}));
    return RunGraph(*graph, inputs, "If");
}
}