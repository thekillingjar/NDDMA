/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_showcase.h"// es构图方式
#include "es_Add.h"
#include "es_Relu.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <iostream>
#include "ge/ge_api.h"

using namespace ge;
using namespace ge::es;
namespace {
es::EsTensorHolder MakeReluAddGraph(es::EsTensorHolder input) {
  /*
  REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_QINT8, DT_BF16}))
    .OP_END_FACTORY_REG(Relu)
  */
  auto relu1 = Relu(input);
  auto relu2 = Relu(relu1);
  auto add = Add(input, relu2);
  return add;
}
}
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
  static uint32_t next = 0;
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

void MakeReluAddGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeReluAddGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_relu_add_graph"));
}

std::unique_ptr<ge::Graph> MakeReluAddGraphByEs() {
  // 1、创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeReluAddGraph");
  // 2、创建输入节点
  auto input = graph_builder->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_ND, {2, 3});
  auto result = MakeReluAddGraph(input);
  // 3、设置输出
  (void) graph_builder->SetOutput(result, 0);
  // 4、构建图
  auto graph = graph_builder->BuildAndReset();
  return graph;
}

int MakeReluAddGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeReluAddGraphByEs();
  std::vector<ge::Tensor> inputs;

  // 准备输入数据
  std::vector<float> input_data(2 * 3, 1.0f);

  // 创建输入tensor
  auto input_tensor = ge::Utils::StubTensor<float>(input_data, {2, 3});
  inputs.push_back(*input_tensor);
  return RunGraph(*graph, inputs, "ReluAdd");
}

}  // namespace es_showcase