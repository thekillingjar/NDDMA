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
#include "es_BatchNorm.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <iostream>
#include "ge/ge_api.h"

using namespace ge;
using namespace ge::es;
namespace {
es::EsTensorHolder MakeBatchNormGraph(es::EsTensorHolder input, es::EsTensorHolder mean, es::EsTensorHolder variance, EsGraphBuilder &graph_builder) {
  auto scale = graph_builder.CreateConst(std::vector<float>{1.0f, 1.0f, 1.0f}, std::vector<int64_t>{3});
  auto offset = graph_builder.CreateConst(std::vector<float>{0.0f, 0.0f, 0.0f}, std::vector<int64_t>{3});
  auto batchnorm = BatchNorm(input, scale, offset, mean, variance, 1e-4f, "NCHW", false);
  return batchnorm.y;
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

void MakeBatchNormGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeBatchNormGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_batchnorm_garph"));
}

std::unique_ptr<ge::Graph> MakeBatchNormGraphByEs() {
  // 1、创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeBatchNormGarph");
  // 2、创建输入节点
  auto input = graph_builder->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 1, 2});
  auto mean = graph_builder->CreateInput(1, "mean", ge::DT_FLOAT, ge::FORMAT_ND, {3});
  auto variance = graph_builder->CreateInput(2, "variance", ge::DT_FLOAT, ge::FORMAT_ND, {3});
  auto result = MakeBatchNormGraph(input, mean, variance, *graph_builder);
  // 3、设置输出
  (void) graph_builder->SetOutput(result, 0);
  // 4、构建图
  auto graph = graph_builder->BuildAndReset();
  return graph;
}

int MakeBatchNormGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeBatchNormGraphByEs();
  std::vector<ge::Tensor> inputs;

  // 准备输入数据 
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> mean_data = {2.0f, 4.0f, 6.0f};
  std::vector<float> variance_data = {0.5f, 0.5f, 0.5f};

  // 创建输入tensor
  auto input_tensor = ge::Utils::StubTensor<float>(input_data, {1, 3, 1, 2}, ge::FORMAT_NCHW);
  auto mean_tensor = ge::Utils::StubTensor<float>(mean_data, {3});
  auto variance_tensor = ge::Utils::StubTensor<float>(variance_data, {3});
  inputs.push_back(*input_tensor);
  inputs.push_back(*mean_tensor);
  inputs.push_back(*variance_tensor);

  return RunGraph(*graph, inputs, "BatchNorm");
}

}  // namespace es_showcase
