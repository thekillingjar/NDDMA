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
#include "es_ConcatV2.h"
#include "utils.h"
#include <memory>
#include "ge/ge_api.h"
using namespace ge;
using namespace ge::es;
namespace {
es::EsTensorHolder MakeConcatV2Graph(es::EsTensorHolder tensor1, es::EsTensorHolder tensor2, es::EsTensorHolder tensor3,
                                         es::EsTensorHolder concat_dim) {
  return ConcatV2({tensor1, tensor2, tensor3}, concat_dim, 3);
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
void MakeConcatV2GraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeConcatV2GraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_concatv2_graph"));
}
std::unique_ptr<ge::Graph> MakeConcatV2GraphByEs() {
  // 1、创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeConcatV2Graph");
  // 2、创建输入节点 在第0维拼接tensor1、tensor2、tensor3
  auto tensor1 = graph_builder->CreateInput(0, "tensor1", ge::DT_FLOAT, ge::FORMAT_ND, {8, 64, 128});
  auto tensor2 = graph_builder->CreateInput(1, "tensor2", ge::DT_FLOAT, ge::FORMAT_ND, {2, 64, 128});
  auto tensor3 = graph_builder->CreateInput(2, "tensor3", ge::DT_FLOAT, ge::FORMAT_ND, {6, 64, 128});
  auto concat_dim = graph_builder->CreateScalar(0);
  /*
    ConcatV2原型注释：
    REG_OP(ConcatV2)
      .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
      .INPUT(concat_dim, TensorType::IndexNumberType())
      .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
      .ATTR(N, Int, 1)
      .OP_END_FACTORY_REG(ConcatV2)
  */
  auto result = MakeConcatV2Graph(tensor1, tensor2, tensor3, concat_dim);
  // 3、设置输出
  (void) graph_builder->SetOutput(result, 0);
  // 4、构建图
  auto graph = graph_builder->BuildAndReset();
  return graph;
}
int MakeConcatV2GraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeConcatV2GraphByEs();
  std::vector<ge::Tensor> inputs;
  std::vector<float> tensor1_data(8 * 64 * 128, 1.0f);
  std::vector<float> tensor2_data(2 * 64 * 128, 2.0f);
  std::vector<float> tensor3_data(6 * 64 * 128, 3.0f);
  inputs.push_back(*ge::Utils::StubTensor<float>(tensor1_data, {8, 64, 128}));
  inputs.push_back(*ge::Utils::StubTensor<float>(tensor2_data, {2, 64, 128}));
  inputs.push_back(*ge::Utils::StubTensor<float>(tensor3_data, {6, 64, 128}));
  return RunGraph(*graph, inputs, "ConcatV2");
}
}

