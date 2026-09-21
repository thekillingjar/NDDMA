/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_Split.h"
#include "utils.h"
#include <memory>
#include "ge/ge_api.h"
using namespace ge;
using namespace ge::es;
namespace {
std::vector<es::EsTensorHolder> MakeSplitGraph(es::EsTensorHolder input, EsGraphBuilder &graph_builder) {
  auto split = Split(graph_builder.CreateScalar(0), input, 4, 4);
  return split;
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

std::unique_ptr<ge::Graph> MakeSplitGraphByEs() {
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeSplitGraph");
  auto input = graph_builder->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_ND, {8, 16, 64});
  /*
  Split算子原型注释：
  REG_OP(Split) 
    .INPUT(split_dim, TensorType({DT_INT32})) 
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, 
     DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, 
     DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, 
     DT_BF16, DT_BOOL})) 
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, 
     DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, 
     DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, 
     DT_BF16, DT_BOOL})) 
    .REQUIRED_ATTR(num_split, Int) 
    .OP_END_FACTORY_REG(Split)
  */
  auto results = MakeSplitGraph(input, *graph_builder);
  for (size_t i = 0; i < results.size(); ++i) {
    (void) graph_builder->SetOutput(results[i], static_cast<int>(i));
  }
  return graph_builder->BuildAndReset();
}

void MakeSplitGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeSplitGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_split_graph"));
}

int MakeSplitGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeSplitGraphByEs();
  std::vector<ge::Tensor> inputs;
  std::vector<float> input_data(8 * 16 * 64);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100);
  }
  inputs.push_back(*ge::Utils::StubTensor<float>(input_data, {8, 16, 64}));
  return RunGraph(*graph, inputs, "Split");
}
}   
