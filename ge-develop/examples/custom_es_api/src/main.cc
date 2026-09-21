/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cstdlib>
#include <iostream>
#include <string>


#include "es_Add.h"
#include "graph/graph.h"
#include "my_ConcatD.h"
#include "my_Conv2D.h"
#include "my_IdentityN.h"


using namespace ge;
using namespace ge::es;

void DumpGraph(ge::Graph &graph, const std::string &prefix) {
     graph.DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString(prefix.c_str()));
}

std::unique_ptr<ge::Graph> MakeOwnGraphByEs() {
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeOwnGraph");
  auto input0 = graph_builder->CreateInput(0, "input0", ge::DT_INT64, ge::FORMAT_ND, {2, 3});
  auto input1 = graph_builder->CreateInput(1, "input1", ge::DT_INT64, ge::FORMAT_ND, {2, 3});
  auto add = Add(input0, input1);
  auto conv2d = MyConv2D(add, ge::FORMAT_NHWC, input1, ge::FORMAT_NHWC, nullptr, nullptr, {1}, {1});
  auto identityN = MyIdentityN({add, conv2d});
  auto concatD = MyConcatD(identityN, 1);
  (void)graph_builder->SetOutput(concatD, 0);
  return graph_builder->BuildAndReset();
}

int main() {
  std::cout << "start main func" << std::endl;
  std::unique_ptr<ge::Graph> graph = MakeOwnGraphByEs();
  DumpGraph(*graph, "make_own_graph");
  std::cout << "dumped graph name is make_own_graph" << std::endl;
}