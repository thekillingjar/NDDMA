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
#include "utils.h"
#include <memory>
#include "ge/ge_api.h"
using namespace ge;
using namespace ge::es;
namespace {
es::EsTensorHolder MakeAddSubMulDivGraph(es::EsTensorHolder input1, es::EsTensorHolder input2) {
  /*
Add原型注释：
REG_OP(Add)
.INPUT(x1, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
.INPUT(x2, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
.OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
.OP_END_FACTORY_REG(Add)
*/
  // 操作符重载
  auto add_result = input1 + input2;
  /*
Sub原型注释：
REG_OP(Sub)
  .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                         DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                         DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
  .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                         DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                         DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                         DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                         DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
  .OP_END_FACTORY_REG(Sub)

*/
  // 操作符重载
  auto sub_result = input1 - input2;
  /*
Mul原型注释：
REG_OP(Mul)
  .INPUT(x1, "T1")
  .INPUT(x2, "T2")
  .OUTPUT(y, "T3")
  .DATATYPE(T1, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                            DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                            DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
  .DATATYPE(T2, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                            DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                            DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
  .DATATYPE(T3, Promote({"T1", "T2"}))
  .OP_END_FACTORY_REG(Mul)
*/
  // 操作符重载
  auto mul_result = input1 * input2;
  /*
Div原型注释：
REG_OP(Div)
  .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                         DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                         DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
  .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                         DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                         DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                         DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                         DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
  .OP_END_FACTORY_REG(Div)
*/
  auto div_result = input1 / input2;
  auto add_sub_mul_div = add_result + sub_result + mul_result + div_result;
  return add_sub_mul_div;
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
    return -1;
  }
  std::vector<ge::Tensor> outputs;
  ret = s->RunGraph(graph_id, inputs, outputs);
  if (ret != ge::SUCCESS) {
    std::cout << "RunGraph failed" << std::endl;
    (void)s->RemoveGraph(graph_id);
    return -1;
  }
  (void)s->RemoveGraph(graph_id);
  ge::Utils::PrintTensorsToFile(outputs, output_prefix);
  return 0;
}
std::unique_ptr<ge::Graph> MakeAddSubMulDivGraphByEs() {
  // 1、创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeAddSubMulDivGraph");
  // 2、创建输入节点
  auto input1 = graph_builder->CreateInput(0, "input1", ge::DT_FLOAT, ge::FORMAT_ND, {2, 2});
  auto input2 = graph_builder->CreateInput(1, "input2", ge::DT_FLOAT, ge::FORMAT_ND, {2, 2});
  auto result = MakeAddSubMulDivGraph(input1, input2);
  // 3、设置输出
  (void) graph_builder->SetOutput(result, 0);
  // 4、构建图
  return graph_builder->BuildAndReset();
}
void MakeAddSubMulDivGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeAddSubMulDivGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_add_sub_mul_div_graph"));
}
int MakeAddSubMulDivGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeAddSubMulDivGraphByEs();
  std::vector<ge::Tensor> inputs;
  inputs.push_back(*ge::Utils::StubTensor<float>({1.0, 2.0, 3.0, 4.0}, {2, 2}));
  inputs.push_back(*ge::Utils::StubTensor<float>({1.0, 2.0, 3.0, 4.0}, {2, 2}));
  return RunGraph(*graph, inputs, "AddSubMulDiv");
}
}