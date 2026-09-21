/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_showcase.h"
#include "es_PromptFlashAttention.h"
#include "es_Reshape.h"
#include "es_BatchMatMul.h"
#include "es_HcomAllReduce.h"
#include "es_Cast.h"
#include "es_AddRmsNorm.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <iostream>
#include "ge/ge_api.h"

using namespace ge;
using namespace ge::es;

namespace {
// Core graph building function
std::vector<es::EsTensorHolder> MakePfaHcomGraph(
    es::EsTensorHolder query, es::EsTensorHolder key, es::EsTensorHolder value,
    es::EsTensorHolder atten_mask, es::EsTensorHolder quant_scale2, es::EsTensorHolder quant_offset2,
    es::EsTensorHolder mm_x2, es::EsTensorHolder arn_x1, es::EsTensorHolder arn_gamma,
    EsGraphBuilder &graph_builder) {
  auto query_fp16 = Cast(query, DT_FLOAT16);
  auto key_fp16 = Cast(key, DT_FLOAT16);
  auto value_fp16 = Cast(value, DT_FLOAT16);
  auto atten_mask_fp16 = Cast(atten_mask, DT_FLOAT16);
  auto quant_scale2_fp16 = Cast(quant_scale2, DT_FLOAT16);
  auto quant_offset2_fp16 = Cast(quant_offset2, DT_FLOAT16);
  auto mm_x2_fp16 = Cast(mm_x2, DT_FLOAT16);
  auto arn_x1_fp16 = Cast(arn_x1, DT_FLOAT16);
  auto arn_gamma_fp16 = Cast(arn_gamma, DT_FLOAT16);
  auto pfa_output = PromptFlashAttention(
    query_fp16, key_fp16, value_fp16,
    nullptr, atten_mask_fp16, nullptr, nullptr,
    nullptr, nullptr, nullptr,
    quant_scale2_fp16, quant_offset2_fp16,
    8, 1.0f, 214748647, 0, "BSH", 8, 0, 1
  );

  auto reshape_output = Reshape(pfa_output, std::vector<int64_t>{2, 128, 512});
  auto mm_output = BatchMatMul(reshape_output, mm_x2_fp16);
  auto hcom_output = HcomAllReduce(mm_output, "sum", "hccl_world_group");
  auto cast_output = Cast(hcom_output, DT_FLOAT16);
  auto arn_output = AddRmsNorm(arn_x1_fp16, cast_output, arn_gamma_fp16);

  auto mm_output_fp32 = Cast(mm_output, DT_FLOAT);
  auto hcom_output_fp32 = Cast(hcom_output, DT_FLOAT);
  auto arn_output_fp32 = Cast(arn_output.y, DT_FLOAT);

  return {mm_output_fp32, hcom_output_fp32, arn_output_fp32};
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

void MakePfaHcomGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakePfaHcomGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_pfa_hcom_graph"));
}

std::unique_ptr<ge::Graph> MakePfaHcomGraphByEs() {
  // 1. 创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakePfaHcomGraph");

  // 2. 创建输入节点 - 使用FP32，在图内部转换为FP16
  auto query = graph_builder->CreateInput(0, "query", ge::DT_FLOAT, ge::FORMAT_ND, {2, 128, 512});
  auto key = graph_builder->CreateInput(1, "key", ge::DT_FLOAT, ge::FORMAT_ND, {2, 128, 512});
  auto value = graph_builder->CreateInput(2, "value", ge::DT_FLOAT, ge::FORMAT_ND, {2, 128, 512});
  auto atten_mask = graph_builder->CreateInput(3, "atten_mask", ge::DT_FLOAT, ge::FORMAT_ND, {2, 128, 128});
  auto quant_scale2 = graph_builder->CreateInput(4, "quant_scale2", ge::DT_FLOAT, ge::FORMAT_ND, {1});
  auto quant_offset2 = graph_builder->CreateInput(5, "quant_offset2", ge::DT_FLOAT, ge::FORMAT_ND, {1});
  auto mm_x2 = graph_builder->CreateInput(6, "mm_x2", ge::DT_FLOAT, ge::FORMAT_ND, {2, 512, 512});
  auto arn_x1 = graph_builder->CreateInput(7, "arn_x1", ge::DT_FLOAT, ge::FORMAT_ND, {2, 128, 512});
  auto arn_gamma = graph_builder->CreateInput(8, "arn_gamma", ge::DT_FLOAT, ge::FORMAT_ND, {512});

  // 3. 构建图
  auto outputs = MakePfaHcomGraph(query, key, value, atten_mask, quant_scale2, quant_offset2,
                                   mm_x2, arn_x1, arn_gamma, *graph_builder);

  // 4. 设置输出
  for (size_t i = 0; i < outputs.size(); ++i) {
    (void) graph_builder->SetOutput(outputs[i], i);
  }

  // 5. 构建图
  return graph_builder->BuildAndReset();
}

int MakePfaHcomGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakePfaHcomGraphByEs();
  std::vector<ge::Tensor> inputs;

  // 准备输入数据
  std::vector<float> query_data(2 * 128 * 512, 1.0f);
  std::vector<float> key_data(2 * 128 * 512, 1.0f);
  std::vector<float> value_data(2 * 128 * 512, 1.0f);
  std::vector<float> atten_mask_data(2 * 128 * 128, 0.0f);
  std::vector<float> quant_scale2_data(1, 1.0f);
  std::vector<float> quant_offset2_data(1, 0.0f);
  std::vector<float> mm_x2_data(2 * 512 * 512, 1.0f);
  std::vector<float> arn_x1_data(2 * 128 * 512, 1.0f);
  std::vector<float> arn_gamma_data(512, 1.0f);

  // 使用FP32类型创建输入（图内部会转换为FP16）
  inputs.push_back(*ge::Utils::StubTensor<float>(query_data, {2, 128, 512}));
  inputs.push_back(*ge::Utils::StubTensor<float>(key_data, {2, 128, 512}));
  inputs.push_back(*ge::Utils::StubTensor<float>(value_data, {2, 128, 512}));
  inputs.push_back(*ge::Utils::StubTensor<float>(atten_mask_data, {2, 128, 128}));
  inputs.push_back(*ge::Utils::StubTensor<float>(quant_scale2_data, {1}));
  inputs.push_back(*ge::Utils::StubTensor<float>(quant_offset2_data, {1}));
  inputs.push_back(*ge::Utils::StubTensor<float>(mm_x2_data, {2, 512, 512}));
  inputs.push_back(*ge::Utils::StubTensor<float>(arn_x1_data, {2, 128, 512}));
  inputs.push_back(*ge::Utils::StubTensor<float>(arn_gamma_data, {512}));

  return RunGraph(*graph, inputs, "PfaHcomTP");
}

}  // namespace es_showcase
