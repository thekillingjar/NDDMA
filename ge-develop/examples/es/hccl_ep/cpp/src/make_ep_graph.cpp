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
#include "es_DynamicQuant.h"
#include "es_HcomAllGather.h"
#include "es_MatMul.h"
#include "es_MoeGatingTopKSoftmax.h"
#include "es_Cast.h"
#include "es_Unsqueeze.h"
#include "es_ConcatV2.h"
#include "es_SplitV.h"
#include "es_Round.h"
#include "es_Reshape.h"
#include "es_GroupedMatmulFinalizeRouting.h"
#include "es_HcomReduceScatter.h"
#include "es_Add.h"
#include "es_GatherV2.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <iostream>
#include "ge/ge_api.h"

using namespace ge;
using namespace ge::es;

namespace {
const int64_t LOCAL_BATCH = 4;      
const int64_t HIDDEN_SIZE = 512;    
const int64_t NUM_EXPERTS = 8;      
const int64_t TOP_K = 2;
const int64_t RANK_SIZE = 2;

es::EsTensorHolder MakeEPGraph(
    es::EsTensorHolder hidden_states,
    es::EsTensorHolder gate_weight,
    es::EsTensorHolder finished,
    es::EsTensorHolder shared_output,
    es::EsTensorHolder expert_weight,
    es::EsTensorHolder group_list,
    EsGraphBuilder &graph_builder) {
  auto [hidden_states_int8, pertoken_scale] = DynamicQuant(hidden_states, nullptr, nullptr, ge::DT_INT8);
  const char* group = "hccl_world_group";
  auto global_hidden_states = HcomAllGather(hidden_states_int8, RANK_SIZE, group, 0, -1);
  auto router_logits = MatMul(hidden_states, gate_weight);
  auto [topk_weights, topk_ids, row_idx] = MoeGatingTopKSoftmax(router_logits, finished, TOP_K);
  auto topk_ids_float = Cast(topk_ids, ge::DT_FLOAT);
  auto row_idx_float = Cast(row_idx, ge::DT_FLOAT);
  auto pertoken_scale_unsqueezed = Unsqueeze(pertoken_scale, std::vector<int64_t>{-1});
  auto topk_cat = ConcatV2({topk_weights, topk_ids_float, row_idx_float, pertoken_scale_unsqueezed}, static_cast<int64_t>(-1), 4);
  auto topk_all = HcomAllGather(topk_cat, RANK_SIZE, group, 0, -1);
  auto split_results = SplitV(topk_all, graph_builder.CreateVector(std::vector<int64_t>{TOP_K, TOP_K, TOP_K, 1}), static_cast<int64_t>(-1), 4, 4);
  auto global_topk_weights = split_results[0];
  auto global_topk_ids_float = split_results[1];
  auto global_row_idx_float = split_results[2];
  auto global_pertoken_scale = split_results[3];
  auto topk_ids_rounded = Round(global_topk_ids_float);
  auto row_idx_rounded = Round(global_row_idx_float);
  auto global_row_idx = Cast(row_idx_rounded, ge::DT_INT64);
  auto global_row_idx_flat = Reshape(global_row_idx, std::vector<int64_t>{-1});
  auto axis_const = graph_builder.CreateConst(std::vector<int32_t>{0}, std::vector<int64_t>{});
  auto dispatched_hidden_states = GatherV2(global_hidden_states, global_row_idx_flat, axis_const);
  auto global_pertoken_scale_squeezed = Reshape(global_pertoken_scale, std::vector<int64_t>{-1});
  auto dispatched_pertoken_scale = GatherV2(global_pertoken_scale_squeezed, global_row_idx_flat, axis_const);
  auto global_topk_weights_flat = Reshape(global_topk_weights, std::vector<int64_t>{-1});
  auto group_list_int64 = Cast(group_list, ge::DT_INT64);
  
  std::vector<float> scale_data(NUM_EXPERTS * HIDDEN_SIZE, 1.0f);
  auto weight_scale = graph_builder.CreateConst(scale_data, std::vector<int64_t>{NUM_EXPERTS, HIDDEN_SIZE});

  auto moe_output = GroupedMatmulFinalizeRouting(
      dispatched_hidden_states,
      expert_weight,
      weight_scale,
      nullptr,
      dispatched_pertoken_scale,
      group_list_int64,
      nullptr,
      global_topk_weights_flat,
      global_row_idx_flat,
      nullptr,
      ge::DT_FLOAT,
      0.0f,
      0,
      false,
      false,
      8
  );

  auto final_hidden_states = HcomReduceScatter(moe_output, "sum", group, RANK_SIZE, 0, -1);
  return Add(final_hidden_states, shared_output);
}
}  // namespace

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

void MakeEPGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeEPGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_ep_graph"));
}

std::unique_ptr<ge::Graph> MakeEPGraphByEs() {
  // 1. 创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeEPGraph");

  // 2. 创建输入节点
  auto hidden_states = graph_builder->CreateInput(0, "hidden_states", ge::DT_FLOAT, ge::FORMAT_ND, {LOCAL_BATCH, HIDDEN_SIZE});
  auto gate_weight = graph_builder->CreateInput(1, "gate_weight", ge::DT_FLOAT, ge::FORMAT_ND, {HIDDEN_SIZE, NUM_EXPERTS});
  auto finished = graph_builder->CreateInput(2, "finished", ge::DT_BOOL, ge::FORMAT_ND, {LOCAL_BATCH});
  auto shared_output = graph_builder->CreateInput(3, "shared_output", ge::DT_FLOAT, ge::FORMAT_ND, {LOCAL_BATCH, HIDDEN_SIZE});
  auto expert_weight = graph_builder->CreateInput(4, "expert_weight", ge::DT_INT8, ge::FORMAT_ND, {NUM_EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE}); 
  auto group_list = graph_builder->CreateInput(5, "group_list", ge::DT_INT32, ge::FORMAT_ND, {NUM_EXPERTS});

  // 3. 构建图
  auto output = MakeEPGraph(hidden_states, gate_weight, finished, shared_output,
                            expert_weight, group_list, *graph_builder);

  // 4. 设置输出
  (void) graph_builder->SetOutput(output, 0);

  // 5. 构建图
  return graph_builder->BuildAndReset();
}

int MakeEPGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeEPGraphByEs();
  std::vector<ge::Tensor> inputs;

  // 准备输入数据
  std::vector<float> hidden_states_data(LOCAL_BATCH * HIDDEN_SIZE, 1.0f);
  std::vector<float> gate_weight_data(HIDDEN_SIZE * NUM_EXPERTS, 0.1f);
  std::vector<float> shared_output_data(LOCAL_BATCH * HIDDEN_SIZE, 0.5f);
  std::vector<int8_t> expert_weight_data(NUM_EXPERTS * HIDDEN_SIZE * HIDDEN_SIZE, 1);
  std::vector<int32_t> group_list_data(NUM_EXPERTS, 0);
  for (int32_t i = 0; i < NUM_EXPERTS; ++i) {
    group_list_data[i] = 2; 
  }

  // 创建输入tensor
  std::vector<uint8_t> finished_data(LOCAL_BATCH, 0);
  auto hidden_states_tensor = ge::Utils::StubTensor<float>(hidden_states_data, {LOCAL_BATCH, HIDDEN_SIZE});
  auto gate_weight_tensor = ge::Utils::StubTensor<float>(gate_weight_data, {HIDDEN_SIZE, NUM_EXPERTS});
  ge::Tensor finished_tensor;
  ge::TensorDesc finished_desc(ge::Shape({LOCAL_BATCH}), ge::FORMAT_ND, ge::DT_BOOL);
  finished_tensor.SetTensorDesc(finished_desc);
  finished_tensor.SetData(finished_data.data(), finished_data.size() * sizeof(uint8_t));
  auto shared_output_tensor = ge::Utils::StubTensor<float>(shared_output_data, {LOCAL_BATCH, HIDDEN_SIZE});
  auto expert_weight_tensor = ge::Utils::StubTensor<int8_t>(expert_weight_data, {NUM_EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE});
  auto group_list_tensor = ge::Utils::StubTensor<int32_t>(group_list_data, {NUM_EXPERTS});

  inputs.push_back(*hidden_states_tensor);
  inputs.push_back(*gate_weight_tensor);
  inputs.push_back(finished_tensor);
  inputs.push_back(*shared_output_tensor);
  inputs.push_back(*expert_weight_tensor);
  inputs.push_back(*group_list_tensor);

  return RunGraph(*graph, inputs, "Ep");
}
}  // namespace es_showcase