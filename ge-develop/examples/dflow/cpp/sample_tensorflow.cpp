/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>

#include "all_ops.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "flow_graph/data_flow.h"
#include "node_builder.h"

using namespace ge;
using namespace dflow;
namespace {
constexpr int32_t kFeedTimeout = 3000;
constexpr int32_t kFetchTimeout = 30000;
/**
 * @brief
 * Build a dataflow graph by DataFlow API
 * The dataflow graph contains 3 flow nodes and DAG shows as following:
 * FlowData      FlowData
 *    \            /
 *     \          /
 *      \        /
 *       \      /
 *       FlowNode0
 *           |
 *           |
 *       FlowOutput
 *
 * @return DataFlow graph
 *
 */
FlowGraph BuildDataFlowGraph() {
  // construct graph
  dflow::FlowGraph flow_graph("flow_graph");

  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);

  std::map<ge::AscendString, ge::AscendString> parser_params = {
      {ge::AscendString(ge::ir_option::INPUT_DATA_NAMES), ge::AscendString("Placeholder,Placeholder_1")}};
  BuildBasicConfig build_cfg = {
      .node_name = "node0", .input_num = 2, .output_num = 1, .compile_cfg = "../config/add_graph.json"};
  auto node0 = BuildTfGraphNode(build_cfg, "../config/add.pb", parser_params).SetInput(0, data0).SetInput(1, data1);
  std::vector<FlowOperator> inputs_operator{data0, data1};
  std::vector<FlowOperator> outputs_operator{node0};
  flow_graph.SetInputs(inputs_operator).SetOutputs(outputs_operator);
  return flow_graph;
}

bool CheckResult(std::vector<ge::Tensor> &result, const std::vector<int32_t> &expect_out) {
  if (result.size() != 1) {
    std::cout << "ERROR=======Fetch data size is expected containing 1 element=" << std::endl;
    return false;
  }
  if (result[0].GetSize() != expect_out.size() * sizeof(int32_t)) {
    std::cout << "ERROR=======Verify data size failed===========" << std::endl;
    std::cout << "Tensor size:" << result[0].GetSize() << std::endl;
    std::cout << "Expect size:" << expect_out.size() * sizeof(int32_t) << std::endl;
    return false;
  }

  int32_t *output_data = reinterpret_cast<int32_t *>(result[0].GetData());
  if (output_data != nullptr) {
    for (size_t k = 0; k < expect_out.size(); ++k) {
      if (expect_out[k] != output_data[k]) {
        std::cout << "ERROR=======Verify data failed===========" << std::endl;
        std::cout << "ERROR======expect:" << expect_out[k] << "  real:" << output_data[k] << std::endl;
        return false;
      }
    }
  }
  return true;
}
}  // namespace

int32_t main() {
  // Build dataflow graph
  auto flow_graph = BuildDataFlowGraph();

  // Initialize
  std::map<ge::AscendString, AscendString> config = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "0"}};
  auto ge_ret = ge::GEInitialize(config);
  if (ge_ret != ge::SUCCESS) {
    std::cout << "ERROR=====GeInitialize failed.=======" << std::endl;
    return ge_ret;
  }

  // Create Session
  std::map<ge::AscendString, ge::AscendString> options;
  std::shared_ptr<ge::Session> session = std::make_shared<ge::Session>(options);
  if (session == nullptr) {
    std::cout << "ERROR=======Create session failed===========" << std::endl;
    ge::GEFinalize();
    return ge_ret;
  }

  // Add graph
  ge_ret = session->AddGraph(0, flow_graph.ToGeGraph());
  if (ge_ret != ge::SUCCESS) {
    std::cout << "ERROR=======Add graph failed===========" << std::endl;
    ge::GEFinalize();
    return ge_ret;
  }

  // Prepare Inputs
  const int64_t element_num = 3;
  std::vector<int64_t> shape = {element_num};
  int32_t input_data[element_num] = {4, 7, 5};
  ge::Tensor input_tensor;
  ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_INT32);
  input_tensor.SetTensorDesc(desc);
  input_tensor.SetData((uint8_t *)input_data, sizeof(int32_t) * element_num);

  ge::DataFlowInfo data_flow_info;
  std::vector<ge::Tensor> inputs_data = {input_tensor, input_tensor};

  // FeedInput
  const size_t loop_num = 4;
  for (size_t i = 0; i < loop_num; ++i) {
    ge_ret = session->FeedDataFlowGraph(0, inputs_data, data_flow_info, kFeedTimeout);
    if (ge_ret != ge::SUCCESS) {
      std::cout << "ERROR=======Feed data failed===========" << std::endl;
      ge::GEFinalize();
      return ge_ret;
    }
  }

  // Verify outputs
  std::vector<int32_t> expect_output = {8, 14, 10};
  for (size_t i = 0; i < loop_num; ++i) {
    std::vector<ge::Tensor> outputs_data;
    ge_ret = session->FetchDataFlowGraph(0, outputs_data, data_flow_info, kFetchTimeout);
    if (ge_ret != ge::SUCCESS) {
      std::cout << "ERROR=======Fetch data failed===========" << std::endl;
      ge::GEFinalize();
      return ge_ret;
    }

    if (!CheckResult(outputs_data, expect_output)) {
      std::cout << "ERROR=======Check result data failed===========" << std::endl;
      ge::GEFinalize();
      return -1;
    }
  }
  std::cout << "TEST=======run case success===========" << std::endl;
  ge::GEFinalize();
  return 0;
}