/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ge_graph_dsl/graph_dsl.h"

#include "ge/ut/ge/test_tools_task_info.h"
#include "framework/executor/ge_executor.h"
#include "graph/execute/model_executor.h"
#include "ge/ge_ir_build.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "init_ge.h"

using namespace std;
using namespace testing;

namespace ge {
class CtrlFlowCompileTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
  }
};

/**
 *     net_output
 *          |
 *        merge
 *       /   \
 *      /    NEG
 *     /      |
 *  square   shape(will be fold)
 *    F|     T/
 *    switch1
 *   /       \
 *  data     data
 */
TEST_F(CtrlFlowCompileTest,  TestSwitchAndMerge) {
  auto pred_node = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_BOOL, {}).InCnt(1).OutCnt(1);
  auto shape_node = OP_CFG(SHAPE).TensorDesc(FORMAT_ND, DT_INT32, {}).InCnt(1).OutCnt(1);
  DEF_GRAPH(g0) {
    CHAIN(NODE("_arg_0", DATA)->NODE("switch", SWITCH)->EDGE(0, 0)->NODE("shape", shape_node)->NODE("neg0", NEG)->NODE("merge", MERGE)->NODE(
        NODE_NAME_NET_OUTPUT,
        NETOUTPUT));
    CHAIN(NODE("_arg_1", pred_node)->NODE("switch", SWITCH)->EDGE(1, 0)->NODE("neg1", NEG)->NODE("merge", MERGE)->NODE(
        NODE_NAME_NET_OUTPUT,
        NETOUTPUT));
  };

  auto graph = ToGeGraph(g0);
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";

  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);
  ModelBufferData model_buffer_data{};

  std::map<string, string> build_options;
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

/**
 *         neg2
 *        d\ /c
 * neg1 - merge     net_output
 *        T|         |
 *      switch      shape
 *     /       \    /
 *    const     data
 */
TEST_F(CtrlFlowCompileTest, TestSwitchAndMerge_Merge_With_Non_Input) {
  ReInitGe();
  GeRunningEnvFaker::BackupEnv();
  auto pred_node = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_BOOL, {}).InCnt(1).OutCnt(1);
  auto shape_node = OP_CFG(SHAPE).TensorDesc(FORMAT_ND, DT_INT32, {}).InCnt(1).OutCnt(1);
  int64_t max_size = -1;
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_INT64);
  GeTensorPtr const_tensor = MakeShared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&max_size), sizeof(int64_t));
  const auto const_0 = OP_CFG(CONSTANTOP).OutCnt(1).Weight(const_tensor);
  DEF_GRAPH(g0) {
    CHAIN(NODE("_arg_0", DATA)
              ->EDGE(0, 0)
              ->NODE("switch", SWITCH)
              ->EDGE(0, 0)
              ->NODE("merge", MERGE)
              ->NODE("neg1", NEG));
    CHAIN(NODE("merge", MERGE)->NODE("neg2", NEG));
    CHAIN(NODE("merge", MERGE)->CTRL_EDGE()->NODE("neg2", NEG));
    CHAIN(NODE("const_0", const_0)->EDGE(0, 1)->NODE("switch", SWITCH));
    CHAIN(NODE("_arg_0", DATA)->NODE("shape", shape_node)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
  };

  auto graph = ToGeGraph(g0);
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);
  ModelBufferData model_buffer_data{};

  DUMP_GRAPH_WHEN("PrepareAfterInferFormatAndShape");
  std::map<string, string> build_options;
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  CHECK_GRAPH(PrepareAfterInferFormatAndShape) {
    EXPECT_EQ(graph->FindNode("merge"), nullptr);
    EXPECT_EQ(graph->FindNode("neg1"), nullptr);
    EXPECT_EQ(graph->FindNode("neg2"), nullptr);
  };
  aclgrphBuildFinalize();
}
}  // namespace ge
