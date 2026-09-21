/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"

using namespace std;
using namespace ge;
namespace {
GeTensorPtr GenerateLoopVarTensor(int64_t data_value) {
  GeTensorDesc data_tensor_desc(GeShape(std::vector<int64_t>({})), FORMAT_ND, DT_INT64);
  return std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)(&data_value), sizeof(int64_t));
}

Graph BuildIteratorFlowCtrlVarGraph(bool with_global_step_var = true, bool with_loop_per_iter_var = true,
                                    bool with_loop_cond_var = true, bool with_loop_inc_var = true,
                                    bool with_loop_reset_value_var = true, int64_t loop_num_per_iter = 10) {
  std::vector<int64_t> shape = {};
  auto data_tensor = GenerateLoopVarTensor(1);
  auto loop_num_data_tensor = GenerateLoopVarTensor(loop_num_per_iter);
  DEF_GRAPH(flow_ctrl_var_init) {
    uint32_t output_port_id_idx = 0;
    if (with_global_step_var) {
      // NODE_NAME_GLOBAL_STEP
      auto global_step_var =
          OP_CFG(VARIABLE).TensorDesc(FORMAT_ND, DT_INT64, shape).InCnt(1).OutCnt(1).Build(NODE_NAME_GLOBAL_STEP);
      auto global_step_const = OP_CFG(CONSTANT)
                                   .TensorDesc(FORMAT_ND, DT_INT64, shape)
                                   .Weight(data_tensor)
                                   .InCnt(1)
                                   .OutCnt(1)
                                   .Build("global_step_const");
      auto global_step_assign =
          OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_INT64, shape).InCnt(2).OutCnt(1).Build("global_step_assign");

      CHAIN(NODE(global_step_var)->NODE(global_step_assign));
      CHAIN(NODE(global_step_const)->EDGE(0, 1)->NODE(global_step_assign));

      CHAIN(NODE(global_step_assign)->EDGE(0, output_port_id_idx)->NODE("NetOutput1", NETOUTPUT));
      output_port_id_idx++;
    }

    if (with_loop_per_iter_var) {
      // NODE_NAME_FLOWCTRL_LOOP_PER_ITER
      auto loop_per_iter_var = OP_CFG(VARIABLE)
                                   .TensorDesc(FORMAT_ND, DT_INT64, shape)
                                   .InCnt(1)
                                   .OutCnt(1)
                                   .Build(NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
      auto loop_per_iter_const = OP_CFG(CONSTANT)
                                     .TensorDesc(FORMAT_ND, DT_INT64, shape)
                                     .Weight(loop_num_data_tensor)
                                     .InCnt(1)
                                     .OutCnt(1)
                                     .Build("loop_per_iter_const");
      auto loop_per_iter_assign =
          OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_INT64, shape).InCnt(2).OutCnt(1).Build("loop_per_iter_assign");

      CHAIN(NODE(loop_per_iter_var)->NODE(loop_per_iter_assign));
      CHAIN(NODE(loop_per_iter_const)->EDGE(0, 1)->NODE(loop_per_iter_assign));
      CHAIN(NODE(loop_per_iter_assign)->EDGE(0, output_port_id_idx)->NODE("NetOutput1", NETOUTPUT));
      output_port_id_idx++;
    }

    if (with_loop_cond_var) {
      // NODE_NAME_FLOWCTRL_LOOP_COND
      auto loop_cond_var = OP_CFG(VARIABLE)
                               .TensorDesc(FORMAT_ND, DT_INT64, shape)
                               .InCnt(1)
                               .OutCnt(1)
                               .Build(NODE_NAME_FLOWCTRL_LOOP_COND);
      auto loop_cond_const = OP_CFG(CONSTANT)
                                 .TensorDesc(FORMAT_ND, DT_INT64, shape)
                                 .Weight(data_tensor)
                                 .InCnt(1)
                                 .OutCnt(1)
                                 .Build("loop_cond_const");
      auto loop_cond_assign =
          OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_INT64, shape).InCnt(2).OutCnt(1).Build("loop_cond_assign");

      CHAIN(NODE(loop_cond_var)->NODE(loop_cond_assign));
      CHAIN(NODE(loop_cond_const)->EDGE(0, 1)->NODE(loop_cond_assign));
      CHAIN(NODE(loop_cond_assign)->EDGE(0, output_port_id_idx)->NODE("NetOutput1", NETOUTPUT));
      output_port_id_idx++;
    }

    if (with_loop_inc_var) {
      // NODE_NAME_FLOWCTRL_LOOP_INCREMENT
      auto loop_inc_var = OP_CFG(VARIABLE)
                              .TensorDesc(FORMAT_ND, DT_INT64, shape)
                              .InCnt(1)
                              .OutCnt(1)
                              .Build(NODE_NAME_FLOWCTRL_LOOP_INCREMENT);
      auto loop_inc_const = OP_CFG(CONSTANT)
                                .TensorDesc(FORMAT_ND, DT_INT64, shape)
                                .Weight(data_tensor)
                                .InCnt(1)
                                .OutCnt(1)
                                .Build("loop_inc_const");
      auto loop_inc_assign =
          OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_INT64, shape).InCnt(2).OutCnt(1).Build("loop_inc_assign");

      CHAIN(NODE(loop_inc_var)->NODE(loop_inc_assign));
      CHAIN(NODE(loop_inc_const)->EDGE(0, 1)->NODE(loop_inc_assign));
      CHAIN(NODE(loop_inc_assign)->EDGE(0, output_port_id_idx)->NODE("NetOutput1", NETOUTPUT));
      output_port_id_idx++;
    }

    if (with_loop_reset_value_var) {
      // NODE_NAME_FLOWCTRL_LOOP_RESETVALUE
      auto loop_reset_value_var = OP_CFG(VARIABLE)
                                      .TensorDesc(FORMAT_ND, DT_INT64, shape)
                                      .InCnt(1)
                                      .OutCnt(1)
                                      .Build(NODE_NAME_FLOWCTRL_LOOP_RESETVALUE);
      auto loop_reset_value_const = OP_CFG(CONSTANT)
                                        .TensorDesc(FORMAT_ND, DT_INT64, shape)
                                        .Weight(data_tensor)
                                        .InCnt(1)
                                        .OutCnt(1)
                                        .Build("loop_reset_value_onst");
      auto loop_reset_value_assign =
          OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_INT64, shape).InCnt(2).OutCnt(1).Build("loop_reset_value_assign");

      CHAIN(NODE(loop_reset_value_var)->NODE(loop_reset_value_assign));
      CHAIN(NODE(loop_reset_value_const)->EDGE(0, 1)->NODE(loop_reset_value_assign));
      CHAIN(NODE(loop_reset_value_assign)->EDGE(0, output_port_id_idx)->NODE("NetOutput1", NETOUTPUT));
      output_port_id_idx++;
    }
  };

  return ToGeGraph(flow_ctrl_var_init);
}

/**
 * Set up a graph with the following network structure
 *        IteratorGetNext
 *              |
 *          MemcpyAsync
 *              |
 *              A
 *              |
 *          NetOutput
 */
Graph BuildIteratorFlowCtrlGraph(uint32_t iterator_num, bool with_global_step = true, bool with_loop_per_iter = true,
                                 bool with_loop_cond = true, bool with_loop_inc = true,
                                 bool with_loop_reset_value = true) {
  DEF_GRAPH(g1) {
    uint32_t output_port_id_idx = 0;
    auto out_desc = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
    AttrUtils::SetInt(out_desc, ATTR_NAME_TRUE_BRANCH_STREAM, TRUE_STREAM_ID);

    auto memcpy_op_desc = std::make_shared<OpDesc>("MemcpyAsync", MEMCPYASYNC);
    AttrUtils::SetBool(memcpy_op_desc, ATTR_NAME_STREAM_CYCLE_EVENT_FLAG, true);

    auto iterator_get_next = std::make_shared<OpDesc>("IteratorGetNext", FRAMEWORKOP);
    AttrUtils::SetStr(iterator_get_next, "original_type", "IteratorGetNext");
    CHAIN(NODE(iterator_get_next)->NODE(memcpy_op_desc)->NODE("A", RESOURCEAPPLYMOMENTUM)->EDGE(0, 0)->NODE(out_desc));

    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_UINT64);
    if (with_global_step) {
      auto loop_global_step = std::make_shared<OpDesc>(NODE_NAME_GLOBAL_STEP, VARIABLE);
      loop_global_step->AddOutputDesc(tensor_desc);
      CHAIN(NODE(loop_global_step)->EDGE(0, output_port_id_idx)->NODE(out_desc));
      output_port_id_idx++;
    }
    if (with_loop_per_iter) {
      auto loop_per_iter = std::make_shared<OpDesc>(NODE_NAME_FLOWCTRL_LOOP_PER_ITER, VARIABLE);
      loop_per_iter->AddOutputDesc(tensor_desc);
      CHAIN(NODE(loop_per_iter)->EDGE(0, output_port_id_idx)->NODE(out_desc));
      output_port_id_idx++;
    }
    if (with_loop_cond) {
      auto loop_cond = std::make_shared<OpDesc>(NODE_NAME_FLOWCTRL_LOOP_COND, VARIABLE);
      loop_cond->AddOutputDesc(tensor_desc);
      CHAIN(NODE(loop_cond)->EDGE(0, output_port_id_idx)->NODE(out_desc));
      output_port_id_idx++;
    }
    if (with_loop_inc) {
      auto loop_increment = std::make_shared<OpDesc>(NODE_NAME_FLOWCTRL_LOOP_INCREMENT, VARIABLE);
      loop_increment->AddOutputDesc(tensor_desc);
      CHAIN(NODE(loop_increment)->EDGE(0, output_port_id_idx)->NODE(out_desc));
      output_port_id_idx++;
    }
    if (with_loop_reset_value) {
      auto loop_reset_value = std::make_shared<OpDesc>(NODE_NAME_FLOWCTRL_LOOP_RESETVALUE, VARIABLE);
      loop_reset_value->AddOutputDesc(tensor_desc);
      CHAIN(NODE(loop_reset_value)->EDGE(0, output_port_id_idx)->NODE(out_desc));
      output_port_id_idx++;
    }

    for (uint32_t i = 0; i < iterator_num; ++i) {
      std::string iterator_name("iterator_");
      iterator_name.append(std::to_string(i));
      auto iterator_desc = std::make_shared<OpDesc>(iterator_name, FRAMEWORKOP);
      AttrUtils::SetStr(iterator_desc, "original_type", "IteratorV2");
      CHAIN(NODE(iterator_desc)->EDGE(0, output_port_id_idx)->NODE(out_desc));
      output_port_id_idx++;
    }
  };
  return ToGeGraph(g1);
}
}  // namespace
class IteratorFlowCtrlTest : public testing::Test {
 protected:
  void SetUp() {
    map<AscendString, AscendString> options;
    EXPECT_EQ(GEInitialize(options), SUCCESS);
  }
  void TearDown() {}
};

TEST_F(IteratorFlowCtrlTest, iterator_loop_normal) {
  bool with_global_step_var = true;
  bool with_global_step_node = true;
  uint32_t iterator_v2_num = 1;
  auto var_graph = BuildIteratorFlowCtrlVarGraph(with_global_step_var);
  auto original_loop_graph = BuildIteratorFlowCtrlGraph(iterator_v2_num, with_global_step_node);
  original_loop_graph.SetNeedIteration(true);
  size_t original_node_size = original_loop_graph.GetAllNodes().size();

  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, std::to_string(ge::TRAIN).c_str());
  Session session(options);
  session.AddGraph(8, var_graph, options);
  session.AddGraph(9, original_loop_graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(8, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(9, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_GT(graph->GetDirectNodesSize(), original_node_size);

    auto get_next_loop_switch_node = graph->FindNode("IteratorGetNext_IteratorCtrl_StreamSwitch");
    EXPECT_NE(get_next_loop_switch_node, nullptr);
  };
}

// multi iterator v2 will insert model exit active node
TEST_F(IteratorFlowCtrlTest, iterator_loop_normal_multi_iterator) {
  bool with_global_step_var = true;
  bool with_global_step_node = true;
  uint32_t iterator_v2_num = 2;
  auto var_graph = BuildIteratorFlowCtrlVarGraph(with_global_step_var);
  auto original_loop_graph = BuildIteratorFlowCtrlGraph(iterator_v2_num, with_global_step_node);
  original_loop_graph.SetNeedIteration(true);
  size_t original_node_size = original_loop_graph.GetAllNodes().size();

  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, std::to_string(ge::TRAIN).c_str());
  options.emplace(ge::VARIABLE_MEMORY_MAX_SIZE, "12800");
  Session session(options);
  session.AddGraph(8, var_graph, options);
  session.AddGraph(9, original_loop_graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(8, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(9, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_GT(graph->GetDirectNodesSize(), original_node_size);
    std::string model_exit_node_name("NetOutput_");
    model_exit_node_name.append(NODE_NAME_STREAM_SWITCH).append("_StreamExitActive");
    auto model_exit_node = graph->FindNode(model_exit_node_name);
    EXPECT_NE(model_exit_node, nullptr);

    auto get_next_loop_switch_node = graph->FindNode("IteratorGetNext_IteratorCtrl_StreamSwitch");
    EXPECT_NE(get_next_loop_switch_node, nullptr);
  };
}

// auto insert var node
TEST_F(IteratorFlowCtrlTest, iterator_loop_normal_no_var_node_in_loop_graph) {
  bool with_global_step_var = true;
  bool with_loop_per_iter_var = true;
  bool with_loop_cond_var = true;
  bool with_loop_inc_var = true;
  bool with_loop_reset_value_var = true;
  auto var_graph = BuildIteratorFlowCtrlVarGraph(with_global_step_var, with_loop_per_iter_var, with_loop_cond_var,
                                                 with_loop_inc_var, with_loop_reset_value_var);

  uint32_t iterator_v2_num = 1;
  bool with_global_step_node = false;
  bool with_loop_per_iter_node = false;
  bool with_loop_cond_node = false;
  bool with_loop_inc_node = false;
  bool with_loop_reset_value_node = false;
  auto original_loop_graph =
      BuildIteratorFlowCtrlGraph(iterator_v2_num, with_global_step_node, with_loop_per_iter_node, with_loop_cond_node,
                                 with_loop_inc_node, with_loop_reset_value_node);
  original_loop_graph.SetNeedIteration(true);
  size_t original_node_size = original_loop_graph.GetAllNodes().size();

  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, std::to_string(ge::TRAIN).c_str());
  options.emplace(ge::VARIABLE_MEMORY_MAX_SIZE, "12800");
  Session session(options);
  session.AddGraph(8, var_graph, options);
  session.AddGraph(9, original_loop_graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(8, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(9, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_GT(graph->GetDirectNodesSize(), original_node_size);
    auto global_step_node = graph->FindNode(NODE_NAME_GLOBAL_STEP);
    EXPECT_NE(global_step_node, nullptr);

    auto get_next_loop_switch_node = graph->FindNode("IteratorGetNext_IteratorCtrl_StreamSwitch");
    EXPECT_NE(get_next_loop_switch_node, nullptr);
  };
}

// no_global_step_var will not use pass
TEST_F(IteratorFlowCtrlTest, iterator_loop_normal_no_loop_per_iter_var) {
  bool with_global_step_var = true;
  bool with_loop_per_iter_var = false;
  bool with_global_step_node = true;
  uint32_t iterator_v2_num = 1;
  auto var_graph = BuildIteratorFlowCtrlVarGraph(with_global_step_var, with_loop_per_iter_var);
  auto original_loop_graph = BuildIteratorFlowCtrlGraph(iterator_v2_num, with_global_step_node);
  original_loop_graph.SetNeedIteration(true);

  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, std::to_string(ge::TRAIN).c_str());
  options.emplace(ge::VARIABLE_MEMORY_MAX_SIZE, "12800");
  Session session(options);
  session.AddGraph(8, var_graph, options);
  session.AddGraph(9, original_loop_graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(8, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(9, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto get_next_loop_switch_node = graph->FindNode("IteratorGetNext_IteratorCtrl_StreamSwitch");
    EXPECT_EQ(get_next_loop_switch_node, nullptr);
  };
}

// no loop_cond_var and no loop_cond_node will pass failed
TEST_F(IteratorFlowCtrlTest, iterator_loop_normal_no_loop_cond) {
  bool with_global_step_var = true;
  bool with_loop_per_iter_var = true;
  bool with_loop_cond_var = false;
  bool with_global_step_node = true;
  bool with_loop_per_iter_node = true;
  bool with_loop_cond_node = false;
  uint32_t iterator_v2_num = 1;
  auto var_graph = BuildIteratorFlowCtrlVarGraph(with_global_step_var, with_loop_per_iter_var, with_loop_cond_var);
  auto original_loop_graph =
      BuildIteratorFlowCtrlGraph(iterator_v2_num, with_global_step_node, with_loop_per_iter_node, with_loop_cond_node);
  original_loop_graph.SetNeedIteration(true);

  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, std::to_string(ge::TRAIN).c_str());
  options.emplace(ge::VARIABLE_MEMORY_MAX_SIZE, "12800");
  Session session(options);
  session.AddGraph(8, var_graph, options);
  session.AddGraph(9, original_loop_graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(8, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(9, inputs);
  EXPECT_NE(ret, SUCCESS);
}

// no loop_inc var and node flow ctrl pass will add node failed pass failed
TEST_F(IteratorFlowCtrlTest, iterator_loop_normal_no_loop_inc) {
  bool with_global_step_var = true;
  bool with_loop_per_iter_var = true;
  bool with_loop_cond_var = true;
  bool with_loop_inc_var = false;
  bool with_global_step_node = true;
  bool with_loop_per_iter_node = true;
  bool with_loop_cond_node = true;
  bool with_loop_inc_node = false;
  uint32_t iterator_v2_num = 1;
  auto var_graph = BuildIteratorFlowCtrlVarGraph(with_global_step_var, with_loop_per_iter_var, with_loop_cond_var,
                                                 with_loop_inc_var);
  auto original_loop_graph = BuildIteratorFlowCtrlGraph(iterator_v2_num, with_global_step_node, with_loop_per_iter_node,
                                                        with_loop_cond_node, with_loop_inc_node);
  original_loop_graph.SetNeedIteration(true);

  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, std::to_string(ge::TRAIN).c_str());
  options.emplace(ge::VARIABLE_MEMORY_MAX_SIZE, "12800");
  Session session(options);
  session.AddGraph(8, var_graph, options);
  session.AddGraph(9, original_loop_graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(8, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(9, inputs);
  EXPECT_NE(ret, SUCCESS);
}

// no loop_reset_value_var var and node flow ctrl pass will add node failed pass failed
TEST_F(IteratorFlowCtrlTest, iterator_loop_normal_no_reset_value) {
  bool with_global_step_var = true;
  bool with_loop_per_iter_var = true;
  bool with_loop_cond_var = true;
  bool with_loop_inc_var = true;
  bool with_loop_reset_value_var = false;
  auto var_graph = BuildIteratorFlowCtrlVarGraph(with_global_step_var, with_loop_per_iter_var, with_loop_cond_var,
                                                 with_loop_inc_var, with_loop_reset_value_var);

  uint32_t iterator_v2_num = 1;
  bool with_global_step_node = true;
  bool with_loop_per_iter_node = true;
  bool with_loop_cond_node = true;
  bool with_loop_inc_node = true;
  bool with_loop_reset_value_node = false;
  auto original_loop_graph =
      BuildIteratorFlowCtrlGraph(iterator_v2_num, with_global_step_node, with_loop_per_iter_node, with_loop_cond_node,
                                 with_loop_inc_node, with_loop_reset_value_node);
  original_loop_graph.SetNeedIteration(true);

  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, std::to_string(ge::TRAIN).c_str());
  options.emplace(ge::VARIABLE_MEMORY_MAX_SIZE, "12800");
  Session session(options);
  session.AddGraph(8, var_graph, options);
  session.AddGraph(9, original_loop_graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(8, inputs);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(9, inputs);
  EXPECT_NE(ret, SUCCESS);
}
