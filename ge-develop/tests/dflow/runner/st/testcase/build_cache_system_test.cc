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
#include <gmock/gmock.h>
#include "nlohmann/json.hpp"
#include "mmpa/mmpa_api.h"
#include "ge/ge_api.h"
#include "flow_graph/data_flow.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "compiler/graph/build/model_cache.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "common/util/mem_utils.h"
#include "graph/debug/ge_op_types.h"
#include "common/model/ge_root_model.h"
#include "common/context/local_context.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace ge;
namespace ge {
namespace {
Graph BuildAddGraph(const std::string &name) {
  DEF_GRAPH(tmp_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto compute_graph = ToComputeGraph(tmp_graph_def);
  compute_graph->SetName(name);
  return GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}

Graph BuildConstantGraph(const std::string &name) {
  std::vector<int64_t> shape = {1, 2, 3};
  vector<int32_t> data_value(1 * 2 * 3, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));

  DEF_GRAPH(tmp_graph_def) {
    auto variable0 = OP_CFG("Variable").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape);
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape);
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("variable0", variable0)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto compute_graph = ToComputeGraph(tmp_graph_def);
  compute_graph->SetName(name);
  return GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}
}  // namespace

class BuildCacheTest : public testing::Test {
  void SetUp() {
    system("mkdir ./build_cache_dir");
  }
  void TearDown() {
    system("rm -fr ./build_cache_dir");
  }
};

TEST_F(BuildCacheTest, ge_root_model_cache_test) {
  map<AscendString, AscendString> options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}, {"ge.runFlag", "0"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "graph_key1"}};
  std::vector<InputTensorInfo> inputs;
  {
    Graph g1 = BuildAddGraph("ge_root_graph");
    Session session(options);
    session.AddGraph(1, g1, graph_options);
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = mmAccess("./build_cache_dir/graph_key1.idx");
    EXPECT_EQ(check_ret, 0);
  }
  {
    Graph g2 = BuildAddGraph("ge_root_graph");
    Session session2(options);
    session2.AddGraph(2, g2, graph_options);
    auto ret = session2.BuildGraph(2, inputs);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(BuildCacheTest, ge_root_model_suspend_cache_test) {
  map<AscendString, AscendString> options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}, {"ge.runFlag", "0"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "graph_key1"}};
  std::vector<InputTensorInfo> inputs;

  {
    DEF_GRAPH(tmp_graph_def) {
      auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
      auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
      auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
      CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
      CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
      ADD_OUTPUT(add, 0);
    };
    auto compute_graph = ToComputeGraph(tmp_graph_def);
    compute_graph->SetName("name_init_by_20000");
    (void)AttrUtils::SetStr(compute_graph, "_suspend_graph_original_name", "name");
    auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
    Session session(options);
    session.AddGraph(1, graph, graph_options);
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = mmAccess("./build_cache_dir/graph_key1.idx");
    EXPECT_EQ(check_ret, 0);
  }
  {
    Graph g2 = BuildAddGraph("ge_root_graph");
    Session session2(options);
    session2.AddGraph(2, g2, graph_options);
    auto ret = session2.BuildGraph(2, inputs);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(BuildCacheTest, ge_root_model_cache_test_with_constant) {
  map<AscendString, AscendString> options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}, {"ge.runFlag", "0"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "graph_key1"}};
  std::vector<InputTensorInfo> inputs;
  {
    Graph g1 = BuildConstantGraph("ge_root_graph");
    Session session(options);
    session.AddGraph(1, g1, graph_options);
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = mmAccess("./build_cache_dir/graph_key1.idx");
    EXPECT_EQ(check_ret, 0);
  }
  {
    Graph g2 = BuildConstantGraph("ge_root_graph");
    Session session2(options);
    session2.AddGraph(2, g2, graph_options);
    auto ret = session2.BuildGraph(2, inputs);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(BuildCacheTest, ge_root_model_cache_use_hash_index) {
  // cannot as file name.
  std::string graph_key = "abc/def/gh";
  std::string expect_hash_idx_name("hash_");
  auto hash_code = std::hash<std::string>{}(graph_key);
  expect_hash_idx_name.append(std::to_string(hash_code)).append(".idx");

  map<AscendString, AscendString> options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}, {"ge.runFlag", "0"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", graph_key.c_str()}};
  std::vector<InputTensorInfo> inputs;
  {
    Graph g1 = BuildAddGraph("ge_root_graph");
    Session session(options);
    session.AddGraph(1, g1, graph_options);
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_NE(ret, SUCCESS);
  }
}

TEST_F(BuildCacheTest, ge_root_model_cache_test_dir_not_exist) {
  map<AscendString, AscendString> options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir_not_exist"},
                                             {"ge.runFlag", "0"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "graph_key1"}};
  std::vector<InputTensorInfo> inputs;
  {
    Graph g1 = BuildAddGraph("ge_root_graph");
    Session session(options);
    session.AddGraph(1, g1, graph_options);
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_NE(ret, SUCCESS);
  }
}

TEST_F(BuildCacheTest, ge_root_model_cache_test_with_dynamic_batch) {
  map<AscendString, AscendString> options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}, {"ge.runFlag", "0"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "graph_key1"}};
  std::vector<InputTensorInfo> inputs;
  {
    Graph g1 = BuildConstantGraph("ge_root_graph");
    Session session(options);
    session.AddGraph(1, g1, graph_options);
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = mmAccess("./build_cache_dir/graph_key1.idx");
    EXPECT_EQ(check_ret, 0);
  }
  {
    Graph g2 = BuildConstantGraph("ge_root_graph");
    Session session2(options);
    session2.AddGraph(2, g2, graph_options);
    auto old_user_input_dims = GetLocalOmgContext().user_input_dims;
    auto old_dynamic_dims = GetLocalOmgContext().dynamic_dims;
    auto type = GetLocalOmgContext().dynamic_node_type;
    GetLocalOmgContext().user_input_dims = {{"data0", {-1, -1, -1, 1}}};
    GetLocalOmgContext().dynamic_dims = "2,2,2;10,10,10";
    GetLocalOmgContext().dynamic_node_type = "DATA";
    auto ret = session2.BuildGraph(2, inputs);
    EXPECT_EQ(ret, SUCCESS);
    GetLocalOmgContext().user_input_dims = old_user_input_dims;
    GetLocalOmgContext().dynamic_dims = old_dynamic_dims;
    GetLocalOmgContext().dynamic_node_type = type;
  }
}

TEST_F(BuildCacheTest, save_flow_model_to_om_fail) {
  ComputeGraphPtr graph = nullptr;
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  auto ret = FlowModelHelper::SaveToOmModel(flow_model, "./ut_cache_dir/flow.om");
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(BuildCacheTest, update_model_task_addr) {
  DEF_GRAPH(test_graph) {
    auto file_constant =
        OP_CFG(FILECONSTANT).InCnt(0).OutCnt(1)
            .Attr("shape", GeShape{})
            .Attr("dtype", DT_FLOAT)
            .Attr("file_id", "fake_id");
    auto ffts_plus_neg = OP_CFG("Neg").InCnt(1).OutCnt(1);
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});
    CHAIN(NODE("file_constant", file_constant)->NODE("ffts_plus_neg", ffts_plus_neg)->NODE("Node_Output", net_output));
  };

  auto compute_graph = ToComputeGraph(test_graph);
  compute_graph->SetName(test_graph.GetName());

  auto op_desc = compute_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);

  auto fc_node = compute_graph->FindFirstNodeMatchType(FILECONSTANT);
  fc_node->GetOpDesc()->SetOutputOffset({0x10000000});
  auto neg_node = compute_graph->FindFirstNodeMatchType("Neg");
  neg_node->GetOpDesc()->SetInputOffset({0x10000000});
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  auto ge_model = MakeShared<ge::GeModel>();
  ge_model->SetName("test");
  ge_model->SetGraph(compute_graph);
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  auto task_def = model_task_def->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto ffts_plus_task = task_def->mutable_ffts_plus_task();
  auto ctx_def = ffts_plus_task->add_ffts_plus_ctx();
  auto mutable_mix_aic_aiv_ctx = ctx_def->mutable_mix_aic_aiv_ctx();
  mutable_mix_aic_aiv_ctx->add_task_addr(0x10000000);
  mutable_mix_aic_aiv_ctx->add_task_addr(0x20000000);
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_root_model->SetModelName("test_model");
  ge_root_model->SetSubgraphInstanceNameToModel("model_name", ge_model);

  map<AscendString, AscendString> options =
      {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}, {"ge.runFlag", "0"}, {"ge.graph_key", "graph_key1"}};
  Session session(options);
  const auto session_id = session.GetSessionId();
  compute_graph->SetSessionID(session_id);
  AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, std::to_string(session_id) + "_1");

  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(compute_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
  }

  {
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(compute_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(compute_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
  }
}
}  // namespace ge
