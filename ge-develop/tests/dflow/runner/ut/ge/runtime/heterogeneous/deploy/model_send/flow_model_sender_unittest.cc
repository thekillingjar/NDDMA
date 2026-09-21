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
#include "graph/debug/ge_attr_define.h"
#include "macro_utils/dt_public_scope.h"
#include "deploy/model_send/flow_model_sender.h"
#include "deploy/resource/resource_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/build/graph_builder.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "generator/ge_generator.h"
#include "graph/ge_local_context.h"

namespace ge {
class FlowModelSenderTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(FlowModelSenderTest, AddDynamicSchedInfo) {
  FlowModelSender flow_model_sender;
  DeployState deploy_state;
  DeployPlan::SubmodelInfo model_info;

  model_info.status_input_queue_indices.push_back(0);
  model_info.status_output_queue_indices.push_back(0);
  auto &sub_model = deploy_state.MutableDeployPlan().MutableSubmodels();
  sub_model["test"] = model_info;
  deploy_state.MutableDeployPlan().dynamic_sched_plan_.model_instances_num_["test"] = 2;

  deployer::SubmodelDesc submodel_desc;
  EXPECT_NO_THROW(flow_model_sender.AddDynamicSchedInfo(deploy_state, "test", submodel_desc));
}

TEST_F(FlowModelSenderTest, AddErrDynamicSchedInfo) {
  FlowModelSender flow_model_sender;
  DeployState deploy_state;
  DeployPlan::SubmodelInfo model_info;

  model_info.status_input_queue_indices.push_back(0);
  model_info.status_output_queue_indices.push_back(0);
  auto &sub_model = deploy_state.MutableDeployPlan().MutableSubmodels();
  sub_model["test"] = model_info;
  deploy_state.MutableDeployPlan().dynamic_sched_plan_.model_instances_num_["test"] = 2;

  deployer::SubmodelDesc submodel_desc;
  EXPECT_NO_THROW(flow_model_sender.AddDynamicSchedInfo(deploy_state, "err_test", submodel_desc));
}

TEST_F(FlowModelSenderTest, CacheLocalModel) {
  int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId();
  GE_MAKE_GUARD(recover, [local_node_id]() {
    ResourceManager::GetInstance().local_node_id_ = local_node_id;
  });
  
  ResourceManager::GetInstance().local_node_id_ = 0;
  FlowModelSender flow_model_sender;
  DeployPlan::SubmodelInfo model_info;
  auto graph = MakeShared<ComputeGraph>("g1");
  model_info.model = MakeShared<FlowModel>(graph);
  model_info.model->SetModelType(PNE_ID_NPU);
  model_info.model->SetSavedModelPath("/a/b/c");
  auto ret = flow_model_sender.CacheLocalModel(model_info);
  EXPECT_EQ(ret, true);
}

TEST_F(FlowModelSenderTest, TestGetSavedModelPath) {
  DeployPlan::SubmodelInfo model_info;
  auto graph = MakeShared<ComputeGraph>("g1");
  (void)AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false);
  graph->SetGraphUnknownFlag(false);
  model_info.model =  MakeShared<FlowModel>(graph);;
  model_info.model->SetIsBuiltinModel(true);
  model_info.model->SetModelType(PNE_ID_UDF);
  model_info.device_info = DeployPlan::DeviceInfo(0, 0, 0);
  FlowModelSender flow_model_sender;
  std::string model_file = "model_file";
  std::string saved_path;

  model_info.model->SetIsBuiltinModel(false);
  const auto back_up = ResourceManager::GetInstance().local_node_id_;
  ResourceManager::GetInstance().local_node_id_ = 0;
  ASSERT_EQ(flow_model_sender.GetSavedFilePath(model_info, model_file, saved_path), FAILED);
  
  model_info.model->SetSavedModelPath("save_path");
  ASSERT_EQ(flow_model_sender.GetSavedFilePath(model_info, model_file, saved_path), SUCCESS);
  ASSERT_EQ(saved_path, "save_path");
  ResourceManager::GetInstance().local_node_id_ = back_up;
}

TEST_F(FlowModelSenderTest, BuildDeployPlanOptions) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  AttrUtils::SetBool(root_graph, "_dflow_is_data_flow_graph", true);
  auto flow_model = MakeShared<FlowModel>(root_graph);
  DeployState deploy_state;
  deploy_state.SetFlowModel(flow_model);
  FlowModelSender flow_model_sender;
  deployer::UpdateDeployPlanRequest request;

  map<std::string, std::string> global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  GE_MAKE_GUARD(recover_sess_cfg, ([&graph_options, &global_options, &sess_options](){
    GetThreadLocalContext().SetGlobalOption(global_options);
    GetThreadLocalContext().SetSessionOption(sess_options);
    GetThreadLocalContext().SetGraphOption(graph_options);
  }));

  map<std::string, std::string> global_mock_options{{OP_WAIT_TIMEOUT, "0"}};
  GetThreadLocalContext().SetGlobalOption(global_mock_options);
  map<std::string, std::string> sess_mock_options{{OP_WAIT_TIMEOUT, "0"}};
  GetThreadLocalContext().SetSessionOption(sess_mock_options);
  map<std::string, std::string> graph_mock_options{{OP_WAIT_TIMEOUT, "0"}};
  GetThreadLocalContext().SetGraphOption(graph_mock_options);
  flow_model_sender.BuildDeployPlanOptions(deploy_state, request);
  EXPECT_EQ(request.options().global_options_size(), 2);
  EXPECT_EQ(request.options().session_options_size(), 1);
  EXPECT_EQ(request.options().graph_options_size(), 1);
}
}  // namespace ge
