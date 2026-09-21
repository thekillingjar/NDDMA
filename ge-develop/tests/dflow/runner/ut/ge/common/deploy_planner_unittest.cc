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
#include "dflow/base/model/endpoint.h"

#include "macro_utils/dt_public_scope.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/passes/graph_builder_utils.h"
#include "graph/build/graph_builder.h"
#include "stub_models.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"

using namespace std;

namespace ge {
namespace {
void AddQueueDef(ModelRelation &model_relation, const std::string &name) {
  Endpoint queue_def(name, EndpointType::kQueue);
  model_relation.endpoints.emplace_back(queue_def);
}
}  // namespace

class DeployPlannerTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(DeployPlannerTest, TestFailedDueToMismatchOfQueueNames) {
  auto root_model = StubModels::BuildRootModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(root_model != nullptr);
  root_model->model_relation_->submodel_endpoint_infos["subgraph-2"].input_endpoint_names[0] = "oops";
  DeployPlan deploy_plan;
  auto flow_model = std::make_shared<FlowModel>();
  ASSERT_TRUE(flow_model != nullptr);
  flow_model->AddSubModel(root_model);
  auto ret = DeployPlanner(flow_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, PARAM_INVALID);
}

TEST_F(DeployPlannerTest, TestBuildDeployPlan_WithQueueBindings) {
  auto root_model = StubModels::BuildRootModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(root_model != nullptr);
  EXPECT_EQ(root_model->GetSubmodels().size(), 2);
  std::cout << root_model->GetSubmodels().size() << std::endl;
  auto model_relation = root_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  auto ret = DeployPlanner(root_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  // data2 -> PC_1, data2 -> PC_2
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 8);  // 4(output) + 2(input) + 2(output group)
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 2);
  std::map<int, std::set<int>> check;
  for (auto to_bind : deploy_plan.GetQueueBindings()) {
    check[to_bind.first].emplace(to_bind.second);
    std::cout << to_bind.first << " -> " << to_bind.second << std::endl;
  }
  ASSERT_EQ(check.size(), 1);
  ASSERT_EQ(check.begin()->second.size(), 2);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}
/**
 *      NetOutput
 *          |
 *        PC_3
 *       /   \.
 *     PC_1  PC2
 *     |      |
 *   data1  data2
 */
TEST_F(DeployPlannerTest, TestBuildDeployPlan_WithoutQueueBindings) {
  auto root_model = StubModels::BuildRootModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(root_model != nullptr);
  EXPECT_EQ(root_model->GetSubmodels().size(), 3);
  auto model_relation = root_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 3);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-3")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-3")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  auto ret = DeployPlanner(root_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 5);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 0);
}

TEST_F(DeployPlannerTest, TestGetQueueInfo) {
  DeployPlan plan;
  const DeployPlan::QueueInfo *queue_info = nullptr;
  ASSERT_EQ(plan.GetQueueInfo(0, queue_info), PARAM_INVALID);
  plan.queues_.resize(1);
  ASSERT_EQ(plan.GetQueueInfo(0, queue_info), SUCCESS);
}

TEST_F(DeployPlannerTest, TestAddingControlInput) {
  auto flow_model = make_shared<FlowModel>();
  auto graph_1 = ge::MakeShared<ComputeGraph>("subgraph-1");
  auto graph_2 = ge::MakeShared<ComputeGraph>("subgraph-2");
  auto graph_3 = ge::MakeShared<ComputeGraph>("subgraph-3");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  auto submodel_2 = StubModels::BuildRootModel(graph_2, false);
  auto submodel_3 = StubModels::BuildRootModel(graph_3, false);
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  ASSERT_TRUE(submodel_3 != nullptr);
  flow_model->AddSubModel(submodel_1);
  flow_model->AddSubModel(submodel_2);
  flow_model->AddSubModel(submodel_3);

  flow_model->model_relation_ = MakeShared<ModelRelation>();
  Endpoint queue_def("ext_queue", EndpointType::kQueue);
  QueueNodeUtils(queue_def).SetDepth(2L);
  flow_model->model_relation_->endpoints.emplace_back(queue_def);
  queue_def.SetName("out_1_0");
  flow_model->model_relation_->endpoints.emplace_back(queue_def);
  queue_def.SetName("out_2_0");
  flow_model->model_relation_->endpoints.emplace_back(queue_def);
  queue_def.SetName("out_3_0");
  flow_model->model_relation_->endpoints.emplace_back(queue_def);
  flow_model->model_relation_->root_model_endpoint_info.external_input_queue_names = {"ext_queue"};
  flow_model->model_relation_->root_model_endpoint_info.output_endpoint_names = {"out_1_0", "out_2_0", "out_3_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].model_name = "subgraph-1";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].external_input_queue_names = {"ext_queue"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].output_endpoint_names = {"out_1_0"};

  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].model_name = "subgraph-2";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].input_endpoint_names = {};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].output_endpoint_names = {"out_2_0"};

  flow_model->model_relation_->submodel_endpoint_infos["subgraph-3"].model_name = "subgraph-3";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-3"].input_endpoint_names = {};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-3"].output_endpoint_names = {"out_3_0"};

  DeployPlan deploy_plan;
  auto ret = DeployPlanner(flow_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  std::cout << deploy_plan.GetAllInputQueueIndices().size()  << std::endl;
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].input_queue_indices.size(), 0);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].control_input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-3"].input_queue_indices.size(), 0);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-3"].control_input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 0);
  ASSERT_EQ(deploy_plan.GetControlInputQueueIndices().size(), 1);
}

TEST_F(DeployPlannerTest, TestAddingControlOutput) {
  auto flow_model = make_shared<FlowModel>();
  auto graph_1 = ge::MakeShared<ComputeGraph>("subgraph-1");
  auto graph_2 = ge::MakeShared<ComputeGraph>("subgraph-2");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  auto submodel_2 = StubModels::BuildRootModel(graph_2, false);
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  flow_model->AddSubModel(submodel_1);
  flow_model->AddSubModel(submodel_2);

  flow_model->model_relation_ = MakeShared<ModelRelation>();
  Endpoint queue_def("in_1_0", EndpointType::kQueue);
  QueueNodeUtils(queue_def).SetDepth(2L);
  flow_model->model_relation_->endpoints.emplace_back(queue_def);
  flow_model->model_relation_->root_model_endpoint_info.input_endpoint_names = {"in_1_0"};
  flow_model->model_relation_->root_model_endpoint_info.output_endpoint_names = {};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].model_name = "subgraph-1";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].input_endpoint_names = {"in_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].output_endpoint_names = {};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].model_name = "subgraph-2";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].input_endpoint_names = {"in_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].output_endpoint_names = {};

  DeployPlan deploy_plan;
  auto ret = DeployPlanner(flow_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].control_output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].control_output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetControlOutputQueueIndices().size(), 2);
}

TEST_F(DeployPlannerTest, TestAddingControlOutput_NotForInvoked) {
  auto flow_model = make_shared<FlowModel>();
  auto graph_1 = ge::MakeShared<ComputeGraph>("subgraph-1");
  auto graph_2 = ge::MakeShared<ComputeGraph>("subgraph-2");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  auto submodel_2 = StubModels::BuildRootModel(graph_2, false);
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  flow_model->AddSubModel(submodel_1);
  flow_model->AddSubModel(submodel_2);

  flow_model->model_relation_ = MakeShared<ModelRelation>();
  Endpoint queue_def("in_1_0", EndpointType::kQueue);
  QueueNodeUtils(queue_def).SetDepth(2L);
  flow_model->model_relation_->endpoints.emplace_back(queue_def);
  queue_def.SetName("invoke_in_1_0");
  flow_model->model_relation_->endpoints.emplace_back(queue_def);

  flow_model->model_relation_->root_model_endpoint_info.input_endpoint_names = {"in_1_0"};
  flow_model->model_relation_->root_model_endpoint_info.output_endpoint_names = {};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].model_name = "subgraph-1";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].input_endpoint_names = {"in_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].output_endpoint_names = {};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].model_name = "subgraph-2";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].input_endpoint_names = {"invoke_in_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].output_endpoint_names = {};
  flow_model->model_relation_->invoked_model_queue_infos["invoke_model_key"] = {{"invoke_in_1_0"},
                                                                                {}};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].invoke_model_keys = {"invoke_model_key"};
  DeployPlan deploy_plan;
  auto ret = DeployPlanner(flow_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].control_output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].control_output_queue_indices.size(), 0);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetControlOutputQueueIndices().size(), 1);
}

TEST_F(DeployPlannerTest, GetEndpointFullName) {
  DeployPlanner deploy_planner(nullptr);
  std::string empty_endpoint_name;
  ModelQueueIndex q_idx{"model_name", "", 0};
  DeployPlan::QueueInfo endpoint_info;
  endpoint_info.model_instance_name = "model_ins_name";
  endpoint_info.device_info = DeployPlan::DeviceInfo(NPU, 1, 1);
  auto full_name_0 = deploy_planner.GetEndpointFullName(endpoint_info, q_idx);
  auto found = full_name_0.find("model_ins_name:0_FROM_@0_1_1");
  EXPECT_NE(found, std::string::npos);

  std::string endpoint_name_1 = std::string(128 - full_name_0.length(), '0');
  endpoint_info.name = endpoint_name_1;
  auto full_name_1 = deploy_planner.GetEndpointFullName(endpoint_info, q_idx);
  auto full_name_2 = deploy_planner.GetEndpointFullName(endpoint_info, q_idx);
  EXPECT_EQ(full_name_1, "deploy_planner.auto_generated:0");
  EXPECT_TRUE(full_name_2 == full_name_1);

  DeployPlanner deploy_planner2(nullptr);
  auto full_name_3 = deploy_planner2.GetEndpointFullName(endpoint_info, q_idx);
  EXPECT_EQ(full_name_3, "deploy_planner.auto_generated:1");

  std::string endpoint_name_4 = std::string(127 - full_name_0.length(), '0');
  endpoint_info.name = endpoint_name_4;
  auto full_name_4 = deploy_planner.GetEndpointFullName(endpoint_info, q_idx);
  found = full_name_4.find("model_ins_name:0_FROM_" + std::string(127 - full_name_0.length(), '0') + "@0_1_1");
  EXPECT_NE(found, std::string::npos);
  EXPECT_EQ(full_name_4.length(), 127);
}

/**
 *       NetOutput
 *          |
 *          |
 *         PC_2
 *         |  \.
 *        PC_1 |
 *      /     \.
 *     |      |
 *   data1  data2
 */
TEST_F(DeployPlannerTest, TestDynamicSchedBuildDeployPlan_WithQueueBindings) {
  auto root_model = StubModels::BuildRootModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(root_model != nullptr);
  EXPECT_EQ(root_model->GetSubmodels().size(), 2);
  std::cout << root_model->GetSubmodels().size() << std::endl;
  auto model_relation = root_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(true);
  auto ret = DeployPlanner(root_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  std::map<int, std::set<int>> check;
  for (auto to_bind : deploy_plan.GetQueueBindings()) {
    check[to_bind.first].emplace(to_bind.second);
    std::cout << to_bind.first << " -> " << to_bind.second << std::endl;
  }

  cout << check.size() << endl;
  cout << check.begin()->second.size() << endl;
  cout << deploy_plan.GetQueueInfoList().size() << endl;
  cout << deploy_plan.GetQueueBindings().size() << endl;
  cout << deploy_plan.GetInputQueueIndices().size() << endl;
  cout << deploy_plan.GetOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size() << endl;

  ASSERT_EQ(check.size(), 3);
  ASSERT_EQ(check.begin()->second.size(), 2);
  // data2 -> PC_1, data2 -> PC_2
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 11);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 4);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size(), 2);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size(), 2);
}

/**
 *      NetOutput
 *          |
 *        PC_3
 *       /   \.
 *     PC_1  PC2
 *     |      |
 *   data1  data2
 */
TEST_F(DeployPlannerTest, TestBuildDynamicSchedDeployPlan_WithoutQueueBindings) {
  auto root_model = StubModels::BuildRootModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(root_model != nullptr);
  EXPECT_EQ(root_model->GetSubmodels().size(), 3);
  auto model_relation = root_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 3);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-3")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-3")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(true);
  auto ret = DeployPlanner(root_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);

  cout << deploy_plan.GetQueueInfoList().size() << endl;
  cout << deploy_plan.GetQueueBindings().size() << endl;
  cout << deploy_plan.GetInputQueueIndices().size() << endl;
  cout << deploy_plan.GetOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size() << endl;

  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 9);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 3);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size(), 3);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size(), 0);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size(), 3);
}

TEST_F(DeployPlannerTest, TestWithFusionInvokeModel) {
  std::string graph_inputs_fusion = "{\"invoke_model_key2\":\"0~2;3,4\", \"invoke_model_key3\":\"1,2\", \"invoke_model_key5\":\"0\"}";
  auto flow_model = make_shared<FlowModel>();
  auto graph_1 = ge::MakeShared<ComputeGraph>("subgraph-1");
  AttrUtils::SetStr(graph_1, "_invoked_model_fusion_inputs", graph_inputs_fusion);
  auto graph_2 = ge::MakeShared<ComputeGraph>("subgraph-2");
  auto graph_3 = ge::MakeShared<ComputeGraph>("subgraph-3");
  auto graph_4 = ge::MakeShared<ComputeGraph>("subgraph-4");
  auto graph_5 = ge::MakeShared<ComputeGraph>("subgraph-5");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  auto submodel_2 = StubModels::BuildRootModel(graph_2, false);
  auto submodel_3 = StubModels::BuildRootModel(graph_3, false);
  auto submodel_4 = StubModels::BuildRootModel(graph_4, false);
  auto submodel_5 = StubModels::BuildRootModel(graph_5, false);
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  ASSERT_TRUE(submodel_3 != nullptr);
  ASSERT_TRUE(submodel_4 != nullptr);
  ASSERT_TRUE(submodel_5 != nullptr);
  flow_model->AddSubModel(submodel_1);
  flow_model->AddSubModel(submodel_2);
  flow_model->AddSubModel(submodel_3);
  flow_model->AddSubModel(submodel_4);
  flow_model->AddSubModel(submodel_5);

  flow_model->model_relation_ = MakeShared<ModelRelation>();
  AddQueueDef(*(flow_model->model_relation_), "external_in_1_0");
  AddQueueDef(*(flow_model->model_relation_), "in_1_0");
  AddQueueDef(*(flow_model->model_relation_), "out_1_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-2_in_5_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-2_in_5_1");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-2_in_5_2");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-2_in_5_3");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-2_in_5_4");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-2_out_1_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-3_in_3_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-3_in_3_1");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-3_in_3_2");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-3_out_1_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-4_in_2_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-4_in_2_1");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-4_out_1_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-5_in_1_0");
  AddQueueDef(*(flow_model->model_relation_), "subgraph-5_out_1_0");

  flow_model->model_relation_->root_model_endpoint_info.external_input_queue_names = {"external_in_1_0"};
  flow_model->model_relation_->root_model_endpoint_info.input_endpoint_names = {"in_1_0"};
  flow_model->model_relation_->root_model_endpoint_info.output_endpoint_names = {"out_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].model_name = "subgraph-1";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].external_input_queue_names = {"external_in_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].input_endpoint_names = {"in_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].output_endpoint_names = {"out_1_0"};

  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].model_name = "subgraph-2";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].input_endpoint_names = {"subgraph-2_in_5_0",
                                                                                             "subgraph-2_in_5_1",
                                                                                             "subgraph-2_in_5_2",
                                                                                             "subgraph-2_in_5_3",
                                                                                             "subgraph-2_in_5_4"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-2"].output_endpoint_names = {"subgraph-2_out_1_0"};

  flow_model->model_relation_->submodel_endpoint_infos["subgraph-3"].model_name = "subgraph-3";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-3"].input_endpoint_names = {"subgraph-3_in_3_0",
                                                                                             "subgraph-3_in_3_1",
                                                                                             "subgraph-3_in_3_2"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-3"].output_endpoint_names = {"subgraph-3_out_1_0"};

  flow_model->model_relation_->submodel_endpoint_infos["subgraph-4"].model_name = "subgraph-4";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-4"].input_endpoint_names = {"subgraph-4_in_2_0", "subgraph-4_in_2_1"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-4"].output_endpoint_names = {"subgraph-4_out_1_0"};

  flow_model->model_relation_->submodel_endpoint_infos["subgraph-5"].model_name = "subgraph-5";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-5"].input_endpoint_names = {"subgraph-5_in_1_0"};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-5"].output_endpoint_names = {"subgraph-5_out_1_0"};

  flow_model->model_relation_->invoked_model_queue_infos["invoke_model_key2"] = {{"subgraph-2_in_5_0",
                                                                                  "subgraph-2_in_5_1",
                                                                                  "subgraph-2_in_5_2",
                                                                                  "subgraph-2_in_5_3",
                                                                                  "subgraph-2_in_5_4"},
                                                                                 {"subgraph-2_out_1_0"}};
  flow_model->model_relation_->invoked_model_queue_infos["invoke_model_key3"] = {{"subgraph-3_in_3_0",
                                                                                  "subgraph-3_in_3_1",
                                                                                  "subgraph-3_in_3_2"},
                                                                                 {"subgraph-3_out_1_0"}};
  flow_model->model_relation_->invoked_model_queue_infos["invoke_model_key4"] = {{"subgraph-4_in_2_0",
                                                                                 "subgraph-4_in_2_1"},
                                                                                 {"subgraph-4_out_1_0"}};
  flow_model->model_relation_->invoked_model_queue_infos["invoke_model_key5"] = {{"subgraph-5_in_1_0"},
                                                                                 {"subgraph-5_out_1_0"}};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].invoke_model_keys = {"invoke_model_key2", "invoke_model_key3",
                                                                                          "invoke_model_key4", "invoke_model_key5"};

  DeployPlan deploy_plan;
  auto ret = DeployPlanner(flow_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].input_queue_indices.size(), 2);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key2"].feed_queue_indices.size(), 5);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key3"].feed_queue_indices.size(), 3);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key4"].feed_queue_indices.size(), 2);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key5"].feed_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].input_queue_indices.size(), 5);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-3"].input_queue_indices.size(), 3);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-3"].output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-4"].input_queue_indices.size(), 2);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-4"].output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-5"].input_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-5"].output_queue_indices.size(), 1);
  auto &feed2_queue_indices = deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key2"].feed_queue_indices;
  ASSERT_EQ(feed2_queue_indices[0], deploy_plan.queues_[feed2_queue_indices[1]].ref_index);
  ASSERT_EQ(feed2_queue_indices[0], deploy_plan.queues_[feed2_queue_indices[2]].ref_index);
  ASSERT_EQ(feed2_queue_indices[3], deploy_plan.queues_[feed2_queue_indices[4]].ref_index);

  auto &feed3_queue_indices = deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key3"].feed_queue_indices;
  ASSERT_EQ(feed3_queue_indices[1], deploy_plan.queues_[feed3_queue_indices[2]].ref_index);

  const auto &invoked_model_queue_infos = deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos;
  ASSERT_EQ(invoked_model_queue_infos.size(), 4);
  ASSERT_EQ(invoked_model_queue_infos.count("invoke_model_key2"), 1);
  ASSERT_EQ(invoked_model_queue_infos.count("invoke_model_key3"), 1);
  ASSERT_EQ(invoked_model_queue_infos.count("invoke_model_key4"), 1);
  ASSERT_EQ(invoked_model_queue_infos.count("invoke_model_key5"), 1);

  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}

TEST_F(DeployPlannerTest, TestParseInvalidFusionInputs) {
  std::string fusion_inputs_string = "1~x;4,5,6";
  std::vector<std::vector<size_t>> fusion_inputs_list;
  auto ret = DeployPlannerBase::ParseInvokedModelFusionInputs(fusion_inputs_string, fusion_inputs_list);
  ASSERT_NE(ret, SUCCESS);

  fusion_inputs_string = "3~1;4,5,6";
  ret = DeployPlannerBase::ParseInvokedModelFusionInputs(fusion_inputs_string, fusion_inputs_list);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(DeployPlannerTest, TestGetInvalidFusionInputs) {
  std::string graph_inputs_fusion = "{\"invoke_model_key2\":\"0~2;3,4\", invalid_json}";
  auto flow_model = std::make_shared<FlowModel>();
  auto graph_1 = ge::MakeShared<ComputeGraph>("graph-1");
  AttrUtils::SetStr(graph_1, "_invoked_model_fusion_inputs", graph_inputs_fusion);
  auto model_1 = StubModels::BuildRootModel(graph_1, false);
  ASSERT_TRUE(model_1 != nullptr);
  flow_model->AddSubModel(model_1);
  std::map<std::string, std::string> fusion_inputs;
  auto ret = DeployPlannerBase::GetInvokedModelFusionInputs(model_1, fusion_inputs);
  ASSERT_NE(ret, SUCCESS);
}
}  // namespace ge
