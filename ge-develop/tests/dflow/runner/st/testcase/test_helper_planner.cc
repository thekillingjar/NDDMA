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

#include "macro_utils/dt_public_scope.h"
#include "deploy/resource/heterogeneous_deploy_planner.h"
#include "deploy/flowrm/flow_route_planner.h"
#include "macro_utils/dt_public_unscope.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "common/model/ge_root_model.h"
#include "framework/common/helper/model_save_helper.h"

using namespace std;

namespace ge {
class HeterogeneousPlannerTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

namespace {
PneModelPtr BuildPneModel(ComputeGraphPtr root_graph) {
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  auto ge_model = MakeShared<ge::GeModel>();
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName(root_graph->GetName());
  ge_model->SetGraph(root_graph);
  ge_root_model->SetModelName(root_graph->GetName());	
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);
  bool is_unknown_shape = false;
  auto ret = ge_root_model->CheckIsUnknownShape(is_unknown_shape);
  EXPECT_EQ(ret, SUCCESS);
  ModelBufferData model_buffer_data{};
  const auto model_save_helper =
      ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  EXPECT_NE(model_save_helper, nullptr);
  model_save_helper->SetSaveMode(false);
  ret = model_save_helper->SaveToOmRootModel(ge_root_model, "NoUse", model_buffer_data, is_unknown_shape);
  EXPECT_EQ(ret, SUCCESS);
  ModelData model_data{};
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;
  PneModelPtr pne_model = FlowModelHelper::ToPneModel(model_data, root_graph);
  return pne_model;
}

void AddQueueDef(ModelRelation &model_relation, const std::string &name) {
  Endpoint queue_def(name, EndpointType::kQueue);
  model_relation.endpoints.emplace_back(queue_def);
}
}  // namespace

/**
 *    NetOutput
 *       |
 *     PC_3     -> submodel_2
 *       |
 *     data1
 * 
 *    NetOutput
 *       |
 *     PC_2     -> submodel_1
 *       |
 *     data1
 * 
 *    NetOutput
 *       |
 *     PC_1     -> root model
 *      |
 *    Data
 */
TEST_F(HeterogeneousPlannerTest, TestFlattenModelRelation_3_level) {
  auto submodel_3 = std::make_shared<FlowModel>();  // leaf
  submodel_3->SetModelName("submodel_3");

  auto submodel_2 = std::make_shared<FlowModel>();  // mid-2
  {
    submodel_2->SetModelName("submodel_2");
    submodel_2->AddSubModel(submodel_3);
    auto model_relation = std::make_shared<ModelRelation>();
    submodel_2->SetModelRelation(model_relation);
    AddQueueDef(*model_relation, "submodel_2_input_0");
    AddQueueDef(*model_relation, "submodel_2_output_0");
    model_relation->root_model_endpoint_info.input_endpoint_names = {"submodel_2_input_0"};
    model_relation->root_model_endpoint_info.output_endpoint_names = {"submodel_2_output_0"};
    model_relation->submodel_endpoint_infos["submodel_3"].model_name = "submodel_3";
    model_relation->submodel_endpoint_infos["submodel_3"].input_endpoint_names = {"submodel_2_input_0"};
    model_relation->submodel_endpoint_infos["submodel_3"].output_endpoint_names = {"submodel_2_output_0"};
  }

  auto submodel_1 = std::make_shared<FlowModel>();  // mid-1
  {
    submodel_1->SetModelName("submodel_1");
    submodel_1->AddSubModel(submodel_2);
    auto model_relation = std::make_shared<ModelRelation>();
    submodel_1->SetModelRelation(model_relation);
    AddQueueDef(*model_relation, "submodel_1_input_0");
    AddQueueDef(*model_relation, "submodel_1_output_0");
    model_relation->root_model_endpoint_info.input_endpoint_names = {"submodel_1_input_0"};
    model_relation->root_model_endpoint_info.output_endpoint_names = {"submodel_1_output_0"};
    model_relation->submodel_endpoint_infos["submodel_2"].model_name = "submodel_2";
    model_relation->submodel_endpoint_infos["submodel_2"].input_endpoint_names = {"submodel_1_input_0"};
    model_relation->submodel_endpoint_infos["submodel_2"].output_endpoint_names = {"submodel_1_output_0"};
  }

  auto root_model = std::make_shared<FlowModel>();  // root
  {
    root_model->SetModelName("root_model");
    root_model->AddSubModel(submodel_1);
    auto model_relation = std::make_shared<ModelRelation>();
    root_model->SetModelRelation(model_relation);
    AddQueueDef(*model_relation, "input_0");
    AddQueueDef(*model_relation, "output_0");
    model_relation->root_model_endpoint_info.input_endpoint_names = {"input_0"};
    model_relation->root_model_endpoint_info.output_endpoint_names = {"output_0"};
    model_relation->submodel_endpoint_infos["submodel_1"].model_name = "submodel_1";
    model_relation->submodel_endpoint_infos["submodel_1"].input_endpoint_names = {"input_0"};
    model_relation->submodel_endpoint_infos["submodel_1"].output_endpoint_names = {"output_0"};
  }

  ModelRelationFlattener flattener(root_model);
  ModelRelation flattened_model_relation;
  std::map<std::string, PneModelPtr> models;
  EXPECT_EQ(flattener.Flatten(flattened_model_relation, models), SUCCESS);
  EXPECT_EQ(models.size(), 1);
  EXPECT_EQ(flattened_model_relation.submodel_endpoint_infos.size(), 1);
  std::set<std::string>
      expected_queue_names = {"input_0", "output_0"};
  std::set<std::string> queue_names;
  for (const auto &queue_def : flattened_model_relation.endpoints) {
    queue_names.emplace(queue_def.GetName());
  }
  EXPECT_EQ(queue_names, expected_queue_names);

  ModelRelationFlattener shallow_flattener(root_model);
  shallow_flattener.max_depth_ = 1;
  EXPECT_EQ(shallow_flattener.Flatten(flattened_model_relation, models), UNSUPPORTED);
}

TEST_F(HeterogeneousPlannerTest, TestLongNameSuccess) {
  auto flow_model = make_shared<FlowModel>();
  flow_model->AddSubModel(BuildPneModel(std::make_shared<ComputeGraph>("subgraph-1")));

  flow_model->model_relation_ = MakeShared<ModelRelation>();
  std::string log_in_name = "in_1_0";
  for (int32_t i = 0; i < 120; ++i) {
    log_in_name += "x";
  }
  AddQueueDef(*(flow_model->model_relation_), log_in_name);
  std::string log_out_name = "out_1_0";
  for (int32_t i = 0; i < 128; ++i) {
    log_out_name += "x";
  }
  AddQueueDef(*(flow_model->model_relation_), log_out_name);

  flow_model->model_relation_->root_model_endpoint_info.input_endpoint_names = {log_in_name};
  flow_model->model_relation_->root_model_endpoint_info.output_endpoint_names = {log_out_name};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].model_name = "subgraph-1";
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].input_endpoint_names = {log_in_name};
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].output_endpoint_names = {log_out_name};

  DeployPlan deploy_plan;
  DeployPlanner planner(flow_model);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousPlannerTest, TestWithFusionInvokeModel) {
  std::string graph_inputs_fusion = "{\"invoke_model_key2\":\"0~2;3,4\", \"invoke_model_key3\":\"1,2\"}";
  auto flow_model = make_shared<FlowModel>();
  auto graph_1 = ge::MakeShared<ComputeGraph>("subgraph-1");
  AttrUtils::SetStr(graph_1, "_invoked_model_fusion_inputs", graph_inputs_fusion);
  flow_model->AddSubModel(BuildPneModel(graph_1));
  flow_model->AddSubModel(BuildPneModel(std::make_shared<ComputeGraph>("subgraph-2")));
  flow_model->AddSubModel(BuildPneModel(std::make_shared<ComputeGraph>("subgraph-3")));
  flow_model->AddSubModel(BuildPneModel(std::make_shared<ComputeGraph>("subgraph-4")));

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
  flow_model->model_relation_->submodel_endpoint_infos["subgraph-1"].invoke_model_keys = {"invoke_model_key2", "invoke_model_key3", "invoke_model_key4"};

  DeployPlan deploy_plan;
  auto ret = DeployPlanner(flow_model).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].input_queue_indices.size(), 2);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key2"].feed_queue_indices.size(), 5);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key3"].feed_queue_indices.size(), 3);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key4"].feed_queue_indices.size(), 2);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-1"].output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].input_queue_indices.size(), 5);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-2"].output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-3"].input_queue_indices.size(), 3);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-3"].output_queue_indices.size(), 1);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-4"].input_queue_indices.size(), 2);
  ASSERT_EQ(deploy_plan.submodels_["subgraph-4"].output_queue_indices.size(), 1);
  auto &feed2_queue_indices = deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key2"].feed_queue_indices;
  ASSERT_EQ(feed2_queue_indices[0], deploy_plan.queues_[feed2_queue_indices[1]].ref_index);
  ASSERT_EQ(feed2_queue_indices[0], deploy_plan.queues_[feed2_queue_indices[2]].ref_index);
  ASSERT_EQ(feed2_queue_indices[3], deploy_plan.queues_[feed2_queue_indices[4]].ref_index);

  auto &feed3_queue_indices = deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos["invoke_model_key3"].feed_queue_indices;
  ASSERT_EQ(feed3_queue_indices[1], deploy_plan.queues_[feed3_queue_indices[2]].ref_index);

  const auto &invoked_model_queue_infos = deploy_plan.submodels_["subgraph-1"].invoked_model_queue_infos;
  ASSERT_EQ(invoked_model_queue_infos.size(), 3);
  ASSERT_EQ(invoked_model_queue_infos.count("invoke_model_key2"), 1);
  ASSERT_EQ(invoked_model_queue_infos.count("invoke_model_key3"), 1);
  ASSERT_EQ(invoked_model_queue_infos.count("invoke_model_key4"), 1);

  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}

TEST_F(HeterogeneousPlannerTest, TestGetInvalidFusionInputs) {
  std::string graph_inputs_fusion = "{\"invoke_model_key2\":\"0~2;3,4\", invalid_json}";
  auto graph_1 = ge::MakeShared<ComputeGraph>("graph-1");
  AttrUtils::SetStr(graph_1, "_invoked_model_fusion_inputs", graph_inputs_fusion);
  auto graph_model = BuildPneModel(graph_1);
  std::map<std::string, std::string> fusion_inputs;
  auto ret = DeployPlannerBase::GetInvokedModelFusionInputs(graph_model, fusion_inputs);
  ASSERT_NE(ret, SUCCESS);
}
}  // namespace ge
