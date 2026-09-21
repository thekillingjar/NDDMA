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
#include "common/model/ge_root_model.h"

#include "macro_utils/dt_public_scope.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "macro_utils/dt_public_unscope.h"

#include "stub_models.h"
#include "graph/build/graph_builder.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/passes/graph_builder_utils.h"
#include "dflow/compiler/data_flow_graph/process_point_loader.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "framework/common/helper/model_save_helper.h"

namespace ge {
namespace {
const char *kSessionId = "123456";

NodePtr AddNode(ut::GraphBuilder &builder, const string &name, const string &type, int in_cnt, int out_cnt) {
  auto node = builder.AddNode(name, type, in_cnt, out_cnt);
  auto op_desc = node->GetOpDesc();
  op_desc->SetInputOffset(std::vector<int64_t>(in_cnt));
  op_desc->SetOutputOffset(std::vector<int64_t>(out_cnt));
  return node;
}

static NodePtr CreateFileConstantNode(const ComputeGraphPtr &graph, const string &name, const string &location) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, FILECONSTANT);
  AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, location);
  AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 0);
  AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 2);

  op_desc->AddOutputDesc(GeTensorDesc());
  GeTensorPtr value = std::make_shared<GeTensor>(GeTensorDesc(), 64);
  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, name);
  std::vector<int64_t> shape = {2,2,2,2};
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  return graph->AddNode(op_desc);
}

void SetSubGraph(const NodePtr &node, const string &name) {
  auto op_desc = node->GetOpDesc();
  auto builder = ut::GraphBuilder(name);

  int num_inputs = static_cast<int>(op_desc->GetInputsSize());
  int num_outputs = static_cast<int>(op_desc->GetOutputsSize());
  auto some_node = AddNode(builder, name + "_node", ADDN, num_inputs, 1);
  for (int i = 0; i < num_inputs; ++i) {
    auto data_node = AddNode(builder, name + "_data_" + std::to_string(i), DATA, 1, 1);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, i);
    builder.AddDataEdge(data_node, 0, some_node, i);
  }
  auto net_output = AddNode(builder, "Node_Output", NETOUTPUT, num_outputs, num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    AttrUtils::SetInt(net_output->GetOpDesc()->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, i);
    builder.AddDataEdge(some_node, 0, net_output, i);
  }

  auto output_op_desc = net_output->GetOpDesc();
  std::vector<std::string> src_name(num_outputs, "out");
  output_op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index(num_outputs, 0);
  output_op_desc->SetSrcIndex(src_index);

  auto subgraph = builder.GetGraph();
  AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  subgraph->SetParentNode(node);
  op_desc->AddSubgraphName(name);
  op_desc->SetSubgraphInstanceName(0, name);
  auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  subgraph->SetParentGraph(root_graph);
  root_graph->AddSubgraph(name, subgraph);
}

Status BuildForPipelinePartitionedGraph(const ComputeGraphPtr &root_graph, const PneModelPtr &root_model) {
  std::unique_ptr<ModelRelation> model_relation;
  GE_CHK_STATUS_RET(ModelRelationBuilder().BuildFromRootGraph(*root_graph, model_relation),
                    "Failed to build ModelRelation from root graph: %s",
                    root_graph->GetName().c_str());
  root_model->SetModelRelation(std::shared_ptr<ModelRelation>(model_relation.release()));
  std::vector<GeRootModelPtr> submodels;
  for (const auto &node : root_graph->GetDirectNode()) {
    auto op_type = NodeUtils::GetNodeType(node);
    if (op_type != PARTITIONEDCALL) {
      continue;
    }

    auto op_desc = node->GetOpDesc();
    // convert to root graph of submodel
    auto subgraph = NodeUtils::GetSubgraph(*node, 0);
    GE_CHECK_NOTNULL(subgraph);

    GE_CHK_STATUS_RET(ProcessPointLoader::PreProcessSubgraphAttrs(subgraph),
                      "Failed to PreProcessSubGraphAttrs failed, graph[%s].", subgraph->GetName().c_str());

    GELOGD("Start to build subgraph: %s", subgraph->GetName().c_str());
    auto submodel = StubModels::BuildGraphModel(subgraph);
    GE_CHK_STATUS_RET_NOLOG(root_model->AddSubModel(submodel));
  }
  return SUCCESS;
}
}  // namespace

Status StubModels::SaveGeRootModelToModelData(const GeRootModelPtr &ge_root_model, ModelData &model_data, ModelBufferData &model_buffer_data) {
  bool is_unknown_shape = false;
  GE_ASSERT_SUCCESS(ge_root_model->CheckIsUnknownShape(is_unknown_shape),
                    "root model(id:%u) CheckIsUnknownShape failed", ge_root_model->GetModelId());
  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  EXPECT_NE(model_save_helper, nullptr);
  model_save_helper->SetSaveMode(false);
  GE_ASSERT_SUCCESS(model_save_helper->SaveToOmRootModel(ge_root_model, "NoUse", 
                          model_buffer_data, is_unknown_shape), "SaveToOmRootModel failed, model id:%u", ge_root_model->GetModelId());
  model_data.model_data = model_buffer_data.data.get();
	model_data.model_len = model_buffer_data.length;
  return SUCCESS;
}

ComputeGraphPtr StubModels::BuildSinglePartitionedCallGraph() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = AddNode(builder, "data1", DATA, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto data2 = AddNode(builder, "data2", DATA, 1, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto partitioned_call_1 = AddNode(builder, "PartitionedCall1", PARTITIONEDCALL, 2, 1);
  auto net_output = AddNode(builder, "NetOutput", NETOUTPUT, 1, 1);

  SetSubGraph(partitioned_call_1, "subgraph-1");

  builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
  builder.AddDataEdge(data2, 0, partitioned_call_1, 1);
  builder.AddDataEdge(partitioned_call_1, 0, net_output, 0);
  return builder.GetGraph();
}

ComputeGraphPtr StubModels::BuildSeriesPartitionedCallGraph() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = AddNode(builder, "data1", DATA, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto partitioned_call_1 = AddNode(builder, "PartitionedCall1", PARTITIONEDCALL, 1, 1);
  auto partitioned_call_2 = AddNode(builder, "PartitionedCall2", PARTITIONEDCALL, 1, 1);
  auto net_output = AddNode(builder, "NetOutput", NETOUTPUT, 1, 1);

  SetSubGraph(partitioned_call_1, "subgraph-1");
  SetSubGraph(partitioned_call_2, "subgraph-2");

  builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
  builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_2, 0);
  builder.AddDataEdge(partitioned_call_2, 0, net_output, 0);
  return builder.GetGraph();
}

ComputeGraphPtr StubModels::BuildParallelPartitionedCallGraph() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = AddNode(builder, "data1", DATA, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto partitioned_call_1 = AddNode(builder, "PartitionedCall1", PARTITIONEDCALL, 1, 1);
  auto partitioned_call_2 = AddNode(builder, "PartitionedCall2", PARTITIONEDCALL, 1, 0);
  auto net_output = AddNode(builder, "NetOutput", NETOUTPUT, 1, 1);

  SetSubGraph(partitioned_call_1, "subgraph-1");
  SetSubGraph(partitioned_call_2, "subgraph-2");

  builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
  builder.AddDataEdge(data1, 0, partitioned_call_2, 0);
  builder.AddDataEdge(partitioned_call_1, 0, net_output, 0);
  builder.AddControlEdge(partitioned_call_2, net_output);
  return builder.GetGraph();
}

ComputeGraphPtr StubModels::BuildParallelPartitionedCallGraphWithMultiOutputs() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = AddNode(builder, "data1", DATA, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto partitioned_call_1 = AddNode(builder, "PartitionedCall1", PARTITIONEDCALL, 1, 1);
  auto partitioned_call_2 = AddNode(builder, "PartitionedCall2", PARTITIONEDCALL, 1, 1);
  auto net_output = AddNode(builder, "NetOutput", NETOUTPUT, 2, 2);

  SetSubGraph(partitioned_call_1, "subgraph-1");
  SetSubGraph(partitioned_call_2, "subgraph-2");

  builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
  builder.AddDataEdge(data1, 0, partitioned_call_2, 0);
  builder.AddDataEdge(partitioned_call_1, 0, net_output, 0);
  builder.AddDataEdge(partitioned_call_2, 0, net_output, 1);
  return builder.GetGraph();
}

ComputeGraphPtr StubModels::BuildGraphWithQueueBindings() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = AddNode(builder, "data1", DATA, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto data2 = AddNode(builder, "data2", DATA, 1, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto partitioned_call_1 = AddNode(builder, "PartitionedCall1", PARTITIONEDCALL, 2, 1);
  auto partitioned_call_2 = AddNode(builder, "PartitionedCall2", PARTITIONEDCALL, 2, 1);
  auto net_output = AddNode(builder, "NetOutput", NETOUTPUT, 1, 1);

  SetSubGraph(partitioned_call_1, "subgraph-1");
  SetSubGraph(partitioned_call_2, "subgraph-2");

  builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
  builder.AddDataEdge(data2, 0, partitioned_call_1, 1);
  builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_2, 0);
  builder.AddDataEdge(data2, 0, partitioned_call_2, 1);
  builder.AddDataEdge(partitioned_call_2, 0, net_output, 0);
  return builder.GetGraph();
}

ComputeGraphPtr StubModels::BuildGraphWithoutNeedForBindingQueues(const std::string &prefix) {
  auto builder = ut::GraphBuilder(prefix + "g1");
  auto data1 = AddNode(builder, prefix + "data1", DATA, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto data2 = AddNode(builder, prefix + "data2", DATA, 1, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto partitioned_call_1 = AddNode(builder, prefix + "PartitionedCall1", PARTITIONEDCALL, 1, 1);
  AttrUtils::SetInt(partitioned_call_1->GetOpDesc(), "_eschedProcessPriority", 2);
  AttrUtils::SetInt(partitioned_call_1->GetOpDesc(), "_eschedEventPriority", 2);
  auto partitioned_call_2 = AddNode(builder, prefix + "PartitionedCall2", PARTITIONEDCALL, 1, 1);
  auto partitioned_call_3 = AddNode(builder, prefix + "PartitionedCall3", PARTITIONEDCALL, 2, 1);
  auto net_output = AddNode(builder, prefix + "NetOutput", NETOUTPUT, 1, 1);

  SetSubGraph(partitioned_call_1, prefix + "subgraph-1");
  SetSubGraph(partitioned_call_2, prefix + "subgraph-2");
  SetSubGraph(partitioned_call_3, prefix + "subgraph-3");

  builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
  builder.AddDataEdge(data2, 0, partitioned_call_2, 0);
  builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_3, 0);
  builder.AddDataEdge(partitioned_call_2, 0, partitioned_call_3, 1);
  builder.AddDataEdge(partitioned_call_3, 0, net_output, 0);

  return builder.GetGraph();
}

FlowModelPtr StubModels::BuildFlowModel(ComputeGraphPtr root_graph, bool pipeline_partitioned) {
  auto flow_model = make_shared<FlowModel>(root_graph);
  flow_model->SetModelType(PNE_ID_NPU);
  auto root_model = BuildRootModel(std::move(root_graph), pipeline_partitioned);
  if (root_model == nullptr) {
    return nullptr;
  }
  if (root_model->GetSubmodels().empty()) {
    flow_model->AddSubModel(root_model, PNE_ID_NPU);
  } else {
    for (auto submodel : root_model->GetSubmodels()) {
      flow_model->AddSubModel(submodel.second, PNE_ID_NPU);
    }
    flow_model->SetModelRelation(root_model->GetModelRelation());
  }
  return flow_model;
}

PneModelPtr StubModels::BuildGraphModel(ComputeGraphPtr root_graph, const std::string &model_type) {
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

  ModelData model_data{};
  ModelBufferData model_buffer_data;
  auto ret = SaveGeRootModelToModelData(ge_root_model, model_data, model_buffer_data);
  EXPECT_EQ(ret, SUCCESS);
  auto graph_model = FlowModelHelper::ToPneModel(model_data, root_graph, model_type);
  graph_model->SetModelName(root_graph->GetName());
  return graph_model;
}

PneModelPtr StubModels::BuildRootModel(ComputeGraphPtr root_graph, bool pipeline_partitioned) {
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  PneModelPtr root_model;
  if (pipeline_partitioned) {
    root_model = MakeShared<ge::FlowModel>(root_graph);
    EXPECT_NE(root_model, nullptr);
    root_model->SetModelName(root_graph->GetName());
    auto ret = BuildForPipelinePartitionedGraph(root_graph, root_model);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_NE(root_model->GetModelRelation(), nullptr);
  } else {
    root_model = BuildGraphModel(root_graph);
  }
  return root_model;
}

DeployPlan StubModels::BuildSimpleDeployPlan(int32_t remote_node_id) {
  DeployPlan deploy_plan;
  deploy_plan.group_entries_.resize(4);
  deploy_plan.group_entries_[0].device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0); // data1->[PC_1:0]
  deploy_plan.group_entries_[0].name = "data1@PC_1:0";
  deploy_plan.group_entries_[1].device_info = DeployPlan::DeviceInfo(0, 0, 0);  // [data1] -> PC_1:0
  deploy_plan.group_entries_[1].name = "data1@PC_1:0";
  deploy_plan.group_entries_[2].device_info = DeployPlan::DeviceInfo(0, 0, 0); // PC_2 -> [NetOutput:0]
  deploy_plan.group_entries_[2].name = "PC_2:0@__tail:0";
  deploy_plan.group_entries_[3].device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0); // [PC_2] -> NetOutput:0
  deploy_plan.group_entries_[3].name = "PC_2:0@__tail:0";

  deploy_plan.queues_.resize(9);
  deploy_plan.queues_[0].device_info = DeployPlan::DeviceInfo(0, 0, 0);
  deploy_plan.queues_[1].device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0);
  deploy_plan.queues_[2].device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0);
  deploy_plan.queues_[3].device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0);
  deploy_plan.queues_[4].device_info = DeployPlan::DeviceInfo(0, 0, 0);
  deploy_plan.queues_[5].device_info = DeployPlan::DeviceInfo(0, 0, 0);
  deploy_plan.groups_[5].emplace_back(0);
  deploy_plan.queues_[6].device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0);
  deploy_plan.groups_[6].emplace_back(1);
  deploy_plan.queues_[7].device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0);
  deploy_plan.groups_[7].emplace_back(2);
  deploy_plan.queues_[8].device_info = DeployPlan::DeviceInfo(0, 0, 0);
  deploy_plan.groups_[8].emplace_back(3);
  deploy_plan.root_model_info_.input_queue_indices.emplace_back(0);
  deploy_plan.root_model_info_.output_queue_indices.emplace_back(4);
  auto &submodel_1 = deploy_plan.submodels_["subgraph-1"];
  submodel_1.input_queue_indices.emplace_back(1);
  submodel_1.output_queue_indices.emplace_back(2);
  submodel_1.device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0);
  auto ge_submodel_1 = make_shared<GeRootModel>();
  ge_submodel_1->SetRootGraph(std::make_shared<ComputeGraph>("subgraph-1"));
  auto model = make_shared<GeModel>();
  auto model_task_def = make_shared<domi::ModelTaskDef>();
  model_task_def->add_task();
  model->SetModelTaskDef(model_task_def);
  model->SetGraph(ge_submodel_1->GetRootGraph());
  model->SetWeight(Buffer(512));
  ge_submodel_1->SetSubgraphInstanceNameToModel("subgraph-1", model);
  ModelData model_data_1{};
  ModelBufferData model_buffer_data_1;
  auto ret = SaveGeRootModelToModelData(ge_submodel_1, model_data_1, model_buffer_data_1);
  EXPECT_EQ(ret, SUCCESS);
  submodel_1.model = FlowModelHelper::ToPneModel(model_data_1, ge_submodel_1->GetRootGraph());

  auto &submodel_2 = deploy_plan.submodels_["subgraph-2"];
  submodel_2.input_queue_indices.emplace_back(2);
  submodel_2.output_queue_indices.emplace_back(3);
  submodel_2.device_info = DeployPlan::DeviceInfo(0, remote_node_id, 0);
  auto ge_submodel_2 = make_shared<GeRootModel>();
  ge_submodel_2->SetRootGraph(std::make_shared<ComputeGraph>("subgraph-2"));
  model = make_shared<GeModel>();
  model->SetModelTaskDef(model_task_def);
  model->SetWeight(Buffer(512));
  model->SetGraph(ge_submodel_2->GetRootGraph());
  ge_submodel_2->SetSubgraphInstanceNameToModel("subgraph-2", model);
  ModelData model_data_2{};
  ModelBufferData model_buffer_data_2;
  ret = SaveGeRootModelToModelData(ge_submodel_2, model_data_2, model_buffer_data_2);
  EXPECT_EQ(ret, SUCCESS);
  submodel_2.model = FlowModelHelper::ToPneModel(model_data_2, ge_submodel_2->GetRootGraph());

  deploy_plan.queue_bindings_.emplace_back(0, 5);
  deploy_plan.queue_bindings_.emplace_back(6, 1);
  deploy_plan.queue_bindings_.emplace_back(3, 7);
  deploy_plan.queue_bindings_.emplace_back(8, 4);
  return deploy_plan;
}

DeployPlan StubModels::BuildSingleModelDeployPlan(int32_t remote_node_id) {
  const int32_t local_node_id = 0;
  DeployPlan deploy_plan;
  deploy_plan.group_entries_.resize(4);
  deploy_plan.group_entries_[0].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[0].name = "data1@PC_1:0";
  deploy_plan.group_entries_[1].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.group_entries_[1].name = "data1@PC_1:0";
  deploy_plan.group_entries_[2].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.group_entries_[2].name = "PC_2:0@__tail:0";
  deploy_plan.group_entries_[3].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[3].name = "PC_2:0@__tail:0";

  deploy_plan.queues_.resize(8);
  // head output
  deploy_plan.queues_[0].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  // model1 output
  deploy_plan.queues_[1].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // model_1 input
  deploy_plan.queues_[2].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // tail input
  deploy_plan.queues_[3].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);

  // input tag
  deploy_plan.queues_[4].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.groups_[4].emplace_back(0);
  deploy_plan.queues_[5].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[5].emplace_back(1);

  // output tag
  deploy_plan.queues_[6].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[6].emplace_back(2);
  deploy_plan.queues_[7].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.groups_[7].emplace_back(3);

  deploy_plan.root_model_info_.input_queue_indices.emplace_back(0);
  deploy_plan.root_model_info_.output_queue_indices.emplace_back(3);

  auto &submodel_1 = deploy_plan.submodels_["subgraph-1"];
  submodel_1.input_queue_indices.emplace_back(2);
  submodel_1.output_queue_indices.emplace_back(1);
  submodel_1.device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  auto ge_submodel_1 = make_shared<GeRootModel>();
  ge_submodel_1->SetRootGraph(std::make_shared<ComputeGraph>("subgraph-1"));
  auto model = make_shared<GeModel>();
  auto model_task_def = make_shared<domi::ModelTaskDef>();
  model_task_def->add_task();
  model->SetModelTaskDef(model_task_def);
  model->SetGraph(ge_submodel_1->GetRootGraph());
  model->SetWeight(Buffer(512));
  ge_submodel_1->SetSubgraphInstanceNameToModel("subgraph-1", model);
  ModelData model_data_1{};
  ModelBufferData model_buffer_data_1;
  auto ret = SaveGeRootModelToModelData(ge_submodel_1, model_data_1, model_buffer_data_1);
  EXPECT_EQ(ret, SUCCESS);
  submodel_1.model = FlowModelHelper::ToPneModel(model_data_1, ge_submodel_1->GetRootGraph());

  deploy_plan.queue_bindings_.emplace_back(0, 4);
  deploy_plan.queue_bindings_.emplace_back(5, 2);
  deploy_plan.queue_bindings_.emplace_back(1, 6);
  deploy_plan.queue_bindings_.emplace_back(7, 3);
  return deploy_plan;
}

DeployPlan StubModels::BuildSingleModelDeployPlanWithDummyQ(int32_t remote_node_id) {
  const int32_t local_node_id = 0;
  DeployPlan deploy_plan;
  deploy_plan.group_entries_.resize(4);
  deploy_plan.group_entries_[0].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[0].name = "data1@PC_1:0";
  deploy_plan.group_entries_[1].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.group_entries_[1].name = "data1@PC_1:0";
  deploy_plan.group_entries_[2].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.group_entries_[2].name = "PC_2:0@__tail:0";
  deploy_plan.group_entries_[3].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[3].name = "PC_2:0@__tail:0";

  deploy_plan.queues_.resize(9);
  // head output
  deploy_plan.queues_[0].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  // model1 output
  deploy_plan.queues_[1].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // model_1 input
  deploy_plan.queues_[2].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // tail input
  deploy_plan.queues_[3].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);

  // input tag
  deploy_plan.queues_[4].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.groups_[4].emplace_back(0);
  deploy_plan.queues_[5].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[5].emplace_back(1);

  // output tag
  deploy_plan.queues_[6].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[6].emplace_back(2);
  deploy_plan.queues_[7].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.groups_[7].emplace_back(3);

  deploy_plan.queues_[7].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.queues_[8].is_dummy = true;

  deploy_plan.root_model_info_.input_queue_indices.emplace_back(0);
  deploy_plan.root_model_info_.output_queue_indices.emplace_back(3);

  auto &submodel_1 = deploy_plan.submodels_["subgraph-1"];
  submodel_1.input_queue_indices.emplace_back(2);
  submodel_1.output_queue_indices.emplace_back(1);
  submodel_1.device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  auto ge_submodel_1 = make_shared<GeRootModel>();
  ge_submodel_1->SetRootGraph(std::make_shared<ComputeGraph>("subgraph-1"));
  auto model = make_shared<GeModel>();
  auto model_task_def = make_shared<domi::ModelTaskDef>();
  model_task_def->add_task();
  model->SetModelTaskDef(model_task_def);
  model->SetGraph(ge_submodel_1->GetRootGraph());
  model->SetWeight(Buffer(512));
  ge_submodel_1->SetSubgraphInstanceNameToModel("subgraph-1", model);
  ModelData model_data_1{};
  ModelBufferData model_buffer_data_1;
  auto ret = SaveGeRootModelToModelData(ge_submodel_1, model_data_1, model_buffer_data_1);
  EXPECT_EQ(ret, SUCCESS);
  submodel_1.model = FlowModelHelper::ToPneModel(model_data_1, ge_submodel_1->GetRootGraph());

  deploy_plan.queue_bindings_.emplace_back(0, 4);
  deploy_plan.queue_bindings_.emplace_back(5, 2);
  deploy_plan.queue_bindings_.emplace_back(1, 6);
  deploy_plan.queue_bindings_.emplace_back(7, 3);
  return deploy_plan;
}

DeployPlan StubModels::BuildSingleModelWithFileConstDeployPlan(const std::string &location, int32_t remote_node_id) {
  const int32_t local_node_id = 0;
  DeployPlan deploy_plan;
  deploy_plan.group_entries_.resize(4);
  deploy_plan.group_entries_[0].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[0].name = "data1@PC_1:0";
  deploy_plan.group_entries_[1].device_info = DeployPlan::DeviceInfo(CPU, local_node_id, 0);
  deploy_plan.group_entries_[1].name = "data1@PC_1:0";
  deploy_plan.group_entries_[2].device_info = DeployPlan::DeviceInfo(CPU, local_node_id, 0);
  deploy_plan.group_entries_[2].name = "PC_2:0@__tail:0";
  deploy_plan.group_entries_[3].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[3].name = "PC_2:0@__tail:0";

  deploy_plan.queues_.resize(8);
  // head output
  deploy_plan.queues_[0].device_info = DeployPlan::DeviceInfo(CPU, local_node_id, 0);
  // model1 output
  deploy_plan.queues_[1].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // model_1 input
  deploy_plan.queues_[2].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // tail input
  deploy_plan.queues_[3].device_info = DeployPlan::DeviceInfo(CPU, local_node_id, 0);

  // input tag
  deploy_plan.queues_[4].device_info = DeployPlan::DeviceInfo(CPU, local_node_id, 0);
  deploy_plan.groups_[4].emplace_back(0);
  deploy_plan.queues_[5].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[5].emplace_back(1);

  // output tag
  deploy_plan.queues_[6].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[6].emplace_back(2);
  deploy_plan.queues_[7].device_info = DeployPlan::DeviceInfo(CPU, local_node_id, 0);
  deploy_plan.groups_[7].emplace_back(3);

  deploy_plan.root_model_info_.input_queue_indices.emplace_back(0);
  deploy_plan.root_model_info_.output_queue_indices.emplace_back(3);

  auto &submodel_1 = deploy_plan.submodels_["subgraph-1"];
  submodel_1.input_queue_indices.emplace_back(2);
  submodel_1.output_queue_indices.emplace_back(1);
  submodel_1.device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  auto ge_submodel_1 = make_shared<GeRootModel>();

  auto graph = std::make_shared<ComputeGraph>("subgraph-1");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  auto const_1 = CreateFileConstantNode(graph, "file_const_1", location);

  ge_submodel_1->SetRootGraph(graph);
  auto model = make_shared<GeModel>();
  auto model_task_def = make_shared<domi::ModelTaskDef>();
  model_task_def->add_task();
  model->SetModelTaskDef(model_task_def);
  model->SetGraph(ge_submodel_1->GetRootGraph());
  model->SetWeight(Buffer(512));
  ge_submodel_1->SetSubgraphInstanceNameToModel("subgraph-1", model);
  ModelData model_data_1{};
  ModelBufferData model_buffer_data_1;
  auto ret = SaveGeRootModelToModelData(ge_submodel_1, model_data_1, model_buffer_data_1);
  EXPECT_EQ(ret, SUCCESS);
  submodel_1.model = FlowModelHelper::ToPneModel(model_data_1, ge_submodel_1->GetRootGraph());

  deploy_plan.queue_bindings_.emplace_back(0, 4);
  deploy_plan.queue_bindings_.emplace_back(5, 2);
  deploy_plan.queue_bindings_.emplace_back(1, 6);
  deploy_plan.queue_bindings_.emplace_back(7, 3);
  return deploy_plan;
}

DeployPlan StubModels::BuildSingleModelDeployPlanWithProxy(int32_t remote_node_id) {
  const int32_t local_node_id = 0;
  DeployPlan deploy_plan;
  deploy_plan.group_entries_.resize(4);
  deploy_plan.group_entries_[0].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[0].name = "data1@PC_1:0";
  deploy_plan.group_entries_[1].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.group_entries_[1].name = "data1@PC_1:0";
  deploy_plan.group_entries_[2].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.group_entries_[2].name = "PC_2:0@__tail:0";
  deploy_plan.group_entries_[3].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.group_entries_[3].name = "PC_2:0@__tail:0";

  deploy_plan.queues_.resize(8);
  // head output
  deploy_plan.queues_[0].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  // model1 output
  deploy_plan.queues_[1].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // model_1 input
  deploy_plan.queues_[2].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  // tail input
  deploy_plan.queues_[3].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);

  // input tag
  deploy_plan.queues_[4].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.groups_[4].emplace_back(0);
  deploy_plan.queues_[5].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[5].emplace_back(1);

  // output tag
  deploy_plan.queues_[6].device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  deploy_plan.groups_[6].emplace_back(2);
  deploy_plan.queues_[7].device_info = DeployPlan::DeviceInfo(NPU, local_node_id, 0);
  deploy_plan.groups_[7].emplace_back(3);

  deploy_plan.root_model_info_.input_queue_indices.emplace_back(0);
  deploy_plan.root_model_info_.output_queue_indices.emplace_back(3);

  auto &submodel_1 = deploy_plan.submodels_["subgraph-1"];
  submodel_1.input_queue_indices.emplace_back(2);
  submodel_1.output_queue_indices.emplace_back(1);
  submodel_1.device_info = DeployPlan::DeviceInfo(NPU, remote_node_id, 0);
  auto ge_submodel_1 = make_shared<GeRootModel>();
  ge_submodel_1->SetRootGraph(std::make_shared<ComputeGraph>("subgraph-1"));
  auto model = make_shared<GeModel>();
  auto model_task_def = make_shared<domi::ModelTaskDef>();
  model_task_def->add_task();
  model->SetModelTaskDef(model_task_def);
  model->SetGraph(ge_submodel_1->GetRootGraph());
  model->SetWeight(Buffer(512));
  ge_submodel_1->SetSubgraphInstanceNameToModel("subgraph-1", model);
  ModelData model_data_1{};
  ModelBufferData model_buffer_data_1;
  auto ret = SaveGeRootModelToModelData(ge_submodel_1, model_data_1, model_buffer_data_1);
  EXPECT_EQ(ret, SUCCESS);
  submodel_1.model = FlowModelHelper::ToPneModel(model_data_1, ge_submodel_1->GetRootGraph());

  deploy_plan.queue_bindings_.emplace_back(0, 4);
  deploy_plan.queue_bindings_.emplace_back(5, 2);
  deploy_plan.queue_bindings_.emplace_back(1, 6);
  deploy_plan.queue_bindings_.emplace_back(7, 3);
  return deploy_plan;
}
}  // namespace ge
