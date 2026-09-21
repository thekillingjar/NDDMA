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
#include <fstream>
#include "mmpa/mmpa_api.h"
#include "dflow/compiler/model/flow_model_cache.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/ge_local_context.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "common/model/ge_root_model.h"
#include "dflow/base/model/endpoint.h"
#include "dflow/base/model/model_deploy_resource.h"
#include "common/helper/model_parser_base.h"
#include "framework/common/types.h"
#include "dflow/compiler/pne/udf/udf_model.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "framework/common/helper/om_file_helper.h"
#include "dflow/base/model/flow_model_om_loader.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/helper/model_save_helper.h"
#include "graph/manager/graph_var_manager.h"

namespace ge {
namespace {
ComputeGraphPtr FakeComputeGraph(const string &graph_name) {
  DEF_GRAPH(graph1) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto op_desc = root_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return root_graph;
}

ComputeGraphPtr FakeGraphWithSubGraph(const string &graph_name, ComputeGraphPtr sub_graph) {
  DEF_GRAPH(graph1) {
    auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto partitioned_call_op =
        OP_CFG(PARTITIONEDCALL).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    CHAIN(NODE("data", data)->NODE(partitioned_call_op)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  root_graph->AddSubGraph(sub_graph);
  sub_graph->SetParentGraph(root_graph);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  return root_graph;
}

ComputeGraphPtr FakeComputeGraphWithConstant(const string &graph_name) {
  DEF_GRAPH(graph1) {
    auto constant_0 = OP_CFG(CONSTANTOP).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE(graph_name + "_constant_0", constant_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto const_op_desc = root_graph->FindFirstNodeMatchType(CONSTANTOP)->GetOpDesc();
  GeTensor weight;
  std::vector<int32_t> data(16, 1);
  weight.SetData((uint8_t *)data.data(), data.size() * sizeof(int32_t));
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({16}));
  weight_desc.SetOriginShape(GeShape({16}));
  weight.SetTensorDesc(weight_desc);
  AttrUtils::SetTensor(const_op_desc, "value", weight);

  auto op_desc = root_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return root_graph;
}

ComputeGraphPtr FakeComputeGraphWithVar(const string &graph_name) {
  DEF_GRAPH(graph1) {
    auto var_0 = OP_CFG(VARIABLE).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("1_ascend_mbatch_batch_0", var_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto op_desc = root_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return root_graph;
}

PneModelPtr BuildPneModel(const string &name, ComputeGraphPtr graph) {
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  GE_ASSERT_SUCCESS(ge_root_model->Initialize(graph));
  auto ge_model = MakeShared<ge::GeModel>();
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName(name);
  ge_model->SetGraph(graph);
  ge_root_model->SetModelName(name);	
  ge_root_model->SetSubgraphInstanceNameToModel(name, ge_model);	
  bool is_unknown_shape = false;
  GE_ASSERT_SUCCESS(ge_root_model->CheckIsUnknownShape(is_unknown_shape));
  ModelBufferData model_buffer_data{};
  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  model_save_helper->SetSaveMode(false);
  GE_ASSERT_SUCCESS(model_save_helper->SaveToOmRootModel(ge_root_model, name, model_buffer_data, is_unknown_shape));
  ModelData model_data{};
  model_data.model_data = model_buffer_data.data.get();
	model_data.model_len = model_buffer_data.length;
  auto graph_model = FlowModelHelper::ToPneModel(model_data, graph);
  graph_model->SetLogicDeviceId("0:0:1");
  return graph_model;
}

UdfModelPtr BuildUdfModel(const string &name, ComputeGraphPtr graph) {
  auto udf_model = MakeShared<UdfModel>(graph);
  udf_model->SetModelName(name);
  auto &udf_model_def = udf_model->MutableUdfModelDef();
  auto *udf_def = udf_model_def.add_udf_def();
  udf_def->set_name(name);
  udf_def->set_func_name(name);
  auto deploy_resource = MakeShared<ModelDeployResource>();
  deploy_resource->resource_type = "x86";
  udf_model->SetDeployResource(deploy_resource);
  udf_model->SetSavedModelPath("./workspace/release/test1_release.tar.gz");
  udf_model->SetNormalizedModelName("test1_release");
  return udf_model;
}

void AddModelQueueDef(const std::shared_ptr<ModelRelation> &model_relation, const std::vector<std::string> &names) {
  for (const auto &name : names) {
    Endpoint queue_def(name, EndpointType::kQueue);
    QueueNodeUtils(queue_def).SetDepth(100L).SetEnqueuePolicy("FIFO").SetNodeAction(kQueueActionControl);
    model_relation->endpoints.emplace_back(queue_def);
  }
}

void AddSubModelQueueInfo(const std::shared_ptr<ModelRelation> &model_relation, const std::string &model_name,
                          const ModelRelation::ModelEndpointInfo &model_queue_info) {
  AddModelQueueDef(model_relation, model_queue_info.input_endpoint_names);
  AddModelQueueDef(model_relation, model_queue_info.output_endpoint_names);
  AddModelQueueDef(model_relation, model_queue_info.external_input_queue_names);
  AddModelQueueDef(model_relation, model_queue_info.external_output_queue_names);
  model_relation->submodel_endpoint_infos.emplace(model_name, model_queue_info);
}

void AddSubModelP2pNodeInfo(const std::shared_ptr<ModelRelation> &model_relation, const std::string &model_name,
                          const ModelRelation::ModelEndpointInfo &model_queue_info) {
  AddModelQueueDef(model_relation, model_queue_info.input_endpoint_names);
  AddModelQueueDef(model_relation, model_queue_info.output_endpoint_names);
  AddModelQueueDef(model_relation, model_queue_info.external_input_queue_names);
  AddModelQueueDef(model_relation, model_queue_info.external_output_queue_names);
  model_relation->submodel_endpoint_infos.emplace(model_name, model_queue_info);
}

FlowModelPtr BuildFlowModelWithOutP2pNode(const string &name, ComputeGraphPtr graph) {
  auto graph1 = FakeComputeGraph("graph1");
  auto graph2 = FakeComputeGraph("graph2");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  auto ge_root_model2 = BuildPneModel("graph2", graph2);
  auto model_relation = MakeShared<ModelRelation>();
  ModelRelation::ModelEndpointInfo model1_queue_info;
  model1_queue_info.input_endpoint_names = {"model1_in1", "model1_in2"};
  model1_queue_info.output_endpoint_names = {"model1_out1", "model1_out2"};
  AddSubModelP2pNodeInfo(model_relation, "model1", model1_queue_info);
  ModelRelation::ModelEndpointInfo model2_queue_info;
  model2_queue_info.input_endpoint_names = {"model2_in1", "model2_in2"};
  model2_queue_info.output_endpoint_names = {"model2_out1", "model2_out2"};
  AddSubModelP2pNodeInfo(model_relation, "model2", model2_queue_info);

  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->SetModelRelation(model_relation);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);
  flow_model->AddSubModel(ge_root_model2, PNE_ID_CPU);
  return flow_model;
}

FlowModelPtr BuildFlowModelWithoutUdf(const string &name, ComputeGraphPtr graph) {
  auto graph1 = FakeComputeGraph("graph1");
  auto graph2 = FakeComputeGraph("graph2");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  auto ge_root_model2 = BuildPneModel("graph2", graph2);
  auto model_relation = MakeShared<ModelRelation>();
  ModelRelation::ModelEndpointInfo model1_queue_info;
  model1_queue_info.input_endpoint_names = {"model1_in1", "model1_in2"};
  model1_queue_info.output_endpoint_names = {"model1_out1", "model1_out2"};
  AddSubModelQueueInfo(model_relation, "model1", model1_queue_info);
  ModelRelation::ModelEndpointInfo model2_queue_info;
  model2_queue_info.input_endpoint_names = {"model2_in1", "model2_in2"};
  model2_queue_info.output_endpoint_names = {"model2_out1", "model2_out2"};
  AddSubModelQueueInfo(model_relation, "model2", model2_queue_info);

  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->SetModelRelation(model_relation);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);
  flow_model->AddSubModel(ge_root_model2, PNE_ID_CPU);
  return flow_model;
}

FlowModelPtr BuildFlowModelWithOutUdfV2(const string &name, ComputeGraphPtr graph) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  auto graph1 = FakeComputeGraphWithConstant("graph1");
  EXPECT_EQ(VarMemAssignUtil::AssignConstantOpMemory(graph1), SUCCESS);
  const auto &constant_node = graph1->FindNode("graph1_constant_0");
  const auto &fake_node = graph1->FindNode("fused_op1");
  fake_node->GetOpDesc()->SetInputOffset(constant_node->GetOpDesc()->GetOutputOffset());
  fake_node->GetOpDesc()->SetOutputOffset(constant_node->GetOpDesc()->GetOutputOffset());
  auto graph2 = FakeComputeGraphWithVar("graph2");
  EXPECT_EQ(VarMemAssignUtil::AssignConstantOpMemory(graph2), SUCCESS);
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  auto ge_root_model2 = BuildPneModel("graph2", graph2);
  auto model_relation = MakeShared<ModelRelation>();
  ModelRelation::ModelEndpointInfo model1_queue_info;
  model1_queue_info.input_endpoint_names = {"model1_in1", "model1_in2"};
  model1_queue_info.output_endpoint_names = {"model1_out1", "model1_out2"};
  AddSubModelQueueInfo(model_relation, "model1", model1_queue_info);
  ModelRelation::ModelEndpointInfo model2_queue_info;
  model2_queue_info.input_endpoint_names = {"model2_in1", "model2_in2"};
  model2_queue_info.output_endpoint_names = {"model2_out1", "model2_out2"};
  AddSubModelQueueInfo(model_relation, "model2", model2_queue_info);

  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->SetModelRelation(model_relation);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);
  flow_model->AddSubModel(ge_root_model2, PNE_ID_CPU);
  return flow_model;
}

FlowModelPtr BuildFlowModelWithUdf(const string &name, ComputeGraphPtr graph) {
  auto graph1 = FakeComputeGraph("graph1");
  auto udf_graph = FakeComputeGraph("udf_graph");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  auto udf_model = BuildUdfModel("udf_model", udf_graph);
  auto model_relation = MakeShared<ModelRelation>();
  ModelRelation::ModelEndpointInfo model1_queue_info;
  model1_queue_info.input_endpoint_names = {"model1_in1", "model1_in2"};
  model1_queue_info.output_endpoint_names = {"model1_out1", "model1_out2"};
  AddSubModelQueueInfo(model_relation, "model1", model1_queue_info);

  ModelRelation::ModelEndpointInfo udf_model_queue_info;
  udf_model_queue_info.input_endpoint_names = {"udf_model_in1", "udf_model_in2"};
  udf_model_queue_info.output_endpoint_names = {"udf_model_out1", "udf_model_out2"};
  udf_model_queue_info.invoke_model_keys = {"invoke_key1"};
  ModelRelation::InvokedModelQueueInfo invoke_model_queue_info;
  invoke_model_queue_info.input_queue_names = {"model1_in1", "model1_in2"};
  invoke_model_queue_info.output_queue_names = {"udf_model_out1", "udf_model_out2"};
  AddSubModelQueueInfo(model_relation, "udf_model", udf_model_queue_info);

  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->SetModelRelation(model_relation);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);
  flow_model->AddSubModel(udf_model, PNE_ID_UDF);
  return flow_model;
}

bool ReadIndexFile(const std::string &index_file, std::vector<ge::CacheFileIndex> &cache_file_list) {
  nlohmann::json json_obj;
  std::ifstream file_stream(index_file);
  if (!file_stream.is_open()) {
    std::cout << "Failed to open cache index file:" << index_file << std::endl;
    return false;
  }

  try {
    file_stream >> json_obj;
  } catch (const nlohmann::json::exception &e) {
    std::cout << "Failed to read cache index file:" << index_file << ", err msg:" << e.what() << std::endl;
    file_stream.close();
    return false;
  }

  try {
    cache_file_list = json_obj["cache_file_list"].get<std::vector<ge::CacheFileIndex>>();
  } catch (const nlohmann::json::exception &e) {
    std::cout << "Failed to read cache index file:" << index_file << ", err msg:" << e.what() << std::endl;
    file_stream.close();
    return false;
  }
  file_stream.close();
  return true;
}

bool CheckCacheResult(const std::string &cache_dir, const std::string &graph_key, size_t expect_cache_size, bool compare_key = true) {
  const auto cache_idx_file = cache_dir + "/" + graph_key + ".idx";
  auto check_ret = mmAccess(cache_idx_file.c_str());
  if (check_ret != 0) {
    std::cout << "Cache index file:" << cache_idx_file << " is not exist" << std::endl;
    return false;
  }

  std::vector<ge::CacheFileIndex> cache_file_list;
  if (!ReadIndexFile(cache_idx_file, cache_file_list)) {
    std::cout << "Faile to read cache index file:" << cache_idx_file << std::endl;
    return false;
  }
  for (auto &idx : cache_file_list) {
    idx.cache_file_name = cache_dir + "/" + idx.cache_file_name;
  }
  if (cache_file_list.size() != expect_cache_size) {
    std::cout << "Cache file size[" << cache_file_list.size() << "] error, expect = " << expect_cache_size << std::endl;
    return false;
  }

  for (const auto &cache_index : cache_file_list) {
    if (compare_key && cache_index.graph_key != graph_key) {
      std::cout << "Cache graph_key[" << cache_index.graph_key << "] error, expect = " << graph_key << std::endl;
      return false;
    }

    if (cache_index.cache_file_name.empty()) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is empty" << std::endl;
      return false;
    }

    check_ret = mmAccess(cache_index.cache_file_name.c_str());
    if (check_ret != 0) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is not exist" << std::endl;
      return false;
    }
  }
  return true;
}

std::string GetCacheFileName(const std::string &cache_dir, const std::string &graph_key) {
  const auto cache_idx_file = cache_dir + "/" + graph_key + ".idx";
  auto check_ret = mmAccess(cache_idx_file.c_str());
  if (check_ret != 0) {
    std::cout << "Cache index file:" << cache_idx_file << " is not exist" << std::endl;
    return "";
  }

  std::vector<ge::CacheFileIndex> cache_file_list;
  if (!ReadIndexFile(cache_idx_file, cache_file_list)) {
    std::cout << "Faile to read cache index file:" << cache_idx_file << std::endl;
    return "";
  }
  for (auto &idx : cache_file_list) {
    idx.cache_file_name = cache_dir + "/" + idx.cache_file_name;
  }

  for (const auto &cache_index : cache_file_list) {
    if (cache_index.graph_key != graph_key) {
      std::cout << "Cache graph_key[" << cache_index.graph_key << "] error, expect = " << graph_key << std::endl;
      continue;
    }

    if (cache_index.cache_file_name.empty()) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is empty" << std::endl;
      return "";
    }
    return cache_index.cache_file_name;
  }
  return "";
}

class MockMmpaForOpenFailed : public MmpaStubApiGe {
 public:
  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return -1;
  }
};

class MockMmpaForFlockFailed : public MmpaStubApiGe {
 public:
  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return INT32_MAX;
  }
};

class MockExchangeService : public ExchangeService {
 public:
  Status CreateQueue(const int32_t device_id,
                     const string &name,
                     const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    return 0;
  }
  Status DestroyQueue(const int32_t device_id, const uint32_t queue_id) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id,
                 const uint32_t queue_id,
                 const void *const data,
                 const size_t size,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id,
                 const uint32_t queue_id,
                 const size_t size,
                 const FillFunc &fill_func,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) override {
    return 0;
  }
  Status Dequeue(const int32_t device_id,
                 const uint32_t queue_id,
                 void *const data,
                 const size_t size,
                 ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueTensor(const int32_t device_id,
                       const uint32_t queue_id,
                       GeTensor &tensor,
                       ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) override {
    return 0;
  }
  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
    return;
  }
};

class MockModelDeployer : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model,
                     DeployResult &deploy_result) override {
    return SUCCESS;
  }
  Status Undeploy(uint32_t model_id) override {
    return SUCCESS;
  }

  std::vector<int32_t> GetLocalNodeIndex() {
    return std::vector<int32_t>{0, 0};
  }
};
class MockExecutionRuntime : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  const std::string &GetCompileHostResourceType() const override{
    if (with_host_resource_) {
      return host_stub_;
    }
    return host_stub2_;
  }
  const std::map<std::string, std::string> &GetCompileDeviceInfo() const override{
    if (with_dev_resource_) {
      return logic_dev_id_to_res_type_;
    }
    return logic_dev_id_to_res_type2_;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

  void SetHostResFlag() { with_host_resource_ = true; }
  void SetDevResFlag() { with_dev_resource_ = true; }
  void ResetCompileResFlag() {
    with_host_resource_ = false;
    with_dev_resource_ = false;
  }
 private:
  MockModelDeployer model_deployer_;
  MockExchangeService exchange_service_;
  bool with_host_resource_ = false;
  bool with_dev_resource_ = false;
  std::string host_stub_ = "stub_host_type";
  std::map<std::string, std::string> logic_dev_id_to_res_type_ = {{"0:0:0", "stub_dev_type"},
                                                                  {"0:0:1", "stub_dev_type"}};
  std::string host_stub2_ = "";
  std::map<std::string, std::string> logic_dev_id_to_res_type2_ = {};
};
}  // namespace

class FlowModelCacheTest : public testing::Test {
 public:
  static void PrepareForCacheConfig(bool cache_manual_check, bool cache_debug_mode) {
    std::string cache_config_file = "./ut_cache_dir/cache.conf";
    {
      nlohmann::json cfg_json = {
                                  {"cache_manual_check", cache_manual_check},
                                  {"cache_debug_mode", cache_debug_mode}};
      std::ofstream json_file(cache_config_file);
      json_file << cfg_json << std::endl;
    }
  }

  static void RemoveCacheConfig() {
    remove("./ut_cache_dir/cache.conf");
  }

 protected:
  static void SetUpTestSuite() {
    origin_session_options_ = GetThreadLocalContext().GetAllSessionOptions();
    origin_graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
    ExecutionRuntime::SetExecutionRuntime(std::make_shared<MockExecutionRuntime>());
    std::string cmd = R"(
mkdir -p ./workspace/release/
cd ./workspace/release/
touch test1_release.om
touch test1_release.so
echo "Hello" > test1_release.om
echo "test1_release" > test1_release.so
tar -cvf test1_release.tar.gz test1_release.om test1_release.so
cd -
)";
  (void)system(cmd.c_str());
  }
  static void TearDownTestSuite() {
    GetThreadLocalContext().SetSessionOption(origin_session_options_);
    GetThreadLocalContext().SetGraphOption(origin_graph_options_);
    (void)system("rm -rf ./workspace");
  }
  void SetUp() {
    VarManager::Instance(0)->Init(0, 0, 0, 0);
    GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().SetGraphOption({});
    (void)system("mkdir ./ut_cache_dir");
  }

  void TearDown() {
    (void)system("rm -fr ./ut_cache_dir");
    VarManager::Instance(0)->Destory();
  }

  static void SetCacheDirOption(const std::string &cache_dir) {
    GetThreadLocalContext().SetSessionOption({{"ge.graph_compiler_cache_dir", cache_dir}});
  }
  static void ClearCacheDirOption() {
    GetThreadLocalContext().SetSessionOption({});
  }
  static void SetGraphKeyOption(const std::string &graph_key) {
    GetThreadLocalContext().SetGraphOption({{"ge.graph_key", graph_key}});
  }
  static void ClearGraphKeyOption() {
    GetThreadLocalContext().SetGraphOption({});
  }

 private:
  static std::map<std::string, std::string> origin_session_options_;
  static std::map<std::string, std::string> origin_graph_options_;
};
std::map<std::string, std::string> FlowModelCacheTest::origin_session_options_;
std::map<std::string, std::string> FlowModelCacheTest::origin_graph_options_;

TEST_F(FlowModelCacheTest, save_cache_no_need_cache) {
  FlowModelCache flow_model_cache;
  auto graph1 = FakeComputeGraph("graph1");
  auto ret = flow_model_cache.Init(graph1);
  EXPECT_EQ(ret, SUCCESS);

  FlowModelPtr flow_model;
  ret = flow_model_cache.TryCacheFlowModel(flow_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FlowModelCacheTest, save_cache_init_graph_no_need_cache) {
  FlowModelCache flow_model_cache;
  auto graph1 = FakeComputeGraph("graph1_init_by_20000");
  (void)AttrUtils::SetStr(graph1, "_suspend_graph_original_name", "graph1");
  auto ret = flow_model_cache.Init(graph1);
  EXPECT_EQ(ret, SUCCESS);

  FlowModelPtr flow_model;
  ret = flow_model_cache.TryCacheFlowModel(flow_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FlowModelCacheTest, save_cache_only_cache_dir_option) {
  SetCacheDirOption("./ut_cache_dir/");
  auto graph1 = FakeComputeGraph("graph1");
  FlowModelCache flow_model_cache;
  auto ret = flow_model_cache.Init(graph1);
  EXPECT_EQ(ret, SUCCESS);
  FlowModelPtr flow_model;
  ret = flow_model_cache.TryCacheFlowModel(flow_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FlowModelCacheTest, save_cache_only_graph_key_option) {
  SetGraphKeyOption("graph_key_1");
  FlowModelCache flow_model_cache;
  auto graph1 = FakeComputeGraph("graph1");
  auto ret = flow_model_cache.Init(graph1);
  EXPECT_EQ(ret, SUCCESS);
  FlowModelPtr flow_model;
  ret = flow_model_cache.TryCacheFlowModel(flow_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FlowModelCacheTest, save_cache_cache_dir_not_exit) {
  SetCacheDirOption("./ut_cache_dir_not_exit/");
  SetGraphKeyOption("graph_key_1");
  FlowModelCache flow_model_cache;
  auto graph1 = FakeComputeGraph("graph1");
  auto ret = flow_model_cache.Init(graph1);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(FlowModelCacheTest, save_and_load_ge_root_model) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_1");
  auto graph = FakeComputeGraph("test_graph_cache");
  auto ge_root_model = BuildPneModel("test_graph_cache", graph);
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->AddSubModel(ge_root_model, PNE_ID_NPU);
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_1", 1);
    EXPECT_EQ(check_ret, true);
  }

  {
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_EQ(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 1);
  }
}

TEST_F(FlowModelCacheTest, save_and_load_flow_model) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", graph);
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
  }
}

TEST_F(FlowModelCacheTest, save_and_load_with_subgraph) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model_with_sub_graph");
  auto sub_graph = FakeComputeGraph("sub_graph");
  auto root_graph = FakeGraphWithSubGraph("root_graph", sub_graph);
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", root_graph);
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(root_graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model_with_sub_graph", 1);
    EXPECT_EQ(check_ret, true);
    // save will not change orgin graph
    EXPECT_EQ(root_graph->GetAllSubgraphs().size(), 1);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    const auto &loaded_graph = load_flow_model->GetRootGraph();
    // flow model root graph not save subgraph
    EXPECT_EQ(loaded_graph->GetAllSubgraphs().size(), 0);
  }
}

TEST_F(FlowModelCacheTest, fake_load_flow_model_failed) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model_load_failed");
  auto sub_graph = FakeComputeGraph("sub_graph");
  auto root_graph = FakeGraphWithSubGraph("root_graph", sub_graph);
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", root_graph);
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(root_graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    std::string cache_file_name = GetCacheFileName("./ut_cache_dir", "graph_key_flow_model_load_failed");
    ModelData model;
    // Load model from file, default 0 priority.
    Status parse_ret = ModelParserBase::LoadFromFile(cache_file_name.c_str(), 0, model);
    GE_MAKE_GUARD(model_guard, [&model]() {
      if (model.model_data != nullptr) {
        delete[] static_cast<char *>(model.model_data);
        model.model_data = nullptr;
      }
    });
    ASSERT_EQ(parse_ret, SUCCESS);
    OmFileLoadHelper om_file_load_helper;
    auto ret = om_file_load_helper.Init(model);
    ASSERT_EQ(ret, SUCCESS);
    auto &model_partitions = om_file_load_helper.GetModelPartitions(0);
    uint8_t *data = const_cast<uint8_t *>(model_partitions[model_partitions.size() - 1].data);
    // modify sub model invalid
    data[0] = 0xFF;
    data[1] = 0xFF;
    FlowModelOmLoader loader;
    FlowModelPtr flow_model_load;
    Status load_ret = loader.LoadToFlowModel(model, flow_model_load);
    EXPECT_NE(load_ret, SUCCESS);
  }
}

TEST_F(FlowModelCacheTest, save_flow_init_graph_model) {
  std::string orig_name = "root_graph";
  auto graph = FakeComputeGraph("root_graph_init_by_graph");
  (void)AttrUtils::SetStr(graph, "_suspend_graph_original_name", orig_name);
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", graph);
  {
    SetCacheDirOption("./ut_cache_dir");
    SetGraphKeyOption("init_graph_key");
    FlowModelCache flow_model_cache;
    FlowModelCache::TryRecordSuspendGraph(orig_name);
  }
  {
    SetGraphKeyOption("other_graph_key");
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "init_graph_key", 1);
    EXPECT_EQ(check_ret, true);
  }
}

TEST_F(FlowModelCacheTest, save_and_load_flow_model_with_constant) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraphWithConstant("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithOutUdfV2("flow_model1", graph);

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraphWithConstant("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
  }
}


TEST_F(FlowModelCacheTest, save_and_load_flow_model_with_var) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraphWithVar("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithOutUdfV2("flow_model1", graph);

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraphWithVar("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
  }
}

TEST_F(FlowModelCacheTest, save_and_load_with_compile_resource) {
  auto *execute_runtime = ExecutionRuntime::GetInstance();
  EXPECT_NE(execute_runtime, nullptr);
  static_cast<MockExecutionRuntime*>(execute_runtime)->SetHostResFlag();
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", graph);
  for (auto &submodel_iter : flow_model->GetSubmodels()) {
    auto &submodel = submodel_iter.second;
    ASSERT_TRUE(submodel != nullptr);
    auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
    ASSERT_TRUE(deploy_resource_ptr != nullptr);
    ModelDeployResource &deploy_resource = *deploy_resource_ptr;
    deploy_resource.resource_list["Memory"] = 1024;
    deploy_resource.resource_list["CpuNum"] = 10;
    submodel->SetDeployResource(deploy_resource_ptr);
  }
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    static_cast<MockExecutionRuntime*>(execute_runtime)->SetDevResFlag();
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
 
    ASSERT_EQ(load_flow_model->GetCompileResource()->host_resource_type, "stub_host_type");
    ASSERT_EQ(load_flow_model->GetCompileResource()->logic_dev_id_to_res_type.size(), 1);
    ASSERT_EQ(load_flow_model->GetCompileResource()->logic_dev_id_to_res_type.count("0:0:0"), 0);
    ASSERT_EQ(load_flow_model->GetCompileResource()->logic_dev_id_to_res_type.count("0:0:1"), 1);
    ASSERT_EQ(load_flow_model->GetCompileResource()->logic_dev_id_to_res_type["0:0:1"], "stub_dev_type");
  }
  static_cast<MockExecutionRuntime*>(execute_runtime)->ResetCompileResFlag();
}

TEST_F(FlowModelCacheTest, save_and_load_flow_model_with_invalid_graph_key) {
  // cannot as file name.
  std::string graph_key = "can/not/be/file/name";
  std::string expect_hash_idx_name("hash_");
  auto hash_code = std::hash<std::string>{}(graph_key);
  expect_hash_idx_name.append(std::to_string(hash_code));

  GeTensorDesc tensor_desc(GeShape({16, 16}), FORMAT_ND, DT_INT16);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("some_var", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);

  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption(graph_key);
  auto graph = FakeComputeGraph("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", graph);
  FlowModelCache flow_model_cache;
  auto ret = flow_model_cache.Init(graph);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(FlowModelCacheTest, save_and_load_flow_model_with_udf) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model_with_udf1");

  GeTensorDesc tensor_desc(GeShape({16, 16}), FORMAT_ND, DT_INT16);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("some_var", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);

  auto graph = FakeComputeGraph("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithUdf("flow_model1", graph);
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ret = FlowModelHelper::SaveToOmModel(flow_model, "./ut_cache_dir/flow.om");
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model_with_udf1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
    uint32_t udf_model_num = 0;
    uint32_t ge_root_model_num = 0;
    for (const auto &pair : load_flow_model->GetSubmodels()) {
      const auto model = pair.second;
      if (model->GetModelType() == PNE_ID_UDF) {
        udf_model_num++;
        ASSERT_NE(model->GetDeployResource(), nullptr);
      } else {
        ge_root_model_num++;
      }
    }
    EXPECT_EQ(udf_model_num, 1);
    EXPECT_EQ(ge_root_model_num, 1);
  }
}


TEST_F(FlowModelCacheTest, save_and_load_flow_model_check_priority) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model_with_udf1");
  auto graph = FakeComputeGraph("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithUdf("flow_model1", graph);
  // fake priority
  std::map<std::string, std::map<std::string, int32_t>> models_esched_priority;

  for (const auto &submodel : flow_model->GetSubmodels()) {
    models_esched_priority[submodel.first] = {{ATTR_NAME_ESCHED_PROCESS_PRIORITY, 1},
                                              {ATTR_NAME_ESCHED_EVENT_PRIORITY, 2}};
  }
  flow_model->SetModelsEschedPriority(models_esched_priority);
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model_with_udf1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
    uint32_t udf_model_num = 0;
    uint32_t ge_root_model_num = 0;
    for (const auto &pair : load_flow_model->GetSubmodels()) {
      const auto model = pair.second;
      if (model->GetModelType() == PNE_ID_UDF) {
        udf_model_num++;
        ASSERT_NE(model->GetDeployResource(), nullptr);
      } else {
        ge_root_model_num++;
      }
    }
    EXPECT_EQ(udf_model_num, 1);
    EXPECT_EQ(ge_root_model_num, 1);
    EXPECT_EQ(load_flow_model->GetModelsEschedPriority(), models_esched_priority);
  }
}
  
TEST_F(FlowModelCacheTest, save_and_load_flow_model_compile_and_deploy) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  GELOGD("flow_model_cache after init");

  constexpr const char *redundant_logic_device_id = "0:0:2:0";
  FlowModelPtr flow_model = BuildFlowModelWithOutP2pNode("flow_model1", graph);
  flow_model->SetRedundantLogicDeviceId(redundant_logic_device_id);
  GELOGD("flow_model_cache after BuildFlowModelWithoutUdf");

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    GELOGD("flow_model_cache after TryCacheModel");
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);

    // load with no index file
    (void)system("rm -fr ./ut_cache_dir/graph_key_flow_model1.idx");
    FlowModelPtr load_flow_model2;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model2);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model2, nullptr);
    EXPECT_NE(load_flow_model2->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model2->GetSubmodels().size(), 2);
    EXPECT_EQ(load_flow_model2->GetLogicDeviceId(), "");
    EXPECT_EQ(load_flow_model2->GetRedundantLogicDeviceId(), "");
    for (const auto &submodel:load_flow_model2->GetSubmodels()) {
      EXPECT_EQ(submodel.second->GetRedundantLogicDeviceId(), redundant_logic_device_id);
    }
  }
}

TEST_F(FlowModelCacheTest, load_and_save_flow_model) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");

  GeTensorDesc tensor_desc(GeShape({16, 16}), FORMAT_ND, DT_INT16);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("some_var", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);
 
  auto fake_graph = FakeComputeGraph("test_load_graph");
  AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
  FlowModelPtr load_flow_model;
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    // not match
    ret = flow_model_cache.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(load_flow_model, nullptr);
    
    // mock var fusion
    GeTensorDesc tensor_desc1(GeShape({16, 16}), FORMAT_NCHW, DT_INT16);
    EXPECT_EQ(VarManager::Instance(0)->RecordStagedVarDesc(0, "some_var", tensor_desc1), SUCCESS);
    EXPECT_EQ(VarManager::Instance(0)->SetChangedGraphId("some_var", 0), SUCCESS);
    TransNodeInfo trans_node_info;
    VarTransRoad fusion_road;
    fusion_road.emplace_back(trans_node_info);
    EXPECT_EQ(VarManager::Instance(0)->SetTransRoad("some_var", fusion_road), SUCCESS);
    
    GeTensorDesc tensor_desc2(GeShape({16, 16}), FORMAT_NHWC, DT_INT16);
    EXPECT_EQ(VarManager::Instance(0)->RenewCurVarDesc("some_var", tensor_desc2), SUCCESS);
    
    auto graph = FakeComputeGraph("root_graph");
    FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", graph);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph);
    EXPECT_EQ(ret, SUCCESS);
    
    // resume var
    EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("some_var", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);
    // match cache
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
  }
}

TEST_F(FlowModelCacheTest, save_and_load_sub_flow_model) {
  auto graph = FakeComputeGraph("graph1");
  graph->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
  graph->SetExtAttr<std::string>("_udf_om_file_for_data_flow_cache", graph->GetName() + "/" + "test1_release.om");
  FlowModelPtr flow_model = BuildFlowModelWithUdf("flow_model1", graph);
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.InitSubmodelCache(graph, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = mmAccess("./ut_cache_dir/graph_key/graph1/graph1.om");
    EXPECT_EQ(check_ret, 0);
    
    auto graph2 = FakeComputeGraph("graph2/test");
    graph2->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    AttrUtils::SetStr(*graph2, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
    ret = flow_model_cache.InitSubmodelCache(graph2, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    check_ret = mmAccess("./ut_cache_dir/graph_key/graph22ftest/buildinfo");
    EXPECT_EQ(check_ret, 0);

    auto graph3 = FakeComputeGraph("graph3");
    graph3->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    graph3->SetExtAttr<std::string>("_udf_om_file_for_data_flow_cache", "invalid.om");
    AttrUtils::SetStr(*graph2, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
    ret = flow_model_cache.InitSubmodelCache(graph3, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    check_ret = mmAccess("./ut_cache_dir/graph_key/graph3/buildinfo");
    EXPECT_EQ(check_ret, 0);
  }
  {
    auto fake_graph = FakeComputeGraph("graph1");
    fake_graph->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
    
    auto fake_graph2 = FakeComputeGraph("graph2/test");
    fake_graph2->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    AttrUtils::SetStr(*fake_graph2, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
    AttrUtils::SetStr(*fake_graph2, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph2, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph2, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);

    auto fake_graph3 = FakeComputeGraph("graph3");
    fake_graph3->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    AttrUtils::SetStr(*fake_graph3, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
    AttrUtils::SetStr(*fake_graph3, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph3, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph3, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);
  }
}

TEST_F(FlowModelCacheTest, save_and_load_sub_flow_model_manual_check) {
  auto graph = FakeComputeGraph("graph1");
  graph->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
  FlowModelPtr flow_model = BuildFlowModelWithUdf("flow_model1", graph);
  PrepareForCacheConfig(true, true);
  GE_MAKE_GUARD(recover, []() { RemoveCacheConfig(); });
  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.InitSubmodelCache(graph, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = mmAccess("./ut_cache_dir/graph_key/graph1/graph1.om");
    EXPECT_EQ(check_ret, 0);

    auto graph2 = FakeComputeGraph("graph2");
    graph2->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    // set compile result to graph
    ge::NamedAttrs current_compile_result;
    ge::NamedAttrs running_res_info;
    running_res_info.SetAttr("Ascend", GeAttrValue::CreateFrom<int64_t>(1));
    ge::AttrUtils::SetNamedAttrs(current_compile_result, "_dflow_running_resource_info", running_res_info);
    ge::NamedAttrs runnable_resources_info;
    runnable_resources_info.SetAttr("Ascend", GeAttrValue::CreateFrom<std::string>("stub_path"));
    ge::AttrUtils::SetNamedAttrs(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
    ge::AttrUtils::SetNamedAttrs(graph2, "_dflow_compiler_result", current_compile_result);

    AttrUtils::SetStr(*graph2, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
    ret = flow_model_cache.InitSubmodelCache(graph2, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
    check_ret = mmAccess("./ut_cache_dir/graph_key/graph2/buildinfo");
    EXPECT_EQ(check_ret, 0);
  }
  {
    auto fake_graph = FakeComputeGraph("graph1");
    fake_graph->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    FlowModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    FlowModelPtr load_flow_model;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model, nullptr);
    EXPECT_NE(load_flow_model->GetModelRelation(), nullptr);
    ASSERT_EQ(load_flow_model->GetSubmodels().size(), 2);

    (void)system("rm -fr ./ut_cache_dir/graph_key/graph1/buildinfo");
    // manual mode, no need to check buildinfo
    FlowModelPtr load_flow_model2;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph, load_flow_model2);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_flow_model2, nullptr);
    auto fake_graph2 = FakeComputeGraph("graph2");
    fake_graph2->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
    AttrUtils::SetStr(*fake_graph2, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
    AttrUtils::SetStr(*fake_graph2, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    system("mkdir -p ./ut_cache_dir/graph_key/graph2 ;touch ./ut_cache_dir/graph_key/graph2/graph2_release.om");
    auto check_ret = mmAccess("./ut_cache_dir/graph_key/graph2/graph2_release.om");
    EXPECT_EQ(check_ret, 0);
    ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph2, "./ut_cache_dir", "graph_key");
    EXPECT_EQ(ret, SUCCESS);
    load_flow_model = nullptr;
    ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph2, load_flow_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_EQ(load_flow_model, nullptr);
    CacheCompileResult compile_result;
    auto matched = flow_model_cache_for_load.TryLoadCompileResultFromCache(compile_result);
    ASSERT_EQ(matched, true);

    {
      std::ofstream buildinfo("./ut_cache_dir/graph_key/graph2/buildinfo");
      buildinfo << "{\"bin_info\":1}\n";
    }
    matched = flow_model_cache_for_load.TryLoadCompileResultFromCache(compile_result);
    ASSERT_EQ(matched, false);
  }
}

TEST_F(FlowModelCacheTest, save_and_load_sub_flow_model_no_need) {
  auto graph = FakeComputeGraph("graph1");
  graph->SetExtAttr<bool>("_is_built_in_for_data_flow", true);
  FlowModelCache flow_model_cache;
  auto ret = flow_model_cache.InitSubmodelCache(graph, "./ut_cache_dir", "graph_key");
  EXPECT_EQ(ret, SUCCESS);

  auto graph_no_cache = FakeComputeGraph("graph_no_cache");
  ret = flow_model_cache.InitSubmodelCache(graph_no_cache, "./ut_cache_dir", "graph_key");
  EXPECT_EQ(ret, SUCCESS);
  FlowModelPtr load_flow_model;
  ret = flow_model_cache.TryLoadFlowModelFromCache(graph_no_cache, load_flow_model);
  EXPECT_EQ(ret, SUCCESS);
  FlowModelPtr flow_model = BuildFlowModelWithUdf("flow_model1", graph_no_cache);
  ret = flow_model_cache.TryCacheFlowModel(flow_model);
  EXPECT_EQ(ret, SUCCESS);
  auto check_ret = mmAccess("./ut_cache_dir/graph_key/graph_no_cache/buildinfo");
  EXPECT_NE(check_ret, 0);
  AttrUtils::SetStr(*graph_no_cache, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
  ret = flow_model_cache.InitSubmodelCache(graph_no_cache, "./ut_cache_dir", "graph_key");
  EXPECT_EQ(ret, SUCCESS);
  ret = flow_model_cache.TryLoadFlowModelFromCache(graph_no_cache, load_flow_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FlowModelCacheTest, save_and_load_sub_flow_model_failed) {
  auto fake_graph2 = FakeComputeGraph("graph2");
  fake_graph2->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
  AttrUtils::SetStr(*fake_graph2, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_NPU);
  AttrUtils::SetStr(*fake_graph2, ATTR_NAME_SESSION_GRAPH_ID, "100_99");

  FlowModelCache flow_model_cache_for_load;
  auto ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph2, "./ut_cache_dir", "graph_key");
  EXPECT_EQ(ret, SUCCESS);
  FlowModelPtr flow_model = BuildFlowModelWithUdf("flow_model1", fake_graph2);
  ret = flow_model_cache_for_load.TryCacheFlowModel(flow_model);
  EXPECT_EQ(ret, SUCCESS);
  auto check_ret = mmAccess("./ut_cache_dir/graph_key/graph2/buildinfo");
  EXPECT_EQ(check_ret, 0);
  fake_graph2->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info_new");
  FlowModelPtr load_flow_model;
  ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph2, load_flow_model);
  EXPECT_EQ(ret, SUCCESS);
  fake_graph2->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
  {
    std::ofstream buildinfo("./ut_cache_dir/graph_key/graph2/buildinfo");
    buildinfo << "error\n";
  }
  ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph2, load_flow_model);
  EXPECT_EQ(ret, FAILED);
  auto fake_graph3 = FakeComputeGraph("graph3");
  AttrUtils::SetStr(*fake_graph3, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_NPU);
  ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph3, "./ut_cache_dir", "graph_key");
  EXPECT_EQ(ret, SUCCESS);
  fake_graph3->SetExtAttr<std::string>("_graph_info_for_data_flow_cache", "graph_info");
  ret = flow_model_cache_for_load.TryCacheFlowModel(flow_model);
  EXPECT_EQ(ret, SUCCESS);
  {
    std::ofstream buildinfo("./ut_cache_dir/graph_key/graph3/buildinfo");
    buildinfo << "{\"no_graph_info\":1}\n";
  }
  FlowModelPtr load_flow_model2;
  ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph3, load_flow_model2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(load_flow_model2, nullptr);
  AttrUtils::SetStr(*fake_graph3, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF);
  ret = flow_model_cache_for_load.InitSubmodelCache(fake_graph3, "./ut_cache_dir", "graph_key");
  EXPECT_EQ(ret, SUCCESS);
  ret = flow_model_cache_for_load.TryLoadFlowModelFromCache(fake_graph3, load_flow_model);
  EXPECT_EQ(ret, SUCCESS);
  auto cache_release_info =
    fake_graph3->TryGetExtAttr<std::string>("_cache_graph_info_for_data_flow_cache", "");
  EXPECT_EQ(cache_release_info, "");
}

TEST_F(FlowModelCacheTest, save_flow_model_open_lock_file_failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForOpenFailed>());
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", graph);

  FlowModelCache flow_model_cache;
  auto ret = flow_model_cache.Init(graph);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(FlowModelCacheTest, save_flow_model_open_lock_failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForFlockFailed>());
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  FlowModelPtr flow_model = BuildFlowModelWithoutUdf("flow_model1", graph);
  FlowModelCache flow_model_cache;
  auto ret = flow_model_cache.Init(graph);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}
}  // namespace ge
