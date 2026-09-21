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
#include "ge_graph_dsl/graph_dsl.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/build/graph_builder.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/helper/file_saver.h"
#include "api/gelib/gelib.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "register/ops_kernel_builder_registry.h"
#include "graph/build/model_data_info.h"
#include "graph/manager/graph_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "framework/common/helper/model_helper.h"
#include "framework/memory/memory_api.h"
#include "stub/gert_runtime_stub.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
const char *const kKernelLibName = "ops_kernel_lib";
const char *kSessionId = "123456";
bool need_sk_append_ws = false;
}  // namespace

class GraphBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    InitGeLib();
  }
  void TearDown() override {
    FinalizeGeLib();
  }
  class FakeOpsKernelInfoStore : public OpsKernelInfoStore {
   public:
    FakeOpsKernelInfoStore() = default;

   private:
    Status Initialize(const std::map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
      return true;
    };
    void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override{};
  };

  class FakeOpsKernelBuilder : public OpsKernelBuilder {
   public:
    FakeOpsKernelBuilder() = default;

   private:
    Status Initialize(const map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    Status CalcOpRunningParam(Node &node) override {
      auto node_op_desc = node.GetOpDesc();
      GE_CHECK_NOTNULL(node_op_desc);
      uint32_t index = 0;
      for (const auto &output_desc_ptr : node_op_desc->GetAllOutputsDescPtr()) {
        GeTensorDesc &desc_temp = *output_desc_ptr;

        uint32_t dim_num = static_cast<uint32_t>(desc_temp.GetShape().GetDimNum());
        GE_IF_BOOL_EXEC(dim_num > DIM_DEFAULT_SIZE, TensorUtils::SetRealDimCnt(desc_temp, dim_num));
        // calculate tensor size
        int64_t size_temp = 0;
        graphStatus graph_status = TensorUtils::GetTensorMemorySizeInBytes(desc_temp, size_temp);
        if (graph_status != GRAPH_SUCCESS) {
          return FAILED;
        }
        TensorUtils::SetSize(desc_temp, size_temp);
        if (node_op_desc->UpdateOutputDesc(index, desc_temp) != SUCCESS) {
          return FAILED;
        }
        index++;
      }
      return SUCCESS;
    };
    Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
      domi::TaskDef task_def;
      tasks.push_back(task_def);
      AttrUtils::SetListInt(node.GetOpDesc(), "_test_task_generate_input_offset", node.GetOpDesc()->GetInputOffset());
      AttrUtils::SetListInt(node.GetOpDesc(), "_test_task_generate_output_offset", node.GetOpDesc()->GetOutputOffset());
      AttrUtils::SetInt(node.GetOpDesc(), "_test_get_run_context_memsize", context.dataMemSize);
      if ((node.GetType() == "SuperKernel") && (need_sk_append_ws)) {
        std::vector<int64_t> append_ws_vec;
        append_ws_vec.emplace_back(100);
        AttrUtils::SetListInt(node.GetOpDesc(), "_append_ws", append_ws_vec);
      }
      return SUCCESS;
    };
  };

  class FakeDNNEngine : public DNNEngine {
   public:
    FakeDNNEngine() = default;
    ~FakeDNNEngine() override = default;
  };

  class AICoreDNNEngine : public DNNEngine {
  public:
      explicit AICoreDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
      ~AICoreDNNEngine() override = default;
  };

  class AICpuDNNEngine : public DNNEngine {
  public:
      explicit AICpuDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
      ~AICpuDNNEngine() override = default;
  };

  class GeLocalDNNEngine : public DNNEngine {
  public:
      explicit GeLocalDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
      ~GeLocalDNNEngine() override = default;
  };

  class HcclDNNEngine : public DNNEngine {
  public:
      explicit HcclDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
      ~HcclDNNEngine() override = default;
  };

  class FakeGraphOptimizer : public GraphOptimizer {
   public:
    FakeGraphOptimizer() = default;
    Status Initialize(const std::map<std::string, std::string> &options,
                      OptimizeUtility *const optimize_utility) override {
      return SUCCESS;
    }
    Status Finalize() override {
      return SUCCESS;
    }
    Status OptimizeOriginalGraph(ComputeGraph &graph) override {
      return SUCCESS;
    }
    Status OptimizeFusedGraph(ComputeGraph &graph) override {
      return SUCCESS;
    }
    Status OptimizeWholeGraph(ComputeGraph &graph) override {
      return SUCCESS;
    }
    Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
      return SUCCESS;
    }
  };

  void InitGeLib() {
    map<string, string> options;
    Status ret = ge::GELib::Initialize(options);
    EXPECT_EQ(ret, SUCCESS);
    auto instance_ptr = ge::GELib::GetInstance();
    EXPECT_NE(instance_ptr, nullptr);

    //  SchedulerConf conf;
    SchedulerConf scheduler_conf;
    scheduler_conf.name = kKernelLibName;
    scheduler_conf.cal_engines[kKernelLibName] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines[kKernelLibName]->name = kKernelLibName;
    scheduler_conf.cal_engines[kKernelLibName]->scheduler_id = kKernelLibName;

    scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->name = "DNN_VM_GE_LOCAL";
    scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->id = "DNN_VM_GE_LOCAL";
    scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->independent = false;
    scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->attach = true;
    scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->skip_assign_stream = true;

    scheduler_conf.cal_engines["AIcoreEngine"] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines["AIcoreEngine"]->name = "AIcoreEngine";
    scheduler_conf.cal_engines["AIcoreEngine"]->id = "AIcoreEngine";
    scheduler_conf.cal_engines["AIcoreEngine"]->independent = false;
    scheduler_conf.cal_engines["AIcoreEngine"]->attach = false;
    scheduler_conf.cal_engines["AIcoreEngine"]->skip_assign_stream = false;

    scheduler_conf.cal_engines["DNN_VM_RTS"] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines["DNN_VM_RTS"]->name = "DNN_VM_RTS";
    scheduler_conf.cal_engines["DNN_VM_RTS"]->id = "DNN_VM_RTS";
    scheduler_conf.cal_engines["DNN_VM_RTS"]->independent = false;
    scheduler_conf.cal_engines["DNN_VM_RTS"]->attach = true;
    scheduler_conf.cal_engines["DNN_VM_RTS"]->skip_assign_stream = false;

    instance_ptr->DNNEngineManagerObj().schedulers_[kKernelLibName] = scheduler_conf;

    DNNEngineAttribute attr_aicore = { "AIcoreEngine", {}, PriorityEnum::COST_0, DEVICE, FORMAT_RESERVED,
                                       FORMAT_RESERVED, true };
    auto aicore_engine = std::make_shared<AICoreDNNEngine>(attr_aicore);
    instance_ptr->DNNEngineManagerObj().engines_map_["AIcoreEngine"] = aicore_engine;

    DNNEngineAttribute attr_aicpu = { "DNN_VM_AICPU", {}, PriorityEnum::COST_2, DEVICE, FORMAT_RESERVED,
                                      FORMAT_RESERVED, true };
    auto aicpu_engine = std::make_shared<AICpuDNNEngine>(attr_aicpu);
    instance_ptr->DNNEngineManagerObj().engines_map_["DNN_VM_AICPU"] = aicpu_engine;

    DNNEngineAttribute attr_hccl = { "hccl", {}, PriorityEnum::COST_1, DEVICE, FORMAT_RESERVED, FORMAT_RESERVED, true };
    auto hccl_engine = std::make_shared<HcclDNNEngine>(attr_hccl);
    instance_ptr->DNNEngineManagerObj().engines_map_["hccl"] = hccl_engine;

    DNNEngineAttribute attr_ge_local = { "DNN_VM_GE_LOCAL", {}, PriorityEnum::COST_9, DEVICE, FORMAT_RESERVED,
                                         FORMAT_RESERVED, true };
    auto ge_local_engine = std::make_shared<GeLocalDNNEngine>(attr_ge_local);
    instance_ptr->DNNEngineManagerObj().engines_map_["DNN_VM_GE_LOCAL"] = ge_local_engine;

    auto engine = std::make_shared<FakeDNNEngine>();
    instance_ptr->DNNEngineManagerObj().engines_map_[kKernelLibName] = engine;

    OpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FakeOpsKernelInfoStore>();
    OpsKernelManager::GetInstance().ops_kernel_store_.emplace(kKernelLibName, ops_kernel_info_store_ptr);
    std::vector<OpInfo> op_infos;
    OpInfo op_info = {kKernelLibName, kKernelLibName, 1, false, false, false, "", ""};
    op_infos.emplace_back(op_info);
    std::vector<std::string> stub_optyes{SEND, RECV, "SuperKernel"};
    for (const auto &op_type : stub_optyes) {
      OpsKernelManager::GetInstance().ops_kernel_info_.emplace(op_type, op_infos);
    }
    auto graph_optimizer = std::make_shared<FakeGraphOptimizer>();
    OpsKernelManager::GetInstance().graph_optimizers_.emplace(kKernelLibName, graph_optimizer);
    OpsKernelBuilderPtr fake_builder = std::make_shared<FakeOpsKernelBuilder>();
    OpsKernelBuilderRegistry::GetInstance().kernel_builders_[kKernelLibName] = fake_builder;
    OpsKernelBuilderRegistry::GetInstance().kernel_builders_["ops_kernel_info_hccl"] = fake_builder;
    OpsKernelBuilderRegistry::GetInstance().kernel_builders_["AIcoreEngine"] = fake_builder;
  }

  void FinalizeGeLib() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
  }

  NodePtr AddNode(ut::GraphBuilder &builder, const string &name, const string &type, int in_cnt, int out_cnt) {
    auto node = builder.AddNode(name, type, in_cnt, out_cnt);
    auto op_desc = node->GetOpDesc();
    op_desc->SetInputOffset(std::vector<int64_t>(in_cnt));
    op_desc->SetOutputOffset(std::vector<int64_t>(out_cnt));
    return node;
  }

  void SetSubGraph(const NodePtr &node, const string &name, bool is_dynamic_subgraph = false,
                   bool has_memory_type = false) {
    auto op_desc = node->GetOpDesc();
    auto builder = ut::GraphBuilder(name);

    int num_inputs = static_cast<int>(op_desc->GetInputsSize());
    int num_outputs = static_cast<int>(op_desc->GetOutputsSize());
    auto some_node = AddNode(builder, name + "_node", ADDN, num_inputs, 1);
    some_node->GetOpDesc()->SetOpEngineName("AIcoreEngine");
    if (has_memory_type) {  // kSubgraphData
      (void)AttrUtils::SetInt(some_node->GetOpDesc(), ATTR_INPUT_MEMORY_TYPE, 0);
    }
    for (int i = 0; i < num_inputs; ++i) {
      auto data_node = AddNode(builder, name + "_data_" + std::to_string(i), DATA, 1, 1);
      data_node->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
      AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_INDEX, i);
      AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, i);
      builder.AddDataEdge(data_node, 0, some_node, i);
    }
    auto net_output = AddNode(builder, "Node_Output", NETOUTPUT, num_outputs, num_outputs);
    net_output->GetOpDesc()->SetOpEngineName("DNN_VM_RTS");
    for (int i = 0; i < num_outputs; ++i) {
      builder.AddDataEdge(some_node, 0, net_output, i);
      AttrUtils::SetInt(net_output->GetOpDesc()->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, i);
    }

    auto subgraph = builder.GetGraph();
    AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
    subgraph->SetGraphUnknownFlag(is_dynamic_subgraph);
    subgraph->SetParentNode(node);
    op_desc->AddSubgraphName(name);
    op_desc->SetSubgraphInstanceName(0, name);
    auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
    subgraph->SetParentGraph(root_graph);
    root_graph->AddSubgraph(name, subgraph);
  }

  static ComputeGraphPtr BuildSubComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
    auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
    auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    return graph;
  }

  ComputeGraphPtr BuildComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto data = builder.AddNode("Data", "Data", 1, 1);
    auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
    transdata->GetOpDesc()->AddSubgraphName("subgraph");
    transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, transdata, 0);
    builder.AddDataEdge(transdata, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    // add subgraph
    transdata->SetOwnerComputeGraph(graph);
    ComputeGraphPtr subgraph = BuildSubComputeGraph();
    subgraph->SetParentGraph(graph);
    subgraph->SetParentNode(transdata);
    graph->AddSubgraph("subgraph", subgraph);
    auto op_desc = netoutput->GetOpDesc();
    std::vector<std::string> src_name{"out"};
    op_desc->SetSrcName(src_name);
    std::vector<int64_t> src_index{0};
    op_desc->SetSrcIndex(src_index);
    return graph;
  }

  void GenerateOmFile(std::string &om_path) {
    const ComputeGraphPtr graph = BuildComputeGraph();
    const GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(graph);
    const auto model_task_def = MakeShared<domi::ModelTaskDef>();
    model_task_def->add_task()->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_END_GRAPH));
    ge_model->SetModelTaskDef(model_task_def);

    TBEKernelStore tbe_kernel_store;
    const auto kernel = MakeShared<OpKernelBin>("hello", std::vector<char>(64, 0));
    tbe_kernel_store.AddTBEKernel(kernel);
    tbe_kernel_store.Build();
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    ModelBufferData model;
    ModelHelper model_helper;
    const std::string path = "./ut_test_model_for_skip.om";
    model_helper.SaveToOmModel(ge_model, "ut_test_model_for_skip.om", model);
    char resolved_path[2000];
    realpath(path.c_str(), resolved_path);
    om_path = resolved_path;
  }

  ComputeGraphPtr BuildCompoundGraph(bool is_dynamic_subgraph = false, bool has_memory_type = false,
                                     bool skip_gen_task = false, bool with_om_path = false) {
    auto builder = ut::GraphBuilder("g1");
    auto data1 = AddNode(builder, "data1", DATA, 1, 1);
    data1->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    if (has_memory_type) {  // kOthers
      (void)AttrUtils::SetInt(data1->GetOpDesc(), ATTR_INPUT_MEMORY_TYPE, 0);
    }
    auto data2 = AddNode(builder, "data2", DATA, 1, 1);
    data2->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
    auto partitioned_call_1 = AddNode(builder, "PartitionedCall1", PARTITIONEDCALL, 1, 1);
    auto partitioned_call_2 = AddNode(builder, "PartitionedCall2", PARTITIONEDCALL, 1, 1);
    auto partitioned_call_3 = AddNode(builder, "PartitionedCall3", PARTITIONEDCALL, 2, 1);
    auto net_output = AddNode(builder, "NetOutput", NETOUTPUT, 1, 1);
    partitioned_call_1->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
    partitioned_call_2->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
    partitioned_call_3->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
    net_output->GetOpDesc()->SetOpEngineName("DNN_VM_RTS");
    if (has_memory_type) {  // kSubgraphNode
      (void)AttrUtils::SetInt(net_output->GetOpDesc(), ATTR_INPUT_MEMORY_TYPE, 0);
    }
    SetSubGraph(partitioned_call_1, "subgraph-1", is_dynamic_subgraph, has_memory_type);
    if (skip_gen_task) {
      (void)AttrUtils::SetBool(partitioned_call_1->GetOpDesc(), ATTR_NAME_SKIP_GEN_TASK, true);
    }
    if (with_om_path) {
      std::string om_path;
      GenerateOmFile(om_path);
      (void)AttrUtils::SetStr(partitioned_call_1->GetOpDesc(), ATTR_NAME_OM_BINARY_PATH, om_path);
    }
    SetSubGraph(partitioned_call_2, "subgraph-2");
    SetSubGraph(partitioned_call_3, "subgraph-3");

    builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
    builder.AddDataEdge(data2, 0, partitioned_call_2, 0);
    builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_3, 0);
    builder.AddDataEdge(partitioned_call_2, 0, partitioned_call_3, 1);
    builder.AddDataEdge(partitioned_call_3, 0, net_output, 0);

    auto root_graph = builder.GetGraph();
    for (const auto &node : root_graph->GetAllNodes()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    }

    AttrUtils::SetBool(root_graph, "is_compound", true);
    return root_graph;
  }

  ComputeGraphPtr BuildDynamicShapeGraph(bool has_memory_type = false) {
    auto root_graph = BuildCompoundGraph(true, has_memory_type);
    AttrUtils::SetBool(root_graph, "is_compound", false);
    AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
    return root_graph;
  }

  ComputeGraphPtr BuildDynamicShapeGraphWithSkipGenTask(bool skip_gen_task = true, bool with_om_path = false) {
    auto root_graph = BuildCompoundGraph(true, false, skip_gen_task, with_om_path);
    AttrUtils::SetBool(root_graph, "is_compound", false);
    AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
    return root_graph;
  }

  ComputeGraphPtr BuildDynamicShapeGraphWithMemoryType() {
    auto root_graph = BuildDynamicShapeGraph(true);
    return root_graph;
  }

  ComputeGraphPtr BuildStaticShapeGraphWithMemoryType() {
    auto root_graph = BuildCompoundGraph(false, true);
    AttrUtils::SetBool(root_graph, "is_compound", false);
    return root_graph;
  }

  ComputeGraphPtr BuildGraphWithConst(bool with_weight = true) {
    DEF_GRAPH(g1) {
      CHAIN(NODE("const1", CONSTANT)->NODE("relu1", RELU)->NODE("netoutput", NETOUTPUT));
    };
    auto root_graph = ToComputeGraph(g1);
    // somehow, GeRunningEnvFaker does not work well
    for (const auto &node : root_graph->GetAllNodes()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
    }
    if (with_weight) {
      auto const_node = root_graph->FindNode("const1");
      EXPECT_NE(const_node, nullptr);
      float_t weight[1 * 1 * 224 * 224] = {0};
      GeTensorDesc weight_desc(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_FLOAT);
      GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
      OpDescUtils::SetWeights(const_node, {tensor});
    }
    return root_graph;
  }

  ComputeGraphPtr BuildGraphWithLabels() {
    DEF_GRAPH(g1) {
      auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_STREAM_LABEL, "0");
      auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_STREAM_LABEL, "1");
      auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_STREAM_LABEL, "2");
      auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_STREAM_LABEL, "3");
      auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_STREAM_LABEL, "4");
      auto add_0 = OP_CFG(ADD).Attr(ATTR_NAME_STREAM_LABEL, "5");
      CHAIN(NODE("_arg_0", data_0)
                ->EDGE(0, 0)
                ->NODE("Conv2D", conv_0)
                ->EDGE(0, 0)
                ->NODE("Relu", relu_0)
                ->EDGE(0, 0)
                ->NODE("Add", add_0)
                ->EDGE(0, 0)
                ->NODE("Node_Output", NETOUTPUT));
      CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("Conv2D", conv_0));
      CHAIN(NODE("_arg_2", data_2)->EDGE(0, 1)->NODE("Add", add_0));
    };
    auto root_graph = ToComputeGraph(g1);
    // somehow, GeRunningEnvFaker does not work well
    for (const auto &node : root_graph->GetAllNodes()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
    }
    (void)ge::AttrUtils::SetInt(*root_graph, ATTR_MODEL_LABEL_NUM, 6);
    return root_graph;
  }
};

TEST_F(GraphBuilderTest, CalcOpParam_fixed) {
  GraphBuilder graph_builder;
  auto root_graph = BuildGraphWithLabels();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  InitGeLib();
  for (const auto &node : root_graph->GetAllNodes()) {
    if (strcmp(node->GetOpDesc()->GetNamePtr(), "Conv2D") == 0) {
      node->GetOpDesc()->SetOpKernelLibName("ops_kernel_info_hccl");
      //node->GetOpDesc()->SetOpEngineName("ops_kernel_info_hccl");
    }
  }
  auto ret = graph_builder.CalcOpParam(root_graph);
  EXPECT_EQ(ret, SUCCESS);
  FinalizeGeLib();
}

TEST_F(GraphBuilderTest, CalcOpParam_failed) {
  GraphBuilder graph_builder;
  auto root_graph = BuildGraphWithLabels();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  FinalizeGeLib();
  auto ret = graph_builder.CalcOpParam(root_graph);
  EXPECT_NE(ret, SUCCESS);
  InitGeLib();
  for (const auto &node : root_graph->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName("");
    node->GetOpDesc()->SetOpEngineName("");
  }
  ret = graph_builder.CalcOpParam(root_graph);
  EXPECT_NE(ret, SUCCESS);
  FinalizeGeLib();
}

TEST_F(GraphBuilderTest, CalcLogIdAndSetAttr_success) {
  GraphBuilder graph_builder;
  auto root_graph = BuildGraphWithLabels();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  std::vector<int64_t> trace_nodes{0};
  for (const auto &node : root_graph->GetAllNodes()) {
    auto ret = graph_builder.CalcLogIdAndSetAttr(node->GetOpDesc(), trace_nodes, 0, 10);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(GraphBuilderTest, SetConstantInputOffset) {
  GraphBuilder graph_builder;
  auto root_graph_with_weights = BuildGraphWithConst();
  AttrUtils::SetStr(root_graph_with_weights, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  auto ret = graph_builder.SetConstantInputOffset(root_graph_with_weights);
  EXPECT_EQ(ret, SUCCESS);
  auto root_graph_without_weights = BuildGraphWithConst(false);
  AttrUtils::SetStr(root_graph_without_weights, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  ret = graph_builder.SetConstantInputOffset(root_graph_without_weights);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, CheckConstOffset_Success_WithSrcConstName) {
  InitGeLib();
  GraphBuilder graph_builder;
  auto root_graph = BuildDynamicShapeGraphWithMemoryType();
  NodePtr add_node = nullptr;
  for (const auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == "subgraph-1_node") {
      add_node = node;
      break;
    }
  }
  ASSERT_NE(add_node, nullptr);
  add_node->GetOpDesc()->SetType("Constant");
  GeTensor weight;
  std::vector<int32_t> data = {8, 4, 4, 0};
  weight.SetData((uint8_t *)data.data(), data.size() * sizeof(int32_t));
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({4}));
  weight_desc.SetOriginShape(GeShape({4}));
  weight.SetTensorDesc(weight_desc);
  AttrUtils::SetTensor(add_node->GetOpDesc(), "value", weight);

  AttrUtils::SetStr(add_node->GetOpDesc(), ATTR_NAME_SRC_CONST_NAME, "subgraph-1_node0");
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  GeRootModelPtr root_model;
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);

  const auto output_offset = add_node->GetOpDesc()->GetOutputOffset();
  ASSERT_EQ(output_offset.size(), 1);
  EXPECT_NE(output_offset[0], 0);
}

TEST_F(GraphBuilderTest, test_build_for_graph_with_label) {
  GraphBuilder graph_builder;
  auto root_graph = BuildGraphWithLabels();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, test_build_for_static_shape_graph_with_const) {
  GraphBuilder graph_builder;
  auto root_graph = BuildGraphWithConst();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, test_build_for_static_shape_graph_with_memory_type) {
  GraphBuilder graph_builder;
  auto root_graph = BuildStaticShapeGraphWithMemoryType();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, test_build_for_dynamic_shape_graph_with_memory_type) {
  GraphBuilder graph_builder;
  auto root_graph = BuildDynamicShapeGraphWithMemoryType();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, test_build_for_dynamic_shape_graph) {
  GraphBuilder graph_builder;
  auto root_graph = BuildDynamicShapeGraph();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, Build_Ok_DynamicShapeGraphWithMultiStreamDisabled) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "0", 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  auto const_node = root_graph->FindNode("data1");
  const_node->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
  auto relu = root_graph->FindNode("relu1");
  relu->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  auto output = root_graph->FindNode("netoutput");
  output->GetOpDesc()->SetOpEngineName("DNN_VM_RTS");
  for (const auto &node : root_graph->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
  }

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  GraphBuilder graph_builder;
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "Enable multi-stream in dynamic graph"), -1);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(GraphBuilderTest, test_dynamic_shape_graph_enable_single_stream) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  auto const_node = root_graph->FindNode("data1");
  const_node->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");
  auto relu = root_graph->FindNode("relu1");
  relu->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  auto output = root_graph->FindNode("netoutput");
  output->GetOpDesc()->SetOpEngineName("DNN_VM_RTS");
  for (const auto &node : root_graph->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
  }

  auto options = GetThreadLocalContext().GetAllGraphOptions();
  options[ENABLE_SINGLE_STREAM] = "true";
  GetThreadLocalContext().SetGraphOption(options);
  GraphBuilder graph_builder;
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  options[ENABLE_SINGLE_STREAM] = "false";
  GetThreadLocalContext().SetGraphOption(options);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(GraphBuilderTest, test_build_for_host_cpu_graph) {
  GraphBuilder graph_builder;
  auto root_graph = BuildDynamicShapeGraph();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  std::map<std::string, std::string> options_map;
  options_map.emplace("ge.exec.placement", "HOST");
  GetThreadLocalContext().SetGraphOption(options_map);
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  std::map<std::string, std::string> options_map_recovery;
  GetThreadLocalContext().SetGraphOption(options_map_recovery);
}

TEST_F(GraphBuilderTest, TestBuildForCompoundGraph) {
  GraphBuilder graph_builder;
  auto root_graph = BuildCompoundGraph();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, TestSpecialInputAndOutputSize) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph = ToComputeGraph(g1);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  int64_t special_size = 224 * 224 * 2 * 8 * 3 / 2;
  auto special_size_node = root_graph->FindNode("relu1");
  EXPECT_NE(special_size_node, nullptr);
  ge::AttrUtils::SetInt(special_size_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE,
                        special_size);
  ge::AttrUtils::SetInt(special_size_node->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_SPECIAL_OUTPUT_SIZE,
                        special_size);
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);

  GeRootModelPtr root_model;
  GraphBuilder graph_builder;
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  auto check_size = [&](const NodePtr &node) {
    EXPECT_NE(node, nullptr);
    int64_t size = 0;
    if (node->GetType() == DATA) {
      EXPECT_EQ(ge::AttrUtils::GetInt(node->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, size),
                true);
    } else if (node->GetType() == NETOUTPUT) {
      EXPECT_EQ(ge::AttrUtils::GetInt(node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_OUTPUT_SIZE, size),
                true);
    }
    EXPECT_EQ(size, special_size);
  };
  check_size(root_graph->FindNode("data1"));
  check_size(root_graph->FindNode("netoutput"));
}

TEST_F(GraphBuilderTest, TestSkAppendWs) {
  InitGeLib();
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph_1 = ToComputeGraph(g1);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph_1->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  AttrUtils::SetStr(root_graph_1, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  auto relu_g1_node = root_graph_1->FindNode("relu1");
  EXPECT_NE(relu_g1_node, nullptr);
  AttrUtils::SetStr(relu_g1_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(relu_g1_node->GetOpDesc(), "supportSuperKernel", 1);

  GeRootModelPtr root_model_1;
  GraphBuilder graph_builder;
  auto ret = graph_builder.Build(root_graph_1, root_model_1);
  EXPECT_EQ(ret, SUCCESS);
  auto sk_1 = root_graph_1->FindFirstNodeMatchType("SuperKernel");
  EXPECT_NE(sk_1, nullptr);
  ComputeGraphPtr sub_graph_1 = nullptr;
  auto sk_1_opdesc = sk_1->GetOpDesc();
  sub_graph_1 = sk_1_opdesc->TryGetExtAttr("_sk_sub_graph", sub_graph_1);
  EXPECT_NE(sub_graph_1, nullptr);
  relu_g1_node = sub_graph_1->FindNode("relu1");
  EXPECT_NE(relu_g1_node, nullptr);
  
  need_sk_append_ws = true;
  DEF_GRAPH(g2) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph_2 = ToComputeGraph(g2);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph_2->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  auto relu_g2_node = root_graph_2->FindNode("relu1");
  EXPECT_NE(relu_g2_node, nullptr);
  AttrUtils::SetStr(relu_g2_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(relu_g2_node->GetOpDesc(), "supportSuperKernel", 1);

  AttrUtils::SetStr(root_graph_2, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);

  GeRootModelPtr root_model_2;
  ret = graph_builder.Build(root_graph_2, root_model_2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(root_model_2, nullptr);
  auto sk_2 = root_graph_2->FindFirstNodeMatchType("SuperKernel");
  EXPECT_NE(sk_2, nullptr);
  ComputeGraphPtr sub_graph_2 = nullptr;
  auto sk_2_opdesc = sk_2->GetOpDesc();
  sub_graph_2 = sk_2_opdesc->TryGetExtAttr("_sk_sub_graph", sub_graph_2);
  EXPECT_NE(sub_graph_2, nullptr);
  relu_g2_node = sub_graph_2->FindNode("relu1");
  EXPECT_NE(relu_g2_node, nullptr);

  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetInputOffset()[0] - 512, relu_g1_node->GetOpDesc()->GetInputOffset()[0]);

  std::vector<int64_t> task_gen_input_offset_relu1_g1;
  (void)AttrUtils::GetListInt(relu_g1_node->GetOpDesc(), "_test_task_generate_input_offset", task_gen_input_offset_relu1_g1);
  EXPECT_EQ(task_gen_input_offset_relu1_g1.size(), 0);
  std::vector<int64_t> task_gen_input_offset_relu1_g2;
  (void)AttrUtils::GetListInt(relu_g2_node->GetOpDesc(), "_test_task_generate_input_offset", task_gen_input_offset_relu1_g2);
  EXPECT_EQ(task_gen_input_offset_relu1_g2.size(), 0);

  auto relu2_g1_node = root_graph_1->FindNode("relu2");
  EXPECT_NE(relu2_g1_node, nullptr);
  auto relu2_g2_node = root_graph_2->FindNode("relu2");
  EXPECT_NE(relu2_g2_node, nullptr);
  std::vector<int64_t> task_gen_output_offset_relu2_g1;
  (void)AttrUtils::GetListInt(relu2_g1_node->GetOpDesc(), "_test_task_generate_output_offset", task_gen_output_offset_relu2_g1);
  EXPECT_EQ(task_gen_output_offset_relu2_g1.size(), 1);
  std::vector<int64_t> task_gen_output_offset_relu2_g2;
  (void)AttrUtils::GetListInt(relu2_g2_node->GetOpDesc(), "_test_task_generate_output_offset", task_gen_output_offset_relu2_g2);
  EXPECT_EQ(task_gen_output_offset_relu2_g2.size(), 1);
  EXPECT_EQ(task_gen_output_offset_relu2_g2[0] - 512, task_gen_output_offset_relu2_g1[0]);
  int64_t origin_mem_size = 0;
  (void)AttrUtils::GetInt(relu2_g1_node->GetOpDesc(), "_test_get_run_context_memsize", origin_mem_size);
  int64_t re_gen_task_mem_size = 0;
  (void)AttrUtils::GetInt(relu2_g2_node->GetOpDesc(), "_test_get_run_context_memsize", re_gen_task_mem_size);
  EXPECT_TRUE(re_gen_task_mem_size - 512 == origin_mem_size);
  need_sk_append_ws = false;
}

TEST_F(GraphBuilderTest, TestAppendWs_fail) {
  DEF_GRAPH(g2) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph_2 = ToComputeGraph(g2);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph_2->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  auto relu_g2_node = root_graph_2->FindNode("relu1");
  EXPECT_NE(relu_g2_node, nullptr);
  std::vector<int64_t> append_ws_vec;
  AttrUtils::SetListInt(relu_g2_node->GetOpDesc(), "_append_ws", append_ws_vec);
  AttrUtils::SetStr(root_graph_2, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);

  GeRootModelPtr root_model_2;
  GraphBuilder graph_builder;
  auto ret = graph_builder.Build(root_graph_2, root_model_2);
  EXPECT_NE(ret, SUCCESS);
  append_ws_vec.emplace_back(0);
  AttrUtils::SetListInt(relu_g2_node->GetOpDesc(), "_append_ws", append_ws_vec);
  ret = graph_builder.Build(root_graph_2, root_model_2);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(GraphBuilderTest, TestAppendWs_multi_stream) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph_1 = ToComputeGraph(g1);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph_1->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  AttrUtils::SetStr(root_graph_1, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  auto relu_g1_node = root_graph_1->FindNode("relu1");
  EXPECT_NE(relu_g1_node, nullptr);
  AttrUtils::SetStr(relu_g1_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "1");
  auto relu2_g1_node = root_graph_1->FindNode("relu2");
  EXPECT_NE(relu2_g1_node, nullptr);
  AttrUtils::SetStr(relu2_g1_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "2");

  GeRootModelPtr root_model_1;
  GraphBuilder graph_builder;
  auto ret = graph_builder.Build(root_graph_1, root_model_1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(root_model_1, nullptr);
  int64_t origin_memory_size = 0;
  EXPECT_EQ(root_model_1->GetSubgraphInstanceNameToModel().size(), 1);
  (void)AttrUtils::GetInt(root_model_1->GetSubgraphInstanceNameToModel().begin()->second,
                          ATTR_MODEL_MEMORY_SIZE, origin_memory_size);
  std::vector<std::vector<int64_t>> origin_sub_memory_infos;
  (void)AttrUtils::GetListListInt(root_model_1->GetSubgraphInstanceNameToModel().begin()->second,
                                  ATTR_MODEL_SUB_MEMORY_INFO, origin_sub_memory_infos);

  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetWorkspaceBytes().size(), 0);
  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetWorkspace().size(), 0);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetWorkspaceBytes().size(), 0);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetWorkspace().size(), 0);

  DEF_GRAPH(g2) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph_2 = ToComputeGraph(g2);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph_2->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  auto relu_g2_node = root_graph_2->FindNode("relu1");
  EXPECT_NE(relu_g2_node, nullptr);
  std::vector<int64_t> append_ws_vec{500, 300};
  AttrUtils::SetListInt(relu_g2_node->GetOpDesc(), "_append_ws", append_ws_vec);
  AttrUtils::SetStr(relu_g2_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "1");
  auto relu2_g2_node = root_graph_2->FindNode("relu2");
  EXPECT_NE(relu2_g2_node, nullptr);
  std::vector<int64_t> append_ws_vec_2{100};
  AttrUtils::SetListInt(relu2_g2_node->GetOpDesc(), "_append_ws", append_ws_vec_2);
  AttrUtils::SetStr(relu2_g2_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "2");
  AttrUtils::SetStr(root_graph_2, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);

  GeRootModelPtr root_model_2;
  ret = graph_builder.Build(root_graph_2, root_model_2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(root_model_2, nullptr);
  int64_t memory_size = 0;
  EXPECT_EQ(root_model_2->GetSubgraphInstanceNameToModel().size(), 1);
  (void)AttrUtils::GetInt(root_model_2->GetSubgraphInstanceNameToModel().begin()->second,
                          ATTR_MODEL_MEMORY_SIZE, memory_size);
  std::vector<std::vector<int64_t>> sub_memory_infos;
  (void)AttrUtils::GetListListInt(root_model_2->GetSubgraphInstanceNameToModel().begin()->second,
                                  ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  EXPECT_EQ((memory_size - 1536), origin_memory_size);
  EXPECT_EQ((sub_memory_infos.size() - 1), origin_sub_memory_infos.size());
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetWorkspaceBytes().size(), 2);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetWorkspace().size(), 2);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetWorkspaceBytes().size(), 1);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetWorkspace().size(), 1);

  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetInputOffset()[0] - 1536, relu_g1_node->GetOpDesc()->GetInputOffset()[0]);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetOutputOffset()[0] - 1536, relu2_g1_node->GetOpDesc()->GetOutputOffset()[0]);
}

TEST_F(GraphBuilderTest, TestAppendWs) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph_1 = ToComputeGraph(g1);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph_1->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  AttrUtils::SetStr(root_graph_1, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);

  GeRootModelPtr root_model_1;
  GraphBuilder graph_builder;
  auto ret = graph_builder.Build(root_graph_1, root_model_1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(root_model_1, nullptr);
  int64_t origin_memory_size = 0;
  EXPECT_EQ(root_model_1->GetSubgraphInstanceNameToModel().size(), 1);
  (void)AttrUtils::GetInt(root_model_1->GetSubgraphInstanceNameToModel().begin()->second,
                          ATTR_MODEL_MEMORY_SIZE, origin_memory_size);
  std::vector<std::vector<int64_t>> origin_sub_memory_infos;
  (void)AttrUtils::GetListListInt(root_model_1->GetSubgraphInstanceNameToModel().begin()->second,
      ATTR_MODEL_SUB_MEMORY_INFO, origin_sub_memory_infos);
  auto relu_g1_node = root_graph_1->FindNode("relu1");
  EXPECT_NE(relu_g1_node, nullptr);
  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetWorkspaceBytes().size(), 0);
  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetWorkspace().size(), 0);
  auto relu2_g1_node = root_graph_1->FindNode("relu2");
  EXPECT_NE(relu2_g1_node, nullptr);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetWorkspaceBytes().size(), 0);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetWorkspace().size(), 0);

  DEF_GRAPH(g2) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto root_graph_2 = ToComputeGraph(g2);
  // somehow, GeRunningEnvFaker does not work well
  for (const auto &node : root_graph_2->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  auto relu_g2_node = root_graph_2->FindNode("relu1");
  EXPECT_NE(relu_g2_node, nullptr);
  std::vector<int64_t> append_ws_vec{500, 300};
  AttrUtils::SetListInt(relu_g2_node->GetOpDesc(), "_append_ws", append_ws_vec);
  auto relu2_g2_node = root_graph_2->FindNode("relu2");
  EXPECT_NE(relu2_g2_node, nullptr);
  std::vector<int64_t> append_ws_vec_2{100};
  AttrUtils::SetListInt(relu2_g2_node->GetOpDesc(), "_append_ws", append_ws_vec_2);
  AttrUtils::SetStr(root_graph_2, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);

  GeRootModelPtr root_model_2;
  ret = graph_builder.Build(root_graph_2, root_model_2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(root_model_2, nullptr);
  int64_t memory_size = 0;
  EXPECT_EQ(root_model_2->GetSubgraphInstanceNameToModel().size(), 1);
  (void)AttrUtils::GetInt(root_model_2->GetSubgraphInstanceNameToModel().begin()->second,
                          ATTR_MODEL_MEMORY_SIZE, memory_size);
  std::vector<std::vector<int64_t>> sub_memory_infos;
  (void)AttrUtils::GetListListInt(root_model_2->GetSubgraphInstanceNameToModel().begin()->second,
                                  ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  EXPECT_EQ((memory_size - 1024), origin_memory_size);
  EXPECT_EQ((sub_memory_infos.size() - 1), origin_sub_memory_infos.size());
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetWorkspaceBytes().size(), 2);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetWorkspace().size(), 2);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetWorkspaceBytes().size(), 1);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetWorkspace().size(), 1);

  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu_g1_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu2_g1_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetInputOffset().size(), 1);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetOutputOffset().size(), 1);
  EXPECT_EQ(relu_g2_node->GetOpDesc()->GetInputOffset()[0] - 1024, relu_g1_node->GetOpDesc()->GetInputOffset()[0]);
  EXPECT_EQ(relu2_g2_node->GetOpDesc()->GetOutputOffset()[0] - 1024, relu2_g1_node->GetOpDesc()->GetOutputOffset()[0]);
}

TEST_F(GraphBuilderTest, evaluate_graph_mode_simple) {
  InitGeLib();
  auto sub_data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  auto var = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  DEF_GRAPH(g1) {
      CHAIN(NODE("sub_data_1", sub_data_1)->EDGE(0, 0)->NODE("sub_add", sub_add));
      CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add", sub_add));
      CHAIN(NODE("sub_add", sub_add)->EDGE(0, 0)->NODE("sub_add1", sub_add1));
      CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add1", sub_add1));
      CHAIN(NODE("sub_add1", sub_add1)->EDGE(0, 0)->NODE("sub_net_output", sub_net_output));
  };

  const auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  for (const auto &node : compute_graph->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  std::map<string, string> options{};
  ModelDataInfo model;
  auto ret = EvaluateGraphResource(options, compute_graph, model);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(model.GetGraphMemorySize(), 4400193996288UL);
  EXPECT_EQ(model.GetVarMemorySize(), 2199023257088UL);
  FinalizeGeLib();
}

TEST_F(GraphBuilderTest, convert_const_or_constant_by_lifecycle) {
  GraphBuilder graph_builder;
  auto root_graph = BuildGraphWithConst();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  map<std::string, std::string> options{{OPTION_CONST_LIFECYCLE, "session"}};
  GetThreadLocalContext().SetSessionOption(options);
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_NE(ret, SUCCESS);
  auto node = root_graph->FindNode("const1");
  EXPECT_EQ(node->GetType(), CONSTANTOP);

  map<std::string, std::string> new_options{{MEMORY_OPTIMIZATION_POLICY, kMemoryPriority}};
  GetThreadLocalContext().SetSessionOption(new_options);
  ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  node = root_graph->FindNode("const1");
  EXPECT_EQ(node->GetType(), CONSTANT);
  map<std::string, std::string> options1{{MEMORY_OPTIMIZATION_POLICY, ""}};
  GetThreadLocalContext().SetSessionOption(options1);
}

TEST_F(GraphBuilderTest, get_graph_available_mem) {
  InitGeLib();
  auto sub_data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1, 1024});
  auto var = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1, 1024 * 1024});
  auto sub_add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1, 1024 * 1024});
  auto sub_add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1, 1024 * 1024});
  auto sub_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1, 1024});
  DEF_GRAPH(g1) {
    CHAIN(NODE("sub_data_1", sub_data_1)->EDGE(0, 0)->NODE("sub_add", sub_add));
    CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add", sub_add));
    CHAIN(NODE("sub_add", sub_add)->EDGE(0, 0)->NODE("sub_add1", sub_add1));
    CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add1", sub_add1));
    CHAIN(NODE("sub_add1", sub_add1)->EDGE(0, 0)->NODE("sub_net_output", sub_net_output));
  };

  const auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  for (const auto &node : compute_graph->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  std::map<string, string> options{};
  ModelDataInfo model;
  auto ret = EvaluateGraphResource(options, compute_graph, model);
  EXPECT_EQ(ret, SUCCESS);
  uint64_t available_mem = 0UL;
  EXPECT_EQ(GetGraphAvailableMemory(compute_graph, available_mem), SUCCESS);
  EXPECT_EQ(available_mem, 27917287424UL);

  DEF_GRAPH(g2) {
    CHAIN(NODE("sub_data_1", sub_data_1)->EDGE(0, 0)->NODE("sub_add", sub_add));
    CHAIN(NODE("var_1", var)->EDGE(0, 1)->NODE("sub_add", sub_add));
    CHAIN(NODE("sub_add", sub_add)->EDGE(0, 0)->NODE("sub_add1", sub_add1));
    CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add1", sub_add1));
    CHAIN(NODE("sub_add1", sub_add1)->EDGE(0, 0)->NODE("sub_net_output", sub_net_output));
  };

  const auto graph2 = ToGeGraph(g2);
  auto compute_graph2 = GraphUtilsEx::GetComputeGraph(graph2);
  for (const auto &node : compute_graph2->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
  compute_graph2->SetSessionID(compute_graph->GetSessionID());
  uint64_t available_mem1 = 0UL;
  EXPECT_EQ(GetGraphAvailableMemory(compute_graph2, available_mem1), SUCCESS);
  EXPECT_EQ(available_mem1, 25769803744);
  FinalizeGeLib();
}

TEST_F(GraphBuilderTest, Build_Ok_OnlineInferWithSessionLevelConst) {
  GraphBuilder graph_builder;
  // graph with Const
  auto root_graph = BuildGraphWithConst();
  EXPECT_EQ(GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANT).size(), 1UL);
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  GeRootModelPtr root_model;
  // 在线推理，同时开启内存优先和session级Const生命周期，预期图中Const转化为Constant
  std::map<std::string, std::string> options_map;
  options_map.emplace(ge::MEMORY_OPTIMIZATION_POLICY, ge::kMemoryPriority);
  options_map.emplace(ge::OPTION_GRAPH_RUN_MODE, "0");
  options_map.emplace(ge::OPTION_CONST_LIFECYCLE, "session");
  GetThreadLocalContext().SetGraphOption(options_map);
  VarManager::Instance(0UL)->Init(0U, 0UL, 0UL, 0UL);
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANT).size(), 0UL);
  EXPECT_EQ(GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANTOP).size(), 1UL);
  std::map<std::string, std::string> options_map_recovery;
  GetThreadLocalContext().SetGraphOption(options_map_recovery);
}

TEST_F(GraphBuilderTest, Build_Ok_OnlineInferWithMemoryPriority) {
  GraphBuilder graph_builder;
  // graph with Constant
  auto root_graph = BuildGraphWithConst();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  const auto const_nodes = GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANT);
  EXPECT_EQ(const_nodes.size(), 1UL);
  for (const auto &node : const_nodes) {
    auto opdesc = node->GetOpDesc();
    ge::OpDescUtilsEx::SetType(opdesc, ge::CONSTANTOP);
  }
  GeRootModelPtr root_model;
  // 在线推理，开启内存优先，预期图中Constant转化为Const
  std::map<std::string, std::string> options_map;
  options_map.emplace(ge::MEMORY_OPTIMIZATION_POLICY, ge::kMemoryPriority);
  options_map.emplace(ge::OPTION_GRAPH_RUN_MODE, "0");

  GetThreadLocalContext().SetGraphOption(options_map);
  VarManager::Instance(0UL)->Init(0U, 0UL, 0UL, 0UL);
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANTOP).size(), 0UL);
  EXPECT_EQ(GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANT).size(), 1UL);
  std::map<std::string, std::string> options_map_recovery;
  GetThreadLocalContext().SetGraphOption(options_map_recovery);
}

TEST_F(GraphBuilderTest, Build_Ok_TrainingWithMemoryPriority) {
  GraphBuilder graph_builder;
  // graph with Constant
  auto root_graph = BuildGraphWithConst();
  AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
  const auto const_nodes = GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANT);
  EXPECT_EQ(const_nodes.size(), 1UL);
  for (const auto &node : const_nodes) {
    auto opdesc = node->GetOpDesc();
    ge::OpDescUtilsEx::SetType(opdesc, ge::CONSTANTOP);
  }
  GeRootModelPtr root_model;
  // 训练场景，开启内存优先，预期图中Constant转化为Const
  std::map<std::string, std::string> options_map;
  options_map.emplace(ge::MEMORY_OPTIMIZATION_POLICY, ge::kMemoryPriority);
  options_map.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  GetThreadLocalContext().SetGraphOption(options_map);
  VarManager::Instance(0UL)->Init(0U, 0UL, 0UL, 0UL);
  auto ret = graph_builder.Build(root_graph, root_model);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANTOP).size(), 0UL);
  EXPECT_EQ(GraphUtils::FindNodesByTypeFromAllNodes(root_graph, ge::CONSTANT).size(), 1UL);
  std::map<std::string, std::string> options_map_recovery;
  GetThreadLocalContext().SetGraphOption(options_map_recovery);
}

}  // namespace ge
