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
#include <memory>

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"
#include "../passes/graph_builder_utils.h"

#include "macro_utils/dt_public_scope.h"
#include "api/gelib/gelib.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "graph/build/task_generator.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "common/preload/model/pre_model_partition_utils.h"

using namespace std;
using namespace testing;
using namespace ge;
namespace {
const char *const kIsInputVar = "INPUT_IS_VAR";
const char *const kIsOutputVar = "OUTPUT_IS_VAR";
const char *const kKernelInfoNameHccl = "ops_kernel_info_hccl";
const char *const kKernelLibName = "ops_kernel_lib";
const char *kSessionId = "123456";
const char *const kPartiallySupported = "partially_supported";
const std::set<std::string> kAicpuKernelLibs = {"aicpu_ascend_kernel", "aicpu_tf_kernel"};
const char *const kMixL2Engine = "_ffts_plus_mix_l2";
}  // namespace

class UtestTaskGeneratorTest : public testing::Test {
 public:
  struct FakeOpsKernelBuilder : OpsKernelBuilder {
    FakeOpsKernelBuilder(){};

   private:
    Status Initialize(const map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    Status CalcOpRunningParam(Node &node) override {
      return SUCCESS;
    };
    Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
      domi::TaskDef task_def;
      tasks.push_back(task_def);
      return SUCCESS;
    };
  };
  struct FakeOpsKernelInfoStore : OpsKernelInfoStore {
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

  ge::ComputeGraphPtr BuildGraphProfilingWithDynamicSubgraph() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto partitioncall = builder.AddNode("partitioncall", PARTITIONEDCALL, 1, 1);
    auto netoutput = builder.AddNode("netoutput", "NetOutput", 2, 0);
    auto op_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(addn1, 0, partitioncall, 0);
    builder.AddDataEdge(partitioncall, 0, netoutput, 0);
    auto sub_builder = ut::GraphBuilder("subgraph");
    auto sub_data = sub_builder.AddNode("subgraph_data", DATA, 1, 1);
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    auto addn = sub_builder.AddNode("subgraph_addn", ADDN, 1, 1);
    auto sub_net_output = sub_builder.AddNode("subgraph_output", NETOUTPUT, 1, 1);
    builder.AddDataEdge(sub_data, 0, addn, 0);
    builder.AddDataEdge(addn, 0, sub_net_output, 0);
    auto sub_graph = sub_builder.GetGraph();
    sub_graph->SetGraphUnknownFlag(true);
    auto root_graph = builder.GetGraph();
    root_graph->SetGraphUnknownFlag(true);

    AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
    sub_graph->SetParentNode(partitioncall);
    partitioncall->GetOpDesc()->AddSubgraphName("subgraph");
    partitioncall->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    sub_graph->SetParentGraph(root_graph);
    root_graph->AddSubgraph("subgraph", sub_graph);

    for (const auto &node : root_graph->GetDirectNode()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
    }

    for (const auto &node : sub_graph->GetDirectNode()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
    }
    return builder.GetGraph();
  }

  ge::ComputeGraphPtr BuildGraphFpProfiling() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto netoutput = builder.AddNode("netoutput", "NetOutput", 2, 0);
    auto op_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    op_desc->SetOpKernelLibName("GE");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(addn1, 0, netoutput, 0);
    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildGraphBpProfiling() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto loop0 = builder.AddNode(NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD, NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD, 1, 1);
    auto loop1 = builder.AddNode(NODE_NAME_FLOWCTRL_LOOP_ASSIGN, NODE_NAME_FLOWCTRL_LOOP_ASSIGN, 1, 1);
    auto netoutput = builder.AddNode("Node_Output", "NetOutput", 2, 0);
    auto data_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(data_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    data_desc->SetOpKernelLibName("GE");
    auto output_desc = netoutput->GetOpDesc();
    output_desc->SetOpKernelLibName("output");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddControlEdge(addn1, netoutput);
    builder.AddControlEdge(loop0, netoutput);
    builder.AddControlEdge(loop1, netoutput);
    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildGraphGetNextProfiling() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "GetNext", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto netoutput = builder.AddNode("Node_Output", "NetOutput", 2, 0);
    auto data_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(data_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    data_desc->SetOpKernelLibName("GE");
    auto output_desc = netoutput->GetOpDesc();
    output_desc->SetOpKernelLibName("output");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddControlEdge(addn1, netoutput);
    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildGraphWithVar(int64_t session_id) {
    // init
    MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
    VarManager::Instance(session_id)->Init(0, 0, 0, 0);
    ge::ut::GraphBuilder builder("graph");
    auto var_input = builder.AddNode("var", "Variable", 1, 1);
    auto const_input = builder.AddNode("const", "Const", 1, 1);
    auto assign = builder.AddNode("assgin", "Assign", 2, 1);
    // add link
    builder.AddDataEdge(var_input, 0, assign, 0);
    builder.AddDataEdge(const_input, 0, assign, 1);
    // set offset
    var_input->GetOpDesc()->SetOutputOffset({10000});
    const_input->GetOpDesc()->SetOutputOffset({1000});
    assign->GetOpDesc()->SetInputOffset({10100, 1000});
    assign->GetOpDesc()->SetOutputOffset({10100});
    // set inner offset
    int64_t inner_offset = 100;
    ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
    ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
    // add var addr
    VarManager::Instance(session_id)->var_resource_->var_offset_map_.emplace(10000, RT_MEMORY_HBM);

    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildHcclGraph() {
    ge::ut::GraphBuilder builder("graph");
    auto hccl_node = builder.AddNode("hccl_phony_node", "HCCL_PHONY", 0, 0);
    auto op_desc = hccl_node->GetOpDesc();
    op_desc->SetOpKernelLibName(kKernelInfoNameHccl);
    op_desc->SetStreamId(0);
    return builder.GetGraph();
  }
  ComputeGraphPtr BuildFftsGraph() {
    auto builder = ut::GraphBuilder("g1");
    auto data = builder.AddNode("data", DATA, 1, 1);
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    auto partitioned_call = builder.AddNode("PartitionedCall", PARTITIONEDCALL, 1, 1);
    ge::AttrUtils::SetBool(partitioned_call->GetOpDesc(), ATTR_NAME_FFTS_SUB_GRAPH, true);
    auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
    builder.AddDataEdge(data, 0, partitioned_call, 0);
    builder.AddDataEdge(partitioned_call, 0, net_output, 0);

    auto sub_builder = ut::GraphBuilder("subgraph");
    auto sub_data = sub_builder.AddNode("data", DATA, 1, 1);
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(sub_data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto addn = sub_builder.AddNode("addn", ADDN, 1, 1);
    auto sub_net_output = sub_builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
    builder.AddDataEdge(sub_data, 0, addn, 0);
    builder.AddDataEdge(addn, 0, sub_net_output, 0);
    auto sub_graph = sub_builder.GetGraph();
    auto root_graph = builder.GetGraph();

    AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
    sub_graph->SetParentNode(partitioned_call);
    partitioned_call->GetOpDesc()->AddSubgraphName("subgraph");
    partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    sub_graph->SetParentGraph(root_graph);
    root_graph->AddSubgraph("subgraph", sub_graph);

    for (const auto &node : root_graph->GetDirectNode()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
    }

    for (const auto &node : sub_graph->GetDirectNode()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
    }

    return root_graph;
  }

  ComputeGraphPtr BuildZeroCopyTestGraph() {
  /**
   *      Data Data
   *        \   /
   *         Add
   *         /  \ 
   *       Cast Netoutput 
   */
    auto builder = ut::GraphBuilder("g1");
    auto data1 = builder.AddNode("data", DATA, 1, 1);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    auto data2 = builder.AddNode("data", DATA, 1, 1);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
    auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
    auto add = builder.AddNode("add", ADD, 2, 1);
    auto cast = builder.AddNode("cast", "Cast", 1, 1);

    builder.AddDataEdge(data1, 0, add, 0);
    builder.AddDataEdge(data2, 0, add, 1);
    builder.AddDataEdge(add, 0, cast, 0);
    builder.AddDataEdge(add, 0, net_output, 0);

    builder.AddControlEdge(data1, add);
    builder.AddControlEdge(data2, add);
    builder.AddControlEdge(add, cast);
    builder.AddControlEdge(add, net_output);

    // set offset
    data1->GetOpDesc()->SetOutputOffset({2622464});
    data2->GetOpDesc()->SetOutputOffset({3802624});
    add->GetOpDesc()->SetInputOffset({2622464, 3802624});
    add->GetOpDesc()->SetOutputOffset({3804672});
    cast->GetOpDesc()->SetOutputOffset({22222});
    net_output->GetOpDesc()->SetOutputOffset({3804672});

    auto root_graph = builder.GetGraph();

    for (const auto &node : root_graph->GetDirectNode()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
      std::vector<int64_t> base_offset_list = {2622464, 3802624, 3804672};
      node->GetOpDesc()->SetInputOffset(base_offset_list);
      std::vector<int64_t> zero_copy_basic_offset = {2622464, 3802624, 3804672};
      std::vector<int64_t> zero_copy_relative_offset = {2622464, 3802624, 3804672};
      (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset);
      (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset);
    }
    return root_graph;
  }

  ComputeGraphPtr BuildFftsPlusGraph() {
    auto builder = ut::GraphBuilder("g1");
    auto data = builder.AddNode("data", DATA, 1, 1);
    auto reshape = builder.AddNode("reshape", RESHAPE, 1, 1);
    ge::AttrUtils::SetBool(reshape->GetOpDesc(), ATTR_NAME_NOTASK, true);
    auto partitioned_call = builder.AddNode("PartitionedCall", PARTITIONEDCALL, 1, 1);
    ge::AttrUtils::SetBool(partitioned_call->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);
    auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
    builder.AddDataEdge(data, 0, reshape, 0);
    builder.AddDataEdge(reshape, 0, partitioned_call, 0);
    builder.AddDataEdge(partitioned_call, 0, net_output, 0);

    auto sub_builder = ut::GraphBuilder("subgraph");
    auto sub_data = sub_builder.AddNode("data", DATA, 1, 1);
    AttrUtils::SetInt(sub_data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    auto addn = sub_builder.AddNode("addn", ADDN, 1, 1);
    auto sub_net_output = sub_builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
    builder.AddDataEdge(sub_data, 0, addn, 0);
    builder.AddDataEdge(addn, 0, sub_net_output, 0);
    auto sub_graph = sub_builder.GetGraph();
    auto root_graph = builder.GetGraph();

    AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, kSessionId);
    sub_graph->SetParentNode(partitioned_call);
    partitioned_call->GetOpDesc()->AddSubgraphName("subgraph");
    partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    sub_graph->SetParentGraph(root_graph);
    root_graph->AddSubgraph("subgraph", sub_graph);
    bool is_ffts_node = false;
    for (const auto &node : root_graph->GetDirectNode()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
      if (AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, is_ffts_node)) {
        (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME, kKernelLibName);
      }
    }

    for (const auto &node : sub_graph->GetDirectNode()) {
      node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
      node->GetOpDesc()->SetOpEngineName(kKernelLibName);
    }

    return root_graph;
  }

  ge::ComputeGraphPtr BuildGraphBpProfilingWithGlobalStep() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto global_step1 = builder.AddNode("ge_global_step", "Variable", 0, 1);
    auto netoutput = builder.AddNode("Node_Output", "NetOutput", 2, 0);
    auto data_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(data_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    data_desc->SetOpKernelLibName("GE");
    auto output_desc = netoutput->GetOpDesc();
    output_desc->SetOpKernelLibName("output");

    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(global_step1, 0, netoutput, 1);
    builder.AddControlEdge(addn1, netoutput);
    return builder.GetGraph();
  }

  ge::ComputeGraphPtr BuildGraphBpProfilingWithSubGraph() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto func_node = builder.AddNode("Case", "Case", 1, 1);
    auto global_step1 = builder.AddNode("ge_global_step", "Variable", 0, 1);
    auto netoutput = builder.AddNode("Node_Output", "NetOutput", 2, 0);
    auto data_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(data_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    data_desc->SetOpKernelLibName("GE");
    auto output_desc = netoutput->GetOpDesc();
    output_desc->SetOpKernelLibName("output");

    std::string subgraph_name_1 = "instance_branch_1";
    ComputeGraphPtr subgraph_1 = std::make_shared<ComputeGraph>(subgraph_name_1);
    subgraph_1->SetParentNode(func_node);
    subgraph_1->SetParentGraph(builder.GetGraph());
    size_t index = func_node->GetOpDesc()->GetSubgraphInstanceNames().size();
    func_node->GetOpDesc()->AddSubgraphName("branch_invalid");
    func_node->GetOpDesc()->AddSubgraphName("branch1");
    func_node->GetOpDesc()->SetSubgraphInstanceName(index, "branch_invalid");
    func_node->GetOpDesc()->SetSubgraphInstanceName(index + 1, subgraph_name_1);

    NodePtr global_step2 = subgraph_1->AddNode(CreateOpDesc("ge_global_step", "Variable", 0, 1));
    NodePtr output_node = subgraph_1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    GraphUtils::AddEdge(global_step2->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(func_node, 0, netoutput, 1);
    builder.AddControlEdge(addn1, netoutput);
    auto root_graph = builder.GetGraph();
    root_graph->AddSubGraph(subgraph_1);
    return root_graph;
  }
  void Init() {
    map<string, string> options;
    Status ret = ge::GELib::Initialize(options);
    EXPECT_EQ(ret, SUCCESS);
    GetThreadLocalContext().global_options_.clear();
    GetThreadLocalContext().session_options_.clear();
    GetThreadLocalContext().graph_options_.clear();
    OpsKernelInfoStorePtr ops_kernel_info_store_ptr = MakeShared<FakeOpsKernelInfoStore>();
    OpsKernelManager::GetInstance().ops_kernel_store_.insert(make_pair(kKernelLibName, ops_kernel_info_store_ptr));

    OpsKernelBuilderPtr fake_builder = MakeShared<FakeOpsKernelBuilder>();
    OpsKernelBuilderRegistry::GetInstance().kernel_builders_[kKernelLibName] = fake_builder;
  }
  void Finalize() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
  }

 protected:
  void SetUp() {
    Init();
  }
  void TearDown() {
    Finalize();
  }
};

namespace {
void ProfilingInit() {
  ProfilingProperties::Instance().is_training_trace_ = true;
  ProfilingProperties::Instance().is_execute_profiling_ = true;
}

void ProfilingExit() {
  ProfilingProperties::Instance().is_training_trace_ = false;
  ProfilingProperties::Instance().is_execute_profiling_ = false;
  ProfilingProperties::Instance().fp_point_ = "";
  ProfilingProperties::Instance().bp_point_ = "";
}

void BuildGraphProfiling(ge::ComputeGraphPtr &graph) {
  GeTensorDesc ge_tensor_desc;
  auto data_op = std::make_shared<OpDesc>("Data", GETNEXT);
  data_op->AddInputDesc(ge_tensor_desc);
  data_op->AddOutputDesc(ge_tensor_desc);
  auto data_node = graph->AddNode(data_op);

  auto data_op1 = std::make_shared<OpDesc>("Data", DATA);
  data_op1->AddInputDesc(ge_tensor_desc);
  data_op1->AddOutputDesc(ge_tensor_desc);
  auto data_node1 = graph->AddNode(data_op1);

  auto flatten_op = std::make_shared<OpDesc>("Add", ADD);
  flatten_op->AddInputDesc(ge_tensor_desc);
  flatten_op->AddInputDesc(ge_tensor_desc);
  flatten_op->AddOutputDesc(ge_tensor_desc);
  auto flatten_node = graph->AddNode(flatten_op);

  auto ar1_op = std::make_shared<OpDesc>("ar1", HCOMALLREDUCE);
  ar1_op->AddInputDesc(ge_tensor_desc);
  ar1_op->AddOutputDesc(ge_tensor_desc);
  auto ar1_node = graph->AddNode(ar1_op);

  auto softmax_op = std::make_shared<OpDesc>("MatMul", MATMUL);
  softmax_op->AddInputDesc(ge_tensor_desc);
  softmax_op->AddOutputDesc(ge_tensor_desc);
  auto softmax_node = graph->AddNode(softmax_op);

  auto ar2_op = std::make_shared<OpDesc>("ar2", HCOMALLREDUCE);
  ar2_op->AddInputDesc(ge_tensor_desc);
  ar2_op->AddOutputDesc(ge_tensor_desc);
  auto ar2_node = graph->AddNode(ar2_op);

  auto netouput_op = std::make_shared<OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  auto netouput_node = graph->AddNode(netouput_op);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), flatten_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), flatten_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(flatten_node->GetOutDataAnchor(0), ar1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ar1_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(softmax_node->GetOutDataAnchor(0), ar2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ar2_node->GetOutDataAnchor(0), netouput_node->GetInDataAnchor(0));
}
void SetGraphNodeKernel(ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    node->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
    node->GetOpDesc()->SetOpEngineName(kKernelLibName);
  }
}

void SetFusiondNode(ComputeGraphPtr &graph) {
  int group_id = 1;
  for (const auto &node : graph->GetAllNodes()) {
    (void)AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_L1_FUSION_GROUP_ID, group_id);
  }
}

void SetPartiallySupportedNode(ComputeGraphPtr &graph) {
  OpsKernelInfoStorePtr ops_kernel_info_store_ptr = MakeShared<UtestTaskGeneratorTest::FakeOpsKernelInfoStore>();
  OpsKernelManager::GetInstance().ops_kernel_store_.insert(make_pair("aicpu_ascend_kernel", ops_kernel_info_store_ptr));

  OpsKernelBuilderPtr fake_builder = MakeShared<UtestTaskGeneratorTest::FakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_["aicpu_ascend_kernel"] = fake_builder;
  for (const auto &node : graph->GetAllNodes()) {
    (void)AttrUtils::SetBool(node->GetOpDesc(), kPartiallySupported, true);
  }
}

void SetMixL2Node(ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == ADD) {
      (void)AttrUtils::SetStr(node->GetOpDesc(), "_alias_engine_name", kKernelLibName);
    }
  }
}
}  // namespace
TEST_F(UtestTaskGeneratorTest, GetTaskInfo_with_profiling_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("TestGraph");
  BuildGraphProfiling(compute_graph);
  SetGraphNodeKernel(compute_graph);

  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  ge::Model model_def;
  ProfilingInit();
  std::vector<std::string> task_index_op_name;
  auto ret = task_generator.GetTaskInfo(compute_graph, 0, model_def);
  EXPECT_EQ(ret, SUCCESS);
  ret = task_generator.GenModelTaskDef(compute_graph, 0, model_def);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(AttrUtils::GetListStr(model_def, ATTR_MODEL_TASK_INDEX_OP_NAME, task_index_op_name));
  EXPECT_FALSE(task_index_op_name.empty());
  ProfilingExit();
}

TEST_F(UtestTaskGeneratorTest, GetTaskInfo_with_mix_l2_nodes_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("TestGraph");
  BuildGraphProfiling(compute_graph);
  SetGraphNodeKernel(compute_graph);
  SetMixL2Node(compute_graph);

  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  ge::Model model_def;
  auto ret = task_generator.GetTaskInfo(compute_graph, 0, model_def);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTaskGeneratorTest, GetTask_with_fusion_nodes_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("TestGraph");
  BuildGraphProfiling(compute_graph);
  SetGraphNodeKernel(compute_graph);
  SetFusiondNode(compute_graph);

  std::string buffer_optimize = "l1_optimize";
  std::map<std::string, std::string> options;
  options[BUFFER_OPTIMIZE] = buffer_optimize;
  GetThreadLocalContext().SetGraphOption(options);
  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  ge::Model model_def;
  auto add_op_desc = std::make_shared<OpDesc>("Add_new", ADD);
  auto add = compute_graph->AddNode(add_op_desc);
  add->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
  add->GetOpDesc()->SetOpEngineName(kKernelLibName);
  auto ret = task_generator.GetTaskInfo(compute_graph, 0, model_def);
  EXPECT_EQ(ret, SUCCESS);
  ret = task_generator.ReGetTaskInfo(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = task_generator.GenModelTaskDef(compute_graph, 0, model_def);
  EXPECT_EQ(ret, SUCCESS);

  std::map<std::string, std::string> options_map_recovery;
  GetThreadLocalContext().SetGraphOption(options_map_recovery);
}

TEST_F(UtestTaskGeneratorTest, GetTaskInfo_with_partially_supported_nodes_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("TestGraph");
  BuildGraphProfiling(compute_graph);
  SetGraphNodeKernel(compute_graph);
  SetPartiallySupportedNode(compute_graph);

  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  ge::Model model_def;
  auto ret = task_generator.GetTaskInfo(compute_graph, 0, model_def);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTaskGeneratorTest, ReGetTaskInfo_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("TestGraph");
  BuildGraphProfiling(compute_graph);
  SetGraphNodeKernel(compute_graph);

  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  ge::Model model_def;
  auto ret = task_generator.GetTaskInfo(compute_graph, 0, model_def);
  EXPECT_EQ(ret, SUCCESS);

  auto add_op_desc = std::make_shared<OpDesc>("Add_new", ADD);
  auto add = compute_graph->AddNode(add_op_desc);
  add->GetOpDesc()->SetOpKernelLibName(kKernelLibName);
  add->GetOpDesc()->SetOpEngineName(kKernelLibName);
  ret = task_generator.ReGetTaskInfo(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTaskGeneratorTest, AutoFindFpOpIndex) {
  auto graph = BuildGraphFpProfiling();
  SetGraphNodeKernel(graph);
  TaskGenerator task_generator;
  ProfilingPoint profiling_point;
  profiling_point.fp_index = -1;
  ProfilingProperties::Instance().SetTrainingTrace(true);
  auto addn1 = graph->FindNode("addn1");
  EXPECT_EQ(task_generator.FindProfilingNodeIndex(graph, profiling_point), SUCCESS);
  // addn1 is fp
  EXPECT_EQ(profiling_point.fp_index, addn1->GetOpDesc()->GetId());

  // mark addn1 is no_task and find again
  AttrUtils::SetBool(addn1->GetOpDesc(), ATTR_NAME_NOTASK, true);
  ProfilingPoint profiling_point2;
  EXPECT_EQ(task_generator.FindProfilingNodeIndex(graph, profiling_point2), SUCCESS);
  // addn1 is not fp
  EXPECT_NE(profiling_point2.fp_index, addn1->GetOpDesc()->GetId());
  ProfilingProperties::Instance().SetTrainingTrace(false);
}

TEST_F(UtestTaskGeneratorTest, AutoFindFpOpIndexCrossPartitionedCall) {
  auto graph = BuildFftsGraph();
  SetGraphNodeKernel(graph);
  TaskGenerator task_generator;
  ProfilingPoint profiling_point;
  profiling_point.fp_index = -1;
  ProfilingProperties::Instance().SetTrainingTrace(true);
  auto partitioned_call = graph->FindNode("PartitionedCall");
  auto sub_graph = NodeUtils::GetSubgraph(*partitioned_call, 0);
  auto addn = sub_graph->FindNode("addn");
  EXPECT_EQ(task_generator.FindProfilingNodeIndex(graph, profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.fp_index, addn->GetOpDesc()->GetId());
  EXPECT_EQ(profiling_point.bp_index, addn->GetOpDesc()->GetId());

  // mark addn is no_task and find again
  AttrUtils::SetBool(addn->GetOpDesc(), ATTR_NAME_NOTASK, true);
  ProfilingPoint profiling_point2;
  EXPECT_EQ(task_generator.FindProfilingNodeIndex(graph, profiling_point2), SUCCESS);
  // addn is not fp
  EXPECT_NE(profiling_point2.fp_index, addn->GetOpDesc()->GetId());
  EXPECT_NE(profiling_point2.bp_index, addn->GetOpDesc()->GetId());
  ProfilingProperties::Instance().SetTrainingTrace(false);
}

TEST_F(UtestTaskGeneratorTest, FindLastBpFromBpNodeWithSubGraph) {
  auto graph = BuildGraphBpProfilingWithSubGraph();
  auto net_output = graph->FindNode("Node_Output");
  // netoutput has no data input, return default value 0
  int64_t bp_index = 0;
  EXPECT_EQ(ProfilingTaskUtils::FindLastBpFromBpNode(graph, net_output.get(), bp_index), 0);
}

TEST_F(UtestTaskGeneratorTest, FindLastBpFromBpNode) {
  auto graph = BuildGraphBpProfilingWithGlobalStep();
  auto net_output = graph->FindNode("Node_Output");
  // netoutput has no data input, return default value 0
  int64_t bp_index = 0;
  EXPECT_EQ(ProfilingTaskUtils::FindLastBpFromBpNode(graph, net_output.get(), bp_index), 0);
}

TEST_F(UtestTaskGeneratorTest, UpdateOpIsVarAttr) {
  int64_t session_id = 0;
  ge::ComputeGraphPtr graph = BuildGraphWithVar(session_id);
  graph->SetSessionID(session_id);
  TaskGenerator task_generator;
  auto assign = graph->FindNode("assgin");
  task_generator.UpdateOpIsVarAttr(assign->GetOpDesc(), session_id);
  // input
  vector<bool> input_var;
  AttrUtils::GetListBool(assign->GetOpDesc(), kIsInputVar, input_var);
  EXPECT_EQ(input_var.size(), 2);
  EXPECT_EQ(input_var[0], true);
  EXPECT_EQ(input_var[1], false);
  // output
  vector<bool> output_var;
  AttrUtils::GetListBool(assign->GetOpDesc(), kIsOutputVar, output_var);
  EXPECT_EQ(output_var.size(), 1);
  EXPECT_EQ(output_var[0], true);

  MemManager::Instance().Finalize();
}

TEST_F(UtestTaskGeneratorTest, AutoFindBpOpIndex) {
  auto graph = BuildGraphBpProfiling();
  SetGraphNodeKernel(graph);
  TaskGenerator task_generator;
  ProfilingPoint profiling_point;
  std::vector<int64_t> all_reduce_nodes;
  ProfilingInit();
  auto net_output = graph->FindNode("Node_Output");
  auto output_desc = net_output->GetOpDesc();
  EXPECT_EQ(task_generator.FindProfilingNodeIndex(graph, profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.end_index.size(), 1);
  EXPECT_EQ(*profiling_point.end_index.begin(), output_desc->GetId());
  ProfilingTaskUtils profiling_task_utils(task_generator.nodes_);
  graph->SetNeedIteration(true);
  EXPECT_EQ(profiling_task_utils.AutoFindBpOpIndex(graph, profiling_point, all_reduce_nodes), SUCCESS);
  EXPECT_EQ(profiling_point.end_index.size(), 3);
  auto loop = graph->FindNode(NODE_NAME_FLOWCTRL_LOOP_ASSIGN);
  auto loop_op = loop->GetOpDesc();
  EXPECT_EQ(profiling_point.bp_index, loop_op->GetId());
  ge::OpDescUtilsEx::SetType(output_desc, "HcomAllReduce");
  output_desc->SetName("hcom");
  EXPECT_EQ(profiling_task_utils.AutoFindBpOpIndex(graph, profiling_point, all_reduce_nodes), SUCCESS);
  EXPECT_EQ(all_reduce_nodes.size(), 1U);
  // empty nodes
  ProfilingPoint profiling_point2;
  profiling_point2.bp_index = -1;
  ProfilingTaskUtils profiling_task_utils2({});
  EXPECT_EQ(profiling_task_utils2.AutoFindBpOpIndex(graph, profiling_point2, all_reduce_nodes), SUCCESS);
  EXPECT_TRUE(profiling_point2.bp_index == -1);
  ProfilingExit();
}

TEST_F(UtestTaskGeneratorTest, FindGetNextProfilingIndex) {
  auto graph = BuildGraphGetNextProfiling();
  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  task_generator.FilterCandidatesNodes(graph);
  ProfilingTaskUtils profiling_task_utils(task_generator.nodes_);
  ProfilingPoint profiling_point;
  EXPECT_EQ(profiling_task_utils.FindProfilingTaskIndex(graph, profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.get_next_node_index.size(), 0U);
  ProfilingProperties::Instance().SetTrainingTrace(true);
  EXPECT_EQ(profiling_task_utils.FindProfilingTaskIndex(graph, profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.get_next_node_index.size(), 1U);
  ProfilingProperties::Instance().SetTrainingTrace(false);
  EXPECT_EQ(profiling_task_utils.FindProfilingTaskIndex(graph, profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.get_next_node_index.size(), 1U);
  auto get_next_node = graph->FindNode("data");
  int64_t node_index = 0;
  int64_t log_id = -1;
  EXPECT_EQ(profiling_task_utils.CalculateLogId(profiling_point.get_next_node_index, node_index,
      20000, log_id), SUCCESS);
  EXPECT_EQ(log_id, 20000);
}

TEST_F(UtestTaskGeneratorTest, GenerateTask) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  OpsKernelInfoStorePtr ops_kernel_info_store_ptr = MakeShared<FakeOpsKernelInfoStore>();
  OpsKernelManager::GetInstance().ops_kernel_store_.insert(make_pair(kKernelInfoNameHccl, ops_kernel_info_store_ptr));

  OpsKernelBuilderPtr fake_builder = MakeShared<FakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_[kKernelInfoNameHccl] = fake_builder;

  auto graph = BuildHcclGraph();
  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  vector<int64_t> all_reduce_nodes;
  map<uint32_t, string> op_name_map;
  ModelPtr model_ptr = MakeShared<ge::Model>();
  EXPECT_EQ(task_generator.GenerateTask(graph, *model_ptr), SUCCESS);
  size_t task_num = 0U;
  for (const auto &iter : task_generator.node_id_2_node_tasks_) {
    task_num += iter.second.size();
  }
  EXPECT_EQ(task_num, 1);

  auto node = graph->FindNode("hccl_phony_node");
  EXPECT_TRUE(node != nullptr);
  auto *ops_kernel_info_store =
      node->GetOpDesc()->TryGetExtAttr<OpsKernelInfoStore *>("OpsKernelInfoStorePtr", nullptr);
  EXPECT_EQ(ops_kernel_info_store, ops_kernel_info_store_ptr.get());
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_.erase(kKernelInfoNameHccl);
}

TEST_F(UtestTaskGeneratorTest, SetFpBpByOptions) {
  std::string fp_name("data");
  std::string bp_name("addn1");
  map<std::string, string> options_map = {{OPTION_EXEC_PROFILING_OPTIONS, R"({"fp_point":"data","bp_point":"addn1"})"}};
  ge::GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);
  ProfilingInit();
  auto graph = BuildGraphBpProfiling();
  graph->SetNeedIteration(true);
  auto fp_node = graph->FindNode(fp_name);
  auto bp_node = graph->FindNode(bp_name);
  EXPECT_NE(fp_node, nullptr);
  EXPECT_NE(bp_node, nullptr);
  SetGraphNodeKernel(graph);
  TaskGenerator task_generator;
  task_generator.FilterCandidatesNodes(graph);
  ProfilingTaskUtils profiling_task_utils(task_generator.nodes_);
  ProfilingPoint profiling_point;
  vector<int64_t> all_reduce_nodes;
  EXPECT_EQ(profiling_task_utils.FindProfilingTaskIndex(graph, profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.fp_index, fp_node->GetOpDesc()->GetId());
  EXPECT_EQ(profiling_point.bp_index, bp_node->GetOpDesc()->GetId());
  graph->SetNeedIteration(false);
  bp_node->GetOpDesc()->SetName("not_env_name");
  bp_node->GetOpDesc()->SetType(HCOMALLREDUCE);
  AttrUtils::SetListStr(bp_node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"addn1"});
  profiling_point.bp_index = -1;
  EXPECT_EQ(profiling_task_utils.FindProfilingTaskIndex(graph, profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.bp_index, bp_node->GetOpDesc()->GetId());
  ProfilingExit();
}

TEST_F(UtestTaskGeneratorTest, SubGraphOfDynamicGraphSkipFind) {
  std::string fp_name("data");
  std::string bp_name("addn1");
  map<std::string, string> options_map = {{OPTION_EXEC_PROFILING_OPTIONS, R"({"fp_point":"data","bp_point":"addn1"})"}};
  ge::GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);
  ProfilingInit();

  auto graph = BuildGraphProfilingWithDynamicSubgraph();
  SetGraphNodeKernel(graph);
  TaskGenerator task_generator;
  task_generator.FilterCandidatesNodes(graph);
  ProfilingTaskUtils profiling_task_utils(task_generator.nodes_);
  ProfilingPoint profiling_point;
  vector<int64_t> all_reduce_nodes;
  EXPECT_EQ(profiling_task_utils.FindProfilingTaskIndex(graph->GetSubgraph("subgraph"), profiling_point), SUCCESS);
  EXPECT_EQ(profiling_point.fp_index, 0);  // 0 is default value, not find keep default value
  EXPECT_EQ(profiling_point.bp_index, 0);  // 0 is default value, not find keep default value
  ProfilingExit();
}

TEST_F(UtestTaskGeneratorTest, GenerateFftsTask) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  OpsKernelInfoStorePtr ops_kernel_info_store_ptr = MakeShared<FakeOpsKernelInfoStore>();
  OpsKernelManager::GetInstance().ops_kernel_store_.insert(make_pair(kKernelLibName, ops_kernel_info_store_ptr));

  OpsKernelBuilderPtr fake_builder = MakeShared<FakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_[kKernelLibName] = fake_builder;

  auto graph = BuildFftsGraph();
  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  map<uint32_t, string> op_name_map;
  mmSetEnv("MAX_COMPILE_CORE_NUMBER", "-1", 1);
  ModelPtr model_ptr = MakeShared<ge::Model>();
  EXPECT_EQ(task_generator.GenerateTask(graph, *model_ptr), SUCCESS);
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_.erase(kKernelLibName);
  unsetenv("MAX_COMPILE_CORE_NUMBER");
}

TEST_F(UtestTaskGeneratorTest, GenerateTaskForZeroCopyTest) {
  const std::set<std::string> kDataOpType{DATA, AIPPDATA, ANN_DATA};
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  OpsKernelInfoStorePtr ops_kernel_info_store_ptr = MakeShared<FakeOpsKernelInfoStore>();
  OpsKernelManager::GetInstance().ops_kernel_store_.insert(make_pair(kKernelLibName, ops_kernel_info_store_ptr));

  auto graph = BuildZeroCopyTestGraph();
  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  PreRuntimeParam runtime_param;
  uint8_t mem_base_addr = 0;
  runtime_param.mem_size = 0x20002000u;
  runtime_param.logic_mem_base = reinterpret_cast<uintptr_t>(&mem_base_addr);
  runtime_param.weight_size = 8;
  ModelPtr model_ptr = MakeShared<ge::Model>();
  std::vector<uint64_t> weights_value(100, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  EXPECT_TRUE(AttrUtils::SetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(model_ptr, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[ge::SOC_VERSION] = "Ascend035";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(task_generator.InitZeroCopyInfo(graph, runtime_param), SUCCESS);
  EXPECT_EQ(task_generator.GenerateTask(graph, *model_ptr), SUCCESS);
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_.erase(kKernelLibName);
}

TEST_F(UtestTaskGeneratorTest, GenerateFftsPlusTask) {
  OpsKernelInfoStorePtr ops_kernel_info_store_ptr = MakeShared<FakeOpsKernelInfoStore>();
  OpsKernelManager::GetInstance().ops_kernel_store_.insert(make_pair(kKernelLibName, ops_kernel_info_store_ptr));

  OpsKernelBuilderPtr fake_builder = MakeShared<FakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_[kKernelLibName] = fake_builder;

  auto graph = BuildFftsPlusGraph();
  RunContext run_context;
  TaskGenerator task_generator(nullptr, 0, &run_context);
  map<uint32_t, string> op_name_map;

  ModelPtr model_ptr = MakeShared<ge::Model>();
  EXPECT_EQ(task_generator.GenerateTask(graph, *model_ptr), SUCCESS);
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_.erase(kKernelLibName);
}
