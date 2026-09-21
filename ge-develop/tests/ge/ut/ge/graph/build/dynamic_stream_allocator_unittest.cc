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

#include <string>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/build/stream/dynamic_stream_allocator.h"
#include "graph/build/stream/stream_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/graph_utils.h"
#include "api/gelib/gelib.h"

namespace ge {
class UtestDynamicStreamAllocator : public testing::Test {
 protected:
  void SetUp() { InitGeLib(); }

  void TearDown() { FinalizeGeLib(); }

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

  void InitGeLib() {
    map<string, string> options;
    Status ret = ge::GELib::Initialize(options);
    EXPECT_EQ(ret, SUCCESS);
    auto instance_ptr = ge::GELib::GetInstance();
    EXPECT_NE(instance_ptr, nullptr);

    //  SchedulerConf conf;
    SchedulerConf scheduler_conf;

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

    scheduler_conf.cal_engines["DNN_VM_AICPU"] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines["DNN_VM_AICPU"]->name = "DNN_VM_AICPU";
    scheduler_conf.cal_engines["DNN_VM_AICPU"]->id = "DNN_VM_AICPU";
    scheduler_conf.cal_engines["DNN_VM_AICPU"]->independent = false;
    scheduler_conf.cal_engines["DNN_VM_AICPU"]->attach = true;
    scheduler_conf.cal_engines["DNN_VM_AICPU"]->skip_assign_stream = false;

    scheduler_conf.cal_engines["DNN_HCCL"] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines["DNN_HCCL"]->name = "DNN_HCCL";
    scheduler_conf.cal_engines["DNN_HCCL"]->id = "DNN_HCCL";
    scheduler_conf.cal_engines["DNN_HCCL"]->independent = true;
    scheduler_conf.cal_engines["DNN_HCCL"]->attach = false;
    scheduler_conf.cal_engines["DNN_HCCL"]->skip_assign_stream = false;

    scheduler_conf.cal_engines["ffts_plus"] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines["ffts_plus"]->name = "ffts_plus";
    scheduler_conf.cal_engines["ffts_plus"]->id = "ffts_plus";
    scheduler_conf.cal_engines["ffts_plus"]->independent = false;
    scheduler_conf.cal_engines["ffts_plus"]->attach = true;
    scheduler_conf.cal_engines["ffts_plus"]->skip_assign_stream = true;

    scheduler_conf.cal_engines["DSAEngine"] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines["DSAEngine"]->name = "DSAEngine";
    scheduler_conf.cal_engines["DSAEngine"]->id = "DSAEngine";
    scheduler_conf.cal_engines["DSAEngine"]->independent = false;
    scheduler_conf.cal_engines["DSAEngine"]->attach = false;
    scheduler_conf.cal_engines["DSAEngine"]->skip_assign_stream = false;

    instance_ptr->DNNEngineManagerObj().schedulers_["aaaaaaa"] = scheduler_conf;

    DNNEngineAttribute attr_aicore = {"AIcoreEngine",  {},  PriorityEnum::COST_0, DEVICE, FORMAT_RESERVED,
                                      FORMAT_RESERVED, true};
    auto aicore_engine = std::make_shared<AICoreDNNEngine>(attr_aicore);
    instance_ptr->DNNEngineManagerObj().engines_map_["AIcoreEngine"] = aicore_engine;

    DNNEngineAttribute attr_aicpu = {"DNN_VM_AICPU",  {},  PriorityEnum::COST_2, DEVICE, FORMAT_RESERVED,
                                     FORMAT_RESERVED, true};
    auto aicpu_engine = std::make_shared<AICpuDNNEngine>(attr_aicpu);
    instance_ptr->DNNEngineManagerObj().engines_map_["DNN_VM_AICPU"] = aicpu_engine;

    DNNEngineAttribute attr_hccl = {"DNN_HCCL",      {},  PriorityEnum::COST_1, DEVICE, FORMAT_RESERVED,
                                    FORMAT_RESERVED, true};
    auto hccl_engine = std::make_shared<HcclDNNEngine>(attr_hccl);
    instance_ptr->DNNEngineManagerObj().engines_map_["DNN_HCCL"] = hccl_engine;

    DNNEngineAttribute attr_ge_local = {"DNN_VM_GE_LOCAL", {},  PriorityEnum::COST_9, DEVICE, FORMAT_RESERVED,
                                        FORMAT_RESERVED,   true};
    auto ge_local_engine = std::make_shared<GeLocalDNNEngine>(attr_ge_local);
    instance_ptr->DNNEngineManagerObj().engines_map_["DNN_VM_GE_LOCAL"] = ge_local_engine;

    DNNEngineAttribute attr_dsa = {"DSAEngine",     {},  PriorityEnum::COST_1, DEVICE, FORMAT_RESERVED,
                                   FORMAT_RESERVED, true};
    auto dsa_engine = std::make_shared<GeLocalDNNEngine>(attr_dsa);
    instance_ptr->DNNEngineManagerObj().engines_map_["DSAEngine"] = dsa_engine;
  }

  void FinalizeGeLib() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
  }

  static SubGraphInfoPtr BuildSubGraph(ComputeGraphPtr compute_graph, const string &engine_name) {
    SubGraphInfoPtr subgraph = std::make_shared<SubGraphInfo>();
    subgraph->SetSubGraph(compute_graph);
    subgraph->SetEngineName(engine_name);
    return subgraph;
  }

  NodePtr AddPlaceHolder(ComputeGraphPtr compute_graph, const string &name) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, "PlaceHolder");
    op_desc->AddInputDesc(GeTensorDesc());
    op_desc->AddOutputDesc(GeTensorDesc());
    NodePtr node = compute_graph->AddNode(op_desc);
    node->SetOwnerComputeGraph(compute_graph);
    return node;
  }

  NodePtr AddEnd(ComputeGraphPtr compute_graph, const string &name) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, "End");
    op_desc->AddInputDesc(GeTensorDesc());
    op_desc->AddOutputDesc(GeTensorDesc());
    NodePtr node = compute_graph->AddNode(op_desc);
    node->SetOwnerComputeGraph(compute_graph);
    return node;
  }

  void AddPlaceHolderAndEnd(SubGraphInfoPtr subgraph, int in_num, int out_num) {
    ComputeGraphPtr compute_graph = subgraph->GetSubGraph();

    std::unordered_map<ge::NodePtr, ge::NodePtr> pld_2_end_map;
    if (in_num == 1) {
      NodePtr node = AddPlaceHolder(compute_graph, "placeholder");
      pld_2_end_map.emplace(node, nullptr);
    } else {
      for (int i = 0; i < in_num; i++) {
        NodePtr node = AddPlaceHolder(compute_graph, "placeholder" + to_string(i + 1));
        pld_2_end_map.emplace(node, nullptr);
      }
    }
    subgraph->SetPld2EndMap(pld_2_end_map);

    std::unordered_map<ge::NodePtr, ge::NodePtr> end_2_pld_map;
    if (out_num == 1) {
      NodePtr node = AddEnd(compute_graph, "end");
      end_2_pld_map.emplace(node, nullptr);
    } else {
      for (int i = 0; i < out_num; i++) {
        NodePtr node = AddEnd(compute_graph, "end" + to_string(i + 1));
        end_2_pld_map.emplace(node, nullptr);
      }
    }

    subgraph->SetEnd2PldMap(end_2_pld_map);
  }

  SubGraphInfoPtr CreateSubgraph(const string &graph_name, const ComputeGraphPtr &graph, const string &node_name,
                                 const string &engine, int in_num = 1, int out_num = 1) {
    ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>(graph_name);
    const auto node = graph->FindNode(node_name);
    compute_graph->AddNode(node->GetOpDesc());

    SubGraphInfoPtr subgraph = BuildSubGraph(compute_graph, engine);
    AddPlaceHolderAndEnd(subgraph, in_num, out_num);

    return subgraph;
  }

  void LinkSubGraph(SubGraphInfoPtr subgraph1, const string &end_name, SubGraphInfoPtr subgraph2,
                    const string &placeholder_name) {
    NodePtr end_node = subgraph1->GetSubGraph()->FindNode(end_name);
    assert(end_node != nullptr);

    NodePtr placeholder_node = subgraph2->GetSubGraph()->FindNode(placeholder_name);
    assert(placeholder_node != nullptr);

    subgraph1->end_to_pld_[end_node] = placeholder_node;
    subgraph2->pld_to_end_[placeholder_node] = end_node;
  }

  int64_t GetStreamId(const ComputeGraphPtr &graph, const string &node_name) {
    const auto node = graph->FindNode(node_name);
    return node->GetOpDesc()->GetStreamId();
  }

  size_t GetSendEventNum(const ComputeGraphPtr &graph, const string &node_name) {
    return GetSendEventList(graph, node_name).size();
  }

  std::vector<uint32_t> GetSendEventList(const ComputeGraphPtr &graph, const string &node_name) {
    const auto node = graph->FindNode(node_name);
    std::vector<uint32_t> send_event_ids;
    AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_SEND_EVENT_IDS, send_event_ids);
    return send_event_ids;
  }

  size_t GetRecvEventNum(const ComputeGraphPtr &graph, const string &node_name) {
    return GetRecvEventList(graph, node_name).size();
  }

  std::vector<uint32_t> GetRecvEventList(const ComputeGraphPtr &graph, const string &node_name) {
    const auto node = graph->FindNode(node_name);
    std::vector<uint32_t> recv_event_ids;
    AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_RECV_EVENT_IDS, recv_event_ids);
    return recv_event_ids;
  }

  void AcParallelEnable() {
    std::map<std::string, std::string> options;
    options[AC_PARALLEL_ENABLE] = "1";
    GetThreadLocalContext().SetGraphOption(options);
  }

  void InvalidAcParallelEnable() {
    std::map<std::string, std::string> options;
    options[AC_PARALLEL_ENABLE] = "-1";
    GetThreadLocalContext().SetGraphOption(options);
  }

  void AcParallelDisable() {
    std::map<std::string, std::string> options;
    options[AC_PARALLEL_ENABLE] = "0";
    GetThreadLocalContext().SetGraphOption(options);
  }

  Graph2SubGraphInfoList CreateSubgraphMap(const ComputeGraphPtr &graph,
                                           const std::vector<std::vector<SubGraphInfoPtr>> &subgraph) {
    Graph2SubGraphInfoList subgraph_map;
    subgraph_map[graph] = subgraph[0];
    if (subgraph.size() == 1U) {
      return subgraph_map;
    }
    const auto &subgraphs = graph->GetAllSubgraphs();
    for (size_t i = 1U; i < subgraph.size(); ++i) {
      subgraph_map.emplace(subgraphs[i - 1U], subgraph[i]);
    }
    return subgraph_map;
  }
};

// aicore, dsa and hccl can assign a stream respectively
TEST_F(UtestDynamicStreamAllocator, aicore_dsa_hccl_multi_stream) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)
              ->NODE("relu", RELU)
              ->NODE("allreduce", HCOMALLREDUCE)
              ->NODE("dsa", "DSARandomUniform")
              ->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto gelocal = CreateSubgraph("gelocal", graph, "data1", "DNN_VM_GE_LOCAL", 0, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto hccl = CreateSubgraph("hccl", graph, "allreduce", "DNN_HCCL", 1, 1);
  auto dsa = CreateSubgraph("dsa", graph, "dsa", "DSAEngine", 1, 1);
  auto gelocal2 = CreateSubgraph("gelocal2", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(gelocal, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", hccl, "placeholder");
  LinkSubGraph(hccl, "end", dsa, "placeholder");
  LinkSubGraph(dsa, "end", gelocal2, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicore, hccl, dsa, gelocal, gelocal2};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 3);
  EXPECT_EQ(allocator.GetEventNum(), 3);
  EXPECT_EQ(GetStreamId(graph, "data1"), 0);
  EXPECT_EQ(GetStreamId(graph, "relu"), 0);
  EXPECT_EQ(GetStreamId(graph, "allreduce"), 1);
  EXPECT_EQ(GetStreamId(graph, "dsa"), 2);
  EXPECT_EQ(GetStreamId(graph, "output"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "relu"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "allreduce"), 1);
  EXPECT_EQ(GetSendEventNum(graph, "allreduce"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "dsa"), 1);
  EXPECT_EQ(GetSendEventNum(graph, "dsa"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "output"), 1);
}

// getnext in white list can assign a stream
TEST_F(UtestDynamicStreamAllocator, getnext_aicore_multi_stream) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("getnext", GETNEXT)->NODE("relu", RELU)->NODE("output", NETOUTPUT));
    CTRL_CHAIN(NODE("getnext")->NODE("output"));
  };
  auto graph = ToComputeGraph(g1);

  auto aicpu = CreateSubgraph("aicpu", graph, "getnext", "DNN_VM_AICPU", 0, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(aicpu, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", gelocal, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicpu, aicore, gelocal};

  graph->FindNode("relu")->GetOpDesc()->SetId(1);
  graph->FindNode("output")->GetOpDesc()->SetId(2);
  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 1);
  EXPECT_EQ(GetStreamId(graph, "getnext"), 1);
  EXPECT_EQ(GetStreamId(graph, "relu"), 0);
  EXPECT_EQ(GetStreamId(graph, "output"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "getnext"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "relu"), 1);
  for (const auto &node : graph->GetDirectNode()) {
    std::cout << "node: " << node->GetName() << ", id: " << node->GetOpDesc()->GetId() << std::endl;
  }
}

// gelocal will not attach to getnext
TEST_F(UtestDynamicStreamAllocator, gelocal_getnext_aicore_multi_stream) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("const", CONSTANT)->NODE("getnext", GETNEXT)->NODE("relu", RELU)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto gelocal = CreateSubgraph("gelocal", graph, "const", "DNN_VM_GE_LOCAL", 0, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "getnext", "DNN_VM_AICPU", 1, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto gelocal2 = CreateSubgraph("gelocal2", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(gelocal, "end", aicpu, "placeholder");
  LinkSubGraph(aicpu, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", gelocal2, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {gelocal, aicpu, aicore, gelocal2};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 2);
  EXPECT_EQ(GetStreamId(graph, "const"), 0);
  EXPECT_EQ(GetStreamId(graph, "getnext"), 1);
  EXPECT_EQ(GetStreamId(graph, "relu"), 0);
  EXPECT_EQ(GetStreamId(graph, "output"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "getnext"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "relu"), 1);
}

// gelocal can attach to getnext
TEST_F(UtestDynamicStreamAllocator, gelocal_getnext_aicore_ac_parallel_enable) {
  AcParallelEnable();
  DEF_GRAPH(g1) {
    CHAIN(NODE("const", CONSTANT)->NODE("getnext", GETNEXT)->NODE("relu", RELU)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto gelocal = CreateSubgraph("gelocal", graph, "const", "DNN_VM_GE_LOCAL", 0, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "getnext", "DNN_VM_AICPU", 1, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto gelocal2 = CreateSubgraph("gelocal2", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(gelocal, "end", aicpu, "placeholder");
  LinkSubGraph(aicpu, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", gelocal2, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {gelocal, aicpu, aicore, gelocal2};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 2);
  EXPECT_EQ(GetStreamId(graph, "const"), 0);
  EXPECT_EQ(GetStreamId(graph, "getnext"), 1);
  EXPECT_EQ(GetStreamId(graph, "relu"), 0);
  EXPECT_EQ(GetStreamId(graph, "output"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "getnext"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "relu"), 1);
  AcParallelDisable();
}

TEST_F(UtestDynamicStreamAllocator, invalid_ac_parallel_enable) {
  InvalidAcParallelEnable();
  DEF_GRAPH(g1) {
    CHAIN(NODE("const", CONSTANT)->NODE("getnext", GETNEXT)->NODE("relu", RELU)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto gelocal = CreateSubgraph("gelocal", graph, "const", "DNN_VM_GE_LOCAL", 0, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "getnext", "DNN_VM_AICPU", 1, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto gelocal2 = CreateSubgraph("gelocal2", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(gelocal, "end", aicpu, "placeholder");
  LinkSubGraph(aicpu, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", gelocal2, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {gelocal, aicpu, aicore, gelocal2};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_NE(ret, SUCCESS);
  AcParallelDisable();
}

// aicpu can not reuse input and can assign a stream
TEST_F(UtestDynamicStreamAllocator, aicore_gelocal_aicpu_ac_parallel_enable) {
  AcParallelEnable();
  DEF_GRAPH(g1) {
    CTRL_CHAIN(NODE("relu", RELU)->NODE("const", CONSTANT)->NODE("where", WHERE)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 0, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "const", "DNN_VM_GE_LOCAL", 1, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "where", "DNN_VM_AICPU", 1, 1);
  auto gelocal2 = CreateSubgraph("gelocal2", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(aicore, "end", gelocal, "placeholder");
  LinkSubGraph(gelocal, "end", aicpu, "placeholder");
  LinkSubGraph(aicpu, "end", gelocal2, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicore, gelocal, aicpu, gelocal2};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 2);
  EXPECT_EQ(GetStreamId(graph, "relu"), 0);
  EXPECT_EQ(GetStreamId(graph, "const"), 0);
  EXPECT_EQ(GetStreamId(graph, "where"), 1);
  EXPECT_EQ(GetStreamId(graph, "output"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "const"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "where"), 1);
  EXPECT_EQ(GetSendEventNum(graph, "where"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "output"), 1);
  AcParallelDisable();
}

// aicpu not in white list can attach to aicore
TEST_F(UtestDynamicStreamAllocator, aicore_aicpu_single_stream) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("relu", RELU)->EDGE(0, 0)->NODE("where", WHERE)->EDGE(0, 0)->NODE("output", NETOUTPUT));
    CHAIN(NODE("relu", RELU)->EDGE(1, 0)->NODE("relu2", RELU)->EDGE(0, 1)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 0, 2);
  auto aicore2 = CreateSubgraph("aicore2", graph, "relu2", "AIcoreEngine", 1, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "where", "DNN_VM_AICPU", 1, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "output", "DNN_VM_GE_LOCAL", 2, 0);

  LinkSubGraph(aicore, "end1", aicpu, "placeholder");
  LinkSubGraph(aicore, "end2", aicore2, "placeholder");
  LinkSubGraph(aicpu, "end", gelocal, "placeholder1");
  LinkSubGraph(aicore2, "end", gelocal, "placeholder2");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicore, aicore2, aicpu, gelocal};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 1);
  EXPECT_EQ(allocator.GetEventNum(), 0);
}

// aicpu can not attach to aicore when ac parallel enable
TEST_F(UtestDynamicStreamAllocator, aicore_aicpu_ac_parallel_enable_multi_stream) {
  AcParallelEnable();
  DEF_GRAPH(g1) {
    CHAIN(NODE("relu", RELU)->EDGE(0, 0)->NODE("where", WHERE)->EDGE(0, 0)->NODE("output", NETOUTPUT));
    CHAIN(NODE("relu", RELU)->EDGE(1, 0)->NODE("relu2", RELU)->EDGE(0, 1)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 0, 2);
  auto aicore2 = CreateSubgraph("aicore2", graph, "relu2", "AIcoreEngine", 1, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "where", "DNN_VM_AICPU", 1, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "output", "DNN_VM_GE_LOCAL", 2, 0);

  LinkSubGraph(aicore, "end1", aicpu, "placeholder");
  LinkSubGraph(aicore, "end2", aicore2, "placeholder");
  LinkSubGraph(aicpu, "end", gelocal, "placeholder1");
  LinkSubGraph(aicore2, "end", gelocal, "placeholder2");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicore, aicore2, aicpu, gelocal};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 2);
  EXPECT_EQ(GetStreamId(graph, "relu"), 0);
  EXPECT_EQ(GetStreamId(graph, "relu2"), 0);
  EXPECT_EQ(GetStreamId(graph, "where"), 1);
  EXPECT_EQ(GetStreamId(graph, "output"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "relu"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "where"), 1);
  EXPECT_EQ(GetSendEventNum(graph, "where"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "output"), 1);
  AcParallelDisable();
}

// aicpu can attach backwards to aicore
TEST_F(UtestDynamicStreamAllocator, aicpu_aicore_single_stream) {
  DEF_GRAPH(g1) { CHAIN(NODE("where", WHERE)->NODE("relu", RELU)->NODE("output", NETOUTPUT)); };
  auto graph = ToComputeGraph(g1);

  auto aicpu = CreateSubgraph("aicpu", graph, "where", "DNN_VM_AICPU", 0, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(aicpu, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", gelocal, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicpu, aicore, gelocal};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 1);
  EXPECT_EQ(allocator.GetEventNum(), 0);
}

// both aicpu and gelocal can attach to aicore
TEST_F(UtestDynamicStreamAllocator, aicore_aicpu_gelocal_ac_parallel_enable_single_stream) {
  AcParallelEnable();
  DEF_GRAPH(g1) {
    CHAIN(NODE("relu", RELU)->EDGE(0, 0)->NODE("where", WHERE)->EDGE(0, 0)->NODE("output", NETOUTPUT));
    CHAIN(NODE("relu", RELU)->EDGE(1, 0)->NODE("noop", NOOP)->EDGE(0, 1)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 0, 2);
  auto gelocal = CreateSubgraph("gelocal", graph, "noop", "DNN_VM_GE_LOCAL", 1, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "where", "DNN_VM_AICPU", 1, 1);
  auto gelocal2 = CreateSubgraph("gelocal2", graph, "output", "DNN_VM_GE_LOCAL", 2, 0);

  LinkSubGraph(aicore, "end1", aicpu, "placeholder");
  LinkSubGraph(aicore, "end2", gelocal, "placeholder");
  LinkSubGraph(aicpu, "end", gelocal2, "placeholder1");
  LinkSubGraph(gelocal, "end", gelocal2, "placeholder2");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicore, aicpu, gelocal, gelocal2};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 1);
  EXPECT_EQ(allocator.GetEventNum(), 0);
  AcParallelDisable();
}

// gelocal will attach to aicore rather than dsa
TEST_F(UtestDynamicStreamAllocator, dsa_aicore_gelocal_multi_stream) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("dsa", "DSARandomUniform")->EDGE(0, 0)->NODE("reshape", RESHAPE));
    CHAIN(NODE("relu", RELU)->EDGE(0, 1)->NODE("reshape"));
  };
  auto graph = ToComputeGraph(g1);

  auto dsa = CreateSubgraph("dsa", graph, "dsa", "DSAEngine", 0, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "reshape", "DNN_VM_GE_LOCAL", 2, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 0, 1);

  LinkSubGraph(dsa, "end", gelocal, "placeholder1");
  LinkSubGraph(aicore, "end", gelocal, "placeholder2");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {dsa, gelocal, aicore};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 1);
  EXPECT_EQ(GetStreamId(graph, "dsa"), 1);
  EXPECT_EQ(GetStreamId(graph, "reshape"), 0);
  EXPECT_EQ(GetStreamId(graph, "relu"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "dsa"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "reshape"), 1);
}

// aicpu can not attach to hccl
TEST_F(UtestDynamicStreamAllocator, hccl_aicpu_multi_stream) {
  DEF_GRAPH(g1) { CHAIN(NODE("allreduce", HCOMALLREDUCE)->NODE("where", WHERE)); };
  auto graph = ToComputeGraph(g1);

  auto hccl = CreateSubgraph("hccl", graph, "allreduce", "DNN_HCCL", 0, 1);
  auto aicpu = CreateSubgraph("aicpu", graph, "where", "DNN_VM_AICPU", 1, 0);

  LinkSubGraph(hccl, "end", aicpu, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {hccl, aicpu};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 1);
  EXPECT_EQ(GetStreamId(graph, "allreduce"), 0);
  EXPECT_EQ(GetStreamId(graph, "where"), 1);
  EXPECT_EQ(GetSendEventNum(graph, "allreduce"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "where"), 1);
}

// gelocal can attach to hccl
TEST_F(UtestDynamicStreamAllocator, gelocal_hccl_multi_stream) {
  DEF_GRAPH(g1) { CHAIN(NODE("const", CONSTANT)->NODE("allreduce", HCOMALLREDUCE)); };
  auto graph = ToComputeGraph(g1);

  auto gelocal = CreateSubgraph("gelocal", graph, "const", "DNN_VM_GE_LOCAL", 0, 1);
  auto hccl = CreateSubgraph("hccl", graph, "allreduce", "DNN_HCCL", 1, 0);

  LinkSubGraph(gelocal, "end", hccl, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {gelocal, hccl};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 1);
  EXPECT_EQ(allocator.GetEventNum(), 0);
}

// assign ffts+ to main stream 0
TEST_F(UtestDynamicStreamAllocator, ffts_aicore_single_stream) {
  DEF_GRAPH(g1) { CHAIN(NODE("where", WHERE)->NODE("relu", RELU)->NODE("output", NETOUTPUT)); };
  auto graph = ToComputeGraph(g1);

  auto ffts = CreateSubgraph("ffts", graph, "where", "ffts_plus", 0, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(ffts, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", gelocal, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {ffts, aicore, gelocal};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 1);
  EXPECT_EQ(allocator.GetEventNum(), 0);
}

// assign If to main stream 0
TEST_F(UtestDynamicStreamAllocator, aicore_if_single_stream) {
  DEF_GRAPH(sub_then) { CHAIN(NODE("data_then", DATA)->NODE("relu_then", RELU)->NODE("output_then", NETOUTPUT)); };
  DEF_GRAPH(sub_else) { CHAIN(NODE("data_else", DATA)->NODE("relu_else", RELU)->NODE("output_else", NETOUTPUT)); };
  DEF_GRAPH(g1) { CHAIN(NODE("relu", RELU)->NODE("if", IF, sub_then, sub_else)->NODE("output", NETOUTPUT)); };
  auto graph = ToComputeGraph(g1);
  auto graph_then = graph->GetSubgraph("sub_then");
  auto graph_else = graph->GetSubgraph("sub_else");
  for (const auto &subgraph : graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }

  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 0, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "if", "DNN_VM_GE_LOCAL", 1, 1);
  auto gelocal_output = CreateSubgraph("gelocal_output", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);
  auto data_then = CreateSubgraph("data_then", graph_then, "data_then", "DNN_VM_GE_LOCAL", 0, 1);
  auto aicore_then = CreateSubgraph("aicore_then", graph_then, "relu_then", "AIcoreEngine", 1, 1);
  auto out_then = CreateSubgraph("out_then", graph_then, "output_then", "DNN_VM_GE_LOCAL", 1, 0);
  auto data_else = CreateSubgraph("data_else", graph_else, "data_else", "DNN_VM_GE_LOCAL", 0, 1);
  auto aicore_else = CreateSubgraph("aicore_else", graph_else, "relu_else", "AIcoreEngine", 1, 1);
  auto out_else = CreateSubgraph("out_else", graph_else, "output_else", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(aicore, "end", gelocal, "placeholder");
  LinkSubGraph(gelocal, "end", gelocal_output, "placeholder");
  LinkSubGraph(data_then, "end", aicore_then, "placeholder");
  LinkSubGraph(aicore_then, "end", out_then, "placeholder");
  LinkSubGraph(data_else, "end", aicore_else, "placeholder");
  LinkSubGraph(aicore_else, "end", out_else, "placeholder");

  Graph2SubGraphInfoList subgraph_map = CreateSubgraphMap(
      graph,
      {{aicore, gelocal, gelocal_output}, {data_then, aicore_then, out_then}, {data_else, aicore_else, out_else}});

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 1);
  EXPECT_EQ(allocator.GetEventNum(), 0);
  EXPECT_EQ(GetStreamId(graph, "if"), 0);
  EXPECT_EQ(GetStreamId(graph_then, "relu_then"), 0);
  EXPECT_EQ(GetStreamId(graph_else, "relu_else"), 0);
}

// hccl in subgraph should assign a stream
TEST_F(UtestDynamicStreamAllocator, aicore_if_hccl_multi_stream) {
  DEF_GRAPH(sub_then) {
    CHAIN(NODE("data_then", DATA)->NODE("allreduce_then", HCOMALLREDUCE)->NODE("output_then", NETOUTPUT));
  };
  DEF_GRAPH(sub_else) {
    CHAIN(NODE("data_else", DATA)->NODE("allreduce_else", HCOMALLREDUCE)->NODE("output_else", NETOUTPUT));
  };
  DEF_GRAPH(g1) { CHAIN(NODE("relu", RELU)->NODE("if", IF, sub_then, sub_else)->NODE("output", NETOUTPUT)); };
  auto graph = ToComputeGraph(g1);
  auto graph_then = graph->GetSubgraph("sub_then");
  auto graph_else = graph->GetSubgraph("sub_else");
  for (const auto &subgraph : graph->GetAllSubgraphs()) {
    subgraph->SetGraphUnknownFlag(true);
  }

  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 0, 1);
  auto gelocal = CreateSubgraph("gelocal", graph, "if", "DNN_VM_GE_LOCAL", 1, 1);
  auto gelocal_output = CreateSubgraph("gelocal_output", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);
  auto data_then = CreateSubgraph("data_then", graph_then, "data_then", "DNN_VM_GE_LOCAL", 0, 1);
  auto hccl_then = CreateSubgraph("hccl_then", graph_then, "allreduce_then", "DNN_HCCL", 1, 1);
  auto out_then = CreateSubgraph("out_then", graph_then, "output_then", "DNN_VM_GE_LOCAL", 1, 0);
  auto data_else = CreateSubgraph("data_else", graph_else, "data_else", "DNN_VM_GE_LOCAL", 0, 1);
  auto hccl_else = CreateSubgraph("hccl_else", graph_else, "allreduce_else", "DNN_HCCL", 1, 1);
  auto out_else = CreateSubgraph("out_else", graph_else, "output_else", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(aicore, "end", gelocal, "placeholder");
  LinkSubGraph(gelocal, "end", gelocal_output, "placeholder");
  LinkSubGraph(data_then, "end", hccl_then, "placeholder");
  LinkSubGraph(hccl_then, "end", out_then, "placeholder");
  LinkSubGraph(data_else, "end", hccl_else, "placeholder");
  LinkSubGraph(hccl_else, "end", out_else, "placeholder");

  Graph2SubGraphInfoList subgraph_map = CreateSubgraphMap(
      graph, {{aicore, gelocal, gelocal_output}, {data_then, hccl_then, out_then}, {data_else, hccl_else, out_else}});

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 2);
  EXPECT_EQ(allocator.GetEventNum(), 4);
  EXPECT_EQ(GetStreamId(graph, "if"), 0);
  EXPECT_EQ(GetStreamId(graph_then, "allreduce_then"), 1);
  EXPECT_EQ(GetStreamId(graph_else, "allreduce_else"), 1);
  EXPECT_EQ(GetRecvEventNum(graph_then, "allreduce_then"), 1);
  EXPECT_EQ(GetSendEventNum(graph_then, "allreduce_then"), 1);
  EXPECT_EQ(GetRecvEventNum(graph_else, "allreduce_else"), 1);
  EXPECT_EQ(GetSendEventNum(graph_else, "allreduce_else"), 1);
}

// ac_parallel_enable can only be "", "0" and "1"
TEST_F(UtestDynamicStreamAllocator, ac_parallel_enable_invalid) {
  std::map<std::string, std::string> options;
  options[AC_PARALLEL_ENABLE] = "2";
  GetThreadLocalContext().SetGraphOption(options);

  DynamicStreamAllocator allocator;
  auto ret = allocator.GetAcParallelEnableConfig();
  EXPECT_NE(ret, SUCCESS);

  AcParallelDisable();
}

TEST_F(UtestDynamicStreamAllocator, only_gelocal) {
  DEF_GRAPH(g1) { CHAIN(NODE("const", CONSTANT)); };
  auto graph = ToComputeGraph(g1);
  auto gelocal = CreateSubgraph("gelocal", graph, "const", "DNN_VM_GE_LOCAL", 0, 1);

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {gelocal};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 1);
  EXPECT_EQ(allocator.GetEventNum(), 0);
  EXPECT_EQ(GetStreamId(graph, "const"), 0);
}

// nodes with same stream_label can be assigned to same stream
TEST_F(UtestDynamicStreamAllocator, aicore_dsa_hccl_stream_label) {
  DEF_GRAPH(g1) {
    auto relu = OP_CFG(RELU).Attr(ATTR_NAME_STREAM_LABEL, "aaa");
    auto allreduce = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_STREAM_LABEL, "aaa");
    CHAIN(NODE("data1", DATA)
              ->NODE("relu", relu)
              ->NODE("allreduce", allreduce)
              ->NODE("dsa", "DSARandomUniform")
              ->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);

  auto gelocal = CreateSubgraph("gelocal", graph, "data1", "DNN_VM_GE_LOCAL", 0, 1);
  auto aicore = CreateSubgraph("aicore", graph, "relu", "AIcoreEngine", 1, 1);
  auto hccl = CreateSubgraph("hccl", graph, "allreduce", "DNN_HCCL", 1, 1);
  auto dsa = CreateSubgraph("dsa", graph, "dsa", "DSAEngine", 1, 1);
  auto gelocal2 = CreateSubgraph("gelocal2", graph, "output", "DNN_VM_GE_LOCAL", 1, 0);

  LinkSubGraph(gelocal, "end", aicore, "placeholder");
  LinkSubGraph(aicore, "end", hccl, "placeholder");
  LinkSubGraph(hccl, "end", dsa, "placeholder");
  LinkSubGraph(dsa, "end", gelocal2, "placeholder");

  Graph2SubGraphInfoList subgraph_map;
  subgraph_map[graph] = {aicore, hccl, dsa, gelocal, gelocal2};

  DynamicStreamAllocator allocator;
  auto ret = allocator.AssignStreamsForDynamicShapeGraph(graph, subgraph_map);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(allocator.GetStreamNum(), 3);
  EXPECT_EQ(allocator.GetEventNum(), 3);
  EXPECT_EQ(GetStreamId(graph, "data1"), 0);
  EXPECT_EQ(GetStreamId(graph, "relu"), 2);
  EXPECT_EQ(GetStreamId(graph, "allreduce"), 2);
  EXPECT_EQ(GetStreamId(graph, "dsa"), 1);
  EXPECT_EQ(GetStreamId(graph, "output"), 0);
  EXPECT_EQ(GetSendEventNum(graph, "data1"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "relu"), 1);
  EXPECT_EQ(GetSendEventNum(graph, "allreduce"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "dsa"), 1);
  EXPECT_EQ(GetSendEventNum(graph, "dsa"), 1);
  EXPECT_EQ(GetRecvEventNum(graph, "output"), 1);
}

TEST_F(UtestDynamicStreamAllocator, dynamic_subgraph_alloc_attached_resource) {
  DEF_GRAPH(sub_then) {
    CHAIN(NODE("data_then", DATA)->NODE("relu_then", RELU)->NODE("output_then", NETOUTPUT));
  };
  DEF_GRAPH(sub_else) {
    CHAIN(NODE("data_else", DATA)->NODE("relu_else", RELU)->NODE("output_else", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("relu", RELU)->NODE("if", IF, sub_then, sub_else)->NODE("output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  auto graph_then = graph->GetSubgraph("sub_then");
  graph_then->SetGraphUnknownFlag(false);
  auto graph_else = graph->GetSubgraph("sub_else");
  graph_else->SetGraphUnknownFlag(true);

  auto relu1 = graph_then->FindNode("relu_then");
  ASSERT_NE(relu1, nullptr);
  auto relu2 = graph_else->FindNode("relu_else");
  ASSERT_NE(relu2, nullptr);
  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), "group", "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), "group", "group0"));


  NamedAttrs stream_attrs1;
  EXPECT_TRUE(AttrUtils::SetStr(stream_attrs1, ATTR_NAME_ATTACHED_STREAM_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(stream_attrs1, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, stream_attrs1));
  NamedAttrs notify_attrs1;
  EXPECT_TRUE(AttrUtils::SetStr(notify_attrs1, ATTR_NAME_ATTACHED_NOTIFY_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(notify_attrs1, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, notify_attrs1));

  NamedAttrs stream_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(stream_attrs2, ATTR_NAME_ATTACHED_STREAM_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(stream_attrs2, ATTR_NAME_ATTACHED_STREAM_KEY, "as2"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, stream_attrs2));
  NamedAttrs notify_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(notify_attrs2, ATTR_NAME_ATTACHED_NOTIFY_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(notify_attrs2, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as2"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, notify_attrs2));

  DynamicStreamAllocator allocator;
  int64_t stream_num{0};
  int64_t event_num{0};
  int64_t notify_num{0};
  vector<uint32_t> notify_type{0};
  EXPECT_EQ(allocator.AssignAttachedResource(graph, stream_num, event_num, notify_num, notify_type), SUCCESS);
  EXPECT_EQ(stream_num, 1);
  EXPECT_EQ(notify_num, 1);
  EXPECT_EQ(event_num, 0);

}
}  // namespace ge
