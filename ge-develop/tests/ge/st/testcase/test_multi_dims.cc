/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "api/atc/main_impl.h"
#include "ge_running_env/path_utils.h"
#include "ge_running_env/atc_utils.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_graph_optimizer.h"

#include "utils/model_factory.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/graph_utils.h"
#include "types.h"
#include "init_ge.h"
#include "graph/passes/multi_batch/subgraph_const_migration_pass.h"
#include "graph/passes/multi_batch/subexpression_migration_pass.h"
#include "graph/passes/pass_manager.h"
#include "framework/omg/omg.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/weights_parser.h"

namespace ge {
class MultiDimsSTest : public AtcTest {};

namespace {
void FakeMultiDimsEngine(GeRunningEnvFaker &ge_env) {
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.InstallDefault();
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
}
}

TEST_F(MultiDimsSTest, MultiBatch) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "ms_1");
  auto path = ModelFactory::GenerateModel_1();
  std::string model_arg = "--model="+path;
  std::string output_arg = "--output="+om_path;
  char *argv[] = {"atc",
                  const_cast<char *>(model_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--framework=1", // FrameworkType
                  "--out_nodes=relu:0",
                  "--soc_version=Ascend310",
                  "--input_format=NCHW",
                  "--output_type=FP32",
                  "--virtual_type=0",
                  "--input_shape=data1:-1,3,16,16",
                  "--dynamic_batch_size=1,2,4,8",
                  "--input_shape_range=",
                  };
  DUMP_GRAPH_WHEN("PreRunAfterPrepare")
  auto ret = main_impl(sizeof(argv)/sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  ReInitGe(); // the main_impl will call GEFinalize, so re-init after call it
  /*
   *
   * root graph:
   *                      netoutput
   *                         |
   *         -------------  case
   *        /     /             \
   *      /     /              mapindex
   *    /     /                 /    \
   * data1 const1   data_multibatch const_multibatch
   *
   *
   *
   * subgraphs:
   *
   *     netoutput
   *        |
   *       relu
   *        |
   *      conv2d
   *      /   \
   *    data0 data1
   */
  CHECK_GRAPH(PreRunAfterPrepare) {
    // mmSetEnv("DUMP_GE_GRAPH", "1", 1);
    // GraphUtils::DumpGEGraphToOnnx(*graph, "SnDebugPrepare");
    // GraphUtils::DumpGEGraphToOnnx(*(graph->GetAllSubgraphs()[0]), "SnDebugPrepareSub0");

    EXPECT_EQ(graph->GetDirectNodesSize(), 7);
    EXPECT_EQ(graph->GetAllNodesSize(), 7 + 5 * 4);
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 4);

    // todo check topo
    // todo check shape in every subgraphs
    // todo check the max shape
  };

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(MultiDimsSTest, MultiBatch_empty_tensor) {
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "ms_1");
  auto path = ModelFactory::GenerateModel_1();
  std::string model_arg = "--model="+path;
  std::string output_arg = "--output="+om_path;
  char *argv[] = {"atc",
                  const_cast<char *>(model_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--framework=1", // FrameworkType
                  "--out_nodes=relu:0",
                  "--soc_version=Ascend310",
                  "--input_format=NCHW",
                  "--output_type=FP32",
                  "--virtual_type=0",
                  "--input_shape=data1:-1,3,16,16",
                  "--dynamic_batch_size=0,2,4,8",
                  "--input_shape_range=",
  };
  DUMP_GRAPH_WHEN("PreRunAfterPrepare")
  auto ret = main_impl(sizeof(argv)/sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  ReInitGe(); // the main_impl will call GEFinalize, so re-init after call it

  CHECK_GRAPH(PreRunAfterPrepare) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 7);
    EXPECT_EQ(graph->GetAllNodesSize(), 7 + 5 * 4);
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 4);
  };

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(MultiDimsSTest, MultiBatch_switch) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  GeRunningEnvFaker ge_env;
  FakeMultiDimsEngine(ge_env);

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "ms_3");
  auto path = ModelFactory::GenerateModel_switch();
  std::string model_arg = "--model="+path;
  std::string output_arg = "--output="+om_path;
  char *argv[] = {"atc",
                  const_cast<char *>(model_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--framework=1", // FrameworkType
                  "--soc_version=Ascend310",
                  "--input_format=ND",
                  "--input_shape=data1:-1,3,4,5;data2:1",
                  "--dynamic_dims=1;2;4;8",
                  };
  auto ret = main_impl(sizeof(argv)/sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  CHECK_GRAPH(PreRunAfterBuild) {
    std::set<uint32_t> events;
    for (const auto &node: graph->GetAllNodes()) {
      if (node->GetType() == "Send") {
        const auto &op_desc = node->GetOpDesc();
        uint32_t event_id = 0;
        AttrUtils::GetInt(op_desc, "event_id", event_id);
        events.emplace(event_id);
      }
    }
    EXPECT_TRUE(events.size() >= 1 && events.size() <= 2);
  };
  ge_env.Reset();
  ge_env.InstallDefault();
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

NodePtr MakeNode(const ComputeGraphPtr &graph, int in_num, int out_num, string name, string type) {
  GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (auto i = 0; i < in_num; ++i) {
    op_desc->AddInputDesc(test_desc);
  }
  for (auto i = 0; i < out_num; ++i) {
    op_desc->AddOutputDesc(test_desc);
  }
  if (type == "Const") {
    uint64_t const_value = 101;
    auto weight = std::make_shared<GeTensor>(op_desc->GetOutputDesc(0), (uint8_t *)&const_value, sizeof(uint64_t));
    AttrUtils::SetTensor(op_desc, ge::ATTR_NAME_WEIGHTS, weight);
  }
  return graph->AddNode(op_desc);
}

void make_original_graph(const ComputeGraphPtr &graph) {
  auto data = MakeNode(graph, 1, 1, "data", "Data");
  {
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  }
  auto const1 = MakeNode(graph, 0, 1, "const1", "Const");
  {
    auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
    GraphUtils::AddEdge(data1->GetOutControlAnchor(), const1->GetInControlAnchor());
    auto cast_node1 = MakeNode(graph, 1, 1, "cast1", "Cast");
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), cast_node1->GetInDataAnchor(0));
  }

  auto const2 = MakeNode(graph, 0, 1, "const2", "Const");
  {
    auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);
    GraphUtils::AddEdge(data2->GetOutControlAnchor(), const2->GetInControlAnchor());
  }

  auto cast_node2 = MakeNode(graph, 1, 1, "cast2", "Cast");
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), cast_node2->GetInDataAnchor(0));
  auto cast_node3 = MakeNode(graph, 1, 1, "cast3", "Cast");
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), cast_node3->GetInDataAnchor(0));

  auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
  GraphUtils::AddEdge(cast_node2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));

  auto out = MakeNode(graph, 1, 0, "out1", "NetOutput");
  GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), out->GetInDataAnchor(0));
}

void make_multibatch_graph(const ComputeGraphPtr &graph) {
  auto index = MakeNode(graph, 1, 1, "index", "Data");
  auto data = MakeNode(graph, 1, 1, "data", "Data");
  auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
  auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
  AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);

  auto case1 = MakeNode(graph, 4, 1, "case", "Case");
  GraphUtils::AddEdge(index->GetOutDataAnchor(0), case1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), case1->GetInDataAnchor(1));
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), case1->GetInDataAnchor(2));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), case1->GetInDataAnchor(3));
  auto output_node = MakeNode(graph, 1, 0, "output", "NetOutput");
  GraphUtils::AddEdge(case1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

  AttrUtils::SetInt(case1->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
  case1->GetOpDesc()->RegisterSubgraphIrName("branches", kDynamic);
  ComputeGraphPtr branch = std::make_shared<ComputeGraph>("test_branch");
  make_original_graph(branch);
  for (int i = 0; i < 2; ++i) {
    std::string name("_ascend_mbatch_batch_" + std::to_string(i));
    std::vector<NodePtr> input_nodes;
    std::vector<NodePtr> output_nodes;
    ComputeGraphPtr subgraph = GraphUtils::CloneGraph(branch, name, input_nodes, output_nodes);

    subgraph->SetName(name);
    subgraph->SetParentNode(case1);
    subgraph->SetParentGraph(graph);
    std::string host_node_name = "cast2" + name;
    auto host_node = subgraph->FindNode(host_node_name);
    host_node->SetHostNode(true);
    graph->AddSubgraph(subgraph->GetName(), subgraph);

    case1->GetOpDesc()->AddSubgraphName(name);
    case1->GetOpDesc()->SetSubgraphInstanceName(i, subgraph->GetName());
  }
}

TEST_F(MultiDimsSTest, test_pass_migration) {
  PassManager pass_manager;
  pass_manager.AddPass("SubexpressionMigrationPass", new (std::nothrow) SubexpressionMigrationPass);
  pass_manager.AddPass("SubgraphConstMigrationPass", new (std::nothrow) SubgraphConstMigrationPass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_multibatch_graph(graph);
  auto ret = pass_manager.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}

using GetGraphCallback = std::function<std::unique_ptr<google::protobuf::Message>(
  const google::protobuf::Message *root_proto, const std::string &graph)>;
class CaffeModelParser : public domi::ModelParser {
  public:
    CaffeModelParser(){}
    ~CaffeModelParser(){}
    domi::Status Parse(const char *file, ge::Graph &graph){
      (void)file;
      std::vector<int64_t> shape{8, 4, -1};
      DEF_GRAPH(test_graph) {
        auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Attr(ATTR_NAME_INDEX, 0);
        auto square = OP_CFG(ADD).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, shape);
        auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(0).TensorDesc(FORMAT_ND, DT_FLOAT, shape);
        CHAIN(NODE("Data", data)->NODE("Square", square)->NODE("NetOutput", netoutput));
      };
      auto cgp = ToComputeGraph(test_graph);

      graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
      return SUCCESS;
    }
    domi::Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph){return SUCCESS;}
    domi::Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph){return SUCCESS;}
    domi::Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph){return SUCCESS;}
    domi::Status ParseProtoWithSubgraph(const google::protobuf::Message *proto, GetGraphCallback callback,
                                              ge::ComputeGraphPtr &graph){return SUCCESS;}
    ge::DataType ConvertToGeDataType(const uint32_t type){return DT_FLOAT;}
    domi::Status ParseAllGraph(const google::protobuf::Message *root_proto,
                                              ge::ComputeGraphPtr &root_graph){return SUCCESS;}
};

class CaffeWeightsParser : public domi::WeightsParser {
  public:
    CaffeWeightsParser(){}
    ~CaffeWeightsParser(){}
    domi::Status Parse(const char *file, ge::Graph &graph){return SUCCESS;}
    domi::Status ParseFromMemory(const char *input, uint32_t lengt, ge::ComputeGraphPtr &graph){return SUCCESS;}
};

std::shared_ptr<domi::ModelParser> FuncTest() {
  std::shared_ptr<domi::ModelParser> ptr = std::make_shared<CaffeModelParser>();
  return ptr;
}

std::shared_ptr<domi::WeightsParser> WeightsFuncTest() {
  std::shared_ptr<domi::WeightsParser> ptr = std::make_shared<CaffeWeightsParser>();
  return ptr;
}

TEST_F(MultiDimsSTest, test_parse_graph) {
  std::vector<int64_t> shape{8, 4, -1};
  DEF_GRAPH(test_graph) {
    auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Attr(ATTR_NAME_INDEX, 0);
    auto square = OP_CFG(ADD).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, shape);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(0).TensorDesc(FORMAT_ND, DT_FLOAT, shape);
    CHAIN(NODE("Data", data)->NODE("Square", square)->NODE("NetOutput", netoutput));
  };
  auto cgp = ToComputeGraph(test_graph);

  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  graph.SaveToFile("./ut_graph2.txt");

  std::map<std::string, std::string> atc_params = {{"in_nodes", ""}};

  const char *model_file = "./ut_graph1.txt";
  const char *weights_file = "./ut_graph2.txt";

  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->RegisterCreator(type, FuncTest);
  domi::ModelParserFactory::Instance()->creator_map_[type] = FuncTest;
  domi::WeightsParserFactory::Instance()->creator_map_[type] = WeightsFuncTest;

  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool is_dynamic_input = false;
  Status ret = ParseGraph(graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  domi::WeightsParserFactory::Instance()->creator_map_.clear();
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
}
}
