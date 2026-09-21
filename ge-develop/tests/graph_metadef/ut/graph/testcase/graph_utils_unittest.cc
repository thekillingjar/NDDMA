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

#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/node.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "mmpa/mmpa_api.h"
#include "common/util/mem_utils.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "graph/utils/ge_dump_graph_whitelist.h"

namespace ge {
extern std::stringstream GetFilePathWhenDumpPathSet(const string &ascend_work_path);
namespace {
bool IfNodeExist(const ComputeGraphPtr &graph, std::function<bool(const NodePtr &)> filter,
                 bool direct_node_flag = true) {
  for (const auto &node : graph->GetNodes(direct_node_flag)) {
    if (filter(node)) {
      return true;
    }
  }
  return false;
}
/*
 *             data1  const1         data2  const2
 *                \    /                \    /
 *                 add1                  add2
 *                   |                    |
 *                 cast1                cast2
 *                   |                    |
 *                square1  var1  var2  square2
 *                     \   /  |  |  \   /
 *                     less1  |  |  less2
 *                          \ |  | /
 *                            mul
 *                             |
 *                             |
 *                             |
 *                          netoutput
 */
void BuildGraphForUnfold(ComputeGraphPtr &graph, ComputeGraphPtr &subgraph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);

  graph = builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &data1 = sub_builder.AddNode("data1", DATA, 1, 1);
  const auto &const1 = sub_builder.AddNode("const1", CONSTANTOP, 0, 1);
  const auto &add1 = sub_builder.AddNode("add1", "Add", 2, 1);
  const auto &cast1 = sub_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &func1 = sub_builder.AddNode("func1", PARTITIONEDCALL, 2, 1);
  const auto &data2 = sub_builder.AddNode("data2", DATA, 1, 1);
  const auto &data3 = sub_builder.AddNode("data3", DATA, 1, 1);
  const auto &const2 = sub_builder.AddNode("const2", CONSTANTOP, 0, 1);
  const auto &add2 = sub_builder.AddNode("add2", "Add", 2, 1);
  const auto &cast2 = sub_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &func2 = sub_builder.AddNode("func2", PARTITIONEDCALL, 2, 1);
  const auto &data4 = sub_builder.AddNode("data4", DATA, 1, 1);
  const auto &mul = sub_builder.AddNode("mul", "Mul", 2, 1);
  const auto &netoutput0 = sub_builder.AddNode("netoutput0", NETOUTPUT, 1, 0);
  sub_builder.AddDataEdge(data1, 0, add1, 0);
  sub_builder.AddDataEdge(const1, 0, add1, 1);
  sub_builder.AddDataEdge(add1, 0, cast1, 0);
  sub_builder.AddDataEdge(cast1, 0, func1, 0);
  sub_builder.AddDataEdge(data2, 0, func1, 1);
  sub_builder.AddDataEdge(data3, 0, add2, 0);
  sub_builder.AddDataEdge(const2, 0, add2, 1);
  sub_builder.AddDataEdge(add2, 0, cast2, 0);
  sub_builder.AddDataEdge(cast2, 0, func2, 0);
  sub_builder.AddDataEdge(data4, 0, func2, 1);
  sub_builder.AddDataEdge(func1, 0, mul, 0);
  sub_builder.AddDataEdge(func2, 0, mul, 1);
  sub_builder.AddDataEdge(mul, 0, netoutput0, 0);

  subgraph = sub_builder.GetGraph();
  subgraph->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
  AttrUtils::SetInt(data4->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);
  AttrUtils::SetInt(netoutput0->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func->GetOpDesc()->AddSubgraphName("f");
  func->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  graph->AddSubGraph(subgraph);
  subgraph->SetParentNode(func);
  subgraph->SetParentGraph(graph);

  auto sub_sub_builder1 = ut::GraphBuilder("sub_sub1");
  const auto &data5 = sub_sub_builder1.AddNode("data5", DATA, 1, 1);
  const auto &data6 = sub_sub_builder1.AddNode("data6", DATA, 1, 1);
  const auto &square1 = sub_sub_builder1.AddNode("square1", "Square", 1, 1);
  const auto &less1 = sub_sub_builder1.AddNode("less1", "Less", 2, 1);
  const auto &netoutput1 = sub_sub_builder1.AddNode("netoutput1", NETOUTPUT, 1, 0);
  sub_sub_builder1.AddDataEdge(data5, 0, square1, 0);
  sub_sub_builder1.AddDataEdge(square1, 0, less1, 0);
  sub_sub_builder1.AddDataEdge(data6, 0, less1, 1);
  sub_sub_builder1.AddDataEdge(less1, 0, netoutput1, 0);

  const auto &sub_subgraph1 = sub_sub_builder1.GetGraph();
  sub_subgraph1->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data5->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data6->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func1->GetOpDesc()->AddSubgraphName("f");
  func1->GetOpDesc()->SetSubgraphInstanceName(0, sub_subgraph1->GetName());
  graph->AddSubGraph(sub_subgraph1);
  sub_subgraph1->SetParentNode(func1);
  sub_subgraph1->SetParentGraph(subgraph);

  auto sub_sub_builder2 = ut::GraphBuilder("sub_sub2");
  const auto &data7 = sub_sub_builder2.AddNode("data7", DATA, 1, 1);
  const auto &data8 = sub_sub_builder2.AddNode("data8", DATA, 1, 1);
  const auto &square2 = sub_sub_builder2.AddNode("square2", "Square", 1, 1);
  const auto &less2 = sub_sub_builder2.AddNode("less2", "Less", 2, 1);
  const auto &netoutput2 = sub_sub_builder2.AddNode("netoutput2", NETOUTPUT, 1, 0);
  sub_sub_builder2.AddDataEdge(data7, 0, square2, 0);
  sub_sub_builder2.AddDataEdge(square2, 0, less2, 0);
  sub_sub_builder2.AddDataEdge(data8, 0, less2, 1);
  sub_sub_builder2.AddDataEdge(less2, 0, netoutput2, 0);

  const auto &sub_subgraph2 = sub_sub_builder2.GetGraph();
  sub_subgraph2->SetGraphUnknownFlag(false);
  AttrUtils::SetInt(data7->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data8->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func2->GetOpDesc()->AddSubgraphName("f");
  func2->GetOpDesc()->SetSubgraphInstanceName(0, sub_subgraph2->GetName());
  graph->AddSubGraph(sub_subgraph2);
  sub_subgraph2->SetParentNode(func2);
  sub_subgraph2->SetParentGraph(subgraph);

  return;
}
/*                                   --------------
 *                                  |              |
 *             data1  const1     data2  const2     |
 *              |  \    /           \    /         |
 *              |   add1             add2          |
 *              |    |                 |           |
 *              |  cast1              cast2        |
 *              |    |                 |           |
 *              |    |                 |           |
 *              |     \               /            |
 *              \      ------  mul ------------------
 *               \              |
 *                \             |
 *                 \            |
 *                  ------- netoutput
 */
void BuildGraphForUnfoldWithControlEdge(ComputeGraphPtr &graph, ComputeGraphPtr &subgraph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(input2, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);

  graph = builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &data1 = sub_builder.AddNode("data1", DATA, 1, 1);
  const auto &const1 = sub_builder.AddNode("const1", CONSTANTOP, 0, 1);
  const auto &add1 = sub_builder.AddNode("add1", "Add", 2, 1);
  const auto &cast1 = sub_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &data2 = sub_builder.AddNode("data2", DATA, 1, 1);
  const auto &const2 = sub_builder.AddNode("const2", CONSTANTOP, 0, 1);
  const auto &add2 = sub_builder.AddNode("add2", "Add", 2, 1);
  const auto &cast2 = sub_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &mul = sub_builder.AddNode("mul", "Mul", 2, 1);
  const auto &netoutput0 = sub_builder.AddNode("netoutput0", NETOUTPUT, 1, 0);
  sub_builder.AddDataEdge(data1, 0, add1, 0);
  sub_builder.AddControlEdge(data1, netoutput0);
  sub_builder.AddDataEdge(const1, 0, add1, 1);
  sub_builder.AddDataEdge(add1, 0, cast1, 0);
  sub_builder.AddDataEdge(cast1, 0, mul, 0);
  sub_builder.AddControlEdge(data2, mul);
  sub_builder.AddDataEdge(data2, 0, add2, 0);
  sub_builder.AddDataEdge(const2, 0, add2, 1);
  sub_builder.AddDataEdge(add2, 0, cast2, 0);
  sub_builder.AddDataEdge(cast2, 0, mul, 1);
  sub_builder.AddDataEdge(mul, 0, netoutput0, 0);

  subgraph = sub_builder.GetGraph();
  subgraph->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput0->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func->GetOpDesc()->AddSubgraphName("f");
  func->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  graph->AddSubGraph(subgraph);
  subgraph->SetParentNode(func);
  subgraph->SetParentGraph(graph);
  return;
}

void BuildGraphWithPlaceholderAndEnd(ComputeGraphPtr &graph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("pld1", PLACEHOLDER, 1, 1);
  const auto &input2 = builder.AddNode("pld2", PLACEHOLDER, 1, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &end = builder.AddNode("end", END, 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  const auto &add2 = builder.AddNode("add2", "Add", 2, 1);
  const auto &add3 = builder.AddNode("add3", "Add", 2, 1);
  builder.AddDataEdge(input1, 0, add1, 0);
  builder.AddDataEdge(input2, 0, add1, 1);
  builder.AddDataEdge(data1, 0, add2, 0);
  builder.AddDataEdge(data2, 0, add2, 1);
  builder.AddDataEdge(add1, 0, add3, 0);
  builder.AddDataEdge(add2, 0, add3, 1);
  builder.AddDataEdge(add3, 0, end, 0);
  graph = builder.GetGraph();
  graph->AddOutputNode(end);
}

ComputeGraphPtr BuildGraphWithSubGraph() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data0 = root_builder.AddNode("data0", "Data", 1, 1);
  const auto &case0 = root_builder.AddNode("case0", "Case", 1, 1);
  const auto &relu0 = root_builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = root_builder.AddNode("relu1", "Relu", 1, 1);
  const auto &netoutput = root_builder.AddNode("netoutput", "NetOutput", 1, 1);
  const auto &root_graph = root_builder.GetGraph();
  root_builder.AddDataEdge(data0, 0, case0, 0);
  root_builder.AddDataEdge(case0, 0, relu0, 0);
  root_builder.AddDataEdge(relu0, 0, relu1, 0);
  root_builder.AddDataEdge(relu1, 0, netoutput, 0);

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &data1 = sub_builder1.AddNode("data1", "Data", 0, 1);
  const auto &sub1_netoutput = sub_builder1.AddNode("sub1_netoutput", "NetOutput", 1, 1);
  sub_builder1.AddDataEdge(data1, 0, sub1_netoutput, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = ut::GraphBuilder("sub2");
  const auto &data2 = sub_builder2.AddNode("data2", "Data", 0, 1);
  const auto &sub2_netoutput = sub_builder2.AddNode("sub2_netoutput", "NetOutput", 1, 1);
  sub_builder2.AddDataEdge(data2, 0, sub2_netoutput, 0);
  const auto &sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case0);
  sub_graph2->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch2");
  case0->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  root_graph->TopologicalSorting();
  return root_graph;
}

ComputeGraphPtr BuildGraphWithConst() {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  AttrUtils::SetStr(const_node->GetOpDesc(), "fake_attr_name", "fake_attr_value");
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  AttrUtils::SetStr(add_node->GetOpDesc(), "fake_attr_name", "fake_attr_value");
  AttrUtils::SetStr(add_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, "fake_attr_value");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  return builder.GetGraph();
}

std::string GetSpecificFilePath(const std::string &file_path, const string &suffix) {
  DIR *dir;
  struct dirent *ent;
  dir = opendir(file_path.c_str());
  if (dir == nullptr) {
    return "";
  }
  while ((ent = readdir(dir)) != nullptr) {
    if (strstr(ent->d_name, suffix.c_str()) != nullptr) {
      std::string d_name(ent->d_name);
      closedir(dir);
      return file_path + "/" + d_name;
    }
  }
  closedir(dir);
  return "";
}

ComputeGraphPtr BuildGraphForIsolateNode(const int in_data_num, const int out_data_num, const int in_ctrl_num,
                                         const int out_ctrl_num) {
  auto graph_builder = ut::GraphBuilder("graph");

  const auto &del_node = graph_builder.AddNode("del_node", "DelNode", in_data_num, out_data_num);

  for (int i = 0; i < in_data_num; ++i) {
    const auto &n = graph_builder.AddNode("in_node_" + std::to_string(i), "InNode", 1, 1);
    graph_builder.AddDataEdge(n, 0, del_node, i);
  }
  for (int i = 0; i < out_data_num; ++i) {
    const auto &n = graph_builder.AddNode("out_node_" + std::to_string(i), "OutNode", 1, 1);
    graph_builder.AddDataEdge(del_node, i, n, 0);
  }

  for (int i = 0; i < in_ctrl_num; ++i) {
    const auto &n = graph_builder.AddNode("in_ctrl_node_" + std::to_string(i), "InCtrlNode", 1, 1);
    graph_builder.AddControlEdge(n, del_node);
  }
  for (int i = 0; i < out_ctrl_num; ++i) {
    const auto &n = graph_builder.AddNode("out_ctrl_node_" + std::to_string(i), "OutCtrlNode", 1, 1);
    graph_builder.AddControlEdge(del_node, n);
  }
  return graph_builder.GetGraph();
}
}  // namespace

namespace {
class UtestComputeGraphBuilder : public ComputeGraphBuilder {
 public:
  virtual ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) {
    auto graph = std::make_shared<ComputeGraph>("test");
    auto op_desc = std::make_shared<OpDesc>("node", "node");
    NodePtr node = graph->AddNode(op_desc);
    std::map<std::string, NodePtr> node_names_;
    node_names_.insert(pair<std::string, NodePtr>("node", node));
    return graph;
  }

  NodePtr GetNode(const std::string &name);
  std::vector<NodePtr> GetAllNodes();
  void BuildNodes(graphStatus &error_code, std::string &error_msg);
};

NodePtr UtestComputeGraphBuilder::GetNode(const std::string &name) {
  return ComputeGraphBuilder::GetNode(name);
}

std::vector<NodePtr> UtestComputeGraphBuilder::GetAllNodes() {
  return ComputeGraphBuilder::GetAllNodes();
}

void UtestComputeGraphBuilder::BuildNodes(graphStatus &error_code, std::string &error_msg) {
  return ComputeGraphBuilder::BuildNodes(error_code, error_msg);
}

} // namespace

class UtestGraphUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
  }
};

TEST_F(UtestGraphUtils, DumpGEGraphUserGraphNameNull_AscendWorkPathNotNull) {
  auto graph = BuildGraphWithConst();
  std::string ascend_work_path = "./test_ge_graph_path";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  GraphUtils::DumpGEGraph(graph, "", true, "");
  ComputeGraphPtr com_graph = std::make_shared<ComputeGraph>("GeTestGraph");

  // test load
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(ascend_work_path);
  std::string dump_graph_path = GetSpecificFilePath(ge::RealPath(dump_file_path.str().c_str()), "_.txt");
  auto state = GraphUtils::LoadGEGraph(dump_graph_path.c_str(), *com_graph);
  ASSERT_EQ(state, true);
  unsetenv("DUMP_GRAPH_PATH");
  unsetenv("DUMP_GE_GRAPH");
  system(("rm -rf " + ascend_work_path).c_str());
}

TEST_F(UtestGraphUtils, DumpGEGraphToOnnxNotAlways) {
  unsetenv("DUMP_GRAPH_PATH");
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void) setenv(kDumpGraphLevel, "4", 1);
  ComputeGraph compute_graph("test_graph0");
  compute_graph.SetGraphID(0);
  const std::string suffit = "always_dump";
  ge::GraphUtils::DumpGEGraphToOnnx(compute_graph, suffit);
  unsetenv(kDumpGraphLevel);

  // test NOT existed dir
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  onnx::ModelProto model_proto;
  ASSERT_EQ(model_proto.ByteSize(), 0);
  // static thing, so follow DumpGEGraphUserGraphNameNull_AscendWorkPathNotNull this case path
  std::string ascend_work_path = "./test_ge_graph_path";
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(ascend_work_path);
  std::string dump_graph_path = GetSpecificFilePath(ge::RealPath(dump_file_path.str().c_str()), suffit);
  bool state = GraphUtils::ReadProtoFromTextFile(dump_graph_path.c_str(), &model_proto);
  ASSERT_EQ(state, false);
  ASSERT_EQ(model_proto.ByteSize(), 0);
}

TEST_F(UtestGraphUtils, DumpGEGraphToOnnxAlways) {
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "4", 1);
  ComputeGraph compute_graph("test_graph0");
  compute_graph.SetGraphID(0);
  const std::string suffit = "always_dump";
  ge::GraphUtils::DumpGEGraphToOnnx(compute_graph, suffit, true);
  unsetenv(kDumpGraphLevel);

  // test existed dir
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  onnx::ModelProto model_proto;
  ASSERT_EQ(model_proto.ByteSize(), 0);
  // static thing, so follow DumpGEGraphUserGraphNameNull_AscendWorkPathNotNull this case path
  std::string ascend_work_path = "./test_ge_graph_path";
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(ascend_work_path);
  std::string dump_graph_path = GetSpecificFilePath(ge::RealPath(dump_file_path.str().c_str()), suffit);
  bool state = GraphUtils::ReadProtoFromTextFile(dump_graph_path.c_str(), &model_proto);
  ASSERT_EQ(state, true);
  ASSERT_NE(model_proto.ByteSize(), 0);
  system(("rm -rf " + ascend_work_path).c_str());
}

TEST_F(UtestGraphUtils, DumpGEGraphToOnnxByPath) {
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void) setenv(kDumpGraphLevel, "4", 1);
  ComputeGraph compute_graph("test_graph0");
  compute_graph.SetGraphID(0);
  const std::string suffix = "aaa/bbb.ccc\\ddd";
  std::string dump_file_path = "/root/";
  ge::GraphUtils::DumpGrphToOnnx(compute_graph, dump_file_path, suffix);
  dump_file_path = "/tmp/";
  ge::GraphUtils::DumpGrphToOnnx(compute_graph, dump_file_path, suffix);
  unsetenv(kDumpGraphLevel);

  // test existed dir
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  onnx::ModelProto model_proto;
  ASSERT_EQ(model_proto.ByteSize(), 0);
  const std::string safe_suffix = "aaa_bbb.ccc_ddd";
  std::string dump_graph_path = GetSpecificFilePath(dump_file_path, safe_suffix);
  bool state = GraphUtils::ReadProtoFromTextFile(dump_graph_path.c_str(), &model_proto);
  ASSERT_EQ(state, true);
  ASSERT_NE(model_proto.ByteSize(), 0);
  system(("rm -f " + dump_graph_path).c_str());
}

TEST_F(UtestGraphUtils, DumpGEGraphWithDumpGEGraphInvalid) {
  auto graph = BuildGraphWithConst();
  std::string ascend_work_path = "./test_ge_graph_path";
  mmSetEnv("DUMP_GE_GRAPH", "0", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  GraphUtils::DumpGEGraph(graph, "", false, "");
  ComputeGraphPtr com_graph = std::make_shared<ComputeGraph>("GeTestGraph");

  // test load
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(ascend_work_path);
  std::string dump_graph_path = ge::RealPath(dump_file_path.str().c_str());
  auto state = GraphUtils::LoadGEGraph(GetSpecificFilePath(dump_graph_path, "_.txt").c_str(), *com_graph);
  ASSERT_EQ(state, false);
  unsetenv("DUMP_GRAPH_PATH");
  unsetenv("DUMP_GE_GRAPH");
  system(("rm -rf " + ascend_work_path).c_str());
}

/*
*               var                               var
*  atomicclean  |  \                             |   \
*         \\    |   assign                       |   assign
*          \\   |   //         =======>          |   //
*           allreduce                         identity  atomicclean
*             |                                 |       //
*            netoutput                        allreduce
*                                               |
*                                           netoutput
 */
TEST_F(UtestGraphUtils, InsertNodeBefore_DoNotMoveCtrlEdgeFromAtomicClean) {
  // build test graph
  auto builder = ut::GraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  const auto &identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  // insert identity before allreduce
  GraphUtils::InsertNodeBefore(allreduce->GetInDataAnchor(0), identity, 0, 0);

  // check assign control-in on identity
  ASSERT_EQ(identity->GetInControlNodes().at(0)->GetName(), "assign");
  // check atomicclean control-in still on allreuce
  ASSERT_EQ(allreduce->GetInControlNodes().at(0)->GetName(), "atomic_clean");
}

TEST_F(UtestGraphUtils, GetSubgraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &case0 = root_builder.AddNode("case0", "Case", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &case1 = sub_builder1.AddNode("case1", "Case", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = ut::GraphBuilder("sub2");
  const auto &sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case1);
  sub_graph2->SetParentGraph(sub_graph1);
  case1->GetOpDesc()->AddSubgraphName("branch1");
  case1->GetOpDesc()->SetSubgraphInstanceName(0, "sub2");
  case1->GetOpDesc()->AddSubgraphName("branch2");
  case1->GetOpDesc()->SetSubgraphInstanceName(1, "not_exist");

  std::vector<ComputeGraphPtr> subgraphs1;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(root_graph, subgraphs1), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs1.size(), 2);

  std::vector<ComputeGraphPtr> subgraphs2;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(sub_graph1, subgraphs2), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs2.size(), 1);

  std::vector<ComputeGraphPtr> subgraphs3;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(sub_graph2, subgraphs3), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs3.empty());
}

TEST_F(UtestGraphUtils, GetSubgraphs_nullptr_graph) {
  std::vector<ComputeGraphPtr> subgraphs;
  ASSERT_NE(GraphUtils::GetSubgraphsRecursively(nullptr, subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(UtestGraphUtils, ReplaceEdgeSrc) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node2, 0);
  ASSERT_EQ(GraphUtils::ReplaceEdgeSrc(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node1->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeSrc(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node3->GetOutDataAnchor(0)), GRAPH_SUCCESS);

  builder.AddControlEdge(node0, node2);
  ASSERT_EQ(GraphUtils::ReplaceEdgeSrc(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node1->GetOutControlAnchor()), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeSrc(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node3->GetOutControlAnchor()), GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceEdgeDst) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node2, 0);
  ASSERT_EQ(GraphUtils::ReplaceEdgeDst(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node1->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeDst(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node3->GetInDataAnchor(0)), GRAPH_SUCCESS);

  builder.AddControlEdge(node0, node2);
  ASSERT_EQ(GraphUtils::ReplaceEdgeDst(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node1->GetInControlAnchor()), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeDst(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node3->GetInControlAnchor()), GRAPH_SUCCESS);
}

/*
 *          data0  data1
 *             \    /|
 *              add1 | data2
 *                 \ |  /|
 *                  add2 | data3
 *                     \ |  /|
 *                      add3 |  data4
 *                         \ |  / | \
 *                          add4  | cast1
 *                              \ | / |
 *                              add5  |
 *                                | \ |
 *                                | cast2
 *                                | /
 *                             netoutput
 */
TEST_F(UtestGraphUtils, BuildSubgraphWithNodes) {
  auto builder = ut::GraphBuilder("root");
  const auto &data0 = builder.AddNode("data0", DATA, 1, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &data3 = builder.AddNode("data3", DATA, 1, 1);
  const auto &data4 = builder.AddNode("data4", DATA, 1, 1);

  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  const auto &add2 = builder.AddNode("add2", "Add", 2, 1);
  const auto &add3 = builder.AddNode("add3", "Add", 2, 1);
  const auto &add4 = builder.AddNode("add4", "Add", 2, 1);
  const auto &add5 = builder.AddNode("add5", "Add", 2, 1);

  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, add1, 0);
  builder.AddDataEdge(data1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(data2, 0, add2, 1);
  builder.AddDataEdge(add2, 0, add3, 0);
  builder.AddDataEdge(data3, 0, add3, 1);
  builder.AddDataEdge(add3, 0, add4, 0);
  builder.AddDataEdge(data4, 0, add4, 1);
  builder.AddDataEdge(data4, 0, cast1, 0);
  builder.AddDataEdge(add4, 0, add5, 0);
  builder.AddDataEdge(cast1, 0, add5, 1);
  builder.AddDataEdge(add5, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, netoutput, 0);

  builder.AddControlEdge(data1, add2);
  builder.AddControlEdge(data2, add3);
  builder.AddControlEdge(data3, add4);
  builder.AddControlEdge(data4, add5);
  builder.AddControlEdge(add5, netoutput);
  builder.AddControlEdge(cast1, cast2);

  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(nullptr, {}, "subgraph1"), nullptr);

  const auto &graph = builder.GetGraph();
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, {}, "subgraph1"), nullptr);

  std::set<NodePtr> nodes = { data1, add2, add3, add4, add5, cast2 };
  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, nodes, "subgraph1"), nullptr);

  ASSERT_TRUE(AttrUtils::SetStr(graph, "_session_graph_id", "_session_graph_id"));
  const auto &subgraph1 = GraphUtils::BuildSubgraphWithNodes(graph, nodes, "subgraph1");
  ASSERT_NE(subgraph1, nullptr);
  ASSERT_EQ(subgraph1->GetParentGraph(), graph);
  ASSERT_TRUE(subgraph1->HasAttr("_session_graph_id"));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "data1"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add2"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add3"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add4"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add5"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "cast2"; }));
  ASSERT_TRUE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetType() == PARTITIONEDCALL; }));
  ASSERT_EQ(graph->GetAllSubgraphs().size(), 1);

  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, {cast1}, "subgraph1"), nullptr);
}

TEST_F(UtestGraphUtils, BuildSubgraphWithOnlyControlNodes) {
  auto builder = ut::GraphBuilder("root");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto op1 = builder.AddNode("square1", "Square", 1, 1);
  auto op2 = builder.AddNode("square2", "Square", 1, 1);
  auto output = builder.AddNode("output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, op1, 0);
  builder.AddDataEdge(data2, 0, op2, 0);
  builder.AddDataEdge(op2, 0, output, 0);
  builder.AddControlEdge(op1, op2);
  auto origin_graph = builder.GetGraph();
  ASSERT_TRUE(AttrUtils::SetStr(origin_graph, "_session_graph_id", "graph_id"));

  auto subgraph = GraphUtils::BuildSubgraphWithNodes(origin_graph, {op1}, "subgraph");
  ASSERT_NE(subgraph, nullptr);
  auto subgraph_output = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(subgraph_output, nullptr);
  ASSERT_FALSE(subgraph_output->GetInControlNodes().empty());
  ASSERT_EQ((*subgraph_output->GetInControlNodes().begin())->GetType(), "Square");
}

TEST_F(UtestGraphUtils, BuildGraphFromNodes_case0) {
  auto builder = ut::GraphBuilder("graph0");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto op1 = builder.AddNode("square1", "Square", 1, 1);
  auto op2 = builder.AddNode("square2", "Square", 1, 1);
  auto output = builder.AddNode("output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, op1, 0);
  builder.AddDataEdge(data2, 0, op2, 0);
  builder.AddDataEdge(op2, 0, output, 0);
  builder.AddControlEdge(op1, op2);
  auto origin_graph = builder.GetGraph();

  auto graph = GraphUtils::BuildGraphFromNodes({op1}, "graph1");
  ASSERT_NE(graph, nullptr);
  auto graph_output = graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(graph_output, nullptr);
  ASSERT_FALSE(graph_output->GetInControlNodes().empty());
  ASSERT_EQ((*graph_output->GetInControlNodes().begin())->GetType(), "Square");
}


TEST_F(UtestGraphUtils, SingleOpScene) {
  auto builder1 = ut::GraphBuilder("root");
  auto data1 = builder1.AddNode("data1", DATA, 0, 1);
  auto graph1 = builder1.GetGraph();
  ASSERT_TRUE(AttrUtils::SetBool(graph1, ATTR_SINGLE_OP_SCENE, true));
  bool is_single_op = GraphUtils::IsSingleOpScene(graph1);
  ASSERT_EQ(is_single_op, true);

  auto builder2 = ut::GraphBuilder("root");
  auto data2 = builder2.AddNode("data2", DATA, 0, 1);
  AttrUtils::SetBool(data2->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);
  auto graph2 = builder2.GetGraph();
  is_single_op = GraphUtils::IsSingleOpScene(graph2);
  ASSERT_EQ(is_single_op, true);

  auto builder3 = ut::GraphBuilder("root");
  auto data3 = builder3.AddNode("data3", DATA, 0, 1);
  auto graph3 = builder3.GetGraph();
  is_single_op = GraphUtils::IsSingleOpScene(graph3);
  ASSERT_EQ(is_single_op, false);
}

TEST_F(UtestGraphUtils, UnfoldSubgraph) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForUnfold(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  const auto &filter = [](const ComputeGraphPtr &graph) {
    const auto &parent_node = graph->GetParentNode();
    if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
      return false;
    }
    if ((parent_node->GetType() != PARTITIONEDCALL) ||
        (parent_node->GetOpDesc()->GetSubgraphInstanceNames().size() != 1)) {
      return false;
    }
    return graph->GetGraphUnknownFlag();
  };
  ASSERT_EQ(GraphUtils::UnfoldSubgraph(subgraph, filter), GRAPH_SUCCESS);

  ASSERT_EQ(graph->GetAllSubgraphs().size(), 1);
  ASSERT_FALSE(graph->GetAllSubgraphs()[0]->GetGraphUnknownFlag());
}

TEST_F(UtestGraphUtils, UnfoldSubgraph_InnerDataHasOutControl) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForUnfoldWithControlEdge(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  const auto &filter = [](const ComputeGraphPtr &graph) {
    const auto &parent_node = graph->GetParentNode();
    if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
      return false;
    }
    if (parent_node->GetType() == PARTITIONEDCALL) {
      return true;
    }
    return false;
  };
  ASSERT_EQ(GraphUtils::UnfoldSubgraph(subgraph, filter), GRAPH_SUCCESS);
  ASSERT_EQ(graph->GetAllSubgraphs().size(), 0);
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

}

TEST_F(UtestGraphUtils, UnfoldSubgraph_ForPartition) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForUnfold(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);
  std::vector<NodePtr> inputs;
  std::vector<NodePtr> outputs;
  const auto &new_graph = GraphUtils::CloneGraph(graph, "", inputs, outputs);
  const auto &node_size_before_unfold = new_graph->GetDirectNode().size();
  const auto &filter = [](const ComputeGraphPtr &graph) {
    const auto &parent_node = graph->GetParentNode();
    if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
      return false;
    }
    if ((parent_node->GetType() != PARTITIONEDCALL) ||
        (parent_node->GetOpDesc()->GetSubgraphInstanceNames().size() != 1)) {
      return false;
    }
    return graph->GetGraphUnknownFlag();
  };
  ASSERT_EQ(GraphUtils::UnfoldGraph(subgraph, new_graph, new_graph->FindNode(subgraph->GetParentNode()->GetName()),
                                       filter), GRAPH_SUCCESS);
  ASSERT_NE(node_size_before_unfold, new_graph->GetDirectNode().size());
}

TEST_F(UtestGraphUtils, GetIndependentCompileGraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioned_call0 = root_builder.AddNode("PartitionedCall", "PartitionedCall", 0, 0);
  const auto &root_graph = root_builder.GetGraph();
  (void)AttrUtils::SetBool(*root_graph, ATTR_NAME_PIPELINE_PARTITIONED, true);

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &data1 = sub_builder1.AddNode("Data", "Data", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(partitioned_call0);
  sub_graph1->SetParentGraph(root_graph);
  partitioned_call0->GetOpDesc()->AddSubgraphName("sub1");
  partitioned_call0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  std::vector<ComputeGraphPtr> independent_compile_subgraphs;
  ASSERT_EQ(GraphUtils::GetIndependentCompileGraphs(root_graph, independent_compile_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(independent_compile_subgraphs.size(), 1);
  ASSERT_EQ(independent_compile_subgraphs[0]->GetName(), "sub1");

  (void)AttrUtils::SetBool(*root_graph, ATTR_NAME_PIPELINE_PARTITIONED, false);
  independent_compile_subgraphs.clear();
  ASSERT_EQ(GraphUtils::GetIndependentCompileGraphs(root_graph, independent_compile_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(independent_compile_subgraphs.size(), 1);
  ASSERT_EQ(independent_compile_subgraphs[0]->GetName(), "root");
}

TEST_F(UtestGraphUtils, InsertNodeAfter) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  std::vector<ComputeGraphPtr> independent_compile_subgraphs;
  ASSERT_EQ(GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), {}, node1, 0, 0), GRAPH_FAILED);
}
  TEST_F(UtestGraphUtils, NoNeedDumpGraphBySuffixIsFalse) {
    std::string suffix;
    bool ret = GraphUtils::NoNeedDumpGraphBySuffix(suffix);
    EXPECT_EQ(ret, true);
  }

TEST_F(UtestGraphUtils, NoNeedDumpGraphBySuffixLevel0) {
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "0", 1);
  EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("Build"), true);
  EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("test"), true);
  unsetenv(kDumpGraphLevel);
}

  TEST_F(UtestGraphUtils, NoNeedDumpGraphBySuffixLevel1) {
    const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
    (void)setenv(kDumpGraphLevel, "1", 1);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix(""), true);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("Build"), false);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("test"), false);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("OptimizeSubGraph"), true);
    unsetenv(kDumpGraphLevel);
  }

  TEST_F(UtestGraphUtils, NoNeedDumpGraphBySuffixLevel2) {
    const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
    (void) setenv(kDumpGraphLevel, "2", 1);
    for (const auto &graph_name : kGeDumpWhitelistFullName) {
      EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix(graph_name), false);
    }
    for (const auto &graph_name : kGeDumpWhitelistKeyName) {
      EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("_" + graph_name), false);
    }
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("PreRunAfterOptimizeGraphPrepare"), true);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("RunCustomAfterInferShape_sub_graph"), true);
    unsetenv(kDumpGraphLevel);
  }

  TEST_F(UtestGraphUtils, NoNeedDumpGraphBySuffixLevel3) {
    const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
    (void)setenv(kDumpGraphLevel, "3", 1);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("Build"), false);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("test"), true);
    unsetenv(kDumpGraphLevel);
  }

  TEST_F(UtestGraphUtils, NoNeedDumpGraphBySuffixLevel4) {
    const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
    (void)setenv(kDumpGraphLevel, "4", 1);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("PreRunBegin"), false);
    EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("test"), true);
    unsetenv(kDumpGraphLevel);
  }

TEST_F(UtestGraphUtils, NoNeedDumpGraphBySuffixWhitelist) {
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "RunBegin", 1);
  EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("PreRunBegin"), false);
  EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("test"), true);

  (void)setenv(kDumpGraphLevel, "RunBegin|test", 1);
  EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("PreRunBegin"), false);
  EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("test"), false);
  EXPECT_EQ(GraphUtils::NoNeedDumpGraphBySuffix("other"), true);
  unsetenv(kDumpGraphLevel);
}

TEST_F(UtestGraphUtils, DumpGEGraph_aclop_success) {
  std::string dump_graph_path = "./test_ge_graph_path";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_LEVEL", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", dump_graph_path.c_str(), 1);
  auto graph = BuildGraphWithSubGraph();
  AttrUtils::SetBool(graph, ATTR_SINGLE_OP_SCENE, true);
  GraphUtils::DumpGEGraph(graph, "test");
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(dump_graph_path);
  std::string file_path = ge::RealPath(dump_file_path.str().c_str());
  // root graph
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("GeTestGraph");
  ASSERT_EQ(GraphUtils::LoadGEGraph(GetSpecificFilePath(file_path, "_aclop_").c_str(), *compute_graph), true);

  unsetenv("DUMP_GE_GRAPH");
  unsetenv("DUMP_GRAPH_LEVEL");
  unsetenv("DUMP_GRAPH_PATH");
  system(("rm -rf " + dump_graph_path).c_str());
}

TEST_F(UtestGraphUtils, DumpGEGraph) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  // test existed dir
  GraphUtils::DumpGEGraph(graph, "", true, "./ge_test_graph_0001.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph("./ge_test_graph_0001.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);

  // test not existed dir
  GraphUtils::DumpGEGraph(graph, "", true, "./test/ge_test_graph_0002.txt");
  ComputeGraphPtr com_graph2 = std::make_shared<ComputeGraph>("GeTestGraph2");
  state = GraphUtils::LoadGEGraph("./test/ge_test_graph_0002.txt", *com_graph2);
  ASSERT_EQ(state, true);

  // test input para user_graph_name, without path
  GraphUtils::DumpGEGraph(graph, "", true, "ge_test_graph_0003.txt");
  ComputeGraphPtr com_graph3 = std::make_shared<ComputeGraph>("GeTestGraph3");
  state = GraphUtils::LoadGEGraph("./ge_test_graph_0003.txt", *com_graph3);
  ASSERT_EQ(state, true);
  unsetenv("DUMP_GE_GRAPH");
}
TEST_F(UtestGraphUtils, DumpGEGraphNoOptionsSucc) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  GEThreadLocalContext &context = GetThreadLocalContext();
  std::map<std::string, std::string> graph_maps;
  std::string key1 = "pk1";
  std::string value1 = "pv1";
  std::string key2 = "pk2";
  std::string value2 = "pv2";
  graph_maps.insert(std::make_pair(key1, value1));
  graph_maps.insert(std::make_pair(key2, value2));
  context.SetGraphOption(graph_maps);

  std::map<std::string, std::string> session_maps;
  key1 = "sk1";
  value1 = "sv1";
  key2 = "sk2";
  value2 = "sv2";
  session_maps.insert(std::make_pair(key1, value1));
  session_maps.insert(std::make_pair(key2, value2));
  context.SetSessionOption(session_maps);

  std::map<std::string, std::string> global_maps;
  key1 = "gk1";
  value1 = "gv1";
  key2 = "gk2";
  value2 = "gv2";
  global_maps.insert(std::make_pair(key1, value1));
  global_maps.insert(std::make_pair(key2, value2));
  context.SetGlobalOption(global_maps);

  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "4", 1);
  const char_t *const kDumpGeGraph = "DUMP_GE_GRAPH";
  (void)setenv(kDumpGeGraph, "3", 1);
  // test existed dir
  system("rm -f ./ge_test_graph_options_wt_0001.txt");
  GraphUtils::DumpGEGraph(graph, "PreRunBegin", false, "./ge_test_graph_options_wt_0001.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph("./ge_test_graph_options_wt_0001.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);
  //check graph option
  ge::NamedAttrs graphOptions;
  EXPECT_EQ(AttrUtils::GetNamedAttrs(com_graph1, "GraphOptions", graphOptions),false);
  ge::NamedAttrs sessionOptions;
  EXPECT_EQ(AttrUtils::GetNamedAttrs(com_graph1, "SessionOptions", sessionOptions),false);
  ge::NamedAttrs globalOptions;
  EXPECT_EQ(AttrUtils::GetNamedAttrs(com_graph1, "GlobalOptions", globalOptions),false);

}

TEST_F(UtestGraphUtils, DumpGEGraphOptionsSucc) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  GEThreadLocalContext &context = GetThreadLocalContext();
  std::map<std::string, std::string> graph_maps;
  std::string key1 = "pk1";
  std::string value1 = "pv1";
  std::string key2 = "pk2";
  std::string value2 = "pv2";
  graph_maps.insert(std::make_pair(key1, value1));
  graph_maps.insert(std::make_pair(key2, value2));
  context.SetGraphOption(graph_maps);

  std::map<std::string, std::string> session_maps;
  key1 = "sk1";
  value1 = "sv1";
  key2 = "sk2";
  value2 = "sv2";
  session_maps.insert(std::make_pair(key1, value1));
  session_maps.insert(std::make_pair(key2, value2));
  context.SetSessionOption(session_maps);

  std::map<std::string, std::string> global_maps;
  key1 = "gk1";
  value1 = "gv1";
  key2 = "gk2";
  value2 = "gv2";
  global_maps.insert(std::make_pair(key1, value1));
  global_maps.insert(std::make_pair(key2, value2));
  context.SetGlobalOption(global_maps);

  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "4", 1);
  const char_t *const kDumpGeGraph = "DUMP_GE_GRAPH";
  (void)setenv(kDumpGeGraph, "1", 1);
  // test existed dir
  system("rm -f ./ge_test_graph_options_wt_0002.txt");
  GraphUtils::DumpGEGraph(graph, "PreRunBegin", false, "./ge_test_graph_options_wt_0002.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph("./ge_test_graph_options_wt_0002.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);
  //check graph option
  ge::NamedAttrs graphOptions;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(com_graph1, "GraphOptions", graphOptions));
  EXPECT_EQ(graphOptions.GetName(), "GraphOptions");
  AnyValue av;
  EXPECT_EQ(graphOptions.GetAttr("pk1", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "pv1");
  EXPECT_EQ(graphOptions.GetAttr("pk2", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "pv2");
  //check session option
  ge::NamedAttrs sessionOptions;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(com_graph1, "SessionOptions", sessionOptions));
  EXPECT_EQ(sessionOptions.GetName(), "SessionOptions");
  EXPECT_EQ(sessionOptions.GetAttr("sk1", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "sv1");

  EXPECT_EQ(sessionOptions.GetAttr("sk2", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "sv2");
  //check global option
  ge::NamedAttrs globalOptions;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(com_graph1, "GlobalOptions", globalOptions));
  EXPECT_EQ(globalOptions.GetName(), "GlobalOptions");
  EXPECT_EQ(globalOptions.GetAttr("gk1", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "gv1");

  EXPECT_EQ(globalOptions.GetAttr("gk2", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "gv2");
}
TEST_F(UtestGraphUtils, DumpGEGraphOptionsLevelNot4) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  GEThreadLocalContext &context = GetThreadLocalContext();
  std::map<std::string, std::string> graph_maps;
  std::string key1 = "pk1";
  std::string value1 = "pv1";
  std::string key2 = "pk2";
  std::string value2 = "pv2";
  graph_maps.insert(std::make_pair(key1, value1));
  graph_maps.insert(std::make_pair(key2, value2));
  context.SetGraphOption(graph_maps);

  std::map<std::string, std::string> session_maps;
  key1 = "sk1";
  value1 = "sv1";
  key2 = "sk2";
  value2 = "sv2";
  session_maps.insert(std::make_pair(key1, value1));
  session_maps.insert(std::make_pair(key2, value2));
  context.SetSessionOption(session_maps);

  std::map<std::string, std::string> global_maps;
  key1 = "gk1";
  value1 = "gv1";
  key2 = "gk2";
  value2 = "gv2";
  global_maps.insert(std::make_pair(key1, value1));
  global_maps.insert(std::make_pair(key2, value2));
  context.SetGlobalOption(global_maps);

  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "1", 1);
  const char_t *const kDumpGeGraph = "DUMP_GE_GRAPH";
  (void)setenv(kDumpGeGraph, "1", 1);
  // test existed dir
  system("rm -f ./ge_test_graph_options_wt_0004.txt");
  GraphUtils::DumpGEGraph(graph, "test", false, "./ge_test_graph_options_wt_0004.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph("./ge_test_graph_options_wt_0004.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);
  //check graph option
  ge::NamedAttrs graphOptions;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(com_graph1, "GraphOptions", graphOptions));
  EXPECT_EQ(graphOptions.GetName(), "GraphOptions");
  AnyValue av;
  EXPECT_EQ(graphOptions.GetAttr("pk1", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "pv1");
  EXPECT_EQ(graphOptions.GetAttr("pk2", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "pv2");
  //check session option
  ge::NamedAttrs sessionOptions;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(com_graph1, "SessionOptions", sessionOptions));
  EXPECT_EQ(sessionOptions.GetName(), "SessionOptions");
  EXPECT_EQ(sessionOptions.GetAttr("sk1", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "sv1");

  EXPECT_EQ(sessionOptions.GetAttr("sk2", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "sv2");
  //check global option
  ge::NamedAttrs globalOptions;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(com_graph1, "GlobalOptions", globalOptions));
  EXPECT_EQ(globalOptions.GetName(), "GlobalOptions");
  EXPECT_EQ(globalOptions.GetAttr("gk1", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "gv1");

  EXPECT_EQ(globalOptions.GetAttr("gk2", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<std::string>(), nullptr);
  EXPECT_EQ(*av.Get<std::string>(), "gv2");
}

TEST_F(UtestGraphUtils, DumpGEGraphOptionsNotPreRunBeginNoDump) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  GEThreadLocalContext &context = GetThreadLocalContext();
  std::map<std::string, std::string> graph_maps;
  std::string key1 = "pk1";
  std::string value1 = "pv1";
  std::string key2 = "pk2";
  std::string value2 = "pv2";
  graph_maps.insert(std::make_pair(key1, value1));
  graph_maps.insert(std::make_pair(key2, value2));
  context.SetGraphOption(graph_maps);

  std::map<std::string, std::string> session_maps;
  key1 = "sk1";
  value1 = "sv1";
  key2 = "sk2";
  value2 = "sv2";
  session_maps.insert(std::make_pair(key1, value1));
  session_maps.insert(std::make_pair(key2, value2));
  context.SetSessionOption(session_maps);

  std::map<std::string, std::string> global_maps;
  key1 = "gk1";
  value1 = "gv1";
  key2 = "gk2";
  value2 = "gv2";
  global_maps.insert(std::make_pair(key1, value1));
  global_maps.insert(std::make_pair(key2, value2));
  context.SetGlobalOption(global_maps);

  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "4", 1);
  const char_t *const kDumpGeGraph = "DUMP_GE_GRAPH";
  (void)setenv(kDumpGeGraph, "1", 1);
  // test existed dir
  system("rm -f ./ge_test_graph_options_wt_0003.txt");
  GraphUtils::DumpGEGraph(graph, "test", false, "./ge_test_graph_options_wt_0003.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph("./ge_test_graph_options_wt_0003.txt", *com_graph1);
  ASSERT_EQ(state, false);
}

TEST_F(UtestGraphUtils, CheckDumpGraphNum) {
  std::map<std::string, std::string> session_option{{"ge.maxDumpFileNum", "100"}};
  GetThreadLocalContext().SetSessionOption(session_option);
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  EXPECT_NO_THROW(
    GraphUtils::DumpGEGrph(graph0, "./", "1");
    GraphUtils::DumpGEGrph(graph0, "./", "1");
    GraphUtils::DumpGEGrph(graph0, "./", "1");
    GraphUtils::DumpGEGrph(graph0, "./", "1");
    GraphUtils::DumpGEGrph(graph0, "./", "1");
  );
}

TEST_F(UtestGraphUtils, CopyRootComputeGraph) {
  auto graph = BuildGraphWithSubGraph();
  // check origin graph size
  ASSERT_EQ(graph->GetAllNodesSize(), 9);
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>(ComputeGraph("dst"));
  // test copy root graph success
  auto ret = GraphUtils::CopyComputeGraph(graph, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(dst_compute_graph->GetAllNodesSize(), 9);
  // test copy subgraph failed
  auto sub1_graph = graph->GetSubgraph("sub1");
  ret = GraphUtils::CopyComputeGraph(sub1_graph, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // test copy dst compute_graph null
  ComputeGraphPtr empty_dst_compute_graph;
  ret = GraphUtils::CopyComputeGraph(graph, empty_dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyComputeGraphWithNodeAndGraphFilter) {
  auto graph = BuildGraphWithSubGraph();
  // check origin graph size
  ASSERT_EQ(graph->GetAllNodesSize(), 5 + 2 + 2);
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>(ComputeGraph("dst"));
  auto node_filter = [&graph](const Node &node) {
    // no filter node which not in root graph
    if (node.GetOwnerComputeGraph()->GetName() != graph->GetName()) {
      return true;
    }
    // filter root graph node when node name == "relu1"
    if (node.GetName() == "relu1") {
      return false;
    }
    // copy other nodes in root graph
    return true;
  };

  auto graph_filter = [&graph](const Node &node, const char *, const ComputeGraphPtr &sub_graph) {
    // sub2 graph not copy
    return sub_graph->GetName() != "sub2";
  };
  // test copy root graph success
  auto ret = GraphUtils::CopyComputeGraph(graph, node_filter, graph_filter, nullptr, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(dst_compute_graph->GetAllNodesSize(), 4 + 2 + 0);
  ASSERT_EQ(dst_compute_graph->GetDirectNodesSize(), 4);
  ASSERT_EQ(dst_compute_graph->GetDirectNode().size(), 4);
  ASSERT_EQ(dst_compute_graph->FindNode("relu1"), nullptr);
  ASSERT_NE(dst_compute_graph->FindNode("relu0"), nullptr);
  auto sub1_graph = dst_compute_graph->GetSubgraph("sub1");
  ASSERT_EQ(sub1_graph->GetDirectNodesSize(), 2);
  ASSERT_NE(sub1_graph->GetDirectNode().at(0U), nullptr);
  ASSERT_NE(sub1_graph->GetDirectNode().at(0U)->GetOpDesc(), nullptr);
  ASSERT_EQ(sub1_graph->GetDirectNode().at(0U)->GetOpDesc()->GetId(),
            graph->GetSubgraph("sub1")->GetDirectNode().at(0U)->GetOpDesc()->GetId());
  ASSERT_NE(sub1_graph, nullptr);
  ASSERT_EQ(dst_compute_graph->GetSubgraph("sub2"), nullptr);
}

TEST_F(UtestGraphUtils, CopyComputeGraphWithoutSubGraphRepeat) {
  auto graph = BuildGraphWithSubGraph();
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>(ComputeGraph("dst"));
  auto graph_filter = [](const Node &node, const char *, const ComputeGraphPtr &sub_graph) {
    // all graph not copy
    return false;
  };
  // test copy root graph success
  auto ret = GraphUtils::CopyComputeGraph(graph, nullptr, graph_filter, nullptr, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(dst_compute_graph->GetDirectNodesSize(), graph->GetDirectNodesSize());
  NodePtr case0_node = dst_compute_graph->FindNode("case0");
  ASSERT_NE(case0_node, nullptr);
  const auto &names = case0_node->GetOpDesc()->GetSubgraphInstanceNames();
  for (const auto &name:names) {
    EXPECT_EQ(name, "");
  }
  ASSERT_EQ(dst_compute_graph->GetSubgraph("sub1"), nullptr);
  ASSERT_EQ(dst_compute_graph->GetSubgraph("sub2"), nullptr);
  ComputeGraphPtr dst_compute_graph2 = std::make_shared<ComputeGraph>(ComputeGraph("dst2"));
  ret = GraphUtils::CopyComputeGraph(dst_compute_graph, nullptr, graph_filter, nullptr, dst_compute_graph2);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(dst_compute_graph2->GetDirectNodesSize(), graph->GetDirectNodesSize());
}

TEST_F(UtestGraphUtils, CopyComputeGraphWithAttrFilter) {
  auto graph = BuildGraphWithConst();
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>(ComputeGraph("dst"));
  const std::string const_node_name = "Const";
  auto const_node_with_weight_src = graph->FindNode(const_node_name);
  auto attr_filter = [&const_node_name](const OpDesc &op_desc, const std::string &attr_name) {
    // keep all attr for nodes which is not `const`
    if (op_desc.GetName() != const_node_name) {
      return true;
    }
    // const node not copy weights
    if (attr_name == ge::ATTR_NAME_WEIGHTS) {
      return false;
    }
    return true;
  };

  // test copy graph success with attr filter
  ASSERT_EQ(GraphUtils::CopyComputeGraph(graph, nullptr, nullptr, attr_filter, dst_compute_graph), GRAPH_SUCCESS);
  auto const_node_with_weight_dst = dst_compute_graph->FindNode(const_node_name);
  ASSERT_NE(const_node_with_weight_dst, nullptr);
  ASSERT_NE(const_node_with_weight_src, const_node_with_weight_dst);
  // src node keep origin weight
  ConstGeTensorPtr weight = nullptr;
  ASSERT_TRUE(AttrUtils::GetTensor(const_node_with_weight_src->GetOpDesc(), ATTR_NAME_WEIGHTS, weight));
  ASSERT_NE(weight, nullptr);
  ASSERT_EQ(weight->GetData().GetSize(), 4096U);
  const uint8_t *buff = weight->GetData().GetData();
  ASSERT_EQ((buff == nullptr), false);
  ASSERT_EQ(buff[0], 7);
  ASSERT_EQ(buff[10], 8);
  // dst node has not weight
  ASSERT_FALSE(AttrUtils::GetTensor(const_node_with_weight_dst->GetOpDesc(), ATTR_NAME_WEIGHTS, weight));
  // dst node has other attr
  std::string str_value;
  ASSERT_TRUE(AttrUtils::GetStr(const_node_with_weight_dst->GetOpDesc(), "fake_attr_name", str_value));
  ASSERT_EQ("fake_attr_value", str_value);
  auto add_node_dst = dst_compute_graph->FindNode("Add");
  ASSERT_NE(add_node_dst, nullptr);
  // other node has all attr copyed
  ASSERT_TRUE(AttrUtils::GetStr(add_node_dst->GetOpDesc(), "fake_attr_name", str_value));
  ASSERT_TRUE(AttrUtils::GetStr(add_node_dst->GetOpDesc(), ATTR_NAME_WEIGHTS, str_value));

  // test copy graph success without attr filter
  ComputeGraphPtr dst_compute_graph2 = std::make_shared<ComputeGraph>(ComputeGraph("dst2"));
  ASSERT_EQ(GraphUtils::CopyComputeGraph(graph, nullptr, nullptr, nullptr, dst_compute_graph2), GRAPH_SUCCESS);
  auto const_node_with_weight_dst2 = dst_compute_graph2->FindNode(const_node_name);
  ASSERT_NE(const_node_with_weight_dst2, nullptr);
  ASSERT_NE(const_node_with_weight_src, const_node_with_weight_dst2);
  ConstGeTensorPtr weight2 = nullptr;
  ASSERT_TRUE(AttrUtils::GetTensor(const_node_with_weight_dst2->GetOpDesc(), ATTR_NAME_WEIGHTS, weight2));
  ASSERT_NE(weight2, nullptr);
  ASSERT_EQ(weight2->GetData().GetSize(), 4096U);
  const uint8_t *buff2 = weight2->GetData().GetData();
  // deep copy
  ASSERT_NE(buff2, buff);
  ASSERT_EQ((buff2 == nullptr), false);
  ASSERT_EQ(buff2[0], 7);
  ASSERT_EQ(buff2[10], 8);
}

TEST_F(UtestGraphUtils, DumpGraphByPath) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 0);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  // test dump_level 0
  auto ret = GraphUtils::DumpGEGraphByPath(graph, "./not-exists-path/test_graph_0.txt", ge::DumpLevel::NO_DUMP);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  ret = GraphUtils::DumpGEGraphByPath(graph, "/", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ((ret != 0), true);
  ret = GraphUtils::DumpGEGraphByPath(graph, "test_graph_0.txt", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ((ret != 0), true);
  ret = GraphUtils::DumpGEGraphByPath(graph, "./test_graph_0.txt", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ(ret, 0);
  ComputeGraphPtr com_graph0 = std::make_shared<ComputeGraph>("TestGraph0");
  bool state = GraphUtils::LoadGEGraph("./test_graph_0.txt", *com_graph0);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph0->GetAllNodesSize(), 4);
  for (auto &node_ptr : com_graph0->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), false);
    }
  }

  // test dump_level 1
  ret = GraphUtils::DumpGEGraphByPath(graph, "./test_graph_1.txt", ge::DumpLevel::DUMP_ALL);
  ASSERT_EQ(ret, 0);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  state = GraphUtils::LoadGEGraph("./test_graph_1.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), true);
      ASSERT_EQ((ge_tensor_ptr == nullptr), false);
      const TensorData tensor_data = ge_tensor_ptr->GetData();
      const uint8_t *buff = tensor_data.GetData();
      ASSERT_EQ((buff == nullptr), false);
      ASSERT_EQ(buff[0], 7);
      ASSERT_EQ(buff[10], 8);
    }
  }
}

TEST_F(UtestGraphUtils, AddEdgeAnchorPtrIsNull) {
  AnchorPtr src;
  AnchorPtr dst;
  int ret = GraphUtils::AddEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, AddEdgeAnchorPtrSuccess) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  int ret = GraphUtils::AddEdge(node0->GetOutAnchor(0), node1->GetInAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  int ret2 = GraphUtils::AddEdge(node0->GetOutAnchor(0), node1->GetInControlAnchor());
  EXPECT_EQ(ret2, GRAPH_SUCCESS);

  int ret3 = GraphUtils::AddEdge(node0->GetOutControlAnchor(), node1->GetInControlAnchor());
  EXPECT_EQ(ret3, GRAPH_SUCCESS);

  int ret4 = GraphUtils::AddEdge(node0->GetOutControlAnchor(), node1->GetInDataAnchor(0));
  EXPECT_EQ(ret4, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, AddEdgeControlAnchorPtrIsNull) {
  OutControlAnchorPtr src;
  InControlAnchorPtr dst;
  int ret = GraphUtils::AddEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, AddEdgeDataAnchorPtrIsNull) {
  OutDataAnchorPtr src;
  InControlAnchorPtr dst;
  int ret = GraphUtils::AddEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveEdgeAnchorPtrIsNull) {
  AnchorPtr src;
  AnchorPtr dst;
  int ret = GraphUtils::RemoveEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveEdgeOutDataAnchorPtrIsNull) {
  OutDataAnchorPtr src;
  InControlAnchorPtr  dst;
  int ret = GraphUtils::RemoveEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveEdgeFail) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node1, 0);
  builder.AddControlEdge(node0, node1);
  int ret = GraphUtils::RemoveEdge(node0->GetOutDataAnchor(0), node1->GetInControlAnchor());
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeBetweenDataAnchorsSuccess) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  NodePtr new_node(node1);
  builder.AddDataEdge(node0, 0, node2, 0);
  builder.AddControlEdge(node0, node2);
  int ret = GraphUtils::InsertNodeBetweenDataAnchors(node0->GetOutDataAnchor(0),
                                                     node2->GetInDataAnchor(0), new_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, RemoveSubgraphRecursivelyRemoveNodeIsNull) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  NodePtr remove_node;
  int ret = GraphUtils::RemoveSubgraphRecursively(compute_graph, remove_node);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, RemoveSubgraphRecursivelyNodeNotInGraph) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  NodePtr remove_node(node0);
  int ret = GraphUtils::RemoveSubgraphRecursively(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveSubgraphRecursivelyNodeHasNoSubgrah) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  compute_graph->AddNode(node0);
  node0->SetOwnerComputeGraph(compute_graph);
  int ret = GraphUtils::RemoveSubgraphRecursively(compute_graph, node0);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphUtils, RemoveNodeWithoutRelinkNodePtrIsNull) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  NodePtr remove_node;
  int ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveNodeWithoutRelinkFail) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  NodePtr remove_node = ComGraphMakeShared<Node>();
  OpDescPtr op_desc = ComGraphMakeShared<OpDesc>();
  remove_node->impl_->op_ = op_desc;
  compute_graph->AddNode(remove_node);
  // owner graph is null
  int ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 0U);
  // owner graph is another
  ComputeGraphPtr compute_graph_another = std::make_shared<ComputeGraph>("Test1");
  remove_node->SetOwnerComputeGraph(compute_graph_another);
  ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveNodesWithoutRelinkl) {
  auto builder = ut::GraphBuilder("root");
  std::unordered_set<NodePtr> remove_nodes;
  size_t node_size = 6U;
  for (auto i = 0U; i < node_size; i++) {
    auto node = builder.AddNode("node" + std::to_string(i), "Relu", 1, 1);
    if (i == 0U) {
      builder.GetGraph()->AddInputNode(node);
    }
    if (i == node_size - 1U) {
      builder.GetGraph()->AddOutputNode(node);
    }
    remove_nodes.emplace(node);
  }
  EXPECT_TRUE(builder.GetGraph()->GetAllNodesSize() == node_size);
  EXPECT_TRUE(builder.GetGraph()->GetInputNodes().size() == 1U);
  EXPECT_TRUE(builder.GetGraph()->GetOutputNodes().size() == 1U);
  int ret = GraphUtils::RemoveNodesWithoutRelink(builder.GetGraph(), remove_nodes);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_TRUE(builder.GetGraph()->GetAllNodesSize() == 0U);
  EXPECT_TRUE(builder.GetGraph()->GetAllNodes().empty());
  EXPECT_TRUE(builder.GetGraph()->GetInputNodes().empty());
  EXPECT_TRUE(builder.GetGraph()->GetOutputNodes().empty());
}

TEST_F(UtestGraphUtils, InsertNodeAfterAddEdgefail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &node1 = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &node2 = graph_builder0.AddNode("data2", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  std::vector<InDataAnchorPtr> dsts;
  dsts.push_back(node1->GetInDataAnchor(0));
  int ret = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node2, 1, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  int ret2 = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node2, 0, 1);
  EXPECT_EQ(ret2, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeAfterTypeIsSwitch) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", SWITCH, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  std::vector<InDataAnchorPtr> dsts;
  dsts.push_back(node0->GetInDataAnchor(0));
  int ret = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node0, 0, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, InsertNodeAfterSrcOwnerComputeGraphNotEqualDstOwnerComputeGraph) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  std::vector<InDataAnchorPtr> dsts;
  dsts.push_back(node1->GetInDataAnchor(0));
  int ret = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node1, 0, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeBeforeGetOwnerComputeGraphFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  int ret = GraphUtils::InsertNodeBefore(node0->GetInDataAnchor(0), node1, 0, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeBeforeInsertCodeGetInDataAnchorFail) {
  auto builder = ut::GraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  const auto &identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  int ret = GraphUtils::InsertNodeBefore(allreduce->GetInDataAnchor(0), identity, 0, 5);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveJustNodeNodeIsNull) {
  ComputeGraph compute_graph("test_graph0");
  int ret = GraphUtils::RemoveJustNode(compute_graph, nullptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveJustNodeFail) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto graph_builder0 = ut::GraphBuilder("Test0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  int ret = GraphUtils::RemoveJustNode(compute_graph, node0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}


TEST_F(UtestGraphUtils, LoadGEGraphComputeGraphIsNull) {
  char_t *file = nullptr;
  ge::ComputeGraph compute_graph("");
  bool ret = GraphUtils::LoadGEGraph(file, compute_graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, LoadGEGraphFileIsNull) {
  char_t *file = nullptr;
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  bool ret = GraphUtils::LoadGEGraph(file, compute_graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, LoadGEGraphComputeGraphPtrSuccess) {
  const char_t *file = "./test_graph_0.txt";
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("");
  bool ret = GraphUtils::LoadGEGraph(file, compute_graph);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, ReadProtoFromTextFileFileIsNull) {
  google::protobuf::Message *proto = nullptr;
  bool ret = GraphUtils::ReadProtoFromTextFile(nullptr, proto);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, DumpGEGraphToOnnxForLongName) {
  EXPECT_NO_THROW(
    setenv("DUMP_GE_GRAPH", "1", 1);
    ComputeGraph compute_graph("test_graph0");
    const std::string suffit = "ge_proto_00000001_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.pbtxt";
    ge::GraphUtils::DumpGEGraphToOnnx(compute_graph, suffit);
    setenv("DUMP_GE_GRAPH", "1", 1);
  );
}

TEST_F(UtestGraphUtils, IsolateNodeNodeIsNull) {
  NodePtr node;
  std::vector<int> io_map = {1, 2, 3};
  int ret = GraphUtils::IsolateNode(node, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeIoMapIsNull) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  std::vector<int> io_map;
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, IsolateNodeIoMapSizeIsGreaterThanOutDataAnchorsSize) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  std::vector<int> io_map = {1, 2, 3, 4};
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOutDataAnchorsIsNull) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 0);
  std::vector<int> io_map = {1};
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeInDataAnchorsIsNull) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 0, 1);
  std::vector<int> io_map = {1};
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeInitializerListTest) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  std::initializer_list<int> io_map;
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsNodeIsNull) {
  NodePtr new_node;
  NodePtr old_node;
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsReplaceOutDataAnchorsFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map;
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsReplaceInDataAnchorsFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsSuccess) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map;
  std::vector<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodesSuccess_all_data_anchors) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &data0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &relu0 = graph_builder0.AddNode("relu0", "Relu", 1, 1);
  const auto &abs0 = graph_builder0.AddNode("abs0", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data0, 0, relu0, 0);
  graph_builder0.AddDataEdge(relu0, 0, abs0, 0);
  const auto &data1 = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &relu1 = graph_builder0.AddNode("relu1", "Relu", 1, 1);
  const auto &abs1 = graph_builder0.AddNode("abs1", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data1, 0, relu1, 0);
  graph_builder0.AddDataEdge(relu1, 0, abs1, 0);
  const auto &add = graph_builder0.AddNode("add", "Add", 2, 1);
  graph_builder0.AddDataEdge(relu0, 0, add, 0);
  graph_builder0.AddDataEdge(relu1, 0, add, 1);
  const auto &out = graph_builder0.AddNode("out", "NetOutput", 1, 1);
  graph_builder0.AddDataEdge(add, 0, out, 0);

  const auto &relu_abs_add = graph_builder0.AddNode("relu_abs_add", "ReluAbsAdd", 2, 1);
  EXPECT_EQ(graph_builder0.GetGraph()->GetDirectNodesSize(), 9);
  std::vector<int> inputs_map{1, 0};
  std::vector<int> outputs_map{4};
  int ret =
      GraphUtils::ReplaceNodesDataAnchors({relu_abs_add}, {relu0, relu1, abs0, abs1, add}, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 数据关系移动
  EXPECT_EQ(relu_abs_add->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*relu_abs_add->GetOutDataNodes().begin(), out);
  EXPECT_EQ(relu_abs_add->GetInDataNodesSize(), 2U);
  EXPECT_EQ(*relu_abs_add->GetInDataNodes().begin(), data1);
  EXPECT_EQ(*(relu_abs_add->GetInDataNodes().begin() + 1), data0);
  EXPECT_EQ(data0->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*(data0->GetOutDataNodes().begin()), relu_abs_add);
  EXPECT_EQ(data1->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*(data1->GetOutDataNodes().begin()), relu_abs_add);
  EXPECT_EQ(out->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*(out->GetInDataNodes().begin()), relu_abs_add);
  EXPECT_EQ(graph_builder0.GetGraph()->GetDirectNodesSize(), 9);
  const auto &noop_node = graph_builder0.GetGraph()->FindFirstNodeMatchType(NOOP);
  EXPECT_TRUE(noop_node == nullptr);
}

TEST_F(UtestGraphUtils, ReplaceNodesSuccess_all_data_anchors_and_keep_old_in_data_anchors) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &data0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &relu0 = graph_builder0.AddNode("relu0", "Relu", 1, 1);
  const auto &abs0 = graph_builder0.AddNode("abs0", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data0, 0, relu0, 0);
  graph_builder0.AddDataEdge(relu0, 0, abs0, 0);
  const auto &data1 = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &relu1 = graph_builder0.AddNode("relu1", "Relu", 1, 1);
  const auto &abs1 = graph_builder0.AddNode("abs1", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data1, 0, relu1, 0);
  graph_builder0.AddDataEdge(relu1, 0, abs1, 0);
  const auto &add = graph_builder0.AddNode("add", "Add", 2, 1);
  graph_builder0.AddDataEdge(relu0, 0, add, 0);
  graph_builder0.AddDataEdge(relu1, 0, add, 1);
  const auto &out = graph_builder0.AddNode("out", "NetOutput", 1, 1);
  graph_builder0.AddDataEdge(add, 0, out, 0);

  const auto &relu_abs_add = graph_builder0.AddNode("relu_abs_add", "ReluAbsAdd", 2, 1);
  EXPECT_EQ(graph_builder0.GetGraph()->GetDirectNodesSize(), 9);
  std::vector<int> inputs_map{1, 0};
  std::vector<int> outputs_map{4};
  int ret =
      GraphUtils::CopyNodesInDataAnchors({relu_abs_add}, {relu0, relu1, abs0, abs1, add}, inputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret =
      GraphUtils::ReplaceNodesOutDataAnchors({relu_abs_add}, {relu0, relu1, abs0, abs1, add}, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 输出数据关系移动，输入数据关系拷贝
  EXPECT_EQ(relu_abs_add->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*relu_abs_add->GetOutDataNodes().begin(), out);
  EXPECT_EQ(relu_abs_add->GetInDataNodesSize(), 2U);
  EXPECT_EQ(*relu_abs_add->GetInDataNodes().begin(), data1);
  EXPECT_EQ(*(relu_abs_add->GetInDataNodes().begin() + 1), data0);
  EXPECT_EQ(data0->GetOutDataNodesSize(), 2U);
  EXPECT_EQ(*(data0->GetOutDataNodes().begin()), relu0);
  EXPECT_EQ(*(data0->GetOutDataNodes().begin() + 1), relu_abs_add);
  EXPECT_EQ(data1->GetOutDataNodesSize(), 2U);
  EXPECT_EQ(*(data1->GetOutDataNodes().begin()), relu1);
  EXPECT_EQ(*(data1->GetOutDataNodes().begin() + 1), relu_abs_add);
  EXPECT_EQ(out->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*(out->GetInDataNodes().begin()), relu_abs_add);
  EXPECT_EQ(graph_builder0.GetGraph()->GetDirectNodesSize(), 9);
  const auto &noop_node = graph_builder0.GetGraph()->FindFirstNodeMatchType(NOOP);
  EXPECT_TRUE(noop_node == nullptr);
}

TEST_F(UtestGraphUtils, ReplaceNodesSuccess_with_ctrl) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &data0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &relu0 = graph_builder0.AddNode("relu0", "Relu", 1, 1);
  const auto &abs0 = graph_builder0.AddNode("abs0", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data0, 0, relu0, 0);
  graph_builder0.AddDataEdge(relu0, 0, abs0, 0);
  const auto &out = graph_builder0.AddNode("out", "NetOutput", 1, 1);
  graph_builder0.AddDataEdge(abs0, 0, out, 0);
  const auto &relu_abs = graph_builder0.AddNode("relu_abs", "ReluAbs", 1, 1);

  // 创建控制关系
  const auto &const_node0 = graph_builder0.AddNode("const0", CONSTANT, 1, 1);
  const auto &const_node1 = graph_builder0.AddNode("const1", CONSTANT, 1, 1);
  const auto &const_node2 = graph_builder0.AddNode("const2", CONSTANT, 1, 1);

  graph_builder0.AddControlEdge(const_node0, relu0);
  graph_builder0.AddControlEdge(const_node0, abs0);
  graph_builder0.AddControlEdge(const_node1, abs0);
  graph_builder0.AddControlEdge(relu0, abs0);
  graph_builder0.AddControlEdge(relu0, const_node2);

  std::vector<int> inputs_map{0};
  std::vector<int> outputs_map{1};
  int ret = GraphUtils::ReplaceNodesDataAnchors({relu_abs}, {relu0, abs0}, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = GraphUtils::InheritExecutionOrder({relu_abs}, {relu0, abs0}, graph_builder0.GetGraph());
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 数据关系移动
  EXPECT_EQ(relu_abs->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*(relu_abs->GetOutDataNodes().begin()), out);
  EXPECT_EQ(relu_abs->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*relu_abs->GetInDataNodes().begin(), data0);
  EXPECT_EQ(data0->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*(data0->GetOutDataNodes().begin()), relu_abs);
  EXPECT_EQ(out->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*(out->GetInDataNodes().begin()), relu_abs);

  // 控制关系拷贝
  const auto &noop_in = graph_builder0.GetGraph()->FindNode("noop_in_relu_abs");
  EXPECT_TRUE(noop_in != nullptr);
  EXPECT_EQ(noop_in->GetInControlNodesSize(), 2U);
  EXPECT_EQ(*noop_in->GetInControlNodes().begin(), const_node0);
  EXPECT_EQ(relu0->GetInControlNodesSize(), 1U);
  EXPECT_EQ(abs0->GetInControlNodesSize(), 3U);
  EXPECT_EQ(*(noop_in->GetInControlNodes().begin() + 1), const_node1);
  EXPECT_EQ(relu_abs->GetInControlNodesSize(), 1U);
  EXPECT_EQ(*(relu_abs->GetInControlNodes().begin()), noop_in);

  const auto &noop_out = graph_builder0.GetGraph()->FindNode("noop_out_relu_abs");
  EXPECT_TRUE(noop_out != nullptr);
  EXPECT_EQ(noop_out->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(*noop_out->GetOutControlNodes().begin(), const_node2);
  EXPECT_EQ(relu_abs->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(*relu_abs->GetOutControlNodes().begin(), noop_out);
}

TEST_F(UtestGraphUtils, ReplaceNodesSuccess_with_ctrl_and_keep_old_in_data_anchors) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &data0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &relu0 = graph_builder0.AddNode("relu0", "Relu", 1, 1);
  const auto &abs0 = graph_builder0.AddNode("abs0", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data0, 0, relu0, 0);
  graph_builder0.AddDataEdge(relu0, 0, abs0, 0);
  const auto &out = graph_builder0.AddNode("out", "NetOutput", 1, 1);
  graph_builder0.AddDataEdge(abs0, 0, out, 0);
  const auto &relu_abs = graph_builder0.AddNode("relu_abs", "ReluAbs", 1, 1);

  // 创建控制关系
  const auto &const_node0 = graph_builder0.AddNode("const0", CONSTANT, 1, 1);
  const auto &const_node1 = graph_builder0.AddNode("const1", CONSTANT, 1, 1);
  const auto &const_node2 = graph_builder0.AddNode("const2", CONSTANT, 1, 1);

  graph_builder0.AddControlEdge(const_node0, relu0);
  graph_builder0.AddControlEdge(const_node0, abs0);
  graph_builder0.AddControlEdge(const_node1, abs0);
  graph_builder0.AddControlEdge(relu0, abs0);
  graph_builder0.AddControlEdge(relu0, const_node2);

  std::vector<int> inputs_map{0};
  std::vector<int> outputs_map{1};
  int ret = GraphUtils::CopyNodesInDataAnchors({relu_abs}, {relu0, abs0}, inputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = GraphUtils::ReplaceNodesOutDataAnchors({relu_abs}, {relu0, abs0}, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = GraphUtils::InheritExecutionOrder({relu_abs}, {relu0, abs0}, graph_builder0.GetGraph());
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 输出数据关系移动，输入数据关系拷贝
  EXPECT_EQ(relu_abs->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*(relu_abs->GetOutDataNodes().begin()), out);
  EXPECT_EQ(relu_abs->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*relu_abs->GetInDataNodes().begin(), data0);
  EXPECT_EQ(data0->GetOutDataNodesSize(), 2U);
  EXPECT_EQ(*(data0->GetOutDataNodes().begin()), relu0);
  EXPECT_EQ(*(data0->GetOutDataNodes().begin() + 1), relu_abs);
  EXPECT_EQ(out->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*(out->GetInDataNodes().begin()), relu_abs);

  // 控制关系拷贝
  const auto &noop_in = graph_builder0.GetGraph()->FindNode("noop_in_relu_abs");
  EXPECT_TRUE(noop_in != nullptr);
  EXPECT_EQ(noop_in->GetInControlNodesSize(), 2U);
  EXPECT_EQ(*noop_in->GetInControlNodes().begin(), const_node0);
  EXPECT_EQ(relu0->GetInControlNodesSize(), 1U);
  EXPECT_EQ(abs0->GetInControlNodesSize(), 3U);
  EXPECT_EQ(*(noop_in->GetInControlNodes().begin() + 1), const_node1);
  EXPECT_EQ(relu_abs->GetInControlNodesSize(), 1U);
  EXPECT_EQ(*(relu_abs->GetInControlNodes().begin()), noop_in);

  const auto &noop_out = graph_builder0.GetGraph()->FindNode("noop_out_relu_abs");
  EXPECT_TRUE(noop_out != nullptr);
  EXPECT_EQ(noop_out->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(*noop_out->GetOutControlNodes().begin(), const_node2);
  EXPECT_EQ(relu_abs->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(*relu_abs->GetOutControlNodes().begin(), noop_out);
}

TEST_F(UtestGraphUtils, ReplaceNodesSuccess_with_data_convert_ctrl) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &data0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &data1 = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &relu0 = graph_builder0.AddNode("relu0", "2In2OutReluFake", 2, 2);
  const auto &relu1 = graph_builder0.AddNode("relu1", "RELU", 1, 1);
  const auto &abs0 = graph_builder0.AddNode("abs0", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data0, 0, relu0, 0);
  graph_builder0.AddDataEdge(data1, 0, relu0, 1);
  graph_builder0.AddDataEdge(relu0, 0, abs0, 0);
  graph_builder0.AddDataEdge(relu0, 1, relu1, 0);
  const auto &out = graph_builder0.AddNode("out", "NetOutput", 1, 1);
  graph_builder0.AddDataEdge(abs0, 0, out, 0);
  const auto &relu_abs = graph_builder0.AddNode("relu_abs", "ReluAbs", 1, 1);

  // 创建控制关系
  const auto &const_node0 = graph_builder0.AddNode("const0", CONSTANT, 1, 1);
  const auto &const_node1 = graph_builder0.AddNode("const1", CONSTANT, 1, 1);
  const auto &const_node2 = graph_builder0.AddNode("const2", CONSTANT, 1, 1);

  graph_builder0.AddControlEdge(const_node0, relu0);
  graph_builder0.AddControlEdge(const_node0, abs0);
  graph_builder0.AddControlEdge(const_node1, abs0);
  graph_builder0.AddControlEdge(relu0, abs0);
  graph_builder0.AddControlEdge(relu0, const_node2);

  std::vector<int> inputs_map{0};
  std::vector<int> outputs_map{2};
  int ret = GraphUtils::ReplaceNodesDataAnchors({relu_abs}, {relu0, abs0}, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = GraphUtils::InheritExecutionOrder({relu_abs}, {relu0, abs0}, graph_builder0.GetGraph(), true);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 数据关系移动
  EXPECT_EQ(relu_abs->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*(relu_abs->GetOutDataNodes().begin()), out);
  EXPECT_EQ(relu_abs->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*relu_abs->GetInDataNodes().begin(), data0);
  EXPECT_EQ(data0->GetOutDataNodesSize(), 1U);
  EXPECT_EQ(*(data0->GetOutDataNodes().begin()), relu_abs);
  EXPECT_EQ(out->GetInDataNodesSize(), 1U);
  EXPECT_EQ(*(out->GetInDataNodes().begin()), relu_abs);

  // 控制关系拷贝
  const auto &noop_in = graph_builder0.GetGraph()->FindNode("noop_in_relu_abs");
  EXPECT_TRUE(noop_in != nullptr);
  EXPECT_EQ(noop_in->GetInControlNodesSize(), 3U);
  EXPECT_EQ(*noop_in->GetInControlNodes().begin(), const_node0);
  // 非io_map的data1数据输入也被转换为了noop_in的控制边
  EXPECT_EQ(*(noop_in->GetInControlNodes().begin() + 1), data1);
  EXPECT_EQ(*(noop_in->GetInControlNodes().begin() + 2), const_node1);
  EXPECT_EQ(relu0->GetInControlNodesSize(), 1U);
  EXPECT_EQ(abs0->GetInControlNodesSize(), 3U);
  EXPECT_EQ(relu_abs->GetInControlNodesSize(), 1U);
  EXPECT_EQ(*(relu_abs->GetInControlNodes().begin()), noop_in);

  const auto &noop_out = graph_builder0.GetGraph()->FindNode("noop_out_relu_abs");
  EXPECT_TRUE(noop_out != nullptr);
  EXPECT_EQ(noop_out->GetOutControlNodesSize(), 2U);
  EXPECT_EQ(*noop_out->GetOutControlNodes().begin(), const_node2);
  // 非io_map的relu1数据输出也被转换为了noop_out的控制边
  EXPECT_EQ(*(noop_out->GetOutControlNodes().begin() + 1), relu1);
  EXPECT_EQ(relu_abs->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(*relu_abs->GetOutControlNodes().begin(), noop_out);
}

TEST_F(UtestGraphUtils, ReplaceNodesFailed_as_diff_ownergraph) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &data0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &relu0 = graph_builder0.AddNode("relu0", "Relu", 1, 1);
  const auto &abs0 = graph_builder0.AddNode("abs0", "Abs", 1, 1);
  graph_builder0.AddDataEdge(data0, 0, relu0, 0);
  graph_builder0.AddDataEdge(relu0, 0, abs0, 0);
  const auto &out = graph_builder0.AddNode("out", "NetOutput", 1, 1);
  graph_builder0.AddDataEdge(abs0, 0, out, 0);
  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &relu_abs = graph_builder1.AddNode("relu_abs", "ReluAbs", 1, 1);
  std::vector<int> inputs_map{0};
  std::vector<int> outputs_map{0};
  int ret = GraphUtils::ReplaceNodesDataAnchors({relu_abs}, {relu0, abs0}, inputs_map, outputs_map);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIONodeIsNull) {
  NodePtr node;
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIOInDataIs0) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node = graph_builder0.AddNode("data1", DATA, 0, 1);
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIOOutDataIs0) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node = graph_builder0.AddNode("data1", DATA, 1, 0);
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIOSuccess) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node = graph_builder0.AddNode("data1", DATA, 1, 1);
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsNodeIsNull) {
  NodePtr new_node;
  NodePtr old_node;
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsReplaceNodeDataAnchorsFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &new_node = builder.AddNode("data1", "node", 1, 1);
  const auto &old_node = builder.AddNode("data0", "node", 1, 1);
  builder.AddDataEdge(new_node, 0, old_node, 0);
  builder.AddControlEdge(new_node, old_node);
  std::vector<int> inputs_map = {0};
  std::vector<int> outputs_map = {0};
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsInitializerListTest) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &new_node = builder.AddNode("data1", "node", 1, 1);
  const auto &old_node = builder.AddNode("data0", "node", 1, 1);
  builder.AddDataEdge(new_node, 0, old_node, 0);
  builder.AddControlEdge(new_node, old_node);
  std::initializer_list<int> inputs_map;
  std::initializer_list<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsInitializerListTest) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &new_node = builder.AddNode("data1", DATA, 1, 1);
  const auto &old_node = builder.AddNode("data0", DATA, 1, 1);
  std::initializer_list<int> inputs_map;
  std::initializer_list<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyInCtrlEdgesNodeIsNull) {
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, CopyInCtrlEdgesSrcCtrlInNodesIsEmpty) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  NodePtr dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyInCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test");
  const auto &src_node = builder.AddNode("src_node", "node", 1, 1);
  NodePtr dst_node = builder.AddNode("dst_node", "node", 1, 1);
  builder.AddDataEdge(src_node, 0, dst_node, 0);
  builder.AddControlEdge(src_node, dst_node);
  int ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, MoveInCtrlEdgesNodeIsNull) {
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::MoveInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, MoveInCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  NodePtr dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = GraphUtils::MoveInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyOutCtrlEdgesNodeIsNull) {
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::CopyOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyOutCtrlEdgesOutCtrlNodesIsEmpty) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  NodePtr dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = GraphUtils::CopyOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyOutCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("src_node", NETOUTPUT, 1, 1);
  NodePtr dst_node = builder.AddNode("dst_node", NETOUTPUT, 1, 1);
  auto graph = builder.GetGraph();
  builder.AddControlEdge(src_node, dst_node);

  int ret = GraphUtils::CopyOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyOutCtrlEdgesSuccess_with_filter) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("src_node", NETOUTPUT, 1, 1);
  const auto &ctrl_node = builder.AddNode("ctrl_node", CONSTANT, 0, 0);
  const auto &ctrl_node2 = builder.AddNode("ctrl_node2", CONSTANT, 0, 0);
  NodePtr dst_node = builder.AddNode("dst_node", NETOUTPUT, 1, 1);
  auto graph = builder.GetGraph();
  builder.AddControlEdge(src_node, ctrl_node);
  builder.AddControlEdge(src_node, ctrl_node2);
  NodeFilter node_filter = [&](const Node &node) { return node.GetName() == ctrl_node2->GetName(); };
  int ret = GraphUtils::CopyOutCtrlEdges(src_node, dst_node, node_filter);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  EXPECT_EQ(dst_node->GetOutControlNodesSize(), src_node->GetOutControlNodesSize() - 1U);
  EXPECT_EQ(dst_node->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(dst_node->GetOutControlNodes().at(0U), ctrl_node2);
}

TEST_F(UtestGraphUtils, MoveOutCtrlEdgesNodeIsNull) {
  auto builder = ut::GraphBuilder("test_graph0");
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::MoveOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, MoveOutCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  NodePtr src_node = builder.AddNode("src_node", NETOUTPUT, 1, 1);
  NodePtr dst_node = builder.AddNode("dst_node", NETOUTPUT, 1, 1);
  int ret = GraphUtils::MoveOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, AppendInputNodeSuccess) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("Test1");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  int ret = GraphUtils::AppendInputNode(compute_graph, node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyGraphDstrGraphIsNull) {
  Graph src_graph("test0");
  Graph dst_graph("");
  int ret = GraphUtilsEx::CopyGraph(src_graph, dst_graph);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UtestGraphUtils, GetUserInputDatas) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &usr_data_node0 = builder.AddNode("node0", DATA, 1, 1);
  const auto &usr_data_node1 = builder.AddNode("node1", DATA, 1, 1);
  const auto &usr_data_node2 = builder.AddNode("node2", REFDATA, 1, 1);
  const auto &usr_data_node3 = builder.AddNode("node3", AIPPDATA, 1, 1);
  const auto &multi_batch_data = builder.AddNode("ascend_mbatch_shape_data", DATA, 1, 1);
  ge::AttrUtils::SetBool(multi_batch_data->GetOpDesc(), "_is_multi_batch_shape_data", true);
  const auto &const_node = builder.AddNode("const_node0", CONSTANT, 0, 0);
  NodePtr output_node = builder.AddNode("output", NETOUTPUT, 1, 1);
  auto graph = builder.GetGraph();
  const auto input_nodes = ge::GraphUtilsEx::GetUserInputDataNodes(graph);
  EXPECT_EQ(input_nodes.size(), 4);
  for (const auto &node : input_nodes) {
    EXPECT_NE(node->GetName(), "ascend_mbatch_shape_data");
  }
}

TEST_F(UtestGraphUtils, CopyComputeGraphDepthGreaterThanKCopyGraphMaxRecursionDepth) {
  ComputeGraphPtr src_compute_graph = std::make_shared<ComputeGraph>("Test0");
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>("Test1");
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  int32_t depth = 20;
  int ret =
      GraphUtils::CopyComputeGraph(src_compute_graph, dst_compute_graph, node_old_2_new, op_desc_old_2_new, depth);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyMembersSrcComputerGraphIsNull) {
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>("Test1");
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  int ret =
      GraphUtils::CopyMembers(nullptr, dst_compute_graph, all_new_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyMembersDstComputerGraphIsNull) {
  ComputeGraphPtr src_compute_graph = std::make_shared<ComputeGraph>("Test0");
  ComputeGraphPtr dst_compute_graph;
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  int ret = GraphUtils::CopyMembers(src_compute_graph, dst_compute_graph, all_new_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CloneGraph) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node0 = builder.AddNode("node0", DATA, 1, 1);
  const auto &node1 = builder.AddNode("node1", NETOUTPUT, 1, 1);
  auto graph = builder.GetGraph();
  (void) AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  std::string prefix;
  std::vector<NodePtr> input_nodes;
  std::vector<NodePtr> output_nodes;
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  ComputeGraphPtr new_compute_graph = GraphUtils::CloneGraph(graph, prefix, input_nodes, output_nodes);
  EXPECT_NE(new_compute_graph, nullptr);
}

TEST_F(UtestGraphUtils, CopyTensorAttrsDstDescIsNull) {
  OpDescPtr dst_desc;
  auto builder = ut::GraphBuilder("Test1");
  const auto &src_node = builder.AddNode("src_node", DATA, 1, 1);
  int ret = GraphUtils::CopyTensorAttrs(dst_desc, src_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyTensorAttrsSrcNodeIsNull) {
  OpDescPtr dst_desc = std::make_shared<OpDesc>("test", "test");
  NodePtr src_node;
  int ret = GraphUtils::CopyTensorAttrs(dst_desc, src_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyTensorAttrsFail) {
  OpDescPtr dst_desc = std::make_shared<OpDesc>();
  auto builder = ut::GraphBuilder("Test1");
  const auto &src_node = builder.AddNode("src_node", DATA, 1, 1);
  int ret = GraphUtils::CopyTensorAttrs(dst_desc, src_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RelinkGraphEdgesNodeIsNull) {
  NodePtr node;
  std::string prefix;
  std::unordered_map<std::string, NodePtr> all_nodes;
  int ret = GraphUtils::RelinkGraphEdges(node, prefix, all_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RelinkGraphEdgesAllNodesIsNull) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node = builder.AddNode("node", DATA, 1, 1);
  std::string prefix;
  std::unordered_map<std::string, NodePtr> all_nodes;
  int ret = GraphUtils::RelinkGraphEdges(node, prefix, all_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RelinkGraphEdgesOutCtlNotEmpty) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  builder.AddControlEdge(node1, node2);
  std::string prefix;
  std::unordered_map<std::string, NodePtr> all_nodes;
  all_nodes.emplace(node1->GetName(), node1);
  int ret = GraphUtils::RelinkGraphEdges(node1, prefix, all_nodes);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, RelinkGraphEdgesFail) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", DATA, 1, 1);
  const auto &node2 = builder.AddNode("node2", DATA, 1, 1);
  std::string prefix;
  std::unordered_map<std::string, NodePtr> all_nodes;
  all_nodes.insert(make_pair("node2", node2));
  int ret = GraphUtils::RelinkGraphEdges(node1, prefix, all_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, GetRefMappingSuccess) {
  auto builder = ut::GraphBuilder("Test1");
  auto graph = builder.GetGraph();
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, FindNodeFromAllNodesGraphIsNull) {
  ComputeGraphPtr graph;
  std::string name;
  NodePtr node = GraphUtils::FindNodeFromAllNodes(graph, name);
  EXPECT_EQ(node, nullptr);
}

TEST_F(UtestGraphUtils, FindNodeFromAllNodesSuccess) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", DATA, 1, 1);
  auto graph = builder.GetGraph();
  std::string name = "node1";
  NodePtr node = GraphUtils::FindNodeFromAllNodes(graph, name);
  EXPECT_EQ(node->GetName(), "node1");
}

TEST_F(UtestGraphUtils, FindNodeFromAllNodesNameIsNull) {
  auto builder = ut::GraphBuilder("Test1");
  auto graph = builder.GetGraph();
  std::string name;
  NodePtr node = GraphUtils::FindNodeFromAllNodes(graph, name);
  EXPECT_EQ(node, nullptr);
}

TEST_F(UtestGraphUtils, HandleInAnchorMappingSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", NETOUTPUT, 1, 1);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::HandleInAnchorMapping(graph, node1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleInAnchorMappingNodeTypeIsMERGE) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", MERGE, 1, 1);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::HandleInAnchorMapping(graph, node1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleSubgraphInputFail) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", DATA, 1, 1);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::HandleSubgraphInput(node1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, HandleSubgraphInputUpdateRefMappingFail) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  graph->SetParentNode(func);

  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::HandleSubgraphInput(input1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, HandleSubgraphInputSuccess) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(func);

  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::HandleSubgraphInput(input1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleMergeInputPeerOutAnchorIsNull) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(func);

  AttrUtils::SetStr(input1->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, "data1");
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::HandleMergeInput(input1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleMergeInputPeerOutAnchorIsNotNull) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  SymbolToAnchors symbol_to_anchors;
  NodeIndexIO node_index_io(func, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(node_index_io);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));

  AnchorToSymbol anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  int ret = GraphUtils::HandleMergeInput(func, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, HandleSubgraphOutput) {
  auto builder = ut::GraphBuilder("test2");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);

  SymbolToAnchors symbol_to_anchors;
  NodeIndexIO node_index_io(func, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(node_index_io);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));

  AnchorToSymbol anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  int ret = GraphUtils::HandleSubgraphOutput(func, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UtestGraphUtils, UnionSymbolMappingSuccess) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

  SymbolToAnchors symbol_to_anchors;
  NodeIndexIO node_index1(input1, 0, kOut);
  NodeIndexIO node_index2(input2, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(node_index1);
  symbol_list.push_back(node_index2);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var2_out_0", symbol_list));

  AnchorToSymbol anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  anchor_to_symbol.insert(pair<std::string, std::string>("data2_out_0", "var2_out_0"));

  std::string symbol;
  int ret = GraphUtils::UnionSymbolMapping(node_index1, node_index2, symbol_to_anchors, anchor_to_symbol, symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

/*
 * data1 var1 data2 var2
 *    \    |   |   /
 *      \  |  /  /
 *        func
 *         |
 *      netoutput
 */
TEST_F(UtestGraphUtils, UpdateRefMappingFailed) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

  SymbolToAnchors symbol_to_anchors;
  NodeIndexIO cur_node_info(input1, 0, kOut);
  NodeIndexIO exist_node_info(input2, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(cur_node_info);
  symbol_list.push_back(exist_node_info);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var2_out_0", symbol_list));

  AnchorToSymbol anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  anchor_to_symbol.insert(pair<std::string, std::string>("data2_out_0", "var2_out_0"));

  std::string symbol;
  int ret = GraphUtils::UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, UpdateRefMappingSymbolToAnchorsIsNull) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

  NodeIndexIO cur_node_info(input1, 0, kOut);
  NodeIndexIO exist_node_info(input2, 0, kOut);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  anchor_to_symbol.insert(pair<std::string, std::string>("data2_out_0", "var2_out_0"));

  std::string symbol;
  int ret = GraphUtils::UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, IsRefFromInputOutDataAnchorPtrIsNull) {
  OutDataAnchorPtr out_data_anchor;
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(out_data_anchor, reuse_in_index);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsRefFromInputFail) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node0->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsRefFromInputPassThroughOK) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node0 = builder.AddNode("node0", NETOUTPUT, 1, 1);
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node0->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsRefFromInputTypeIsMergeSuccess) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node0 = builder.AddNode("node0", MERGE, 1, 1);
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node0->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsRefFromInputTypeIsReshapeSuccess) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node0 = builder.AddNode("node0", RESHAPE, 1, 1);
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node0->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(reuse_in_index, 0);
}

TEST_F(UtestGraphUtils, IsRefFromInputRefOpFail) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node1 = builder.AddNode("node", "node", 1, 1);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_REFERENCE, true);

  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node1->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsNoPaddingRefFromInputSuccess) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node1 = builder.AddNode("node", "node", 1, 1);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);

  int32_t reuse_in_index;
  bool ret = GraphUtils::IsNoPaddingRefFromInput(node1->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsNodeInGraphRecursivelySuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test0");
  Node node;
  node.SetOwnerComputeGraph(graph);

  bool ret = GraphUtils::IsNodeInGraphRecursively(graph, node);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsNodeInGraphRecursivelyFail) {
  auto builder = ut::GraphBuilder("test0");
  Node node;
  node.SetOwnerComputeGraph(builder.GetGraph());
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  bool ret = GraphUtils::IsNodeInGraphRecursively(graph, node);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsUnknownShapeGraphFail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  bool ret = GraphUtils::IsUnknownShapeGraph(graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsUnknownShapeGraphGraphIsNull) {
  ComputeGraphPtr graph;
  bool ret = GraphUtils::IsUnknownShapeGraph(graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsUnknownShapeGraphSuccess) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("add", "Add", 2, 1, FORMAT_NHWC, DT_FLOAT, {16, 228, 228, 3});
  auto graph = builder.GetGraph();

  auto add_node = graph->FindNode("add");
  auto out_desc = add_node->GetOpDesc()->MutableOutputDesc(0);
  out_desc->SetShape(GeShape({-1, 228, 228, 3}));

  bool ret = GraphUtils::IsUnknownShapeGraph(graph);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, UnfoldSubgraphSuccess) {
  ut::GraphBuilder builder = ut::GraphBuilder("test0");
  auto graph = builder.GetGraph();
  std::function<bool(const ComputeGraphPtr &)> filter;
  int ret = GraphUtils::UnfoldSubgraph(graph, filter);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, MergeInputNodesFail) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node1 = builder.AddNode("node", DATA, 1, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(node1);

  int ret = GraphUtils::MergeInputNodes(graph, node1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, MergeInputNodesWithIndex) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node1 = builder.AddNode("node", DATA, 1, 1);
  AttrUtils::SetInt(node1->GetOpDesc(), "index", 0);
  const auto &abs = builder.AddNode("abs_in_subgraph", "abs", 1, 0);
  builder.AddDataEdge(node1, 0, abs, 0);
  auto graph = builder.GetGraph();
  auto builder1 = ut::GraphBuilder("parent_of_test0");
  const auto &data0 = builder1.AddNode("data", DATA, 1, 1);
  const auto &relu = builder1.AddNode("target_node", "relu", 1, 1);
  builder1.AddDataEdge(data0, 0, relu, 0);
  graph->SetParentNode(relu);

  int ret = GraphUtils::MergeInputNodes(graph, relu);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(data0->GetOutDataNodes().size() == 1U);
  EXPECT_TRUE((*data0->GetOutDataNodes().begin())->GetName() == "abs_in_subgraph");
}

TEST_F(UtestGraphUtils, MergeNetOutputNodeSuccess) {
  auto builder = ut::GraphBuilder("test2");
  const auto &node1 = builder.AddNode("node", DATA, 1, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(node1);

  int ret = GraphUtils::MergeNetOutputNode(graph, node1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphUtils, RemoveJustNodeGraphImplIsNull) {
  ComputeGraph compute_graph("");
  compute_graph.impl_ = nullptr;
  auto graph_builder0 = ut::GraphBuilder("Test0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  int ret = GraphUtils::RemoveJustNode(compute_graph, node0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveJustNodes) {
  auto graph_builder0 = ut::GraphBuilder("Test0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &node1 = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &node2 = graph_builder0.AddNode("data2", DATA, 1, 1);
  EXPECT_EQ(graph_builder0.GetGraph()->GetDirectNodesSize(), 3U);
  std::unordered_set<NodePtr> remove_nodes;
  remove_nodes.insert(node0);
  remove_nodes.insert(node1);
  EXPECT_EQ(GraphUtils::RemoveJustNodes(graph_builder0.GetGraph(), remove_nodes), GRAPH_SUCCESS);
  EXPECT_EQ(graph_builder0.GetGraph()->GetDirectNodesSize(), 1U);
  // remove nodes not in graph, also return success
  EXPECT_EQ(GraphUtils::RemoveJustNodes(graph_builder0.GetGraph(), remove_nodes), GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, GetNodeFail) {
  UtestComputeGraphBuilder graph;
  NodePtr node_ptr = graph.GetNode("node1");
  EXPECT_EQ(node_ptr, nullptr);
}

TEST_F(UtestGraphUtils, GetAllNodeNodeSizeIs0) {
  UtestComputeGraphBuilder graph;
  std::vector<NodePtr> node_ptr = graph.GetAllNodes();
  EXPECT_EQ(node_ptr.size(), 0);
}

TEST_F(UtestGraphUtils, BuildExistNodesTest) {
  PartialGraphBuilder builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";
  builder.BuildExistNodes(err, msg);
  EXPECT_TRUE(err == GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");

  builder.exist_nodes_.push_back(nullptr);
  builder.BuildExistNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  builder.exist_nodes_.clear();
  auto gbuilder = ut::GraphBuilder("test2");
  auto node = gbuilder.AddNode("node", DATA, 1, 1);
  auto opdsc = std::make_shared<OpDesc>("node1", "node");
  builder.AddExistNode(node);
  builder.AddNode(opdsc);
  EXPECT_EQ(builder.exist_nodes_.size(), 1);
  builder.BuildExistNodes(err, msg);
  EXPECT_TRUE(err ==  GRAPH_FAILED);
  EXPECT_NE(msg, "");

  err = GRAPH_SUCCESS;
  msg = "";
  builder.owner_graph_ = node->GetOwnerComputeGraph();
  builder.BuildExistNodes(err, msg);
  EXPECT_TRUE(err == GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
}

TEST_F(UtestGraphUtils, PartialGraphBuilderBuildTest) {
  PartialGraphBuilder par_graph_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";
  ComputeGraphPtr computer_graph;
  computer_graph = par_graph_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "graph is NULL.");
  EXPECT_EQ(computer_graph, nullptr);

  auto builder = ut::GraphBuilder("test1");
  auto node = builder.AddNode("node", DATA, 1, 1);
  par_graph_builder.SetOwnerGraph(node->GetOwnerComputeGraph());
  computer_graph = par_graph_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "graph is NULL.");
  EXPECT_EQ(computer_graph, nullptr);
}

TEST_F(UtestGraphUtils, CompleteGraphBuilderBuilder) {
  CompleteGraphBuilder complete_builder("");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  complete_builder.Build(err, msg);
  EXPECT_TRUE(err ==  GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
}

TEST_F(UtestGraphUtils, CompleteGraphBuilderBuildGraphTargets) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  //node_names_ is null
  complete_builder.AddTarget("Data_1");
  complete_builder.BuildGraphTargets(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
}

TEST_F(UtestGraphUtils, BuildNetOutputNodeWithLinkTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";
  auto builder = ut::GraphBuilder("test2");
  auto node = builder.AddNode("node", DATA, 1, 1);
  auto node2 = builder.AddNode("node2", NETOUTPUT, 1, 0);
  complete_builder.owner_graph_ = node->GetOwnerComputeGraph();

  OpDescPtr net_output_desc;
  std::vector<OutDataAnchorPtr> peer_out_anchors;
  complete_builder.BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  err = GRAPH_SUCCESS;
  msg = "";
  net_output_desc = std::make_shared<OpDesc>("test", "test");
  complete_builder.AddTarget("Data_1");
  complete_builder.BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  err = GRAPH_SUCCESS;
  msg = "";
  uint32_t index = 1;
  complete_builder.input_mapping_.insert(pair<uint32_t, uint32_t>(1, 0));
  auto ret_node = complete_builder.AddDataNode(index, err, msg);
  EXPECT_EQ(ret_node, complete_builder.node_names_["Data_1"]);
  complete_builder.BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, err, msg);
  EXPECT_TRUE(err == GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
}

TEST_F(UtestGraphUtils, AddDataNodeTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test2");
  auto node = builder.AddNode("node", DATA, 1, 1);

  uint32_t index = 1;
  complete_builder.input_mapping_.insert(pair<uint32_t, uint32_t>(1, 1));
  complete_builder.owner_graph_ = node->GetOwnerComputeGraph();

  auto ret_node = complete_builder.AddDataNode(index, err, msg);
  EXPECT_TRUE(err == GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
  EXPECT_EQ(ret_node, complete_builder.node_names_["Data_1"]);
}

TEST_F(UtestGraphUtils, AddNetOutputNodeTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  // graph_outputs_ and graph_targets_ is null
  complete_builder.AddNetOutputNode(err, msg);
  EXPECT_TRUE(err == GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");

  // node_names_ is null
  complete_builder.AddTarget("out");
  complete_builder.graph_outputs_.push_back(pair<std::string, uint32_t>("out", 0));
  complete_builder.AddNetOutputNode(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  // node is nullptr
  err = GRAPH_SUCCESS;
  msg = "";
  complete_builder.node_names_.insert(pair<std::string, NodePtr>("out", nullptr));
  complete_builder.AddNetOutputNode(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "AddNetOutputNode failed: node is NULL.");
  err = GRAPH_SUCCESS;
  msg = "";
  auto compute_graph = ge::ComGraphMakeSharedAndThrow<ComputeGraph>("test");
  complete_builder.owner_graph_ = compute_graph;
  auto data_node = compute_graph->AddNode(OpDescBuilder("out", "Relu").AddInput("x").AddOutput("y").Build());
  complete_builder.node_names_["out"] = data_node;
  complete_builder.output_mapping_.emplace(0, 0);
  complete_builder.AddNetOutputNode(err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
}

TEST_F(UtestGraphUtils, AddRetValNodesTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  //node_names_ is null
  complete_builder.graph_outputs_.push_back(pair<std::string, uint32_t>("Data_1", 0));
  complete_builder.AddRetValNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "AddRetValNode failed: node Data_1 does not exist in graph.");

  //node_names_ node is nullptr
  err = GRAPH_SUCCESS;
  msg = "";
  complete_builder.node_names_.insert(pair<std::string, NodePtr>("Data_1", nullptr));
  complete_builder.AddRetValNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "AddRetValNode failed: node is NULL.");

  //node_names_ node is not nullptr
  auto builder = ut::GraphBuilder("test2");
  auto node = builder.AddNode("node", DATA, 1, 0);
  complete_builder.owner_graph_ = node->GetOwnerComputeGraph();

  complete_builder.node_names_.clear();
  complete_builder.node_names_.insert(pair<std::string, NodePtr>("Data_1", node));
  complete_builder.output_mapping_.insert(pair<uint32_t, uint32_t>(0, 0));
  err = GRAPH_SUCCESS;
  msg = "";
  complete_builder.AddRetValNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
}

TEST_F(UtestGraphUtils, BuildCtrlLinksTest) {
  PartialGraphBuilder par_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test1");
  auto node = builder.AddNode("node_input", DATA, 1, 1);
  auto node2 = builder.AddNode("node_output", NETOUTPUT, 1, 1);
  par_builder.SetOwnerGraph(node->GetOwnerComputeGraph());

  par_builder.AddControlLink("node_input", "node_output");
  ComputeGraphPtr graph;
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);

  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_input", nullptr));
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_output", nullptr));
  err = GRAPH_SUCCESS;
  msg = "";
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);

  par_builder.node_names_.clear();
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_input", node));
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_output", node2));
  err = GRAPH_SUCCESS;
  msg = "";
  graph = par_builder.Build(err, msg);
  EXPECT_TRUE(err == GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
  EXPECT_EQ(graph, node->GetOwnerComputeGraph());
}

TEST_F(UtestGraphUtils, BuildDataLinksTest) {
  PartialGraphBuilder par_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test1");
  auto node = builder.AddNode("node_input", DATA, 1, 1);
  auto node2 = builder.AddNode("node_output", NETOUTPUT, 1, 1);
  par_builder.SetOwnerGraph(node->GetOwnerComputeGraph());

  par_builder.AddDataLink("node_input", 1, "node_output", 1);
  ComputeGraphPtr graph;
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);

  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_input", nullptr));
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_output", nullptr));
  err = GRAPH_SUCCESS;
  msg = "";
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);
}

TEST_F(UtestGraphUtils, PostProcessTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test2");
  auto node1 = builder.AddNode("node1", DATA, 1, 1);
  auto owner_graph = node1->GetOwnerComputeGraph();
  complete_builder.owner_graph_ = owner_graph;

  auto builder2 = ut::GraphBuilder("test3");
  auto node2 = builder2.AddNode("node", "node", 1, 1);
  complete_builder.parent_node_ = node2;
  auto parent_graph = complete_builder.parent_node_->GetOwnerComputeGraph();

  std::string graph_id;
  AttrUtils::SetStr(parent_graph, ATTR_NAME_SESSION_GRAPH_ID, graph_id);

  AnyValue any_value;
  any_value.SetValue(1);
  complete_builder.parent_node_->GetOwnerComputeGraph()->SetAttr(ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, any_value);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);

  complete_builder.PostProcess(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "Copy attr _dynamic_shape_partitioned failed.");
}


TEST_F(UtestGraphUtils, GetRefMappingTest) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test0");
  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  graph->AddNode(op_desc);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ComputeGraphBuilderBuildNodesTest) {
  UtestComputeGraphBuilder utest_graph_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  //owner_graph_ is null
  utest_graph_builder.BuildNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "graph is NULL.");

  //nodes_ is null
  auto builder = ut::GraphBuilder("test1");
  auto node1 = builder.AddNode("node1", DATA, 1, 1);
  auto owner_graph = node1->GetOwnerComputeGraph();
  utest_graph_builder.owner_graph_ = owner_graph;
  err = GRAPH_SUCCESS;
  msg = "";
  utest_graph_builder.nodes_.push_back(nullptr);
  utest_graph_builder.BuildNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "op_desc is NULL.");
}


TEST_F(UtestGraphUtils, FindNodeByTypeFromAllGraphs) {
  auto graph = BuildGraphWithSubGraph();
  ASSERT_NE(graph, nullptr);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "Data");
  EXPECT_EQ(nodes.size(), 3);
  const auto &bare_nodes = GraphUtils::FindBareNodesByTypeFromAllNodes(graph, "Data");
  EXPECT_EQ(bare_nodes.size(), 3);
  EXPECT_EQ(nodes.at(0).get(), bare_nodes.at(0));
  EXPECT_EQ(nodes.at(1).get(), bare_nodes.at(1));
  EXPECT_EQ(nodes.at(2).get(), bare_nodes.at(2));
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkPlaceholder) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_placeholder");
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, "PlaceHolder");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "PlaceHolder");
  EXPECT_EQ(nodes.size(), 0);
  const auto &bare_nodes = GraphUtils::FindBareNodesByTypeFromAllNodes(graph, "PlaceHolder");
  EXPECT_EQ(bare_nodes.size(), 0);
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkEnd) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_end");
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, "End");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "End");
  EXPECT_EQ(nodes.size(), 0);
  const auto &bare_nodes = GraphUtils::FindBareNodesByTypeFromAllNodes(graph, "End");
  EXPECT_EQ(bare_nodes.size(), 0);
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkAdd) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_end");
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, "Add");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "Add");
  EXPECT_EQ(nodes.size(), 0);
  const auto &bare_nodes = GraphUtils::FindBareNodesByTypeFromAllNodes(graph, "Add");
  EXPECT_EQ(bare_nodes.size(), 0);
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkData) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_end");
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, DATA);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, DATA);
  EXPECT_EQ(nodes.size(), 0);
  const auto &bare_nodes = GraphUtils::FindBareNodesByTypeFromAllNodes(graph, "Data");
  EXPECT_EQ(bare_nodes.size(), 0);
}

TEST_F(UtestGraphUtils, FindNodeByTypeFromAllGraphsNullInput) {
  ComputeGraphPtr graph = nullptr;
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "Data");
  EXPECT_EQ(nodes.size(), 0);
  const auto &bare_nodes = GraphUtils::FindBareNodesByTypeFromAllNodes(graph, "Data");
  EXPECT_EQ(bare_nodes.size(), 0);
}
namespace {
void CheckAnchor(const std::list<NodeIndexIO> &all_anchors_of_symbol,
                 const std::unordered_set<std::string> &expect_anchors) {
  for (auto iter_e = all_anchors_of_symbol.begin(); iter_e != all_anchors_of_symbol.end(); ++iter_e) {
    EXPECT_EQ(expect_anchors.count((*iter_e).ToString()), 1);
  }
}

void PrintAnchors(const SymbolToAnchors &symbol_to_anchors) {
  std::stringstream ss;
  for (const auto &pair : symbol_to_anchors) {
    ss << pair.first << " : ";
    ss << "[ ";
    for (const auto &anchor : pair.second) {
      ss << anchor.ToString() << "|";
    }
    ss << " ]";
  }
  std::cout << ss.str() << std::endl;
}
}  // namespace
/*
   refdata(a) const(b)
     \       /
       assign
         |(a)
         |
      transdata
         |(a)
         |
     netoutput
*/
TEST_F(UtestGraphUtils, GetRefMappingWithRefData) {
  auto builder = ut::GraphBuilder("test1");
  const auto &refdata = builder.AddNode("refdata", REFDATA, 1, 1);
  const auto &const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 2, 1);
  const auto &transdata = builder.AddNode("transdata", "TransData", 1, 1);
  AttrUtils::SetStr(transdata->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, "refdata");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(refdata, 0, assign, 0);
  builder.AddDataEdge(const1, 0, assign, 1);
  builder.AddDataEdge(assign, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 当前图共5个symbol
  EXPECT_EQ(symbol_to_anchors.size(), 5);
  PrintAnchors(symbol_to_anchors);
  // 校验transdata输出和refdata共享一个symbol
  NodeIndexIO transdata_out_info(transdata, 0, kOut);
  auto iter = anchor_to_symbol.find(transdata_out_info.ToString());
  EXPECT_NE(iter, anchor_to_symbol.end());
  std::string symbol_transdata = iter->second;

  NodeIndexIO refdata_info(refdata, 0, kOut);
  iter = anchor_to_symbol.find(refdata_info.ToString());
  EXPECT_NE(iter, anchor_to_symbol.end());
  std::string symbol_ref_data = iter->second;

  EXPECT_STREQ(symbol_transdata.c_str(), symbol_ref_data.c_str());

  // 校验图中refdata的symbol, 有4个tensor共享
  auto iter_a = symbol_to_anchors.find(symbol_transdata);
  EXPECT_NE(iter_a, symbol_to_anchors.end());
  EXPECT_EQ(iter_a->second.size(), 4);

  NodeIndexIO assing_in_0_info(assign, 0, kIn);
  NodeIndexIO netoutput_in_0_info(netoutput, 0, kIn);
  std::unordered_set<std::string> expect_anchors_set{refdata_info.ToString(), transdata_out_info.ToString(),
                                                     assing_in_0_info.ToString(), netoutput_in_0_info.ToString()};
  CheckAnchor(iter_a->second, expect_anchors_set);
}
/*
  refdata(a)
    |
  identity  const(b)
     \       /
       assign
         |(a)
         |
      transdata
         |
         |
     netoutput
  如果refdata和assign中间插入了identity, HandleOutAnchorMapping中，assign上带有ref_var_src_var_name值是refdata，但是同时
  assign是输出引用输入，导致符号建立错误。
*/
TEST_F(UtestGraphUtils, GetRefMappingWithRefData_Failed_BecauseInsertIdentity) {
  auto builder = ut::GraphBuilder("test1");
  const auto &refdata = builder.AddNode("refdata", REFDATA, 0, 1);
  const auto &identity = builder.AddNode("identity", IDENTITY, 1, 1);
  const auto &const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 2, 1);
  const auto &transdata = builder.AddNode("transdata", "TransData", 1, 1);
  AttrUtils::SetStr(assign->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, "refdata");
  AttrUtils::SetBool(assign->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  assign->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  assign->GetOpDesc()->MutableAllOutputName() = {{"x", 0}};
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(refdata, 0, identity, 0);
  builder.AddDataEdge(identity, 0, assign, 0);
  builder.AddDataEdge(const1, 0, assign, 1);
  builder.AddDataEdge(assign, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}
/*
  refdata(a)  const(b)
     \       /
       assign
         |(a)
         |
      transdata
         |
         |
     netoutput
*/
TEST_F(UtestGraphUtils, GetRefMappingWithRefData_Success) {
  auto builder = ut::GraphBuilder("test1");
  const auto &refdata = builder.AddNode("refdata", REFDATA, 0, 1);
  const auto &const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 2, 1);
  const auto &transdata = builder.AddNode("transdata", "TransData", 1, 1);
  AttrUtils::SetStr(assign->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, "refdata");
  AttrUtils::SetBool(assign->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  assign->GetOpDesc()->MutableAllInputName() = {{"x", 0}};
  assign->GetOpDesc()->MutableAllOutputName() = {{"x", 0}};
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(refdata, 0, assign, 0);
  builder.AddDataEdge(const1, 0, assign, 1);
  builder.AddDataEdge(assign, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}
/*
   data   data
     \       /
       merge
         |
         |
      cast
         |
         |
     netoutput
*/
TEST_F(UtestGraphUtils, GetRefMappingWithMergeOp) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 当前图共2个symbol
  EXPECT_EQ(symbol_to_anchors.size(), 2);
  PrintAnchors(symbol_to_anchors);
  // 校验merge输出和input1,input2共享一个symbol
  NodeIndexIO merge_out(merge, 0, kOut);
  auto iter = anchor_to_symbol.find(merge_out.ToString());
  EXPECT_NE(iter, anchor_to_symbol.end());
  std::string symbol_merge = iter->second;

  NodeIndexIO input1_info(input1, 0, kOut);
  iter = anchor_to_symbol.find(input1_info.ToString());
  EXPECT_NE(iter, anchor_to_symbol.end());
  std::string symbol_input1 = iter->second;
  EXPECT_STREQ(symbol_merge.c_str(), symbol_input1.c_str());

  NodeIndexIO input2_info(input2, 0, kOut);
  iter = anchor_to_symbol.find(input2_info.ToString());
  EXPECT_NE(iter, anchor_to_symbol.end());
  std::string symbol_input2 = iter->second;
  EXPECT_STREQ(symbol_merge.c_str(), symbol_input2.c_str());
}

TEST_F(UtestGraphUtils, GetRefMappingWithSubgraphOp) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1);
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 1, 1);
  const auto &out = root_builder.AddNode("out", NETOUTPUT, 1, 1);
  root_builder.AddDataEdge(data, 0, partitioncall_0, 0);
  root_builder.AddDataEdge(partitioncall_0, 0, out, 0);
  const auto &root_graph = root_builder.GetGraph();

  int64_t index = 0;
  auto sub_builder = ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_data = sub_builder.AddNode("partitioncall_0_data", DATA, 1, 1);
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", index);
  const auto &partitioncall_0_cast = sub_builder.AddNode("partitioncall_0_cast", "Cast", 1, 1);
  const auto &partitioncall_0_netoutput = sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", index);
  sub_builder.AddDataEdge(partitioncall_0_data, 0, partitioncall_0_cast, 0);
  sub_builder.AddDataEdge(partitioncall_0_cast, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph("partitioncall_0_sub", sub_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("partitioncall_0_sub");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");
  NodePtr node = GraphUtils::FindNodeFromAllNodes(const_cast<ComputeGraphPtr &>(root_graph), "partitioncall_0_cast");
  EXPECT_NE(node, nullptr);
  node = GraphUtils::FindNodeFromAllNodes(const_cast<ComputeGraphPtr &>(sub_graph), "partitioncall_0_cast");
  EXPECT_NE(node, nullptr);
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(root_graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 当前图共2个symbol
  EXPECT_EQ(symbol_to_anchors.size(), 2);
  PrintAnchors(symbol_to_anchors);
  // 校验partitioncall_0输出和partitioncall_0_cast输出,partitioncall_0_netoutput的输入输出，out的输入输出共享一个symbol
  NodeIndexIO partitioncall_0_out(partitioncall_0, 0, kOut);
  auto iter = anchor_to_symbol.find(partitioncall_0_out.ToString());
  EXPECT_NE(iter, anchor_to_symbol.end());
  std::string symbol_partitioncall_0_out = iter->second;
  auto iter_a = symbol_to_anchors.find(symbol_partitioncall_0_out);
  EXPECT_NE(iter_a, symbol_to_anchors.end());
  EXPECT_EQ(iter_a->second.size(), 6U);
  std::unordered_set<std::string> expect_anchors{partitioncall_0_out.ToString()};
  NodeIndexIO partitioncall_0_cast_out(partitioncall_0_cast, 0, kOut);
  expect_anchors.emplace(partitioncall_0_cast_out.ToString());
  NodeIndexIO partitioncall_0_netoutput_out(partitioncall_0_netoutput, 0, kOut);
  expect_anchors.emplace(partitioncall_0_netoutput_out.ToString());
  NodeIndexIO partitioncall_0_netoutput_in(partitioncall_0_netoutput, 0, kIn);
  expect_anchors.emplace(partitioncall_0_netoutput_in.ToString());
  NodeIndexIO out_in(out, 0, kIn);
  expect_anchors.emplace(out_in.ToString());
  NodeIndexIO out_out(out, 0, kOut);
  expect_anchors.emplace(out_out.ToString());
  CheckAnchor(iter_a->second, expect_anchors);
}

TEST_F(UtestGraphUtils, InfershapeIfNeedOk) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1, FORMAT_NHWC, DT_FLOAT, {16, 228, 228, 3});
  auto cast = builder.AddNode("cast", "Cast", 1, 1, FORMAT_NHWC, DT_FLOAT, {16, 228, 228, 3});
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 1, FORMAT_NHWC, DT_FLOAT, {5, 228, 228, 3});
  AttrUtils::SetBool(cast->GetOpDesc(), "isNeedInfer", true);
  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  cast->GetOpDesc()->AddInferFunc(stub_func);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  builder.AddDataEdge(data, 0, cast, 0);
  builder.AddDataEdge(cast, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  EXPECT_EQ(GraphUtilsEx::InferShapeInNeed(graph), GRAPH_SUCCESS);
  std::vector<int64_t> expect_shape = {16, 228, 228, 3};
  EXPECT_EQ(netoutput->GetOpDesc()->GetInputDesc(0).GetShape().GetDims(), expect_shape);
  int64_t parent_node_index = -1;
  AttrUtils::GetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, parent_node_index);
  EXPECT_EQ(parent_node_index, 0);
}

TEST_F(UtestGraphUtils, LoadGraph_parse_fail) {
  const std::string file_name = "./test.txt";
  system(("touch " + file_name).c_str());
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph(file_name.c_str(), *com_graph1);
  ASSERT_EQ(state, false);
  state = GraphUtils::LoadGEGraph(file_name.c_str(), com_graph1);
  ASSERT_EQ(state, false);
  state = GraphUtils::LoadGEGraph(nullptr, *com_graph1);
  ASSERT_EQ(state, false);
  state = GraphUtils::LoadGEGraph(nullptr, com_graph1);
  ASSERT_EQ(state, false);
  system(("rm -f " + file_name).c_str());
}

TEST_F(UtestGraphUtils, InsertNodeBeforeOpdesc) {
  // build test graph
  auto builder = ut::GraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
 // const auto &identity = builder.AddNode("identity", "Identity", 1, 1);
  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var, 0, allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();
  auto identity_op_desc = ge::MakeShared<ge::OpDesc>("temp", "Identity");
  ASSERT_NE(identity_op_desc, nullptr);
  identity_op_desc->AddInputDesc(ge::GeTensorDesc());
  identity_op_desc->AddOutputDesc(ge::GeTensorDesc());

  std::string expect_super_kernel_scope = "_test";
  (void)AttrUtils::SetStr(allreduce->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_SCOPE, expect_super_kernel_scope);
  // insert identity before allreduce
  auto identity_node = GraphUtils::InsertNodeBefore(allreduce->GetInDataAnchor(0), identity_op_desc, 0, 0);
  ASSERT_NE(identity_node, nullptr);
  std::string super_kernel_scope;
  (void)AttrUtils::GetStr(identity_node->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_SCOPE, super_kernel_scope);
  EXPECT_EQ(super_kernel_scope, expect_super_kernel_scope);
  // check assign control-in on identity
  ASSERT_EQ(identity_node->GetInControlNodes().at(0)->GetName(), "assign");
  ASSERT_EQ(identity_node->GetInDataNodes().at(0)->GetName(), "var");
  // check atomicclean control-in still on allreuce
  ASSERT_EQ(allreduce->GetInControlNodes().at(0)->GetName(), "atomic_clean");
  ASSERT_EQ(allreduce->GetInDataNodes().at(0)->GetName(), "temp");
}

TEST_F(UtestGraphUtils, InsertNodeAfterTypeIsSwitchOpDesc) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", SWITCH, 1, 1);
  const auto &node1 = graph_builder0.AddNode("all_reduce", "HcomAllReduce", 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  graph_builder0.AddDataEdge(node0, 0, node1, 0);
  std::vector<InDataAnchorPtr> dsts;
  dsts.push_back(node1->GetInDataAnchor(0));
  auto identity_op_desc = ge::MakeShared<ge::OpDesc>("temp", "Identity");
  ASSERT_NE(identity_op_desc, nullptr);
  identity_op_desc->AddInputDesc(ge::GeTensorDesc());
  identity_op_desc->AddOutputDesc(ge::GeTensorDesc());
  std::string expect_usr_stream_label = "_test";
  (void)AttrUtils::SetStr(node0->GetOpDesc(), public_attr::USER_STREAM_LABEL, expect_usr_stream_label);
  auto identity_node = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, identity_op_desc, 0, 0);
  ASSERT_NE(identity_node, nullptr);
  std::string usr_stream_label;
  (void)AttrUtils::GetStr(identity_node->GetOpDesc(), public_attr::USER_STREAM_LABEL, usr_stream_label);
  EXPECT_EQ(usr_stream_label, expect_usr_stream_label);
  ASSERT_EQ(node1->GetInDataNodes().at(0)->GetName(), "temp");
  ASSERT_EQ(identity_node->GetInDataNodes().at(0)->GetName(), "data0");
}

TEST_F(UtestGraphUtils, Single_output_2_multi_inputs) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data1 = builder.AddNode("Data1", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto relu1 = builder.AddNode("Relu1", "Relu", 1, 1);
  auto relu2 = builder.AddNode("Relu2", "Relu", 1, 1);
  auto relu3 = builder.AddNode("Relu3", "Relu", 1, 1);
  auto relu4 = builder.AddNode("Relu4", "Relu", 1, 1);
  auto relu5 = builder.AddNode("Relu5", "Relu", 1, 1);
  auto relu6 = builder.AddNode("Relu6", "Relu", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 3, 0);
  builder.AddDataEdge(data1, 0, add_node, 0);
  builder.AddDataEdge(data2, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, relu1, 0);
  builder.AddDataEdge(add_node, 0, relu2, 0);
  builder.AddDataEdge(add_node, 0, relu3, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  builder.AddDataEdge(relu2, 0, netoutput, 0);
  builder.AddDataEdge(relu3, 0, netoutput, 0);
  builder.AddControlEdge(add_node, relu4);
  builder.AddControlEdge(add_node, relu5);
  builder.AddControlEdge(add_node, relu6);
  builder.AddControlEdge(relu4, netoutput);
  builder.AddControlEdge(relu5, netoutput);
  builder.AddControlEdge(relu6, netoutput);
  auto graph = builder.GetGraph();

  std::vector<std::string> expected_dfs_names =
      {"Data1", "Data2", "Add", "Relu6", "Relu5", "Relu4", "Relu3", "Relu2", "Relu1", "Netoutput"};
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> dfs_names;
  for (auto &node : graph->GetAllNodes()) {
    dfs_names.push_back(node->GetName());
  }
  EXPECT_EQ(dfs_names, expected_dfs_names);

  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  (void)setenv(kDumpGraphLevel, "1", 1);
  const char_t *const kDumpGeGraph = "DUMP_GE_GRAPH";
  (void)setenv(kDumpGeGraph, "2", 1);

  GraphUtils::DumpGEGraph(graph, "", true, "./ge_test_graph_single_output_2_multi_inputs.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph("./ge_test_graph_single_output_2_multi_inputs.txt", com_graph1);
  EXPECT_EQ(state, true);
  EXPECT_EQ(com_graph1->TopologicalSorting(), GRAPH_SUCCESS);
  dfs_names.clear();
  for (auto &node : graph->GetAllNodes()) {
    dfs_names.push_back(node->GetName());
  }
  EXPECT_EQ(dfs_names, expected_dfs_names);
  system("rm -f ./ge_test*.txt");
}

TEST_F(UtestGraphUtils, CanReplace_no_attr) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);

  auto op_desc = merge->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 0U);
}

TEST_F(UtestGraphUtils, CanReplace_invalid_attr) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);

  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0, 0}});
  auto op_desc = merge->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, FAILED);
  EXPECT_EQ(inplace_index_list.size(), 0U);
}

TEST_F(UtestGraphUtils, CanReplace_Invalid_input_index) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 2}});

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphUtils, CanReplace_Invalid_output_index) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {1, 0}});

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphUtils, CanReplace_Diff_streamid) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 2);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  const auto &atomicclean = builder.AddNode("atomicclean", ATOMICADDRCLEAN, 1, 0);
  atomicclean->GetOpDesc()->SetStreamId(10);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 1}});
  auto op_desc = merge->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  op_desc->SetStreamId(1);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input1, 1, atomicclean, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 0U);
}

TEST_F(UtestGraphUtils, CanReplace_Invalid_streamid) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 2);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  const auto &atomicclean = builder.AddNode("atomicclean", ATOMICADDRCLEAN, 1, 0);
  atomicclean->GetOpDesc()->SetStreamId(-1);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 1}});
  auto op_desc = merge->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  op_desc->SetStreamId(1);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input1, 1, atomicclean, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 0U);
}

TEST_F(UtestGraphUtils, CanReplace_special_streamid) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 2);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  const auto &atomicclean = builder.AddNode("atomicclean", ATOMICADDRCLEAN, 1, 0);
  input1->GetOpDesc()->SetStreamId(-1);
  input2->GetOpDesc()->SetStreamId(-1);
  atomicclean->GetOpDesc()->SetStreamId(-1);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 1}});
  auto op_desc = merge->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  op_desc->SetStreamId(-1);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input1, 1, atomicclean, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 1U);
  EXPECT_EQ(inplace_index_list[0].size(), 2U);
}

TEST_F(UtestGraphUtils, CanReplace_Not_max_topid) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 2);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  const auto &atomicclean = builder.AddNode("atomicclean", ATOMICADDRCLEAN, 1, 1);

  auto op_desc = atomicclean->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input1, 1, atomicclean, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(atomicclean, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 0U);
}

TEST_F(UtestGraphUtils, CanReplace_Out_node_has_ref_attr) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 2);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  const auto &atomicclean = builder.AddNode("atomicclean", ATOMICADDRCLEAN, 1, 1);

  AttrUtils::SetBool(atomicclean->GetOpDesc(), ATTR_NAME_REFERENCE, true);

  // 设置属性
  auto op_desc = merge->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input1, 1, atomicclean, 0);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 0U);
}

TEST_F(UtestGraphUtils, CanReplace_fusion_node) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  AttrUtils::SetGraph(merge->GetOpDesc(), "_original_fusion_graph", builder.GetGraph());

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 1U);
  EXPECT_EQ(inplace_index_list[0].size(), 1U);
  EXPECT_EQ(inplace_index_list[0][0], 0U);
}

/*
   data   data
     \       /
       merge
         |
         |
      cast
         |
         |
     netoutput
*/
TEST_F(UtestGraphUtils, CanReplace_Success) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  auto op_desc = merge->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 1U);
  EXPECT_EQ(inplace_index_list[0].size(), 1U);
  EXPECT_EQ(inplace_index_list[0][0], 0U);
}

TEST_F(UtestGraphUtils, CanReplace_2inputs_Success) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &merge = builder.AddNode("merge", MERGE, 2, 1);
  // 设置属性
  AttrUtils::SetListListInt(merge->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 1}});
  auto op_desc = merge->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  const auto &cast = builder.AddNode("cast", "CAST", 1, 1);
  const auto &out = builder.AddNode("out", NETOUTPUT, 1, 1);
  builder.AddDataEdge(input1, 0, merge, 0);
  builder.AddDataEdge(input2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, cast, 0);
  builder.AddDataEdge(cast, 0, out, 0);
  auto graph = builder.GetGraph();

  // 调用GetSupportInplaceOutput接口
  std::map<size_t, std::vector<size_t>> inplace_index_list;
  auto ret = GraphUtils::GetSupportInplaceOutput(merge, inplace_index_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(inplace_index_list.size(), 1U);
  EXPECT_EQ(inplace_index_list[0].size(), 2U);
  EXPECT_EQ(inplace_index_list[0][0], 0U);
  EXPECT_EQ(inplace_index_list[0][1], 1U);
}

/*
    Data    Data
    |        |
  - Relu    Relu                                    Data   Data
 |    |       |                                      |   /  |    
 |   Cast0   Cast             Add0 -->              Add    Relu   
 |    \     /                                        \     /       
 |----> Add  ---- Relu                                 Add  
         / \         |                                         
  Cast <-    Add1 ----
                                                   
*/
TEST_F(UtestGraphUtils, TestExpandNodeWithGraphControlEdge) {
  auto builder = ut::GraphBuilder("test_expand_node_with_graph");
  const auto &data0 = builder.AddNode("data0", DATA, 0, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &relu0 = builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  const auto &cast0 = builder.AddNode("cast0", "Cast", 1, 1);
  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &add0 = builder.AddNode("add0", "Add", 2, 1);
  const auto &relu2 = builder.AddNode("relu2", "Relu", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);

  builder.AddDataEdge(data0, 0, relu0, 0);
  builder.AddDataEdge(data1, 0, relu1, 0);
  builder.AddDataEdge(relu0, 0, cast0, 0);
  builder.AddDataEdge(relu1, 0, cast1, 0);
  builder.AddDataEdge(cast0, 0, add0, 0);
  builder.AddDataEdge(cast1, 0, add0, 1);
  builder.AddDataEdge(add0, 0, relu2, 0);
  builder.AddDataEdge(add0, 0, add1, 0);
  builder.AddDataEdge(relu2, 0, add1, 1);
  builder.AddControlEdge(relu0, add0);
  builder.AddControlEdge(add0, cast2);
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{cast2, 0}, {add1, 0}};
  graph->SetOutputSize(2U);
  graph->SetGraphOutNodesInfo(output_nodes);

  std::vector<std::string> origin_node_sort;
  for (const auto &node : graph->GetDirectNode()) {
    origin_node_sort.emplace_back(node->GetName());
  }

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 0, 1);
  const auto &sub_data1 = sub_builder.AddNode("sub_data1", DATA, 0, 1);
  const auto &sub_add0 = sub_builder.AddNode("sub_add0", "Add", 2, 1);
  const auto &sub_relu0 = sub_builder.AddNode("sub_relu0", "Relu", 1, 1);
  const auto &sub_add1 = sub_builder.AddNode("sub_add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(sub_data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  sub_builder.AddDataEdge(sub_data0, 0, sub_add0, 0);
  sub_builder.AddDataEdge(sub_data1, 0, sub_add0, 1);
  sub_builder.AddDataEdge(sub_data1, 0, sub_relu0, 0);
  sub_builder.AddDataEdge(sub_add0, 0, sub_add1, 0);
  sub_builder.AddDataEdge(sub_relu0, 0, sub_add1, 1);
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_add1, 0}};
  sub_graph->SetOutputSize(1U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(add0, sub_graph), SUCCESS);
  const auto add0_node = graph->FindNode("add0");
  EXPECT_EQ(add0_node, nullptr);
  const auto subgraph_data0_node = graph->FindNode("sub_data0");
  EXPECT_EQ(subgraph_data0_node, nullptr);
  const auto subgraph_data1_node = graph->FindNode("sub_data1");
  EXPECT_EQ(subgraph_data1_node, nullptr);

  const auto sub_add0_node = graph->FindNode("sub_add0");
  EXPECT_EQ(sub_add0_node, sub_add0);
  const auto add_in_data_anchors = sub_add0_node->GetAllInDataAnchors();
  const auto add_in_data_anchor_0 = add_in_data_anchors.at(0);
  const auto peer_out_add_in_data_anchor_0 = add_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor_0->GetOwnerNode()->GetName(), "cast0");
  const auto add_in_data_anchor_1 = add_in_data_anchors.at(1);
  const auto peer_out_add_in_data_anchor_1 = add_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor_1->GetOwnerNode()->GetName(), "cast1");

  const auto add_in_control_anchor = sub_add0_node->GetInControlAnchor();
  ASSERT_NE(add_in_control_anchor, nullptr);
  const auto peer_out_add_in_control_anchors = add_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_add_in_control_anchors.size(), 2U);
  EXPECT_EQ(peer_out_add_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");
  EXPECT_EQ(peer_out_add_in_control_anchors.at(1)->GetOwnerNode()->GetName(), "relu0");
  const auto add_out_data_anchor = sub_add0_node->GetOutDataAnchor(0);
  ASSERT_NE(add_out_data_anchor, nullptr);
  const auto peer_in_add_out_data_anchors = add_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_add_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add1");
  const auto add_out_control_anchor = sub_add0_node->GetOutControlAnchor();
  ASSERT_NE(add_out_control_anchor, nullptr);
  EXPECT_EQ(add_out_control_anchor->GetPeerInControlAnchors().size(), 0U);

  const auto sub_relu0_node = graph->FindNode("sub_relu0");
  EXPECT_EQ(sub_relu0_node, sub_relu0);
  const auto relu0_in_data_anchor = sub_relu0_node->GetInDataAnchor(0);
  const auto peer_out_relu0_in_data_anchor = relu0_in_data_anchor->GetPeerOutAnchor();
  ASSERT_NE(peer_out_relu0_in_data_anchor, nullptr);
  EXPECT_EQ(peer_out_relu0_in_data_anchor->GetOwnerNode()->GetName(), "cast1");

  const auto relu0_in_control_anchor = sub_relu0_node->GetInControlAnchor();
  ASSERT_NE(relu0_in_control_anchor, nullptr);
  const auto peer_out_relu0_in_control_anchors = relu0_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_relu0_in_control_anchors.size(), 1U);
  EXPECT_EQ(peer_out_relu0_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");
  const auto relu0_out_data_anchor = sub_relu0_node->GetOutDataAnchor(0);
  ASSERT_NE(relu0_out_data_anchor, nullptr);
  const auto peer_in_relu0_out_data_anchors = relu0_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_relu0_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_relu0_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add1");
  const auto relu0_out_control_anchor = sub_relu0_node->GetOutControlAnchor();
  ASSERT_NE(relu0_out_control_anchor, nullptr);
  EXPECT_EQ(relu0_out_control_anchor->GetPeerInControlAnchors().size(), 0U);

  const auto sub_add1_node = graph->FindNode("sub_add1");
  EXPECT_EQ(sub_add1_node, sub_add1);
  const auto add1_in_data_anchors = sub_add1_node->GetAllInDataAnchors();
  const auto add1_in_data_anchor_0 = add1_in_data_anchors.at(0);
  const auto peer_out_add1_in_data_anchor_0 = add1_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_0->GetOwnerNode()->GetName(), "sub_add0");
  const auto add1_in_data_anchor_1 = add1_in_data_anchors.at(1);
  const auto peer_out_add1_in_data_anchor_1 = add1_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_1->GetOwnerNode()->GetName(), "sub_relu0");

  const auto add1_in_control_anchor = sub_add1_node->GetInControlAnchor();
  ASSERT_NE(add1_in_control_anchor, nullptr);
  const auto peer_out_add1_in_control_anchors = add1_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_add1_in_control_anchors.size(), 0U);

  const auto add1_out_data_anchor = sub_add1_node->GetOutDataAnchor(0);
  ASSERT_NE(add1_out_data_anchor, nullptr);
  const auto peer_in_add1_out_data_anchors = add1_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add1_out_data_anchors.size(), 2);
  EXPECT_EQ(peer_in_add1_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "relu2");
  EXPECT_EQ(peer_in_add1_out_data_anchors.at(1)->GetOwnerNode()->GetName(), "add1");
  const auto add1_out_control_anchor = sub_add1_node->GetOutControlAnchor();
  ASSERT_NE(add1_out_control_anchor, nullptr);
  EXPECT_EQ(add1_out_control_anchor->GetPeerInControlAnchors().size(), 1U);
  EXPECT_EQ(add1_out_control_anchor->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");

  // 验证topo序
  std::vector<std::string> expect_sort;
  for (const auto &origin_node_name : origin_node_sort) {
    if (origin_node_name == "add0") {
      for (const auto &subgraph_node : sub_graph->GetDirectNode()) {
        if ((subgraph_node->GetType() != "Data") && (subgraph_node->GetType() != NETOUTPUT)) {
          expect_sort.emplace_back(subgraph_node->GetName());
        }
      }
      continue;
    }
    expect_sort.emplace_back(origin_node_name);
  }
  size_t index = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), expect_sort[index]);
    index++;
  }
}

/*
    Data    Data
    |        |
  - Relu    Relu                                    Data   Data         then:  Data           else: Data
 |    |       |                                       |     |                    |                   |
 |   Cast0   Cast             Add1 -->                |     |                   Relu                Cast
 |    \     /                                          \    /             
 |----> Add  ---- Relu                                   If          ->  
         / \         |                                         
  Cast <-    Add1 ----                                 

*/
TEST_F(UtestGraphUtils, TestExpandNodeWithGraphWithSubGraph) {
  auto builder = ut::GraphBuilder("test_expand_node_with_graph");
  const auto &data0 = builder.AddNode("data0", DATA, 0, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &relu0 = builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  const auto &cast0 = builder.AddNode("cast0", "Cast", 1, 1);
  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &add0 = builder.AddNode("add0", "Add", 2, 1);
  const auto &relu2 = builder.AddNode("relu2", "Relu", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);

  builder.AddDataEdge(data0, 0, relu0, 0);
  builder.AddDataEdge(data1, 0, relu1, 0);
  builder.AddDataEdge(relu0, 0, cast0, 0);
  builder.AddDataEdge(relu1, 0, cast1, 0);
  builder.AddDataEdge(cast0, 0, add0, 0);
  builder.AddDataEdge(cast1, 0, add0, 1);
  builder.AddDataEdge(add0, 0, relu2, 0);
  builder.AddDataEdge(add0, 0, add1, 0);
  builder.AddDataEdge(relu2, 0, add1, 1);
  builder.AddControlEdge(relu0, add0);
  builder.AddControlEdge(add0, cast2);
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{cast2, 0}, {add1, 0}};
  graph->SetOutputSize(2U);
  graph->SetGraphOutNodesInfo(output_nodes);

  std::vector<std::string> origin_node_sort;
  for (const auto &node : graph->GetDirectNode()) {
    std::cout << "origin node: " << node->GetName() << std::endl;
    origin_node_sort.emplace_back(node->GetName());
  }

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 0, 1);
  const auto &sub_data1 = sub_builder.AddNode("sub_data1", DATA, 0, 1);
  const auto &sub_if = sub_builder.AddNode("sub_if", "If", 2, 1);
  // 设置属性
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(sub_data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  sub_builder.AddDataEdge(sub_data0, 0, sub_if, 0);
  sub_builder.AddDataEdge(sub_data1, 0, sub_if, 1);

  auto then_graph_builder = ut::GraphBuilder("then_graph");
  const auto &then_graph_data0 = then_graph_builder.AddNode("then_graph_data0", DATA, 0, 1);
  const auto &then_graph_relu0 = then_graph_builder.AddNode("then_graph_relu0", "Relu", 1, 1);
  then_graph_builder.AddDataEdge(then_graph_data0, 0, then_graph_relu0, 0);
  AttrUtils::SetInt(then_graph_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(then_graph_data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto then_graph = then_graph_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> then_graph_output_nodes{{then_graph_relu0, 0}};
  then_graph->SetOutputSize(1U);
  then_graph->SetGraphOutNodesInfo(then_graph_output_nodes);

  auto else_graph_builder = ut::GraphBuilder("else_graph");
  const auto &else_graph_data0 = else_graph_builder.AddNode("else_graph_data0", DATA, 0, 1);
  const auto &else_graph_cast0 = else_graph_builder.AddNode("else_graph_cast0", "Cast", 1, 1);
  else_graph_builder.AddDataEdge(else_graph_data0, 0, else_graph_cast0, 0);
  AttrUtils::SetInt(else_graph_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(else_graph_data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto else_graph = else_graph_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> else_graph_output_nodes{{else_graph_cast0, 0}};
  else_graph->SetOutputSize(1U);
  else_graph->SetGraphOutNodesInfo(else_graph_output_nodes);
 
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_if, 0}};
  sub_graph->SetOutputSize(1U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  then_graph->SetGraphUnknownFlag(false);
  sub_if->GetOpDesc()->AddSubgraphName(then_graph->GetName());
  sub_if->GetOpDesc()->SetSubgraphInstanceName(0, then_graph->GetName());
  sub_graph->AddSubGraph(then_graph);
  then_graph->SetParentNode(sub_if);
  then_graph->SetParentGraph(sub_graph);

  else_graph->SetGraphUnknownFlag(false);
  sub_if->GetOpDesc()->AddSubgraphName(else_graph->GetName());
  sub_if->GetOpDesc()->SetSubgraphInstanceName(1, else_graph->GetName());
  sub_graph->AddSubGraph(else_graph);
  else_graph->SetParentNode(sub_if);
  else_graph->SetParentGraph(sub_graph);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(add0, sub_graph), SUCCESS);
  const auto subgraphs = graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);
  const auto add0_node = graph->FindNode("add0");
  EXPECT_EQ(add0_node, nullptr);
  const auto subgraph_data0_node = graph->FindNode("sub_data0");
  EXPECT_EQ(subgraph_data0_node, nullptr);
  const auto subgraph_data1_node = graph->FindNode("sub_data1");
  EXPECT_EQ(subgraph_data1_node, nullptr);
  const auto sub_if_node = graph->FindNode("sub_if");
  EXPECT_EQ(sub_if_node, sub_if);
  const auto if_in_data_anchors = sub_if_node->GetAllInDataAnchors();
  const auto if_in_data_anchor_0 = if_in_data_anchors.at(0);
  const auto peer_out_if_in_data_anchor_0 = if_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_if_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_if_in_data_anchor_0->GetOwnerNode()->GetName(), "cast0");
  const auto if_in_data_anchor_1 = if_in_data_anchors.at(1);
  const auto peer_out_if_in_data_anchor_1 = if_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_if_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_if_in_data_anchor_1->GetOwnerNode()->GetName(), "cast1");

  const auto if_in_control_anchor = sub_if_node->GetInControlAnchor();
  ASSERT_NE(if_in_control_anchor, nullptr);
  const auto peer_out_if_in_control_anchors = if_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_if_in_control_anchors.size(), 2U);
  EXPECT_EQ(peer_out_if_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");
  EXPECT_EQ(peer_out_if_in_control_anchors.at(1)->GetOwnerNode()->GetName(), "relu0");
  const auto if_out_data_anchor = sub_if_node->GetOutDataAnchor(0);
  ASSERT_NE(if_out_data_anchor, nullptr);
  const auto peer_in_if_out_data_anchors = if_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_if_out_data_anchors.size(), 2);
  EXPECT_EQ(peer_in_if_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "relu2");
  EXPECT_EQ(peer_in_if_out_data_anchors.at(1)->GetOwnerNode()->GetName(), "add1");
  const auto if_out_control_anchor = sub_if_node->GetOutControlAnchor();
  ASSERT_NE(if_out_control_anchor, nullptr);
  EXPECT_EQ(if_out_control_anchor->GetPeerInControlAnchors().size(), 1U);
  EXPECT_EQ(if_out_control_anchor->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");
  auto subgraph_names_index = sub_if_node->GetOpDesc()->GetSubgraphNameIndexes();
  EXPECT_EQ(subgraph_names_index.size(), 2);
  const auto subgraph_then_graph = subgraphs[subgraph_names_index["then_graph"]];
  ASSERT_NE(subgraph_then_graph, nullptr);
  EXPECT_EQ(subgraph_then_graph->GetParentGraph(), graph);
  EXPECT_EQ(subgraph_then_graph->GetParentNode(), sub_if_node);
  const auto subgraph_else_graph = subgraphs[subgraph_names_index["else_graph"]];
  ASSERT_NE(subgraph_else_graph, nullptr);
  EXPECT_EQ(subgraph_else_graph->GetParentGraph(), graph);
  EXPECT_EQ(subgraph_else_graph->GetParentNode(), sub_if_node);
  // 验证topo序
  std::vector<std::string> expect_sort;
  for (const auto &origin_node_name : origin_node_sort) {
    if (origin_node_name == "add0") {
      for (const auto &subgraph_node : sub_graph->GetDirectNode()) {
        if ((subgraph_node->GetType() != "Data") && (subgraph_node->GetType() != NETOUTPUT)) {
          expect_sort.emplace_back(subgraph_node->GetName());
        }
      }
      continue;
    }
    expect_sort.emplace_back(origin_node_name);
  }
  size_t index = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), expect_sort[index]);
    index++;
  }
}

/*
  Data    Data            Data                   Data                                   Data
    \    /                 |                      |                 -->                 /  \
      If                  Cast                   Relu                                  Relu  Relu
                                                                                       \    /
                                                                                        Add
*/
TEST_F(UtestGraphUtils, TestExpandNodeWithGraphNodeInSubGraph) {
  auto builder = ut::GraphBuilder("root_graph");
  const auto &data0 = builder.AddNode("data0", DATA, 0, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &if_op = builder.AddNode("if_op", "If", 2, 1);
  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  builder.AddDataEdge(data0, 0, if_op, 0);
  builder.AddDataEdge(data1, 0, if_op, 1);

  auto then_graph_builder = ut::GraphBuilder("then_graph");
  const auto &then_graph_data0 = then_graph_builder.AddNode("then_graph_data0", DATA, 0, 1);
  const auto &then_graph_relu0 = then_graph_builder.AddNode("then_graph_relu0", "Relu", 1, 1);
  then_graph_builder.AddDataEdge(then_graph_data0, 0, then_graph_relu0, 0);
  AttrUtils::SetInt(then_graph_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(then_graph_data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto then_graph = then_graph_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> then_graph_output_nodes{{then_graph_relu0, 0}};
  then_graph->SetOutputSize(1U);
  then_graph->SetGraphOutNodesInfo(then_graph_output_nodes);

  auto else_graph_builder = ut::GraphBuilder("else_graph");
  const auto &else_graph_data0 = else_graph_builder.AddNode("else_graph_data0", DATA, 0, 1);
  const auto &else_graph_cast0 = else_graph_builder.AddNode("else_graph_cast0", "Cast", 1, 1);
  else_graph_builder.AddDataEdge(else_graph_data0, 0, else_graph_cast0, 0);
  AttrUtils::SetInt(else_graph_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(else_graph_data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto else_graph = else_graph_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> else_graph_output_nodes{{else_graph_cast0, 0}};
  else_graph->SetOutputSize(1U);
  else_graph->SetGraphOutNodesInfo(else_graph_output_nodes);
 
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{if_op, 0}};
  graph->SetOutputSize(1U);
  graph->SetGraphOutNodesInfo(output_nodes);

  then_graph->SetGraphUnknownFlag(false);
  if_op->GetOpDesc()->AddSubgraphName(then_graph->GetName());
  if_op->GetOpDesc()->SetSubgraphInstanceName(0, then_graph->GetName());
  graph->AddSubGraph(then_graph);
  then_graph->SetParentNode(if_op);
  then_graph->SetParentGraph(graph);

  else_graph->SetGraphUnknownFlag(false);
  if_op->GetOpDesc()->AddSubgraphName(else_graph->GetName());
  if_op->GetOpDesc()->SetSubgraphInstanceName(1, else_graph->GetName());
  graph->AddSubGraph(else_graph);
  else_graph->SetParentNode(if_op);
  else_graph->SetParentGraph(graph);

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 0, 1);
  const auto &sub_relu0 = sub_builder.AddNode("sub_relu0", "Relu", 1, 1);
  const auto &sub_relu1 = sub_builder.AddNode("sub_relu1", "Relu", 1, 1);
  const auto &sub_add = sub_builder.AddNode("sub_add", "Add", 2, 1);
  sub_builder.AddDataEdge(sub_data0, 0, sub_relu0, 0);
  sub_builder.AddDataEdge(sub_data0, 0, sub_relu1, 0);
  sub_builder.AddDataEdge(sub_relu0, 0, sub_add, 0);
  sub_builder.AddDataEdge(sub_relu1, 0, sub_add, 1);
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_add, 0}};
  sub_graph->SetOutputSize(1U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(then_graph_relu0, sub_graph), SUCCESS);
  const auto subgraphs = graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);
  EXPECT_EQ(then_graph->FindNode("then_graph_relu0"), nullptr);
  EXPECT_EQ(then_graph->FindNode("sub_data0"), nullptr);

  const auto sub_relu0_node = then_graph->FindNode("sub_relu0");
  EXPECT_EQ(sub_relu0_node, sub_relu0);
  const auto relu0_in_data_anchor = sub_relu0_node->GetInDataAnchor(0);
  const auto peer_out_relu0_in_data_anchor = relu0_in_data_anchor->GetPeerOutAnchor();
  ASSERT_NE(peer_out_relu0_in_data_anchor, nullptr);
  EXPECT_EQ(peer_out_relu0_in_data_anchor->GetOwnerNode()->GetName(), "then_graph_data0");
  const auto relu0_out_data_anchor = sub_relu0_node->GetOutDataAnchor(0);
  ASSERT_NE(relu0_out_data_anchor, nullptr);
  const auto peer_in_relu0_out_data_anchors = relu0_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_relu0_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_relu0_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add");

  const auto sub_relu1_node = then_graph->FindNode("sub_relu1");
  EXPECT_EQ(sub_relu1_node, sub_relu1);
  const auto relu1_in_data_anchor = sub_relu1_node->GetInDataAnchor(0);
  const auto peer_out_relu1_in_data_anchor = relu1_in_data_anchor->GetPeerOutAnchor();
  ASSERT_NE(peer_out_relu1_in_data_anchor, nullptr);
  EXPECT_EQ(peer_out_relu1_in_data_anchor->GetOwnerNode()->GetName(), "then_graph_data0");
  const auto relu1_out_data_anchor = sub_relu1_node->GetOutDataAnchor(0);
  ASSERT_NE(relu1_out_data_anchor, nullptr);
  const auto peer_in_relu1_out_data_anchors = relu1_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_relu1_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_relu1_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add");

  const auto sub_add_node = then_graph->FindNode("sub_add");
  EXPECT_EQ(sub_add_node, sub_add);
  const auto add_in_data_anchor0 = sub_add_node->GetInDataAnchor(0);
  const auto peer_out_add_in_data_anchor0 = add_in_data_anchor0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor0, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor0->GetOwnerNode()->GetName(), "sub_relu0");
  const auto add_in_data_anchor1 = sub_add_node->GetInDataAnchor(1);
  const auto peer_out_add_in_data_anchor1 = add_in_data_anchor1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor1, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor1->GetOwnerNode()->GetName(), "sub_relu1");
  const auto add_out_data_anchor = sub_add_node->GetOutDataAnchor(0);
  ASSERT_NE(add_out_data_anchor, nullptr);
  const auto peer_in_add_out_data_anchors = add_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add_out_data_anchors.size(), 1); // 连给子图NetOutput
  const auto output_node_info = then_graph->GetGraphOutNodesInfo();
  EXPECT_EQ(output_node_info.size(), 1);
  EXPECT_EQ(output_node_info[0].first, sub_add_node);
  EXPECT_EQ(output_node_info[0].second, 0);
}

/*
    Data    Data
    |        |
   Relu    Relu                                    Data   Data
     |       |                                      |     |    
    Cast0   Cast   Data            Clip -->         |     |
      \     /      |                                  \    /       
        Clip ------                                    Clip
         |   \  
         |    ---- Relu                                   
        / \          |                                         
  Cast <-    Add1 ----
                                                   
*/
TEST_F(UtestGraphUtils, TestExpandNodeWithGraphInputNotMatch) {
  auto builder = ut::GraphBuilder("test_expand_node_with_graph");
  const auto &data0 = builder.AddNode("data0", DATA, 1, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &relu0 = builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  const auto &cast0 = builder.AddNode("cast0", "Cast", 1, 1);
  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &clip0 = builder.AddNode("clip0", "ClipByValue", 3, 1);
  const auto &relu2 = builder.AddNode("relu2", "Relu", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);

  builder.AddDataEdge(data0, 0, relu0, 0);
  builder.AddDataEdge(data1, 0, relu1, 0);
  builder.AddDataEdge(relu0, 0, cast0, 0);
  builder.AddDataEdge(relu1, 0, cast1, 0);
  builder.AddDataEdge(cast0, 0, clip0, 0);
  builder.AddDataEdge(cast1, 0, clip0, 1);
  builder.AddDataEdge(data2, 0, clip0, 2);
  builder.AddDataEdge(clip0, 0, relu2, 0);
  builder.AddDataEdge(clip0, 0, add1, 0);
  builder.AddDataEdge(relu2, 0, add1, 1);
  builder.AddControlEdge(relu1, clip0);
  builder.AddControlEdge(clip0, cast2);
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{cast2, 0}, {add1, 0}};
  graph->SetOutputSize(2U);
  graph->SetGraphOutNodesInfo(output_nodes);

  std::vector<std::string> origin_node_sort;
  for (const auto &node : graph->GetDirectNode()) {
    std::cout << "origin node: " << node->GetName() << std::endl;
    origin_node_sort.emplace_back(node->GetName());
  }

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 1, 1);
  const auto &sub_data2 = sub_builder.AddNode("sub_data2", DATA, 1, 1);
  const auto &sub_clip0 = sub_builder.AddNode("sub_clip0", "ClipByValue", 3, 1);
  // 设置属性
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(sub_data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
  sub_builder.AddDataEdge(sub_data0, 0, sub_clip0, 0);
  sub_builder.AddDataEdge(sub_data2, 0, sub_clip0, 2);
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_clip0, 0}};
  sub_graph->SetOutputSize(1U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(clip0, sub_graph), SUCCESS);
  const auto clip0_node = graph->FindNode("clip0");
  EXPECT_EQ(clip0_node, nullptr);
  const auto subgraph_data0_node = graph->FindNode("sub_data0");
  EXPECT_EQ(subgraph_data0_node, nullptr);
  const auto subgraph_data2_node = graph->FindNode("sub_data2");
  EXPECT_EQ(subgraph_data2_node, nullptr);

  const auto sub_clip0_node = graph->FindNode("sub_clip0");
  EXPECT_EQ(sub_clip0_node, sub_clip0);
  const auto clip0_in_data_anchors = sub_clip0_node->GetAllInDataAnchors();
  const auto clip0_in_data_anchor_0 = clip0_in_data_anchors.at(0);
  const auto peer_out_clip0_in_data_anchor_0 = clip0_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_clip0_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_clip0_in_data_anchor_0->GetOwnerNode()->GetName(), "cast0");

  const auto clip0_in_data_anchor_1 = clip0_in_data_anchors.at(1);
  const auto peer_out_clip0_in_data_anchor_1 = clip0_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_EQ(peer_out_clip0_in_data_anchor_1, nullptr);

  const auto clip0_in_data_anchor_2 = clip0_in_data_anchors.at(2);
  const auto peer_out_clip0_in_data_anchor_2 = clip0_in_data_anchor_2->GetPeerOutAnchor();
  ASSERT_NE(peer_out_clip0_in_data_anchor_2, nullptr);
  EXPECT_EQ(peer_out_clip0_in_data_anchor_2->GetOwnerNode()->GetName(), "data2");

  const auto clip0_in_control_anchor = sub_clip0_node->GetInControlAnchor();
  ASSERT_NE(clip0_in_control_anchor, nullptr);
  const auto peer_out_clip0_in_control_anchors = clip0_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_clip0_in_control_anchors.size(), 2U);
  EXPECT_EQ(peer_out_clip0_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu1");
  EXPECT_EQ(peer_out_clip0_in_control_anchors.at(1)->GetOwnerNode()->GetName(), "relu1");
  const auto clip0_out_data_anchor = sub_clip0_node->GetOutDataAnchor(0);
  ASSERT_NE(clip0_out_data_anchor, nullptr);
  const auto peer_in_clip0_out_data_anchors = clip0_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_clip0_out_data_anchors.size(), 2);
  EXPECT_EQ(peer_in_clip0_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "relu2");
  EXPECT_EQ(peer_in_clip0_out_data_anchors.at(1)->GetOwnerNode()->GetName(), "add1");
  const auto clip0_out_control_anchor = sub_clip0_node->GetOutControlAnchor();
  ASSERT_NE(clip0_out_control_anchor, nullptr);
  EXPECT_EQ(clip0_out_control_anchor->GetPeerInControlAnchors().size(), 1U);
  EXPECT_EQ(clip0_out_control_anchor->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");

  // 验证topo序
  std::vector<std::string> expect_sort;
  for (const auto &origin_node_name : origin_node_sort) {
    if (origin_node_name == "clip0") {
      for (const auto &subgraph_node : sub_graph->GetDirectNode()) {
        if ((subgraph_node->GetType() != "Data") && (subgraph_node->GetType() != NETOUTPUT)) {
          expect_sort.emplace_back(subgraph_node->GetName());
        }
      }
      continue;
    }
    expect_sort.emplace_back(origin_node_name);
  }
  size_t index = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), expect_sort[index]);
    index++;
  }
}

/*
    Data    Data
    |        |
  - Relu    Relu                                    Data   Data
 |    |       |                                      |   /  |    
 |   Cast0   Cast             Add0 -->              Add    Relu   
 |    \     /                                        \     /  |     
 |----> Add  ---- Relu                                 Add    |
         / \         |                                     NetOutput  
  Cast <-    Add1 ----                                        
                                                   
*/
TEST_F(UtestGraphUtils, TestExpandNodeWithNetOutput) {
  auto builder = ut::GraphBuilder("test_expand_node_with_graph");
  const auto &data0 = builder.AddNode("data0", DATA, 0, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &relu0 = builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  const auto &cast0 = builder.AddNode("cast0", "Cast", 1, 1);
  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &add0 = builder.AddNode("add0", "Add", 2, 1);
  const auto &relu2 = builder.AddNode("relu2", "Relu", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);

  builder.AddDataEdge(data0, 0, relu0, 0);
  builder.AddDataEdge(data1, 0, relu1, 0);
  builder.AddDataEdge(relu0, 0, cast0, 0);
  builder.AddDataEdge(relu1, 0, cast1, 0);
  builder.AddDataEdge(cast0, 0, add0, 0);
  builder.AddDataEdge(cast1, 0, add0, 1);
  builder.AddDataEdge(add0, 0, relu2, 0);
  builder.AddDataEdge(add0, 0, add1, 0);
  builder.AddDataEdge(relu2, 0, add1, 1);
  builder.AddControlEdge(relu0, add0);
  builder.AddControlEdge(add0, cast2);
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{cast2, 0}, {add1, 0}};
  graph->SetOutputSize(2U);
  graph->SetGraphOutNodesInfo(output_nodes);

  std::vector<std::string> origin_node_sort;
  for (const auto &node : graph->GetDirectNode()) {
    std::cout << "origin node: " << node->GetName() << std::endl;
    origin_node_sort.emplace_back(node->GetName());
  }

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 0, 1);
  const auto &sub_data1 = sub_builder.AddNode("sub_data1", DATA, 0, 1);
  const auto &sub_add0 = sub_builder.AddNode("sub_add0", "Add", 2, 1);
  const auto &sub_relu0 = sub_builder.AddNode("sub_relu0", "Relu", 1, 1);
  const auto &sub_add1 = sub_builder.AddNode("sub_add1", "Add", 2, 1);
  const auto &sub_output = sub_builder.AddNode("sub_output", "NetOutput", 1, 1);
  // 设置属性
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(sub_data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  sub_builder.AddDataEdge(sub_data0, 0, sub_add0, 0);
  sub_builder.AddDataEdge(sub_data1, 0, sub_add0, 1);
  sub_builder.AddDataEdge(sub_data1, 0, sub_relu0, 0);
  sub_builder.AddDataEdge(sub_add0, 0, sub_add1, 0);
  sub_builder.AddDataEdge(sub_relu0, 0, sub_add1, 1);
  sub_builder.AddDataEdge(sub_relu0, 0, sub_output, 0);
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_relu0, 0}};
  sub_graph->SetOutputSize(1U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(add0, sub_graph), SUCCESS);
  const auto add0_node = graph->FindNode("add0");
  EXPECT_EQ(add0_node, nullptr);
  const auto subgraph_data0_node = graph->FindNode("sub_data0");
  EXPECT_EQ(subgraph_data0_node, nullptr);
  const auto subgraph_data1_node = graph->FindNode("sub_data1");
  EXPECT_EQ(subgraph_data1_node, nullptr);
  const auto output_node = graph->FindNode("sub_output");
  EXPECT_EQ(output_node, nullptr);
  const auto sub_add0_node = graph->FindNode("sub_add0");
  EXPECT_EQ(sub_add0_node, sub_add0);
  const auto add_in_data_anchors = sub_add0_node->GetAllInDataAnchors();
  const auto add_in_data_anchor_0 = add_in_data_anchors.at(0);
  const auto peer_out_add_in_data_anchor_0 = add_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor_0->GetOwnerNode()->GetName(), "cast0");
  const auto add_in_data_anchor_1 = add_in_data_anchors.at(1);
  const auto peer_out_add_in_data_anchor_1 = add_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor_1->GetOwnerNode()->GetName(), "cast1");

  const auto add_in_control_anchor = sub_add0_node->GetInControlAnchor();
  ASSERT_NE(add_in_control_anchor, nullptr);
  const auto peer_out_add_in_control_anchors = add_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_add_in_control_anchors.size(), 2U);
  EXPECT_EQ(peer_out_add_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");
  EXPECT_EQ(peer_out_add_in_control_anchors.at(1)->GetOwnerNode()->GetName(), "relu0");
  const auto add_out_data_anchor = sub_add0_node->GetOutDataAnchor(0);
  ASSERT_NE(add_out_data_anchor, nullptr);
  const auto peer_in_add_out_data_anchors = add_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_add_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add1");
  const auto add_out_control_anchor = sub_add0_node->GetOutControlAnchor();
  ASSERT_NE(add_out_control_anchor, nullptr);
  EXPECT_EQ(add_out_control_anchor->GetPeerInControlAnchors().size(), 0U);

  const auto sub_relu0_node = graph->FindNode("sub_relu0");
  EXPECT_EQ(sub_relu0_node, sub_relu0);
  const auto relu0_in_data_anchor = sub_relu0_node->GetInDataAnchor(0);
  const auto peer_out_relu0_in_data_anchor = relu0_in_data_anchor->GetPeerOutAnchor();
  ASSERT_NE(peer_out_relu0_in_data_anchor, nullptr);
  EXPECT_EQ(peer_out_relu0_in_data_anchor->GetOwnerNode()->GetName(), "cast1");

  const auto relu0_in_control_anchor = sub_relu0_node->GetInControlAnchor();
  ASSERT_NE(relu0_in_control_anchor, nullptr);
  const auto peer_out_relu0_in_control_anchors = relu0_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_relu0_in_control_anchors.size(), 1U);
  EXPECT_EQ(peer_out_relu0_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");
  const auto relu0_out_data_anchor = sub_relu0_node->GetOutDataAnchor(0);
  ASSERT_NE(relu0_out_data_anchor, nullptr);
  const auto peer_in_relu0_out_data_anchors = relu0_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_relu0_out_data_anchors.size(), 3);
  EXPECT_EQ(peer_in_relu0_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add1");
  EXPECT_EQ(peer_in_relu0_out_data_anchors.at(1)->GetOwnerNode()->GetName(), "relu2");
  EXPECT_EQ(peer_in_relu0_out_data_anchors.at(2)->GetOwnerNode()->GetName(), "add1");
  const auto relu0_out_control_anchor = sub_relu0_node->GetOutControlAnchor();
  ASSERT_NE(relu0_out_control_anchor, nullptr);
  EXPECT_EQ(relu0_out_control_anchor->GetPeerInControlAnchors().size(), 1U);
  EXPECT_EQ(relu0_out_control_anchor->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");

  const auto sub_add1_node = graph->FindNode("sub_add1");
  EXPECT_EQ(sub_add1_node, sub_add1);
  const auto add1_in_data_anchors = sub_add1_node->GetAllInDataAnchors();
  const auto add1_in_data_anchor_0 = add1_in_data_anchors.at(0);
  const auto peer_out_add1_in_data_anchor_0 = add1_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_0->GetOwnerNode()->GetName(), "sub_add0");
  const auto add1_in_data_anchor_1 = add1_in_data_anchors.at(1);
  const auto peer_out_add1_in_data_anchor_1 = add1_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_1->GetOwnerNode()->GetName(), "sub_relu0");

  const auto add1_in_control_anchor = sub_add1_node->GetInControlAnchor();
  ASSERT_NE(add1_in_control_anchor, nullptr);
  const auto peer_out_add1_in_control_anchors = add1_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_add1_in_control_anchors.size(), 0U);

  const auto add1_out_data_anchor = sub_add1_node->GetOutDataAnchor(0);
  ASSERT_NE(add1_out_data_anchor, nullptr);
  const auto peer_in_add1_out_data_anchors = add1_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add1_out_data_anchors.size(), 0);
  const auto add1_out_control_anchor = sub_add1_node->GetOutControlAnchor();
  ASSERT_NE(add1_out_control_anchor, nullptr);
  EXPECT_EQ(add1_out_control_anchor->GetPeerInControlAnchors().size(), 0U);

  // 验证topo序
  std::vector<std::string> expect_sort;
  for (const auto &origin_node_name : origin_node_sort) {
    if (origin_node_name == "add0") {
      for (const auto &subgraph_node : sub_graph->GetDirectNode()) {
        if ((subgraph_node->GetType() != "Data") && (subgraph_node->GetType() != "NetOutput")) {
          expect_sort.emplace_back(subgraph_node->GetName());
        }
      }
      continue;
    }
    expect_sort.emplace_back(origin_node_name);
  }
  size_t index = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), expect_sort[index]);
    index++;
  }
}

/*
    Data    Data
    |        |
  - Relu    Relu                                    Data   Data
 |    |       |                                    / |   /  |    
 |   Cast0   Cast             Add0 -->            |  Add    Relu   
 |    \     /                                     \   \     /       
 |----> Add  ---- Relu                             --   Add  
         / \         |                                         
  Cast <-    Add1 ----
                                                   
*/
TEST_F(UtestGraphUtils, TestExpandNodeWithDataControl) {
  auto builder = ut::GraphBuilder("test_expand_node_with_graph");
  const auto &data0 = builder.AddNode("data0", DATA, 0, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &relu0 = builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  const auto &cast0 = builder.AddNode("cast0", "Cast", 1, 1);
  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &add0 = builder.AddNode("add0", "Add", 2, 1);
  const auto &relu2 = builder.AddNode("relu2", "Relu", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);

  builder.AddDataEdge(data0, 0, relu0, 0);
  builder.AddDataEdge(data1, 0, relu1, 0);
  builder.AddDataEdge(relu0, 0, cast0, 0);
  builder.AddDataEdge(relu1, 0, cast1, 0);
  builder.AddDataEdge(cast0, 0, add0, 0);
  builder.AddDataEdge(cast1, 0, add0, 1);
  builder.AddDataEdge(add0, 0, relu2, 0);
  builder.AddDataEdge(add0, 0, add1, 0);
  builder.AddDataEdge(relu2, 0, add1, 1);
  builder.AddControlEdge(relu0, add0);
  builder.AddControlEdge(add0, cast2);
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{cast2, 0}, {add1, 0}};
  graph->SetOutputSize(2U);
  graph->SetGraphOutNodesInfo(output_nodes);

  std::vector<std::string> origin_node_sort;
  for (const auto &node : graph->GetDirectNode()) {
    std::cout << "origin node: " << node->GetName() << std::endl;
    origin_node_sort.emplace_back(node->GetName());
  }

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 0, 1);
  const auto &sub_data1 = sub_builder.AddNode("sub_data1", DATA, 0, 1);
  const auto &sub_add0 = sub_builder.AddNode("sub_add0", "Add", 2, 1);
  const auto &sub_relu0 = sub_builder.AddNode("sub_relu0", "Relu", 1, 1);
  const auto &sub_add1 = sub_builder.AddNode("sub_add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(sub_data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  sub_builder.AddDataEdge(sub_data0, 0, sub_add0, 0);
  sub_builder.AddDataEdge(sub_data1, 0, sub_add0, 1);
  sub_builder.AddDataEdge(sub_data1, 0, sub_relu0, 0);
  sub_builder.AddDataEdge(sub_add0, 0, sub_add1, 0);
  sub_builder.AddDataEdge(sub_relu0, 0, sub_add1, 1);
  sub_builder.AddControlEdge(sub_data0, sub_add1);
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_add1, 0}};
  sub_graph->SetOutputSize(1U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(add0, sub_graph), SUCCESS);
  const auto add0_node = graph->FindNode("add0");
  EXPECT_EQ(add0_node, nullptr);
  const auto subgraph_data0_node = graph->FindNode("sub_data0");
  EXPECT_EQ(subgraph_data0_node, nullptr);
  const auto subgraph_data1_node = graph->FindNode("sub_data1");
  EXPECT_EQ(subgraph_data1_node, nullptr);

  const auto sub_add0_node = graph->FindNode("sub_add0");
  EXPECT_EQ(sub_add0_node, sub_add0);
  const auto add_in_data_anchors = sub_add0_node->GetAllInDataAnchors();
  const auto add_in_data_anchor_0 = add_in_data_anchors.at(0);
  const auto peer_out_add_in_data_anchor_0 = add_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor_0->GetOwnerNode()->GetName(), "cast0");
  const auto add_in_data_anchor_1 = add_in_data_anchors.at(1);
  const auto peer_out_add_in_data_anchor_1 = add_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add_in_data_anchor_1->GetOwnerNode()->GetName(), "cast1");

  const auto add_in_control_anchor = sub_add0_node->GetInControlAnchor();
  ASSERT_NE(add_in_control_anchor, nullptr);
  const auto peer_out_add_in_control_anchors = add_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_add_in_control_anchors.size(), 2U);
  EXPECT_EQ(peer_out_add_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");
  EXPECT_EQ(peer_out_add_in_control_anchors.at(1)->GetOwnerNode()->GetName(), "relu0");
  const auto add_out_data_anchor = sub_add0_node->GetOutDataAnchor(0);
  ASSERT_NE(add_out_data_anchor, nullptr);
  const auto peer_in_add_out_data_anchors = add_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_add_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add1");
  const auto add_out_control_anchor = sub_add0_node->GetOutControlAnchor();
  ASSERT_NE(add_out_control_anchor, nullptr);
  EXPECT_EQ(add_out_control_anchor->GetPeerInControlAnchors().size(), 0U);

  const auto sub_relu0_node = graph->FindNode("sub_relu0");
  EXPECT_EQ(sub_relu0_node, sub_relu0);
  const auto relu0_in_data_anchor = sub_relu0_node->GetInDataAnchor(0);
  const auto peer_out_relu0_in_data_anchor = relu0_in_data_anchor->GetPeerOutAnchor();
  ASSERT_NE(peer_out_relu0_in_data_anchor, nullptr);
  EXPECT_EQ(peer_out_relu0_in_data_anchor->GetOwnerNode()->GetName(), "cast1");

  const auto relu0_in_control_anchor = sub_relu0_node->GetInControlAnchor();
  ASSERT_NE(relu0_in_control_anchor, nullptr);
  const auto peer_out_relu0_in_control_anchors = relu0_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_relu0_in_control_anchors.size(), 1U);
  EXPECT_EQ(peer_out_relu0_in_control_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");
  const auto relu0_out_data_anchor = sub_relu0_node->GetOutDataAnchor(0);
  ASSERT_NE(relu0_out_data_anchor, nullptr);
  const auto peer_in_relu0_out_data_anchors = relu0_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_relu0_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_relu0_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "sub_add1");
  const auto relu0_out_control_anchor = sub_relu0_node->GetOutControlAnchor();
  ASSERT_NE(relu0_out_control_anchor, nullptr);
  EXPECT_EQ(relu0_out_control_anchor->GetPeerInControlAnchors().size(), 0U);

  const auto sub_add1_node = graph->FindNode("sub_add1");
  EXPECT_EQ(sub_add1_node, sub_add1);
  const auto add1_in_data_anchors = sub_add1_node->GetAllInDataAnchors();
  const auto add1_in_data_anchor_0 = add1_in_data_anchors.at(0);
  const auto peer_out_add1_in_data_anchor_0 = add1_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_0->GetOwnerNode()->GetName(), "sub_add0");
  const auto add1_in_data_anchor_1 = add1_in_data_anchors.at(1);
  const auto peer_out_add1_in_data_anchor_1 = add1_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_1->GetOwnerNode()->GetName(), "sub_relu0");

  const auto add1_in_control_anchor = sub_add1_node->GetInControlAnchor();
  ASSERT_NE(add1_in_control_anchor, nullptr);
  const auto peer_out_add1_in_control_anchors = add1_in_control_anchor->GetPeerOutControlAnchors();
  EXPECT_EQ(peer_out_add1_in_control_anchors.size(), 0U);

  const auto add1_out_data_anchor = sub_add1_node->GetOutDataAnchor(0);
  ASSERT_NE(add1_out_data_anchor, nullptr);
  const auto peer_in_add1_out_data_anchors = add1_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add1_out_data_anchors.size(), 2);
  EXPECT_EQ(peer_in_add1_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "relu2");
  EXPECT_EQ(peer_in_add1_out_data_anchors.at(1)->GetOwnerNode()->GetName(), "add1");
  const auto add1_out_control_anchor = sub_add1_node->GetOutControlAnchor();
  ASSERT_NE(add1_out_control_anchor, nullptr);
  EXPECT_EQ(add1_out_control_anchor->GetPeerInControlAnchors().size(), 1U);
  EXPECT_EQ(add1_out_control_anchor->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetName(), "cast2");

  // 验证topo序
  std::vector<std::string> expect_sort;
  for (const auto &origin_node_name : origin_node_sort) {
    if (origin_node_name == "add0") {
      for (const auto &subgraph_node : sub_graph->GetDirectNode()) {
        if ((subgraph_node->GetType() != "Data") && (subgraph_node->GetType() != "NetOutput")) {
          expect_sort.emplace_back(subgraph_node->GetName());
        }
      }
      continue;
    }
    expect_sort.emplace_back(origin_node_name);
  }
  size_t index = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), expect_sort[index]);
    index++;
  }
}

/*
    Data    Data   Data
    |        |      |
    \        |     /                                           Data    Data   Data
          identity                                --->          \      /  \     /
          /      \                                                Add0       Add1      
        Relu0   Relu1                                         
*/
TEST_F(UtestGraphUtils, TestExpandNodeWithOutputNotMatch) {
  auto builder = ut::GraphBuilder("test_expand_node_output_not_match");
  const auto &data0 = builder.AddNode("data0", DATA, 0, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &identityN = builder.AddNode("identityn0", "IdentityN", 3, 3);
  const auto &relu0 = builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = builder.AddNode("relu1", "Relu", 1, 1);

  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);

  builder.AddDataEdge(data0, 0, identityN, 0);
  builder.AddDataEdge(data1, 0, identityN, 1);
  builder.AddDataEdge(data2, 0, identityN, 2);
  builder.AddDataEdge(identityN, 0, relu0, 0);
  builder.AddDataEdge(identityN, 1, relu1, 0);
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{relu0, 0}, {relu1, 0}};
  graph->SetOutputSize(2U);
  graph->SetGraphOutNodesInfo(output_nodes);

  std::vector<std::string> origin_node_sort;
  for (const auto &node : graph->GetDirectNode()) {
    origin_node_sort.emplace_back(node->GetName());
  }

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 0, 1);
  const auto &sub_data1 = sub_builder.AddNode("sub_data1", DATA, 0, 1);
  const auto &sub_data2 = sub_builder.AddNode("sub_data2", DATA, 0, 1);
  const auto &sub_add0 = sub_builder.AddNode("sub_add0", "Add", 2, 1);
  const auto &sub_add1 = sub_builder.AddNode("sub_add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(sub_data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  AttrUtils::SetInt(sub_data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
  sub_builder.AddDataEdge(sub_data0, 0, sub_add0, 0);
  sub_builder.AddDataEdge(sub_data1, 0, sub_add0, 1);
  sub_builder.AddDataEdge(sub_data1, 0, sub_add1, 0);
  sub_builder.AddDataEdge(sub_data2, 0, sub_add1, 1);
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_add0, 0}, {sub_add1, 0}};
  sub_graph->SetOutputSize(2U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(identityN, sub_graph), SUCCESS);
  const auto identityn_node = graph->FindNode("identityn0");
  EXPECT_EQ(identityn_node, nullptr);
  const auto subgraph_data0_node = graph->FindNode("sub_data0");
  EXPECT_EQ(subgraph_data0_node, nullptr);
  const auto subgraph_data1_node = graph->FindNode("sub_data1");
  EXPECT_EQ(subgraph_data1_node, nullptr);
  const auto subgraph_data2_node = graph->FindNode("sub_data2");
  EXPECT_EQ(subgraph_data2_node, nullptr);

  const auto sub_add0_node = graph->FindNode("sub_add0");
  EXPECT_EQ(sub_add0_node, sub_add0);
  const auto add0_in_data_anchors = sub_add0_node->GetAllInDataAnchors();
  const auto add0_in_data_anchor_0 = add0_in_data_anchors.at(0);
  const auto peer_out_add0_in_data_anchor_0 = add0_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add0_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add0_in_data_anchor_0->GetOwnerNode()->GetName(), "data0");
  const auto add0_in_data_anchor_1 = add0_in_data_anchors.at(1);
  const auto peer_out_add0_in_data_anchor_1 = add0_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add0_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add0_in_data_anchor_1->GetOwnerNode()->GetName(), "data1");

  const auto add0_out_data_anchor = sub_add0_node->GetOutDataAnchor(0);
  ASSERT_NE(add0_out_data_anchor, nullptr);
  const auto peer_in_add0_out_data_anchors = add0_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add0_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_add0_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "relu0");

  const auto sub_add1_node = graph->FindNode("sub_add1");
  EXPECT_EQ(sub_add1_node, sub_add1);
  const auto add1_in_data_anchors = sub_add1_node->GetAllInDataAnchors();
  const auto add1_in_data_anchor_0 = add1_in_data_anchors.at(0);
  const auto peer_out_add1_in_data_anchor_0 = add1_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_0->GetOwnerNode()->GetName(), "data1");
  const auto add1_in_data_anchor_1 = add1_in_data_anchors.at(1);
  const auto peer_out_add1_in_data_anchor_1 = add1_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_1->GetOwnerNode()->GetName(), "data2");

  const auto add1_out_data_anchor = sub_add1_node->GetOutDataAnchor(0);
  ASSERT_NE(add1_out_data_anchor, nullptr);
  const auto peer_in_add1_out_data_anchors = add1_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add1_out_data_anchors.size(), 1);
  EXPECT_EQ(peer_in_add1_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "relu1");

  // 验证topo序
  std::vector<std::string> expect_sort;
  for (const auto &origin_node_name : origin_node_sort) {
    if (origin_node_name == "identityn0") {
      for (const auto &subgraph_node : sub_graph->GetDirectNode()) {
        if ((subgraph_node->GetType() != "Data") && (subgraph_node->GetType() != "NetOutput")) {
          expect_sort.emplace_back(subgraph_node->GetName());
        }
      }
      continue;
    }
    expect_sort.emplace_back(origin_node_name);
  }
  size_t index = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), expect_sort[index]);
    index++;
  }
}

/*
    Data    Data     Data
    |        |         |
    \        |   /     Relu                                        Data    Data   Data
          identity     |                           --->              \      /  \     /
                \     /                                                Add0       Add1      
                 Add                                
*/
TEST_F(UtestGraphUtils, TestExpandNodeWithOutputWithOutNodeInfos) {
  auto builder = ut::GraphBuilder("test_expand_node_output_not_match");
  const auto &data0 = builder.AddNode("data0", DATA, 0, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 0, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 0, 1);
  const auto &identityN = builder.AddNode("identityn0", "IdentityN", 3, 3);
  const auto &relu0 = builder.AddNode("relu0", "Relu", 1, 1);
  const auto &add0 = builder.AddNode("add0", "Add", 2, 1);

  // 设置属性
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);

  builder.AddDataEdge(data0, 0, identityN, 0);
  builder.AddDataEdge(data1, 0, identityN, 1);
  builder.AddDataEdge(data2, 0, identityN, 2);
  builder.AddDataEdge(data2, 0, relu0, 0);
  builder.AddDataEdge(identityN, 1, add0, 0);
  builder.AddDataEdge(relu0, 0, add0, 1);
  auto graph = builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{identityN, 0}, {identityN, 1}, {add0, 0}};
  graph->SetOutputSize(3U);
  graph->SetGraphOutNodesInfo(output_nodes);

  std::vector<std::string> origin_node_sort;
  for (const auto &node : graph->GetDirectNode()) {
    origin_node_sort.emplace_back(node->GetName());
  }

  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &sub_data0 = sub_builder.AddNode("sub_data0", DATA, 0, 1);
  const auto &sub_data1 = sub_builder.AddNode("sub_data1", DATA, 0, 1);
  const auto &sub_data2 = sub_builder.AddNode("sub_data2", DATA, 0, 1);
  const auto &sub_add0 = sub_builder.AddNode("sub_add0", "Add", 2, 1);
  const auto &sub_add1 = sub_builder.AddNode("sub_add1", "Add", 2, 1);
  // 设置属性
  AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(sub_data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
  AttrUtils::SetInt(sub_data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
  sub_builder.AddDataEdge(sub_data0, 0, sub_add0, 0);
  sub_builder.AddDataEdge(sub_data1, 0, sub_add0, 1);
  sub_builder.AddDataEdge(sub_data1, 0, sub_add1, 0);
  sub_builder.AddDataEdge(sub_data2, 0, sub_add1, 1);
  auto sub_graph = sub_builder.GetGraph();
  std::vector<std::pair<NodePtr, int32_t>> sub_output_nodes{{sub_add0, 0}, {sub_add1, 0}};
  sub_graph->SetOutputSize(2U);
  sub_graph->SetGraphOutNodesInfo(sub_output_nodes);

  EXPECT_EQ(GraphUtils::ExpandNodeWithGraph(identityN, sub_graph), SUCCESS);
  const auto identityn_node = graph->FindNode("identityn0");
  EXPECT_EQ(identityn_node, nullptr);
  const auto subgraph_data0_node = graph->FindNode("sub_data0");
  EXPECT_EQ(subgraph_data0_node, nullptr);
  const auto subgraph_data1_node = graph->FindNode("sub_data1");
  EXPECT_EQ(subgraph_data1_node, nullptr);
  const auto subgraph_data2_node = graph->FindNode("sub_data2");
  EXPECT_EQ(subgraph_data2_node, nullptr);

  const auto sub_add0_node = graph->FindNode("sub_add0");
  EXPECT_EQ(sub_add0_node, sub_add0);
  const auto add0_in_data_anchors = sub_add0_node->GetAllInDataAnchors();
  const auto add0_in_data_anchor_0 = add0_in_data_anchors.at(0);
  const auto peer_out_add0_in_data_anchor_0 = add0_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add0_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add0_in_data_anchor_0->GetOwnerNode()->GetName(), "data0");
  const auto add0_in_data_anchor_1 = add0_in_data_anchors.at(1);
  const auto peer_out_add0_in_data_anchor_1 = add0_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add0_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add0_in_data_anchor_1->GetOwnerNode()->GetName(), "data1");

  const auto add0_out_data_anchor = sub_add0_node->GetOutDataAnchor(0);
  ASSERT_NE(add0_out_data_anchor, nullptr);
  const auto peer_in_add0_out_data_anchors = add0_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add0_out_data_anchors.size(), 1UL); // 连接NetOutput

  const auto sub_add1_node = graph->FindNode("sub_add1");
  EXPECT_EQ(sub_add1_node, sub_add1);
  const auto add1_in_data_anchors = sub_add1_node->GetAllInDataAnchors();
  const auto add1_in_data_anchor_0 = add1_in_data_anchors.at(0);
  const auto peer_out_add1_in_data_anchor_0 = add1_in_data_anchor_0->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_0, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_0->GetOwnerNode()->GetName(), "data1");
  const auto add1_in_data_anchor_1 = add1_in_data_anchors.at(1);
  const auto peer_out_add1_in_data_anchor_1 = add1_in_data_anchor_1->GetPeerOutAnchor();
  ASSERT_NE(peer_out_add1_in_data_anchor_1, nullptr);
  EXPECT_EQ(peer_out_add1_in_data_anchor_1->GetOwnerNode()->GetName(), "data2");

  const auto add1_out_data_anchor = sub_add1_node->GetOutDataAnchor(0);
  ASSERT_NE(add1_out_data_anchor, nullptr);
  const auto peer_in_add1_out_data_anchors = add1_out_data_anchor->GetPeerInDataAnchors();
  EXPECT_EQ(peer_in_add1_out_data_anchors.size(), 2); // 连接NetOutput
  EXPECT_EQ(peer_in_add1_out_data_anchors.at(0)->GetOwnerNode()->GetName(), "add0");

  // 验证输出NodeInfo
  const auto output_node_infos = graph->GetGraphOutNodesInfo();
  std::vector<std::pair<std::string, int32_t>> expect_node = {{"sub_add0", 0}, {"sub_add1", 0}, {"add0", 0}};
  ASSERT_EQ(output_node_infos.size(), expect_node.size());
  for (size_t i = 0UL; i < output_node_infos.size(); i++) {
    EXPECT_EQ(output_node_infos[i].first->GetName(), expect_node[i].first);
    EXPECT_EQ(output_node_infos[i].second, expect_node[i].second);
  }
  // 验证topo序
  std::vector<std::string> expect_sort;
  for (const auto &origin_node_name : origin_node_sort) {
    if (origin_node_name == "identityn0") {
      for (const auto &subgraph_node : sub_graph->GetDirectNode()) {
        if ((subgraph_node->GetType() != "Data") && (subgraph_node->GetType() != "NetOutput")) {
          expect_sort.emplace_back(subgraph_node->GetName());
        }
      }
      continue;
    }
    expect_sort.emplace_back(origin_node_name);
  }
  size_t index = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), expect_sort[index]);
    index++;
  }
}
TEST_F(UtestGraphUtils, CreateGraphPtrFromComputeGraphOk) {
  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->GetName(), "test_graph");
  auto cg2 = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(compute_graph.get(), cg2.get());
}
TEST_F(UtestGraphUtils, CreateGraphPtrFromComputeGraph_nullptr) {
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(nullptr);
  ASSERT_EQ(graph, nullptr);
}

TEST_F(UtestGraphUtils, ConvertInDataEdgesToInCtrlEdges_nullptr) {
  EXPECT_EQ(GRAPH_PARAM_INVALID, GraphUtils::ConvertInDataEdgesToInCtrlEdges(nullptr, nullptr, nullptr));
}

TEST_F(UtestGraphUtils, ConvertOutDataEdgesToOutCtrlEdges_nullptr) {
  EXPECT_EQ(GRAPH_PARAM_INVALID, GraphUtils::ConvertOutDataEdgesToOutCtrlEdges(nullptr, nullptr, nullptr));
}

TEST_F(UtestGraphUtils, IsolateNodeNodeWithNoOpOptimize) {
  auto graph = BuildGraphForIsolateNode(30, 30, 10, 10);
  auto node = graph->FindNode("del_node");
  // 断掉所有连边
  std::vector<int> io_map = {};
  auto ret = GraphUtils::IsolateNode(node, io_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto noop = graph->FindFirstNodeMatchType(NOOP);
  EXPECT_NE(noop, nullptr);
  EXPECT_EQ(noop->GetInControlNodesSize(), 40UL);
  EXPECT_EQ(noop->GetOutControlNodesSize(), 40UL);
}

TEST_F(UtestGraphUtils, IsolateNodeNodeWithNoOpOptimize_AllCtrlEdge) {
  // 达到阈值触发NoOp优化
  int io_num = 32;
  auto graph = BuildGraphForIsolateNode(0, 0, io_num, io_num);
  auto node = graph->FindNode("del_node");
  std::vector<int> io_map{};
  auto ret = GraphUtils::IsolateNode(node, io_map);
  GraphUtils::DumpGEGraphToOnnx(*graph, "TestIsolateNode_After");
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto noop = graph->FindFirstNodeMatchType(NOOP);
  EXPECT_NE(noop, nullptr);
  EXPECT_EQ(noop->GetInControlNodesSize(), io_num);
  EXPECT_EQ(noop->GetOutControlNodesSize(), io_num);

  auto in_node = graph->FindNode("in_ctrl_node_0");
  auto out_node = graph->FindNode("out_ctrl_node_0");
  EXPECT_NE(in_node, nullptr);
  EXPECT_NE(out_node, nullptr);
  EXPECT_EQ(in_node->GetOutControlNodesSize(), 1);
  EXPECT_EQ(out_node->GetInControlNodesSize(), 1);
}

TEST_F(UtestGraphUtils, IsolateNodeNodeWithOutNoOpOptimize_AllCtrlEdge) {
  // 未达到阈值，未触发NoOp优化
  int io_num = 30;
  auto graph = BuildGraphForIsolateNode(0, 0, io_num, io_num);
  auto node = graph->FindNode("del_node");
  std::vector<int> io_map{};
  auto ret = GraphUtils::IsolateNode(node, io_map);
  GraphUtils::DumpGEGraphToOnnx(*graph, "TestIsolateNode_After");
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto noop = graph->FindFirstNodeMatchType(NOOP);
  EXPECT_EQ(noop, nullptr);

  auto in_node = graph->FindNode("in_ctrl_node_0");
  auto out_node = graph->FindNode("out_ctrl_node_0");
  EXPECT_NE(in_node, nullptr);
  EXPECT_NE(out_node, nullptr);
  EXPECT_EQ(in_node->GetOutControlNodesSize(), io_num);
  EXPECT_EQ(out_node->GetInControlNodesSize(), io_num);
}

TEST_F(UtestGraphUtils, IsolateNodeNodeWithNoOpOptimize_SetIoMap_NoInAnchr) {
  int io_num = 32;
  auto graph = BuildGraphForIsolateNode(io_num, io_num, 0, 0);
  auto node = graph->FindNode("del_node");
  // 指定输入0和所有输出建立数据边，因此该输入节点不需要连控制边到NoOp
  std::vector<int> io_map(io_num, 0);
  auto ret = GraphUtils::IsolateNode(node, io_map);
  GraphUtils::DumpGEGraphToOnnx(*graph, "TestIsolateNode_After");
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto noop = graph->FindFirstNodeMatchType(NOOP);
  EXPECT_NE(noop, nullptr);
  EXPECT_EQ(noop->GetInControlNodesSize(), io_num - 1);
  EXPECT_EQ(noop->GetOutControlNodesSize(), io_num);

  auto in_node = graph->FindNode("in_node_0");
  auto out_node = graph->FindNode("out_node_0");
  EXPECT_NE(in_node, nullptr);
  EXPECT_NE(out_node, nullptr);
  EXPECT_NE(out_node->GetInDataAnchor(0), nullptr);
  EXPECT_NE(out_node->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(out_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), in_node);
}

TEST_F(UtestGraphUtils, IsolateNodeNodeWithNoOpOptimize_SetIoMap_NoOutAnchr) {
  auto graph_builder = ut::GraphBuilder("graph");
  const auto &del_node = graph_builder.AddNode("del_node", "DelNode", 2, 510);

  const auto &n1 = graph_builder.AddNode("in_node_1", "InNode", 1, 1);
  graph_builder.AddDataEdge(n1, 0, del_node, 0);

  const auto &n2 = graph_builder.AddNode("in_node_2", "InNode", 1, 1);
  graph_builder.AddDataEdge(n2, 0, del_node, 1);

  const auto &n3 = graph_builder.AddNode("out_node_0", "OutNode", 2, 1);
  graph_builder.AddDataEdge(del_node, 0, n3, 0);
  graph_builder.AddDataEdge(del_node, 1, n3, 1);

  for (int i = 2; i < 510; ++i) {
    const auto &n = graph_builder.AddNode("out_node" + std::to_string(i), "OutNode", 1, 1);
    graph_builder.AddDataEdge(del_node, i, n, 0);
  }
  auto graph = graph_builder.GetGraph();


  auto node = graph->FindNode("del_node");
  setenv("DUMP_GE_GRAPH", "1", 1);
  setenv("DUMP_GRAPH_LEVEL", "1", 1);
  setenv("DUMP_GRAPH_PATH", "/home/yangyongqiang/code/dump_graph", 1);
  dlog_setlevel(0, 0, 0);
  GraphUtils::DumpGEGraphToOnnx(*graph, "TestIsolateNode_Before");

  // 指定输出0和所有输入建立数据边，因此该输出节点不需要连控制边到NoOp
  std::vector<int> io_map = {0, 1};
  auto ret = GraphUtils::IsolateNode(node, io_map);
  GraphUtils::DumpGEGraphToOnnx(*graph, "TestIsolateNode_After");
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto noop = graph->FindFirstNodeMatchType(NOOP);
  EXPECT_NE(noop, nullptr);
  EXPECT_EQ(noop->GetInControlNodesSize(), 2);
  EXPECT_EQ(noop->GetOutControlNodesSize(), 508);

  auto in_node_1 = graph->FindNode("in_node_1");
  auto in_node_2 = graph->FindNode("in_node_2");
  auto out_node_0 = graph->FindNode("out_node_0");
  EXPECT_NE(in_node_1, nullptr);
  EXPECT_NE(in_node_2, nullptr);
  EXPECT_NE(out_node_0, nullptr);
  EXPECT_NE(out_node_0->GetInDataAnchor(0), nullptr);
  EXPECT_NE(out_node_0->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(out_node_0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), in_node_1);
  EXPECT_NE(out_node_0->GetInDataAnchor(1), nullptr);
  EXPECT_NE(out_node_0->GetInDataAnchor(1)->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(out_node_0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode(), in_node_2);
}

}  // namespace ge
