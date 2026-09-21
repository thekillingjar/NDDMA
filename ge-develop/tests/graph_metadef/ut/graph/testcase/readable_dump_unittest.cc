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
#include <google/protobuf/text_format.h>

#include "graph/utils/graph_utils.h"
#include "graph/graph_buffer.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_builder_utils.h"
#include "ge_ir.pb.h"
#include "node_adapter.h"
#include "tensor_adapter.h"
#include "tensor_utils.h"
#include "debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ge_context.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/readable_dump.h"
#include "graph/operator_reg.h"
#include "es_graph_builder.h"
#include "normal_graph/operator_impl.h"
#include "default_attr_utils.h"
#include "graph_dump_utils.h"
#include "node_utils.h"
#include "common/util/tiling_utils.h"
#include "recover_ir_utils.h"

using namespace ge;
namespace {
ComputeGraphPtr BuildGraphWithConst(const std::string &graph_name = "graph") {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder(graph_name);
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  AttrUtils::SetStr(const_node->GetOpDesc(), "fake_attr_name", "fake_attr_value");
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  RecoverIrUtils::RecoverOpDescIrDefinition(add_node->GetOpDesc(), "Add");
  AttrUtils::SetStr(add_node->GetOpDesc(), "fake_attr_name", "fake_attr_value");
  AttrUtils::SetStr(add_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, "fake_attr_value");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  return builder.GetGraph();
}
} // namespace


class UtestReadableDump : public testing::Test {
 protected:
  void SetUp() {
    unsetenv("DUMP_GRAPH_PATH");
    unsetenv("DUMP_GE_GRAPH");
    unsetenv("DUMP_GRAPH_FORMAT");
    unsetenv("NPU_COLLECT_PATH");
  }

  void TearDown() {
    unsetenv("DUMP_GRAPH_PATH");
    unsetenv("DUMP_GE_GRAPH");
    unsetenv("DUMP_GRAPH_FORMAT");
  }

  static std::string GetFilePathWhenDumpPathSet(const ComputeGraphPtr graph, const string &ascend_work_path, const string &suffix = "") {
    std::string real_path_file;
    EXPECT_EQ(ge::SUCCESS, GraphUtils::GenDumpReadableTxtFileName(graph, "test", "", real_path_file));
    auto real_path = getParentDirectory(real_path_file);
    return real_path;
  }
  static std::vector<string> GetSpecificFilePath(const std::string &file_path, const string &suffix) {
    DIR *dir;
    struct dirent *ent;
    dir = opendir(file_path.c_str());
    std::vector<string> file_vec{};
    if (dir == nullptr) {
      return file_vec;
    }
    while ((ent = readdir(dir)) != nullptr) {
      if (strstr(ent->d_name, suffix.c_str()) != nullptr) {
        std::string d_name(ent->d_name);
        file_vec.emplace_back(d_name);
      }
    }
    closedir(dir);
    return file_vec;
  }
  static std::string getParentDirectory(const std::string& filepath) {
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
      return filepath.substr(0, lastSlash);
    }
    return "";
  }

};

TEST_F(UtestReadableDump, test_GenReadableDumpTextFile) {
  auto graph = BuildGraphWithConst("GenReadableDumpTextFile");
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  GraphUtils::DumpGEGraphToReadable(graph, "test", true, "");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(graph, ascend_work_path, "test");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "test");

  EXPECT_TRUE(!dump_graph_files.empty());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_GenDumpReadableTxtFileName) {
  auto graph = BuildGraphWithConst("GenDumpReadableTxtFileName");
  std::string real_path_name;
  EXPECT_EQ(ge::SUCCESS, GraphUtils::GenDumpReadableTxtFileName(graph, "test_dump_filename", "", real_path_name));
  EXPECT_TRUE(real_path_name.find("ge_readable_") != real_path_name.npos);
  EXPECT_TRUE(real_path_name.find("_graph_0_test_dump_filename.txt") != real_path_name.npos);

  graph->SetAttr(ATTR_SINGLE_OP_SCENE, GeAttrValue::CreateFrom<>(true));
  EXPECT_EQ(ge::SUCCESS, GraphUtils::GenDumpReadableTxtFileName(graph, "test_dump_filename", "", real_path_name));
  EXPECT_TRUE(real_path_name.find("_aclop_graph_0_test_dump_filename.txt") != real_path_name.npos);
}

TEST_F(UtestReadableDump, test_WriteReadableDumpToOStream) {
  std::stringstream readable_ss("test write file to os");
  std::ostringstream readable_os;
  GraphUtils::WriteReadableDumpToOStream(readable_ss, readable_os);
  EXPECT_EQ(readable_os.str(), readable_ss.str());
}

TEST_F(UtestReadableDump, test_DumpToFile) {
  auto compute_graph = BuildGraphWithConst("DumpToFile");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(Graph::DumpFormat::kReadable, "test_dump_to_file"));
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "test_dump_to_file");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "test_dump_to_file");

  EXPECT_EQ(1, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvDefault) {
  auto compute_graph = BuildGraphWithConst("DumpEnvDefault");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(2, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvSingle) {
  auto compute_graph = BuildGraphWithConst("DumpEnvSingle");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  mmSetEnv("DUMP_GRAPH_FORMAT", "ReAdAbLe", 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(1, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvMultiple) {
  auto compute_graph = BuildGraphWithConst("DumpEnvMultiple");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  mmSetEnv("DUMP_GRAPH_FORMAT", "onNx    |gE_PrOtO|ReAdAbLe", 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(3, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvUnknown) {
  auto compute_graph = BuildGraphWithConst("DumpEnvUnknown");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  mmSetEnv("DUMP_GRAPH_FORMAT", "txt", 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(0, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_GraphDumpToTile) {
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  auto compute_graph = BuildGraphWithConst("GraphDumpToTile");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(ge::Graph::DumpFormat::kReadable, ""));
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(ge::Graph::DumpFormat::kOnnx, ""));
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(ge::Graph::DumpFormat::kTxt, ""));
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_GraphDumpToOStream) {
  auto compute_graph = BuildGraphWithConst("GraphDumpToOStream");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::ostringstream readable_os;
  EXPECT_EQ(ge::SUCCESS, graph.Dump(ge::Graph::DumpFormat::kReadable, readable_os));
  EXPECT_EQ(ge::SUCCESS, graph.Dump(ge::Graph::DumpFormat::kOnnx, readable_os));
  EXPECT_EQ(ge::SUCCESS, graph.Dump(ge::Graph::DumpFormat::kTxt, readable_os));
}

REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const);
REG_OP(phony_mul_i)
    .INPUT(x, TensorType::ALL())
    .OPTIONAL_INPUT(opt_x, TensorType::ALL())
    .DYNAMIC_INPUT(dx1, TensorType::All())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(phony_mul_i);
REG_OP(Phony1)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::NumberType())
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Phony1);
REG_OP(phony_multi_attr)
    .OPTIONAL_INPUT(x, TensorType::ALL())
    .ATTR(li, ListInt, {10, 10, 10})
    .ATTR(f, Float, 0.0)
    .ATTR(s, String, "s")
    .ATTR(b, Bool, true)
    .ATTR(lf, ListFloat, {0.1, 0.2})
    .ATTR(lb, ListBool, {false, true})
    .ATTR(opt_data_type, Type, DT_INT64)
    .ATTR(opt_list_data_type, ListType, {DT_FLOAT, DT_DOUBLE})
    .ATTR(opt_list_list_int, ListListInt, {{1,2,3}, {3,2,1}})
    .ATTR(opt_tensor, Tensor, Tensor())
    .ATTR(opt_list_string, ListString, {"test"})
    .OUTPUT(y, TensorType::NumberType())
    .DYNAMIC_OUTPUT(dy, TensorType::All())
    .OP_END_FACTORY_REG(phony_multi_attr);
REG_OP(phony_mix_ios)
    .INPUT(x1, TensorType::All())
    .INPUT(x2, TensorType::All())
    .DYNAMIC_INPUT(dx1, TensorType::All())
    .OUTPUT(y1, TensorType::All())
    .OUTPUT(y2, TensorType::All())
    .DYNAMIC_OUTPUT(dy, TensorType::All())
    .OP_END_FACTORY_REG(phony_mix_ios);

TEST_F(UtestReadableDump, test_GenSimple) {
  auto compute_graph = BuildGraphWithConst("GenSimple");
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto node1 = graph.GetDirectNode().at(0);
  auto node2 = graph.GetDirectNode().at(1);
  GNode node_netOutput;
  for (const auto &node: graph.GetAllNodes()) {
    AscendString node_type;
    node.GetType(node_type);
    if (node_type == "NetOutput") {
      node_netOutput = node;
    }
  }

  graph.AddControlEdge(node1, node2);
  graph.AddControlEdge(node2, node_netOutput);
  std::string readable_dump = R"(graph("GenSimple"):
  %Data : [#users=1] = Node[type=Data] ()
  %Const : [#users=1] = Node[type=Const] ()
  %Add : [#users=1] = Node[type=Add] (inputs = (x1=%Data, x2=%Const))

  return (%Add)
)";
  std::stringstream readable_ss;
  ReadableDump::GenReadableDump(readable_ss, compute_graph);
  EXPECT_EQ(readable_dump, readable_ss.str());
}

TEST_F(UtestReadableDump, test_OptionalInputsAndControlEdges) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph_OptionalInputsAndControlEdges");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto const_op = op::Const("const").set_attr_value(ge::TensorAdapter::AsTensor(*ge_tensor));
  auto const_node = graph.AddNodeByOp(const_op);

  auto data1_op = op::Data("data1");
  auto data1_node = graph.AddNodeByOp(data1_op);
  graph.AddDataEdge(const_node, 0, data1_node, 0);

  auto data2_op = op::Data("data2");
  auto data2_node = graph.AddNodeByOp(data2_op);
  graph.AddDataEdge(const_node, 0, data2_node, 0);

  auto phony_mul_i_op = op::phony_mul_i("test").create_dynamic_input_byindex_dx1(2, 1);
  auto phony_mul_i_node = graph.AddNodeByOp(phony_mul_i_op);
  graph.AddDataEdge(const_node, 0, phony_mul_i_node, 0);
  graph.AddDataEdge(data1_node, 0, phony_mul_i_node, 2);
  graph.AddDataEdge(data2_node, 0, phony_mul_i_node, 3);

  // 更新OPTIONAL_INPUT opt_x的描述信息
  ge::TensorDesc tensor_desc;
  phony_mul_i_node.GetInputDesc(3, tensor_desc);
  phony_mul_i_node.UpdateInputDesc(3, tensor_desc);

  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);
  graph.AddControlEdge(const_node, netoutput_gnode);

  std::string readable_dump = R"(graph("graph_OptionalInputsAndControlEdges"):
  %const : [#users=1] = Node[type=Const] (attrs = {value: [0.000000]})
  %data1 : [#users=1] = Node[type=Data] (inputs = (x=%const), attrs = {index: 0})
  %data2 : [#users=1] = Node[type=Data] (inputs = (x=%const), attrs = {index: 0})
  %test : [#users=1] = Node[type=phony_mul_i] (inputs = (x=%const, opt_x=%data2, dx1_1=%data1))
)";

  std::stringstream readable_ss;
  ReadableDump::GenReadableDump(readable_ss, compute_graph);
  EXPECT_EQ(readable_dump, readable_ss.str());
}

TEST_F(UtestReadableDump, test_GenComplex) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph_complex");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 5, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto data_op = op::Data("data");
  auto data_node = graph.AddNodeByOp(data_op);
  auto const_op = op::Const("const").set_attr_value(ge::TensorAdapter::AsTensor(*ge_tensor));
  auto const_node = graph.AddNodeByOp(const_op);

  auto phony1_op = op::Phony1("phony1").set_attr_N(1);
  auto phony1_node = graph.AddNodeByOp(phony1_op);
  graph.AddDataEdge(data_node, 0, phony1_node, 0);

  auto phony_multi_attr_op1 = op::phony_multi_attr("phony_multi_attr_node1");
  auto phony_multi_attr_node1 = graph.AddNodeByOp(phony_multi_attr_op1);
  auto phony_multi_attr_op2 = op::phony_multi_attr("phony_multi_attr_node2");
  phony_multi_attr_op2.create_dynamic_output_dy(2);
  auto phony_multi_attr_node2 = graph.AddNodeByOp(phony_multi_attr_op2);
  graph.AddDataEdge(data_node, 0, phony_multi_attr_node2, 0);
  // 更新OPTIONAL_INPUT x的描述信息
  ge::TensorDesc tensor_desc;
  phony_multi_attr_node2.GetInputDesc(0, tensor_desc);
  phony_multi_attr_node2.UpdateInputDesc(0, tensor_desc);

  auto phony_mix_ios_op = op::phony_mix_ios("phony_mix_ios")
                              .create_dynamic_input_byindex_dx1(2, 2);
  phony_mix_ios_op.create_dynamic_output_dy(2);
  auto phony_mix_ios_node = graph.AddNodeByOp(phony_mix_ios_op);
  graph.AddDataEdge(data_node, 0, phony_mix_ios_node, 0);
  graph.AddDataEdge(phony1_node, 0, phony_mix_ios_node, 1);
  graph.AddDataEdge(phony_multi_attr_node2, 0, phony_mix_ios_node, 2);
  graph.AddDataEdge(phony_multi_attr_node2, 1, phony_mix_ios_node, 3);

  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);

  graph.AddDataEdge(phony1_node, 0, netoutput_gnode, 0);
  graph.AddDataEdge(phony_multi_attr_node2, 0, netoutput_gnode, 1);
  graph.AddDataEdge(phony_multi_attr_node2, 1, netoutput_gnode, 2);
  graph.AddDataEdge(phony_mix_ios_node, 0, netoutput_gnode, 3);
  graph.AddDataEdge(phony_mix_ios_node, 1, netoutput_gnode, 4);

  std::string readable_dump = R"(graph("graph_complex"):
  %data : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %const : [#users=1] = Node[type=Const] (attrs = {value: [0.000000]})
  %phony1 : [#users=1] = Node[type=Phony1] (inputs = (x=%data), attrs = {N: 1})
  %phony_multi_attr_node1 : [#users=1] = Node[type=phony_multi_attr] (attrs = {li: {10, 10, 10}, f: 0.000000, s: "s", b: true, lf: {0.100000, 0.200000}, lb: {false, true}, opt_data_type: DT_INT64, opt_list_data_type: {DT_FLOAT, DT_DOUBLE}, opt_list_list_int: {{1, 2, 3}, {3, 2, 1}}, opt_tensor: <empty>, opt_list_string: {"test"}})
  %phony_multi_attr_node2 : [#users=3] = Node[type=phony_multi_attr] (inputs = (x=%data), attrs = {li: {10, 10, 10}, f: 0.000000, s: "s", b: true, lf: {0.100000, 0.200000}, lb: {false, true}, opt_data_type: DT_INT64, opt_list_data_type: {DT_FLOAT, DT_DOUBLE}, opt_list_list_int: {{1, 2, 3}, {3, 2, 1}}, opt_tensor: <empty>, opt_list_string: {"test"}})
  %ret : [#users=2] = get_element[node=%phony_multi_attr_node2](0)
  %ret_1 : [#users=2] = get_element[node=%phony_multi_attr_node2](1)
  %ret_2 : [#users=0] = get_element[node=%phony_multi_attr_node2](2)
  %phony_mix_ios : [#users=4] = Node[type=phony_mix_ios] (inputs = (x1=%data, x2=%phony1, dx1_0=%ret, dx1_1=%ret_1))
  %ret_3 : [#users=1] = get_element[node=%phony_mix_ios](0)
  %ret_4 : [#users=1] = get_element[node=%phony_mix_ios](1)
  %ret_5 : [#users=0] = get_element[node=%phony_mix_ios](2)
  %ret_6 : [#users=0] = get_element[node=%phony_mix_ios](3)

  return (output_0=%phony1, output_1=%ret, output_2=%ret_1, output_3=%ret_3, output_4=%ret_4)
)";

  std::stringstream readable_ss;
  ReadableDump::GenReadableDump(readable_ss, compute_graph);
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_MemSet)
    .REQUIRED_ATTR(sizes, ListInt)
    .ATTR(dtypes, ListType, {})
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(phony_MemSet);

TEST_F(UtestReadableDump, test_EmptyListAttr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmptyListAttr");
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto data_op = op::Data("data");
  auto data_node = graph.AddNodeByOp(data_op);

  std::vector<ge::DataType> dtype_vec = {DT_FLOAT};
  auto memset_op = op::phony_MemSet("test").set_attr_sizes({2560}).set_attr_dtypes(dtype_vec);
  auto memset_node = graph.AddNodeByOp(memset_op);

  graph.AddControlEdge(data_node, memset_node);

  std::string readable_dump = R"(graph("graph_EmptyListAttr"):
  %data : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %test : [#users=0] = Node[type=phony_MemSet] (attrs = {sizes: {2560}, dtypes: {DT_FLOAT}})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_attr_tensors)
    .ATTR(t1, Tensor, Tensor())
    .ATTR(t2, Tensor, Tensor())
    .ATTR(t3, Tensor, Tensor())
    .ATTR(t4, Tensor, Tensor())
    .ATTR(t5, Tensor, Tensor())
    .ATTR(t6, Tensor, Tensor())
    .ATTR(t7, Tensor, Tensor())
    .ATTR(t8, Tensor, Tensor())
    .ATTR(t9, Tensor, Tensor())
    .ATTR(t10, Tensor, Tensor())
    .ATTR(t11, Tensor, Tensor())
    .OP_END_FACTORY_REG(phony_attr_tensors);

TEST_F(UtestReadableDump, test_AttrTensorValues) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmptyListAttr");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  es::EsGraphBuilder es_builder("test_attr_tensor");
  std::vector<float> data1 = {-5.5, -4.4, -3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3, 4.4, 5.5};
  std::vector<int64_t> dims1 = {11};
  auto tensor1 = es_builder.CreateTensor<float>(data1, dims1, ge::DT_FLOAT);
  std::vector<int8_t> data2 = {-5, -4, -3, -2, -1, 0};
  std::vector<int64_t> dims2 = {6};
  auto tensor2 = es_builder.CreateTensor<int8_t>(data2, dims2, ge::DT_INT8);
  std::vector<int16_t> data3 = {-5, -4, -3, -2, -1};
  std::vector<int64_t> dims3 = {5};
  auto tensor3 = es_builder.CreateTensor<int16_t>(data3, dims3, ge::DT_INT16);
  std::vector<int32_t> data4 = {-5, -4, -3, -2};
  std::vector<int64_t> dims4 = {4};
  auto tensor4 = es_builder.CreateTensor<int32_t>(data4, dims4, ge::DT_INT32);
  std::vector<int64_t> data5 = {-5, -4, -3};
  std::vector<int64_t> dims5 = {3};
  auto tensor5 = es_builder.CreateTensor<int64_t>(data5, dims5, ge::DT_INT64);
  std::vector<uint8_t> data6 = {5, 4};
  std::vector<int64_t> dims6 = {2};
  auto tensor6 = es_builder.CreateTensor<uint8_t>(data6, dims6, ge::DT_UINT8);
  std::vector<uint16_t> data7 = {5};
  std::vector<int64_t> dims7 = {1};
  auto tensor7 = es_builder.CreateTensor<uint16_t>(data7, dims7, ge::DT_UINT16);
  std::vector<uint32_t> data8 = {0};
  std::vector<int64_t> dims8 = {};
  auto tensor8 = es_builder.CreateTensor<uint32_t>(data8, dims8, ge::DT_UINT32);
  std::vector<uint64_t> data9 = {5, 6};
  std::vector<int64_t> dims9 = {2};
  auto tensor9 = es_builder.CreateTensor<uint64_t>(data9, dims9, ge::DT_UINT64);
  std::vector data10 = {true, false, true, false};
  std::vector<int64_t> dims10 = {4};
  auto tensor10 = es_builder.CreateBoolTensor(data10, dims10);
  // Create FP16 tensor
  std::vector<float> fp16_data_float = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<uint16_t> fp16_data;
  fp16_data.reserve(fp16_data_float.size());
  for (const auto &val : fp16_data_float) {
    fp16_data.push_back(optiling::Float32ToFloat16(val));
  }
  std::vector<int64_t> dims11 = {5};
  auto tensor11 = es_builder.CreateTensor<uint16_t>(fp16_data, dims11, ge::DT_FLOAT16);

  auto op = op::phony_attr_tensors("test_attr_tensor")
                .set_attr_t1(*tensor1)
                .set_attr_t2(*tensor2)
                .set_attr_t3(*tensor3)
                .set_attr_t4(*tensor4)
                .set_attr_t5(*tensor5)
                .set_attr_t6(*tensor6)
                .set_attr_t7(*tensor7)
                .set_attr_t8(*tensor8)
                .set_attr_t9(*tensor9)
                .set_attr_t10(*tensor10)
                .set_attr_t11(*tensor11);
  (void) graph.AddNodeByOp(op);
  std::string readable_dump = R"(graph("graph_EmptyListAttr"):
  %test_attr_tensor : [#users=0] = Node[type=phony_attr_tensors] (attrs = {t1: [-5.500000 -4.400000 -3.300000 ... 3.300000 4.400000 5.500000], t2: [-5 -4 -3 -2 -1 0], t3: [-5 -4 -3 -2 -1], t4: [-5 -4 -3 -2], t5: [-5 -4 -3], t6: [5 4], t7: [5], t8: [0], t9: [5 6], t10: [true false true false], t11: [1.000000 2.000000 3.000000 4.000000 5.000000]})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_VariableV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(phony_VariableV2)

TEST_F(UtestReadableDump, test_EmptyString) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmptyListAttr");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto op = op::phony_VariableV2("test_empty_string");
  (void) graph.AddNodeByOp(op);

  std::string readable_dump = R"(graph("graph_EmptyListAttr"):
  %test_empty_string : [#users=1] = Node[type=phony_VariableV2] (attrs = {index: 0, value: <empty>})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(phony_Squeeze)

TEST_F(UtestReadableDump, test_phony_Squeeze_WithInputNoAttr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_WithInputNoAttr");
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto data_op = op::Data("data");
  auto data_node = graph.AddNodeByOp(data_op);

  auto squeeze_op = op::phony_Squeeze("phony_Squeeze");
  auto squeeze_node = graph.AddNodeByOp(squeeze_op);

  graph.AddDataEdge(data_node, 0, squeeze_node, 0);

  std::string readable_dump = R"(graph("graph_WithInputNoAttr"):
  %data : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %phony_Squeeze : [#users=1] = Node[type=phony_Squeeze] (inputs = (x=%data))
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_UnsupportedAttr)
.ATTR(unsupported_attr, ListAscendString, {})
.OP_END_FACTORY_REG(phony_UnsupportedAttr)

TEST_F(UtestReadableDump, test_UnsupportedAttr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_UnsupportedAttr");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto op = op::phony_UnsupportedAttr("test_unsupported_attr");
  (void) graph.AddNodeByOp(op);
  std::string readable_dump = R"(graph("graph_UnsupportedAttr"):
  %test_unsupported_attr : [#users=0] = Node[type=phony_UnsupportedAttr] ()
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_attr_empty)
    .OPTIONAL_INPUT(x, TensorType::ALL())
    .ATTR(li, ListInt, {})
    .ATTR(f, Float, 0.0)
    .ATTR(s, String, "")
    .ATTR(b, Bool, true)
    .ATTR(lf, ListFloat, {})
    .ATTR(lb, ListBool, {})
    .ATTR(opt_data_type, Type, DT_INT64)
    .ATTR(opt_list_data_type, ListType, {DT_FLOAT, DT_DOUBLE})
    .ATTR(opt_list_list_int, ListListInt, {{}, {2, 3}, {}})
    .ATTR(opt_tensor, Tensor, Tensor())
    .ATTR(opt_list_string, ListString, {""})
    .OUTPUT(y, TensorType::NumberType())
    .DYNAMIC_OUTPUT(dy, TensorType::All())
    .OP_END_FACTORY_REG(phony_attr_empty);

TEST_F(UtestReadableDump, test_EmtpyDefaultValues) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmtpyDefaultValues");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto op = op::phony_attr_empty("test_empty_attr");
  (void) graph.AddNodeByOp(op);
  auto op_desc = OperatorImpl::GetOpDesc(op);
  auto attr_list_int = ge::AttrString::GetDefaultValueString(op_desc, "li", "VT_LIST_INT");
  auto attr_list_list_int = ge::AttrString::GetDefaultValueString(op_desc, "opt_list_list_int", "VT_LIST_LIST_INT");
  EXPECT_EQ("{}", attr_list_int);
  EXPECT_EQ("{{}, {2, 3}, {}}", attr_list_list_int);

  std::string readable_dump = R"(graph("graph_EmtpyDefaultValues"):
  %test_empty_attr : [#users=1] = Node[type=phony_attr_empty] (attrs = {f: 0.000000, b: true, opt_data_type: DT_INT64, opt_list_data_type: {DT_FLOAT, DT_DOUBLE}, opt_list_list_int: {{2, 3}}, opt_tensor: <empty>})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(Add)
    .INPUT(x1, TensorType::All())
    .INPUT(x2, TensorType::All())
    .OUTPUT(y, TensorType::All())
    .OP_END_FACTORY_REG(Add)
REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If)
REG_OP(While)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .GRAPH(body)
    .ATTR(parallel_iterations, Int, 10)
    .OP_END_FACTORY_REG(While)
REG_OP(Case)
    .INPUT(branch_index, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .DYNAMIC_GRAPH(branches)
    .OP_END_FACTORY_REG(Case)
REG_OP(For)
    .INPUT(start, DT_INT32)
    .INPUT(limit, DT_INT32)
    .INPUT(delta, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(body)
    .OP_END_FACTORY_REG(For)

// 测试子图内联展示功能
TEST_F(UtestReadableDump, test_SubgraphInlineDisplay) {
  // 创建主图
  ut::GraphBuilder builder = ut::GraphBuilder("main_graph");
  auto main_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(main_graph);
  auto cond_op = op::Data("cond").set_attr_index(0);
  auto cond_node = graph.AddNodeByOp(cond_op);
  auto input0_op = op::Data("input0").set_attr_index(1);
  auto input0_node = graph.AddNodeByOp(input0_op);
  auto input1_op = op::Data("input1").set_attr_index(2);
  auto input1_node = graph.AddNodeByOp(input1_op);
  auto if_op = op::If("if_node")
                   .create_dynamic_input_byindex_input(2, 1)
                   .create_dynamic_output_output(1);
  auto if_node_gnode = graph.AddNodeByOp(if_op);
  graph.AddDataEdge(cond_node, 0, if_node_gnode, 0);
  graph.AddDataEdge(input0_node, 0, if_node_gnode, 1);
  graph.AddDataEdge(input1_node, 0, if_node_gnode, 2);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);
  graph.AddDataEdge(if_node_gnode, 0, netoutput_gnode, 0);
  
  // 创建then子图
  ut::GraphBuilder then_builder = ut::GraphBuilder("then_branch");
  auto then_compute_graph = then_builder.GetGraph();
  auto then_graph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(then_compute_graph);
  auto then_data0_op = op::Data("Data_0").set_attr_index(0);
  auto then_data0_node = then_graph_obj.AddNodeByOp(then_data0_op);
  auto then_data1_op = op::Data("Data_1").set_attr_index(1);
  auto then_data1_node = then_graph_obj.AddNodeByOp(then_data1_op);
  auto then_add_op = op::Add("Add_0");
  auto then_add_node = then_graph_obj.AddNodeByOp(then_add_op);
  then_graph_obj.AddDataEdge(then_data0_node, 0, then_add_node, 0);
  then_graph_obj.AddDataEdge(then_data1_node, 0, then_add_node, 1);
  const auto &then_netoutput = then_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode then_netoutput_gnode = NodeAdapter::Node2GNode(then_netoutput);
  then_graph_obj.AddDataEdge(then_add_node, 0, then_netoutput_gnode, 0);
  auto then_graph = then_compute_graph;
  
  // 创建else子图
  ut::GraphBuilder else_builder = ut::GraphBuilder("else_branch");
  auto else_compute_graph = else_builder.GetGraph();
  auto else_graph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(else_compute_graph);
  auto else_data0_op = op::Data("Data_0").set_attr_index(0);
  auto else_data0_node = else_graph_obj.AddNodeByOp(else_data0_op);
  auto else_data1_op = op::Data("Data_1").set_attr_index(1);
  auto else_data1_node = else_graph_obj.AddNodeByOp(else_data1_op);
  auto else_add_op = op::Add("Add_1");
  auto else_add_node = else_graph_obj.AddNodeByOp(else_add_op);
  else_graph_obj.AddDataEdge(else_data0_node, 0, else_add_node, 0);
  else_graph_obj.AddDataEdge(else_data1_node, 0, else_add_node, 1);
  const auto &else_netoutput = else_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode else_netoutput_gnode = NodeAdapter::Node2GNode(else_netoutput);
  else_graph_obj.AddDataEdge(else_add_node, 0, else_netoutput_gnode, 0);
  auto else_graph = else_compute_graph;
  
  // 将子图添加到主图的If节点
  NodeUtils::SetSubgraph(*NodeAdapter::GNode2Node(if_node_gnode), 0, then_graph);
  NodeUtils::SetSubgraph(*NodeAdapter::GNode2Node(if_node_gnode), 1, else_graph);
  
  std::string readable_dump = R"(graph("main_graph"):
  %cond : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %input0 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %input1 : [#users=1] = Node[type=Data] (attrs = {index: 2})
  %if_node : [#users=1] = Node[type=If] (inputs = (cond=%cond, input_0=%input0, input_1=%input1), attrs = {then_branch: %then_branch, else_branch: %else_branch})

  return (%if_node)

graph("then_branch"):
  %Data_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Data_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %Add_0 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_0, x2=%Data_1))

  return (%Add_0)

graph("else_branch"):
  %Data_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Data_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %Add_1 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_0, x2=%Data_1))

  return (%Add_1)
)";
  
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, main_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

// 测试嵌套子图内联展示功能
TEST_F(UtestReadableDump, test_NestedSubgraphInlineDisplay) {
  // 创建主图
  ut::GraphBuilder builder = ut::GraphBuilder("main_graph");
  auto main_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(main_graph);
  
  auto cond_op = op::Data("cond").set_attr_index(0);
  auto cond_node = graph.AddNodeByOp(cond_op);
  auto input_op = op::Data("input").set_attr_index(1);
  auto input_node = graph.AddNodeByOp(input_op);
  auto if_op = op::If("If_0")
                   .create_dynamic_input_byindex_input(1, 1)
                   .create_dynamic_output_output(2);
  auto if_node_gnode = graph.AddNodeByOp(if_op);
  graph.AddDataEdge(cond_node, 0, if_node_gnode, 0);
  graph.AddDataEdge(input_node, 0, if_node_gnode, 1);
  
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 2, 0);
  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);
  graph.AddDataEdge(if_node_gnode, 0, netoutput_gnode, 0);
  graph.AddDataEdge(if_node_gnode, 1, netoutput_gnode, 1);
  
  // 创建then子图（包含While节点）
  ut::GraphBuilder then_builder = ut::GraphBuilder("then_branch");
  auto then_graph = then_builder.GetGraph();
  auto then_compute_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(then_graph);
  auto then_data_op = op::Data("Data_0").set_attr_index(0);
  auto then_data_node = then_compute_graph.AddNodeByOp(then_data_op);
  auto while_op = op::While("While_0")
                      .create_dynamic_input_input(1)
                      .create_dynamic_output_output(4);
  auto while_node_gnode = then_compute_graph.AddNodeByOp(while_op);
  then_compute_graph.AddDataEdge(then_data_node, 0, while_node_gnode, 0);
  auto then_phony1_op = op::Phony1("Phony1_0").set_attr_N(1);
  auto then_phony1_gnode = then_compute_graph.AddNodeByOp(then_phony1_op);
  then_compute_graph.AddDataEdge(then_data_node, 0, then_phony1_gnode, 0);
  const auto &then_netoutput = then_builder.AddNode("netoutput", NETOUTPUT, 3, 0);
  GNode then_netoutput_gnode = NodeAdapter::Node2GNode(then_netoutput);
  then_compute_graph.AddDataEdge(while_node_gnode, 0, then_netoutput_gnode, 0);
  then_compute_graph.AddDataEdge(then_phony1_gnode, 0, then_netoutput_gnode, 1);
  then_compute_graph.AddDataEdge(while_node_gnode, 1, then_netoutput_gnode, 2);

  // 创建While的cond子图
  ut::GraphBuilder cond_builder = ut::GraphBuilder("while_01_cond");
  auto cond_graph = cond_builder.GetGraph();
  auto cond_compute_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cond_graph);
  auto cond_data_op = op::Data("Data_1").set_attr_index(0);
  auto cond_data_node = cond_compute_graph.AddNodeByOp(cond_data_op);
  auto cond_add_op = op::Add("Add_0");
  auto cond_add_node = cond_compute_graph.AddNodeByOp(cond_add_op);
  cond_compute_graph.AddDataEdge(cond_data_node, 0, cond_add_node, 0);
  cond_compute_graph.AddDataEdge(cond_data_node, 0, cond_add_node, 1);
  auto cond_phony1_op = op::Phony1("Phony1_1").set_attr_N(1);
  auto cond_phony1_node = cond_compute_graph.AddNodeByOp(cond_phony1_op);
  cond_compute_graph.AddDataEdge(cond_add_node, 0, cond_phony1_node, 0);
  const auto &cond_netoutput = cond_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode cond_netoutput_gnode = NodeAdapter::Node2GNode(cond_netoutput);
  cond_compute_graph.AddDataEdge(cond_phony1_node, 0, cond_netoutput_gnode, 0);
  
  // 创建While的body子图
  ut::GraphBuilder body_builder = ut::GraphBuilder("while_01_body");
  auto body_graph = body_builder.GetGraph();
  auto body_compute_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(body_graph);
  auto body_data_op = op::Data("Data_2").set_attr_index(0);
  auto body_data_node = body_compute_graph.AddNodeByOp(body_data_op);
  auto body_add_op = op::Add("Add_1");
  auto body_add_node = body_compute_graph.AddNodeByOp(body_add_op);
  body_compute_graph.AddDataEdge(body_data_node, 0, body_add_node, 0);
  body_compute_graph.AddDataEdge(body_data_node, 0, body_add_node, 1);
  auto body_phony1_op = op::Phony1("Phony1_2").set_attr_N(1);
  auto body_phony1_node = body_compute_graph.AddNodeByOp(body_phony1_op);
  body_compute_graph.AddDataEdge(body_data_node, 0, body_phony1_node, 0);
  const auto &body_netoutput = body_builder.AddNode("netoutput", NETOUTPUT, 3, 0);
  GNode body_netoutput_gnode = NodeAdapter::Node2GNode(body_netoutput);
  body_compute_graph.AddDataEdge(body_add_node, 0, body_netoutput_gnode, 0);
  body_compute_graph.AddDataEdge(body_phony1_node, 0, body_netoutput_gnode, 1);
  body_compute_graph.AddDataEdge(body_data_node, 0, body_netoutput_gnode, 2);

  // 将then子图添加到主图的If节点
  auto if_node = NodeAdapter::GNode2Node(if_node_gnode);
  NodeUtils::SetSubgraph(*if_node, 0, then_graph);
  // 将cond和body子图添加到While节点
  auto while_node = NodeAdapter::GNode2Node(while_node_gnode);
  NodeUtils::SetSubgraph(*while_node, 0, cond_graph);
  NodeUtils::SetSubgraph(*while_node, 1, body_graph);

  std::string readable_dump = R"(graph("main_graph"):
  %cond : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %input : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %If_0 : [#users=2] = Node[type=If] (inputs = (cond=%cond, input_0=%input), attrs = {then_branch: %then_branch})
  %ret : [#users=1] = get_element[node=%If_0](0)
  %ret_1 : [#users=1] = get_element[node=%If_0](1)

  return (output_0=%ret, output_1=%ret_1)

graph("then_branch"):
  %Data_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %While_0 : [#users=4] = Node[type=While] (inputs = (input_0=%Data_0), attrs = {cond: %while_01_cond, body: %while_01_body, parallel_iterations: 10})
  %ret : [#users=1] = get_element[node=%While_0](0)
  %ret_1 : [#users=1] = get_element[node=%While_0](1)
  %ret_2 : [#users=0] = get_element[node=%While_0](2)
  %ret_3 : [#users=0] = get_element[node=%While_0](3)
  %Phony1_0 : [#users=1] = Node[type=Phony1] (inputs = (x=%Data_0), attrs = {N: 1})

  return (output_0=%ret, output_1=%Phony1_0, output_2=%ret_1)

graph("while_01_cond"):
  %Data_1 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Add_0 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_1, x2=%Data_1))
  %Phony1_1 : [#users=1] = Node[type=Phony1] (inputs = (x=%Add_0), attrs = {N: 1})

  return (%Phony1_1)

graph("while_01_body"):
  %Data_2 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Add_1 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_2, x2=%Data_2))
  %Phony1_2 : [#users=1] = Node[type=Phony1] (inputs = (x=%Data_2), attrs = {N: 1})

  return (output_0=%Add_1, output_1=%Phony1_2, output_2=%Data_2)
)";
  
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, main_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

// 测试Case节点多个子图输入（显示 IR 参数名）
TEST_F(UtestReadableDump, test_CaseSubgraphMultipleInputsIndex) {
  // 创建主图
  auto builder = ut::GraphBuilder("main_graph");
  auto main_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(main_graph);
  
  auto branch_index_op = op::Data("branch_index").set_attr_index(0);
  auto branch_index_node = graph.AddNodeByOp(branch_index_op);
  auto input0_op = op::Data("input0").set_attr_index(1);
  auto input0_node = graph.AddNodeByOp(input0_op);
  auto input1_op = op::Data("input1").set_attr_index(2);
  auto input1_node = graph.AddNodeByOp(input1_op);
  
  auto case_op = op::Case("Case_0")
                     .create_dynamic_input_byindex_input(2, 1)
                     .create_dynamic_output_output(1)
                     .create_dynamic_subgraph_branches(3);
  auto case_node_gnode = graph.AddNodeByOp(case_op);
  graph.AddDataEdge(branch_index_node, 0, case_node_gnode, 0);
  graph.AddDataEdge(input0_node, 0, case_node_gnode, 1);
  graph.AddDataEdge(input1_node, 0, case_node_gnode, 2);
  
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);
  graph.AddDataEdge(case_node_gnode, 0, netoutput_gnode, 0);
  
  // 创建第一个branch子图（包含2个Data节点）
  auto branch0_builder = ut::GraphBuilder("branch_0");
  auto branch0_compute_graph = branch0_builder.GetGraph();
  auto branch0_graph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(branch0_compute_graph);
  
  auto branch0_data0_op = op::Data("Data_0").set_attr_index(0);
  auto branch0_data0_node = branch0_graph_obj.AddNodeByOp(branch0_data0_op);
  auto branch0_data1_op = op::Data("Data_1").set_attr_index(1);
  auto branch0_data1_node = branch0_graph_obj.AddNodeByOp(branch0_data1_op);
  
  auto branch0_add_op = op::Add("Add_0");
  auto branch0_add_node = branch0_graph_obj.AddNodeByOp(branch0_add_op);
  branch0_graph_obj.AddDataEdge(branch0_data0_node, 0, branch0_add_node, 0);
  branch0_graph_obj.AddDataEdge(branch0_data1_node, 0, branch0_add_node, 1);
  
  const auto &branch0_netoutput = branch0_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode branch0_netoutput_gnode = NodeAdapter::Node2GNode(branch0_netoutput);
  branch0_graph_obj.AddDataEdge(branch0_add_node, 0, branch0_netoutput_gnode, 0);


  // 创建第二个branch子图（包含2个Data节点）
  auto branch1_builder = ut::GraphBuilder("branch_1");
  auto branch1_compute_graph = branch1_builder.GetGraph();
  auto branch1_graph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(branch1_compute_graph);

  auto branch1_data0_op = op::Data("Data_0").set_attr_index(0);
  auto branch1_data0_node = branch1_graph_obj.AddNodeByOp(branch1_data0_op);
  auto branch1_data1_op = op::Data("Data_1").set_attr_index(1);
  auto branch1_data1_node = branch1_graph_obj.AddNodeByOp(branch1_data1_op);

  auto branch1_add_op = op::Add("Add_0");
  auto branch1_add_node = branch1_graph_obj.AddNodeByOp(branch1_add_op);
  branch1_graph_obj.AddDataEdge(branch1_data0_node, 0, branch1_add_node, 0);
  branch1_graph_obj.AddDataEdge(branch1_data1_node, 0, branch1_add_node, 1);

  const auto &branch1_netoutput = branch1_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode branch1_netoutput_gnode = NodeAdapter::Node2GNode(branch1_netoutput);
  branch1_graph_obj.AddDataEdge(branch1_add_node, 0, branch1_netoutput_gnode, 0);
  
  // 将子图添加到主图的Case节点
  NodeUtils::SetSubgraph(*NodeAdapter::GNode2Node(case_node_gnode), 0, branch0_compute_graph);
  NodeUtils::SetSubgraph(*NodeAdapter::GNode2Node(case_node_gnode), 1, branch1_compute_graph);

  std::string readable_dump = R"(graph("main_graph"):
  %branch_index : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %input0 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %input1 : [#users=1] = Node[type=Data] (attrs = {index: 2})
  %Case_0 : [#users=1] = Node[type=Case] (inputs = (branch_index=%branch_index, input_0=%input0, input_1=%input1), attrs = {branches0: %branch_0, branches1: %branch_1})

  return (%Case_0)

graph("branch_0"):
  %Data_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Data_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %Add_0 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_0, x2=%Data_1))

  return (%Add_0)

graph("branch_1"):
  %Data_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Data_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %Add_0 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_0, x2=%Data_1))

  return (%Add_0)
)";
  
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, main_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

// 测试 For 节点多个子图输入（显示 IR 参数名，动态输入 index 从1 开始）
TEST_F(UtestReadableDump, test_ForSubgraphMultipleInputsIndex) {
  // 创建主图
  auto builder = ut::GraphBuilder("main_graph");
  auto main_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(main_graph);
  
  // 创建 For 的三个固定输入：start, limit, delta
  auto ge_tensor_start = std::make_shared<GeTensor>();
  int32_t start_data = 0;
  ge_tensor_start->SetData(reinterpret_cast<uint8_t *>(&start_data), sizeof(int32_t));
  ge_tensor_start->MutableTensorDesc().SetDataType(DT_INT32);
  auto start_op = op::Const("start").set_attr_value(ge::TensorAdapter::AsTensor(*ge_tensor_start));
  auto start_node = graph.AddNodeByOp(start_op);
  
  auto ge_tensor_limit = std::make_shared<GeTensor>();
  int32_t limit_data = 5;
  ge_tensor_limit->SetData(reinterpret_cast<uint8_t *>(&limit_data), sizeof(int32_t));
  ge_tensor_limit->MutableTensorDesc().SetDataType(DT_INT32);
  auto limit_op = op::Const("limit").set_attr_value(ge::TensorAdapter::AsTensor(*ge_tensor_limit));
  auto limit_node = graph.AddNodeByOp(limit_op);
  
  auto ge_tensor_delta = std::make_shared<GeTensor>();
  int32_t delta_data = 1;
  ge_tensor_delta->SetData(reinterpret_cast<uint8_t *>(&delta_data), sizeof(int32_t));
  ge_tensor_delta->MutableTensorDesc().SetDataType(DT_INT32);
  auto delta_op = op::Const("delta").set_attr_value(ge::TensorAdapter::AsTensor(*ge_tensor_delta));
  auto delta_node = graph.AddNodeByOp(delta_op);
  
  // 创建 For 的动态输入
  auto input0_op = op::Data("input0").set_attr_index(0);
  auto input0_node = graph.AddNodeByOp(input0_op);
  auto input1_op = op::Data("input1").set_attr_index(1);
  auto input1_node = graph.AddNodeByOp(input1_op);
  
  auto for_op = op::For("For_0")
                    .create_dynamic_input_byindex_input(2, 3)
                    .create_dynamic_output_output(2);
  auto for_node_gnode = graph.AddNodeByOp(for_op);
  graph.AddDataEdge(start_node, 0, for_node_gnode, 0);
  graph.AddDataEdge(limit_node, 0, for_node_gnode, 1);
  graph.AddDataEdge(delta_node, 0, for_node_gnode, 2);
  graph.AddDataEdge(input0_node, 0, for_node_gnode, 3);
  graph.AddDataEdge(input1_node, 0, for_node_gnode, 4);
  
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);
  graph.AddDataEdge(for_node_gnode, 0, netoutput_gnode, 0);
  
  // 创建body子图（包含2个Data节点：Data_0 对应 input0，Data_1 对应 input1）
  auto body_builder = ut::GraphBuilder("body");
  auto body_compute_graph = body_builder.GetGraph();
  auto body_graph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(body_compute_graph);
  
  auto body_data0_op = op::Data("Data_0").set_attr_index(0);
  auto body_data0_node = body_graph_obj.AddNodeByOp(body_data0_op);
  auto body_data1_op = op::Data("Data_1").set_attr_index(1);
  auto body_data1_node = body_graph_obj.AddNodeByOp(body_data1_op);
  
  auto body_add_op = op::Add("Add_0");
  auto body_add_node = body_graph_obj.AddNodeByOp(body_add_op);
  body_graph_obj.AddDataEdge(body_data0_node, 0, body_add_node, 0);
  body_graph_obj.AddDataEdge(body_data1_node, 0, body_add_node, 1);
  
  const auto &body_netoutput = body_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode body_netoutput_gnode = NodeAdapter::Node2GNode(body_netoutput);
  body_graph_obj.AddDataEdge(body_add_node, 0, body_netoutput_gnode, 0);
  
  // 将子图添加到主图的For节点
  NodeUtils::SetSubgraph(*NodeAdapter::GNode2Node(for_node_gnode), 0, body_compute_graph);
  
  std::string readable_dump = R"(graph("main_graph"):
  %start : [#users=1] = Node[type=Const] (attrs = {value: [0]})
  %limit : [#users=1] = Node[type=Const] (attrs = {value: [5]})
  %delta : [#users=1] = Node[type=Const] (attrs = {value: [1]})
  %input0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %input1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %For_0 : [#users=2] = Node[type=For] (inputs = (start=%start, limit=%limit, delta=%delta, input_1=%input0, input_2=%input1), attrs = {body: %body})
  %ret : [#users=1] = get_element[node=%For_0](0)
  %ret_1 : [#users=0] = get_element[node=%For_0](1)

  return (%ret)

graph("body"):
  %Data_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Data_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %Add_0 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_0, x2=%Data_1))

  return (%Add_0)
)";

  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, main_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

// 测试子图嵌套深度不受限
TEST_F(UtestReadableDump, test_SubgraphDepthMoreThan10) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr current_graph;
  NodePtr current_node;
  
  // 创建根图
  ut::GraphBuilder root_builder = ut::GraphBuilder("graph0");
  root_graph = root_builder.GetGraph();
  auto graph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  
  // 在根图中创建一个 If 节点，用于引用第一个子图
  auto root_if_op = op::If("If_0")
                        .create_dynamic_input_byindex_input(1, 1)
                        .create_dynamic_output_output(1);
  auto root_if_node_gnode = graph_obj.AddNodeByOp(root_if_op);
  current_node = NodeAdapter::GNode2Node(root_if_node_gnode);
  
  // 创建根图的输入和输出节点
  auto root_data_op = op::Data("data0").set_attr_index(0);
  auto root_data_node_gnode = graph_obj.AddNodeByOp(root_data_op);
  graph_obj.AddDataEdge(root_data_node_gnode, 0, root_if_node_gnode, 1);
  const auto &root_netoutput = root_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode root_netoutput_gnode = NodeAdapter::Node2GNode(root_netoutput);
  graph_obj.AddDataEdge(root_if_node_gnode, 0, root_netoutput_gnode, 0);
  
  current_graph = root_graph;
  
  // 创建深度嵌套的子图链（深度为 12）
  for (int i = 1; i <= 12; ++i) {
    std::string subgraph_name = "graph" + std::to_string(i);
    ut::GraphBuilder sub_builder = ut::GraphBuilder(subgraph_name);
    auto sub_compute_graph = sub_builder.GetGraph();
    auto sub_graph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(sub_compute_graph);
    
    // 在子图中创建一个 If 节点
    auto if_op = op::If("If_" + std::to_string(i))
                     .create_dynamic_input_byindex_input(1, 1)
                     .create_dynamic_output_output(1);
    auto if_node_gnode = sub_graph_obj.AddNodeByOp(if_op);
    auto if_node = NodeAdapter::GNode2Node(if_node_gnode);
    
    // 创建子图的输入节点
    auto sub_data_op = op::Data("Data_" + std::to_string(i)).set_attr_index(0);
    auto sub_data_node_gnode = sub_graph_obj.AddNodeByOp(sub_data_op);
    sub_graph_obj.AddDataEdge(sub_data_node_gnode, 0, if_node_gnode, 1);
    
    // 创建子图的输出节点
    const auto &sub_netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
    GNode sub_netoutput_gnode = NodeAdapter::Node2GNode(sub_netoutput);
    sub_graph_obj.AddDataEdge(if_node_gnode, 0, sub_netoutput_gnode, 0);
    
    // 设置子图的父子关系
    sub_compute_graph->SetParentGraph(current_graph);
    sub_compute_graph->SetParentNode(current_node);
    current_graph->AddSubgraph(subgraph_name, sub_compute_graph);
    
    // 设置当前节点的子图引用
    NodeUtils::SetSubgraph(*current_node, 0, sub_compute_graph);
    
    // 更新为下一个子图
    current_graph = sub_compute_graph;
    current_node = if_node;
  }

  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, root_graph));
}

// 测试 IR 信息异常时使用位置参数格式
TEST_F(UtestReadableDump, test_DumpNodeWithoutIrInfo) {
  ut::GraphBuilder builder = ut::GraphBuilder("main_graph");
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // 创建输入节点
  auto input0_op = op::Data("input0").set_attr_index(0);
  auto input0_node = graph.AddNodeByOp(input0_op);
  auto input1_op = op::Data("input1").set_attr_index(1);
  auto input1_node = graph.AddNodeByOp(input1_op);

  // 使用 GraphBuilder::AddNode 直接创建，不恢复 IR 定义
  auto node_without_ir = builder.AddNode("PartitionedCall_0", "PartitionedCall", 2, 1);

  // 将节点添加到图中并连接输入
  auto partitioned_call_node = NodeAdapter::Node2GNode(node_without_ir);
  graph.AddDataEdge(input0_node, 0, partitioned_call_node, 0);
  graph.AddDataEdge(input1_node, 0, partitioned_call_node, 1);

  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);
  graph.AddDataEdge(partitioned_call_node, 0, netoutput_gnode, 0);

  // 创建子图
  std::string subgraph_name = "Conv2D_17function_graph_1";
  ut::GraphBuilder subgraph_builder = ut::GraphBuilder(subgraph_name);
  auto subgraph_compute_graph = subgraph_builder.GetGraph();
  auto subgraph_obj = ge::GraphUtilsEx::CreateGraphFromComputeGraph(subgraph_compute_graph);

  auto subgraph_data0_op = op::Data("Data_0").set_attr_index(0);
  auto subgraph_data0_node = subgraph_obj.AddNodeByOp(subgraph_data0_op);
  auto subgraph_data1_op = op::Data("Data_1").set_attr_index(1);
  auto subgraph_data1_node = subgraph_obj.AddNodeByOp(subgraph_data1_op);

  auto subgraph_add_op = op::Add("Add_0");
  auto subgraph_add_node = subgraph_obj.AddNodeByOp(subgraph_add_op);
  subgraph_obj.AddDataEdge(subgraph_data0_node, 0, subgraph_add_node, 0);
  subgraph_obj.AddDataEdge(subgraph_data1_node, 0, subgraph_add_node, 1);

  const auto &subgraph_netoutput = subgraph_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  GNode subgraph_netoutput_gnode = NodeAdapter::Node2GNode(subgraph_netoutput);
  subgraph_obj.AddDataEdge(subgraph_add_node, 0, subgraph_netoutput_gnode, 0);

  // 使用子图实际名称作为 IR 名称
  auto partitioned_call_node_ptr = NodeAdapter::GNode2Node(partitioned_call_node);
  auto op_desc = partitioned_call_node_ptr->GetOpDesc();
  subgraph_compute_graph->SetParentGraph(compute_graph);
  subgraph_compute_graph->SetParentNode(partitioned_call_node_ptr);
  (void)op_desc->AddSubgraphName(subgraph_name);
  (void)op_desc->SetSubgraphInstanceName(0, subgraph_name);
  compute_graph->AddSubgraph(subgraph_compute_graph->GetName(), subgraph_compute_graph);

  std::string readable_dump = R"(graph("main_graph"):
  %input0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %input1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %PartitionedCall_0 : [#users=1] = Node[type=PartitionedCall] (inputs = (_input_0=%input0, _input_1=%input1), attrs = {_graph_0: %Conv2D_17function_graph_1})

  return (%PartitionedCall_0)

graph("Conv2D_17function_graph_1"):
  %Data_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Data_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %Add_0 : [#users=1] = Node[type=Add] (inputs = (x1=%Data_0, x2=%Data_1))

  return (%Add_0)
)";

  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}
