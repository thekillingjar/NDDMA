/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/buffer.h"
#include "graph/attr_store.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
constexpr size_t kSmallBufferSize = 32UL;
constexpr size_t kLargeBufferSize = 536870912U;
class SubModel : public Model
{
public:

  SubModel();
  SubModel(const std::string &name, const std::string &custom_version);

  virtual ~SubModel();

};

SubModel::SubModel(){}
SubModel::SubModel(const std::string &name, const std::string &custom_version):Model(name,custom_version){}

SubModel::~SubModel() = default;

}

static Model BuildModelWithConst(bool large_weight) {
  Model model("model_name/main_model", "custom version3.0");
  auto compute_graph = std::make_shared<ComputeGraph>("graph_name/main_graph");
  size_t buffer_size = large_weight ? (kLargeBufferSize) : kSmallBufferSize;
  // input
  for (int i = 0; i < 4; i++) {
    std::string inpu_node_name = "test/const" + std::to_string(i);
    auto input_op = std::make_shared<OpDesc>(inpu_node_name, "Const");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = compute_graph->AddNode(input_op);
    GeTensor ge_tensor;
    auto aligned_ptr = std::make_shared<AlignedPtr>(buffer_size);  // 500m
    auto ptr = aligned_ptr->MutableGet();
    *ptr = 7;
    *(ptr + 10) = 8;
    *(ptr + buffer_size - 2) = 9;
    ge_tensor.SetData(aligned_ptr, buffer_size);
    AttrUtils::SetTensor(input_op, ATTR_NAME_WEIGHTS, ge_tensor);
  }
  model.SetGraph(compute_graph);
  auto sub_compute_graph = std::make_shared<ComputeGraph>("sub_graph");
  auto sub_graph_input_op = std::make_shared<OpDesc>("sub_graph_test", "TestOp2");
  sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto sub_graph_input = sub_compute_graph->AddNode(sub_graph_input_op);

  auto parent_input_op = std::make_shared<OpDesc>("parenttest", "TestOp2");
  parent_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto parent_input = compute_graph->AddNode(parent_input_op);
  for (int i = 0; i < 4; i++) {
    std::string inpu_node_name = "subgraph/const" + std::to_string(i);
    auto sub_input_op = std::make_shared<OpDesc>(inpu_node_name, "Const");
    sub_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_input = sub_compute_graph->AddNode(sub_input_op);
    GeTensor ge_tensor;
    auto aligned_ptr = std::make_shared<AlignedPtr>(buffer_size);  // 500m
    auto ptr = aligned_ptr->MutableGet();
    *ptr = 7;
    *(ptr + 10) = 8;
    *(ptr + buffer_size - 2) = 9;
    ge_tensor.SetData(aligned_ptr, buffer_size);
    AttrUtils::SetTensor(sub_input_op, ATTR_NAME_WEIGHTS, ge_tensor);
  }
  std::string sub_graph = "sub_graph";
  parent_input_op->AddSubgraphName(sub_graph);
  parent_input_op->SetSubgraphInstanceName(0, sub_graph);
  sub_compute_graph->SetParentNode(parent_input);
  sub_compute_graph->SetParentGraph(compute_graph);
  compute_graph->AddSubgraph(sub_compute_graph);
  compute_graph->TopologicalSorting();
  return model;
}

static Model BuildModelWithLargeConst() {
  return BuildModelWithConst(true);
}

static Graph BuildGraph() {
  ge::OpDescPtr add_op(new ge::OpDesc("add1", "Add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  return graph;
}

class ModelUt : public testing::Test {};
TEST_F(ModelUt, SetGet) {
  auto md = SubModel();
  auto md2 = SubModel("md2", "test");
  EXPECT_EQ(md.GetName(),"");
  md.SetName("tt");
  EXPECT_EQ(md.GetName(),"tt");
  EXPECT_EQ(md2.GetName(),"md2");
  md2.SetName("md2tt");
  EXPECT_EQ(md2.GetName(),"md2tt");
  EXPECT_EQ(md.GetVersion(),0);
  EXPECT_EQ(md2.GetVersion(),0);
  EXPECT_EQ(md2.GetPlatformVersion(),"test");

  auto graph = BuildGraph();
  EXPECT_EQ(graph.IsValid(),true);
  md2.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  auto g = md2.GetGraph();
  EXPECT_NE(&g, nullptr);
  Buffer buf = Buffer(1024);
  EXPECT_EQ(buf.GetSize(),1024);
  EXPECT_EQ(md2.IsValid(),true);
  ProtoAttrMap attr = AttrStore::Create(512);
  AttrId id = 1;
  int val = 100;
  attr.Set<int>(id, val);
  const int* v = attr.Get<int>(id);
  EXPECT_EQ(*v,val);
  md2.SetAttr(attr);
  EXPECT_EQ(md2.Save(buf,true), GRAPH_SUCCESS);
}

TEST_F(ModelUt, Load) {
  auto md = SubModel("md2", "test");
  auto graph = BuildGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  uint8_t b[5];
  memset(b,1,5);
  EXPECT_EQ(md.Load((const uint8_t*)b, 5, md),GRAPH_FAILED);

  std::string msg = "package lm;\nmessage helloworld{\nrequired int32     id = 1;\nrequired string    str = 2;\noptional int32     opt = 3;}";
  std::ofstream outfile;
  outfile.open("./test_load.proto");
  outfile << msg;
  EXPECT_EQ(md.LoadFromFile("./test_load.proto"), GRAPH_SUCCESS);
  system("rm -rf ./test_load.proto");
  outfile.close();

}

TEST_F(ModelUt, Save) {
  EXPECT_NO_THROW(
    auto md = SubModel("md2", "test");
    auto graph = BuildGraph();
    md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
    std::stringstream ss;
    ss << "./test_save.proto";
    md.SaveToFile(ss.str());
    std::string cmd = "rm -rf " + ss.str();
    system(cmd.c_str());
  );
}

TEST_F(ModelUt, Save_Failure) {
  auto md = SubModel("md2", "test");
  auto graph = BuildGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  std::stringstream fn;
  fn << "/tmp/";
  for (int i = 0; i < 4096; i++){
    fn << "a";
  }
  fn << ".proto";
  md.SaveToFile(fn.str());
  EXPECT_EQ(md.SaveToFile("/proc/non.proto"), GRAPH_FAILED);
}

TEST_F(ModelUt, Load_Longname) {
  auto md = SubModel("md2", "test");
  std::stringstream fn;
  fn << "/tmp/";
  for (int i = 0; i < 4096; i++){
    fn << "a";
  }
  fn << ".proto";
  EXPECT_EQ(md.LoadFromFile(fn.str()),GRAPH_FAILED);
}

TEST_F(ModelUt, Load_Nonfilename) {
  auto md = SubModel("md2", "test");
  EXPECT_EQ(md.LoadFromFile("/tmp/non-exsit"),GRAPH_FAILED);
}

TEST_F(ModelUt, SaveLargeModelWithoutSeparate) {
  auto md = BuildModelWithLargeConst();
  Buffer buf = Buffer(1024);
  EXPECT_EQ(buf.GetSize(),1024);
  EXPECT_EQ(md.IsValid(),true);
  EXPECT_EQ(md.SaveWithoutSeparate(buf), GRAPH_FAILED);
}

TEST_F(ModelUt, SaveLargeModelWithRealPath) {
  auto md = BuildModelWithLargeConst();
  std::string tmp_file_name = "./";

  std::string real_path = ge::RealPath(tmp_file_name.c_str()) + "/model.air";
  std::string clear_cmd = "rm -rf " + real_path;
  system(clear_cmd.c_str());
  EXPECT_EQ(md.SaveToFile(real_path), GRAPH_SUCCESS);
  Model model_back(nullptr, nullptr);
  EXPECT_EQ(model_back.LoadFromFile(real_path), GRAPH_SUCCESS);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 10);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == "Const") {
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
      ASSERT_EQ(buff[kLargeBufferSize - 2], 9); // value is ok for def serialize
    }
  }
  auto sub_graph = com_graph1->GetSubgraph("sub_graph");
  ASSERT_EQ((sub_graph == nullptr), false);
  ASSERT_EQ(sub_graph->GetAllNodesSize(), 5);
  system("rm -rf /tmp/test/air_weight");
  system(clear_cmd.c_str());
}

TEST_F(ModelUt, SaveLargeModelWithRelatedPath) {
  auto md = BuildModelWithLargeConst();
  std::string file_name = "./temp/model.air";
  std::string clear_cmd = "rm -rf " + file_name;
  system(clear_cmd.c_str());
  EXPECT_EQ(md.SaveToFile(file_name), GRAPH_SUCCESS);
  Model model_back;
  EXPECT_EQ(model_back.LoadFromFile(file_name), GRAPH_SUCCESS);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 10);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == "Const") {
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
      ASSERT_EQ(buff[kLargeBufferSize - 2], 9); // value is ok for def serialize
    }
  }
  auto sub_graph = com_graph1->GetSubgraph("sub_graph");
  ASSERT_EQ((sub_graph == nullptr), false);
  ASSERT_EQ(sub_graph->GetAllNodesSize(), 5);
  system("rm -rf ./temp/air_weight");
  system(clear_cmd.c_str());
}

TEST_F(ModelUt, SaveLargeModelSeparateWithRelatedPath) {
  auto md = BuildModelWithConst(false);
  auto graph = md.GetGraph();
  // const node0 reuse const node1 weight
  auto const_node0 = graph->FindNode("test/const0");
  ASSERT_NE(const_node0, nullptr);
  auto const_node1 = graph->FindNode("test/const1");
  ASSERT_NE(const_node1, nullptr);
  ASSERT_TRUE(AttrUtils::SetBool(const_node0->GetOpDesc(), ATTR_NAME_IS_REUSE_EXTERNAL_WEIGHT, true));
  std::string op_tag = const_node1->GetType() + "_" + graph->GetName() + "_" + const_node1->GetName() + "_file";
  std::string regulated_op_tag = ge::GetRegulatedName(op_tag);
  std::string reuse_offset_path = "air_weight/model_name_main_model/" + regulated_op_tag;
  ASSERT_EQ(AttrUtils::SetInt(const_node0->GetOpDesc(), ge::ATTR_NAME_LENGTH, kSmallBufferSize), true);
  ASSERT_EQ(AttrUtils::SetStr(const_node0->GetOpDesc(), ge::ATTR_NAME_LOCATION, reuse_offset_path), true);
  std::string file_name = "./temp/model.air";
  system("rm -rf ./temp/model.air");
  EXPECT_EQ(md.SaveToFile(file_name, true), GRAPH_SUCCESS);
  Model model_back;
  EXPECT_EQ(model_back.LoadFromFile("./temp/model.air"), GRAPH_SUCCESS);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 10);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == "Const") {
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
      ASSERT_EQ(buff[kSmallBufferSize - 2], 9); // value is ok for def serialize
    }
  }
  auto sub_graph = com_graph1->GetSubgraph("sub_graph");
  ASSERT_EQ((sub_graph == nullptr), false);
  ASSERT_EQ(sub_graph->GetAllNodesSize(), 5);
  system("rm -rf ./temp/air_weight");
  system("rm -rf ./temp/model.air");
}

TEST_F(ModelUt, SaveLargeModelWithRelatedPath2) {
  auto md = BuildModelWithLargeConst();
  std::string tmp_file_name = "./";

  std::string real_path = ge::RealPath(tmp_file_name.c_str()) + "/model.air";
  std::string clear_cmd = "rm -rf " + real_path;
  system(clear_cmd.c_str());
  EXPECT_EQ(md.SaveToFile(real_path), GRAPH_SUCCESS);
  Model model_back;
  EXPECT_EQ(model_back.LoadFromFile(real_path), GRAPH_SUCCESS);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 10);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == "Const") {
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
      ASSERT_EQ(buff[kLargeBufferSize - 2], 9); // value is ok for def serialize
    }
  }
  auto sub_graph = com_graph1->GetSubgraph("sub_graph");
  ASSERT_EQ((sub_graph == nullptr), false);
  ASSERT_EQ(sub_graph->GetAllNodesSize(), 5);
  system("rm -rf ./air_weight");
  system(clear_cmd.c_str());
}

TEST_F(ModelUt, SaveLargeModelWithRelatedPath3) {
  auto md = BuildModelWithLargeConst();
  std::string file_name = "model.air";
  std::string clear_cmd = "rm -rf " + file_name;
  system(clear_cmd.c_str());
  EXPECT_EQ(md.SaveToFile(file_name), GRAPH_SUCCESS);
  Model model_back;
  EXPECT_EQ(model_back.LoadFromFile("model.air"), GRAPH_SUCCESS);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 10);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == "Const") {
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
      ASSERT_EQ(buff[kLargeBufferSize - 2], 9); // value is ok for def serialize
    }
  }
  auto sub_graph = com_graph1->GetSubgraph("sub_graph");
  ASSERT_EQ((sub_graph == nullptr), false);
  ASSERT_EQ(sub_graph->GetAllNodesSize(), 5);
  system("rm -rf ./air_weight");
  system(clear_cmd.c_str());
}

TEST_F(ModelUt, LoadLargeModelWithWrongWeight) {
  auto md = BuildModelWithLargeConst();
  auto graph = md.GetGraph();
  graph->SetName("graph");
  std::string file_name = "./model.air";
  std::string clear_cmd = "rm -rf " + file_name;
  system(clear_cmd.c_str());
  EXPECT_EQ(md.SaveToFile(file_name), GRAPH_SUCCESS);
  std::string weight_path = "./air_weight/model_name_main_model/Const_graph_test_const0_file";
  char real_path[1280];
  EXPECT_NE(realpath(weight_path.c_str(), real_path), nullptr);
  std::ofstream ofs(real_path, std::ios::out | std::ofstream::app);
  const char_t *data = "a";
  if (ofs.is_open()) {
    ofs << data << std::endl;
    ofs.close();
  }

  Model model_back;
  EXPECT_NE(model_back.LoadFromFile("./model.air"), GRAPH_SUCCESS);
  system("rm -rf ./air_weight");
  system(clear_cmd.c_str());
}

TEST_F(ModelUt, SaveModelWithAscendWorkPath) {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  mmSetEnv("ASCEND_WORK_PATH", current_path, 1);
  auto md = BuildModelWithLargeConst();
  std::string file_name = "model.air";
  std::string clear_cmd = "rm -rf " + file_name;
  system(clear_cmd.c_str());
  EXPECT_EQ(md.SaveToFile(file_name), GRAPH_SUCCESS);
  Model model_back;
  std::string file_path = current_path;
  file_path += "/" + file_name;
  EXPECT_EQ(model_back.LoadFromFile(file_path), GRAPH_SUCCESS);
  unsetenv("ASCEND_WORK_PATH");
  system(clear_cmd.c_str());
}
}  // namespace ge
