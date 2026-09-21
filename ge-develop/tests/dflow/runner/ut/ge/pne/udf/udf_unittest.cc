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
#include <thread>
#include "graph/testcase/ge_graph/graph_builder_utils.h"
#include "macro_utils/dt_public_scope.h"
#include "dflow/compiler/pne/udf/udf_attr_utils.h"
#include "dflow/compiler/pne/udf/udf_model.h"
#include "dflow/compiler/pne/udf/udf_model_builder.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "macro_utils/dt_public_unscope.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "dflow/compiler/data_flow_graph/function_compile.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
static ComputeGraphPtr BuildComputeGraph(const std::string &path) {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto flow_func = builder.AddNode("flow_func", "FlowFunc", 2, 1);
  if (flow_func == nullptr) {
    return nullptr;
  }
  auto op_desc = flow_func->GetOpDesc();
  if (op_desc == nullptr) {
    return nullptr;
  }
  AnyValue bin_path = AnyValue::CreateFrom(path);
  auto ret = op_desc->SetAttr("bin_path", bin_path);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }
  std::string name = "func_name";
  AnyValue func_name = AnyValue::CreateFrom(name);
  ret = op_desc->SetAttr("func_name", func_name);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  std::vector<std::vector<int64_t>> shapes = {{1, 2, 3}};
  AnyValue output_shapes = AnyValue::CreateFrom(shapes);
  ret = op_desc->SetAttr("output_shapes", output_shapes);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  std::vector<DataType> types = {DT_FLOAT};
  AnyValue output_types = AnyValue::CreateFrom(types);
  ret = op_desc->SetAttr("output_types", output_types);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  std::string s = "";
  AnyValue s_attr = AnyValue::CreateFrom(s);
  ret = op_desc->SetAttr("s_attr", s_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  int64_t i = 0;
  AnyValue i_attr = AnyValue::CreateFrom(i);
  ret = op_desc->SetAttr("i_attr", i_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  float f = 0;
  AnyValue f_attr = AnyValue::CreateFrom(f);
  ret = op_desc->SetAttr("f_attr", f_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  bool b = false;
  AnyValue b_attr = AnyValue::CreateFrom(b);
  ret = op_desc->SetAttr("b_attr", b_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  DataType data_type = DataType::DT_FLOAT;
  AnyValue data_type_attr = AnyValue::CreateFrom(data_type);
  ret = op_desc->SetAttr("data_type_attr", data_type_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  std::vector<int64_t> list_i = {0};
  AnyValue list_i_attr = AnyValue::CreateFrom(list_i);
  ret = op_desc->SetAttr("list_i_attr", list_i_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  std::vector<std::string> list_s = {""};
  AnyValue list_s_attr = AnyValue::CreateFrom(list_s);
  ret = op_desc->SetAttr("list_s_attr", list_s_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  std::vector<float> list_f = {0};
  AnyValue list_f_attr = AnyValue::CreateFrom(list_f);
  ret = op_desc->SetAttr("list_f_attr", list_f_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  std::vector<bool> list_b;
  AnyValue list_b_attr = AnyValue::CreateFrom(list_b);
  ret = op_desc->SetAttr("list_b_attr", list_b_attr);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  ge::NamedAttrs compile_results;
  ge::NamedAttrs current_compile_result;
  std::vector<std::string> runnable_resources_info;
  runnable_resources_info.reserve(2);
  runnable_resources_info.emplace_back("X86");
  runnable_resources_info.emplace_back("Aarch64");
  ge::AttrUtils::SetListStr(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
  compile_results.SetAttr("pp0", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
  (void)ge::AttrUtils::SetNamedAttrs(op_desc, "_dflow_compiler_result", compile_results);
  (void)AttrUtils::SetStr(op_desc, "_dflow_process_point_release_pkg", "./temp/release");

  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, flow_func, 0);
  builder.AddDataEdge(data2, 0, flow_func, 1);
  builder.AddDataEdge(flow_func, 0, netoutput, 0);

  return builder.GetGraph();
}

static ComputeGraphPtr BuildBuiltInUdfComputeGraph() {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto builtin_flow_func = builder.AddNode("builtin_flow_func", "FlowFunc", 1, 1);
  if (builtin_flow_func == nullptr) {
    return nullptr;
  }
  auto builtin_op_desc = builtin_flow_func->GetOpDesc();
  if (builtin_op_desc == nullptr) {
    return nullptr;
  }

  std::string builtin_name = "_BuiltIn_func";
  AnyValue builtin_func_name = AnyValue::CreateFrom(builtin_name);
  (void)builtin_op_desc->SetAttr("func_name", builtin_func_name);

  ge::NamedAttrs compile_results;
  ge::NamedAttrs current_compile_result;
  std::vector<std::string> runnable_resources_info;
  runnable_resources_info.reserve(2);
  runnable_resources_info.emplace_back("X86");
  runnable_resources_info.emplace_back("Aarch64");
  ge::AttrUtils::SetListStr(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
  compile_results.SetAttr("pp0", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
  (void)ge::AttrUtils::SetNamedAttrs(builtin_op_desc, "_dflow_compiler_result", compile_results);
  (void)AttrUtils::SetStr(builtin_op_desc, "_dflow_process_point_release_pkg", "");

  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, builtin_flow_func, 0);
  builder.AddDataEdge(builtin_flow_func, 0, netoutput, 0);

  return builder.GetGraph();
}

static ComputeGraphPtr BuildBuiltInUdfComputeGraphWithMapInput() {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto builtin_flow_func = builder.AddNode("builtin_flow_func", "FlowFunc", 1, 1);
  if (builtin_flow_func == nullptr) {
    return nullptr;
  }
  auto builtin_op_desc = builtin_flow_func->GetOpDesc();
  if (builtin_op_desc == nullptr) {
    return nullptr;
  }

  std::string builtin_name = "_BuiltIn_func";
  AnyValue builtin_func_name = AnyValue::CreateFrom(builtin_name);
  (void)builtin_op_desc->SetAttr("func_name", builtin_func_name);

  ge::NamedAttrs compile_results;
  ge::NamedAttrs current_compile_result;
  std::vector<std::string> runnable_resources_info;
  runnable_resources_info.reserve(2);
  runnable_resources_info.emplace_back("X86");
  runnable_resources_info.emplace_back("Aarch64");
  ge::AttrUtils::SetListStr(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
  compile_results.SetAttr("pp0", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
  (void)ge::AttrUtils::SetNamedAttrs(builtin_op_desc, "_dflow_compiler_result", compile_results);
  (void)AttrUtils::SetStr(builtin_op_desc, "_dflow_process_point_release_pkg", "");

  std::vector<NamedAttrs> funcs_attr(2);
  funcs_attr[0].SetName("_BuiltIn_func0");
  std::vector<int64_t> input_map0 = {0};
  bool stream_input = true;
  funcs_attr[0].SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX, AnyValue::CreateFrom(input_map0));
  funcs_attr[0].SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_STREAM_INPUT, AnyValue::CreateFrom(stream_input));
  funcs_attr[1].SetName("_BuiltIn_func1");
  std::vector<int64_t> input_map1 = {1};
  funcs_attr[1].SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX, AnyValue::CreateFrom(input_map1));
  AttrUtils::SetListNamedAttrs(builtin_op_desc, dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST, funcs_attr);

  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, builtin_flow_func, 0);
  builder.AddDataEdge(builtin_flow_func, 0, netoutput, 0);

  return builder.GetGraph();
}

class UdfProcessNodeEngineTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string cmd =
        "mkdir -p temp/release; cd temp; mkdir build; touch build/compile_lock; cd release; touch libtest.so; tar "
        "-czvf release.tar.gz "
        "libtest.so";
    (void)system(cmd.c_str());
  }
  void TearDown() override {
    std::string cmd = "rm -rf temp";
    (void)system(cmd.c_str());
  }
};

TEST_F(UdfProcessNodeEngineTest, InitAndFinalize) {
  UdfProcessNodeEngine udf_engine;
  std::map<std::string, std::string> options;
  EXPECT_EQ(udf_engine.Initialize(options), SUCCESS);
  EXPECT_EQ(udf_engine.GetEngineName(), PNE_ID_UDF);
  EXPECT_EQ(udf_engine.Finalize(), SUCCESS);
}

TEST_F(UdfProcessNodeEngineTest, BuildGraph) {
  UdfProcessNodeEngine udf_engine;
  std::map<std::string, std::string> options;
  EXPECT_EQ(udf_engine.Initialize(options), SUCCESS);
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  PneModelPtr model = nullptr;
  uint32_t graph_id = 0;
  std::vector<GeTensor> inputs;
  EXPECT_EQ(udf_engine.BuildGraph(graph_id, compute_graph, options, inputs, model), SUCCESS);
  EXPECT_EQ(udf_engine.Finalize(), SUCCESS);
  (void)remove(path.c_str());
}

class UdfModelTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string cmd =
        "mkdir -p temp/release; cd temp; mkdir build; touch build/compile_lock; cd release; touch libtest.so; tar "
        "-czvf release.tar.gz "
        "libtest.so";
    (void)system(cmd.c_str());
  }
  void TearDown() override {
    std::string cmd = "rm -rf temp";
    (void)system(cmd.c_str());
  }
};

TEST_F(UdfModelTest, SerializeAndUnSerializeBuiltInModel) {
  UdfProcessNodeEngine udf_engine;
  std::map<std::string, std::string> options;
  EXPECT_EQ(udf_engine.Initialize(options), SUCCESS);
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  PneModelPtr model = nullptr;
  uint32_t graph_id = 0;
  std::vector<GeTensor> inputs;
  EXPECT_EQ(udf_engine.BuildGraph(graph_id, compute_graph, options, inputs, model), SUCCESS);
  if (model != nullptr) {
    model->SetIsBuiltinModel(true);
    ModelBufferData buffer;
    EXPECT_EQ(model->SerializeModel(buffer), SUCCESS);
    EXPECT_EQ(model->UnSerializeModel(buffer), SUCCESS);
  }
  EXPECT_EQ(udf_engine.Finalize(), SUCCESS);
  (void)remove(path.c_str());
}

TEST_F(UdfModelTest, SerializeAndUnSerializeUserDefineModel) {
  UdfProcessNodeEngine udf_engine;
  std::map<std::string, std::string> options;
  EXPECT_EQ(udf_engine.Initialize(options), SUCCESS);
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  PneModelPtr model = nullptr;
  uint32_t graph_id = 0;
  std::vector<GeTensor> inputs;
  EXPECT_EQ(udf_engine.BuildGraph(graph_id, compute_graph, options, inputs, model), SUCCESS);
  if (model != nullptr) {
    model->SetIsBuiltinModel(false);
    model->SetSavedModelPath(path);
    ModelBufferData buffer;
    EXPECT_EQ(model->SerializeModel(buffer), SUCCESS);
    EXPECT_EQ(model->UnSerializeModel(buffer), FAILED);
  }
  EXPECT_EQ(udf_engine.Finalize(), SUCCESS);
  (void)remove(path.c_str());
}

class UdfModelBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string cmd =
        "mkdir -p temp/release; cd temp; mkdir build; cd release; touch libtest.so; tar "
        "-czvf release.tar.gz "
        "libtest.so";
    (void)system(cmd.c_str());
  }
  void TearDown() override {
    std::string cmd = "rm -rf temp";
    (void)system(cmd.c_str());
  }
};

TEST_F(UdfModelBuilderTest, Build_Failed_without_lock_file) {
  class MockMmpaOpen : public ge::MmpaStubApiGe {
   public:
    virtual INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) {
      return INT32_MAX;
    }
  };
  ComputeGraphPtr compute_graph = BuildComputeGraph("path");
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), FAILED);
}

TEST_F(UdfModelBuilderTest, Build_Failed_lock) {
  class MockMmpaOpen : public ge::MmpaStubApiGe {
   public:
    virtual INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) {
      return INT32_MAX;
    }
  };
  ComputeGraphPtr compute_graph = BuildComputeGraph("path");
  UdfModel udf_model{compute_graph};
  std::string cmd = "touch ./temp/build/compile_lock";
  (void)system(cmd.c_str());
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaOpen>());
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(UdfModelBuilderTest, Build_builtin_udf) {
  ComputeGraphPtr compute_graph = BuildBuiltInUdfComputeGraph();
  std::vector<CompileConfigJson::BufCfg> buf_cfg = {{100,256,100,"normal"}};
  compute_graph->SetExtAttr("_user_buf_cfg", buf_cfg);
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), SUCCESS);
}

TEST_F(UdfModelBuilderTest, Build_builtin_udf_WithMapInput) {
  ComputeGraphPtr compute_graph = BuildBuiltInUdfComputeGraphWithMapInput();
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), SUCCESS);
}

TEST_F(UdfModelBuilderTest, Get_normalize_name) {
  const auto normalize_name = UdfModelBuilder::GetInstance().GenNormalizeModelName("udf_model_!@_//_**-");
  EXPECT_EQ(normalize_name.find("!"), std::string::npos);
  EXPECT_EQ(normalize_name.find("@"), std::string::npos);
  EXPECT_EQ(normalize_name.find("-"), std::string::npos);
  EXPECT_EQ(normalize_name.find("*"), std::string::npos);
  EXPECT_NE(normalize_name.find("_release"), std::string::npos);
}

TEST_F(UdfModelBuilderTest, Get_normalize_name_over_length) {
  std::string model_name(260, 'x');
  const auto normalize_name = UdfModelBuilder::GetInstance().GenNormalizeModelName(model_name);
  EXPECT_LT(normalize_name.size(), 255);
}

TEST_F(UdfModelBuilderTest, Gen_release_for_buitlin_model) {
  ComputeGraphPtr compute_graph = BuildBuiltInUdfComputeGraph();
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), SUCCESS);
}

TEST_F(UdfModelBuilderTest, Gen_release_for_udf_without_cache_on_host) {
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  std::string cmd = "touch ./temp/build/compile_lock";
  (void)system(cmd.c_str());
  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), SUCCESS);
}

TEST_F(UdfModelBuilderTest, Gen_release_for_udf_without_cache_on_device) {
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  std::string cmd = "touch ./temp/build/compile_lock";
  (void)system(cmd.c_str());
  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  for (const NodePtr &node : compute_graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    ge::AttrUtils::SetStr(op_desc, "_dflow_final_location", "Ascend");
  }
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), SUCCESS);
}

TEST_F(UdfModelBuilderTest, Gen_release_for_udf_with_cache) {
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  std::string cmd = "touch ./temp/build/compile_lock";
  (void)system(cmd.c_str());
  cmd = "cd ./temp/release; touch 1.so; touch test_release.om; cd ../..";
  (void)system(cmd.c_str());

  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  std::string release_info;
  (void)UdfModelBuilder::GetInstance().GetReleaseInfo("./temp/release", release_info);
  compute_graph->SetExtAttr("_cache_graph_info_for_data_flow_cache", release_info);
  const std::string om_file_path = "./temp/release/test_release.om";
  compute_graph->SetExtAttr("_cache_graph_udf_om_file", om_file_path);
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), SUCCESS);
  cmd = "rm -rf ./temp";
  (void)system(cmd.c_str());
}

TEST_F(UdfModelBuilderTest, Gen_release_for_udf_with_cache_failed) {
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  std::string cmd = "touch ./temp/build/compile_lock";
  (void)system(cmd.c_str());
  cmd = "cd ./temp/release; touch 1.so; touch test_release.tar.gz; cd ../..";
  (void)system(cmd.c_str());

  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  std::string release_info;
  (void)UdfModelBuilder::GetInstance().GetReleaseInfo("./temp/release", release_info);
  compute_graph->SetExtAttr("_cache_graph_info_for_data_flow_cache", release_info);
  const std::string om_file_path = "./temp/release/test_release";
  compute_graph->SetExtAttr("_cache_graph_udf_om_file", om_file_path);
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), PARAM_INVALID);
  cmd = "rm -rf ./temp";
  (void)system(cmd.c_str());
}

TEST_F(UdfModelBuilderTest, Gen_release_for_udf_with_old) {
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  std::string cmd = "touch ./temp/build/compile_lock";
  (void)system(cmd.c_str());
  cmd = "cd ./temp/release; touch 1.so; touch test_release.tar.gz; cd ../..";
  (void)system(cmd.c_str());

  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  std::string release_info;
  (void)UdfModelBuilder::GetInstance().GetReleaseInfo("./temp/release", release_info);
  compute_graph->SetExtAttr("_cache_graph_info_for_data_flow_cache", release_info);
  UdfModel udf_model{compute_graph};
  EXPECT_EQ(UdfModelBuilder::GetInstance().Build(udf_model), FAILED);
  cmd = "rm -rf ./temp";
  (void)system(cmd.c_str());
}

namespace {
Status GetDirInfo(const std::string &dir_path, std::string &dir_info) {
  std::string cmd = "cd " + dir_path + ";ls -R";
  cmd += std::string(" ./ 2>&1");
  std::cout << "cmd:" << cmd << std::endl;
  FILE *pipe = popen(cmd.c_str(), "r");
  GE_CHECK_NOTNULL(pipe);
  dir_info = "";
  constexpr int32_t buffer_len = 128;
  char buffer[buffer_len];
  while (!feof(pipe)) {
    if (fgets(buffer, buffer_len, pipe) != nullptr) {
      dir_info += buffer;
    }
  }
  const auto ret = pclose(pipe);
  std::cout << "cmd result:" << dir_info << std::endl;
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED, "Failed to get release info, ret[%d], errmsg[%s]", ret, dir_info.c_str());
  return SUCCESS;
}

void GenerateCmdFile() {
  system("rm -rf tar.sh tar1.sh");
  std::string tar_cmd;
  const std::string release_pkg = "./workspace/_build/Test/release/";
  const std::string normalize_name = "Test_release";
  UdfModelBuilder::GetInstance().GenerateTarCmd(release_pkg, normalize_name, true, tar_cmd);
  std::ofstream tar_str("./tar.sh");
  tar_str << tar_cmd;
  UdfModelBuilder::GetInstance().GenerateTarCmd(release_pkg, normalize_name, false, tar_cmd);
  std::ofstream tar1_str("./tar1.sh");
  tar1_str << tar_cmd;
}
}

// 修改udf model builder GenerateTarCmd后本地验证，线上执行shell命令执行报错
TEST_F(UdfModelBuilderTest, Udf_generate_tar_shell) {
GenerateCmdFile();
std::string cmd = R"(
mkdir -p ./workspace/_build/Test/release/
mkdir -p ./workspace/verify
cd ./workspace/_build/Test/release/
touch Test_release.om
touch 1.so
touch test_release.tar.gz
mkdir subdir
touch subdir/2.so
echo 1>Test_release.om
echo 1>1.so
echo 1>subdir/2.so
cd -
)";
(void)system(cmd.c_str());
  std::string path = "./bin_path";
  {
    std::ofstream bin(path);
    bin << path;
  }
  ComputeGraphPtr compute_graph = BuildComputeGraph(path);
  UdfModel udf_model{compute_graph};

  const std::string  expect_result = "./:\n1.so.68b329da9893e34099c7d8ad5cb9c940\n2.so.68b329da9893e34099c7d8ad5cb9c940"
                                     "\nTest_release.om\nTest_release.om_dir\n\n./Test_release.om_dir:\n1.so\n"
                                     "subdir\n\n./Test_release.om_dir/subdir:\n2.so\n";
  const std::string  expect1_result = "./:\nTest_release.om\nTest_release.om_dir\n\n./Test_release.om_dir:"
                                     "\n1.so\nsubdir\n\n./Test_release.om_dir/subdir:\n2.so\n";
  auto ret = system("bash ./tar1.sh");
  if (ret == 0) {
    std::cout << "Verify tar pack" << std::endl;
    ASSERT_EQ(system("tar -xzf ./workspace/_build/Test/release/Test_release.tar.gz -C ./workspace/verify/"), 0);
    std::string real_result;
    GetDirInfo("./workspace/verify/udf_resource/", real_result);
    EXPECT_EQ(real_result, expect1_result);
  }

  (void)system("mv ./workspace/verify/udf_resource/*.om ./workspace/_build/Test/release/;"
               " rm -rf ./workspace/verify/udf_resource");
  (void)system("rm -rf ./workspace/_build/Test/release/Test_release.tar.gz");
  ret = system("bash ./tar.sh");
  if (ret == 0) {
    std::cout << "Verify tar pack" << std::endl;
    ASSERT_EQ(system("tar -xzf ./workspace/_build/Test/release/Test_release.tar.gz -C ./workspace/verify/"), 0);
    std::string real_result1;
    GetDirInfo("./workspace/verify/udf_resource/", real_result1);
    EXPECT_EQ(real_result1, expect_result);
  }

  (void)system("rm -rf ./workspace");
  (void)system("rm -rf tar.sh tar1.sh");
}
}  // namespace ge
