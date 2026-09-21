/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/model_serialize.h"
#include "graph/utils/graph_utils_ex.h"
#include "api/atc/main_impl.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/weights_parser.h"
#include "framework/common/types.h"
#include "graph/types.h"
#include "graph/ge_global_options.h"

#include "ge_running_env/atc_utils.h"
#include "proto/ge_ir.pb.h"
#include "framework/common/scope_guard.h"
#include "graph/ge_local_context.h"
#include "graph/passes/graph_builder_utils.h"
#include "dlog_pub.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "ge_runtime_stub/include/common/share_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "register/optimization_option_registry.h"
#include "depends/mmpa/src/mmpa_stub.h"

using namespace domi;
using namespace ge;

DECLARE_bool(help);
DECLARE_int32(virtual_type);
DECLARE_string(model);
DECLARE_string(oo_level);
const char *const kEnvName = "ASCEND_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";

class UtestMain : public AtcTest {};
namespace {
class AtcFileFactory {
 public:
  static std::string GetFileRealName(const std::string &file_name);
  static std::string Generatefile1(const std::string &file_type, const std::string &file_name);
  static std::string GenerateModel(const std::string &file_type, const std::string &file_name, bool with_fusion = false);
  static void RemoveFile(const char *path) {
    remove(path);
  }
};

std::string AtcFileFactory::GetFileRealName(const std::string &file_name) {
  std::string pwd = __FILE__;
  std::size_t idx = pwd.find_last_of("/");
  pwd = pwd.substr(0, idx);
  return pwd + "/" + file_name;
}

std::string AtcFileFactory::Generatefile1(const std::string &file_type, const std::string &file_name) {
  std::string pwd = __FILE__;
  std::size_t idx = pwd.find_last_of("/");
  pwd = pwd.substr(0, idx);
  std::string om_file = pwd + "/" + file_name;
  return file_type + om_file;
}

std::string AtcFileFactory::GenerateModel(const std::string &file_type, const std::string &file_name, bool with_fusion) {
  std::string pwd = __FILE__;
  std::size_t idx = pwd.find_last_of("/");
  pwd = pwd.substr(0, idx);
  const std::string om_file = pwd + "/" + file_name;
  Model model("model_name", "custom version3.0");
  {
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
    auto op_desc = std::make_shared<OpDesc>("test", "test");
    op_desc->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    computeGraph->AddNode(op_desc);
    auto out_put = std::make_shared<OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
    out_put->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    computeGraph->AddNode(out_put);
    model.SetGraph(computeGraph);
  }

  if (with_fusion) {
    std::vector<Buffer> b_list(2U);
    ge::proto::OpDef fusion_op_def;
    auto &fusion_attr_map = *fusion_op_def.mutable_attr();
    fusion_attr_map["fusion_scope"].set_i(0x00010000);
    const auto group_op_l1_id = fusion_op_def.SerializeAsString();
    b_list[0] = Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(group_op_l1_id.data()), group_op_l1_id.length());
    fusion_attr_map["fusion_scope"].set_i(0x00000001);
    const auto group_op_ub_id = fusion_op_def.SerializeAsString();
    b_list[1] = Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(group_op_ub_id.data()), group_op_ub_id.length());
    (void)AttrUtils::SetListBytes(model, MODEL_ATTR_FUSION_MODEL_DEF, b_list);
  }

  ModelSerialize serialize;
  proto::ModelDef model_def;
  serialize.SerializeModel(model, false, model_def); //success
  GraphUtils::WriteProtoToTextFile(model_def, om_file.c_str());
  return file_type + om_file;
}

class StubTensorFlowModelParser : public domi::ModelParser {
 public:
  StubTensorFlowModelParser() {}
  ~StubTensorFlowModelParser() {}
  domi::Status Parse(const char *file, ge::Graph &graph) {
    DEF_GRAPH(g1) {
      CHAIN(NODE("cons1", "Const")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    };
    graph = ToGeGraph(g1);
    return ge::SUCCESS;
  }
  domi::Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) {
    return ge::SUCCESS;
  }
  domi::Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph) {
    return ge::SUCCESS;
  }
  domi::Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) {
    return ge::SUCCESS;
  }
  domi::Status ParseProtoWithSubgraph(const google::protobuf::Message *proto, GetGraphCallback callback,
                                      ge::ComputeGraphPtr &graph) {
    return ge::SUCCESS;
  }
  ge::DataType ConvertToGeDataType(const uint32_t type) {
    return DT_FLOAT;
  }
  domi::Status ParseAllGraph(const google::protobuf::Message *root_proto, ge::ComputeGraphPtr &root_graph) {
    return ge::SUCCESS;
  }
};

REGISTER_MODEL_PARSER_CREATOR(TENSORFLOW, StubTensorFlowModelParser);
REGISTER_MODEL_PARSER_CREATOR(CAFFE, StubTensorFlowModelParser);
REGISTER_MODEL_PARSER_CREATOR(MINDSPORE, StubTensorFlowModelParser);
REGISTER_MODEL_PARSER_CREATOR(ONNX, StubTensorFlowModelParser);

class StubWeightsParser : public domi::WeightsParser {
 public:
  StubWeightsParser() {}
  ~StubWeightsParser() {}
  domi::Status Parse(const char *file, ge::Graph &graph) {
    return ge::SUCCESS;
  }
  domi::Status ParseFromMemory(const char *input, uint32_t lengt, ge::ComputeGraphPtr &graph) {
    return ge::SUCCESS;
  }
};
REGISTER_WEIGHTS_PARSER_CREATOR(CAFFE, StubWeightsParser);
REGISTER_WEIGHTS_PARSER_CREATOR(ONNX, StubWeightsParser);
REGISTER_WEIGHTS_PARSER_CREATOR(TENSORFLOW, StubWeightsParser);

static std::string ConstructOppEnv() {
  std::string path = GetModelPath();
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/') + 1);
  std::string opp_path = path + "opp_oo_test/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());
  system("pwd");
  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  return opp_path;
}
}  // namespace

TEST_F(UtestMain, MainImplTest_socversion_and_mode_fail01) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=30",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend910B",
                  "--input_format=NCHW",
                  "--allow_hf32=true"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_socversion_and_mode_fail02) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend035",
                  "--input_format=NCHW",
                  "--allow_hf32=true"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_socversion_and_mode_match) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=30",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend035A",
                  "--input_format=NCHW",
                  "--allow_hf32=true"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_singleop) {
  std::string singleop_arg = AtcFileFactory::Generatefile1("--singleop=", "add_int.json");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "./");
  char *argv[] = {"atc", const_cast<char *>(singleop_arg.c_str()), const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\""};
  //未打桩，st里面覆盖全流程，当前ut只是覆盖此文件相关代码
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestMain, MainImplTest_singleop_allow_hf32) {
  std::string singleop_arg = AtcFileFactory::Generatefile1("--singleop=", "add_int.json");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "./");
  char *argv[] = {"atc", const_cast<char *>(singleop_arg.c_str()), const_cast<char *>(output_arg.c_str()),
                  "--allow_hf32=true", "--soc_version=\"Ascend310\""};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestMain, MainImplTest_singleopFail) {
  std::string singleop_arg = AtcFileFactory::Generatefile1("--singleop=", "add_int.json");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "./");
  char *argv[] = {"atc", const_cast<char *>(singleop_arg.c_str()), const_cast<char *>(output_arg.c_str()),
                  "--display_model_info=1", "--soc_version=\"Ascend310\""};
  // --singleop与--display_model_info参数冲突
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, -1);

  char *argv1[] = {"atc", const_cast<char *>(singleop_arg.c_str()), "--output=",
                  "--display_model_info=0", "--soc_version=\"Ascend310\""};
  // --singleop中--output参数必选
  ret = main_impl(sizeof(argv1) / sizeof(argv1[0]), argv1);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestMain, MainImplTest_log_level_err) {
  char *argv[] = {"atc", "--log=1"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, MainImplTest_check_mem_info) {
  char *argv[] = {"atc", "--auto_tune_mode=\"RA\""};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, MainImplTest_generate_om_model_tf) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW",
                  "--allow_hf32=true"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_TRUE(flgs::GetUserOptions().find("allow_hf32") != flgs::GetUserOptions().end());
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_global_options) {
  std::string opp_path = "./opp/";
  system(("mkdir -p " + opp_path).c_str());
  setenv(kEnvName, opp_path.c_str(), 1);

  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());

  system("pwd");

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_x86_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_x86_proto_path).c_str());
  inner_x86_proto_path += kOpsProto;
  system(("touch " + inner_x86_proto_path).c_str());
  system(("echo 'ops proto x86 ' > " + inner_x86_proto_path).c_str());

  std::string inner_x86_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_x86_tiling_path).c_str());
  inner_x86_tiling_path += kOpMaster;
  system(("touch " + inner_x86_tiling_path).c_str());
  system(("echo 'op tiling_x86 ' > " + inner_x86_tiling_path).c_str());

  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string op_arg = AtcFileFactory::Generatefile1("--op_precision_mode=", "op_precision.ini");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--deterministic=1",
                  const_cast<char *>(op_arg.c_str()),
                  "--input_format=NCHW",
                  "--host_env_os=linux",
                  "--host_env_cpu=x86_64"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  auto &options = GetMutableGlobalOptions();
  auto it = options.find(ge::DETERMINISTIC);
  EXPECT_NE(it, options.end());
  if (it != options.end()) {
    EXPECT_EQ(it->second == "1", true);
  }
  it = options.find(ge::OP_PRECISION_MODE);
  EXPECT_NE(it, options.end());
  if (it != options.end()) {
    EXPECT_EQ(it->second == AtcFileFactory::GetFileRealName("op_precision.ini"), true);  }
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_generalized_build_mode_error) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW",
                  "--shape_generalized_build_mode=shape_precise1"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, -1);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_status_check_error) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW",
                  "--shape_generalized_build_mode=shape_precise",
                  "--status_check=3"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, -1);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_status_check_success) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW",
                  "--shape_generalized_build_mode=shape_generalized",
                  "--status_check=1"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_allow_hf32_invalid) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW",
                  "--shape_generalized_build_mode=shape_generalized",
                  "--allow_hf32=1"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_generate_om_model_caffe) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string invalid_weight_arg = AtcFileFactory::Generatefile1("--weight=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=0",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  const_cast<char *>(invalid_weight_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_generate_om_model_onnx) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=5",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_generate_om_model_autofuse) {
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(UtestMain, MainImplTest_convert_model_to_json_pb_dump_mode) {
  std::string om_arg = AtcFileFactory::Generatefile1("--om=", "add.pb");
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {"atc",
                  "--mode=1",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(json_arg.c_str()),
                  "--dump_mode=1"

  };
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.json").c_str());
}

TEST_F(UtestMain, MainImplTest_1) {
  std::string om_arg = AtcFileFactory::Generatefile1("--om=", "add.pb");
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {"atc",
                  "--mode=6",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(json_arg.c_str()),
                  "--dump_mode=1"

  };
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.json").c_str());
}

TEST_F(UtestMain, MainImplTest_convert_model_to_json_pb) {
  std::string om_arg = AtcFileFactory::Generatefile1("--om=", "add.pb");
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {
      "atc", "--mode=1", "--framework=3", const_cast<char *>(om_arg.c_str()), const_cast<char *>(json_arg.c_str())

  };
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.json").c_str());
}

TEST_F(UtestMain, MainImplTest_convert_model_to_json_txt_failed) {
  std::string om_arg = AtcFileFactory::Generatefile1(
      "--om=", "ge_proto_00000261_partition0_rank31_new_sub_graph102_SecondPartitioning.txt");
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {"atc", const_cast<char *>(om_arg.c_str()), const_cast<char *>(json_arg.c_str()), "--mode=5"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);  // txt file not exist
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.json").c_str());
}

TEST_F(UtestMain, MainImplTest_convert_model_to_json_prototxt_failed) {
  std::string om_arg = AtcFileFactory::Generatefile1(
      "--om=", "ge_proto_00000261_partition0_rank31_new_sub_graph102_SecondPartitioning.prototxt");
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {"atc", const_cast<char *>(om_arg.c_str()), const_cast<char *>(json_arg.c_str()), "--mode=5"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);  // txt file is invalid
}

TEST_F(UtestMain, MainImplTest_convert_empty_model_to_json_prototxt_failed) {
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {"atc", const_cast<char *>(json_arg.c_str()), "--mode=5"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);  // txt file is empty
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.json").c_str());
}

TEST_F(UtestMain, MainImplTest_convert_model_to_json_txt_successful) {
  std::string om_arg = AtcFileFactory::GenerateModel("--om=", "tmp.txt");
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {"atc", const_cast<char *>(om_arg.c_str()), const_cast<char *>(json_arg.c_str()), "--mode=5"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.json").c_str());
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.txt").c_str());
}

TEST_F(UtestMain, MainImplTest_convert_model_to_json_with_fusion) {
  std::string om_arg = AtcFileFactory::GenerateModel("--om=", "tmp.txt", true);
  std::string json_arg = AtcFileFactory::Generatefile1("--json=", "tmp.json");
  char *argv[] = {"atc", const_cast<char *>(om_arg.c_str()), const_cast<char *>(json_arg.c_str()), "--mode=5"};
  EXPECT_EQ(main_impl(sizeof(argv) / sizeof(argv[0]), argv), 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.json").c_str());
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.txt").c_str());
}

TEST_F(UtestMain, MainImplTest_generate_om_model_virtual_type) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--virtual_type=2",
                  "--framework=5",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, GeFlags_param_ok01) {
  char *argv[] = {"atc",
                  "--virtual_type=0"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestMain, GeFlags_param_ok02) {
  char *argv[] = {"atc",
                  "--virtual_type=1"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_virtual_type, 1);
}

TEST_F(UtestMain, GeFlags_param_ok03) {
  char *argv[] = {"atc",
                  "--model=model_value"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_model, "model_value");
}

TEST_F(UtestMain, GeFlags_param_ok04) {
  char *argv[] = {"atc",
                  "--model",
                  "model_value"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_model, "model_value");
}

TEST_F(UtestMain, GeFlags_param_help01) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help02) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "-help"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help03) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "-h"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help04) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=true"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help05) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=True"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help06) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=T"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help07) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=y"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help08) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=yes"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help09) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=1"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, true);
}

TEST_F(UtestMain, GeFlags_param_help10) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=false"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_help, false);
}

TEST_F(UtestMain, GeFlags_param_help11) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=f"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_help, false);
}

TEST_F(UtestMain, GeFlags_param_help12) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=n"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_help, false);
}

TEST_F(UtestMain, GeFlags_param_help13) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=no"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_help, false);
}

TEST_F(UtestMain, GeFlags_param_help14) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=0"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(FLAGS_help, false);
}

TEST_F(UtestMain, GeFlags_param_help15) {
  FLAGS_help = false;
  char *argv[] = {"atc",
                  "--help=invalid_bool"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_help, false);
}

TEST_F(UtestMain, GeFlags_param_include_dash) {
  char *argv[] = {"atc",
                  "--virtual-type=0"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestMain, GeFlags_param_single_minus) {
  char *argv[] = {"atc",
                  "-virtual_type=0"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestMain, GeFlags_flag_value_empty01) {
  char *argv[] = {"atc",
                  "--virtual_type="};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_flag_value_empty02) {
  char *argv[] = {"atc",
                  "--model="};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestMain, GeFlags_no_flag_value) {
  char *argv[] = {"atc",
                  "--virtual_type"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_data_type_error) {
  char *argv[] = {"atc",
                  "--virtual_type=string_value"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_flag_name_error01) {
  char *argv[] = {"atc",
                  "--virtual_type_err=0"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_flag_name_error02) {
  char *argv[] = {"atc",
                  "--virtual_type_err"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_non_option_parameter01) {
  char *argv[] = {"atc",
                  "virtual_type=0"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_non_option_parameter02) {
  char *argv[] = {"atc",
                  "virtual_type=0",
                  "--mode=0",
                  "--framework=5"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_dump_mode_error) {
  char *argv[] = {"atc", "--dump_mode=3"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_status_check_error) {
  char *argv[] = {"atc", "--status_check=3"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_op_compiler_cache_mode_error) {
  char *argv[] = {"atc", "--op_compiler_cache_mode=\"atc\""};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_framework_error) {
  char *argv[] = {"atc", "--framework=6"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_op_debug_level_error) {
  char *argv[] = {"atc", "--op_debug_level=3.6"};
  int32_t ret = ge::flgs::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_mode_1_framework_error) {
  char *argv[] = {"atc",
                  "--mode=1",
                  "--framework=1",};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_framework_0_version_error) {
  std::string weight = AtcFileFactory::Generatefile1("--weight=", "add.pb");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=0",
                  const_cast<char *>(weight.c_str()),
                  "--soc_version=Ascend910B1"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlagsFramework0_Fail_UnsupportedVersion) {
  std::string weight = AtcFileFactory::Generatefile1("--weight=", "add.pb");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=0",
                  const_cast<char *>(weight.c_str()),
                  "--soc_version=Ascend910_9391"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_framework_0_weight_error) {
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=0",
                  "--weight=***",
                  "--soc_version=\"Ascend310\""};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_framework_0_weight_none) {
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=0",
                  "--soc_version=\"Ascend310\""};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_framework_3_weight_warn) {
  std::string weight = AtcFileFactory::Generatefile1("--weight=", "add.pb");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=3",
                  const_cast<char *>(weight.c_str()),
                  "--soc_version=\"Ascend310\""};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, GeFlags_framework_5_weight_warn) {
  std::string weight = AtcFileFactory::Generatefile1("--weight=", "add.pb");
  char *argv[] = {"atc",
                  "--mode=0",
                  "--framework=5",
                  const_cast<char *>(weight.c_str()),
                  "--soc_version=\"Ascend310\""};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
}

TEST_F(UtestMain, MainImplTest_status_check_success_3) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--mode=3",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW",
                  "--shape_generalized_build_mode=shape_generalized",
                  "--status_check=1"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_status_check_fail_mindspore) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=1",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--out_nodes=relu:0",
                  "--soc_version=Ascend310",
                  "--input_format=NCHW",
                  "--output_type=FP32",
                  "--status_check=0"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

TEST_F(UtestMain, MainImplTest_status_check_fail_mindspore_2) {
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--model=3"
                  "--framework=1",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--out_nodes=relu:0",
                  "--soc_version=Ascend310",
                  "--input_format=NCHW",
                  "--output_type=FP32",
                  "--status_check=0",
                  "--cluster_config=fake.json"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
}

namespace {
ComputeGraphPtr MakeGraph() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto output = builder.AddNode("output", "NetOutput", 1, 1);
  builder.AddDataEdge(data, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, output, 0);
  return builder.GetGraph();
}
}

TEST_F(UtestMain, MainImplTest_single_air_graph) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  map<std::string, std::string> global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });
  GE_MAKE_GUARD(recover_global_cfg, [&global_options](){
    GetThreadLocalContext().SetGlobalOption(global_options);
  });
  GE_MAKE_GUARD(recover_context_cfg, [&](){
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  });
  std::string opp_path = "./opp/";
  system(("mkdir -p " + opp_path).c_str());
  setenv(kEnvName, opp_path.c_str(), 1);

  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());

  system("pwd");

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_x86_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_x86_proto_path).c_str());
  inner_x86_proto_path += kOpsProto;
  system(("touch " + inner_x86_proto_path).c_str());
  system(("echo 'ops proto x86 ' > " + inner_x86_proto_path).c_str());

  std::string inner_x86_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_x86_tiling_path).c_str());
  inner_x86_tiling_path += kOpMaster;
  system(("touch " + inner_x86_tiling_path).c_str());
  system(("echo 'op tiling_x86 ' > " + inner_x86_tiling_path).c_str());

  std::string model_1 = "model1";
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  auto data = compute_graph->FindNode("data");
  EXPECT_NE(data, nullptr);
  EXPECT_NE(data->GetOpDesc()->MutableInputDesc(0), nullptr);
  data->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1,2,3,4,5}));
  data->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(FORMAT_NC1HWC0);
  bool enable_storage_format_spread = true;
  (void)AttrUtils::SetBool(data->GetOpDesc(), "_enable_storage_format_spread", enable_storage_format_spread);
  bool is_origin_format_set = true;
  (void)AttrUtils::SetBool(data->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, is_origin_format_set);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  graph.SaveToFile(model_1.c_str());
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=1",
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend910\"",
                  "--model=model1",
                  "--host_env_os=linux",
                  "--host_env_cpu=x86_64",
                  "--log=error",
                  "--allow_hf32=false"
                  };
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  remove(model_1.c_str());
}

TEST_F(UtestMain, MainImplTest_MindSpore_Input_Shape) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  map<std::string, std::string> global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });
  GE_MAKE_GUARD(recover_global_cfg, [&global_options](){
    GetThreadLocalContext().SetGlobalOption(global_options);
  });
  GE_MAKE_GUARD(recover_context_cfg, [&](){
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  });

  std::string opp_path = "./opp/";
  system(("mkdir -p " + opp_path).c_str());
  setenv(kEnvName, opp_path.c_str(), 1);

  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());

  system("pwd");

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_x86_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_x86_proto_path).c_str());
  inner_x86_proto_path += kOpsProto;
  system(("touch " + inner_x86_proto_path).c_str());
  system(("echo 'ops proto x86 ' > " + inner_x86_proto_path).c_str());

  std::string inner_x86_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_x86_tiling_path).c_str());
  inner_x86_tiling_path += kOpMaster;
  system(("touch " + inner_x86_tiling_path).c_str());
  system(("echo 'op tiling_x86 ' > " + inner_x86_tiling_path).c_str());

  std::string model_1 = "model1";
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  graph.SaveToFile(model_1.c_str());
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=1",
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend910\"",
                  "--model=model1",
                  "--input_shape=data:-1,1",
                  "--dynamic_batch_size=1,2"
  };
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  remove(model_1.c_str());
}

TEST_F(UtestMain, MainImplTest_MindSpore_Invalid_Input_Shape) {
  map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  map<std::string, std::string> global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });
  GE_MAKE_GUARD(recover_global_cfg, [&global_options](){
    GetThreadLocalContext().SetGlobalOption(global_options);
  });
  GE_MAKE_GUARD(recover_context_cfg, [&](){
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  });

  std::string opp_path = "./opp/";
  system(("mkdir -p " + opp_path).c_str());
  setenv(kEnvName, opp_path.c_str(), 1);

  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());

  system("pwd");

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_x86_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_x86_proto_path).c_str());
  inner_x86_proto_path += kOpsProto;
  system(("touch " + inner_x86_proto_path).c_str());
  system(("echo 'ops proto x86 ' > " + inner_x86_proto_path).c_str());

  std::string inner_x86_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_x86_tiling_path).c_str());
  inner_x86_tiling_path += kOpMaster;
  system(("touch " + inner_x86_tiling_path).c_str());
  system(("echo 'op tiling_x86 ' > " + inner_x86_tiling_path).c_str());

  std::string model_1 = "model1";
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  graph.SaveToFile(model_1.c_str());
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv_invalid_type[] = {"atc",
                  "--framework=1",
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend910\"",
                  "--model=model1",
                  "--input_shape=addn1:-1,1",
                  "--dynamic_batch_size=1,2"
  };
  int32_t ret = main_impl(sizeof(argv_invalid_type) / sizeof(argv_invalid_type[0]), argv_invalid_type);
  EXPECT_NE(ret, 0);

  char *argv_invalid_name[] = {"atc",
                  "--framework=1",
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=\"Ascend910\"",
                  "--model=model1",
                  "--input_shape=data1:-1,1",
                  "--dynamic_batch_size=1,2"
  };
  ret = main_impl(sizeof(argv_invalid_name) / sizeof(argv_invalid_name[0]), argv_invalid_name);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  remove(model_1.c_str());
}
namespace ge {
Status CallAmctInterface(ge::Graph &graph, std::map<std::string, std::string> &options);
}

char g_handleStub;
bool g_isError = false;
using amctStatus = int32_t;
amctStatus amctGraphCalibration(ge::Graph &graph, const std::map<std::string, std::string> &options) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  if (compute_graph != nullptr) {
    return static_cast<amctStatus>(ge::GRAPH_NOT_CHANGED);
  } else {
    return static_cast<amctStatus>(ge::GRAPH_FAILED);
  }
}

class MockMmpa : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (string("libamctacl.so") == file_name) {
      return (void *) &g_handleStub;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (g_isError) {
      return nullptr;
    }
    if (std::string(func_name) == "amctGraphCalibration") {
      return (void *) &amctGraphCalibration;
    }
    return dlsym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    return 0;
  }
};

TEST_F(UtestMain, MainImplTest_amct_interface) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  Graph graph("test");
  std::map<std::string, std::string> options;
  auto ret = CallAmctInterface(graph, options);
  EXPECT_EQ(ret, 0);

  options[COMPRESSION_OPTIMIZE_CONF] = "path_test";
  g_isError = true;
  ret = CallAmctInterface(graph, options);
  EXPECT_NE(ret, 0);

  g_isError = false;
  ret = CallAmctInterface(graph, options);
  EXPECT_NE(ret, 0);

  graph.SetValid();
  ret = CallAmctInterface(graph, options);
  EXPECT_EQ(ret, 0);
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestMain, GeFlags_oo_help) {
  char *argv[] = {"atc",
                  "--help"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestMain, GeFlags_oo_init) {
  const auto opp_path = ConstructOppEnv();
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend910B",
                  "--oo_level=O1",
                  "--oo_constant_folding=true"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_oo_level, "O1");
  std::string opt_value;
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");
  system(("rm -rf " + opp_path).c_str());

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestMain, GeFlags_oo_init_param_valid) {
  const auto opp_path = ConstructOppEnv();
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  std::string opt_value;

  char *argv1[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend910B",
                  "--oo_level=O1",
                  "--oo_constant_folding=T"};
  auto ret = main_impl(sizeof(argv1) / sizeof(argv1[0]), argv1);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_oo_level, "O1");
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");

  char *argv2[] = {"atc",
                   "--framework=3",
                   const_cast<char *>(om_arg.c_str()),
                   const_cast<char *>(output_arg.c_str()),
                   "--soc_version=Ascend910B",
                   "--oo_level=O1",
                   "--oo_constant_folding=1"};
  ret = main_impl(sizeof(argv2) / sizeof(argv2[0]), argv2);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_oo_level, "O1");
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");

  char *argv3[] = {"atc",
                   "--framework=3",
                   const_cast<char *>(om_arg.c_str()),
                   const_cast<char *>(output_arg.c_str()),
                   "--soc_version=Ascend910B",
                   "--oo_level=O1",
                   "--oo_constant_folding=0"};
  ret = main_impl(sizeof(argv3) / sizeof(argv3[0]), argv3);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_oo_level, "O1");
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");

  char *argv4[] = {"atc",
                   "--framework=3",
                   const_cast<char *>(om_arg.c_str()),
                   const_cast<char *>(output_arg.c_str()),
                   "--soc_version=Ascend910B",
                   "--oo_level=O1",
                   "--oo_constant_folding=t1"};
  ret = main_impl(sizeof(argv4) / sizeof(argv4[0]), argv4);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_oo_level, "O1");

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestMain, GeFlags_oo_init_param_invalid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  const auto opp_path = ConstructOppEnv();
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend910B",
                  "--oo_level=O4",
                  "--oo_constant_folding=true"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_oo_level, "O4");
  std::string opt_value;
  EXPECT_NE(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);

  char *argv4[] = {"atc",
                   "--framework=3",
                   const_cast<char *>(om_arg.c_str()),
                   const_cast<char *>(output_arg.c_str()),
                   "--soc_version=Ascend910B",
                   "--oo_level=O1",
                   "--oo_constant_folding=f1"};
  ret = main_impl(sizeof(argv4) / sizeof(argv4[0]), argv4);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(FLAGS_oo_level, "O1");
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestMain, GeFlags_export_compile_stat_invalid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  const auto opp_path = ConstructOppEnv();
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend910B",
                  "--export_compile_stat=3"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  std::string opt_value;
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestMain, GeFlags_export_compile_stat_valid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  flgs::GetUserOptions().clear();
  const auto opp_path = ConstructOppEnv();
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--soc_version=Ascend910B",
                  "--export_compile_stat=1"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  std::string opt_value;
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "1");

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

// 校验输入hint shape的index校验失败
TEST_F(UtestMain, MainImplTest_generate_om_model_autofuse_dynamic_shape_index_invalid) {
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--input_hint_shape=-1:[2]",
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  unsetenv("AUTOFUSE_FLAGS");
}

// input_hint_shape校验失败，没有用[]
TEST_F(UtestMain, MainImplTest_generate_om_model_autofuse_dynamic_shape_shape_invalid) {
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--input_hint_shape=1:2",
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  unsetenv("AUTOFUSE_FLAGS");
}

// test input hint shape symbolic failed scenario
TEST_F(UtestMain, MainImplTest_generate_om_model_autofuse_dynamic_shape_symbolic_failed) {
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--input_shape=Placeholder:-1;Placeholder_1:-1",
                  "--input_hint_shape=0:[2];1:[3]",
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  unsetenv("AUTOFUSE_FLAGS");
}

// test input hint shape and input dynamic param failed
TEST_F(UtestMain, MainImplTest_generate_om_model_autofuse_hint_shape_with_dyna_param_failed) {
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  std::string om_arg = AtcFileFactory::Generatefile1("--model=", "add.pb");
  std::string output_arg = AtcFileFactory::Generatefile1("--output=", "tmp");
  char *argv[] = {"atc",
                  "--framework=3",
                  const_cast<char *>(om_arg.c_str()),
                  const_cast<char *>(output_arg.c_str()),
                  "--input_shape=Placeholder:-1;Placeholder_1:-1",
                  "--input_hint_shape=0:[3];1:[3]",
                  "--dynamic_dims=2;4;8;16",
                  "--soc_version=\"Ascend310\"",
                  "--input_format=NCHW"};
  int32_t ret = main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  EXPECT_NE(ret, 0);
  AtcFileFactory::RemoveFile(AtcFileFactory::Generatefile1("", "tmp.om").c_str());
  unsetenv("AUTOFUSE_FLAGS");
}