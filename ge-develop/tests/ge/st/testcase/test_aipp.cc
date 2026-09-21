/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <cstdlib>

#include "api/atc/main_impl.h"
#include "ge_running_env/path_utils.h"
#include "ge_running_env/atc_utils.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/fake_graph_optimizer.h"

#include "utils/model_factory.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/graph_utils.h"
#include "types.h"
#include "init_ge.h"
#include "graph/preprocess/insert_op/insert_aipp_op_util.h"
#include "graph/operator_factory_impl.h"

namespace ge {
class AippSTest : public AtcTest {
  void SetUp() override {
    auto aipp_stub_infer_func = [](Operator &op) -> graphStatus {
      auto images_desc = op.GetInputDesc("images");
      auto images_shape = images_desc.GetShape().GetDims();
      int64_t size = 1;
      for (auto dim : images_shape) {
        size *= dim;  // RGB888_U8 size is n*3*h*w (c is 3 in GenerateModel)
      }
      size = abs(size);
      images_desc.SetSize(size);
      (void)op.UpdateOutputDesc("features", images_desc);
      images_desc.SetDataType(DT_UINT8);
      (void)op.UpdateInputDesc("images", images_desc);
      return 0;
    };
    // aipp infershape need register
    GeRunningEnvFaker ge_env;
    ge_env.InstallDefault().Install(FakeOp(AIPP).InfoStoreAndBuilder("AicoreLib").InferShape(aipp_stub_infer_func));
    auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
    ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
    OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("Const", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("Conv2d", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("Relu", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("Aipp", aipp_stub_infer_func);
  }
  void TearDown() override {
    /** DT场景验证不同入口功能, 需要重置ctx全局对象. atc在SetDynamicInputSizeOptions中设置动态输入选项,
     * 若不清空, 全局ctx会在session用例中被设置为LocalOmgContext, 导致用例失败
     */
    ge::OmgContext ctx;
    domi::GetContext() = ctx;
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Const");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Conv2d");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Relu");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Aipp");
  }
};

TEST_F(AippSTest, StaticAipp_StaticShape) {
  auto path = ModelFactory::GenerateModel_2(false);
  std::string model_arg = "--model=" + path;
  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "static_aipp_static_shape");
  std::string output_arg = "--output=" + om_path;

  std::string conf_path = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_static.cfg";
  char real_path[PATH_MAX] = {};
  realpath(conf_path.c_str(), real_path);
  std::string insert_conf_arg = "--insert_op_conf=" + std::string(real_path);
  char *argv[] = {
      "atc",
      const_cast<char *>(model_arg.c_str()),
      const_cast<char *>(output_arg.c_str()),
      const_cast<char *>(insert_conf_arg.c_str()),
      "--framework=1",  // FrameworkType
      "--mode=0",       // Aipp only support mode 0
      "--out_nodes=relu:0",
      "--virtual_type=0",
      "--soc_version=Ascend310",
      "--input_format=NCHW",
      "--output_type=FP32",
  };
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp")
  main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  // EXPECT_EQ(ret, 0);
  ReInitGe();  // the main_impl will call GEFinalize, so re-init after call it

  CHECK_GRAPH(PrepareAfterInsertAipp) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 7 + 2);  // TODO: other check
  };
}

TEST_F(AippSTest, DynamicAipp_Case) {
  auto path = ModelFactory::GenerateModel_1();
  std::string model_arg = "--model=" + path;
  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "dynamic_aipp");
  std::string output_arg = "--output=" + om_path;

  std::string conf_path = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_dynamic.cfg";
  char real_path[PATH_MAX] = {};
  realpath(conf_path.c_str(), real_path);
  std::string insert_conf_arg = "--insert_op_conf=" + std::string(real_path);
  char *argv[] = {
      "atc",
      const_cast<char *>(model_arg.c_str()),
      const_cast<char *>(output_arg.c_str()),
      const_cast<char *>(insert_conf_arg.c_str()),
      "--framework=1",  // FrameworkType
      "--mode=0",       // Aipp only support mode 0
      "--out_nodes=relu:0",
      "--soc_version=Ascend310",
      "--virtual_type=0",
      "--input_format=NCHW",
      "--output_type=FP32",
      "--input_shape=data1:-1,3,16,16",
      "--dynamic_batch_size=1,2",
  };
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp")
  OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {
    auto input = op.GetInputDesc(0);
    op.GetOutputDesc(0).SetShape(input.GetShape());
    return GRAPH_SUCCESS;});
  main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  // EXPECT_EQ(ret, 0);
  ReInitGe();  // the main_impl will call GEFinalize, so re-init after call it

  CHECK_GRAPH(PrepareAfterInsertAipp) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 9);
    EXPECT_EQ(graph->GetAllNodesSize(), 9 + 5 * 2);
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);
    // TODO: other check
  };
}

TEST_F(AippSTest, StaticAipp_Case) {
  auto path = ModelFactory::GenerateModel_2();
  std::string model_arg = "--model=" + path;
  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "aipp_static");
  std::string output_arg = "--output=" + om_path;

  std::string conf_path = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_static.cfg";
  char real_path[PATH_MAX] = {};
  realpath(conf_path.c_str(), real_path);
  std::string insert_conf_arg = "--insert_op_conf=" + std::string(real_path);
  char *argv[] = {
      "atc",
      const_cast<char *>(model_arg.c_str()),
      const_cast<char *>(output_arg.c_str()),
      const_cast<char *>(insert_conf_arg.c_str()),
      "--framework=1",  // FrameworkType
      "--mode=0",       // Aipp only support mode 0
      "--out_nodes=relu:0",
      "--virtual_type=0",
      "--soc_version=Ascend310",
      "--input_format=NCHW",
      "--output_type=FP32",
      "--input_shape=data1:-1,3,16,16",
      "--dynamic_batch_size=1,2",
  };
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp")
  main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  // EXPECT_EQ(ret, 0); // TODO RECOVER
  ReInitGe();  // the main_impl will call GEFinalize, so re-init after call it

  CHECK_GRAPH(PrepareAfterInsertAipp) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 8);
    // TODO: other check
  };
}


TEST_F(AippSTest, StaticAipp_Config_Invalid) {
  InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
  instance.Init();
  GraphManagerOptions options;

  options.insert_op_file = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_conf_check.cfg";
  options.dynamic_image_size = "244,244;112,112";
  auto ret = instance.Parse(options);
  ASSERT_NE(ret, SUCCESS);
}
// TODO NEED RECOVER
/*TEST_F(AippSTest, StaticInsertAippInSubGraph) {
  auto path = ModelFactory::GenerateModel_4();
  std::string model_arg = "--model=" + path;
  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "aipp_static");
  std::string output_arg = "--output=" + om_path;

  std::string conf_path = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_subgraph_conf.cfg";
  char real_path[PATH_MAX] = {};
  realpath(conf_path.c_str(), real_path);
  std::string insert_conf_arg = "--insert_op_conf=" + std::string(real_path);
  char *argv[] = {
      "atc",
      const_cast<char *>(model_arg.c_str()),
      const_cast<char *>(output_arg.c_str()),
      const_cast<char *>(insert_conf_arg.c_str()),
      "--framework=1",  // FrameworkType
      "--mode=0",       // Aipp only support mode 0
      "--out_nodes=relu:0",
      "--virtual_type=0",
      "--soc_version=Ascend310",
      "--input_format=NCHW",
      "--output_type=FP32",
      "--enable_small_channel=1"
      "--input_shape=data1:128,3,224,224;data2:128,3,224,224;data3:128,3,224,224;data4:128,3,224,224",
  };
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp")
  main_impl(sizeof(argv) / sizeof(argv[0]), argv);
  // EXPECT_EQ(ret, SUCCESS);
  ReInitGe();  // the main_impl will call GEFinalize, so re-init after call it
  string aipp_data_name0;
  string aipp_data_name1;
  CHECK_GRAPH(PrepareAfterInsertAipp) {
    size_t aipp_cnt = 0, node_size = 0;
    for (auto node : graph->GetAllNodes()) {
      node_size++;
      if (node->GetType() != AIPP) {
        continue;
      }
      string node_name = node->GetInAnchor(0)->GetPeerAnchors().at(0)->GetOwnerNode()->GetName();
      if (node_name == "data001") {
        aipp_data_name0 = node_name;
      }
      if (node_name == "data11") {
        aipp_data_name1 = node_name;
      }
      aipp_cnt++;
    }

    EXPECT_EQ(aipp_cnt, 2U);
    EXPECT_EQ(aipp_data_name0, "data001");
    EXPECT_EQ(aipp_data_name1, "data11");
  };
}*/
}  // namespace ge
