/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "nlohmann/json.hpp"
#include "gtest/gtest.h"
#include "proto/dflow.pb.h"
#include "dflow/inc/data_flow/flow_graph/model_pp.h"
#include "dflow/flow_graph/data_flow_attr_define.h"

using namespace testing;
namespace ge {
class ModelPpTest : public Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(ModelPpTest, Serialize_success) {
  dflow::ModelPp model_pp("modelPpName", "./testModelPp.om");
  std::string pp_config_file = "./config.json";
  {
    nlohmann::json pp1_cfg_json = {{"invoke_model_fusion_inputs""0~7"}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  model_pp.SetCompileConfig(nullptr).SetCompileConfig(pp_config_file.c_str());
  AscendString str;
  model_pp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.name(), "modelPpName");
  ASSERT_EQ(process_point.compile_cfg_file(), pp_config_file);
  const auto &extend_attrs = process_point.pp_extend_attrs();
  const auto &find_ret = extend_attrs.find(dflow::INNER_PP_CUSTOM_ATTR_MODEL_PP_MODEL_PATH);
  ASSERT_FALSE(find_ret == extend_attrs.cend());
  EXPECT_EQ(find_ret->second, "./testModelPp.om");
  (void)system("rm -rf ./config.json");
}

TEST_F(ModelPpTest, Serialize_failed) {
  dflow::ModelPp model_pp("modelPpName", nullptr);
  AscendString str;
  model_pp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.name(), "modelPpName");
  const auto &extend_attrs = process_point.pp_extend_attrs();
  const auto &find_ret = extend_attrs.find(dflow::INNER_PP_CUSTOM_ATTR_MODEL_PP_MODEL_PATH);
  ASSERT_TRUE(find_ret == extend_attrs.cend());
}
}  // namespace ge
