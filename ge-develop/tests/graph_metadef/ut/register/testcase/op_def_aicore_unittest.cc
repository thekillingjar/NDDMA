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
#include <vector>
#include <string>
#include "register/op_def_registry.h"

namespace ops {

namespace {

class OpDefAICoreUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpDefAICoreUT, AICoreTest) {
  OpAICoreDef aicoreDef;
  OpAICoreConfig config;
  config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(true)
      .PrecisionReduceFlag(true);
  std::map<ge::AscendString, ge::AscendString> cfgs = config.GetCfgInfo();
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "true");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "true");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "true");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "true");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "true");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "true");
  config.DynamicCompileStaticFlag(false)
      .DynamicFormatFlag(false)
      .DynamicRankSupportFlag(false)
      .DynamicShapeSupportFlag(false)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(false);
  cfgs = config.GetCfgInfo();
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "false");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "false");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "false");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "false");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "false");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "false");
  aicoreDef.AddConfig("ascend310p", config);
  aicoreDef.AddConfig("ascend910", config);
  aicoreDef.AddConfig("ascend310p", config);
  std::map<ge::AscendString, OpAICoreConfig> aicfgs = aicoreDef.GetAICoreConfigs();
  EXPECT_TRUE(aicfgs.find("ascend310p") != aicfgs.end());
  EXPECT_EQ(aicfgs.size(), 2);
  aicoreDef.AddConfig("ascend310p");
  aicfgs = aicoreDef.GetAICoreConfigs();
  config = aicfgs["ascend310p"];
  cfgs = config.GetCfgInfo();
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "true");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "true");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "true");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "true");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "false");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "true");

  // test ZeroEleOutputLaunch
  aicoreDef.LaunchWithZeroEleOutputTensors(true);
  EXPECT_EQ(aicoreDef.GetZeroEleOutputLaunchFlag(), true);
  // test default aicore config
  OpAICoreConfig configDefault("ascend310p");
  cfgs = configDefault.GetCfgInfo();
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "true");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "true");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "true");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "true");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "false");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "true");

  OpAICPUDef aicpuDef;
  aicpuDef.ExtendCfgInfo("opInfo.opsFlag", "OPS_FLAG_OPEN");
  auto& cfgInfo = aicpuDef.GetCfgInfo();
  EXPECT_EQ(cfgInfo["opInfo.opsFlag"], "OPS_FLAG_OPEN");

  OpHostCPUDef opHostCPUDef;
  opHostCPUDef.ExtendCfgInfo("opInfo.opsFlagHost", "OPS_FLAG_OPEN");
  auto& cfgInfoHost = opHostCPUDef.GetCfgInfo();
  EXPECT_EQ(cfgInfoHost["opInfo.opsFlagHost"], "OPS_FLAG_OPEN");
}

}  // namespace
}  // namespace ops
