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
#include "register/op_config_registry.h"
#include "register/device_op_impl_registry.h"
#include "register/opdef/op_config_registry_impl.h"

namespace ops {
namespace {
class OpDefFactoryUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

class AddAscendC : public OpDef {
 public:
  AddAscendC(const char *name) : OpDef(name) {}
};

OP_ADD(AddAscendC, None);

class AddCustomRegMacro : public OpDef {
  public:
  AddCustomRegMacro(const char *name) : OpDef(name) {}
};

REGISTER_OP_AICORE_CONFIG(AddCustomRegMacro, ascendxxxy, []() {
  ops::OpAICoreConfig config("ascendxxxy");
  return config;
});

OP_ADD(AddCustomRegMacro);

class AddCustomAddConfigWithRegMacro : public OpDef {
  public:
  AddCustomAddConfigWithRegMacro(const char *name) : OpDef(name) {
    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(false)
      .DynamicFormatFlag(false)
      .DynamicRankSupportFlag(false)
      .DynamicShapeSupportFlag(false)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(false);
    this->AICore().AddConfig("ascend111y", aicoreConfig);
  }
};

REGISTER_OP_AICORE_CONFIG(AddCustomAddConfigWithRegMacro, ascend111y, []() {
  ops::OpAICoreConfig config;
  config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(true)
      .PrecisionReduceFlag(true);
  return config;
});

OP_ADD(AddCustomAddConfigWithRegMacro);


TEST_F(OpDefFactoryUT, OpDefFactoryTest) {
  auto &ops = OpDefFactory::GetAllOp();
  EXPECT_EQ(ops.size(), 3);
  EXPECT_EQ(std::string(ops[0].GetString()), "AddAscendC");
  EXPECT_EQ(std::string(ops[1].GetString()), "AddCustomRegMacro");
  EXPECT_EQ(std::string(ops[2].GetString()), "AddCustomAddConfigWithRegMacro");
  if (std::string(ops[0].GetString()) == "AddAscendC") {
    OpDef opDef = OpDefFactory::OpDefCreate(ops[0].GetString());
    EXPECT_EQ(opDef.GetOpType(), "AddAscendC");
  }
}

TEST_F(OpDefFactoryUT, DeviceOpImplRegisterUt) {
  auto op_device_register_tmp = optiling::DeviceOpImplRegister("AddAscendC");
  optiling::DeviceOpImplRegister op_device_register_tmp2 = optiling::DeviceOpImplRegister("AddAscendC");
  optiling::DeviceOpImplRegister op_device_register_move(std::move(op_device_register_tmp));
  optiling::DeviceOpImplRegister op_device_register = op_device_register_move;
  op_device_register.Tiling((optiling::SinkTilingFunc)nullptr);
  EXPECT_EQ(ops::OpDefFactory::OpIsTilingSink("AddAscendC"), true);
}

TEST_F(OpDefFactoryUT, RegisterOpAICoreConfigTest) {
  auto regConfigs = GetOpAllAICoreConfig("AddCustomRegMacro");
  EXPECT_EQ(regConfigs.size(), 1);
  auto it = regConfigs.cbegin();
  EXPECT_EQ(std::string((it->first).GetString()), "ascendxxxy");
  EXPECT_NE(it->second, nullptr);

  OpDef opDef = OpDefFactory::OpDefCreate("AddCustomRegMacro");
  std::map<ge::AscendString, OpAICoreConfig> aicfgs = opDef.AICore().GetAICoreConfigs();
  EXPECT_TRUE(aicfgs.find("ascendxxxy") != aicfgs.end());
  EXPECT_EQ(aicfgs.size(), 1);
  OpAICoreConfig config = aicfgs["ascendxxxy"];
  std::map<ge::AscendString, ge::AscendString> cfgs = config.GetCfgInfo();
  // shoud be default config
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "true");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "true");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "true");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "true");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "false");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "true");

  ops::OpConfigRegistry configRegistry;
  configRegistry.RegisterOpAICoreConfig(nullptr, nullptr, nullptr);
  configRegistry.RegisterOpAICoreConfig("AddCustomNullptr", nullptr, nullptr);

  OpConfigRegistryImpl::GetInstance().AddAICoreConfig(nullptr, nullptr, nullptr);
  OpConfigRegistryImpl::GetInstance().AddAICoreConfig("AddCustomNullptr", nullptr, nullptr);
  OpConfigRegistryImpl::GetInstance().GetOpAllAICoreConfig(nullptr);
}

TEST_F(OpDefFactoryUT, AddConfigWithRegMacroTest) {
  auto regConfigs = GetOpAllAICoreConfig("AddCustomAddConfigWithRegMacro");
  EXPECT_EQ(regConfigs.size(), 1);
  auto it = regConfigs.cbegin();
  EXPECT_EQ(std::string((it->first).GetString()), "ascend111y");
  EXPECT_NE(it->second, nullptr);

  OpDef opDef = OpDefFactory::OpDefCreate("AddCustomAddConfigWithRegMacro");
  std::map<ge::AscendString, OpAICoreConfig> aicfgs = opDef.AICore().GetAICoreConfigs();
  EXPECT_TRUE(aicfgs.find("ascend111y") != aicfgs.end());
  EXPECT_EQ(aicfgs.size(), 1);
  OpAICoreConfig config = aicfgs["ascend111y"];
  std::map<ge::AscendString, ge::AscendString> cfgs = config.GetCfgInfo();
  // custom config set by opdef addconfig should overwrite the config set by REGISTER_OP_AICORE_CONFIG
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "false");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "false");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "false");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "false");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "false");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "false");
}
}  // namespace
}  // namespace ops
