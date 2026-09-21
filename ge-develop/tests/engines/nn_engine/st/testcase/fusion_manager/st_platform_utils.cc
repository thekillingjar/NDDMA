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
#include <nlohmann/json.hpp>
#include "ge/ge_api_types.h"
#include "common/util/json_util.h"

#define protected public
#define private public
#include "platform/platform_info_def.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#undef private
#undef protected

using namespace std;
using namespace fe;

class PlatFormUtilsST: public testing::Test
{
 protected:
  static void SetUpTestCase() {
    std::cout << "PlatFormUtilsST SetUpTestCase" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "PlatFormUtilsST TearDownTestCase" << std::endl;
  }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(PlatFormUtilsST, init_success_case1)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "10");
  options.emplace(ge::BUFFER_OPTIMIZE, "l2_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend910B");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend910B");
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), false);
  EXPECT_EQ(instance.GetAICoreNum(), 10);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_infos.GetSocVersion(), "Ascend910B");
  EXPECT_EQ(optional_infos.GetAICoreNum(), 10);
  EXPECT_EQ(optional_infos.GetCoreType(), "AiCore");
  EXPECT_EQ(optional_infos.GetL1FusionFlag(), "false");

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_info.soc_version, "Ascend910B");
  EXPECT_EQ(optional_info.ai_core_num, 10);
  EXPECT_EQ(optional_info.core_type, "AiCore");
  EXPECT_EQ(optional_info.l1_fusion_flag, "false");
}

TEST_F(PlatFormUtilsST, init_success_case2)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::CORE_TYPE, "VectorCore");
  options.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend035");
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), false);
  EXPECT_EQ(instance.GetAICoreNum(), 1);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_infos.GetSocVersion(), "Ascend035");
  EXPECT_EQ(optional_infos.GetAICoreNum(), 1);
  EXPECT_EQ(optional_infos.GetCoreType(), "VectorCore");
  EXPECT_EQ(optional_infos.GetL1FusionFlag(), "true");

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_info.soc_version, "Ascend035");
  EXPECT_EQ(optional_info.ai_core_num, 1);
  EXPECT_EQ(optional_info.core_type, "VectorCore");
  EXPECT_EQ(optional_info.l1_fusion_flag, "true");
}

TEST_F(PlatFormUtilsST, init_fail_case1) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::CORE_TYPE, "CubeCore");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsST, init_fail_case2) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "CubeCore");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsST, init_fail_case3) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "0");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsST, init_fail_case4) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "-1");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsST, init_fail_case5) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "2");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsST, init_fail_case6) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::SOC_VERSION, "Ascend000");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}
