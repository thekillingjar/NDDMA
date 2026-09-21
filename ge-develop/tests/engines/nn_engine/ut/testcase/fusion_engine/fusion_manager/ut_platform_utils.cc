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

#include "fe_llt_utils.h"

using namespace std;
using namespace fe;

class PlatFormUtilsUT: public testing::Test
{
 protected:
  static void SetUpTestCase() {
    std::cout << "PlatFormUtilsUT SetUpTestCase" << std::endl;
    InitPlatformInfo("Ascend910A");
  }
  static void TearDownTestCase() {
    std::cout << "PlatFormUtilsUT TearDownTestCase" << std::endl;
  }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(PlatFormUtilsUT, init_success_case_cloud)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "10");
  options.emplace(ge::BUFFER_OPTIMIZE, "l2_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend910B");
  Status status = instance.Initialize(options);
  ASSERT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend910B");
  EXPECT_EQ(instance.GetShortSocVersion(), "Ascend910");
  EXPECT_EQ(instance.GetIsaArchVersion(), ISAArchVersion::EN_ISA_ARCH_V100);
  EXPECT_EQ(instance.GetIsaArchVersionStr(), "v100");
  EXPECT_EQ(instance.IsSupportFixpipe(), false);
  EXPECT_EQ(instance.IsEnableAllowHF32(), false);
  EXPECT_EQ(instance.IsSupportContextSwitch(), false);
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), false);
  EXPECT_EQ(instance.GetDsaWorkspaceSize(), 0);
  EXPECT_EQ(instance.GetCubeVecState(), CubeVecStateNew::CUBE_VEC_UNKNOWN);
  EXPECT_EQ(instance.GetFftsMode(), AICoreMode::FFTS_MODE_NO_FFTS);
  EXPECT_EQ(instance.GetL2Type(), L2Type::Cache);
  EXPECT_EQ(instance.IsEnableL2CacheCmo(), false);
  EXPECT_EQ(instance.IsEnableL2CacheRc(), false);
  EXPECT_EQ(instance.IsSupportVectorEngine(), false);
  EXPECT_EQ(instance.IsSpecifiedMemBase(), false);
  EXPECT_EQ(instance.IsDCSoc(), false);
  EXPECT_EQ(instance.GetAICoreNum(), 10);
  EXPECT_EQ(instance.IsHardwareSupportCoreSync(), false);
  std::vector<uint32_t> vir_type_list = {2,4,8,16};
  EXPECT_EQ(instance.GetVirTypeList(), vir_type_list);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  ASSERT_EQ(ret, 0);
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

TEST_F(PlatFormUtilsUT, init_success_case_mini)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend310");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend310");
  EXPECT_EQ(instance.GetShortSocVersion(), "Ascend310");
  EXPECT_EQ(instance.GetIsaArchVersion(), ISAArchVersion::EN_ISA_ARCH_V100);
  EXPECT_EQ(instance.GetIsaArchVersionStr(), "v100");
  EXPECT_EQ(instance.IsSupportFixpipe(), false);
  EXPECT_EQ(instance.IsEnableAllowHF32(), false);
  EXPECT_EQ(instance.IsSupportContextSwitch(), false);
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), false);
  EXPECT_EQ(instance.GetDsaWorkspaceSize(), 0);
  EXPECT_EQ(instance.GetCubeVecState(), CubeVecStateNew::CUBE_VEC_UNKNOWN);
  EXPECT_EQ(instance.GetFftsMode(), AICoreMode::FFTS_MODE_NO_FFTS);
  EXPECT_EQ(instance.GetL2Type(), L2Type::Buff);
  EXPECT_EQ(instance.IsEnableL2CacheCmo(), false);
  EXPECT_EQ(instance.IsEnableL2CacheRc(), false);
  EXPECT_EQ(instance.IsSupportVectorEngine(), false);
  EXPECT_EQ(instance.IsSpecifiedMemBase(), false);
  EXPECT_EQ(instance.IsDCSoc(), false);
  EXPECT_EQ(instance.GetAICoreNum(), 2);
  EXPECT_EQ(instance.IsHardwareSupportCoreSync(), false);
  std::vector<uint32_t> vir_type_list = {};
  EXPECT_EQ(instance.GetVirTypeList(), vir_type_list);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_infos.GetSocVersion(), "Ascend310");
  EXPECT_EQ(optional_infos.GetAICoreNum(), 2);
  EXPECT_EQ(optional_infos.GetCoreType(), "AiCore");
  EXPECT_EQ(optional_infos.GetL1FusionFlag(), "true");

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_info.soc_version, "Ascend310");
  EXPECT_EQ(optional_info.ai_core_num, 2);
  EXPECT_EQ(optional_info.core_type, "AiCore");
  EXPECT_EQ(optional_info.l1_fusion_flag, "true");
}

TEST_F(PlatFormUtilsUT, init_success_case_ascend310p)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::CORE_TYPE, "VectorCore");
  options.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend310P1");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend310P1");
  EXPECT_EQ(instance.GetShortSocVersion(), "Ascend310P");
  EXPECT_EQ(instance.GetIsaArchVersion(), ISAArchVersion::EN_ISA_ARCH_V200);
  EXPECT_EQ(instance.GetIsaArchVersionStr(), "v200");
  EXPECT_EQ(instance.IsSupportFixpipe(), false);
  EXPECT_EQ(instance.IsEnableAllowHF32(), false);
  EXPECT_EQ(instance.IsSupportContextSwitch(), false);
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), false);
  EXPECT_EQ(instance.GetDsaWorkspaceSize(), 0);
  EXPECT_EQ(instance.GetCubeVecState(), CubeVecStateNew::CUBE_VEC_UNKNOWN);
  EXPECT_EQ(instance.GetFftsMode(), AICoreMode::FFTS_MODE_NO_FFTS);
  EXPECT_EQ(instance.GetL2Type(), L2Type::Cache);
  EXPECT_EQ(instance.IsEnableL2CacheCmo(), false);
  EXPECT_EQ(instance.IsEnableL2CacheRc(), false);
  EXPECT_EQ(instance.IsSupportVectorEngine(), true);
  EXPECT_EQ(instance.IsSpecifiedMemBase(), false);
  EXPECT_EQ(instance.IsDCSoc(), true);
  EXPECT_EQ(instance.GetAICoreNum(), 10);
  EXPECT_EQ(instance.IsHardwareSupportCoreSync(), false);
  std::vector<uint32_t> vir_type_list = {1,2,4,8};
  EXPECT_EQ(instance.GetVirTypeList(), vir_type_list);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_infos.GetSocVersion(), "Ascend310P1");
  EXPECT_EQ(optional_infos.GetAICoreNum(), 10);
  EXPECT_EQ(optional_infos.GetCoreType(), "VectorCore");
  EXPECT_EQ(optional_infos.GetL1FusionFlag(), "true");

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_info.soc_version, "Ascend310P1");
  EXPECT_EQ(optional_info.ai_core_num, 10);
  EXPECT_EQ(optional_info.core_type, "VectorCore");
  EXPECT_EQ(optional_info.l1_fusion_flag, "true");
}

TEST_F(PlatFormUtilsUT, init_success_case_ascend910b)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend910B1");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend910B1");
  EXPECT_EQ(instance.GetShortSocVersion(), "Ascend910B");
  EXPECT_EQ(instance.GetIsaArchVersion(), ISAArchVersion::EN_ISA_ARCH_V220);
  EXPECT_EQ(instance.GetIsaArchVersionStr(), "v220");
  EXPECT_EQ(instance.IsSupportFixpipe(), true);
  EXPECT_EQ(instance.IsEnableAllowHF32(), true);
  EXPECT_EQ(instance.IsSupportContextSwitch(), false);
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), true);
  EXPECT_EQ(instance.GetDsaWorkspaceSize(), 48);
  EXPECT_EQ(instance.GetCubeVecState(), CubeVecStateNew::CUBE_VEC_SPLIT);
  EXPECT_EQ(instance.GetFftsMode(), AICoreMode::FFTS_MODE_FFTS_PLUS);
  EXPECT_EQ(instance.GetL2Type(), L2Type::Cache);
  EXPECT_EQ(instance.IsEnableL2CacheCmo(), false);
  EXPECT_EQ(instance.IsEnableL2CacheRc(), false);
  EXPECT_EQ(instance.IsSupportVectorEngine(), false);
  EXPECT_EQ(instance.IsSpecifiedMemBase(), false);
  EXPECT_EQ(instance.IsDCSoc(), false);
  EXPECT_EQ(instance.GetAICoreNum(), 24);
  EXPECT_EQ(instance.IsHardwareSupportCoreSync(), true);
  std::vector<uint32_t> vir_type_list = {2,3,6,12,24};
  EXPECT_EQ(instance.GetVirTypeList(), vir_type_list);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_infos.GetSocVersion(), "Ascend910B1");
  EXPECT_EQ(optional_infos.GetAICoreNum(), 24);
  EXPECT_EQ(optional_infos.GetCoreType(), "AiCore");
  EXPECT_EQ(optional_infos.GetL1FusionFlag(), "true");

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_info.soc_version, "Ascend910B1");
  EXPECT_EQ(optional_info.ai_core_num, 24);
  EXPECT_EQ(optional_infos.GetCoreType(), "AiCore");
  EXPECT_EQ(optional_info.l1_fusion_flag, "true");
}

TEST_F(PlatFormUtilsUT, init_success_case_ascend310b)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend310B1");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend310B1");
  EXPECT_EQ(instance.GetShortSocVersion(), "Ascend310B");
  EXPECT_EQ(instance.GetIsaArchVersion(), ISAArchVersion::EN_ISA_ARCH_V300);
  EXPECT_EQ(instance.GetIsaArchVersionStr(), "v300");
  EXPECT_EQ(instance.IsSupportFixpipe(), true);
  EXPECT_EQ(instance.IsEnableAllowHF32(), true);
  EXPECT_EQ(instance.IsSupportContextSwitch(), false);
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), true);
  EXPECT_EQ(instance.GetDsaWorkspaceSize(), 0);
  EXPECT_EQ(instance.GetCubeVecState(), CubeVecStateNew::CUBE_VEC_FUSE);
  EXPECT_EQ(instance.GetFftsMode(), AICoreMode::FFTS_MODE_NO_FFTS);
  EXPECT_EQ(instance.GetL2Type(), L2Type::Cache);
  EXPECT_EQ(instance.IsEnableL2CacheCmo(), true);
  EXPECT_EQ(instance.IsEnableL2CacheRc(), false);
  EXPECT_EQ(instance.IsSupportVectorEngine(), false);
  EXPECT_EQ(instance.IsSpecifiedMemBase(), false);
  EXPECT_EQ(instance.IsDCSoc(), false);
  EXPECT_EQ(instance.GetAICoreNum(), 1);
  EXPECT_EQ(instance.IsHardwareSupportCoreSync(), false);
  std::vector<uint32_t> vir_type_list = {};
  EXPECT_EQ(instance.GetVirTypeList(), vir_type_list);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_infos.GetSocVersion(), "Ascend310B1");
  EXPECT_EQ(optional_infos.GetAICoreNum(), 1);
  EXPECT_EQ(optional_infos.GetCoreType(), "AiCore");
  EXPECT_EQ(optional_infos.GetL1FusionFlag(), "true");

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_info.soc_version, "Ascend310B1");
  EXPECT_EQ(optional_info.ai_core_num, 1);
  EXPECT_EQ(optional_info.core_type, "AiCore");
  EXPECT_EQ(optional_info.l1_fusion_flag, "true");
}

TEST_F(PlatFormUtilsUT, init_success_case_nano)
{
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(instance.GetSocVersion(), "Ascend035");
  EXPECT_EQ(instance.GetShortSocVersion(), "Ascend035");
  EXPECT_EQ(instance.GetIsaArchVersion(), ISAArchVersion::EN_ISA_ARCH_V350);
  EXPECT_EQ(instance.GetIsaArchVersionStr(), "v350");
  EXPECT_EQ(instance.IsSupportFixpipe(), true);
  EXPECT_EQ(instance.IsEnableAllowHF32(), false);
  EXPECT_EQ(instance.IsSupportContextSwitch(), true);
  EXPECT_EQ(instance.IsEnableCubeHighPrecision(), false);
  EXPECT_EQ(instance.GetDsaWorkspaceSize(), 0);
  EXPECT_EQ(instance.GetCubeVecState(), CubeVecStateNew::CUBE_VEC_UNKNOWN);
  EXPECT_EQ(instance.GetFftsMode(), AICoreMode::FFTS_MODE_NO_FFTS);
  EXPECT_EQ(instance.GetL2Type(), L2Type::Cache);
  EXPECT_EQ(instance.IsEnableL2CacheCmo(), false);
  EXPECT_EQ(instance.IsEnableL2CacheRc(), false);
  EXPECT_EQ(instance.IsSupportVectorEngine(), false);
  EXPECT_EQ(instance.IsSpecifiedMemBase(), true);
  EXPECT_EQ(instance.IsDCSoc(), false);
  EXPECT_EQ(instance.GetAICoreNum(), 1);
  EXPECT_EQ(instance.IsHardwareSupportCoreSync(), false);
  std::vector<uint32_t> vir_type_list = {};
  EXPECT_EQ(instance.GetVirTypeList(), vir_type_list);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  uint32_t ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_infos.GetSocVersion(), "Ascend035");
  EXPECT_EQ(optional_infos.GetAICoreNum(), 1);
  EXPECT_EQ(optional_infos.GetCoreType(), "AiCore");
  EXPECT_EQ(optional_infos.GetL1FusionFlag(), "true");

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  ret = PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(optional_info.soc_version, "Ascend035");
  EXPECT_EQ(optional_info.ai_core_num, 1);
  EXPECT_EQ(optional_info.core_type, "AiCore");
  EXPECT_EQ(optional_info.l1_fusion_flag, "true");
}

TEST_F(PlatFormUtilsUT, init_fail_case1) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::CORE_TYPE, "CubeCore");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsUT, init_fail_case2) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "CubeCore");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsUT, init_fail_case3) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "0");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsUT, init_fail_case4) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "-1");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsUT, init_fail_case5) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::AICORE_NUM, "2");
  options.emplace(ge::SOC_VERSION, "Ascend035");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsUT, init_fail_case6) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::SOC_VERSION, "Ascend000");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(PlatFormUtilsUT, init_fail_case7) {
  PlatformUtils instance;
  std::map<std::string, std::string> options;
  options.emplace(ge::SOC_VERSION, "Ascend910B");
  options.emplace(ge::CORE_TYPE, "AAAAC");
  Status status = instance.Initialize(options);
  EXPECT_EQ(status, FAILED);
}
