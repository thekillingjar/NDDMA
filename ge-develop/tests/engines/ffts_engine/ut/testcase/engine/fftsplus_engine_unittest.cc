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
#include <iostream>
#include <list>
#include "ffts_llt_utils.h"
#define private public
#define protected public
#include "engine/fftsplus_engine.h"
#include "engine/engine_manager.h"
#include "inc/ffts_configuration.h"
#include "inc/ffts_utils.h"
#include "inc/ffts_type.h"
#undef private
#undef protected

using namespace std;
using namespace ffts;
using namespace ge;

class FFTSPlusEngineUTEST : public testing::Test {
 protected:
	void SetUp() {
	}

	void TearDown() {
	}
};

TEST_F(FFTSPlusEngineUTEST, Initialize_failed) {
  map<string, string> options;
  Status ret = Initialize(options);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(FFTSPlusEngineUTEST, Initialize_success) {
  map<string, string> options;
  options["ge.socVersion"] = "soc_version_";
  Configuration &config = Configuration::Instance();
  config.is_init_ = true;
  Status ret = Initialize(options);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusEngineUTEST, Finalize_success) {
  Status ret = Finalize();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusEngineUTEST, GetCompositeEngines_success1) {
  std::map<string, std::set<std::string>> compound_engine_contains;
  std::map<std::string, std::string> compound_engine_2_kernel_lib_name;
  GetCompositeEngines(compound_engine_contains, compound_engine_2_kernel_lib_name);
  EXPECT_EQ(0, compound_engine_2_kernel_lib_name.size());
}

TEST_F(FFTSPlusEngineUTEST, GetCompositeEngines_success2) {
  std::map<string, std::set<std::string>> compound_engine_contains;
  std::map<std::string, std::string> compound_engine_2_kernel_lib_name;
  std::string kFFTSPlusCoreName = "ffts_plus";
  string pre_soc_version = EngineManager::Instance(kFFTSPlusCoreName).GetSocVersion();
  EngineManager::Instance(kFFTSPlusCoreName).soc_version_ = "ascend910b2";

  std::string path = GetCodeDir() + "/tests/engines/ffts_engine/config/data/platform_config";
  std::string real_path = RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);

  std::map<std::string, std::vector<std::string>> a;
  std::map<std::string, std::vector<std::string>> b;
  a = b;
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos opti_compilation_infos;
  std::map<std::string, fe::PlatFormInfos> platform_infos_map;
  platform_infos.Init();
  std::map<std::string, std::string> res;
  res[kFFTSMode] = kFFTSPlus;
  platform_infos.SetPlatformRes(kSocInfo, res);
  platform_infos_map["ascend910b2"] = platform_infos;
  fe::PlatformInfoManager::Instance().platform_infos_map_ = platform_infos_map;
  std::map<std::string, OpsKernelInfoStorePtr> op_kern_infos;
  GetOpsKernelInfoStores(op_kern_infos);
  GetCompositeEngines(compound_engine_contains, compound_engine_2_kernel_lib_name);
  bool ffts_flag = false;
  ffts::Configuration::Instance().SetFftsOptimizeEnable();
  GetFFTSPlusSwitch(ffts_flag);
  EXPECT_EQ(true, ffts_flag);
  EngineManager::Instance(kFFTSPlusCoreName).soc_version_ = pre_soc_version;
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  EXPECT_EQ(1, compound_engine_2_kernel_lib_name.size());
}

TEST_F(FFTSPlusEngineUTEST, GetFFTSPlusSwitch_1) {
  bool ffts_flag = false;
  ffts::Configuration::Instance().SetFftsOptimizeEnable();
  GetFFTSPlusSwitch(ffts_flag);
  EXPECT_EQ(false, ffts_flag);
}
