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

#define private public
#define protected public
#include "graph/ge_local_context.h"
#include "engine/fftsplus_engine.h"
#include "engine/engine_manager.h"
#include "inc/ffts_configuration.h"
#include "register/optimization_option_registry.h"
#include "common/aicore_util_constants.h"
#undef private
#undef protected

using namespace std;
using namespace ffts;
using namespace ge;

class FFTSPlusConfigurationSTEST : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(FFTSPlusConfigurationSTEST, Initialize_failed){
  Configuration &config = Configuration::Instance();
  config.is_init_ = false;
  Status ret = config.Initialize();
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(FFTSPlusConfigurationSTEST, Finalize_success1) {
  Configuration &config = Configuration::Instance();
  config.is_init_ = false;
  Status ret = config.Finalize();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusConfigurationSTEST, Finalize_success2) {
  Configuration &config = Configuration::Instance();
  config.is_init_ = true;
  Status ret = config.Finalize();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusConfigurationSTEST, GetFftsOptimizeEnableByEnv_on) {
  Configuration &config = Configuration::Instance();
  config.runmode_ = "";
  setenv("ASCEND_ENHANCE_ENABLE", "1", 1);
  auto res = config.GetFftsOptimizeEnable();
  unsetenv("ASCEND_ENHANCE_ENABLE");
  EXPECT_EQ("ffts", res);
}

TEST_F(FFTSPlusConfigurationSTEST, GetFftsOptimizeEnableByEnv_off) {
  Configuration &config = Configuration::Instance();
  config.runmode_ = "";
  setenv("ASCEND_ENHANCE_ENABLE", "0", 1);
  auto res = config.GetFftsOptimizeEnable();
  unsetenv("ASCEND_ENHANCE_ENABLE");
  EXPECT_EQ("no_ffts", res);
}

TEST_F(FFTSPlusConfigurationSTEST, GetFftsMergeLimitHBMRatio) {
  Configuration &config = Configuration::Instance();
  config.hbm_ratio_ = 0;
  uint32_t hbm_ratio = config.GetFftsMergeLimitHBMRatio();
  EXPECT_EQ(hbm_ratio, 50);

  config.hbm_ratio_ = 0;
  setenv("HBM_RATIO", "0", 1);
  hbm_ratio = config.GetFftsMergeLimitHBMRatio();
  EXPECT_EQ(hbm_ratio, 50);

  config.hbm_ratio_ = 0;
  setenv("HBM_RATIO", "14", 1);
  hbm_ratio = config.GetFftsMergeLimitHBMRatio();
  EXPECT_EQ(hbm_ratio, 14);

  setenv("HBM_RATIO", "abc", 1);
  hbm_ratio = config.GetFftsMergeLimitHBMRatio();
  EXPECT_EQ(hbm_ratio, 14);

  config.hbm_ratio_ = 0;
  hbm_ratio = config.GetFftsMergeLimitHBMRatio();
  EXPECT_EQ(hbm_ratio, 50);

  unsetenv("HBM_RATIO");
}

TEST_F(FFTSPlusConfigurationSTEST, GetFftsEnableWithO1Level1) {
  Configuration &config = Configuration::Instance();
  config.runmode_ = "";
  setenv("ASCEND_ENHANCE_ENABLE", "1", 1);

  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_[fe::kComLevelO1Opt] = fe::kStrFalse;

  auto res = config.GetFftsOptimizeEnable();
  unsetenv("ASCEND_ENHANCE_ENABLE");
  EXPECT_EQ("no_ffts", res);
  (void)ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_.clear();
}