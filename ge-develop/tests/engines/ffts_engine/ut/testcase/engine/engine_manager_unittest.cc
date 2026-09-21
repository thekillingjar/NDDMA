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
#include "engine/engine_manager.h"
#include "inc/ffts_configuration.h"
#undef private
#undef protected

using namespace std;
using namespace ffts;
using namespace ge;

class EngineManagerUTEST : public testing::Test {
 protected:
	void SetUp() {
	}

	void TearDown() {
	}
};

TEST_F(EngineManagerUTEST, Initialize_success1) {
  EngineManager em;
  map<string, string> options;
  std::string engine_name;
  std::string soc_version;
  Configuration &config = Configuration::Instance();
  config.is_init_ = true;
  Status ret = em.Initialize(options, engine_name, soc_version);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(EngineManagerUTEST, Initialize_success2) {
  EngineManager em;
  map<string, string> options;
  std::string engine_name;
  std::string soc_version;
  em.inited_ = true;
  Status ret = em.Initialize(options, engine_name, soc_version);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(EngineManagerUTEST, Finalize_success1) {
  EngineManager em;
  Status ret = em.Finalize();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(EngineManagerUTEST, Finalize_success2) {
  EngineManager em;
  em.inited_ = true;
  Status ret = em.Finalize();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(EngineManagerUTEST, GetGraphOptimizerObjs_success1) {
  EngineManager em;
  map<string, GraphOptimizerPtr> graph_optimizers;
  std::string engine_name;
  em.GetGraphOptimizerObjs(graph_optimizers, engine_name);
  EXPECT_EQ(0, graph_optimizers.size());
}