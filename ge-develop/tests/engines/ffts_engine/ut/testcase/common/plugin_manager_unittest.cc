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
#include "graph/compute_graph.h"
#include "common/ffts_plugin_manager.h"
#include "inc/ffts_utils.h"
#undef private
#undef protected

using namespace std;
using namespace ffts;

class PluginManagerUTEST : public testing::Test {
 protected:
	void SetUp() {
	}

	void TearDown() {
	}
};

TEST_F(PluginManagerUTEST, Close_Handle_Fail) {
  PluginManager pm("so_name");
  Status ret = pm.CloseHandle();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(PluginManagerUTEST, Close_Handle_Fail_2) {
  PluginManager pm("so_name");
	int a = 1;
	void *h = &a;
	pm.handle = h;
  Status ret = pm.CloseHandle();
}
