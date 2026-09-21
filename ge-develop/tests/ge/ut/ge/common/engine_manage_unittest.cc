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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "engines/manager/engine/engine_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestEngineManage : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestEngineManage, Normal) {
    EXPECT_NO_THROW(auto p1 = std::make_shared<EngineManager>());
}

TEST_F(UtestEngineManage, Abnormal) {
    auto p1 = std::make_shared<EngineManager>();
    std::map<std::string, DNNEnginePtr> engines_map;
    GetDNNEngineObjs(engines_map);
    const std::string fake_engine = "fake_engine";
    const std::string ai_core = "AIcoreEngine";
    DNNEnginePtr fake_engine_ptr = nullptr;

    Status status;
    (void)EngineManager::GetEngine(fake_engine);
    DNNEnginePtr aicore_engine_ptr = EngineManager::GetEngine(ai_core);
    status = EngineManager::RegisterEngine(ai_core, aicore_engine_ptr);
    EXPECT_EQ(status, ge::FAILED);
    status = EngineManager::RegisterEngine(fake_engine, fake_engine_ptr);
    EXPECT_EQ(status, ge::FAILED);
}

} // namespace ge
