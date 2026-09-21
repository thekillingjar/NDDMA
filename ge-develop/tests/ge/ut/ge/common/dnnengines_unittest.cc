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
#include "engines/manager/engine/dnnengines.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestDnnengines : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestDnnengines, Normal) {
    EXPECT_NO_THROW(auto p1 = std::make_shared<AICoreDNNEngine>("name"));
    EXPECT_NO_THROW(auto p2 = std::make_shared<VectorCoreDNNEngine>("name"));
    EXPECT_NO_THROW(auto p3 = std::make_shared<AICpuDNNEngine>("name"));
    EXPECT_NO_THROW(auto p4 = std::make_shared<AICpuTFDNNEngine>("name"));
    EXPECT_NO_THROW(auto p5 = std::make_shared<GeLocalDNNEngine>("name"));
    EXPECT_NO_THROW(auto p6 = std::make_shared<HostCpuDNNEngine>("name"));
    EXPECT_NO_THROW(auto p7 = std::make_shared<RtsDNNEngine>("name"));
    EXPECT_NO_THROW(auto p8 = std::make_shared<HcclDNNEngine>("name"));
    EXPECT_NO_THROW(auto p9 = std::make_shared<FftsPlusDNNEngine>("name"));
    EXPECT_NO_THROW(auto p10 = std::make_shared<DSADNNEngine>("name"));
    EXPECT_NO_THROW(auto rts_fftsplus = std::make_shared<RtsFftsPlusDNNEngine>("RTS_FFTS_PLUS"));
    EXPECT_NO_THROW(auto aicpu_fftsplus = std::make_shared<AICpuFftsPlusDNNEngine>("AICPU_FFTS_PLUS"));
    EXPECT_NO_THROW(auto aicpu_ascend_fftsplus = std::make_shared<AICpuAscendFftsPlusDNNEngine>("AICPU_ASCEND_FFTS_PLUS"));
}



} // namespace ge
