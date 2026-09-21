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
#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_info_store.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {
namespace ge_local{

class UtestGeLocalOpsKernelInfo : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestGeLocalOpsKernelInfo, Normal) {
  auto p = std::make_shared<GeLocalOpsKernelInfoStore>();
  const std::map<std::string, std::string> options = {};
  EXPECT_EQ(p->Initialize(options), SUCCESS);
  EXPECT_EQ(p->CreateSession(options), SUCCESS);
  EXPECT_EQ(p->DestroySession(options), SUCCESS);
  std::map<std::string, ge::OpInfo> infos;
  p->GetAllOpsKernelInfo(infos);
  const auto opdesc = std::make_shared<OpDesc>();
  OpDescPtr opdesc1 = nullptr;
  std::string reason = "test";
  EXPECT_EQ(p->CheckSupported(opdesc1, reason), false);
  EXPECT_EQ(p->CheckSupported(opdesc, reason), false);
  auto node = make_shared<ge::Node>();
  EXPECT_EQ(p->SetCutSupportedInfo(node), SUCCESS);
  EXPECT_EQ(p->Finalize(), SUCCESS);
}

}
} // namespace ge
