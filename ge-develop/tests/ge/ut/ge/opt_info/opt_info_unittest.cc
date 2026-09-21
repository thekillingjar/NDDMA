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

#include "macro_utils/dt_public_scope.h"
#include "opt_info/ge_opt_info.h"
#include "graph/ge_local_context.h"
#include "ge/ge_api_types.h"
#include "depends/runtime/src/runtime_stub.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UTEST_opt_info : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTEST_opt_info, get_opt_info_success) {
  std::map<std::string, std::string> options = {{ge::SOC_VERSION, "Ascend910"}};
  GetThreadLocalContext().SetGlobalOption(options);
  auto ret = GeOptInfo::SetOptInfo();
  EXPECT_EQ(ret, ge::SUCCESS);
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  auto itr = graph_options.find("opt_module.fe");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.pass");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.op_tune");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
}

TEST_F(UTEST_opt_info, get_opt_info_all) {
  std::map<std::string, std::string> global_options = {{ge::SOC_VERSION, "Ascend310"}};
  GetThreadLocalContext().SetGlobalOption(global_options);
  auto ret = GeOptInfo::SetOptInfo();
  EXPECT_EQ(ret, ge::SUCCESS);
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  auto itr = graph_options.find("opt_module.fe");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.pass");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.op_tune");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.rl_tune");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.aoe");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
}

TEST_F(UTEST_opt_info, get_opt_info_failed) {
  std::map<std::string, std::string> options;
  GetThreadLocalContext().SetGlobalOption(options);
  auto ret = GeOptInfo::SetOptInfo();
  EXPECT_EQ(ret, ge::FAILED);
}
}  // namespace ge
