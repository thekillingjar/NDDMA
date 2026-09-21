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
#include "ge/ge_api.h"
#include "graph/compute_graph.h"
#include "framework/common/types.h"
#include "graph/ge_local_context.h"

#include "ge_graph_dsl/graph_dsl.h"

namespace ge {
class STEST_opt_info : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(STEST_opt_info, get_opt_info_all) {
  std::map<std::string, std::string> options = {{ge::SOC_VERSION, "Ascend310"}};
  GetThreadLocalContext().SetGlobalOption(options);

  /**
   *      data1   data2
   *        \    /
   *         add
   */
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->NODE("add"));
  };

  auto graph = ToGeGraph(g1);

  // new session & add graph
  Session session(options);
  auto ret = session.AddGraph(1, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  auto itr = graph_options.find("opt_module.fe");
  ASSERT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.pass");
  ASSERT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.op_tune");
  ASSERT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.rl_tune");
  ASSERT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
  itr = graph_options.find("opt_module.aoe");
  ASSERT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "ALL");
}

TEST_F(STEST_opt_info, get_opt_info_success) {
  std::map<std::string, std::string> options = {{ge::SOC_VERSION, "Ascend910"}};
  GetThreadLocalContext().SetGlobalOption(options);

  /**
   *      data1   data2
   *        \    /
   *         add
   */
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->NODE("add"));
  };

  auto graph = ToGeGraph(g1);

  // new session & add graph
  Session session(options);
  auto ret = session.AddGraph(1, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

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
}  // namespace ge
