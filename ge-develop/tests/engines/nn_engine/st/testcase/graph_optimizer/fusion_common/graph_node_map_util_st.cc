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
#include <string>
#include <vector>

#define protected public
#define private public
#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class GRAPH_NODE_MAP_UTIL_STEST: public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(GRAPH_NODE_MAP_UTIL_STEST, ReCreateNodeTypeMapInGraph_suc1) {
  GraphNodeMapUtil graph_node_map_util;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_0 = graph->AddNode(op_desc_0);
  Status ret = graph_node_map_util.ReCreateNodeTypeMapInGraph(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}
