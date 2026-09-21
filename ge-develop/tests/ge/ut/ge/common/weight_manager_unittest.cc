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
#include "common/host_resource_center/weight_manager.h"
#include "common/host_resource_center/host_resource_center.h"
#include "common/share_graph.h"

namespace ge {
class WeightManagerUT : public testing::Test {};
TEST_F(WeightManagerUT, TakeoverResourceSuccess_SimpleGraph) {
  WeightManager weight_manager;
  auto graph = gert::ShareGraph::BuildCtrlToConstGraph();
  auto const_node = graph->FindFirstNodeMatchType("Const");

  // takeover reousrce on graph
  for (const auto &node : graph->GetAllNodes()) {
    EXPECT_EQ(weight_manager.TakeoverResources(node->GetOpDesc()), SUCCESS);
  }

  // try find resource
  auto host_resource_ptr = weight_manager.GetResource(const_node->GetOpDesc(), 0);
  EXPECT_NE(host_resource_ptr, nullptr);

  // check
  auto weight_resource_ptr = dynamic_cast<const WeightResource *>(host_resource_ptr);
  EXPECT_NE(weight_resource_ptr, nullptr);
  auto weight_ptr = weight_resource_ptr->GetWeight();
  EXPECT_NE(weight_ptr, nullptr);
  EXPECT_EQ(weight_ptr->GetTensorDesc().GetDataType(), DT_INT32);

  float *weight_data = (float *)weight_ptr->GetData().data();
  std::vector<int32_t> expect_output = {0};
  for (size_t i = 0U; i < expect_output.size(); ++i) {
    EXPECT_EQ(weight_data[i], expect_output[i]);
  }
  std::shared_ptr<HostResource> dummy;
  EXPECT_NE(weight_manager.AddResource(const_node->GetOpDesc(), 0, dummy), SUCCESS);
}

TEST_F(WeightManagerUT, TakeoverResourceSuccess_WithFunctionNode) {
  WeightManager weight_mananger;
  auto graph = gert::ShareGraph::BuildDynamicAndStaticGraph();
  EXPECT_EQ(graph->TopologicalSorting(), SUCCESS);

  NodePtr sub_const1;
  NodePtr sub_const2;
  // takeover reousrce on graph
  for (const auto &node : graph->GetAllNodes()) {
    EXPECT_EQ(weight_mananger.TakeoverResources(node->GetOpDesc()), SUCCESS);
    if (node->GetName() == "const_1") {
      sub_const1 = node;
    }
    if (node->GetName() == "const_2") {
      sub_const2 = node;
    }
  }

  EXPECT_NE(sub_const1, nullptr);
  EXPECT_NE(sub_const2, nullptr);
  // try find resource
  auto sub_const1_host_resource_ptr = weight_mananger.GetResource(sub_const1->GetOpDesc(), 0);
  EXPECT_NE(sub_const1_host_resource_ptr, nullptr);
  auto sub_const2_host_resource_ptr = weight_mananger.GetResource(sub_const2->GetOpDesc(), 0);
  EXPECT_NE(sub_const2_host_resource_ptr, nullptr);
}

TEST_F(WeightManagerUT, TakeoverResource_WithDuplicateKey_SUCCESS) {
  WeightManager weight_mananger;
  auto graph = gert::ShareGraph::BuildDynamicAndStaticGraph();

  NodePtr sub_const1;
  NodePtr sub_const2;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "const_1") {
      sub_const1 = node;
      sub_const1->GetOpDesc()->SetId(0);
    }
    if (node->GetName() == "const_2") {
      sub_const2 = node;
      sub_const2->GetOpDesc()->SetId(0);
    }
  }
  EXPECT_NE(sub_const1, nullptr);
  EXPECT_NE(sub_const2, nullptr);

  // takeover reousrce on graph
  EXPECT_EQ(weight_mananger.TakeoverResources(sub_const1->GetOpDesc()), SUCCESS);
  EXPECT_EQ(weight_mananger.TakeoverResources(sub_const2->GetOpDesc()), SUCCESS);
}

TEST_F(WeightManagerUT, GetResourceBeforeTakeover_GotNullptr) {
  WeightManager weight_mananger;
  auto graph = gert::ShareGraph::BuildCtrlToConstGraph();
  auto const_node = graph->FindFirstNodeMatchType("Const");

  // try find resource without takeover
  auto host_resource_ptr = weight_mananger.GetResource(const_node->GetOpDesc(), 0);
  EXPECT_EQ(host_resource_ptr, nullptr);
}

TEST_F(WeightManagerUT, TakeoverResourceSuccess_GetHostResourceMgr) {
  HostResourceCenter center;
  auto graph = gert::ShareGraph::BuildCtrlToConstGraph();
  auto const_node = graph->FindFirstNodeMatchType("Const");

  EXPECT_EQ(center.TakeOverHostResources(graph), SUCCESS);

  auto weight_manager = center.GetHostResourceMgr(HostResourceType::kWeight);
  auto host_resource_ptr = weight_manager->GetResource(const_node->GetOpDesc(), 0);
  EXPECT_NE(host_resource_ptr, nullptr);

  // check
  auto weight_resource_ptr = dynamic_cast<const WeightResource *>(host_resource_ptr);
  EXPECT_NE(weight_resource_ptr, nullptr);
  auto weight_ptr = weight_resource_ptr->GetWeight();
  EXPECT_NE(weight_ptr, nullptr);
  EXPECT_EQ(weight_ptr->GetTensorDesc().GetDataType(), DT_INT32);

  float *weight_data = (float *)weight_ptr->GetData().data();
  std::vector<int32_t> expect_output = {0};
  for (size_t i = 0U; i < expect_output.size(); ++i) {
    EXPECT_EQ(weight_data[i], expect_output[i]);
  }
}
}  // namespace ge
