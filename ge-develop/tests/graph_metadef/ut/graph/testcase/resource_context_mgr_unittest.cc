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
#include "graph/resource_context_mgr.h"
#include "graph_builder_utils.h"

namespace ge {
namespace {
  struct TestResourceContext : ResourceContext {
    std::vector<GeShape> shapes;
    std::string resource_type;
  };
}

class ResourceInferenceContextMgrTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(ResourceInferenceContextMgrTest, TestSetAndGetResourceContext) {
  // prepare resource_context
  string resource_key = "123";
  std::vector<GeShape> resource_shapes = {GeShape({1,1,2,3})};
  TestResourceContext *resource_context = new TestResourceContext();
  resource_context->shapes = resource_shapes;
  resource_context->resource_type = "normal";

  // set resource_context to mgr
  ResourceContextMgr resource_context_mgr;
  resource_context_mgr.SetResourceContext(resource_key, resource_context);

  TestResourceContext *test_resource_context =
      dynamic_cast<TestResourceContext *>(resource_context_mgr.GetResourceContext(resource_key));
  // check result
  auto ret_shape = test_resource_context->shapes.at(0);
  auto ret_type = test_resource_context->resource_type;
  ASSERT_EQ(ret_shape.GetDims(), resource_context->shapes.at(0).GetDims());
  ASSERT_EQ(ret_type, resource_context->resource_type);
}

TEST_F(ResourceInferenceContextMgrTest, TestRegsiterNodesReliedOnResource) {
  string resource_key = "123";
  auto builder = ut::GraphBuilder("g");

  auto read_node_1 = builder.AddNode("stackpop", "stackPop", 1, 1);
  auto read_node_2 = builder.AddNode("tensorAarrayRead", "TensorArrayRead", 1, 1);
  ResourceContextMgr resource_context_mgr;
  // register one node
  resource_context_mgr.RegisterNodeReliedOnResource(resource_key, read_node_1);
  auto read_nodes = resource_context_mgr.MutableNodesReliedOnResource(resource_key);
  ASSERT_EQ(read_nodes.size(), 1);

  // register second node
  resource_context_mgr.RegisterNodeReliedOnResource(resource_key, read_node_2);
  read_nodes = resource_context_mgr.MutableNodesReliedOnResource(resource_key);
  ASSERT_EQ(read_nodes.size(), 2);
  vector<NodePtr> expect_read_nodes = {read_node_1, read_node_2};
  for (const auto &expect_node : expect_read_nodes) {
    ASSERT_TRUE(read_nodes.count(expect_node) > 0);
  }
}

TEST_F(ResourceInferenceContextMgrTest, TestRegsiterDuplicateNodeReliedOnResource) {
  string resource_key = "123";
  auto builder = ut::GraphBuilder("g");

  auto read_node = builder.AddNode("stack", "stack", 1, 1);
  ResourceContextMgr resource_context_mgr;
  resource_context_mgr.RegisterNodeReliedOnResource(resource_key, read_node);
  // check add same node to context
  resource_context_mgr.RegisterNodeReliedOnResource(resource_key, read_node);
  auto read_nodes = resource_context_mgr.MutableNodesReliedOnResource(resource_key);
  ASSERT_EQ(read_nodes.size(), 1);
}
} // namespace ge
