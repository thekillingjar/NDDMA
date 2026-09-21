/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/data_dependent_interpreter.h"
#include <gtest/gtest.h>
#include "graph/node.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "register/op_impl_registry.h"
#include "faker/node_faker.h"
#include "faker/space_registry_faker.h"
#include "common/checker.h"

namespace gert {
namespace {
// todo 把注册做成stub的庄能力，不影响其他流程
IMPL_OP(DDIT002).InputsDataDependency({0, 2});
IMPL_OP(DDIT1).InputsDataDependency({1});
IMPL_OP(DDIT003).TilingInputsDataDependency({1, 2});
IMPL_OP(DDIT4);
IMPL_OP(DDIT5).TilingInputsDataDependency({1, 2}, {TilingPlacement::TILING_ON_AICPU});

/*
 * ub graph:
 *
 *     NetOutput
 *        |
 *       Foo
 *        |
 *      DDIT002
 *     /    |   \
 * data0 data1 data2
 */
ge::NodePtr FakeUbNode02() {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto ub_graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data0", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 0)
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data1", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 1)
                   .IoNum(0, 1)
                   .Build();
  auto data2 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data2", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 2)
                   .IoNum(0, 1)
                   .Build();

  auto ddit02 =
      ComputeNodeFaker(ub_graph).NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ddit02->GetOpDesc()->SetOpInferDepends({"x", "z"});

  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), ddit02->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), ddit02->GetInDataAnchor(1)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), ddit02->GetInDataAnchor(2)));

  auto foo = ComputeNodeFaker(ub_graph).NameAndType("Foo", "Foo").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(ddit02->GetOutDataAnchor(0), foo->GetInDataAnchor(0)));

  auto netoutput = ComputeNodeFaker(ub_graph).NameAndType("NetOutput", "NetOutput").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(foo->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0)));

  GE_DUMP(ub_graph, "TestUbGraph");

  GE_ASSERT_TRUE(ge::AttrUtils::SetGraph(node->GetOpDesc(), "_original_fusion_graph", ub_graph));

  return node;
}

/*
 * ub graph:
 *
 *     NetOutput
 *        |
 *       Foo
 *        |
 *      ddit02
 *     /    |   \
 * const0 data1 data2
 */
ge::NodePtr FakeUbNode02OnlyHasTwoData() {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto ub_graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data2 = ComputeNodeFaker(ub_graph)
                   .NameAndType("const0", "Const")
                   .IoNum(0, 1)
                   .Build();
  auto data0 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data1", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 0)
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data2", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 1)
                   .IoNum(0, 1)
                   .Build();

  auto ddit02 =
      ComputeNodeFaker(ub_graph).NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ddit02->GetOpDesc()->SetOpInferDepends({"x", "z"});

  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), ddit02->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), ddit02->GetInDataAnchor(1)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), ddit02->GetInDataAnchor(2)));

  auto foo = ComputeNodeFaker(ub_graph).NameAndType("Foo", "Foo").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(ddit02->GetOutDataAnchor(0), foo->GetInDataAnchor(0)));

  auto netoutput = ComputeNodeFaker(ub_graph).NameAndType("NetOutput", "NetOutput").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(foo->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0)));

  GE_DUMP(ub_graph, "TestUbGraph");

  GE_ASSERT_TRUE(ge::AttrUtils::SetGraph(node->GetOpDesc(), "_original_fusion_graph", ub_graph));

  return node;
}
/*
 * ub graph:
 *
 *     NetOutput
 *        |
 *       Foo
 *        |
 *      ddit02
 *     /    |   \
 * data0 data1 data2
 */
ge::NodePtr FakeUbNode02DataDoesNotHasIndex() {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto ub_graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data0", "Data")
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data1", "Data")
                   .IoNum(0, 1)
                   .Build();
  auto data2 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data2", "Data")
                   .IoNum(0, 1)
                   .Build();

  auto ddit02 =
      ComputeNodeFaker(ub_graph).NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ddit02->GetOpDesc()->SetOpInferDepends({"x", "z"});

  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), ddit02->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), ddit02->GetInDataAnchor(1)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), ddit02->GetInDataAnchor(2)));

  auto foo = ComputeNodeFaker(ub_graph).NameAndType("Foo", "Foo").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(ddit02->GetOutDataAnchor(0), foo->GetInDataAnchor(0)));

  auto netoutput = ComputeNodeFaker(ub_graph).NameAndType("NetOutput", "NetOutput").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(foo->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0)));

  GE_DUMP(ub_graph, "TestUbGraph");

  GE_ASSERT_TRUE(ge::AttrUtils::SetGraph(node->GetOpDesc(), "_original_fusion_graph", ub_graph));

  return node;
}
/*
 * ub graph:
 *
 *     NetOutput
 *        |
 *       Foo
 *        |
 *      ddit02
 *     /    |   \
 * data0 data1 data2
 */
ge::NodePtr FakeUbNode02TypeFoo() {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "Foo").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();

  auto ub_graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data0", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 0)
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data1", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 1)
                   .IoNum(0, 1)
                   .Build();
  auto data2 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data2", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 2)
                   .IoNum(0, 1)
                   .Build();

  auto ddit02 =
      ComputeNodeFaker(ub_graph).NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ddit02->GetOpDesc()->SetOpInferDepends({"x", "z"});

  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), ddit02->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), ddit02->GetInDataAnchor(1)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), ddit02->GetInDataAnchor(2)));

  auto foo = ComputeNodeFaker(ub_graph).NameAndType("Foo", "Foo").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(ddit02->GetOutDataAnchor(0), foo->GetInDataAnchor(0)));

  auto netoutput = ComputeNodeFaker(ub_graph).NameAndType("NetOutput", "NetOutput").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(foo->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0)));

  GE_DUMP(ub_graph, "TestUbGraph");

  GE_ASSERT_TRUE(ge::AttrUtils::SetGraph(node->GetOpDesc(), "_original_fusion_graph", ub_graph));

  return node;
}
/*
 * ub graph:
 *
 *     NetOutput
 *        |
 *       Foo
 *        |
 *       ddit1
 *     /    |   \
 * data0 data2 data1
 */
ge::NodePtr FakeUbNode1() {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "DDIT1").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  node->GetOpDesc()->SetOpInferDepends({"y"});

  auto ub_graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data0", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 0)
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data1", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 1)
                   .IoNum(0, 1)
                   .Build();
  auto data2 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data2", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 2)
                   .IoNum(0, 1)
                   .Build();

  auto ddit02 = ComputeNodeFaker(ub_graph).NameAndType("Test", "DDIT1").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ddit02->GetOpDesc()->SetOpInferDepends({"y"});

  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), ddit02->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), ddit02->GetInDataAnchor(2)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), ddit02->GetInDataAnchor(1)));

  auto foo = ComputeNodeFaker(ub_graph).NameAndType("Foo", "Foo").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(ddit02->GetOutDataAnchor(0), foo->GetInDataAnchor(0)));

  auto netoutput = ComputeNodeFaker(ub_graph).NameAndType("NetOutput", "NetOutput").IoNum(1, 1).Build();
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(foo->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0)));

  GE_DUMP(ub_graph, "TestUbGraph");

  GE_ASSERT_TRUE(ge::AttrUtils::SetGraph(node->GetOpDesc(), "_original_fusion_graph", ub_graph));

  return node;
}
}  // namespace
class DataDependentInterpreterUT : public testing::Test {};
TEST_F(DataDependentInterpreterUT, SimpleNode_ReturnTrue_V2V1True) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto space_registry = SpaceRegistryFaker().Build();
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = space_registry;
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(2, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}

TEST_F(DataDependentInterpreterUT, SimpleNode_ReturnTrue_V2V1True_UseRegistryV2) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto space_registry = SpaceRegistryFaker().Build();
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = space_registry;
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(2, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);

  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(2, ret), ge::GRAPH_SUCCESS); // 兼容
  ASSERT_TRUE(ret);
}

TEST_F(DataDependentInterpreterUT, SimpleNode_TilingDepend_ReturnTrue_V2V1False_UseRegistryV2) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT_error").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  auto space_registry = SpaceRegistryFaker().Build();
  OpImplSpaceRegistryV2Array space_registries;
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = space_registry;
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_BINARY_SOURCE,
                              static_cast<int64_t>(ge::OppImplVersion::kVersionEnd));
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  auto node2 = ComputeNodeFaker().NameAndType("Test", "DDIT4").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node2->GetOpDesc(), space_registries).IsTilingInputDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
}

TEST_F(DataDependentInterpreterUT, SimpleNode_TilingDepend_ReturnTrue_V2V1False) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT_error").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  auto space_registry = SpaceRegistryFaker().Build();
  OpImplSpaceRegistryV2Array space_registries;
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = space_registry;
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_BINARY_SOURCE,
                              static_cast<int64_t>(ge::OppImplVersion::kVersionEnd));
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  auto node2 = ComputeNodeFaker().NameAndType("Test", "DDIT4").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node2->GetOpDesc(), space_registries).IsTilingInputDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
}

TEST_F(DataDependentInterpreterUT, SimpleNode_TilingDepend_ReturnTrue_V2V1) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT003").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().BuildMainSpace();
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(1UL, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(0UL, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsTilingInputDataDependent(2UL, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}

TEST_F(DataDependentInterpreterUT, SimpleNode_TilingDependPlacement_ReturnTrue_V2V1False) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT1").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  auto space_registry = SpaceRegistryFaker().Build();
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = space_registry;
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsSupportTilingDependPlacement(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_BINARY_SOURCE,
                              static_cast<int64_t>(ge::OppImplVersion::kOppKernel));
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsSupportTilingDependPlacement(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  auto node2 = ComputeNodeFaker().NameAndType("Test", "DDIT5").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node2, nullptr);
  // space_registries.fill()
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = nullptr;
  ASSERT_EQ(DataDependentInterpreter(node2->GetOpDesc(), space_registries).IsSupportTilingDependPlacement(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  auto node3 = ComputeNodeFaker().NameAndType("Test", "DDIT_error").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node3, nullptr);
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = space_registry;
  ASSERT_EQ(DataDependentInterpreter(node3->GetOpDesc(), space_registries).IsSupportTilingDependPlacement(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
}

TEST_F(DataDependentInterpreterUT, SimpleNode_TilingDependPlacement_ReturnTrue_V2V1) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT003").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().BuildMainSpace();
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsSupportTilingDependPlacement(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  auto node2 = ComputeNodeFaker().NameAndType("Test", "DDIT5").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node2, nullptr);
  ASSERT_EQ(DataDependentInterpreter(node2->GetOpDesc(), space_registries).IsSupportTilingDependPlacement(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
  ASSERT_EQ(DataDependentInterpreter(node2->GetOpDesc(), space_registries).IsSupportTilingDependPlacement(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}

TEST_F(DataDependentInterpreterUT, SimpleNode_ReturnFalse_V2V1False) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});
  bool ret = true;
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
}
TEST_F(DataDependentInterpreterUT, SimpleNode_ReturnTrueAndLogWarning_V2FalseV1True) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"y"});
  bool ret = false;
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}
TEST_F(DataDependentInterpreterUT, SimpleNode_ReturnTrue_V2TrueV1False) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);

  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().BuildMainSpace();
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(2, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}
TEST_F(DataDependentInterpreterUT, UbGraphNode_ReturnTrue_V2V1UbGraphTrue) {
  auto node = FakeUbNode02();
  ASSERT_NE(node, nullptr);

  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(2, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}
TEST_F(DataDependentInterpreterUT, UbGraphNode_ReturnFalse_V2V1UbGraphFalse) {
  auto node = FakeUbNode02();
  ASSERT_NE(node, nullptr);

  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  bool ret = true;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(ret);
}
TEST_F(DataDependentInterpreterUT, UbGraphNode_ReturnTrueAndLogWarning_V2V1TrueUbGraphFalse) {
  auto node = FakeUbNode1();
  ASSERT_NE(node, nullptr);

  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  bool ret = true;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(1, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}
TEST_F(DataDependentInterpreterUT, UbGraphNode_ReturnTrue_V2V1FalseUbGraphTrue) {
  auto node = FakeUbNode02TypeFoo();
  ASSERT_NE(node, nullptr);

  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  DataDependentInterpreter ddi(node->GetOpDesc(), space_registries);

  bool ret = true;
  ASSERT_EQ(ddi.IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  ret = true;
  ASSERT_EQ(ddi.IsDataDependent(2, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}
TEST_F(DataDependentInterpreterUT, UbGraphNode_Failed_InvalidDataInUbGraph) {
  auto node = FakeUbNode02DataDoesNotHasIndex();
  ASSERT_NE(node, nullptr);
  bool ret;
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  ASSERT_NE(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
}
TEST_F(DataDependentInterpreterUT, UbGraphNode_Failed_DataIndexMissmatch) {
  auto node = FakeUbNode02OnlyHasTwoData();
  ASSERT_NE(node, nullptr);
  bool ret;
  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  ASSERT_NE(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(2, ret), ge::GRAPH_SUCCESS);
}
TEST_F(DataDependentInterpreterUT, SimpleNode_With_EmptyRegistry) {
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT002").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), OpImplSpaceRegistryV2Array()).IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}
TEST_F(DataDependentInterpreterUT, OnlyV1Node_ReturnTrueAndLogWarning_V1True) {
  auto node = ComputeNodeFaker().NameAndType("Test", "FooNotRegister").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  OpImplSpaceRegistryV2Array space_registries;
  space_registries[static_cast<size_t>(ge::OppImplVersion::kOpp)] = SpaceRegistryFaker().Build();
  bool ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(0, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
  ret = false;
  ASSERT_EQ(DataDependentInterpreter(node->GetOpDesc(), space_registries).IsDataDependent(2, ret), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ret);
}
}  // namespace gert
