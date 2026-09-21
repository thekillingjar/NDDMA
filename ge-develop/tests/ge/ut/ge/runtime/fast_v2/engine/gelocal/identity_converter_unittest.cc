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
#include <memory>
#include "graph/node.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "kernel/tensor_attr.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "graph/utils/graph_utils.h"
#include "register/node_converter_registry.h"
#include "common/const_data_helper.h"
#include "graph/utils/graph_dump_utils.h"

namespace gert {
using namespace ge;
using namespace bg;
LowerResult LoweringIdentityLikeNode(const ge::NodePtr &node, const LowerInput &lower_input);
class IdentityConverterUt : public bg::BgTestAutoCreate3StageFrame {};
namespace {
/*
 *
 * netoutput
 *   |
 * identity
 *   |
 * data1
 */
ComputeGraphPtr BuildIdentityGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("identity", "Identity")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(data1->GetOpDesc(), "index", 0)) {
    return nullptr;
  }
  return graph;
}

/*
 *
 * netoutput
 *   |
 * MemcpyAddrAsync
 *   |
 * const
 */
ComputeGraphPtr BuildMemcpyAddrAsycnGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("memcpy_addr_async", "MemcpyAddrAsync")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(data1->GetOpDesc(), "index", 0)) {
    return nullptr;
  }
  return graph;
}

/*
 *
 * netoutput
 *   |   |
 * identityN
 *   |    \
 * data1  data2
 */
ComputeGraphPtr BuildIdentityNGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->EDGE(0,0)->NODE("identityn", "IdentityN")->EDGE(0,0)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0,1)->NODE("identityn", "IdentityN")->EDGE(1,0)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8,3,224,224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(data1->GetOpDesc(), "index", 0)) {
    return nullptr;
  }

  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8,3,224,224}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8,3,224,224}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(data2->GetOpDesc(), "index", 1)) {
    return nullptr;
  }
  return graph;
}
}

TEST_F(IdentityConverterUt, ConvertIdentity_InputPlacementUnknown_Ok) {
  auto graph = BuildIdentityGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  auto identity = graph->FindNode("identity");

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  address_holder->SetPlacement(kTensorPlacementEnd);
  LowerInput lower_input{{shape_holder}, {address_holder}, &lgd};
  auto ret = LoweringIdentityLikeNode(identity, lower_input);
  ge::DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredIdentityGraph");
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 1);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  // EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodesSize(), 3);
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "MakeSureTensorAtDevice");
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);

  FastNodeTopoChecker checker(ret.out_addrs[0]);
  // stream, allocator, src_addr, tensorsize, src_shape, datatype
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SplitRtStreams", 0},
                                                            {"SelectL2Allocator", 0},
                                                            {"Data", 0},
                                                            {"CalcTensorSizeFromShape", 0},
                                                            {"Data", 0},
                                                            {"Const", 0}}), true),
"success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<FastSrcNode>({{"FreeMemory", 0}})), "success");
}

TEST_F(IdentityConverterUt, ConvertIdentity_InputPlacementDevice_Ok) {
  auto graph = BuildIdentityGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  auto identity = graph->FindNode("identity");

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  address_holder->SetPlacement(kOnDeviceHbm);
  LowerInput lower_input{{shape_holder}, {address_holder}, &lgd};
  auto ret = LoweringIdentityLikeNode(identity, lower_input);
  ge::DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredIdentityGraph");
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 1);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  // EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodesSize(), 3);
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "AllocMemHbm");
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);

  FastNodeTopoChecker checker(ret.out_addrs[0]);
  // src_addr, tensor_size, allocator, stream
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SelectL2Allocator", 0}, {"CalcTensorSizeFromShape", 0}}), true), "success");
}

TEST_F(IdentityConverterUt, ConvertIdentity_InputPlacementHost_Ok) {
  auto graph = BuildIdentityGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  auto identity = graph->FindNode("identity");

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  address_holder->SetPlacement(kOnHost);
  LowerInput lower_input{{shape_holder}, {address_holder}, &lgd};
  auto ret = LoweringIdentityLikeNode(identity, lower_input);
  ge::DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredIdentityGraph");
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 1);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  // EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodesSize(), 3);
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "CopyTensorDataH2H");
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnHost);

  FastNodeTopoChecker checker(ret.out_addrs[0]);
  // src_addr, src_shape, datatype
  EXPECT_EQ(checker.StrictConnectFrom(
                std::vector<FastSrcNode>({{"Data", 0}, {"InferShape", 0}, {"Const", 0}, {"InnerData", 0}})),
            "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<FastSrcNode>({{"FreeMemory", 0}})), "success");
}

TEST_F(IdentityConverterUt, ConvertIdentityN_InputPlacementAllDevice_Ok) {
  auto graph = BuildIdentityNGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  auto identity_n = graph->FindNode("identityn");

  auto shape_holder_0 = ValueHolder::CreateFeed(0);
  auto shape_holder_1 = ValueHolder::CreateFeed(1);
  auto address_holder_0 = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  auto address_holder_1 = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  address_holder_0->SetPlacement(kOnDeviceHbm);
  address_holder_1->SetPlacement(kOnDeviceHbm);
  LowerInput lower_input{{shape_holder_0, shape_holder_1}, {address_holder_0, address_holder_1}, &lgd};
  auto ret = LoweringIdentityLikeNode(identity_n, lower_input);
  ge::DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredIdentityGraph");
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 2);
  EXPECT_EQ(ret.out_addrs.size(), 2);
  EXPECT_EQ(ret.out_shapes.size(), 2);
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "AllocMemHbm");
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(ret.out_addrs[1]->GetFastNode()->GetType(), "AllocMemHbm");
  EXPECT_EQ(ret.out_addrs[1]->GetPlacement(), kOnDeviceHbm);

  FastNodeTopoChecker checker(ret.out_addrs[0]);
  // src_addr, tensor_size, allocator, stream
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SelectL2Allocator", 0}, {"CalcTensorSizeFromShape", 0}}), true), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<FastSrcNode>({{"CopyD2D", 1}, {"FreeMemory", 0}})), "success");

  FastNodeTopoChecker checker_1(ret.out_addrs[1]);
  // makesure, tensor_size, allocator, stream
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SelectL2Allocator", 0}, {"CalcTensorSizeFromShape", 0}}), true), "success");
}

TEST_F(IdentityConverterUt, ConvertIdentityN_InputPlacementMix_UnknownAndHost_Ok) {
  auto graph = BuildIdentityNGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  auto identity_n = graph->FindNode("identityn");

  auto shape_holder_0 = ValueHolder::CreateFeed(0);
  auto shape_holder_1 = ValueHolder::CreateFeed(1);
  auto address_holder_0 = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  auto address_holder_1 = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  address_holder_0->SetPlacement(kTensorPlacementEnd);
  address_holder_1->SetPlacement(kOnHost);
  LowerInput lower_input{{shape_holder_0, shape_holder_1}, {address_holder_0, address_holder_1}, &lgd};
  auto ret = LoweringIdentityLikeNode(identity_n, lower_input);
  ge::DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredIdentityGraph");
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 2);
  EXPECT_EQ(ret.out_addrs.size(), 2);
  EXPECT_EQ(ret.out_shapes.size(), 2);
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "MakeSureTensorAtDevice");
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(ret.out_addrs[1]->GetFastNode()->GetType(), "CopyTensorDataH2H");
  EXPECT_EQ(ret.out_addrs[1]->GetPlacement(), kOnHost);

  FastNodeTopoChecker checker(ret.out_addrs[0]);
  // stream, allocator, src_addr, tensor_size, src_shape, datatype
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SplitRtStreams", 0},
                                                            {"SelectL2Allocator", 0},
                                                            {"Data", 0},
                                                            {"CalcTensorSizeFromShape", 0},
                                                            {"Data", 0},
                                                            {"Const", 0}}), true), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<FastSrcNode>({{"FreeMemory", 0}})), "success");
  
  FastNodeTopoChecker checker_1(ret.out_addrs[1]);
  // src_addr, src_shape, datatype
  EXPECT_EQ(checker_1.StrictConnectFrom(
                std::vector<FastSrcNode>({{"Data", 0}, {"InferShape", 1}, {"Const", 0}, {"InnerData", 0}})),
            "success");
  EXPECT_EQ(checker_1.StrictConnectTo(0, std::vector<FastSrcNode>({{"FreeMemory", 0}})), "success");
}

TEST_F(IdentityConverterUt, ConvertMemcpyAddrAsync_InputPlacementHost_Ok) {
  auto graph = BuildMemcpyAddrAsycnGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  auto memcpy_addr_async = graph->FindNode("memcpy_addr_async");

  auto shape_holder = ValueHolder::CreateFeed(0);
  auto address_holder = DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0);
  address_holder->SetPlacement(kOnHost);
  LowerInput lower_input{{shape_holder}, {address_holder}, &lgd};
  auto ret = LoweringIdentityLikeNode(memcpy_addr_async, lower_input);
  ge::DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredIdentityGraph");
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 1);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  // EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodesSize(), 3);
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "CopyTensorDataH2H");
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnHost);

  FastNodeTopoChecker checker(ret.out_addrs[0]);
  // src_addr, src_shape, datatype
  EXPECT_EQ(
      checker.StrictConnectFrom(std::vector<FastSrcNode>({{"Data", 0}, {"Data", 0}, {"Const", 0}, {"InnerData", 0}})),
      "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<FastSrcNode>({{"FreeMemory", 0}})), "success");
}
}  // namespace gert
