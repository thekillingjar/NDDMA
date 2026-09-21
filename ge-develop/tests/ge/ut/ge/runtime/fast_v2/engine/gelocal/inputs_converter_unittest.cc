/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/gelocal/inputs_converter.h"
#include <gtest/gtest.h>
#include <memory>
#include "graph/node.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "kernel/tensor_attr.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "common/summary_checker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "lowering/exe_graph_attrs.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph/operator_reg.h"
#include "graph/ge_context.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
namespace ge {
    REG_OP(ConstPlaceHolder)
        .OUTPUT(y, TensorType::ALL())
        .REQUIRED_ATTR(origin_shape, ListInt)
        .REQUIRED_ATTR(origin_format, Int)
        .REQUIRED_ATTR(storage_shape, ListInt)
        .REQUIRED_ATTR(storage_format, Int)
        .REQUIRED_ATTR(expand_dim_rules, String)
        .REQUIRED_ATTR(dtype, Type)
        .REQUIRED_ATTR(addr, Int)
        .REQUIRED_ATTR(size, Int)
        .ATTR(placement, Int, 1)
        .OP_END_FACTORY_REG(ConstPlaceHolder)
}
namespace gert {
using namespace ge;
class InputsConverterUt : public bg::BgTestAutoCreate3StageFrame {
 public:
  void ExpectFromInit(const bg::ValueHolderPtr &holder, const std::string &node_type, int32_t index) {
    auto parent_node = holder->GetFastNode();
    ASSERT_NE(parent_node, nullptr);
    auto graph = ge::FastNodeUtils::GetSubgraphFromNode(parent_node, 0);

    ASSERT_NE(graph, nullptr);
    auto init_netoutput = ExecuteGraphUtils::FindFirstNodeMatchType(graph, "InnerNetOutput");
    ASSERT_NE(init_netoutput, nullptr);
    const auto in_data_edge = init_netoutput->GetInDataEdgeByIndex(holder->GetOutIndex());
    ASSERT_NE(in_data_edge, nullptr);
    const auto addr_src_node = in_data_edge->src;
    ASSERT_NE(addr_src_node, nullptr);

    EXPECT_EQ(addr_src_node->GetType(), node_type)
        << "Expect node type " << node_type << ", actual " << addr_src_node->GetType();
    EXPECT_EQ(in_data_edge->src_output, index)
        << "Expect node out index " << index << ", actual " << in_data_edge->src_output;
  }
};
namespace {
/*
 *
 * netoutput
 *   |
 * const1
 */
ComputeGraphPtr BuildConstGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("const1", "Const")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto const1 = graph->FindNode("const1");
  const1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  auto tensor = std::make_shared<GeTensor>(const1->GetOpDesc()->GetOutputDesc(0));
  std::vector<float> data(8 * 3 * 224 * 224);
  for (size_t i = 0; i < 8 * 3 * 224 * 224; ++i) {
    data[i] = static_cast<float>(i);
  }
  tensor->SetData(reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float));
  if (!AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor)) {
    return nullptr;
  }
  return graph;
}

void ConstructStringTensor(std::shared_ptr<GeTensor> &ge_tensor) {
  auto elem_cnt = ge_tensor->GetTensorDesc().GetShape().GetShapeSize();
  uint64_t single_str_size = 64U;
  uint8_t *addr = reinterpret_cast<uint8_t*>(ge_tensor->MutableData().GetData());
  uint64_t offset = elem_cnt * sizeof(ge::StringHead);

  for(int64_t i = 0;i < elem_cnt; i++) {
    ge::StringHead *string_head = reinterpret_cast<ge::StringHead*>(addr);
    string_head->len = single_str_size;
    string_head->addr = offset;
    *(reinterpret_cast<char *>(addr) + offset + 0) = 'a';
    *(reinterpret_cast<char *>(addr) + offset + 1) = 'b';
    *(reinterpret_cast<char *>(addr) + offset + 2) = 'c';
    *(reinterpret_cast<char *>(addr) + offset + 3) = '\0';
    offset += single_str_size;
    addr += sizeof(ge::StringHead);
  }
}

void CheckStringTensor(void *data, const size_t elem_cnt) {
  uint64_t single_str_size = 64U;
  uint8_t *addr = reinterpret_cast<uint8_t*>(data);
  uint64_t offset = elem_cnt * sizeof(ge::StringHead);

  for(size_t i = 0; i < elem_cnt; i++) {
    ge::StringHead *string_head = reinterpret_cast<ge::StringHead*>(addr);
    ASSERT_EQ(string_head->len, (int64_t)single_str_size);
    ASSERT_EQ(string_head->addr, offset);
    ASSERT_STRCASEEQ((reinterpret_cast<char *>(addr) + offset + 0), "abc");
    offset += single_str_size;
    addr += sizeof(ge::StringHead);
  }
}
/*
 *
 * netoutput
 *   |
 * const1
 */
ComputeGraphPtr BuildConstWithStringGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const1", "Const")->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);
  auto const1 = graph->FindNode("const1");
  const1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({2}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({2}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_STRING);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_STRING);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);

  uint64_t single_str_size = 64U;
  auto elem_cnt = const1->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetShapeSize();
  uint64_t total_size = elem_cnt * (single_str_size + sizeof(ge::StringHead));
  vector<uint8_t> data(total_size);
  auto ge_tensor = std::make_shared<GeTensor>(const1->GetOpDesc()->GetOutputDesc(0));
  ge_tensor->SetData(reinterpret_cast<uint8_t *>(data.data()), data.size());
  (void)ge::TensorUtils::SetSize(ge_tensor->MutableTensorDesc(), total_size);

  ConstructStringTensor(ge_tensor);

  if (!AttrUtils::SetTensor(const1->GetOpDesc(), "value", ge_tensor)) {
    return nullptr;
  }
  return graph;
}

ComputeGraphPtr BuildConstWithMaxLengthTensor() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const1", "Const")->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);
  auto const1 = graph->FindNode("const1");
  const1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({2}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({2}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_STRING);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_STRING);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);

  uint64_t single_str_size = 1500000000U;
  auto elem_cnt = const1->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetShapeSize();
  uint64_t total_size = elem_cnt * (single_str_size + sizeof(ge::StringHead));
  vector<uint8_t> data(total_size);
  auto ge_tensor = std::make_shared<GeTensor>(const1->GetOpDesc()->GetOutputDesc(0));
  ge_tensor->SetData(reinterpret_cast<uint8_t *>(data.data()), data.size());
  (void)ge::TensorUtils::SetSize(ge_tensor->MutableTensorDesc(), total_size);

  ConstructStringTensor(ge_tensor);

  if (!AttrUtils::SetTensor(const1->GetOpDesc(), "value", ge_tensor)) {
    return nullptr;
  }
  return graph;
}
/*
 *
 * netoutput
 *   |
 * const1
 */
ComputeGraphPtr BuildEmptyTensorConstGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("const1", "Const")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto const1 = graph->FindNode("const1");
  const1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({0}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({0}));
  const1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  auto tensor = std::make_shared<GeTensor>(const1->GetOpDesc()->GetOutputDesc(0));
  std::vector<float> data(1);
  tensor->SetData(reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float));
  if (!AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor)) {
    return nullptr;
  }
  return graph;
}
/*
 *
 * netoutput
 *   |
 * data1
 */
ComputeGraphPtr BuildDataGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
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
 * aippData1
 */
ComputeGraphPtr BuildAippDataGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("aippData1", "AippData")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto aippData1 = graph->FindNode("aippData1");
  aippData1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  aippData1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  aippData1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  aippData1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  aippData1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  aippData1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  return graph;
}

/*
 *
 * netoutput
 *   |
 * refdata
 */
ComputeGraphPtr BuildRefDataGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata", "RefData")->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  auto refdata = graph->FindNode("refdata");
  refdata->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  refdata->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  refdata->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  refdata->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  refdata->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  refdata->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(refdata->GetOpDesc(), "index", 0)) {
    return nullptr;
  }
  return graph;
}
/*
 *        NetOutput            +--------------+  +-----------+
 *            |                |Then Graph    |  |Else Graph |
 *          cast               |              |  |           |
 *           |                 | NetOutput    |  | NetOutput |
 *          if <----------->   |   |          |  |   |       |
 *           |                 | assign       |  |  Cast     |
 *           |                 |   |   \      |  |   |       |
 *        /    \               |RefData const |  |  Refdata  |
 * pred(Data)  input(RefData)  |   |          |  |   |       |
 *                             | Data         |  | Data      |
 *                             +--------------+  +-----------+
 */
ComputeGraphPtr BuildRefDataInSubGraph() {
  std::vector<int64_t> shape = {8, 3, 16, 16};  // HWCN
  auto refdata1 = OP_CFG("RefData")
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .Attr(ATTR_NAME_INDEX, 1)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("input");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("pred", "Data")->NODE("if", "If")->NODE("cast", "Cast")->NODE("NetOutput", "NetOutput"));
                   CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE("if", "If"));
                 };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");

  vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_NCHW, DT_INT32);
  GeTensorPtr tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANTOP).Weight(tensor).Build("const1");

  auto assign =
      OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InNames({"ref", "value"}).OutNames({"ref"}).Build("assign");

  auto then_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE("input", "RefData")->NODE(assign)->NODE("cast", "Cast")->NODE("NetOutput1", "NetOutput"));
                   CHAIN(NODE(const1)->EDGE(0, 1)->NODE("assign", "Assign"));
                 };
    return ToComputeGraph(g);
  }();
  then_graph->SetName("then");
  auto data_node = then_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto netoutput_node = then_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto assign_node = then_graph->FindFirstNodeMatchType("Assign");
  assign_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(assign_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");

  auto else_graph = []() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE("input", "RefData")->NODE("cast1", "Cast")->NODE("NetOutput", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  data_node = else_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto cast_node = else_graph->FindFirstNodeMatchType("Cast");
  cast_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  AttrUtils::SetStr(cast_node->GetOpDesc(), "_kernel_bin_id", "te_add_12345");
  auto else_netoutput_node = else_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(else_netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  then_graph->SetGraphUnknownFlag(false);
  else_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"cast"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}
}  // namespace

TEST_F(InputsConverterUt, ConvertConstOk) {
  auto graph = BuildConstGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  auto const1 = graph->FindNode("const1");
  auto ret = LoweringConstNode(const1, {{}, {}, &lgd});
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  auto root_graph = ret.out_addrs[0]->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(root_graph)
                .StrictDirectNodeTypes({{"Init", 1}, {"Main", 1}, {"DeInit", 1}}),
            "success");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "Init");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode(), ret.out_shapes[0]->GetFastNode());
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnHost);

  ExpectFromInit(ret.out_addrs[0], "SplitConstTensor", 1);
  ExpectFromInit(ret.out_shapes[0], "SplitConstTensor", 0);

  auto init_node = ret.out_addrs[0]->GetFastNode();
  ASSERT_NE(init_node, nullptr);
  ASSERT_EQ(ret.out_addrs[0]->GetFastNode(), ret.out_shapes[0]->GetFastNode());
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);
  auto split_tensor_node = ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "SplitConstTensor");
  ASSERT_NE(split_tensor_node, nullptr);

  FastNodeTopoChecker checker(split_tensor_node);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"Const", 0}, {"Const", 0}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, {{"InnerNetOutput", 0}}), "success");
  EXPECT_EQ(checker.StrictConnectTo(1, std::vector<FastSrcNode>({{"InnerNetOutput", 1}})), "success");

  auto de_init_node= ExecuteGraphUtils::FindFirstNodeMatchType(root_graph, "DeInit");
  ASSERT_NE(de_init_node, nullptr);
  auto de_init_graph = FastNodeUtils::GetSubgraphFromNode(de_init_node, 0);
  ASSERT_NE(de_init_graph, nullptr);
  auto free_mem_node = ExecuteGraphUtils::FindFirstNodeMatchType(de_init_graph, "FreeMemory");
  // 这个地方没有FreeMemory节点了，Splitensor没有guarder，输出的tensordata的生命周期跟随输入的Tensor
  ASSERT_EQ(free_mem_node, nullptr);

  auto const_node = split_tensor_node->GetInDataEdgeByIndex(0)->src;
  ASSERT_NE(const_node, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(const_node->GetOpDescBarePtr(), "value", buffer));
  auto tensor = reinterpret_cast<Tensor *>(buffer.GetData());
  EXPECT_EQ(tensor->GetPlacement(), kOnHost);
  EXPECT_EQ(tensor->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(tensor->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(tensor->GetFormat().GetExpandDimsType(), ExpandDimsType());

  EXPECT_EQ(tensor->GetOriginShape().GetDimNum(), 4);
  EXPECT_EQ(tensor->GetStorageShape().GetDimNum(), 4);
  EXPECT_EQ(tensor->GetOriginShape().GetDim(0), 8);
  EXPECT_EQ(tensor->GetOriginShape().GetDim(1), 3);
  EXPECT_EQ(tensor->GetOriginShape().GetDim(2), 224);
  EXPECT_EQ(tensor->GetOriginShape().GetDim(3), 224);
  EXPECT_EQ(tensor->GetDataType(), ge::DT_FLOAT);
  auto data = tensor->GetData<float>();
  for (size_t i = 0; i < 8 * 3 * 224 * 224; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(InputsConverterUt, ConvertConstWithStringTypeOk) {
  auto graph = BuildConstWithStringGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  auto const1 = graph->FindNode("const1");
  auto ret = LoweringConstNode(const1, {{}, {}, &lgd});
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  auto root_graph = ret.out_addrs[0]->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(root_graph)
                .StrictDirectNodeTypes({{"Init", 1}, {"Main", 1}, {"DeInit", 1}}),
            "success");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "Init");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode(), ret.out_shapes[0]->GetFastNode());
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnHost);

  ExpectFromInit(ret.out_addrs[0], "SplitConstTensor", 1);
  ExpectFromInit(ret.out_shapes[0], "SplitConstTensor", 0);

  auto init_node = ret.out_addrs[0]->GetFastNode();
  ASSERT_NE(init_node, nullptr);
  ASSERT_EQ(ret.out_addrs[0]->GetFastNode(), ret.out_shapes[0]->GetFastNode());
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);
  auto split_tensor_node = ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "SplitConstTensor");
  ASSERT_NE(split_tensor_node, nullptr);

  FastNodeTopoChecker checker(split_tensor_node);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"Const", 0}, {"Const", 0}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, {{"InnerNetOutput", 0}}), "success");
  EXPECT_EQ(checker.StrictConnectTo(1, std::vector<FastSrcNode>({{"InnerNetOutput", 1}})), "success");

  auto de_init_node= ExecuteGraphUtils::FindFirstNodeMatchType(root_graph, "DeInit");
  ASSERT_NE(de_init_node, nullptr);
  auto de_init_graph = FastNodeUtils::GetSubgraphFromNode(de_init_node, 0);
  ASSERT_NE(de_init_graph, nullptr);
  auto free_mem_node = ExecuteGraphUtils::FindFirstNodeMatchType(de_init_graph, "FreeMemory");
  // 这个地方没有FreeMemory节点了，Splitensor没有guarder，输出的tensordata的生命周期跟随输入的Tensor
  ASSERT_EQ(free_mem_node, nullptr);

  auto const_node = split_tensor_node->GetInDataEdgeByIndex(0)->src;
  ASSERT_NE(const_node, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(const_node->GetOpDescPtr(), "value", buffer));
  auto tensor = reinterpret_cast<Tensor *>(buffer.GetData());
  EXPECT_EQ(tensor->GetPlacement(), kOnHost);
  EXPECT_EQ(tensor->GetOriginShape().GetDimNum(), 1);
  EXPECT_EQ(tensor->GetStorageShape().GetDimNum(), 1);
  EXPECT_EQ(tensor->GetOriginShape().GetDim(0), 2);
  EXPECT_EQ(tensor->GetDataType(), ge::DT_STRING);

  auto data = tensor->GetData<char *>();
  CheckStringTensor(data, 2);
}

TEST_F(InputsConverterUt, ConvertConstWithEmptyTensorOk) {
  auto graph = BuildEmptyTensorConstGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  auto const1 = graph->FindNode("const1");
  auto ret = LoweringConstNode(const1, {{}, {}, &lgd});
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  auto root_graph = ret.out_addrs[0]->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(root_graph)
                .StrictDirectNodeTypes({{"Init", 1}, {"Main", 1}, {"DeInit", 1}}),
            "success");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "Init");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode(), ret.out_shapes[0]->GetFastNode());
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnHost);

  ExpectFromInit(ret.out_addrs[0], "SplitConstTensor", 1);
  ExpectFromInit(ret.out_shapes[0], "SplitConstTensor", 0);

  auto init_node = ret.out_addrs[0]->GetFastNode();
  ASSERT_NE(init_node, nullptr);
  ASSERT_EQ(ret.out_addrs[0]->GetFastNode(), ret.out_shapes[0]->GetFastNode());
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);
  auto split_tensor_node = ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "SplitConstTensor");
  ASSERT_NE(split_tensor_node, nullptr);

  FastNodeTopoChecker checker(split_tensor_node);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"Const", 0}, {"Const", 0}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, {{"InnerNetOutput", 0}}), "success");
  EXPECT_EQ(checker.StrictConnectTo(1, std::vector<FastSrcNode>({{"InnerNetOutput", 1}})), "success");

  auto de_init_node= ExecuteGraphUtils::FindFirstNodeMatchType(root_graph, "DeInit");
  ASSERT_NE(de_init_node, nullptr);
  auto de_init_graph = FastNodeUtils::GetSubgraphFromNode(de_init_node, 0);
  ASSERT_NE(de_init_graph, nullptr);
  auto free_mem_node = ExecuteGraphUtils::FindFirstNodeMatchType(de_init_graph, "FreeMemory");
  // 这个地方没有FreeMemory节点了，Splitensor没有guarder，输出的tensordata的生命周期跟随输入的Tensor
  ASSERT_EQ(free_mem_node, nullptr);

  auto const_node = split_tensor_node->GetInDataEdgeByIndex(0)->src;
  ASSERT_NE(const_node, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(const_node->GetOpDescPtr(), "value", buffer));
  auto tensor = reinterpret_cast<Tensor *>(buffer.GetData());
  EXPECT_EQ(tensor->GetPlacement(), kOnHost);
  EXPECT_EQ(tensor->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(tensor->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(tensor->GetFormat().GetExpandDimsType(), ExpandDimsType());

  EXPECT_EQ(tensor->GetOriginShape().GetDimNum(), 1);
  EXPECT_EQ(tensor->GetStorageShape().GetDimNum(), 1);
  EXPECT_EQ(tensor->GetOriginShape().GetDim(0), 0);
  EXPECT_EQ(tensor->GetStorageShape().GetDim(0), 0);
  EXPECT_EQ(tensor->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(tensor->GetSize(), 0);
}

TEST_F(InputsConverterUt, CopyTensorOk) {
  auto op_desc = std::make_shared<OpDesc>("test", "test");
  {
    Tensor tensor;
    tensor.MutableOriginShape() = {8, 3, 224, 224};
    tensor.MutableStorageShape() = {8, 3, 224, 224};
    AttrUtils::SetZeroCopyBytes(op_desc, "value",
                                ge::Buffer::CopyFrom(reinterpret_cast<uint8_t *>(&tensor), sizeof(tensor)));
  }

  ge::Buffer buffer;

  ASSERT_TRUE(AttrUtils::GetZeroCopyBytes(op_desc, "value", buffer));

  auto tensor1 = reinterpret_cast<Tensor *>(buffer.GetData());
  EXPECT_EQ(tensor1->GetOriginShape().GetDim(0), 8);
  EXPECT_EQ(tensor1->GetOriginShape().GetDim(1), 3);
  EXPECT_EQ(tensor1->GetOriginShape().GetDim(2), 224);
  EXPECT_EQ(tensor1->GetOriginShape().GetDim(3), 224);
  EXPECT_EQ(tensor1->GetStorageShape().GetDim(0), 8);
  EXPECT_EQ(tensor1->GetStorageShape().GetDim(1), 3);
  EXPECT_EQ(tensor1->GetStorageShape().GetDim(2), 224);
  EXPECT_EQ(tensor1->GetStorageShape().GetDim(3), 224);
}
ge::FastNode *FindDataNode(const ge::ExecuteGraph *graph, int64_t expect_index) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "Data") {
      int64_t index;
      ge::AttrUtils::GetInt(node->GetOpDescPtr(), "index", index);
      if (index == expect_index) {
        return node;
      }
    }
  }
  return nullptr;
}

/*
 * exepct graph:
 *       FreeTensorMemory
 *            |
 *        SplitTensor
 *            |
 *           Data
 */
TEST_F(InputsConverterUt, ConvertDataOk) {
  auto graph = BuildDataGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};
  auto data1 = graph->FindNode("data1");
  auto ret = LoweringDataNode(data1, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredDataGraph");
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodes().size(), 10);
  auto data_node = FindDataNode(ret.out_addrs[0]->GetExecuteGraph(), 0);
  ASSERT_NE(data_node, nullptr);
  EXPECT_EQ(data_node->GetDataOutNum(), 1);

  auto sink_tensor_node = ExecuteGraphUtils::FindFirstNodeMatchType(ret.out_addrs[0]->GetExecuteGraph(),"SplitDataTensor");
  ASSERT_NE(sink_tensor_node, nullptr);
  EXPECT_EQ(sink_tensor_node->GetDataOutNum(), 2);
  EXPECT_EQ(sink_tensor_node->GetDataInNum(), 2);

  // 这个地方没有FreeMemory节点了，Splitensor没有guarder，输出的tensordata的生命周期跟随输入的Tensor
  auto free_memory_node = ExecuteGraphUtils::FindFirstNodeMatchType(ret.out_addrs[0]->GetExecuteGraph(), "FreeMemory");
  EXPECT_EQ(free_memory_node, nullptr);
  EXPECT_EQ(data_node->GetOutDataNodesByIndex(0).front(), sink_tensor_node);
}

/*
 * exepct graph:
 *       netoutput
 *            |
 *           Data
 */
TEST_F(InputsConverterUt, ConvertDataWithFrozen) {
  auto graph = BuildDataGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};
  auto data1 = graph->FindNode("data1");
  const auto op_desc = data1->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetBool(op_desc, "frozen_input", true));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "addr", 1111));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "size", 8 * 3 * 224 * 224 * sizeof(float)));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "placement", ge::Placement::kPlacementDevice));
  std::vector<int64_t> input_shape = {8, 3, 224, 224};
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "storage_shape", input_shape));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "origin_shape", input_shape));
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));

  auto ret = LoweringDataNode(data1, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  auto root_graph = ret.out_addrs[0]->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(root_graph)
                .StrictDirectNodeTypes({{"Init", 1}, {"Main", 1}, {"DeInit", 1}}),
            "success");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "Init");

  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);

  const auto output_shape_node = bg::HolderOnInit(ret.out_shapes[0])->GetFastNode();
  ASSERT_NE(output_shape_node, nullptr);
  ge::Buffer buf;
  ge::AttrUtils::GetZeroCopyBytes(output_shape_node->GetOpDescPtr(), gert::kConstValue, buf);
  auto storage_shape = reinterpret_cast<StorageShape *>(buf.GetData());
  ASSERT_NE(storage_shape, nullptr);

  ASSERT_EQ(storage_shape->GetOriginShape().GetDimNum(), input_shape.size());
  for (size_t i = 0U; i < storage_shape->GetOriginShape().GetDimNum(); ++i) {
    EXPECT_EQ(storage_shape->GetOriginShape().GetDim(i), input_shape[i]);
  }
  ASSERT_EQ(storage_shape->GetStorageShape().GetDimNum(), input_shape.size());
  for (size_t i = 0U; i < storage_shape->GetStorageShape().GetDimNum(); ++i) {
    EXPECT_EQ(storage_shape->GetStorageShape().GetDim(i), input_shape[i]);
  }
}

TEST_F(InputsConverterUt, LoweringUnfedDataNode) {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("x", "UnfedData")->NODE("foo", "Foo")->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};
  auto data = graph->FindNode("x");
  auto ret = LoweringUnfedDataNode(data, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweringUnfedDataNode");
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodes().size(), 1 /*stream*/ + 2 /*lower*/ + 2/*SplitRtStreams const*/);

  auto unfed_data = ExecuteGraphUtils::FindFirstNodeMatchType(ret.out_addrs[0]->GetExecuteGraph(), "UnfedData");
  ASSERT_NE(unfed_data, nullptr);
  EXPECT_EQ(unfed_data->GetDataOutNum(), 2);
  EXPECT_EQ(unfed_data->GetDataInNum(), 0);
}

/*
 * exepct graph:
 *       FreeTensorMemory
 *            |
 *        SplitTensor
 *            |
 *           AippData
 *
 */
TEST_F(InputsConverterUt, ConvertAippDataOk) {
  auto graph = BuildAippDataGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};
  auto aippData1 = graph->FindNode("aippData1");
  ASSERT_TRUE(AttrUtils::SetInt(aippData1->GetOpDesc(), "index", 1)); // set no match data index
  auto ret = LoweringDataNode(aippData1, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredDataGraph");

  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodes().size(), 10);
  auto data_node = FindDataNode(ret.out_addrs[0]->GetExecuteGraph(), 0);
  ASSERT_NE(data_node, nullptr);
  EXPECT_EQ(data_node->GetDataOutNum(), 1);

  auto sink_tensor_node = ExecuteGraphUtils::FindFirstNodeMatchType(ret.out_addrs[0]->GetExecuteGraph(), "SplitDataTensor");
  ASSERT_NE(sink_tensor_node, nullptr);
  EXPECT_EQ(sink_tensor_node->GetDataOutNum(), 2);
  EXPECT_EQ(sink_tensor_node->GetDataInNum(), 2);
  EXPECT_EQ(data_node->GetOutDataNodesByIndex(0).front(), sink_tensor_node);
}
TEST_F(InputsConverterUt, ConvertConstWithMaxLengthTensor) {
  auto graph = BuildConstWithMaxLengthTensor();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  auto const1 = graph->FindNode("const1");
  auto ret = LoweringConstNode(const1, {{}, {}, &lgd});
  ASSERT_TRUE(ret.result.IsSuccess());
}

/*
 * exepct graph:
 *       FreeTensorMemory
 *            |
 *        SplitTensor
 *            |
 *           Data
 *
 */
TEST_F(InputsConverterUt, ConvertRefDataOk) {
  auto graph = BuildRefDataGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};
  auto refdata = graph->FindNode("refdata");
  auto ret = LoweringRefDataNode(refdata, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  DumpGraph(ret.out_addrs[0]->GetExecuteGraph(), "LoweredRefDataGraph");
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodes().size(), 10);
  auto data_node = FindDataNode(ret.out_addrs[0]->GetExecuteGraph(), 0);
  ASSERT_NE(data_node, nullptr);
  EXPECT_EQ(data_node->GetDataOutNum(), 1);

  auto sink_tensor_node = ExecuteGraphUtils::FindFirstNodeMatchType(ret.out_addrs[0]->GetExecuteGraph(),"SplitDataTensor");
  ASSERT_NE(sink_tensor_node, nullptr);
  EXPECT_EQ(sink_tensor_node->GetDataOutNum(), 2);
  EXPECT_EQ(sink_tensor_node->GetDataInNum(), 2);

  // 这个地方没有FreeMemory节点了，Splitensor没有guarder，输出的tensordata的生命周期跟随输入的Tensor
  auto free_memory_node = ExecuteGraphUtils::FindFirstNodeMatchType(ret.out_addrs[0]->GetExecuteGraph(), "FreeMemory");
  EXPECT_EQ(free_memory_node, nullptr);

  EXPECT_EQ(data_node->GetOutDataNodesByIndex(0).front(), sink_tensor_node);

  // 校验global data中有refdata的地址
  auto builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    return {};
  };
  auto refdata_shape_and_addr = global_lowering_data.GetOrCreateUniqueValueHolder("refdata", builder);
  EXPECT_EQ(refdata_shape_and_addr.size(), 2);
  EXPECT_NE(refdata_shape_and_addr[1], nullptr);
}

/*
 * refdata in subgraph share one addr with refdata in root graph
 */
TEST_F(InputsConverterUt, ConvertRefDataInSubGraphOk) {
  auto graph = BuildRefDataInSubGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};

  auto refdata = graph->FindNode("input");
  // lowering refdata in root graph
  auto ret = LoweringRefDataNode(refdata, lower_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  EXPECT_EQ(ret.out_addrs[0]->GetExecuteGraph()->GetAllNodes().size(), 11);


  // 校验global data中有refdata的地址
  auto builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    return {};
  };
  auto refdata_shape_and_addr = global_lowering_data.GetOrCreateUniqueValueHolder("input", builder);
  EXPECT_EQ(refdata_shape_and_addr.size(), 2);
  EXPECT_NE(refdata_shape_and_addr[1], nullptr);

  // lowering refdata in subgraph
  auto then_graph = graph->GetSubgraph("then");
  auto refdata_in_then = then_graph->FindNode("input");
  auto then_ret = LoweringRefDataNode(refdata_in_then, lower_input);
  ASSERT_TRUE(then_ret.result.IsSuccess());
  EXPECT_EQ(then_ret.order_holders.size(), 0);
  EXPECT_EQ(then_ret.out_shapes.size(), 1);
  EXPECT_EQ(then_ret.out_addrs.size(), 1);
  EXPECT_EQ(then_ret.out_addrs[0], refdata_shape_and_addr[1]);
}

/*
 * refdata in subgraph lowering before refdata in root graph
 */
TEST_F(InputsConverterUt, ConvertRefDataInSubGraphFailed) {
  auto graph = BuildRefDataInSubGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};

  // lowering refdata in subgraph
  auto then_graph = graph->GetSubgraph("then");
  auto refdata_in_then = then_graph->FindNode("input");
  auto then_ret = LoweringRefDataNode(refdata_in_then, lower_input);
  ASSERT_FALSE(then_ret.result.IsSuccess());
}

/*
 * netoutput
 *   |
 * ConstPlaceHolder
 */
ComputeGraphPtr BuildConstPlaceHolderGraph() {
  DEF_GRAPH(g1) {
                    CHAIN(NODE("ConstPlaceHolder", "ConstPlaceHolder")->NODE("NetOutput", "NetOutput"));
                };
  vector<int64_t > shape_ori({1, 2, 3});
  vector<int64_t > shape_sto({3, 2, 1});
  auto graph = ToComputeGraph(g1);
  auto ConstPlaceHolder = graph->FindNode("ConstPlaceHolder");
  auto constplaceholder_op_desc1 = ConstPlaceHolder->GetOpDesc();
  ge::AttrUtils::SetListInt(constplaceholder_op_desc1, "origin_shape", shape_ori);
  ge::AttrUtils::SetListInt(constplaceholder_op_desc1, "storage_shape", shape_sto);
  DataType data_type = DT_FLOAT;
  ge::AttrUtils::SetDataType(constplaceholder_op_desc1, "dtype", data_type); // float
  int64_t data_length = 24L;
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "size", data_length);
  int64_t placement = 1L;
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "placement", placement); // device
  int64_t device_addr = 20000;
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "addr", device_addr);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"ConstPlaceHolder"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

TEST_F(InputsConverterUt, ConvertConstPlaceHolderOk) {
  auto graph = BuildConstPlaceHolderGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  auto ConstPlaceHolder1 = graph->FindNode("ConstPlaceHolder");

  auto ret = LoweringConstPlaceHolderNode(ConstPlaceHolder1, {{}, {}, &global_lowering_data});
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 0);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);
  auto root_graph = ret.out_addrs[0]->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(root_graph)
                .StrictDirectNodeTypes({{"Init", 1}, {"Main", 1}, {"DeInit", 1}}),
            "success");
  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), "Init");

  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);

  const auto output_shape_node = bg::HolderOnInit(ret.out_shapes[0])->GetFastNode();
  ASSERT_NE(output_shape_node, nullptr);
  ge::Buffer buf;
  ge::AttrUtils::GetZeroCopyBytes(output_shape_node->GetOpDescPtr(), gert::kConstValue, buf);
  auto storage_shape = reinterpret_cast<StorageShape *>(buf.GetData());
  ASSERT_NE(storage_shape, nullptr);
  std::vector<int64_t> original_shape;
  std::vector<int64_t> storage_shape_use;
  ge::AttrUtils::GetListInt(ConstPlaceHolder1->GetOpDesc(), "origin_shape", original_shape);
  ge::AttrUtils::GetListInt(ConstPlaceHolder1->GetOpDesc(), "storage_shape", storage_shape_use);
  ASSERT_EQ(storage_shape->GetOriginShape().GetDimNum(), original_shape.size());
  for (size_t i = 0U; i < storage_shape->GetOriginShape().GetDimNum(); ++i) {
    EXPECT_EQ(storage_shape->GetOriginShape().GetDim(i), original_shape[i]);
  }
  ASSERT_EQ(storage_shape->GetStorageShape().GetDimNum(), storage_shape_use.size());
  for (size_t i = 0U; i < storage_shape->GetStorageShape().GetDimNum(); ++i) {
    EXPECT_EQ(storage_shape->GetStorageShape().GetDim(i), storage_shape_use[i]);
  }
}

/*
 * netoutput
 *   |
 * ConstPlaceHolder
 */
ComputeGraphPtr BuildMultiBatchShapeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("case", "Case")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 2)->NODE("case"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 3)->NODE("case"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 4)->NODE("case"));
    CTRL_CHAIN(NODE("data1")->NODE("shape_data", "DATA"));
    CTRL_CHAIN(NODE("data2")->NODE("shape_data"));
    CTRL_CHAIN(NODE("data3")->NODE("shape_data"));
    CHAIN(NODE("shape_data")->NODE("mapIndex", "MapIndex"));
    CHAIN(NODE("const", "Constant")->EDGE(0, 1)->NODE("mapIndex")->EDGE(0, 0)->NODE("case"));
  };
  auto graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 1, 100}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 1, 100}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  std::vector<int32_t> unknown_dim_index_1 = {0, 3};
  (void)ge::AttrUtils::SetListInt(data1->GetOpDesc(), "_dynamic_batch_unknown_dim_index", unknown_dim_index_1);
  if (!AttrUtils::SetInt(data1->GetOpDesc(), "index", 0)) {
    return nullptr;
  }
  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 1, 100}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 1, 100}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  std::vector<int32_t> unknown_dim_index_2 = {0, 2};
  (void)ge::AttrUtils::SetListInt(data2->GetOpDesc(), "_dynamic_batch_unknown_dim_index", unknown_dim_index_2);
  if (!AttrUtils::SetInt(data2->GetOpDesc(), "index", 1)) {
    return nullptr;
  }
  auto data3 = graph->FindNode("data3");
  data3->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 1, 100}));
  data3->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 1, 100}));
  data3->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data3->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data3->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data3->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  std::vector<int32_t> unknown_dim_index_3 = {3};
  (void)ge::AttrUtils::SetListInt(data3->GetOpDesc(), "_dynamic_batch_unknown_dim_index", unknown_dim_index_3);
  if (!AttrUtils::SetInt(data3->GetOpDesc(), "index", 2)) {
    return nullptr;
  }
  auto data4 = graph->FindNode("data4");
  data4->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 1, 100}));
  data4->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 1, 100}));
  data4->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data4->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data4->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data4->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  if (!AttrUtils::SetInt(data4->GetOpDesc(), "index", 3)) {
    return nullptr;
  }
  auto shape_data = graph->FindNode("shape_data");
  shape_data->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({5}));
  shape_data->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({5}));
  shape_data->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  shape_data->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
  shape_data->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  shape_data->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
  std::vector<int32_t> unknown_shape_data_index = {0, 1, 2};
  (void)ge::AttrUtils::SetListInt(shape_data->GetOpDesc(), "_dynamic_batch_unknown_data_index", unknown_shape_data_index);
  if (!AttrUtils::SetInt(shape_data->GetOpDesc(), "index", 4)) {
    return nullptr;
  }
  graph->SetGraphID(0);
  (void)ge::AttrUtils::SetBool(graph, "_enable_dynamic_batch", true);
  return graph;
}

TEST_F(InputsConverterUt, ConvertMultiShapeNodeOk) {
  auto graph = BuildMultiBatchShapeGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_lowering_data = GlobalDataFaker(root_model).Build();
  LowerInput lower_input = {{}, {}, &global_lowering_data};
  auto data1 = graph->FindNode("data1");
  auto ret1 = LoweringDataNode(data1, lower_input);
  data1->GetOpDesc()->SetExtAttr(kLoweringResult, PlacedLoweringResult(data1, std::move(ret1)));
  auto data2 = graph->FindNode("data2");
  auto ret2 = LoweringDataNode(data2, lower_input);
  data2->GetOpDesc()->SetExtAttr(kLoweringResult, PlacedLoweringResult(data2, std::move(ret2)));
  auto data3 = graph->FindNode("data3");
  auto ret3 = LoweringDataNode(data3, lower_input);
  data3->GetOpDesc()->SetExtAttr(kLoweringResult, PlacedLoweringResult(data3, std::move(ret3)));
  auto data4 = graph->FindNode("data4");
  auto ret4 = LoweringDataNode(data4, lower_input);
  data4->GetOpDesc()->SetExtAttr(kLoweringResult, PlacedLoweringResult(data4, std::move(ret4)));
  auto shape_data = graph->FindNode("shape_data");
  auto shape_ret = LoweringMultiBatchShapeDataNode(shape_data, lower_input);
  ASSERT_TRUE(shape_ret.result.IsSuccess());
  EXPECT_EQ(shape_ret.order_holders.size(), 0);
  EXPECT_EQ(shape_ret.out_addrs.size(), 1);
  EXPECT_EQ(shape_ret.out_shapes.size(), 1);
  EXPECT_EQ(shape_ret.out_addrs[0]->GetFastNode()->GetType(), "GetCurDynamicShape");
}
}  // namespace gert