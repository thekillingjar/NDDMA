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
#include "graph/node.h"
#include "faker/global_data_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "common/const_data_helper.h"
#include "register/node_converter_registry.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "lowering/exe_graph_attrs.h"
#include "common/summary_checker.h"

namespace gert {
using namespace ge;
LowerResult LoweringFileConstantNode(const ge::NodePtr &node, const LowerInput &lower_input);
namespace {
void SetDefaultOutputTensorDesc(const ge::NodePtr &file_constant) {
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({5, 5}));
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({5, 5}));
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
}
void SetDefaultAttr(const ge::NodePtr &file_constant, const std::string &file_path_config = "test_weight_convert.bin") {
  // set attr
  std::vector<int64_t> shape = {5, 5};
  std::vector<int64_t> original_shape = {1, 5, 5};
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "offset", 0);
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "length", 0);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "location", "");
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_path", file_path_config);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_id", "");
  ge::AttrUtils::SetDataType(file_constant->GetOpDesc(), "dtype", DT_BF16);
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "shape", shape);
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "original_shape", original_shape);
}
/*
 * netoutput
 *   |
 * FileConstant
 */
ComputeGraphPtr BuildFileConstantGraph(const std::vector<std::string> &file_path_config = {"test_weight_convert.bin"}) {
  (void)file_path_config;
  DEF_GRAPH(g1) {
                  CHAIN(NODE("FileConstant", "FileConstant")->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);
  auto file_constant = graph->FindNode("FileConstant");
  SetDefaultOutputTensorDesc(file_constant);
  SetDefaultAttr(file_constant);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"FileConstant"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  return graph;
}

/*
 *            netoutput
 *            /      \
 * file_constant_0  file_constant_1
 */
ComputeGraphPtr Build2FileConstantGraph(const std::vector<std::string> &file_path_config) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("file_constant_0", "FileConstant")->NODE("NetOutput", "NetOutput"));
                  CHAIN(NODE("file_constant_1", "FileConstant")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
                };
  auto graph = ToComputeGraph(g1);
  GE_ASSERT_TRUE(file_path_config.size() == 2U);
  auto file_constant_0 = graph->FindNode("file_constant_0");
  SetDefaultOutputTensorDesc(file_constant_0);
  SetDefaultAttr(file_constant_0, file_path_config[0]);
  auto file_constant_1 = graph->FindNode("file_constant_1");
  SetDefaultOutputTensorDesc(file_constant_1);
  SetDefaultAttr(file_constant_1, file_path_config[1]);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"file_constant_0", "file_constant_1"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

ComputeGraphPtr Build2FileConstantGraphWithDiffShape(const std::vector<std::string> &file_path_config) {
  auto compute_graph = Build2FileConstantGraph(file_path_config);
  auto file_constant_node = compute_graph->FindFirstNodeMatchType(FILECONSTANT);
  ge::AttrUtils::SetListInt(file_constant_node->GetOpDesc(), "shape", {});
  ge::AttrUtils::SetListInt(file_constant_node->GetOpDesc(), "original_shape", {});
  return compute_graph;
}

struct GraphBuilderWrapper{
  using GraphBuilderFunc = std::function<ComputeGraphPtr(const std::vector<std::string> &file_path_config)>;
  using GraphBuilderParam = const std::vector<std::string>;
  GraphBuilderFunc graph_builder_func;
  GraphBuilderParam file_path_config;
};
}  // namespace

class FileConstantConverterUT : public bg::BgTestAutoCreate3StageFrame {
 public:
  static void EXPECT_TestLoweringFileConstantNode(
      const GraphBuilderWrapper &graph_builder_wrapper,
      const std::vector<std::map<std::string, size_t>> &init_graph_node_types_to_counts,
      const std::vector<std::map<std::string, size_t>> &deinit_graph_node_types_to_counts, LoweringGlobalData &lgd) {
    LowerResult ret;
    auto graph = graph_builder_wrapper.graph_builder_func(graph_builder_wrapper.file_path_config);
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
    lgd = GlobalDataFaker(root_model).Build();
    lgd.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
    lgd.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
    bg::LowerConstDataNode(lgd);
    size_t k_file_constant_cnt = 0U;
    for (auto &node : graph->GetAllNodes()) {
      if (node->GetType() == FILECONSTANT) {
        ret = LoweringFileConstantNode(node, {{}, {}, &lgd});
        ASSERT_TRUE(ret.result.IsSuccess());
        EXPECT_EQ(ret.order_holders.size(), 1);
        EXPECT_EQ(ret.out_addrs.size(), 1);
        EXPECT_EQ(ret.out_shapes.size(), 1);

        const auto &main_graph = bg::ValueHolder::GetCurrentFrame()->GetExecuteGraph();
        EXPECT_EQ(main_graph->GetDirectNodesSize(), 4U); // data*2, const, splitRtTensor

        ASSERT_TRUE(k_file_constant_cnt < init_graph_node_types_to_counts.size());
        ASSERT_TRUE(k_file_constant_cnt < deinit_graph_node_types_to_counts.size());
        EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit));
        const auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(ret.out_addrs[0]->GetFastNode(), 0U);
        EXPECT_EQ(
            ExeGraphSummaryChecker(init_graph).StrictDirectNodeTypes(init_graph_node_types_to_counts[k_file_constant_cnt]),
            "success");

        const auto &root_graph = ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
        ASSERT_NE(root_graph, nullptr);
        const auto &deinit_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph, GetExecuteGraphTypeStr(ExecuteGraphType::kDeInit));
        ASSERT_NE(deinit_node, nullptr);
        const auto &deinit_graph = ge::FastNodeUtils::GetSubgraphFromNode(deinit_node, 0U);
        ASSERT_NE(deinit_graph, nullptr);
        EXPECT_EQ(
            ExeGraphSummaryChecker(deinit_graph).StrictDirectNodeTypes(deinit_graph_node_types_to_counts[k_file_constant_cnt]),
            "success");
        ++k_file_constant_cnt;
      }
    }
  }
};

TEST_F(FileConstantConverterUT, ConvertFileConstantOk) {
  auto graph = BuildFileConstantGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  bg::CreateStreamNodeOnInitGraph(lgd);
  auto FileConstant = graph->FindNode("FileConstant");
  auto ret = LoweringFileConstantNode(FileConstant, {{}, {}, &lgd});

  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 1);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);

  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit));
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);

  const auto output_shape_node = bg::HolderOnInit(ret.out_shapes[0])->GetFastNode();
  ASSERT_NE(output_shape_node, nullptr);
  ge::Buffer buf;
  ge::AttrUtils::GetZeroCopyBytes(output_shape_node->GetOpDescBarePtr(), kConstValue, buf);
  auto storage_shape = reinterpret_cast<StorageShape *>(buf.GetData());
  ASSERT_NE(storage_shape, nullptr);
  std::vector<int64_t> original_shape;
  ge::AttrUtils::GetListInt(FileConstant->GetOpDesc(), "original_shape", original_shape);
  ASSERT_EQ(storage_shape->GetOriginShape().GetDimNum(), original_shape.size());
  for (size_t i = 0U; i < storage_shape->GetOriginShape().GetDimNum(); ++i) {
    EXPECT_EQ(storage_shape->GetOriginShape().GetDim(i), original_shape[i]);
  }
}

TEST_F(FileConstantConverterUT, ConvertFileConstant_Success_WithUserDeviceMem) {
  auto graph = BuildFileConstantGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  bg::CreateStreamNodeOnInitGraph(lgd);

  const size_t mem_size = 100U;
  uint8_t user_mem[mem_size];
  ge::FileConstantMem file_constant_mem{"test_weight_convert.bin", (void *)user_mem, mem_size};
  lgd.SetFileConstantMem({file_constant_mem});
  auto FileConstant = graph->FindNode("FileConstant");
  auto ret = LoweringFileConstantNode(FileConstant, {{}, {}, &lgd});
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 1);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);

  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit));
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);

  const auto output_addr_node = bg::HolderOnInit(ret.out_addrs[0])->GetFastNode();
  ASSERT_NE(output_addr_node, nullptr);
  auto in_nodes = output_addr_node->GetInDataNodes();
  ASSERT_EQ(in_nodes.size(), 2U);
  EXPECT_EQ(output_addr_node->GetType(), "FileConstantUserMemKernel");
}

TEST_F(FileConstantConverterUT, ConvertFileConstant_Failed_WithUserDeviceMem) {
  auto graph = BuildFileConstantGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  bg::CreateStreamNodeOnInitGraph(lgd);

  const size_t mem_size = 100U;
  uint8_t user_mem[mem_size];
  ge::FileConstantMem file_constant_mem{"test_weight_convert.bin", (void *)user_mem, mem_size - 1U};
  lgd.SetFileConstantMem({file_constant_mem});
  auto FileConstant = graph->FindNode("FileConstant");
  auto ret = LoweringFileConstantNode(FileConstant, {{}, {}, &lgd});
  ASSERT_FALSE(ret.result.IsSuccess());
}

// 用户设置了其他FileConstant的地址
TEST_F(FileConstantConverterUT, ConvertFileConstant_Success_UserSetOtherNode) {
  auto graph = BuildFileConstantGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData lgd = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(lgd);
  bg::CreateStreamNodeOnInitGraph(lgd);

  const size_t mem_size = 100U;
  uint8_t user_mem[mem_size];
  ge::FileConstantMem file_constant_mem{"not_exist.bin", (void *)user_mem, mem_size};
  lgd.SetFileConstantMem({file_constant_mem});
  auto FileConstant = graph->FindNode("FileConstant");
  auto ret = LoweringFileConstantNode(FileConstant, {{}, {}, &lgd});
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_EQ(ret.order_holders.size(), 1);
  EXPECT_EQ(ret.out_addrs.size(), 1);
  EXPECT_EQ(ret.out_shapes.size(), 1);

  EXPECT_EQ(ret.out_addrs[0]->GetFastNode()->GetType(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit));
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnDeviceHbm);

  const auto output_addr_node = bg::HolderOnInit(ret.out_addrs[0])->GetFastNode();
  ASSERT_NE(output_addr_node, nullptr);
  ASSERT_NE(output_addr_node->GetType(), "FileConstantUserMemKernel");
}

// 用例描述: 计算图上单个FileConstant节点，lowering节点数量符合预期
TEST_F(FileConstantConverterUT, LoweringFileConstantNode_ExeGraphNodeStaticOK) {
  LoweringGlobalData lgd;
  const std::vector<std::string> file_path_config({"test_weight_convert.bin"});
  const GraphBuilderWrapper graph_builder_wrapper{BuildFileConstantGraph, file_path_config};
  const std::map<std::string, size_t> init_graph_node_types_to_count = {{"ConstData", 3},
                                                                        {"Const", 8},
                                                                        {"Data", 1},
                                                                        {"CalcTensorSizeFromShape", 1},
                                                                        {"CreateL1Allocator", 1},
                                                                        {"CreateL2Allocators", 1},
                                                                        {"CreateInitL2Allocator", 1},
                                                                        {"SplitRtStreams", 1},
                                                                        {"GetFileConstantWeightDir", 1},
                                                                        {"FileConstantKernel", 1},
                                                                        {"InnerNetOutput", 1}};

  const std::map<std::string, size_t> deinit_graph_node_types_to_count = {{"InnerData", 1}, {"FreeMemory", 1}};
  EXPECT_TestLoweringFileConstantNode(graph_builder_wrapper, {init_graph_node_types_to_count},
                                      {deinit_graph_node_types_to_count}, lgd);
}

// 用例描述: 计算图上存在2个文件路径相同、shape相同的FileConstant节点，分别lowering，out_shape各自构造，out_addr复用
TEST_F(FileConstantConverterUT, LoweringFileConstantNode_Only1OutAddrResult_2SameFileConstantNodes) {
  LoweringGlobalData lgd;
  const std::vector<std::string> file_path_config({"test_weight_convert.bin", "test_weight_convert.bin"});
  const GraphBuilderWrapper graph_builder_wrapper{Build2FileConstantGraph, file_path_config};
  const std::map<std::string, size_t> init_graph_node_types_to_count_0 = {{"ConstData", 3},
                                                                          {"Const", 8},
                                                                          {"Data", 1},
                                                                          {"CalcTensorSizeFromShape", 1},
                                                                          {"CreateL1Allocator", 1},
                                                                          {"CreateL2Allocators", 1},
                                                                          {"CreateInitL2Allocator", 1},
                                                                          {"SplitRtStreams", 1},
                                                                          {"GetFileConstantWeightDir", 1},
                                                                          {"FileConstantKernel", 1},
                                                                          {"InnerNetOutput", 1}};
  const std::map<std::string, size_t> init_graph_node_types_to_count_1 = {{"ConstData", 3},
                                                                          {"Const", 9},
                                                                          {"Data", 1},
                                                                          {"CalcTensorSizeFromShape", 1},
                                                                          {"CreateL1Allocator", 1},
                                                                          {"CreateL2Allocators", 1},
                                                                          {"CreateInitL2Allocator", 1},
                                                                          {"SplitRtStreams", 1},
                                                                          {"GetFileConstantWeightDir", 1},
                                                                          {"FileConstantKernel", 1},
                                                                          {"InnerNetOutput", 1}};

  const std::map<std::string, size_t> deinit_graph_node_types_to_count = {{"InnerData", 1}, {"FreeMemory", 1}};
  EXPECT_TestLoweringFileConstantNode(graph_builder_wrapper,
                                      {init_graph_node_types_to_count_0, init_graph_node_types_to_count_1},
                                      {deinit_graph_node_types_to_count, deinit_graph_node_types_to_count}, lgd);
}

// 用例描述: 计算图上存在2个文件路径相同但是shape不同的FileConstant节点，分别lowering，out_shape各自构造，out_addr复用
TEST_F(FileConstantConverterUT, LoweringFileConstantNode_Only1OutAddrResult_SameLocationButDiffShape) {
  LoweringGlobalData lgd;
  const std::vector<std::string> file_path_config({"test_weight_convert.bin", "test_weight_convert.bin"});
  const GraphBuilderWrapper graph_builder_wrapper{Build2FileConstantGraphWithDiffShape, file_path_config};
  const std::map<std::string, size_t> init_graph_node_types_to_count_0 = {{"ConstData", 3},
                                                                          {"Const", 8},
                                                                          {"Data", 1},
                                                                          {"CalcTensorSizeFromShape", 1},
                                                                          {"CreateL1Allocator", 1},
                                                                          {"SplitRtStreams", 1},
                                                                          {"CreateL2Allocators", 1},
                                                                          {"CreateInitL2Allocator", 1},
                                                                          {"GetFileConstantWeightDir", 1},
                                                                          {"FileConstantKernel", 1},
                                                                          {"InnerNetOutput", 1}};
  const std::map<std::string, size_t> init_graph_node_types_to_count_1 = {{"ConstData", 3},
                                                                          {"Const", 9},
                                                                          {"Data", 1},
                                                                          {"CalcTensorSizeFromShape", 1},
                                                                          {"CreateL1Allocator", 1},
                                                                          {"SplitRtStreams", 1},
                                                                          {"CreateL2Allocators", 1},
                                                                          {"CreateInitL2Allocator", 1},
                                                                          {"GetFileConstantWeightDir", 1},
                                                                          {"FileConstantKernel", 1},
                                                                          {"InnerNetOutput", 1}};
  const std::map<std::string, size_t> deinit_graph_node_types_to_count_0 = {{"InnerData", 1}, {"FreeMemory", 1}};
  EXPECT_TestLoweringFileConstantNode(graph_builder_wrapper,
                                      {init_graph_node_types_to_count_0, init_graph_node_types_to_count_1},
                                      {deinit_graph_node_types_to_count_0, deinit_graph_node_types_to_count_0}, lgd);
}

// 用例描述: 计算图上存在2个文件路径不同的FileConstant节点，分别lowering，均有lowering结果
TEST_F(FileConstantConverterUT, LoweringFileConstantNode_Get2FileConstantLowerResult_2DiffFileConstantNodes) {
  LoweringGlobalData lgd;
  const std::vector<std::string> file_path_config({"test_weight_convert1.bin", "test_weight_convert2.bin"});
  const GraphBuilderWrapper graph_builder_wrapper{Build2FileConstantGraph, file_path_config};
  const std::map<std::string, size_t> init_graph_node_types_to_count_0 = {{"ConstData", 3},
                                                                          {"Const", 8},
                                                                          {"Data", 1},
                                                                          {"CalcTensorSizeFromShape", 1},
                                                                          {"CreateL1Allocator", 1},
                                                                          {"SplitRtStreams", 1},
                                                                          {"CreateL2Allocators", 1},
                                                                          {"CreateInitL2Allocator", 1},
                                                                          {"GetFileConstantWeightDir", 1},
                                                                          {"FileConstantKernel", 1},
                                                                          {"InnerNetOutput", 1}};
  const std::map<std::string, size_t> init_graph_node_types_to_count_1 = {{"ConstData", 3},
                                                                          {"Const", 12},
                                                                          {"Data", 1},
                                                                          {"CalcTensorSizeFromShape", 2},
                                                                          {"CreateL1Allocator", 1},
                                                                          {"SplitRtStreams", 1},
                                                                          {"CreateL2Allocators", 1},
                                                                          {"CreateInitL2Allocator", 1},
                                                                          {"GetFileConstantWeightDir", 2},
                                                                          {"FileConstantKernel", 2},
                                                                          {"InnerNetOutput", 1}};
  const std::map<std::string, size_t> deinit_graph_node_types_to_count_0 = {{"InnerData", 1}, {"FreeMemory", 1}};
  const std::map<std::string, size_t> deinit_graph_node_types_to_count_1 = {{"InnerData", 2}, {"FreeMemory", 2}};
  EXPECT_TestLoweringFileConstantNode(graph_builder_wrapper,
                                      {init_graph_node_types_to_count_0, init_graph_node_types_to_count_1},
                                      {deinit_graph_node_types_to_count_0, deinit_graph_node_types_to_count_1}, lgd);
}
}  // namespace gert
