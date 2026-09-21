/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_infer_shape_range.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "engine/gelocal/inputs_converter.h"
#include "engine/aicore/converter/aicore_node_converter.h"
#include "register/node_converter_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "register/op_impl_registry.h"
#include "exe_graph/lowering/lowering_global_data.h"

#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "graph/buffer.h"
#include "kernel/common_kernel_impl/compatible_utils.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "common/share_graph.h"
#include "faker/aicpu_taskdef_faker.h"
#include "common/const_data_helper.h"

namespace gert {
namespace bg {
namespace {
using namespace ge;
ge::graphStatus InferShapeForAdd(InferShapeContext *context) {
  return SUCCESS;
}
IMPL_OP(Add).InferShape(InferShapeForAdd);
} // namespace
class BgInferShapeRangeUT : public BgTestAutoCreate3StageFrame {};

TEST_F(BgInferShapeRangeUT, NodeNullFailed) {
  LoweringGlobalData global_data;
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto ret = bg::InferShapeRange(nullptr, {}, global_data);
  EXPECT_FALSE(ret.IsSuccess());
}

TEST_F(BgInferShapeRangeUT, InputNumDiffFailed) {
  auto graph = ShareGraph::AicoreGraph();
  auto add_node = graph->FindNode("add1");
  LoweringGlobalData global_data;
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto ret = bg::InferShapeRange(add_node, {}, global_data);
  EXPECT_FALSE(ret.IsSuccess());
}

TEST_F(BgInferShapeRangeUT, ConstructInferShapeRangeOk) {
  auto graph = ShareGraph::ThirdAicpuOpGraph();
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);

  auto non_zero_node = graph->FindNode("nonzero");

  auto infer_shape_range_result = bg::InferShapeRange(non_zero_node,
                                                      {add_ret.out_shapes[0], data2_ret.out_shapes[0]},
                                                      global_data);
  ASSERT_TRUE(infer_shape_range_result.IsSuccess());
  auto max_shapes = infer_shape_range_result.GetAllMaxShapes();
  ASSERT_EQ(max_shapes.size(), 1);

  EXPECT_EQ(max_shapes[0]->GetFastNode()->GetType(), "InferShapeRange");

  auto exe_graph = max_shapes[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();

  auto find_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "FindInferShapeRangeFunc");
  ASSERT_NE(find_node, nullptr);
  ASSERT_EQ(find_node->GetInDataNodes().size(), 2);
  auto node_type = *find_node->GetInDataNodes().begin();
  ASSERT_NE(node_type, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(node_type, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), "NonZero");

  EXPECT_TRUE(ExeGraphComparer::ExpectConnectTo(find_node, "InferShapeRange", 0, 2));
}

TEST_F(BgInferShapeRangeUT, ConstructInferShapeRangeNullNodeFailed) {
  LoweringGlobalData global_data;
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  auto result = bg::InferShapeRange(nullptr, {}, global_data);
  ASSERT_FALSE(result.IsSuccess());
}

TEST_F(BgInferShapeRangeUT, ConstructInferShapeRangeInputsNumNotMatchFailed) {
  auto graph = ShareGraph::AicoreGraph();
  auto add_node = graph->FindNode("add1");

  LoweringGlobalData global_data;
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  auto result = bg::InferShapeRange(add_node, {}, global_data);
  ASSERT_FALSE(result.IsSuccess());
}

TEST_F(BgInferShapeRangeUT, GetAllShapeRangesOk) {
  auto graph = ShareGraph::ThirdAicpuOpGraph();

  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);

  auto non_zero_node = graph->FindNode("nonzero");

  auto infer_shape_range_result = bg::InferShapeRange(non_zero_node,
                                                      {data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                                                      global_data);
  ASSERT_TRUE(infer_shape_range_result.IsSuccess());
  auto shape_ranges = infer_shape_range_result.GetAllShapeRanges();
  ASSERT_EQ(shape_ranges.size(), 1U);

  EXPECT_EQ(shape_ranges[0]->GetFastNode()->GetType(), "InferShapeRange");
}

TEST_F(BgInferShapeRangeUT, GetMinShapeRangesOk) {
  auto graph = ShareGraph::ThirdAicpuOpGraph();

  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);

  auto non_zero_node = graph->FindNode("nonzero");

  auto infer_shape_range_result = bg::InferShapeRange(non_zero_node,
                                                      {data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                                                      global_data);
  ASSERT_TRUE(infer_shape_range_result.IsSuccess());
  auto shape_ranges = infer_shape_range_result.GetAllMinShapes();
  ASSERT_EQ(shape_ranges.size(), 1U);

  EXPECT_EQ(shape_ranges[0]->GetFastNode()->GetType(), "InferShapeRange");
}


TEST_F(BgInferShapeRangeUT, InferMaxShapeOk) {
  auto graph = ShareGraph::ThirdAicpuOpGraph();

  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCore.c_str());
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);

  auto non_zero_node = graph->FindNode("nonzero");

  auto max_shape_holders = bg::InferMaxShape(non_zero_node,
                                             {data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                                             global_data);
  ASSERT_EQ(max_shape_holders.size(), 1U);

  EXPECT_EQ(max_shape_holders[0]->GetFastNode()->GetType(), "InferShapeRange");
}

TEST_F(BgInferShapeRangeUT, InferMaxShapeNullNodeFailed) {
  LoweringGlobalData global_data;
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  auto max_shape_holders = bg::InferMaxShape(nullptr, {}, global_data);
  ASSERT_TRUE(max_shape_holders.empty());
}


/**
 * 测试当mul算子不支持新版infershape函数时，lowering产生v1版本的infershape执行图
 *
 *         CompatibleInferShapeRange
 *            /       \     \_________________________
 *           /         |                            |
 * all-shapes  FindCompatibleInferShapeRangeFunc       op
 *                      |
 *                   node-type
 *
 * 校验点：
 * 1、校验执行图连边关系
 *    （1）FindCompatibleInferShapeFunc是InferShapeV1的第三个输入
 *    （2）operator buffer const是InferShapeV1的第一个输入
 * 2、校验operator const中的值是否正确
 */
TEST_F(BgInferShapeRangeUT, ConstructV1InferShapeRangeOk) {
  auto graph = ShareGraph::BuildCompatibleInferShapeRangeGraph();

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("TestNoInferShapeRange", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto test_no_infer = graph->FindNode("test_no_infer");
  auto max_shape_holders = bg::InferMaxShape(test_no_infer, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                                             global_data);
  ASSERT_EQ(max_shape_holders.size(), 1U);
  EXPECT_EQ(max_shape_holders[0]->GetFastNode()->GetType(), "CompatibleInferShapeRange");
  auto exe_graph = max_shape_holders[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();

  // CHECK exec graph connection
  auto find_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "FindCompatibleInferShapeRangeFunc");
  auto node_type_const = find_node->GetInDataNodes().at(0);
  ASSERT_NE(node_type_const, nullptr);
  // check node_type const content
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(node_type_const, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), "TestNoInferShapeRange");
  EXPECT_TRUE(ExeGraphComparer::ExpectConnectTo(node_type_const, "FindCompatibleInferShapeRangeFunc", 0, 0));

  // check op buffer content
  auto compatible_infer_shape = max_shape_holders[0]->GetFastNode();
  EXPECT_TRUE(ExeGraphComparer::ExpectConnectTo(find_node, "CompatibleInferShapeRange", 0, 1));
  auto create_op_from_buffer = compatible_infer_shape->GetInDataNodes().at(0);
  EXPECT_EQ(create_op_from_buffer->GetType(), "CreateOpFromBuffer");

  // check mul operator content
  auto op_buffer_const = create_op_from_buffer->GetInDataNodes().at(0);
  auto op_buffer_size_const = create_op_from_buffer->GetInDataNodes().at(1);
  ge::Buffer op_buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(op_buffer_const, op_buffer));
  ASSERT_NE(op_buffer.GetData(), nullptr);
  ge::Buffer op_buffer_size_holder;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(op_buffer_size_const, op_buffer_size_holder));
  ASSERT_NE(op_buffer_size_holder.GetData(), nullptr);
  auto op_buffer_size = reinterpret_cast<size_t *>(op_buffer_size_holder.GetData());

  ge::OpDescPtr test_op_desc = nullptr;
  kernel::KernelCompatibleUtils::UnSerializeOpDesc(op_buffer.GetData(), *op_buffer_size, test_op_desc);
  EXPECT_STREQ(test_op_desc->GetName().c_str(), "test_no_infer");
  EXPECT_EQ(test_op_desc->GetType(), "TestNoInferShapeRange");

  // check shape on operator
  auto input_0_shape = test_op_desc->GetInputDesc(0).GetShape().GetDims();
  EXPECT_EQ(input_0_shape[0], -1);
  EXPECT_EQ(input_0_shape[1], 2);
  EXPECT_EQ(input_0_shape[2], 3);
  EXPECT_EQ(input_0_shape[3], 4);
}
}  // namespace bg
}  // namespace gert