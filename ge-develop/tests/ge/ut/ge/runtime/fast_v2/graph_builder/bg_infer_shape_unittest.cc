/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_infer_shape.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "engine/gelocal/inputs_converter.h"
#include "register/node_converter_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "register/op_impl_registry.h"
#include "common/const_data_helper.h"

#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "graph/buffer.h"
#include "kernel/common_kernel_impl/compatible_utils.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "common/share_graph.h"
#include "common/summary_checker.h"
#include "lowering/pass/constant_expression_motion.h"

namespace gert {
namespace bg {
namespace {
using namespace ge;
void SetNoStorage(const ge::OpDescPtr &op_desc, Format format, DataType dt, std::initializer_list<int64_t> in_shape,
                  std::initializer_list<int64_t> out_shape) {
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetFormat(format);
    op_desc->MutableInputDesc(i)->SetOriginFormat(format);
    op_desc->MutableInputDesc(i)->SetShape(GeShape(in_shape));
    op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(in_shape));
    op_desc->MutableInputDesc(i)->SetDataType(dt);
    op_desc->MutableInputDesc(i)->SetOriginDataType(dt);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetFormat(format);
    op_desc->MutableOutputDesc(i)->SetOriginFormat(format);
    op_desc->MutableOutputDesc(i)->SetShape(GeShape(out_shape));
    op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(out_shape));
    op_desc->MutableOutputDesc(i)->SetDataType(dt);
    op_desc->MutableOutputDesc(i)->SetOriginDataType(dt);
  }
}

/*
 *
 *   test_node
 *    /  \
 * data1 data2
 */
ComputeGraphPtr BuildSingleNodeGraph(const std::string &node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("test_node", node_type)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("test_node", node_type));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  SetNoStorage(data1->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8,3,224,224}, {8,3,224,224});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  SetNoStorage(data2->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8,3,224,224}, {8,3,224,224});
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto test_node = graph->FindNode("test_node");
  SetNoStorage(test_node->GetOpDesc(), FORMAT_NCHW, DT_FLOAT, {8,3,224,224}, {-1,-1,-1,-1});
  return graph;
}

ge::graphStatus InferShapeForAdd(InferShapeContext *context) {
  return SUCCESS;
}
IMPL_OP(Add).InferShape(InferShapeForAdd);
} // namespace
class BgInferShapeUT : public BgTestAutoCreateFrame {
 protected:
  void SetUp() override {
    ValueHolder::PushGraphFrame();
    auto init = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    auto main = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
    auto de_init = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});

    ValueHolder::PushGraphFrame(init, "Init");
    init_frame_ = ValueHolder::PopGraphFrame({}, {});

    ValueHolder::PushGraphFrame(de_init, "DeInit");
    de_init_frame_ = ValueHolder::PopGraphFrame();

    ValueHolder::PushGraphFrame(main, "Main");
  }

  void TearDown() override {
    BgTest::TearDown();
    init_frame_.reset();
    de_init_frame_.reset();
  }

  std::unique_ptr<GraphFrame> init_frame_;
  std::unique_ptr<GraphFrame> de_init_frame_;
};

// TEST_F(BgInferShapeUT, ConstructInferStorageShapeOk) {}

TEST_F(BgInferShapeUT, ConstructInferShapeOk) {
  auto graph = ShareGraph::AicoreGraph();
  auto add_node = graph->FindNode("add1");

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto out_shapes = InferStorageShape(nullptr, {}, global_data);
  ASSERT_EQ(out_shapes.size(), 0);

  out_shapes = InferStorageShape(add_node, {}, global_data);
  ASSERT_EQ(out_shapes.size(), 0);

  out_shapes = InferStorageShape(add_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, global_data);
  ASSERT_EQ(out_shapes.size(), 1);
  EXPECT_EQ(out_shapes[0]->GetFastNode()->GetType(), "InferShape");

  auto main_frame = ValueHolder::PopGraphFrame({out_shapes}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 4},
                                                                     {"InferShape", 1},
                                                                     {"Const", 1},
                                                                     {"SplitDataTensor", 2},
                                                                     {"InnerData", 4},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"ConstData", 3},
                                                                     {"Const", 6},
                                                                     {"Data", 1},
                                                                     {"CreateInitL2Allocator", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"GetSpaceRegistry", 1},
                                                                     {"FindInferShapeFunc", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);
}


/**
 * 测试当mul算子不支持新版infershape函数时，lowering产生v1版本的infershape执行图
 *
 *             CompatibleInferShape
 *            /       \     \________________________
 *           /         |                            |
 * all-shapes    FindCompatibleInferShapeFunc       op
 *                      |
 *                   node-type
 *
 * 校验点：
 * 1、校验执行图连边关系
 *    （1）FindCompatibleInferShapeFunc是InferShapeV1的第三个输入
 *    （2）operator buffer const是InferShapeV1的第一个输入
 * 2、校验operator const中的值是否正确
 */
TEST_F(BgInferShapeUT, ConstructV1InferShapeOk) {
  std::string node_type = "Dummy";
  auto graph = BuildSingleNodeGraph(node_type);
  auto test_node = graph->FindNode("test_node");

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore(node_type, false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto out_shapes = InferStorageShape(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, global_data);
  ASSERT_EQ(out_shapes.size(), 1);
  EXPECT_EQ(out_shapes[0]->GetFastNode()->GetType(), "CompatibleInferShape");

  // CHECK exec graph connection
  auto find_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(),
                                                                 "FindCompatibleInferShapeFunc");
  auto node_type_const = find_node->GetInDataNodes().at(0);
  ASSERT_NE(node_type_const, nullptr);
  // check node_type const content
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(node_type_const, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), node_type.c_str());
  EXPECT_TRUE(ExeGraphComparer::ExpectConnectTo(node_type_const, "FindCompatibleInferShapeFunc", 0, 0));

  // check op buffer content
  auto compatible_infer_shape = out_shapes[0]->GetFastNode();
  ConnectFromInitToMain(find_node, 0, compatible_infer_shape, 1);
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
  EXPECT_STREQ(test_op_desc->GetName().c_str(), "test_node");
  EXPECT_EQ(test_op_desc->GetType(), node_type);

  // check shape on operator
  auto input_0_shape = test_op_desc->GetInputDesc(0).GetShape().GetDims();
  EXPECT_EQ(input_0_shape[0], 8);
  EXPECT_EQ(input_0_shape[1], 3);
  EXPECT_EQ(input_0_shape[2], 224);
  EXPECT_EQ(input_0_shape[3], 224);
}

// TEST_F(BgInferShapeUT, InputNumDiff) {}
// TEST_F(BgInferShapeUT, DataDependency) {}

TEST_F(BgInferShapeUT, GetFrameworkOpNodeTypeOK) {
  const string real_node_type = "Add";
  auto graph = ShareGraph::FrameworkOPGraph(real_node_type);
  auto add_node = graph->FindNode("add1");

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto out_shapes = InferStorageShape(add_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, global_data);
  ASSERT_EQ(out_shapes.size(), 1);

  auto main_frame = ValueHolder::PopGraphFrame({out_shapes}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 4},
                                                                     {"InferShape", 1},
                                                                     {"SplitDataTensor", 2},
                                                                     {"InnerData", 4},
                                                                     {"Const", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"ConstData", 3},
                                                                     {"Const", 6},
                                                                     {"Data", 1},
                                                                     {"CreateInitL2Allocator", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"GetSpaceRegistry", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"FindInferShapeFunc", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);
}

TEST_F(BgInferShapeUT, GetCompatibleFrameworkOpNodeTypeOK) {
  const string real_node_type = "TestCompatibleNodeType";
  auto graph = ShareGraph::FrameworkOPGraph(real_node_type);
  auto add_node = graph->FindNode("add1");

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("TestCompatibleNodeType", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto out_shapes = InferStorageShape(add_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, global_data);
  ASSERT_EQ(out_shapes.size(), 1);

  auto find_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(),
                                                                 "FindCompatibleInferShapeFunc");
  ASSERT_EQ(find_node->GetInDataNodes().size(), 1);
  auto node_type = *find_node->GetInDataNodes().begin();
  ASSERT_NE(node_type, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(node_type, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), "TestCompatibleNodeType");
}
}  // namespace bg
}  // namespace gert