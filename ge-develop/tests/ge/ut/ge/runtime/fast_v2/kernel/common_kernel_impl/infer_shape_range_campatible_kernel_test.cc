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

#include "exe_graph/runtime/storage_shape.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "graph/operator_factory.h"
#include "graph/operator_factory_impl.h"
#include "graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_reg.h"
#include "common/util.h"
#include "depends/checker/shape_checker.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph_builder/bg_compatible_utils.h"

namespace gert {
namespace kernel {
ge::graphStatus InferShapeRange(KernelContext *context);
}

struct InferShapeRangeCompatibleKernelTest : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

ge::graphStatus StubCopyInferShapeRangeV1(ge::Operator &op) {
  std::vector<std::pair<int64_t, int64_t>> input_ranges;
  (void) op.GetInputDesc(0).GetShapeRange(input_ranges);
  const auto &op_dsec = ge::OpDescUtils::GetOpDescFromOperator(op);
  const auto &output_name = op_dsec->GetOutputNameByIndex(0);
  auto output_tensor = op.GetOutputDesc(0);
  output_tensor.SetShapeRange(input_ranges);
  op.UpdateOutputDesc(output_name.c_str(), output_tensor);
  return ge::GRAPH_SUCCESS;
}

// only support shape_data_type int64
// use op.GetInputConstData
ge::graphStatus StubInferShapeRangeDependV1(ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape"});
  auto x_desc = op_desc->MutableInputDesc("x");
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
  std::string shape_input_name = "shape";

  ge::Tensor shape_tensor;
  op.GetInputConstData(shape_input_name.c_str(), shape_tensor);

  ge::GeShape output_shape;
  auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
  const int64_t* shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(shape_tensor.GetData()));
  GE_CHECK_NOTNULL(shape_data);
  std::vector<int64_t> out_dims;
  std::vector<std::pair<int64_t, int64_t>> input_ranges;
  (void) x_desc->GetShapeRange(input_ranges);
  std::vector<std::pair<int64_t, int64_t>> output_ranges;
  // 输出range为输入range+value
  for (size_t i = 0U; i < input_ranges.size(); ++i) {
    auto min_range = input_ranges[i].first + (*shape_data);
    auto max_range = input_ranges[i].second + (*shape_data);
    out_dims.emplace_back(-1);
    output_ranges.emplace_back(std::pair<int64_t, int64_t>(min_range, max_range));
  }
  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape(out_dims));
  td->SetOriginShape(ge::GeShape(out_dims));
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetShapeRange(output_ranges);
  td->SetOriginShapeRange(output_ranges);
  return ge::GRAPH_SUCCESS;
}

// only support shape_data_type int64
// use OpDescUtils::GetInputConstData
ge::graphStatus StubReshapeInferShapeRangeV1WithOpDescUtils(ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape"});
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
  std::string shape_input_name = "shape";

  auto shape_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName("shape"));
  const ge::GeTensor *shape_tensor = ge::OpDescUtils::GetInputConstData(op, shape_idx);

  ge::GeShape output_shape;
  auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
  const int64_t* shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(shape_tensor->GetData().GetData()));
  GE_CHECK_NOTNULL(shape_data);
  std::vector<int64_t> out_dims;
  std::vector<std::pair<int64_t, int64_t>> input_ranges;
  (void) x_desc->GetShapeRange(input_ranges);
  std::vector<std::pair<int64_t, int64_t>> output_ranges;
  // 输出range为输入range+value
  for (size_t i = 0U; i < input_ranges.size(); ++i) {
    auto min_range = input_ranges[i].first + (*shape_data);
    auto max_range = input_ranges[i].second + (*shape_data);
    out_dims.emplace_back(-1);
    output_ranges.emplace_back(std::pair<int64_t, int64_t>(min_range, max_range));
  }
  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape(out_dims));
  td->SetOriginShape(ge::GeShape(out_dims));
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetShapeRange(output_ranges);
  td->SetOriginShapeRange(output_ranges);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus StubInfershapeRangeV1(ge::Operator &) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(InferShapeRangeCompatibleKernelTest, test_find_InferShapeFuncV1_fail) {
  auto run_context = BuildKernelRunContext(1, 1);
  std::string node_type = "stub_infershape";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FindCompatibleInferShapeRangeFunc")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(InferShapeRangeCompatibleKernelTest, test_find_InferShapeFuncV1_success) {
  auto run_context = BuildKernelRunContext(1, 1);
  std::string node_type = "stub_infershape";
  auto a = [](ge::Operator &v) { return StubInfershapeRangeV1(v);}; // simulate operator_reg
  ge::InferShapeFuncRegister(node_type.c_str(), a);
  ge::InferShapeFunc target_infer_func;
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  run_context.value_holder[1].Set(const_cast<ge::InferShapeFunc *>(&target_infer_func), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("FindCompatibleInferShapeRangeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto op_func = ge::OperatorFactoryImpl::GetInferShapeFunc(node_type);
  // build a operator
  ge::Operator op;
  auto infer_func_2 = reinterpret_cast<ge::InferShapeFunc *>(run_context.value_holder[1].GetValue<void *>());
  ASSERT_EQ((*infer_func_2)(op), ge::GRAPH_SUCCESS);
  ASSERT_EQ(target_infer_func(op), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeRangeCompatibleKernelTest, test_infershape_InferShapeRangeFuncV1_success) {
  // build operator with single input and single output
  ge::ComputeGraphPtr test_graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_NCHW)), ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)), ge::GRAPH_SUCCESS);
  auto test_node = test_graph->AddNode(desc_ptr);

  auto op_desc_data = std::make_shared<ge::OpDesc>("op_desc_data", "Data");
  op_desc_data->AddOutputDesc(ge::GeTensorDesc());
  auto data = test_graph->AddNode(op_desc_data);
  ge::GraphUtils::AddEdge(data->GetOutDataAnchor(0), test_node->GetInDataAnchor(0));

  ge::InferShapeFunc a = [](ge::Operator &v) { return StubCopyInferShapeRangeV1(v);}; // simulate operator_reg
  auto op = ge::OpDescUtils::CreateOperatorFromNode(test_node);

  auto run_context = KernelRunContextFaker()
                         .IrInstanceNum({1})
                         .NodeIoNum(1, 1)
                         .KernelIONum(3, 3)
                         .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_NCHW)
                         .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_NCHW)
                         .Build();
  Shape min1_range{1, 1, 1, 1};
  Shape max1_range{16, 64, -1, -1};
  Range<Shape> input_range{&min1_range, &max1_range};
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&op), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(&a), nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(&input_range), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShapeRange")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShapeRange")->run_func(run_context),
            ge::GRAPH_SUCCESS);
  auto out1 = run_context.value_holder[3].GetPointer<Range<Shape>>();
  auto out2 = run_context.value_holder[4].GetPointer<Shape>();
  auto out3 = run_context.value_holder[5].GetPointer<Shape>();
  ASSERT_NE(out1, nullptr);
  ASSERT_NE(out2, nullptr);
  ASSERT_NE(out3, nullptr);
  EXPECT_EQ(out1->GetMin(), out2);
  EXPECT_EQ(out1->GetMax(), out3);

  EXPECT_EQ(min1_range, *out2);
  EXPECT_EQ(max1_range, *out3);

  run_context.FreeValue(3);
  run_context.FreeValue(4);
  run_context.FreeValue(5);
}

// TestNoInferShapeRange from {{1, 16}, {1, 64}, {1, 64}, {1, 64}} to {{2, 18}, {2, 66}, {2, 66}, {1, 66}}
TEST_F(InferShapeRangeCompatibleKernelTest, test_infershape_InferShapeRangeFuncV1_with_dependency_success) {
  // build TestNoInferShapeRange operator
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "ReShape");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_NCHW)), ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("shape", ge::GeTensorDesc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT64)), ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)), ge::GRAPH_SUCCESS);
  desc_ptr->SetOpInferDepends({"shape"});

  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  auto test = tmp_graph->AddNode(desc_ptr);
  auto op_desc_data = std::make_shared<ge::OpDesc>("op_desc_data", "Data");
  op_desc_data->AddOutputDesc(ge::GeTensorDesc());
  auto data = tmp_graph->AddNode(op_desc_data);
  auto op_desc_data2 = std::make_shared<ge::OpDesc>("op_desc_data2", "Data");
  op_desc_data2->AddOutputDesc(ge::GeTensorDesc());
  auto data2 = tmp_graph->AddNode(op_desc_data2);
  ge::GraphUtils::AddEdge(data->GetOutDataAnchor(0), test->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), test->GetInDataAnchor(1));
  auto op = ge::OpDescUtils::CreateOperatorFromNode(test);

  ge::InferShapeFunc a = [](ge::Operator &v) { return StubInferShapeRangeDependV1(v);}; // simulate operator_reg

  auto run_context = KernelRunContextFaker()
                         .IrInstanceNum({2})
                         .NodeIoNum(2, 1)
                         .KernelIONum(4, 3)
                         .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_NCHW)
                         .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_NCHW)
                         .Build();
  gert::Tensor min_range_tensor = {{{1, 1, 1, 1}, {1, 1, 1, 1}},        // shape
                                   {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                                   kOnDeviceHbm,                        // placement
                                   ge::DT_FLOAT16,                      // data type
                                   (void *)0x0};
  gert::Tensor max_range_tensor = {{{16, 64, 64, 64}, {16, 64, 64, 64}},   // shape
                                   {ge::FORMAT_ND, ge::FORMAT_ND, {}},     // format
                                   kOnDeviceHbm,                           // placement
                                   ge::DT_FLOAT16,                         // data type
                                   (void *)0x0};
  TensorRange input_range{&min_range_tensor, &max_range_tensor};

  // build input shape & input tensor
  size_t total_size;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {1};
  input_tensor->MutableStorageShape() = {1};

  auto tensor_data = reinterpret_cast<int64_t *>(input_tensor->GetAddr());
  input_tensor->SetData(gert::TensorData{tensor_data, nullptr});
  tensor_data[0] = 2;
  TensorRange input_range2{input_tensor, input_tensor};

  run_context.value_holder[0].Set(reinterpret_cast<void *>(&op), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(&a), nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(&input_range), nullptr);
  run_context.value_holder[3].Set(reinterpret_cast<void *>(&input_range2), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShapeRange")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShapeRange")->run_func(run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(registry.FindKernelFuncs("CompatibleInferShapeRange")->trace_printer(run_context).empty());
  for (auto &s : registry.FindKernelFuncs("CompatibleInferShapeRange")->trace_printer(run_context)) {
    std::cout << s << std::endl;
  }

  StorageShape expect_out_min_shape({3, 3, 3, 3}, {3, 3, 3, 3});
  StorageShape expect_out_max_shape({18, 66, 66, 66}, {18, 66, 66, 66});
  auto infer_range_context = run_context.GetContext<KernelContext>();
  auto shape_range = infer_range_context->GetOutputPointer<Range<Shape>>(0UL);
  ASSERT_NE(shape_range, nullptr);
  ShapeChecker::CheckShape(*shape_range->GetMin(), expect_out_min_shape.GetOriginShape());
  ShapeChecker::CheckShape(*shape_range->GetMax(), expect_out_max_shape.GetOriginShape());

  auto tensor_range = infer_range_context->GetOutputPointer<TensorRange>(0UL);
  ASSERT_NE(tensor_range, nullptr);
  ShapeChecker::CheckShape(tensor_range->GetMin()->GetOriginShape(), expect_out_min_shape.GetOriginShape());
  EXPECT_EQ(tensor_range->GetMin()->GetStorageShape().GetDimNum(), 0UL);
  EXPECT_EQ(tensor_range->GetMin()->GetAddr(), nullptr);
  ShapeChecker::CheckShape(tensor_range->GetMax()->GetOriginShape(), expect_out_max_shape.GetOriginShape());
  ShapeChecker::CheckShape(tensor_range->GetMax()->GetStorageShape(), expect_out_max_shape.GetStorageShape());
  EXPECT_EQ(tensor_range->GetMax()->GetAddr(), nullptr);
  run_context.FreeAll();
}
}