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
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "graph/operator_factory.h"
#include "graph/operator_factory_impl.h"
#include "graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_reg.h"

#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "graph/buffer.h"
#include "graph_builder/bg_compatible_utils.h"

namespace gert {
namespace kernel {
ge::graphStatus InferShape(KernelContext *context);
}

struct InferShapeCompatibleKernelTest : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

ge::graphStatus StubCopyInferShapeV1(ge::Operator &op) {
  auto input_shape = op.GetInputDesc(0).GetShape();
  const auto &op_dsec = ge::OpDescUtils::GetOpDescFromOperator(op);
  const auto &output_name = op_dsec->GetOutputNameByIndex(0);
  auto output_tensor = op.GetOutputDesc(0);
  output_tensor.SetShape(input_shape);
  op.UpdateOutputDesc(output_name.c_str(), output_tensor);
  // check get origin format success
  auto input_format = op.GetInputDesc(0).GetFormat();
  auto output_format = op.GetOutputDescByName("y").GetFormat();
  if ((input_format != ge::FORMAT_ND) || (output_format != ge::FORMAT_ND)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// only support shape_data_type int64
// use op.GetInputConstData
ge::graphStatus StubReshapeInferShapeV1(ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape"});
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
  std::string shape_input_name = "shape";

  ge::Tensor shape_tensor;
  op.GetInputConstData(shape_input_name.c_str(), shape_tensor);
  // const ge::GeTensor *tensor = ge::OpDescUtils::GetInputConstData(op, shape_idx);

  ge::GeShape output_shape;
  auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
  auto &shape_ref = shape_tensor_desc->MutableShape();
  auto shape_dims = shape_ref.GetDims();
  // stub shape data type int64
  int64_t dim_num = shape_dims[0];
  const int64_t *shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(shape_tensor.GetData()));
  std::vector<int64_t> out_dims;
  int64_t product = 1;
  for (int64_t i = 0; i < dim_num; i++) {
    auto dim = shape_data[i];
    if (dim != 0 && product > (INT64_MAX / dim)) {
      return ge::GRAPH_PARAM_INVALID;
    }
    out_dims.push_back(dim);
    product *= dim;
  }

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape(out_dims));
  td->SetOriginShape(ge::GeShape(out_dims));
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StubOptionalInputInferShapeV1(ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(1);
  auto out_desc = op_desc->MutableOutputDesc(0);
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());

  out_desc->SetShape(ge::GeShape(x_shape));
  out_desc->SetOriginShape(ge::GeShape(x_shape));
  out_desc->SetDataType(x_desc->GetDataType());
  out_desc->SetOriginDataType(x_desc->GetDataType());
  return ge::GRAPH_SUCCESS;
}

// only support shape_data_type int64
// use OpDescUtils::GetInputConstData
ge::graphStatus StubReshapeInferShapeV1WithOpDescUtils(ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape"});
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
  std::string shape_input_name = "shape";

  auto shape_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName("shape"));
  const ge::GeTensor *tensor = ge::OpDescUtils::GetInputConstData(op, shape_idx);

  ge::GeShape output_shape;
  auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
  auto &shape_ref = shape_tensor_desc->MutableShape();
  auto shape_dims = shape_ref.GetDims();
  // stub shape data type int64
  int64_t dim_num = shape_dims[0];
  const int64_t *shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(tensor->GetData().GetData()));
  std::vector<int64_t> out_dims;
  int64_t product = 1;
  for (int64_t i = 0; i < dim_num; i++) {
    auto dim = shape_data[i];
    if (dim != 0 && product > (INT64_MAX / dim)) {
      return ge::GRAPH_PARAM_INVALID;
    }
    out_dims.push_back(dim);
    product *= dim;
  }

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape(out_dims));
  td->SetOriginShape(ge::GeShape(out_dims));
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus StubInfershapeV1(ge::Operator &) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(InferShapeCompatibleKernelTest, test_create_op_from_buffer_success) {
  StorageShape input{{1, 2, 2, 256}, {1, 2, 2, 256}};
  StorageShape output;
  // build operator with single input and single output
  ge::ComputeGraphPtr test_graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  auto test_node = test_graph->AddNode(desc_ptr);
  auto op_desc_data = std::make_shared<ge::OpDesc>("op_desc_data", "Data");
  op_desc_data->AddOutputDesc(ge::GeTensorDesc());
  auto data = test_graph->AddNode(op_desc_data);
  ge::GraphUtils::AddEdge(data->GetOutDataAnchor(0), test_node->GetInDataAnchor(0));

  auto buffer = bg::CompatibleUtils::SerializeNode(test_node);
  auto op = new (std::nothrow) ge::Operator();

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set(buffer.GetData(), nullptr);
  run_context.value_holder[1].Set((void *)buffer.GetSize(), nullptr);
  run_context.value_holder[2].SetWithDefaultDeleter(op);
  ASSERT_EQ(registry.FindKernelFuncs("CreateOpFromBuffer")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op->GetName(), "test");
}

TEST_F(InferShapeCompatibleKernelTest, test_find_InferShapeFuncV1_fail) {
  auto run_context = BuildKernelRunContext(1, 1);
  std::string node_type = "stub_infershape";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FindCompatibleInferShapeFunc")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(InferShapeCompatibleKernelTest, test_find_InferShapeRangeFuncV1_success) {
  auto run_context = BuildKernelRunContext(1, 1);
  std::string node_type = "stub_infershape";
  auto a = [](ge::Operator &v) { return StubInfershapeV1(v); };  // simulate operator_reg
  ge::InferShapeFuncRegister(node_type.c_str(), a);
  ge::InferShapeFunc target_infer_func;
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  run_context.value_holder[1].Set(const_cast<ge::InferShapeFunc *>(&target_infer_func), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("FindCompatibleInferShapeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto op_func = ge::OperatorFactoryImpl::GetInferShapeFunc(node_type);
  // build a operator
  ge::Operator op;
  auto infer_func_2 = reinterpret_cast<ge::InferShapeFunc *>(run_context.value_holder[1].GetValue<void *>());
  ASSERT_EQ((*infer_func_2)(op), ge::GRAPH_SUCCESS);
  ASSERT_EQ(target_infer_func(op), ge::GRAPH_SUCCESS);
}

/*
 * 补充一个异常场景UT，兼容模式下InfershapeRange函数找不到返回失败
 * */
TEST_F(InferShapeCompatibleKernelTest, test_find_InferShapeRangeFuncV1_failed) {
  auto run_context = BuildKernelRunContext(1, 1);
  std::string node_type = "no_infershaperange";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  run_context.value_holder[1].Set(nullptr, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("FindCompatibleInferShapeFunc")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(InferShapeCompatibleKernelTest, test_infershape_InferShapeRangeFuncV1_success) {
  StorageShape input{{1, 2, 2, 256}, {1, 2, 2, 256}};
  StorageShape output;
  // build operator with single input and single output
  ge::ComputeGraphPtr test_graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  auto test_node = test_graph->AddNode(desc_ptr);
  auto op_desc_data = std::make_shared<ge::OpDesc>("op_desc_data", "Data");
  op_desc_data->AddOutputDesc(ge::GeTensorDesc());
  auto data = test_graph->AddNode(op_desc_data);
  ge::GraphUtils::AddEdge(data->GetOutDataAnchor(0), test_node->GetInDataAnchor(0));

  ge::InferShapeFunc a = [](ge::Operator &v) { return StubCopyInferShapeV1(v); };  // simulate operator_reg
  auto op = ge::OpDescUtils::CreateOperatorFromNode(test_node);
  auto run_context = KernelRunContextFaker()
                         .NodeIoNum(1, 1)
                         .IrInputNum(1)
                         .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_NCHW)
                         .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_NCHW)
                         .Inputs({&op, &a, &input})
                         .Outputs({&output})
                         .Build();

  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShape")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(output.GetStorageShape().GetDimNum(), 4);
  ASSERT_EQ(output.GetOriginShape().GetDimNum(), 4);
  ASSERT_EQ(output.GetStorageShape().GetShapeSize(), input.GetStorageShape().GetShapeSize());
  ASSERT_EQ(output.GetOriginShape().GetShapeSize(), input.GetOriginShape().GetShapeSize());
}

// reshape from {2,24} to {24,2}
TEST_F(InferShapeCompatibleKernelTest, test_infershape_InferShapeFuncV1_with_dependency_success) {
  // build reshape operator
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("reshape", "Reshape");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("shape", ge::GeTensorDesc(ge::GeShape({2}), ge::FORMAT_NCHW, ge::DT_INT64)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  desc_ptr->SetOpInferDepends({"shape"});

  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  auto reshape = tmp_graph->AddNode(desc_ptr);
  auto op_desc_data = std::make_shared<ge::OpDesc>("op_desc_data", "Data");
  op_desc_data->AddOutputDesc(ge::GeTensorDesc());
  auto data = tmp_graph->AddNode(op_desc_data);
  auto op_desc_data2 = std::make_shared<ge::OpDesc>("op_desc_data2", "Data");
  op_desc_data2->AddOutputDesc(ge::GeTensorDesc());
  auto data2 = tmp_graph->AddNode(op_desc_data2);
  ge::GraphUtils::AddEdge(data->GetOutDataAnchor(0), reshape->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), reshape->GetInDataAnchor(1));
  auto op = ge::OpDescUtils::CreateOperatorFromNode(reshape);

  // build input shape & input tensor
  gert::Tensor input_shape_tensor = {{{2, 24}, {2, 24}},    // shape
                                     {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                                     kOnDeviceHbm,                                // placement
                                     ge::DT_FLOAT16,                              // data type
                                     (void *)0x0};
  size_t total_size;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(2, ge::DT_INT64, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {2};
  input_tensor->MutableStorageShape() = {2};

  auto tensor_data = reinterpret_cast<int64_t *>(input_tensor->GetAddr());
  input_tensor->SetData(gert::TensorData{tensor_data, nullptr});
  tensor_data[0] = 24;
  tensor_data[1] = 2;
  gert::Tensor output_shape_tensor;
  ge::InferShapeFunc infer_func_a = [](ge::Operator &v) {
    return StubReshapeInferShapeV1(v);
  };  // simulate operator_reg
  auto run_context = KernelRunContextFaker()
                          .NodeIoNum(2, 1)
                          .IrInputNum(2)
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_NCHW)
                          .Inputs({&op, &infer_func_a, &input_shape_tensor, input_tensor})
                          .Outputs({&output_shape_tensor})
                          .Build();

  run_context.value_holder[0].Set(&op, nullptr);
  run_context.value_holder[1].Set((void *)&infer_func_a, nullptr);  // inferfunc just copy shape from input to output
  run_context.value_holder[2].Set(&input_shape_tensor, nullptr);
  run_context.value_holder[3].Set(input_tensor, nullptr);
  run_context.value_holder[4].Set(&output_shape_tensor, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShape")->outputs_creator(nullptr, run_context),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShape")->run_func(run_context), ge::GRAPH_SUCCESS);
  // todo uncomment below after metadef fix
  auto output_shape = run_context.GetContext<KernelContext>()->GetOutputPointer<StorageShape>(0);
  ASSERT_NE(output_shape, nullptr);
  EXPECT_EQ(output_shape->GetStorageShape().GetDimNum(), 2);
  EXPECT_EQ(output_shape->GetOriginShape().GetDimNum(), 2);
  EXPECT_EQ(output_shape->GetStorageShape().GetShapeSize(), input_shape_tensor.GetStorageShape().GetShapeSize());
  EXPECT_EQ(output_shape->GetOriginShape().GetShapeSize(), input_shape_tensor.GetOriginShape().GetShapeSize());
  EXPECT_EQ(output_shape->GetOriginShape().GetDim(0), 24);
  EXPECT_EQ(output_shape->GetOriginShape().GetDim(1), 2);
  auto output_tensor = run_context.GetContext<KernelContext>()->GetOutputPointer<Tensor>(0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(output_tensor->GetStorageShape().GetDimNum(), 2);
  EXPECT_EQ(output_tensor->GetOriginShape().GetDimNum(), 2);
  EXPECT_EQ(output_tensor->GetStorageShape().GetShapeSize(), input_shape_tensor.GetStorageShape().GetShapeSize());
  EXPECT_EQ(output_tensor->GetOriginShape().GetShapeSize(), input_shape_tensor.GetOriginShape().GetShapeSize());
  EXPECT_EQ(output_tensor->GetOriginShape().GetDim(0), 24);
  EXPECT_EQ(output_tensor->GetOriginShape().GetDim(1), 2);
  EXPECT_EQ(output_tensor->GetAddr(), nullptr);

  gert::Tensor output_shape_tensor2;
  // test infershape func using op_desc_utils
  ge::InferShapeFunc infer_func_b = [](ge::Operator &v) {
    return StubReshapeInferShapeV1WithOpDescUtils(v);
  };                                                                // simulate operator_reg
  run_context.value_holder[1].Set((void *)&infer_func_b, nullptr);  // inferfunc just copy shape from input to output
  run_context.value_holder[4].Set(&output_shape_tensor2, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShape")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto output_shape2 = run_context.GetContext<KernelContext>()->GetOutputPointer<StorageShape>(0);
  EXPECT_EQ(output_shape2->GetStorageShape().GetDimNum(), 2);
  EXPECT_EQ(output_shape2->GetOriginShape().GetDimNum(), 2);
  EXPECT_EQ(output_shape2->GetStorageShape().GetShapeSize(), input_shape_tensor.GetStorageShape().GetShapeSize());
  EXPECT_EQ(output_shape2->GetOriginShape().GetShapeSize(), input_shape_tensor.GetOriginShape().GetShapeSize());
  EXPECT_EQ(output_shape2->GetOriginShape().GetDim(0), 24);
  EXPECT_EQ(output_shape2->GetOriginShape().GetDim(1), 2);
  auto output_tensor2 = run_context.GetContext<KernelContext>()->GetOutputPointer<Tensor>(0);
  EXPECT_EQ(output_tensor2->GetStorageShape().GetDimNum(), 2);
  EXPECT_EQ(output_tensor2->GetOriginShape().GetDimNum(), 2);
  EXPECT_EQ(output_tensor2->GetStorageShape().GetShapeSize(), input_shape_tensor.GetStorageShape().GetShapeSize());
  EXPECT_EQ(output_tensor2->GetOriginShape().GetShapeSize(), input_shape_tensor.GetOriginShape().GetShapeSize());
  EXPECT_EQ(output_tensor2->GetOriginShape().GetDim(0), 24);
  EXPECT_EQ(output_tensor2->GetOriginShape().GetDim(1), 2);
  EXPECT_EQ(output_tensor2->GetAddr(), nullptr);
  run_context.FreeAll();
}

TEST_F(InferShapeCompatibleKernelTest, test_infershape_InferShapeFuncV1_withOptionalInput_success) {
  // build reshape operator
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("foo", "Foo");
  EXPECT_EQ(desc_ptr->AddOptionalInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_RESERVED, ge::DT_UNDEFINED)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("y", ge::GeTensorDesc(ge::GeShape({2}), ge::FORMAT_NCHW, ge::DT_INT64)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("z", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);

  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  auto foo_node = tmp_graph->AddNode(desc_ptr);
  auto op_desc_data = std::make_shared<ge::OpDesc>("op_desc_data", "Data");
  op_desc_data->AddOutputDesc(ge::GeTensorDesc());
  auto data = tmp_graph->AddNode(op_desc_data);
  ge::GraphUtils::AddEdge(data->GetOutDataAnchor(0), foo_node->GetInDataAnchor(1));
  auto op = ge::OpDescUtils::CreateOperatorFromNode(foo_node);

  // build input shape & input tensor
  gert::StorageShape input_shape = {{2, 24}, {2, 24}};
  gert::StorageShape output_shape = {{}, {}};

  ge::InferShapeFunc infer_func_a = [](ge::Operator &v) {
    return StubOptionalInputInferShapeV1(v);
  };  // simulate operator_reg
  auto run_context = KernelRunContextFaker()
      .NodeIoNum(1, 1)
      .IrInputNum(1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_NCHW)
      .Inputs({&op, &infer_func_a, &input_shape})
      .Outputs({&output_shape})
      .Build();

  run_context.value_holder[0].Set(&op, nullptr);
  run_context.value_holder[1].Set((void *)&infer_func_a, nullptr);  // inferfunc just copy shape from input to output
  run_context.value_holder[2].Set(&input_shape, nullptr);
  run_context.value_holder[3].Set(&output_shape, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("CompatibleInferShape")->run_func(run_context), ge::GRAPH_SUCCESS);

  ASSERT_EQ(output_shape.GetStorageShape().GetDimNum(), 2);
  ASSERT_EQ(output_shape.GetOriginShape().GetDimNum(), 2);
  ASSERT_EQ(output_shape.GetStorageShape().GetShapeSize(), input_shape.GetStorageShape().GetShapeSize());
  ASSERT_EQ(output_shape.GetOriginShape().GetShapeSize(), input_shape.GetOriginShape().GetShapeSize());
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 2);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 24);
}

/*
 * Trace 异常场景: 如果context的输入小于预期个数，则trace会打印相关异常信息，而非正常打印
 * CompatibleInferShapeRange/CompatibleInferShape: 至少两个以上输入，前两个为operator跟infershaperange v1 func,
 * 剩下为输入shape
 * */
TEST_F(InferShapeCompatibleKernelTest, test_InferShapeV1FuncTrace_failed) {
  auto run_context = BuildKernelRunContext(1, 1);
  std::string node_type = "CompatibleInferShapeFunc";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  auto ret1 = registry.FindKernelFuncs("CompatibleInferShape")->trace_printer(run_context);
  ASSERT_EQ(ret1.size(), 3);
  ASSERT_STREQ(ret1[1].c_str(),
               "input original shapes : Trace failed, input num < input_shape_start_index, context->GetInputNum:1, "
               "input_shape_start_index:2");

  auto ret2 = registry.FindKernelFuncs("CompatibleInferShapeRange")->trace_printer(run_context);
  ASSERT_EQ(ret1.size(), 3);
  ASSERT_STREQ(ret2[1].c_str(),
               "Trace failed, input num < input_range_start_index, context->GetInputNum:1, input_range_start_index:2");
}
}  // namespace gert