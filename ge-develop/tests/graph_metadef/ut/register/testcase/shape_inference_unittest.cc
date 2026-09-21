/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator_factory_impl.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "register/shape_inference.h"
#include "utils/op_desc_utils.h"
#include <gtest/gtest.h>
#include "graph/utils/graph_utils.h"
#include "graph/attr_value.h"
#include "graph/operator_factory.h"
#include "common/ge_common/ge_inner_error_codes.h"

#include <op_desc_utils_ex.h>
#include "graph/op_desc.h"
#include "graph/normal_graph/op_desc_impl.h"

#include <dlog_pub.h>
#include <ge_common/debug/ge_log.h>
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/inference_context.h"
#include "graph/ct_infer_shape_context.h"
#include "graph/ct_infer_shape_range_context.h"
#include "graph/utils/tensor_adapter.h"
#include "faker/space_registry_faker.h"
#include "graph/custom_op_factory.h"

namespace ge{
REG_OP(Const)
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
        .ATTR(value, Tensor, Tensor())
        .OP_END_FACTORY_REG(Const);
}
namespace gert {
using namespace ge;
class MyCustomOp : public BaseCustomOp {
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  };
};
REG_AUTO_MAPPING_OP(MyCustomOp);

class ShapeInferenceUT : public testing::Test {};
// infer from output
REG_OP(FixIOOp_OutputIsFix)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T")
    .OUTPUT(fix_output, "T2")
    .DATATYPE(T2, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(FixIOOp_OutputIsFix);
// 无可选输入，无动态输入，正常流程，infer shape & infer data type
TEST_F(ShapeInferenceUT, CallInferV2Func_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };
  IMPL_OP(FixIOOp_OutputIsFix).InferShape(infer_shape_func)
      .InferDataType(infer_data_type_func)
      .InferShapeRange(infer_shape_range_func)
      .OutputShapeDependOnCompute({0});

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FixIOOp_OutputIsFix");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();

  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  ASSERT_NE(call_infer_data_type, nullptr);
  ASSERT_NE(call_infer_shape_v2, nullptr);
  ASSERT_NE(call_infer_shape_range, nullptr);
  auto status = call_infer_data_type(op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  uint64_t unknown_shape_type;
  ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type));
  ASSERT_EQ(unknown_shape_type, 3U);
  status = call_infer_shape_range(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 4);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
}

REG_OP(OptionalInput3Input3Output)
    .INPUT(input1, "T")
    .OPTIONAL_INPUT(input2, "T")
    .INPUT(input3, "T")
    .OUTPUT(output1, "T2")
    .OUTPUT(output2, "T2")
    .OUTPUT(output3, "T2")
    .DATATYPE(T2, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(OptionalInput3Input3Output);
// 未实例化的optional input测试
TEST_F(ShapeInferenceUT, CallInferV2Func_OptionalInputWithOutInstance) {
  auto op = OperatorFactory::CreateOperator("test2", "OptionalInput3Input3Output");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  // input1
  GeShape shape1({1, 2, 3, 4});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc2);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto option_input_shape = context->GetOptionalInputShape(1U);
    if (option_input_shape != nullptr) {
      return GRAPH_FAILED;
    }
    auto output = context->GetOutputShape(0);
    const auto input_shape = context->GetInputShape(1U);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  IMPL_OP(OptionalInput3Input3Output).InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("OptionalInput3Input3Output");
  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_shape_range = nullptr;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 实例化的optional input测试
TEST_F(ShapeInferenceUT, CallInferV2Func_OptionalInputWithInstance) {
  auto op = OperatorFactory::CreateOperator("test3", "OptionalInput3Input3Output");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  // input1
  GeShape shape1({1, 2, 3, 4});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input2
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // input3
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    // update option input to output0
    const auto input_shape = context->GetOptionalInputShape(1U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    // update input3 to output2
    const auto input_shape2 = context->GetInputShape(2U);
    auto output2 = context->GetOutputShape(1);
    for (size_t dim = 0UL; dim < input_shape2->GetDimNum(); dim++) {
      output2->AppendDim(input_shape2->GetDim(dim));
    }
    output2->SetDimNum(input_shape2->GetDimNum());
    return GRAPH_SUCCESS;
  };
  IMPL_OP(OptionalInput3Input3Output).InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("OptionalInput3Input3Output");
  op_impl_func->infer_shape = infer_shape_func;


  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  const auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
}

REG_OP(TwoOptionalInputsOp)
    .INPUT(input0, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output1, TensorType({DT_FLOAT16, DT_INT8))
    .OP_END_FACTORY_REG(TwoOptionalInputsOp);
// 只实例化第二个可选输入，预期第一个可选输入dtype为undefined，第二个可选输入为算子上的dtype
TEST_F(ShapeInferenceUT, CallInferV2Func_Rt2ConetxtGetRightDtype_JustInstantializeOptionalInputs) {
  auto op = OperatorFactory::CreateOperator("test2", "TwoOptionalInputsOp");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  // input0
  GeShape shape0({1, 2, 3, 4});
  GeTensorDesc tensor_desc0(shape0, Format::FORMAT_NCHW, DT_FLOAT);
  tensor_desc0.SetOriginShape(shape0);
  tensor_desc0.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc0);
  // optional input 1
  GeShape shape1({4, 3, 2});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_RESERVED, DT_UNDEFINED);
  tensor_desc1.SetOriginDataType(DT_UNDEFINED);
  op_desc->UpdateInputDesc(1, tensor_desc1);
  // optional input 2
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc2);

  const auto infer_datatype_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto option_input_dtype_1 = context->GetOptionalInputDataType(1U);
    if (option_input_dtype_1 != ge::DT_UNDEFINED) {
      return GRAPH_FAILED;
    }
    const auto option_input_dtype_2 = context->GetOptionalInputDataType(2U);
    if (option_input_dtype_2 != ge::DT_FLOAT16) {
      return GRAPH_FAILED;
    }
    auto ret = context->SetOutputDataType(0U, option_input_dtype_2 == ge::DT_FLOAT16 ? ge::DT_INT8 : DT_FLOAT);
    return ret;
  };
  IMPL_OP(TwoOptionalInputsOp).InferShape(nullptr).InferDataType(infer_datatype_func).InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("TwoOptionalInputsOp");
  op_impl_func->infer_datatype = infer_datatype_func;

  const auto call_infer_dtype_v2 = OperatorFactoryImpl::GetInferDataTypeFunc();
  ASSERT_NE(call_infer_dtype_v2, nullptr);
  auto status = call_infer_dtype_v2(op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), ge::DT_INT8);
}

// 动态输入的input测试
REG_OP(DynamicInput3Input3Output3)
    .INPUT(input1, "T")
    .DYNAMIC_INPUT(dyn_input, "D")
        .INPUT(input3, "T")
        .OUTPUT(output1, "T2")
        .OUTPUT(output2, "T2")
        .OUTPUT(output3, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(DynamicInput3Input3Output3);
const auto INFER_SHAPE_FUNC = [](gert::InferShapeContext *context) -> graphStatus {
  // update input3 input to output0
  const auto input_shape = context->GetInputShape(1U);
  auto output = context->GetOutputShape(0);
  for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
    output->AppendDim(input_shape->GetDim(dim));
  }
  output->SetDimNum(input_shape->GetDimNum());
  // update dyn_input_0 to output1, dyn_input_1 to output2
  const auto input_shape2 = context->GetInputShape(2U);
  auto output2 = context->GetOutputShape(1);
  for (size_t dim = 0UL; dim < input_shape2->GetDimNum(); dim++) {
    output2->AppendDim(input_shape2->GetDim(dim));
  }
  output2->SetDimNum(input_shape2->GetDimNum());

  const auto input_shape3 = context->GetInputShape(3U);
  auto output3 = context->GetOutputShape(2);
  for (size_t dim = 0UL; dim < input_shape3->GetDimNum(); dim++) {
    output3->AppendDim(input_shape3->GetDim(dim));
  }
  output3->SetDimNum(input_shape3->GetDimNum());
  return GRAPH_SUCCESS;
};
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, 4});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInput3Input3Output3");
  op_impl_func->infer_shape = INFER_SHAPE_FUNC;
  op_impl_func->infer_shape_range = nullptr;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 动态输入的input测试 动态轴-2
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_2) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({-2});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
    .InferDataType(nullptr)
    .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInput3Input3Output3");
  op_impl_func->infer_shape = INFER_SHAPE_FUNC;
  op_impl_func->infer_shape_range = nullptr;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 0);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 动态输入的input测试 动态轴-1, shape range 不设值
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_no_shaperange) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInput3Input3Output3");
  op_impl_func->infer_shape = INFER_SHAPE_FUNC;
  op_impl_func->infer_shape_range = nullptr;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 动态输入的input测试 动态轴-1, shape range 设值
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_shaperange) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {2, 2}, {3, 3}, {22, 999}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInput3Input3Output3");
  op_impl_func->infer_shape = INFER_SHAPE_FUNC;
  op_impl_func->infer_shape_range = nullptr;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)op_desc->GetOutputDesc(2U).GetShapeRange(shape_range);
  ASSERT_EQ(shape_range.size(), 4U);
  for (size_t i = 0UL; i < shape_range.size(); ++i) {
    ASSERT_EQ(shape_range[i].first, range[i].first);
    ASSERT_EQ(shape_range[i].second, range[i].second);
  }
}

// 动态输入的input测试 动态轴-1, shape range 设值,min大于max异常场景
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_shaperange_min_bigger_max) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {2, 2}, {3, 3}, {999, 22}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
    .InferDataType(nullptr)
    .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInput3Input3Output3");
  op_impl_func->infer_shape = INFER_SHAPE_FUNC;
  op_impl_func->infer_shape_range = nullptr;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, ge::PARAM_INVALID);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)op_desc->GetOutputDesc(2U).GetShapeRange(shape_range);
  ASSERT_EQ(shape_range.size(), 0U);
}

// 动态输入的input测试 动态轴-1, shape range 设值, min大于max, max为-1的正常场景
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_shaperange_min_bigger_max_success) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, -1}, {2, -1}, {3, -1}, {999, -1}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
    .InferDataType(nullptr)
    .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInput3Input3Output3");
  op_impl_func->infer_shape = INFER_SHAPE_FUNC;
  op_impl_func->infer_shape_range = nullptr;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, ge::GRAPH_SUCCESS);
}

// 二类算子值依赖测试
REG_OP(Type2_1Input_1Output)
    .INPUT(input1, "T")
        .OPTIONAL_INPUT(input2, "T")
        .INPUT(input3, "T")
        .OUTPUT(output1, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(Type2_1Input_1Output);
TEST_F(ShapeInferenceUT, CallInferV2Func_Type2ValueDepend) {
  // construct const input
  auto const_input = ge::op::Const("const_input");
  ge::TensorDesc td{ge::Shape(std::vector<int64_t>({1, 2, 3, 4})), FORMAT_NCHW, DT_UINT8};
  ge::Tensor tensor(td);
  std::vector<uint8_t> val{0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58};
  tensor.SetData(val);
  const_input.set_attr_value(tensor);
  // const input link to op
  auto op = op::Type2_1Input_1Output("test5");
  op.set_input_input1(const_input);
  op.set_input_input3(const_input);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 3);
  // input1
  ge::GeShape shape1({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  op_desc->UpdateInputDesc(2, tensor_desc1);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    // update input3(因为option输入未实例化，所以是第二个) value to output0
    const auto data = context->GetInputTensor(1U)->GetData<uint8_t>();
    std::vector<int64_t> dims = {data[0], data[1], data[2], data[3]};
    ge::Shape input_shape(dims);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape.GetDimNum(); dim++) {
      output->AppendDim(input_shape.GetDim(dim));
    }
    output->SetDimNum(input_shape.GetDimNum());
    return GRAPH_SUCCESS;
  };
  IMPL_OP(Type2_1Input_1Output).InferShape(infer_shape_func).InputsDataDependency({2})
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("Type2_1Input_1Output");
  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->SetInputDataDependency(2);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  const auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &shape = op_desc->GetOutputDesc(0U).GetShape();
  ASSERT_EQ(shape.GetDimNum(), 4);
  ASSERT_EQ(shape.GetDim(0U), 85);
  ASSERT_EQ(shape.GetDim(1U), 86);
  ASSERT_EQ(shape.GetDim(2U), 87);
  ASSERT_EQ(shape.GetDim(3U), 88);
}

// 二类算子值依赖测试,带shape range
REG_OP(Type2_3Input_2Output)
  .INPUT(input1, "T")
  .OPTIONAL_INPUT(input2, "T")
  .INPUT(input3, "T")
  .OUTPUT(output1, "T2")
  .OUTPUT(output2, "T2")
  .DATATYPE(T2, TensorType({DT_BOOL}))
  .OP_END_FACTORY_REG(Type2_3Input_2Output);
TEST_F(ShapeInferenceUT, CallInferV2Func_Type2ValueDepend_unknow_shaperange) {
  // construct const input
  auto const_input = ge::op::Const("const_input");
  ge::TensorDesc td{ge::Shape(std::vector<int64_t>({1, 2, 3, 4})), FORMAT_NCHW, DT_UINT8};
  ge::Tensor tensor(td);
  std::vector<uint8_t> val{0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58};
  tensor.SetData(val);
  const_input.set_attr_value(tensor);
  // const input link to op
  auto op = op::Type2_3Input_2Output("test5");
  op.set_input_input1(const_input);
  op.set_input_input3(const_input);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 3);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {2, 2}, {3, 3}, {22, 999}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  ge::GeShape shape3({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    // update input3(因为option输入未实例化，所以是第二个) value to output0
    const auto data = context->GetInputTensor(1U)->GetData<uint8_t>();
    std::vector<int64_t> dims = {data[0], data[1], data[2], data[3]};
    ge::Shape input_shape(dims);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape.GetDimNum(); dim++) {
      output->AppendDim(input_shape.GetDim(dim));
    }
    output->SetDimNum(input_shape.GetDimNum());

    const auto input_shape1 = context->GetInputShape(0U);
    auto output1 = context->GetOutputShape(1);
    for (size_t dim = 0UL; dim < input_shape1->GetDimNum(); dim++) {
      output1->AppendDim(input_shape1->GetDim(dim));
    }
    output1->SetDimNum(input_shape1->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };
  IMPL_OP(Type2_3Input_2Output).InferShape(infer_shape_func).InputsDataDependency({2})
    .InferDataType(nullptr)
    .InferShapeRange(infer_shape_range_func);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("Type2_3Input_2Output");
  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->SetInputDataDependency(2);
  op_impl_func->infer_shape_range = infer_shape_range_func;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &shape = op_desc->GetOutputDesc(0U).GetShape();
  ASSERT_EQ(shape.GetDimNum(), 4);
  ASSERT_EQ(shape.GetDim(0U), 85);
  ASSERT_EQ(shape.GetDim(1U), 86);
  ASSERT_EQ(shape.GetDim(2U), 87);
  ASSERT_EQ(shape.GetDim(3U), 88);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)op_desc->GetOutputDesc(0U).GetShapeRange(shape_range);
  ASSERT_EQ(shape_range.size(), 4U);
  for (size_t i = 0UL; i < shape_range.size(); ++i) {
    ASSERT_EQ(shape_range[i].first, range[i].first);
    ASSERT_EQ(shape_range[i].second, range[i].second);
  }
}
TEST_F(ShapeInferenceUT, CallInferV2Func_skip_shaperange_infer_when_input_without_range) {
  // construct const input
  auto const_input = ge::op::Const("const_input");
  ge::TensorDesc td{ge::Shape(std::vector<int64_t>({1, 2, 3, 4})), FORMAT_NCHW, DT_UINT8};
  ge::Tensor tensor(td);
  std::vector<uint8_t> val{0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58};
  tensor.SetData(val);
  const_input.set_attr_value(tensor);
  // const input link to op
  auto op = op::Type2_3Input_2Output("test5");
  op.set_input_input1(const_input);
  op.set_input_input3(const_input);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 3);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  ge::GeShape shape3({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);

  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus { return GRAPH_FAILED; };
  IMPL_OP(Type2_3Input_2Output).InputsDataDependency({2})
      .InferDataType(nullptr)
      .InferShapeRange(infer_shape_range_func);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("Type2_3Input_2Output");
  op_impl_func->SetInputDataDependency(2);
  op_impl_func->infer_shape_range = infer_shape_range_func;

  auto status = InferShapeRangeOnCompile(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS); // success means not called infer_shape_range_func
}

// 资源类算子测试
REG_OP(RegisterAndGetReiledOnResource)
    .INPUT(input1, "T")
        .OUTPUT(output1, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(RegisterAndGetReiledOnResource);
TEST_F(ShapeInferenceUT, CallInferV2Func_RegisterAndGetReiledOnResource) {
  auto op = OperatorFactory::CreateOperator("test6", "RegisterAndGetReiledOnResource");
  const char_t *resource_key = "224";
  auto read_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());
  read_inference_context->RegisterReliedOnResourceKey(AscendString(resource_key));
  op.SetInferenceContext(read_inference_context);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);    // simulate read_op register relied resource
  // input1
  ge::GeShape shape1({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    auto ct_context = reinterpret_cast<gert::CtInferShapeContext *>(context);
    const auto &read_inference_context = ct_context->GetInferenceContext();
    const auto &reiled_keys = read_inference_context->GetReliedOnResourceKeys();
    const char_t *resource_key_ = "224";
    // check result
    EXPECT_EQ(reiled_keys.empty(), false);
    EXPECT_EQ(*reiled_keys.begin(), resource_key_);
    if (reiled_keys.empty() ||
        (*reiled_keys.begin() != resource_key_)) {
      return GRAPH_FAILED;
    }
    auto out_shape = context->GetOutputShape(0UL);
    out_shape->SetDimNum(1UL);
    out_shape->SetDim(0UL, std::strtol(resource_key_, nullptr, 10));
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto ct_context = reinterpret_cast<gert::CtInferShapeRangeContext *>(context);
    const auto &read_inference_context = ct_context->GetInferenceContext();
    const auto &reiled_keys = read_inference_context->GetReliedOnResourceKeys();
    const char_t *resource_key_ = "224";
    // check result
    EXPECT_EQ(reiled_keys.empty(), false);
    EXPECT_EQ(*reiled_keys.begin(), resource_key_);
    return GRAPH_SUCCESS;
  };
  IMPL_OP(RegisterAndGetReiledOnResource)
      .InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(infer_shape_range_func);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("RegisterAndGetReiledOnResource");
  op_impl_func->infer_shape = infer_shape_func;

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  const auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &shape = op_desc->GetOutputDesc(0U).GetShape();
  ASSERT_EQ(shape.GetDim(0), std::strtol(resource_key, nullptr, 10));
}

// 默认infer datatype测试
REG_OP(TestDefaultInferDataType)
    .INPUT(input1, "T")
        .OUTPUT(output1, "T")
        .DATATYPE(T, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(TestDefaultInferDataType);
TEST_F(ShapeInferenceUT, CallInferV2Func_TestDefaultInferShape) {
  auto op = OperatorFactory::CreateOperator("test7", "TestDefaultInferDataType");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);    // simulate read_op register relied resource
  // input1
  ge::GeShape shape1({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_BOOL);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_BOOL);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    return GRAPH_SUCCESS;
  };
  IMPL_OP(TestDefaultInferDataType)
      .InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);
  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto status = call_infer_data_type(op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &data_type = op_desc->GetOutputDesc(0U).GetDataType();
  ASSERT_EQ(data_type, DT_BOOL);
}
TEST_F(ShapeInferenceUT, AdaptFuncRegisterOk) {
  ASSERT_NE(OperatorFactoryImpl::GetInferShapeV2Func(), nullptr);
  ASSERT_NE(OperatorFactoryImpl::GetInferShapeRangeFunc(), nullptr);
  ASSERT_NE(OperatorFactoryImpl::GetInferDataTypeFunc(), nullptr);
}
TEST_F(ShapeInferenceUT, CallInferV2Func_no_inferfunc_failed_FAST_IGNORE_INFER_ERRIR_is_empty) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;  // make v1 is null

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FixIOOp_OutputIsFix");
  op_impl_func->infer_shape = nullptr;  // make v2 is null
  op_impl_func->infer_datatype = nullptr;
  op_impl_func->infer_shape_range = nullptr;

  mmSetEnv("IGNORE_INFER_ERROR", "111", 1);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_PARAM_INVALID);
  unsetenv("IGNORE_INFER_ERROR");
}

TEST_F(ShapeInferenceUT, CallInferV2Func_NoSpaceRegistry_NotSupportSymbolicInfer_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "AddUt");  // not support symbolic infer dtype
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;  // make v1 is null
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, ge::GRAPH_PARAM_INVALID);
}

TEST_F(ShapeInferenceUT, CallInferV2Func_NoSpaceRegistry_SupportSymbolicInfer_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");  // support symbolic infer dtype
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr; // make v1 is null
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, ge::GRAPH_PARAM_INVALID);
}

TEST_F(ShapeInferenceUT, CallInferV2Func_NoCustomInferdtype_NotSupportSymbolicInfer_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "AddUt"); // not support symbolic infer dtype
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1,1,1,1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr; // make v1 is null

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("AddUt");
  op_impl_func->infer_shape = nullptr; // make v2 is null
  op_impl_func->infer_datatype = nullptr; // make v2 infer dtype is null
  op_impl_func->infer_shape_range = nullptr;

  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, ge::GRAPH_PARAM_INVALID);
}

TEST_F(ShapeInferenceUT, CallInferV2Func_NoInferShape_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "AddUt");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1,1,1,1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr; // make v1 is null

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("AddUt");
  op_impl_func->infer_shape = nullptr; // make v2 is null
  op_impl_func->infer_datatype = nullptr; // make v2 infer dtype is null
  op_impl_func->infer_shape_range = nullptr;

  auto status = InferShapeOnCompile(op, op_desc);
  ASSERT_NE(status, ge::GRAPH_SUCCESS);
}

TEST_F(ShapeInferenceUT, CallInferFormatFunc_OptionalInput) {
  auto op = OperatorFactory::CreateOperator("OptionalTest", "OptionalInput3Input3Output");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc, nullptr);
  // option_input
  GeShape shape({1, 1, 1, 1});
  const auto format = Format::FORMAT_NCHW;
  const auto ori_format = Format::FORMAT_ND;
  GeTensorDesc tensor_desc(shape, format);
  tensor_desc.SetOriginFormat(ori_format);
  tensor_desc.SetOriginShape(shape);
  op_desc->UpdateInputDesc(1U, tensor_desc);

  auto infer_format_func = [](gert::InferFormatContext *context) -> UINT32 {
    const auto option_input_format = context->GetOptionalInputFormat(1U);
    EXPECT_NE(option_input_format, nullptr);
    EXPECT_EQ(option_input_format->GetOriginFormat(), Format::FORMAT_ND);
    EXPECT_EQ(option_input_format->GetStorageFormat(), Format::FORMAT_NCHW);

    const auto option_input_shape = context->GetOptionalInputShape(1U);
    EXPECT_NE(option_input_shape, nullptr);
    EXPECT_EQ(option_input_shape->GetDimNum(), 4UL);
    EXPECT_EQ(option_input_shape->GetDim(0U), 1L);

    auto input_0 = context->GetRequiredInputFormat(0U);
    input_0->SetOriginFormat(Format::FORMAT_NCHW);
    input_0->SetStorageFormat(Format::FORMAT_NCHW);

    auto output_0 = context->GetOutputFormat(0U);
    output_0->SetOriginFormat(Format::FORMAT_NCHW);
    output_0->SetStorageFormat(Format::FORMAT_NCHW);
    return GRAPH_SUCCESS;
  };
  IMPL_OP(OptionalInput3Input3Output).InferFormat(infer_format_func);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("OptionalInput3Input3Output");
  op_impl_func->infer_format_func = infer_format_func;

  const auto call_infer_format_v2 = OperatorFactoryImpl::GetInferFormatV2Func();
  EXPECT_NE(call_infer_format_v2, nullptr);
  ASSERT_EQ(call_infer_format_v2(op, op_desc), GRAPH_SUCCESS);

  const auto &input_0 = op_desc->GetInputDesc(0U);
  EXPECT_EQ(input_0.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(input_0.GetFormat(), Format::FORMAT_NCHW);

  const auto &output_0 = op_desc->GetOutputDesc(0U);
  EXPECT_EQ(output_0.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(output_0.GetFormat(), Format::FORMAT_NCHW);
}

TEST_F(ShapeInferenceUT, CallInferFormatFunc_DynamicInput) {
  auto op = op::DynamicInput3Input3Output3("DynamicTest");
  op.create_dynamic_input_byindex_dyn_input(2, 1);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);

  // dynamic_input index=1
  GeShape shape({1, 1, 1, 1});
  const auto format = Format::FORMAT_NCHW;
  const auto ori_format = Format::FORMAT_ND;
  GeTensorDesc tensor_desc(shape, format);
  tensor_desc.SetOriginFormat(ori_format);
  tensor_desc.SetOriginShape(shape);
  op_desc->UpdateInputDesc(1U, tensor_desc);

  auto infer_format_func = [](gert::InferFormatContext *context) -> UINT32 {
    const auto dynamic_input_format = context->GetDynamicInputFormat(1U, 0U);
    EXPECT_NE(dynamic_input_format, nullptr);
    EXPECT_EQ(dynamic_input_format->GetOriginFormat(), Format::FORMAT_ND);
    EXPECT_EQ(dynamic_input_format->GetStorageFormat(), Format::FORMAT_NCHW);

    const auto dynamic_input_shape = context->GetDynamicInputShape(1U, 0U);
    EXPECT_NE(dynamic_input_shape, nullptr);
    EXPECT_EQ(dynamic_input_shape->GetDimNum(), 4UL);
    EXPECT_EQ(dynamic_input_shape->GetDim(0U), 1L);

    auto input_0 = context->GetRequiredInputFormat(2U);
    input_0->SetOriginFormat(Format::FORMAT_NCHW);
    input_0->SetStorageFormat(Format::FORMAT_NCHW);

    auto output_0 = context->GetOutputFormat(0U);
    output_0->SetOriginFormat(Format::FORMAT_NCHW);
    output_0->SetStorageFormat(Format::FORMAT_NCHW);
    return GRAPH_SUCCESS;
  };
  IMPL_OP(DynamicInput3Input3Output3).InferFormat(infer_format_func);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInput3Input3Output3");
  op_impl_func->infer_format_func = infer_format_func;

  const auto call_infer_format_v2 = OperatorFactoryImpl::GetInferFormatV2Func();
  EXPECT_NE(call_infer_format_v2, nullptr);
  ASSERT_EQ(call_infer_format_v2(op, op_desc), GRAPH_SUCCESS);

  const auto &input_0 = op_desc->GetInputDesc(3U);
  EXPECT_EQ(input_0.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(input_0.GetFormat(), Format::FORMAT_NCHW);

  const auto &output_0 = op_desc->GetOutputDesc(0U);
  EXPECT_EQ(output_0.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(output_0.GetFormat(), Format::FORMAT_NCHW);
}

TEST_F(ShapeInferenceUT, CallInferFormatFunc_ValueDependInput) {
  // construct const input
  auto const_input = ge::op::Const("const_input");
  ge::TensorDesc td{ge::Shape(std::vector<int64_t>({1, 2, 3, 4})), FORMAT_NCHW, DT_UINT8};
  ge::Tensor tensor(td);
  std::vector<uint8_t> val{0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58};
  tensor.SetData(val);
  const_input.set_attr_value(tensor);
  // const input link to op
  auto op = op::Type2_1Input_1Output("test5");
  op.set_input_input1(const_input);
  op.set_input_input3(const_input);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 3);
  // input1
  ge::GeShape shape1({1, 2, 3, 4});
  ge::GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  op_desc->UpdateInputDesc(2, tensor_desc1);


  auto infer_format_func = [](gert::InferFormatContext *context) -> UINT32 {
    const auto data = context->GetInputTensor(1U)->GetData<uint8_t>();
    EXPECT_EQ(data[0], 85);

    auto output_0 = context->GetOutputFormat(0U);
    output_0->SetOriginFormat(Format::FORMAT_NCHW);
    output_0->SetStorageFormat(Format::FORMAT_NCHW);
    return GRAPH_SUCCESS;
  };
  IMPL_OP(Type2_1Input_1Output).InferFormat(infer_format_func).InputsDataDependency({2});

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("Type2_1Input_1Output");
  op_impl_func->infer_format_func = infer_format_func;
  op_impl_func->SetInputDataDependency(2);

  const auto call_infer_format_v2 = OperatorFactoryImpl::GetInferFormatV2Func();
  EXPECT_NE(call_infer_format_v2, nullptr);
  ASSERT_EQ(call_infer_format_v2(op, op_desc), GRAPH_SUCCESS);

  const auto &output_0 = op_desc->GetOutputDesc(0U);
  EXPECT_EQ(output_0.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(output_0.GetFormat(), Format::FORMAT_NCHW);
}

REG_OP(DynamicInputDynamicOutput)
    .DYNAMIC_INPUT(dyn_input, "D")
    .DYNAMIC_OUTPUT(dyn_output, "D")
    .OUTPUT(out, "T")
    .OP_END_FACTORY_REG(DynamicInputDynamicOutput);
TEST_F(ShapeInferenceUT, CallInferFormatFunc_DynamicOutput) {
  auto op = op::DynamicInputDynamicOutput("DynamicTest");
  op.create_dynamic_input_byindex_dyn_input(1, 0);
  op.create_dynamic_output_dyn_output(1);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);

  // dynamic_input
  GeShape shape({1, 1, 1, 1});
  const auto format = Format::FORMAT_NCHW;
  const auto ori_format = Format::FORMAT_ND;
  GeTensorDesc tensor_desc(shape, format);
  tensor_desc.SetOriginFormat(ori_format);
  tensor_desc.SetOriginShape(shape);
  op_desc->UpdateInputDesc(0U, tensor_desc);
  op_desc->UpdateOutputDesc(0U, tensor_desc);
  op_desc->UpdateOutputDesc(1U, tensor_desc);

  auto infer_format_func = [](gert::InferFormatContext *context) -> UINT32 {
    const auto dynamic_input_format = context->GetDynamicInputFormat(0U, 0U);
    EXPECT_NE(dynamic_input_format, nullptr);
    EXPECT_EQ(dynamic_input_format->GetOriginFormat(), Format::FORMAT_ND);
    EXPECT_EQ(dynamic_input_format->GetStorageFormat(), Format::FORMAT_NCHW);

    const auto dynamic_input_shape = context->GetDynamicInputShape(0U, 0U);
    EXPECT_NE(dynamic_input_shape, nullptr);
    EXPECT_EQ(dynamic_input_shape->GetDimNum(), 4UL);
    EXPECT_EQ(dynamic_input_shape->GetDim(0U), 1L);


    auto output_0 = context->GetDynamicOutputFormat(0U, 0U);
    output_0->SetOriginFormat(Format::FORMAT_NCHW);
    output_0->SetStorageFormat(Format::FORMAT_NCHW);

    auto output_1 = context->GetRequiredOutputFormat(0U);
    output_1->SetOriginFormat(Format::FORMAT_NCHW);
    output_1->SetStorageFormat(Format::FORMAT_NCHW);
    return GRAPH_SUCCESS;
  };
  IMPL_OP(DynamicInputDynamicOutput).InferFormat(infer_format_func);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DynamicInputDynamicOutput");
  op_impl_func->infer_format_func = infer_format_func;

  const auto call_infer_format_v2 = OperatorFactoryImpl::GetInferFormatV2Func();
  EXPECT_NE(call_infer_format_v2, nullptr);
  ASSERT_EQ(call_infer_format_v2(op, op_desc), GRAPH_SUCCESS);

  const auto &output_0 = op_desc->GetOutputDesc(0U);
  EXPECT_EQ(output_0.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(output_0.GetFormat(), Format::FORMAT_NCHW);

  const auto &output_1 = op_desc->GetOutputDesc(1U);
  EXPECT_EQ(output_1.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(output_1.GetFormat(), Format::FORMAT_NCHW);
}


// infer from output
REG_OP(CustomAdd)
    .INPUT(input1, "T")
    .INPUT(input2, "T")
    .OUTPUT(output, "T")
    .OP_END_FACTORY_REG(CustomAdd);
TEST_F(ShapeInferenceUT, CallInferFunc_CustomOp_With_InferShape_Success) {
  auto op = OperatorFactory::CreateOperator("test1", "CustomAdd");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomAdd");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  CustomOpFactory::RegisterCustomOpCreator("CustomAdd", []()->std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MyCustomOp>();
  });

  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 4);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
}

TEST_F(ShapeInferenceUT, CallInferFunc_CustomOp_Without_InferShape_Success) {
  auto op = OperatorFactory::CreateOperator("test1", "CustomAdd");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomAdd");
  (void) op_impl_func;
  CustomOpFactory::RegisterCustomOpCreator("CustomAdd", []()->std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MyCustomOp>();
  });

  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();

  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  ASSERT_NE(call_infer_data_type, nullptr);
  ASSERT_NE(call_infer_shape_v2, nullptr);
  ASSERT_NE(call_infer_shape_range, nullptr);

  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  // 走shape为-2
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_UNDEFINED);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 0);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), -2);
}

TEST_F(ShapeInferenceUT, CallInferFunc_CustomOp_Without_InferShape_with_origin_shape_Success) {
  auto op = OperatorFactory::CreateOperator("test1", "CustomAdd");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomAdd");
  (void) op_impl_func;
  CustomOpFactory::RegisterCustomOpCreator("CustomAdd", []()->std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MyCustomOp>();
  });

  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();

  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  ASSERT_NE(call_infer_data_type, nullptr);
  ASSERT_NE(call_infer_shape_v2, nullptr);
  ASSERT_NE(call_infer_shape_range, nullptr);

  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  // 走shape为-2
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 4);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
}
}  // namespace gert
