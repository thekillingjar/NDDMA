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

#include "graph/op_desc.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/ge_tensor.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/transformer_utils.h"
#include "graph/common_error_codes.h"
#include "graph/operator_factory_impl.h"
#include "register/op_tiling_registry.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_kernel_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/infer_format_context.h"
#include "faker/space_registry_faker.h"

namespace ge {
class UtestOpDescUtilsEx : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpDescUtilsEx, OpVerify_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
  const auto verify_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  op_desc->AddVerifierFunc(verify_func);
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtilsEx, InferShapeAndType_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
  const auto add_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  op_desc->AddInferFunc(add_func);
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtilsEx, InferDataSlice_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), NO_DEPENDENCE_FUNC);
  const auto infer_data_slice_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  auto op = std::make_shared<Operator>();
  op_desc->SetType("test");
  OperatorFactoryImpl::RegisterInferDataSliceFunc("test",infer_data_slice_func);
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), GRAPH_SUCCESS);
}

REG_OP(FixInfer_OutputIsFix)
  .INPUT(fix_input1, "T")
  .INPUT(fix_input2, "T")
  .OUTPUT(fix_output, "T2")
  .DATATYPE(T2, TensorType({DT_BOOL}))
  .OP_END_FACTORY_REG(FixInfer_OutputIsFix);
TEST_F(UtestOpDescUtilsEx, CallInferFormatFunc_success) {
  auto op = OperatorFactory::CreateOperator("test", "FixInfer_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("test");
  const auto infer_format_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferFormatFunc("test", infer_format_func);
  EXPECT_EQ(OpDescUtilsEx::CallInferFormatFunc(op_desc, op), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtilsEx, SetType_success) {
  auto op_desc = std::make_shared<OpDesc>();
  string type = "tmp";
  OpDescUtilsEx::SetType(op_desc, type);
  EXPECT_EQ(op_desc->GetType(), type);
}

TEST_F(UtestOpDescUtilsEx, SetTypeAndResetFuncHandle_success) {
  auto op_desc = std::make_shared<OpDesc>();
  string type = "tmp";
  OpDescUtilsEx::SetTypeAndResetFuncHandle(op_desc, type);
  EXPECT_EQ(op_desc->GetType(), type);
  EXPECT_EQ(op_desc->GetInferFunc(), nullptr);
  EXPECT_EQ(op_desc->GetInferFormatFunc(), nullptr);
  EXPECT_EQ(op_desc->GetInferValueRangeFunc(), nullptr);
  EXPECT_EQ(op_desc->GetVerifyFunc(), nullptr);
  EXPECT_EQ(op_desc->GetInferDataSliceFunc(), nullptr);
}

TEST_F(UtestOpDescUtilsEx, CallInferFormatFunc_v2_success) {
  auto op = OperatorFactory::CreateOperator("test", "FixInfer_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  const auto infer_format_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferFormatFunc("FixInfer_OutputIsFix", infer_format_func);

  const auto infer_format_func_v2 = [](gert::InferFormatContext *context) -> UINT32 {
    auto output_1 = context->GetRequiredOutputFormat(0U);
    output_1->SetOriginFormat(Format::FORMAT_NCHW);
    output_1->SetStorageFormat(Format::FORMAT_NCHW);
    return GRAPH_SUCCESS;
  };
  IMPL_OP(FixInfer_OutputIsFix).InferFormat(infer_format_func_v2);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FixInfer_OutputIsFix");
  op_impl_func->infer_format_func = infer_format_func_v2;

  EXPECT_EQ(OpDescUtilsEx::CallInferFormatFunc(op_desc, op), GRAPH_SUCCESS);
  const auto &output_0 = op_desc->GetOutputDesc(0U);
  EXPECT_EQ(output_0.GetOriginFormat(), Format::FORMAT_NCHW);
  EXPECT_EQ(output_0.GetFormat(), Format::FORMAT_NCHW);

}
}