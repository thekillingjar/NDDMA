/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_datatype_context.h"
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "faker/kernel_run_context_faker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "faker/space_registry_faker.h"

namespace gert {
class InferDataTypeContextUT : public testing::Test {};
TEST_F(InferDataTypeContextUT, GetInputDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT8;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputDataTypes({&in_datatype1, &in_datatype2})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetInputDataType(0), in_datatype1);
  EXPECT_EQ(context->GetInputDataType(1), in_datatype2);
}

TEST_F(InferDataTypeContextUT, GetDynamicInputDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT8;
  ge::DataType in_datatype2 = ge::DT_INT4;
  ge::DataType in_datatype3 = ge::DT_INT8;
  ge::DataType in_datatype4 = ge::DT_INT4;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
                            .IrInstanceNum({1, 2, 0, 1})
                            .NodeIoNum(4, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .InputDataTypes({&in_datatype1, &in_datatype2, &in_datatype3, &in_datatype4})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOptionalInputDataType(0), in_datatype1);
  EXPECT_EQ(context->GetDynamicInputDataType(1, 0), in_datatype2);
  EXPECT_EQ(context->GetDynamicInputDataType(1, 1), in_datatype3);

  EXPECT_EQ(context->GetOptionalInputDataType(2), ge::DataType::DT_UNDEFINED);

  EXPECT_EQ(context->GetOptionalInputDataType(3), in_datatype4);
}

TEST_F(InferDataTypeContextUT, GetOutDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT4;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputDataTypes({&in_datatype1, &in_datatype2})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), out_datatype);

  EXPECT_EQ(context->GetOutputDataType(1), ge::DataType::DT_UNDEFINED);
}

TEST_F(InferDataTypeContextUT, SetOutputDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT4;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType origin_out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputDataTypes({&in_datatype1, &in_datatype2})
      .OutputDataTypes({&origin_out_datatype})
      .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), origin_out_datatype);
  EXPECT_EQ(context->SetOutputDataType(0, ge::DT_INT32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST_F(InferDataTypeContextUT, Retpeat_register_InferDataType_InferOutDataTypeByFirstInput_success) {
  IMPL_OP(TestFoo1)
      .InferOutDataTypeSameWithFirstInput();
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TestFoo1");
  ASSERT_NE(funcs, nullptr);
  auto infer_func = funcs->infer_datatype;
  EXPECT_NE(infer_func, nullptr);

  ge::DataType in_datatype1 = ge::DT_INT4;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType origin_out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputDataTypes({&in_datatype1, &in_datatype2})
      .OutputDataTypes({&origin_out_datatype})
      .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), origin_out_datatype);
  EXPECT_EQ(infer_func(context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputDataType(0), in_datatype1);
}

TEST_F(InferDataTypeContextUT, Retpeat_register_InferDataType_InferOutDataTypeByFirstInput_failed) {
  IMPL_OP(TestFoo)
  .InferOutDataTypeSameWithFirstInput();
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  auto infer_func = funcs->infer_datatype;
  EXPECT_NE(infer_func, nullptr);

  ge::DataType in_datatype1 = ge::DT_UNDEFINED;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType origin_out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputDataTypes({&in_datatype1, &in_datatype2})
      .OutputDataTypes({&origin_out_datatype})
      .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), origin_out_datatype);
  EXPECT_NE(infer_func(context), ge::GRAPH_SUCCESS);
}
TEST_F(InferDataTypeContextUT, GetRequiredInputDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT8;
  ge::DataType in_datatype2 = ge::DT_INT4;
  ge::DataType in_datatype3 = ge::DT_INT8;
  ge::DataType in_datatype4 = ge::DT_INT4;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
                            .IrInstanceNum({1, 2, 0, 1})
                            .NodeIoNum(4, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .InputDataTypes({&in_datatype1, &in_datatype2, &in_datatype3, &in_datatype4})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetRequiredInputDataType(0), in_datatype1);
  EXPECT_EQ(context->GetDynamicInputDataType(1, 0), in_datatype2);
  EXPECT_EQ(context->GetDynamicInputDataType(1, 1), in_datatype3);

  EXPECT_EQ(context->GetOptionalInputDataType(2), ge::DataType::DT_UNDEFINED);

  EXPECT_EQ(context->GetRequiredInputDataType(3), in_datatype4);
}
}  // namespace gert
