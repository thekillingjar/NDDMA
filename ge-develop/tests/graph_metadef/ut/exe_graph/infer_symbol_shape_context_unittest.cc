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
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "faker/kernel_run_context_faker.h"
#include "exe_graph/runtime/symbolic_tensor.h"

namespace gert {
class InferSymbolShapeContextUT : public testing::Test {};
TEST_F(InferSymbolShapeContextUT, GetInputShapeOk) {
  auto in_0 = ge::Symbol(8);
  auto in_1 = ge::Symbol(3);
  auto in_2 = ge::Symbol(224);
  auto in_3 = ge::Symbol(224);
  gert::SymbolTensor in_tensor1({in_0, in_1, in_2, in_3}, {});
  gert::SymbolTensor in_tensor2({in_0, in_1, in_2, in_3}, {});
  gert::SymbolShape out_shape({in_0, in_1, in_2, in_3});

  auto context_holder = InferSymbolShapeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Inputs({&in_tensor1, &in_tensor2})
                            .Outputs({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferSymbolShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputSymbolShape(0), nullptr);
  EXPECT_EQ(*context->GetInputSymbolShape(0), in_tensor1.GetOriginSymbolShape());

  ASSERT_NE(context->GetInputSymbolTensor(1), nullptr);
  EXPECT_EQ(*context->GetInputSymbolShape(1), in_tensor2.GetOriginSymbolShape());

  EXPECT_EQ(context->GetInputSymbolTensor(2), nullptr);

  ASSERT_NE(context->GetOutputSymbolShape(0), nullptr);
  EXPECT_EQ(*context->GetOutputSymbolShape(0), out_shape);
}

TEST_F(InferSymbolShapeContextUT, GetDynamicInputShapeOk) {
  auto sym_8 = ge::Symbol(8);
  auto sym_3 = ge::Symbol(3);
  auto sym_2 = ge::Symbol(2);
  auto sym_4 = ge::Symbol(4);
  auto sym_224 = ge::Symbol(224);
  auto sym_128 = ge::Symbol(128);
  auto sym_16 = ge::Symbol(16);
  gert::SymbolTensor in_tensor1({sym_8, sym_3, sym_224, sym_224});
  gert::SymbolTensor in_tensor2({sym_2, sym_2, sym_3, sym_8});
  gert::SymbolTensor in_tensor3({sym_3, sym_2, sym_3, sym_8});
  gert::SymbolTensor in_tensor4({sym_4, sym_2, sym_3, sym_8});
  gert::SymbolShape out_shape({sym_8, sym_3, sym_224, sym_224});

  auto context_holder = InferSymbolShapeContextFaker()
                            .IrInputInstanceNum({1, 2, 0, 1})
                            .NodeIoNum(4, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .Inputs({&in_tensor1, &in_tensor2, &in_tensor3, &in_tensor4})
                            .Outputs({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferSymbolShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOptionalInputSymbolShape(0), nullptr);
  EXPECT_EQ(*context->GetOptionalInputSymbolShape(0), in_tensor1.GetOriginSymbolShape());

  ASSERT_NE(context->GetDynamicInputSymbolShape(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputSymbolShape(1, 0), in_tensor2.GetOriginSymbolShape());

  ASSERT_NE(context->GetDynamicInputSymbolShape(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputSymbolShape(1, 1), in_tensor3.GetOriginSymbolShape());

  EXPECT_EQ(context->GetOptionalInputSymbolShape(2), nullptr);

  ASSERT_NE(context->GetOptionalInputSymbolShape(3), nullptr);
  EXPECT_EQ(*context->GetOptionalInputSymbolShape(3), in_tensor4.GetOriginSymbolShape());
}

TEST_F(InferSymbolShapeContextUT, GetOutShapeOk) {
  auto sym_8 = ge::Symbol(8);
  auto sym_3 = ge::Symbol(3);
  auto sym_2 = ge::Symbol(2);
  auto sym_4 = ge::Symbol(4);
  auto sym_224 = ge::Symbol(224);
  auto sym_16 = ge::Symbol(16);
  gert::SymbolShape in_shape1({sym_8, sym_3, sym_224, sym_224});
  gert::SymbolShape in_shape2({sym_2, sym_2, sym_3, sym_8});
  gert::SymbolShape out_shape({sym_8, sym_3, sym_224, sym_224});

  auto context_holder = InferSymbolShapeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Inputs({&in_shape1, &in_shape2})
                            .Outputs({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferSymbolShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOutputSymbolShape(0), nullptr);
  EXPECT_EQ(*context->GetOutputSymbolShape(0), out_shape);

  EXPECT_EQ(context->GetOutputSymbolShape(1), nullptr);
}

TEST_F(InferSymbolShapeContextUT, GetOptionalInputTensorFailed_NotSetOptionalInput) {
  auto sym_8 = ge::Symbol(8);
  auto sym_3 = ge::Symbol(3);
  auto sym_224 = ge::Symbol(224);
  gert::SymbolShape in_shape1({sym_8, sym_3, sym_224, sym_224});

  auto infer_shape_func_addr = reinterpret_cast<void *>(0x11);

  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .IrInstanceNum({1, 0})
                            .KernelIONum(2, 0)
                            .NodeIoNum(1, 0)
                            .Inputs({&in_shape1, infer_shape_func_addr})
                            .Build();
  auto context = context_holder.GetContext<InferSymbolShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputSymbolTensor(0), nullptr);
  EXPECT_EQ(*context->GetInputSymbolShape(0), in_shape1);

  EXPECT_EQ(context->GetOptionalInputSymbolTensor(1), nullptr);
  EXPECT_NE(context->GetInputSymbolTensor(1), nullptr);
}

TEST_F(InferSymbolShapeContextUT, GetInputSymbolTensorValueOK) {
  auto sym_8 = ge::Symbol(8);
  auto sym_3 = ge::Symbol(3);
  auto sym_4 = ge::Symbol(4);
  auto sym_224 = ge::Symbol(224);
  auto sym_16 = ge::Symbol(16);
  gert::SymbolTensor in_tensor1({sym_3, sym_8, sym_16, sym_16});
  gert::SymbolTensor in_tensor2({sym_4}, {sym_3, sym_16, sym_224, sym_16});

  auto context_holder = InferSymbolShapeContextFaker()
                            .IrInputNum(2)
                            .IrInputInstanceNum({1, 1})
                            .NodeIoNum(2, 0)
                            .Inputs({reinterpret_cast<void *>(&in_tensor1), reinterpret_cast<void *>(&in_tensor2)})
                            .Build();
  auto context = context_holder.GetContext<InferSymbolShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputSymbolShape(0), nullptr);
  EXPECT_EQ(*context->GetInputSymbolShape(0), in_tensor1.GetOriginSymbolShape());

  EXPECT_NE(context->GetInputSymbolTensor(1), nullptr);
  EXPECT_EQ(context->GetInputSymbolTensor(1)->GetSymbolicValue()->size(), 4);
  EXPECT_EQ(context->GetInputSymbolTensor(1)->GetSymbolicValue()->at(0), sym_3);
  EXPECT_EQ(context->GetInputSymbolTensor(1)->GetSymbolicValue()->at(1), sym_16);
  EXPECT_EQ(context->GetInputSymbolTensor(1)->GetSymbolicValue()->at(2), sym_224);
  EXPECT_EQ(context->GetInputSymbolTensor(1)->GetSymbolicValue()->at(3), sym_16);
}

TEST_F(InferSymbolShapeContextUT, GetOptionalInputTensorOK_SetOptionalInput) {
  auto sym_8 = ge::Symbol(8);
  auto sym_3 = ge::Symbol(3);
  auto sym_4 = ge::Symbol(4);
  auto sym_224 = ge::Symbol(224);
  auto sym_16 = ge::Symbol(16);
  gert::SymbolTensor in_tensor1({sym_8, sym_3, sym_224, sym_224});
  gert::SymbolTensor in_tensor2({sym_4}, {sym_3, sym_16, sym_224, sym_16});

  auto context_holder =
      InferSymbolShapeContextFaker()
          .IrInputNum(2)
          .IrInputInstanceNum({1, 1})
          .NodeIoNum(2, 0)
          .Inputs({reinterpret_cast<void *>(&in_tensor1),
                   reinterpret_cast<void *>(&in_tensor2)})  // 接口内部会在末尾填充一个infershape的函数地址
          .Build();
  auto context = context_holder.GetContext<InferSymbolShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputSymbolShape(0), nullptr);
  EXPECT_EQ(*context->GetInputSymbolShape(0), in_tensor1.GetOriginSymbolShape());

  EXPECT_NE(context->GetOptionalInputSymbolTensor(1), nullptr);
  ASSERT_EQ(context->GetOptionalInputSymbolTensor(1)->GetSymbolicValue()->size(), 4);
  EXPECT_EQ(context->GetOptionalInputSymbolTensor(1)->GetSymbolicValue()->at(0), sym_3);
  EXPECT_EQ(context->GetOptionalInputSymbolTensor(1)->GetSymbolicValue()->at(1), sym_16);
  EXPECT_EQ(context->GetOptionalInputSymbolTensor(1)->GetSymbolicValue()->at(2), sym_224);
  EXPECT_EQ(context->GetOptionalInputSymbolTensor(1)->GetSymbolicValue()->at(3), sym_16);

  EXPECT_EQ(context->GetOptionalInputSymbolTensor(2), nullptr);
}

TEST_F(InferSymbolShapeContextUT, GetRequiredInputTensorOk) {
  auto sym_8 = ge::Symbol(8);
  auto sym_3 = ge::Symbol(3);
  auto sym_4 = ge::Symbol(4);
  auto sym_224 = ge::Symbol(224);
  auto sym_16 = ge::Symbol(16);
  gert::SymbolTensor in_tensor1({sym_8, sym_3, sym_224, sym_224});
  gert::SymbolTensor in_tensor2({sym_3, sym_16, sym_224, sym_16});
  gert::SymbolTensor in_tensor3({sym_4}, {sym_8, sym_3, sym_224, sym_224});
  gert::SymbolTensor in_tensor4({sym_4}, {sym_3, sym_8, sym_16, sym_16});

  gert::SymbolShape symbol_shape;


  auto context_holder =
      InferSymbolShapeContextFaker()
          .IrInputInstanceNum({1, 2, 0, 1})
          .NodeIoNum(4, 1)
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
          .Inputs({reinterpret_cast<void *>(&in_tensor1), reinterpret_cast<void *>(&in_tensor2),
                   reinterpret_cast<void *>(&in_tensor3), reinterpret_cast<void *>(&in_tensor4)})
          .Outputs({&symbol_shape})
          .Build();
  auto context = context_holder.GetContext<InferSymbolShapeContext>();
  ASSERT_NE(context, nullptr);

  // 第一个输入是必选输入，获取到shape
  ASSERT_NE(context->GetRequiredInputSymbolShape(0), nullptr);
  EXPECT_EQ(*context->GetRequiredInputSymbolShape(0), in_tensor1.GetOriginSymbolShape());

  // 第二个输入是动态输入，有两个实例，第一个实例获取到shape
  ASSERT_NE(context->GetDynamicInputSymbolShape(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputSymbolShape(1, 0), in_tensor2.GetOriginSymbolShape());

  // 第二个输入是动态输入，有两个实例，第一个实例获取到value
  ASSERT_NE(context->GetDynamicInputSymbolTensor(1, 1), nullptr);
  ASSERT_EQ(context->GetDynamicInputSymbolTensor(1, 1)->GetSymbolicValue()->size(), 4);
  EXPECT_EQ(context->GetDynamicInputSymbolTensor(1, 1)->GetSymbolicValue()->at(0), sym_8);
  EXPECT_EQ(context->GetDynamicInputSymbolTensor(1, 1)->GetSymbolicValue()->at(1), sym_3);
  EXPECT_EQ(context->GetDynamicInputSymbolTensor(1, 1)->GetSymbolicValue()->at(2), sym_224);
  EXPECT_EQ(context->GetDynamicInputSymbolTensor(1, 1)->GetSymbolicValue()->at(3), sym_224);

  // 第三个输入是可选输入，由于可选输入未被设置，获取到shape为nullptr
  EXPECT_EQ(context->GetOptionalInputSymbolShape(2), nullptr);

  // 第四个输入是必选输入
  ASSERT_NE(context->GetRequiredInputSymbolTensor(3), nullptr);
  ASSERT_EQ(context->GetRequiredInputSymbolTensor(3)->GetSymbolicValue()->size(), 4);
  EXPECT_EQ(context->GetRequiredInputSymbolTensor(3)->GetSymbolicValue()->at(0), sym_3);
  EXPECT_EQ(context->GetRequiredInputSymbolTensor(3)->GetSymbolicValue()->at(1), sym_8);
  EXPECT_EQ(context->GetRequiredInputSymbolTensor(3)->GetSymbolicValue()->at(2), sym_16);
  EXPECT_EQ(context->GetRequiredInputSymbolTensor(3)->GetSymbolicValue()->at(3), sym_16);
}
}  // namespace gert