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
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "graph/compute_graph.h"
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "faker/fake_value.h"

namespace gert {
class ExtendedContextUT : public testing::Test {};
TEST_F(ExtendedContextUT, GetAttrs) {
  StorageShape i1;
  StorageShape i2;
  StorageShape o1;

  auto ge_tensor = FakeGeTensorHolder()
                       .DataType(ge::DT_FLOAT16)
                       .OriginFormat(ge::FORMAT_NCHW)
                       .StorageFormat(ge::FORMAT_NC1HWC0)
                       .OriginShape({8, 3, 224, 224})
                       .StorageShape({8, 1, 224, 224, 16})
                       .Build();
  auto holder = InferShapeContextFaker()
                    .IrInputNum(2)
                    .InputShapes({&i1, &i2})
                    .OutputShapes({&o1})
                    .NodeIoNum(2, 1)
                    .NodeAttrs({{"int", ge::AnyValue::CreateFrom<int64_t>(10)},
                                {"str", ge::AnyValue::CreateFrom<std::string>("Hello!")},
                                {"bool", ge::AnyValue::CreateFrom<bool>(true)},
                                {"float", ge::AnyValue::CreateFrom<float>(10.101)},
                                {"list_int", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3, 4096})},
                                {"tensor", ge::AnyValue::CreateFrom<ge::GeTensor>(*ge_tensor)}})
                    .Build();

  auto context = holder.GetContext<ExtendedKernelContext>();

  // 下面的代码可以出现在infershape或tiling的函数中：
  auto attrs = context->GetAttrs();
  ASSERT_NE(attrs, nullptr);
  ASSERT_EQ(attrs->GetAttrNum(), 6);

  auto attr_0 = attrs->GetAttrPointer<int64_t>(0);
  ASSERT_NE(attr_0, nullptr);
  EXPECT_EQ(*attr_0, 10);

  auto attr1 = attrs->GetAttrPointer<char>(1);
  ASSERT_NE(attr1, nullptr);
  EXPECT_STREQ(attr1, "Hello!");

  auto attr2 = attrs->GetAttrPointer<bool>(2);
  ASSERT_NE(attr2, nullptr);
  EXPECT_TRUE(*attr2);

  auto attr3 = attrs->GetAttrPointer<float>(3);
  ASSERT_NE(attr3, nullptr);
  EXPECT_FLOAT_EQ(*attr3, 10.101);

  auto attr4 = attrs->GetAttrPointer<ContinuousVector>(4);
  ASSERT_NE(attr4, nullptr);
  ASSERT_EQ(attr4->GetSize(), 3);
  auto data = reinterpret_cast<const int64_t *>(attr4->GetData());
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 3);
  EXPECT_EQ(data[2], 4096);

  auto attr5 = attrs->GetAttrPointer<gert::Tensor>(5);
  ASSERT_NE(attr5, nullptr);
  EXPECT_EQ(attr5->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(attr5->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(attr5->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(attr5->GetOriginShape(), Shape({8,3,224,224}));
  EXPECT_EQ(attr5->GetStorageShape(), Shape({8,1,224,224,16}));
  EXPECT_EQ(memcmp(attr5->GetData<uint8_t>(), ge_tensor->GetData().GetData(), ge_tensor->GetData().GetSize()), 0);

  EXPECT_EQ(attrs->GetAttrPointer<bool>(6), nullptr);
}
}  // namespace gert