/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/tensor.h"
#include "graph/ge_tensor.h"
#include <gtest/gtest.h>
namespace gert {
class TensorUT : public testing::Test {};
TEST_F(TensorUT, ConstructOk) {
  Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                kOnDeviceHbm,                                // placement
                ge::DT_FLOAT16,                              //dt
                nullptr};
  const Tensor &t = tensor;

  EXPECT_EQ(t.GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(t.GetStorageShape(), Shape({16, 3, 224, 224}));

  EXPECT_EQ(t.GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(t.GetStorageFormat(), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(t.GetExpandDimsType(), ExpandDimsType{});

  EXPECT_EQ(t.GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(t.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(t.GetAddr(), nullptr);
  EXPECT_EQ(t.GetData<int64_t>(), nullptr);
}

TEST_F(TensorUT, GetDataAddrFollowingOk) {
  Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                kFollowing,                                // placement
                ge::DT_FLOAT16,                              //dt
                nullptr};
  const Tensor &t = tensor;

  EXPECT_EQ(t.GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(t.GetStorageShape(), Shape({16, 3, 224, 224}));

  EXPECT_EQ(t.GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(t.GetStorageFormat(), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(t.GetExpandDimsType(), ExpandDimsType{});

  EXPECT_EQ(t.GetPlacement(), kFollowing);
  EXPECT_EQ(t.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(t.GetAddr(), &tensor + 1);
  EXPECT_EQ(t.GetData<int64_t>(), reinterpret_cast<const int64_t *>(&tensor + 1));
}

TEST_F(TensorUT, GetCopiedDataAddrFollowingOk) {
  Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}},       // shape
                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                kFollowing,                                // placement
                ge::DT_FLOAT16,                              //dt
                nullptr};
  Tensor t = {{}, {}, {}, {}, nullptr};
  memcpy(static_cast<void *>(&t), static_cast<void *>(&tensor), sizeof(tensor));

  EXPECT_EQ(t.GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(t.GetStorageShape(), Shape({16, 3, 224, 224}));

  EXPECT_EQ(t.GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(t.GetStorageFormat(), ge::FORMAT_FRACTAL_NZ);
  EXPECT_EQ(t.GetExpandDimsType(), ExpandDimsType{});

  EXPECT_EQ(t.GetPlacement(), kFollowing);
  EXPECT_EQ(t.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(t.GetAddr(), &t + 1);
  EXPECT_EQ(t.GetData<int64_t>(), reinterpret_cast<int64_t *>(&t + 1));
}

TEST_F(TensorUT, SetGetShapeOk) {
  Tensor t = {{}, {}, {}, {}, nullptr};
  const Tensor &ct = t;
  t.MutableOriginShape() = Shape{8,3,224,224};
  t.MutableStorageShape() = Shape{8,1,224,224,16};
  EXPECT_EQ(t.GetOriginShape(), Shape({8,3,224,224}));
  EXPECT_EQ(t.GetStorageShape(), Shape({8,1,224,224,16}));
  EXPECT_EQ(ct.GetOriginShape(), Shape({8,3,224,224}));
  EXPECT_EQ(ct.GetStorageShape(), Shape({8,1,224,224,16}));
}

TEST_F(TensorUT, SetGetFormatOk) {
  Tensor tensor = {{}, {}, {}, {}, nullptr};
  const Tensor &t = tensor;
  tensor.SetOriginFormat(ge::FORMAT_NHWC);
  tensor.SetStorageFormat(ge::FORMAT_NC1HWC0);

  EXPECT_EQ(t.GetOriginFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(t.GetStorageFormat(), ge::FORMAT_NC1HWC0);

  EXPECT_EQ(t.GetFormat().GetOriginFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(t.GetFormat().GetStorageFormat(), ge::FORMAT_NC1HWC0);
}

TEST_F(TensorUT, SetGetPlacementOk) {
  Tensor t = {{}, {}, {}, {}, nullptr};
  const Tensor &ct = t;
  t.SetPlacement(kOnHost);
  EXPECT_EQ(t.GetPlacement(), kOnHost);
  EXPECT_EQ(ct.GetPlacement(), kOnHost);
}

TEST_F(TensorUT, SetGetDataTypeOk) {
  Tensor t = {{}, {}, {}, {}, nullptr};
  const Tensor &ct = t;
  t.SetDataType(ge::DT_FLOAT16);
  EXPECT_EQ(t.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(ct.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(TensorUT, SetGetAddrOk) {
  Tensor t = {{}, {}, {}, {}, nullptr};
  const Tensor &ct = t;
  void *a = &t;

  TensorData td(a, nullptr);
  t.SetData(std::move(td));
  EXPECT_EQ(t.GetAddr(), a);
  EXPECT_EQ(ct.GetAddr(), a);

  EXPECT_EQ(t.GetData<Tensor>(), &t);
  EXPECT_EQ(ct.GetData<Tensor>(), &t);
}

TEST_F(TensorUT, GetTensorDataOk) {
  Tensor t = {{}, {}, {}, {}, nullptr};
  auto a = reinterpret_cast<void *>(10);
  t.MutableTensorData() = TensorData{a, nullptr};
  EXPECT_EQ(t.GetAddr(), a);
}

TEST_F(TensorUT, GetTensorPlacementOk) {
  Tensor t = {{}, {}, kOnHost, {}, nullptr};
  EXPECT_EQ(t.GetPlacement(), kOnHost);
}

TEST_F(TensorUT, GetTensorSizeOk) {
  StorageShape sh({1, 2, 3}, {1, 2, 3});
  Tensor t = {sh, {}, {}, ge::DT_FLOAT, nullptr};
  EXPECT_EQ(t.GetSize(), 24);
}

TEST_F(TensorUT, CreateFollowingCheckTotalSize) {
  size_t total_size;
  auto ptr = Tensor::CreateFollowing(32, ge::DataType::DT_INT8, total_size);
  EXPECT_NE(ptr, nullptr);
  auto tensor = reinterpret_cast<Tensor *>(ptr.get());
  EXPECT_EQ(tensor->GetSize(), 32);
}

TEST_F(TensorUT, CreateFollowingWithTensorCheckTotalSize) {
  size_t total_size;
  auto ptr = Tensor::CreateFollowing(ge::DT_FLOAT, 8U, total_size);
  EXPECT_NE(ptr, nullptr);
  auto tensor = reinterpret_cast<Tensor *>(ptr.get());
  EXPECT_EQ(tensor->GetSize(), 8);
}

TEST_F(TensorUT, CreateFollowingWithTensorUseStringTypeCheckTotalSize) {
  size_t total_size;
  auto ptr = Tensor::CreateFollowing(ge::DT_STRING, 160U, total_size);
  EXPECT_NE(ptr, nullptr);
  auto tensor = reinterpret_cast<Tensor *>(ptr.get());
  EXPECT_EQ(tensor->GetSize(), 160);
}
}  // namespace gert
