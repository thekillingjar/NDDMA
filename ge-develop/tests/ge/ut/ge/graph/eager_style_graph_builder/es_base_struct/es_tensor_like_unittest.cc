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
#include "es_tensor_like.h"
#include "es_graph_builder.h"

using namespace ge::es;
class EsTensorLikeUT : public ::testing::Test {
 protected:
  void SetUp() override {
    builder = new EsGraphBuilder("tensor_like_test");
    c_builder = builder->GetCGraphBuilder();
    tensor = builder->CreateScalar(int64_t(123));
  }
  void TearDown() override {
    delete builder;
  }
  EsGraphBuilder *builder;
  EsCGraphBuilder *c_builder;
  EsTensorHolder tensor{nullptr};
};

TEST_F(EsTensorLikeUT, Constructor_FromTensorHolder) {
  EsTensorLike tensor_like(tensor);
  auto *owner_builder = tensor_like.GetOwnerBuilder();
  EXPECT_NE(owner_builder, nullptr);
}

TEST_F(EsTensorLikeUT, Constructor_FromNullptr) {
  EsTensorLike tensor_like(nullptr);
  auto *owner_builder = tensor_like.GetOwnerBuilder();
  EXPECT_EQ(owner_builder, nullptr);
}

TEST_F(EsTensorLikeUT, Constructor_FromInt64) {
  EsTensorLike tensor_like(int64_t(42));
  EsCGraphBuilder *owner_builder = tensor_like.GetOwnerBuilder();
  EXPECT_EQ(owner_builder, nullptr);
}

TEST_F(EsTensorLikeUT, Constructor_FromFloat) {
  EsTensorLike tensor_like(3.14f);
  EsCGraphBuilder *owner_builder = tensor_like.GetOwnerBuilder();
  EXPECT_EQ(owner_builder, nullptr);
}

TEST_F(EsTensorLikeUT, Constructor_FromVectorInt64) {
  std::vector<int64_t> vec = {1, 2, 3, 4};
  EsTensorLike tensor_like(vec);
  EsCGraphBuilder *owner_builder = tensor_like.GetOwnerBuilder();
  EXPECT_EQ(owner_builder, nullptr);
}

TEST_F(EsTensorLikeUT, Constructor_FromVectorFloat) {
  std::vector<float> vec = {1.1f, 2.2f, 3.3f};
  EsTensorLike tensor_like(vec);
  EsCGraphBuilder *owner_builder = tensor_like.GetOwnerBuilder();
  EXPECT_EQ(owner_builder, nullptr);
}

TEST_F(EsTensorLikeUT, GetCTensorHolder_FromTensor) {
  EsTensorLike tensor_like(tensor);
  auto result = tensor_like.ToTensorHolder(c_builder);
  EXPECT_EQ(result.GetCTensorHolder(), tensor.GetCTensorHolder());
}

TEST_F(EsTensorLikeUT, GetCTensorHolder_FromInt64) {
  EsTensorLike tensor_like(int64_t(42));
  auto result = tensor_like.ToTensorHolder(c_builder);
  EXPECT_NE(result.GetCTensorHolder(), nullptr);
}

TEST_F(EsTensorLikeUT, GetCTensorHolder_FromFloat) {
  EsTensorLike tensor_like(3.14f);
  auto result = tensor_like.ToTensorHolder(c_builder);
  EXPECT_NE(result.GetCTensorHolder(), nullptr);
}

TEST_F(EsTensorLikeUT, GetCTensorHolder_FromVectorInt64) {
  EsTensorLike tensor_like(std::vector<int64_t>{1, 2, 3, 4, 5});
  auto result = tensor_like.ToTensorHolder(c_builder);
  EXPECT_NE(result.GetCTensorHolder(), nullptr);
}

TEST_F(EsTensorLikeUT, GetCTensorHolder_FromVectorFloat) {
  EsTensorLike tensor_like(std::vector<float>{1.1f, 2.2f, 3.3f});
  auto result = tensor_like.ToTensorHolder(c_builder);
  EXPECT_NE(result.GetCTensorHolder(), nullptr);
}

TEST_F(EsTensorLikeUT, ResolveBuilder_WithTensor) {
  auto *owner_builder = ResolveBuilder(tensor, 3.14f);
  EXPECT_NE(owner_builder, nullptr);
  EXPECT_EQ(owner_builder, c_builder);

  auto *owner_builder1 = ResolveBuilder(3.14f, nullptr, tensor);
  EXPECT_NE(owner_builder1, nullptr);
  EXPECT_EQ(owner_builder1, c_builder);
}

TEST_F(EsTensorLikeUT, ResolveBuilder_NoTensor) {
  auto *owner_builder = ResolveBuilder(3.14f);
  EXPECT_EQ(owner_builder, nullptr);

  auto *owner_builder1 = ResolveBuilder(3.14f, int64_t(40), std::vector<float>{3.14, 1.0});
  EXPECT_EQ(owner_builder1, nullptr);
}

TEST_F(EsTensorLikeUT, ResolveBuilder_WithVector) {
  auto input_tensor1 = builder->CreateInput(1, "input1", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  auto input_tensor2 = builder->CreateInput(2, "input2", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  auto input_tensor3 = builder->CreateInput(3, "input3", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  std::vector input_vec{input_tensor1, input_tensor2, input_tensor3};
  auto *owner_builder = ResolveBuilder(3.14f, input_vec, tensor, std::vector<float>{3.14, 1.0});
  EXPECT_NE(owner_builder, nullptr);
  EXPECT_EQ(owner_builder, c_builder);
}
