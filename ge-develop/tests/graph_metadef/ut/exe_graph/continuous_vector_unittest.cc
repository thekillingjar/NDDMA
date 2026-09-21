/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/continuous_vector.h"
#include "securec.h"
#include <gtest/gtest.h>
namespace gert {
class ContinuousVectorUT : public testing::Test {};
TEST_F(ContinuousVectorUT, CreateOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  auto c_vec = reinterpret_cast<const ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 16);
  EXPECT_EQ(c_vec->GetSize(), 0);
  EXPECT_EQ(c_vec->GetCapacity(), 16);
}
TEST_F(ContinuousVectorUT, SetSizeOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->SetSize(8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(vec->GetSize(), 8);
  EXPECT_EQ(vec->SetSize(16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(vec->GetSize(), 16);
  EXPECT_EQ(vec->SetSize(0), ge::GRAPH_SUCCESS);
  EXPECT_EQ(vec->GetSize(), 0);
}
TEST_F(ContinuousVectorUT, SetSizeFailedOutOfBounds) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_NE(vec->SetSize(17), ge::GRAPH_SUCCESS);
}
TEST_F(ContinuousVectorUT, CreateNone) {
  auto vec_holder = ContinuousVector::Create<size_t>(0);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 0);
}
TEST_F(ContinuousVectorUT, WriteOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(2);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 2);

  EXPECT_EQ(vec->SetSize(2), ge::GRAPH_SUCCESS);
  auto data = reinterpret_cast<size_t *>(vec->MutableData());
  data[0] = 1024;
  data[1] = 2048;
  EXPECT_EQ(vec->GetSize(), 2);
  EXPECT_EQ(reinterpret_cast<const size_t *>(vec->GetData())[0], 1024);
  EXPECT_EQ(reinterpret_cast<const size_t *>(vec->GetData())[1], 2048);
}
TEST_F(ContinuousVectorUT, TypedOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->SetSize(4), ge::GRAPH_SUCCESS);
  auto data = reinterpret_cast<size_t *>(vec->MutableData());
  data[0] = 1024;
  data[1] = 2048;
  data[2] = 4096;
  data[3] = 8192;

  auto t_vec = reinterpret_cast<const TypedContinuousVector<size_t> *>(vec);
  EXPECT_EQ(t_vec->GetSize(), 4);
  EXPECT_EQ(t_vec->GetCapacity(), 16);
  EXPECT_EQ(t_vec->GetData()[0], 1024);
  EXPECT_EQ(t_vec->GetData()[1], 2048);
  EXPECT_EQ(t_vec->GetData()[2], 4096);
  EXPECT_EQ(t_vec->GetData()[3], 8192);
  auto mt_vec = reinterpret_cast<TypedContinuousVector<size_t> *>(vec);
  EXPECT_EQ(mt_vec->MutableData()[0], 1024);
  EXPECT_EQ(mt_vec->MutableData()[1], 2048);
  EXPECT_EQ(mt_vec->MutableData()[2], 4096);
  EXPECT_EQ(mt_vec->MutableData()[3], 8192);
}

TEST_F(ContinuousVectorUT, GetOverHeadLengthOk) {
  EXPECT_EQ(ContinuousVectorVector::GetOverHeadLength(0), sizeof(ContinuousVectorVector));
  EXPECT_EQ(ContinuousVectorVector::GetOverHeadLength(1), sizeof(ContinuousVectorVector));
  EXPECT_EQ(ContinuousVectorVector::GetOverHeadLength(3), sizeof(ContinuousVectorVector) + (3U - 1U) * sizeof(size_t));
}

TEST_F(ContinuousVectorUT, AddOk) {
  std::vector<std::vector<int64_t>> vector_vector_list{{0, 2}, {1, 1}, {2, 4, 3}, {}, {1}};

  size_t inner_vector_num = vector_vector_list.size();
  size_t total_length = ContinuousVectorVector::GetOverHeadLength(inner_vector_num);
  for (const auto &inner_vector : vector_vector_list) {
    size_t inner_vector_length = sizeof(ContinuousVector) + sizeof(int64_t) * inner_vector.size();
    total_length += inner_vector_length;
  }

  std::vector<uint8_t> buf(total_length);
  auto cvv = new (buf.data()) ContinuousVectorVector();
  ASSERT_NE(cvv, nullptr);
  cvv->Init(inner_vector_num);

  for (const auto &inner_vector : vector_vector_list) {
    auto cv = cvv->Add<int64_t>(inner_vector.size());
    cv->Init(inner_vector.size());
    cv->SetSize(inner_vector.size());
    if (inner_vector.empty()) {
      continue;
    }
    auto ret = memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(int64_t),
             inner_vector.data(), inner_vector.size() * sizeof(int64_t));
    ASSERT_EQ(ret, EOK);
  }

  auto new_cvv = reinterpret_cast<ContinuousVectorVector *>(buf.data());
  ASSERT_EQ(new_cvv->GetSize(), vector_vector_list.size());
  for (size_t i = 0U; i < vector_vector_list.size(); ++i) {
    auto new_cv = new_cvv->Get(i);
    ASSERT_EQ(new_cv->GetSize(), vector_vector_list[i].size());
    ASSERT_EQ(new_cv->GetSize(), new_cv->GetCapacity());
    const int64_t *data = reinterpret_cast<const int64_t *>(new_cv->GetData());
    for (size_t j = 0U; j < new_cv->GetSize(); ++j) {
      EXPECT_EQ(data[j], vector_vector_list[i][j]);
    }
  }
}

TEST_F(ContinuousVectorUT, AddFailed_WithoutInit) {
  std::vector<uint8_t> buf(100);
  auto cvv = new (buf.data()) ContinuousVectorVector();
  ASSERT_NE(cvv, nullptr);
  auto cv = cvv->Add<int64_t>(2);
  EXPECT_EQ(cv, nullptr);
}

TEST_F(ContinuousVectorUT, AddFailed_OutBounds) {
  std::vector<uint8_t> buf(500);
  auto cvv = new (buf.data()) ContinuousVectorVector();
  ASSERT_NE(cvv, nullptr);
  cvv->Init(2);
  auto cv = cvv->Add<int64_t>(2);
  EXPECT_NE(cv, nullptr);
  cv = cvv->Add<int64_t>(2);
  EXPECT_NE(cv, nullptr);
  cv = cvv->Add<int64_t>(2);
  EXPECT_EQ(cv, nullptr);
}
}  // namespace gert
