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
#include "graph/cache_policy/compile_cache_desc.h"

namespace ge {
class UtestCompileCachePolicyHasher : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderConstrutFromPtr) {
  uint8_t data1[2] = {0U,1U};
  BinaryHolder holder1(data1, sizeof(data1));
  const uint8_t *dataPtr = holder1.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size1 = holder1.GetDataLen();
  ASSERT_EQ(size1, sizeof(data1));
  ASSERT_EQ(dataPtr[0], 0);
  ASSERT_EQ(dataPtr[1], 1);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderCopyConstrut) {
  uint8_t data1[2] = {0U,1U};
  BinaryHolder holder1(data1, sizeof(data1));
  const uint8_t *dataPtr = holder1.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size1 = holder1.GetDataLen();
  ASSERT_EQ(size1, sizeof(data1));

  BinaryHolder holder2 = holder1;
  ASSERT_EQ((holder1 != holder2), false);
  ASSERT_NE(holder1.GetDataPtr(), holder2.GetDataPtr());
  BinaryHolder holder3(holder1);
  ASSERT_NE(holder1.GetDataPtr(), holder3.GetDataPtr());
  ASSERT_EQ((holder1 != holder3), false);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderMoveConstrut) {
  uint8_t data1[2] = {0U,1U};
  BinaryHolder holder1(data1, sizeof(data1));

  BinaryHolder holder2 = std::move(holder1);
  ASSERT_EQ(holder1.GetDataPtr(), nullptr);
  ASSERT_NE(holder2.GetDataPtr(), nullptr);
  ASSERT_EQ(holder1.GetDataLen(), 0);
  ASSERT_EQ(holder2.GetDataLen(), sizeof(data1));

  const uint8_t *dataPtr = holder2.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size2 = holder2.GetDataLen();
  ASSERT_EQ(size2, sizeof(data1));
  ASSERT_EQ(dataPtr[0], 0);
  ASSERT_EQ(dataPtr[1], 1);

  BinaryHolder holder3(std::move(holder2));
  ASSERT_EQ(holder2.GetDataPtr(), nullptr);
  ASSERT_NE(holder3.GetDataPtr(), nullptr);
  ASSERT_EQ(holder2.GetDataLen(), 0);
  ASSERT_EQ(holder3.GetDataLen(), sizeof(data1));
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHoldercreateFromUniquePtr) {
  auto data_ptr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[10]);
  ASSERT_NE(data_ptr, nullptr);
  const uint8_t *data_ptr_real = data_ptr.get();
  auto holder2 = BinaryHolder::createFrom(std::move(data_ptr), 10);
  ASSERT_NE(holder2->GetDataPtr(), nullptr);
  ASSERT_EQ(holder2->GetDataPtr(), data_ptr_real);
  ASSERT_EQ(holder2->GetDataLen(), 10);
  ASSERT_EQ(data_ptr, nullptr);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHoldercreateFromUniquePtrFail) {
  auto holder2 = BinaryHolder::createFrom(nullptr, 0);
  ASSERT_EQ(holder2->GetDataPtr(), nullptr);
  ASSERT_EQ(holder2->GetDataLen(), 0);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderEqual) {
  uint8_t data1[8] = {0U,1U,2U,3U,4U,5U,6U,7U};
  uint8_t data2[8] = {0U,1U,2U,3U,4U,5U,6U,7U};

  BinaryHolder holder1(data1, sizeof(data1));
  const uint8_t *dataPtr = holder1.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size1 = holder1.GetDataLen();
  ASSERT_EQ(size1, sizeof(data1));

  BinaryHolder holder2(data2, sizeof(data2));
  ASSERT_EQ((holder1 != holder2), false);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderDiffBecauseLength) {
  uint8_t data1[8] = {0U,1U,2U,3U,4U,5U,6U,7U};
  uint8_t data2[9] = {0U,1U,2U,3U,4U,5U,7U,9U,11U};

  BinaryHolder holder1(data1, sizeof(data1));
  ASSERT_EQ(holder1.GetDataLen(), sizeof(data1));
  BinaryHolder holder2(data2, sizeof(data2));
  ASSERT_EQ(holder2.GetDataLen(), sizeof(data2));
  ASSERT_EQ((holder1 != holder2), true);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderDiffBecauseVaule) {
  uint8_t data1[8] = {0U,1U,2U,3U,4U,5U,6U,7U};
  uint8_t data2[8] = {1U,1U,2U,3U,4U,5U,6U,7U};

  BinaryHolder holder1(data1, sizeof(data1));
  ASSERT_EQ(holder1.GetDataLen(), sizeof(data1));
  BinaryHolder holder2(data2, sizeof(data2));
  ASSERT_EQ(holder2.GetDataLen(), sizeof(data2));
  ASSERT_EQ((holder1 != holder2), true);
}

TEST_F(UtestCompileCachePolicyHasher, TestGetCacheDescHashWithoutShape) { 
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("1111");
  TensorInfoArgs tensor_info_args(FORMAT_ND, FORMAT_ND, DT_BF16);
  cache_desc->AddTensorInfo(tensor_info_args);
  CacheHashKey id = cache_desc->GetCacheDescHash();
  cache_desc->SetOpType("2222");
  CacheHashKey id_another = cache_desc->GetCacheDescHash();
  ASSERT_NE(id, id_another);
}
}
