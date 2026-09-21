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
#include <memory>
#include "vector.h"

#define VECTOR_BASIC_STEP 8

using namespace testing;

class UtestVectorTest : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(UtestVectorTest, VectorCaseBasic) {
  Vector a;
  uint8_t value = 1;
  InitVector(&a, sizeof(uint8_t));
  EXPECT_EQ(ReSizeVector(&a, 0), 0);
  EXPECT_EQ(CapacityVector(&a, 0), 0);
  EmplaceBackVector(&a, &value);
  value++;
  EXPECT_EQ(ReSizeVector(&a, 0), 1);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 0), 1);
  EXPECT_EQ(ReSizeVector(&a, 2), 2);

  EXPECT_EQ((uint8_t*)VectorAt(&a, 1), ((uint8_t*)VectorAt(&a, 0)) + 1);
  *(uint8_t*)VectorAt(&a, 1) = 2;
  RemoveVector(&a, 0);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 0), 2);
  EXPECT_EQ(ReSizeVector(&a, 0), 1);
  DeInitVector(&a);
  EXPECT_EQ((a.data == NULL), true);
  EXPECT_EQ(a.size, 0);
  EXPECT_EQ(a.capacity, 0);
  EXPECT_EQ(a.itemSize, 0);
}

TEST_F(UtestVectorTest, VectorCaseBasic_capacity) {
  Vector a;
  uint8_t value = 1;
  InitVector(&a, sizeof(uint8_t));
  EXPECT_EQ(ReSizeVector(&a, 1), 1);
  EXPECT_EQ(CapacityVector(&a, 0), 1);
  *(uint8_t*)VectorAt(&a, 0) = value;
  EmplaceBackVector(&a, &value);
  EXPECT_EQ(CapacityVector(&a, 0), VECTOR_BASIC_STEP);
  EXPECT_EQ(ReSizeVector(&a, 0), 2);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 0), value);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 1), value);

  for (int i = 2; i < VECTOR_BASIC_STEP + 1; i++) {
    value = (uint8_t)i;
    EmplaceBackVector(&a, &value);
  }
  value = 1;
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 0), value);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 1), value);
  for (int i = 2; i < VECTOR_BASIC_STEP + 1; i++) {
    value = (uint8_t)i;
    EXPECT_EQ(*(uint8_t*)VectorAt(&a, i), value);
  }
  EXPECT_EQ(CapacityVector(&a, 0), VECTOR_BASIC_STEP * 2);

  DeInitVector(&a);
}

TEST_F(UtestVectorTest, VectorCaseBasic_Emplace) {
  Vector a;
  uint8_t value = 1;
  InitVector(&a, sizeof(uint8_t));
  EmplaceVector(&a, 0, &value);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 0), value);
  value++;
  EmplaceVector(&a, 1, &value);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 1), value);
  value++;
  EmplaceVector(&a, 0, &value);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 0), value);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 1), 1);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 2), 2);
  value++;

  EmplaceHeadVector(&a, &value);
  EXPECT_EQ(*(uint8_t*)VectorAt(&a, 0), value);
  EXPECT_EQ(ReSizeVector(&a, 0), 4);
  DeInitVector(&a);
}

TEST_F(UtestVectorTest, VectorCaseBasic_new) {
  Vector *a = NewVector(uint8_t);
  uint8_t value = 1;
  InitVector(a, sizeof(uint8_t));
  EXPECT_EQ(ReSizeVector(a, 1), 1);
  EXPECT_EQ(CapacityVector(a, 0), 1);
  *(uint8_t*)VectorAt(a, 0) = value;
  EmplaceBackVector(a, &value);
  EXPECT_EQ(CapacityVector(a, 0), VECTOR_BASIC_STEP);
  EXPECT_EQ(ReSizeVector(a, 0), 2);
  EXPECT_EQ(*(uint8_t*)VectorAt(a, 0), value);
  EXPECT_EQ(*(uint8_t*)VectorAt(a, 1), value);

  for (int i = 2; i < VECTOR_BASIC_STEP + 1; i++) {
    value = (uint8_t)i;
    EmplaceBackVector(a, &value);
  }
  value = 1;
  EXPECT_EQ(*(uint8_t*)VectorAt(a, 0), value);
  EXPECT_EQ(*(uint8_t*)VectorAt(a, 1), value);
  for (int i = 2; i < VECTOR_BASIC_STEP + 1; i++) {
    value = (uint8_t)i;
    EXPECT_EQ(*(uint8_t*)VectorAt(a, i), value);
  }
  EXPECT_EQ(CapacityVector(a, 0), VECTOR_BASIC_STEP * 2);

  DestroyVector(a);
}
