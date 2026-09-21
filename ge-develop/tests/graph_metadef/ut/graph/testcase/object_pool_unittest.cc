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
#include <vector>

#include "graph/utils/object_pool.h"
#include "graph/ge_tensor.h"

using std::vector;
namespace ge {
class UTObjectPool : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTObjectPool, Add) {
  ObjectPool<GeTensor, 10> object_pool_;
  ASSERT_TRUE(object_pool_.IsEmpty());

  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);

  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {

  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);
  object_pool_.Release(std::move(ge_tensor));
  ASSERT_EQ(object_pool_.handlers_.size(), 1);
}

TEST_F(UTObjectPool, UniqueToShared) {
  ObjectPool<GeTensor, 10> object_pool_;
  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);

  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {

  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);

  {
    std::shared_ptr<GeTensor> shared_tensor(ge_tensor.get(), [](GeTensor *){});
  }
  ASSERT_NE(ge_tensor, nullptr);
  object_pool_.Release(std::move(ge_tensor));
  ASSERT_EQ(object_pool_.handlers_.size(), 1);
}

TEST_F(UTObjectPool, GetFromFull) {
  ObjectPool<GeTensor, 1> object_pool_;

  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);
  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {
  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);
  object_pool_.Release(std::move(ge_tensor));

  ASSERT_TRUE(object_pool_.IsFull());
  auto tmp = object_pool_.Acquire();
  ASSERT_TRUE(object_pool_.IsEmpty());
}


TEST_F(UTObjectPool, AutoRelease) {
  ObjectPool<GeTensor, 10> object_pool_;
  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);

  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {
  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);
  {
    std::queue<std::unique_ptr<GeTensor>> shared_tensors_;
    shared_tensors_.push(std::move(ge_tensor));

  }
  ASSERT_EQ(ge_tensor, nullptr);
  object_pool_.Release(std::move(ge_tensor));
  ASSERT_TRUE(object_pool_.IsEmpty());
}
}
