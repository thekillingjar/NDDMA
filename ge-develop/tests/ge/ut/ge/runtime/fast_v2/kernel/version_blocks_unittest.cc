/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/version_blocks.h"
#include <gtest/gtest.h>
#include <list>
#include "faker/multi_stream_allocator_faker.h"
namespace gert {
namespace memory {
class VersionBlocksUT : public testing::Test {};

TEST_F(VersionBlocksUT, ToAllBit_Ok) {
  auto all_bit = BaseVersionBlocks::ToAllBit(4, 0);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(0)) == 0U);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(1)) != 0U);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(2)) != 0U);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(3)) != 0U);
}
TEST_F(VersionBlocksUT, ToAllBit_Ok1) {
  auto all_bit = BaseVersionBlocks::ToAllBit(4, 1);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(0)) != 0U);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(1)) == 0U);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(2)) != 0U);
  ASSERT_TRUE((all_bit & BaseVersionBlocks::ToBit(3)) != 0U);
}
// 遍历方法数据不好构造，在l2_allocator中一起测试
}  // namespace memory
}  // namespace gert