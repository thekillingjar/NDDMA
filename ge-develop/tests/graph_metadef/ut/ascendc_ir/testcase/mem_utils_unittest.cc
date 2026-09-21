/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/utils/mem_utils.h"

namespace ge {
class UtestMemUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
TEST(UtestMemUtils, CreateTQueConfigSuccess) {
  auto tque = MemUtils::CreateTQueConfig(Position::kPositionVecIn, 1, 2);
  EXPECT_EQ(tque.pos_, Position::kPositionVecIn);
  EXPECT_EQ(tque.queue_attr_.buf_num, 2);
  EXPECT_EQ(tque.queue_attr_.depth, 1);
}

TEST(UtestMemUtils, CreateTQueConfigFailed) {
  EXPECT_EQ(MemUtils::CreateTQueConfig(Position::kPositionGM, 1, 2).pos_, Position::kPositionInvalid);
}

TEST(UtestMemUtils, CreateTBufConfigIncSuccess) {
  auto tbuf = MemUtils::CreateTBufConfig(Position::kPositionVecIn);
  EXPECT_EQ(tbuf.pos_, Position::kPositionVecIn);
  auto old = tbuf.buf_attr_.id;
  auto tbuf2 = MemUtils::CreateTBufConfig(Position::kPositionVecOut);
  EXPECT_EQ(tbuf2.buf_attr_.id - old, 1);
}

TEST(UtestMemUtils, CreateTQueConfigIncSuccess) {
  auto tque1 = MemUtils::CreateTQueConfig(Position::kPositionVecIn, 1, 2);
  auto old = tque1.queue_attr_.id;
  auto tque2 = MemUtils::CreateTQueConfig(Position::kPositionVecIn, 2, 4);
  EXPECT_EQ(tque1.queue_attr_.depth, 1);
  EXPECT_EQ(tque1.queue_attr_.buf_num, 2);
  EXPECT_EQ(tque2.queue_attr_.depth, 2);
  EXPECT_EQ(tque2.queue_attr_.buf_num, 4);
  EXPECT_EQ(tque2.queue_attr_.id - old, 1);
}

TEST(UtestMemUtils, CreateTQueConfigBindTensorsSuccess) {
  auto tque = MemUtils::CreateTQueConfig(Position::kPositionVecIn, 10, 20);
  auto tque1 = MemUtils::CreateTQueConfig(Position::kPositionVecIn, 10, 20);
  auto old = tque1.queue_attr_.id;
  EXPECT_EQ(tque1.queue_attr_.depth, 10);
  EXPECT_EQ(tque1.queue_attr_.buf_num, 20);
  AscTensorAttr output1;
  AscTensorAttr output2;
  AscTensorAttr output3;
  AscTensorAttr output4;
  AscTensorAttr output5;
  AscTensorAttr output6;
  tque1.BindTensors(output1, output2);
  tque1.BindTensors(output3, output4, output5);
  tque1.BindTensors(output6);
  EXPECT_EQ(output1.que.id, old);
  EXPECT_EQ(output1.buf.id, kIdNone);
  EXPECT_EQ(output1.que.depth, 10);
  EXPECT_EQ(output1.que.buf_num, 20);
  EXPECT_EQ(output2.que.id, old);
  EXPECT_EQ(output2.buf.id, kIdNone);
  EXPECT_EQ(output3.que.id, old);
  EXPECT_EQ(output3.buf.id, kIdNone);
  EXPECT_EQ(output4.que.id, old);
  EXPECT_EQ(output4.buf.id, kIdNone);
  EXPECT_EQ(output5.que.id, old);
  EXPECT_EQ(output5.buf.id, kIdNone);
  EXPECT_EQ(output6.que.id, old);
  EXPECT_EQ(output6.buf.id, kIdNone);
  EXPECT_EQ(output6.que.depth, 10);
  EXPECT_EQ(output6.que.buf_num, 20);
}

TEST(UtestMemUtils, CreateTBufConfigBindTensorsSuccess) {
  auto tbuf = MemUtils::CreateTBufConfig(Position::kPositionVecIn);
  auto tbuf1 = MemUtils::CreateTBufConfig(Position::kPositionVecIn);
  auto old = tbuf1.buf_attr_.id;
  EXPECT_EQ(tbuf1.pos_, Position::kPositionVecIn);
  AscTensorAttr output1;
  AscTensorAttr output2;
  AscTensorAttr output3;
  AscTensorAttr output4;
  AscTensorAttr output5;
  AscTensorAttr output6;
  tbuf1.BindTensors(output1, output2);
  tbuf1.BindTensors(output3, output4, output5);
  tbuf1.BindTensors(output6);
  EXPECT_EQ(output1.buf.id, old);
  EXPECT_EQ(output2.buf.id, old);
  EXPECT_EQ(output3.buf.id, old);
  EXPECT_EQ(output4.buf.id, old);
  EXPECT_EQ(output5.buf.id, old);
  EXPECT_EQ(output6.buf.id, old);
}

TEST(UtestMemUtils, MergeScopeSuccess) {
  auto tbuf = MemUtils::CreateTBufConfig(Position::kPositionVecIn);
  auto tbuf1 = MemUtils::CreateTBufConfig(Position::kPositionVecIn);
  auto tbuf2 = MemUtils::CreateTBufConfig(Position::kPositionVecIn);
  auto tbuf3 = MemUtils::CreateTBufConfig(Position::kPositionVecIn);
  AscTensorAttr output1;
  output1.opt.merge_scope = 2;
  AscTensorAttr output2;
  output2.opt.merge_scope = 3;
  AscTensorAttr output3;
  output3.opt.merge_scope = 4;
  AscTensorAttr output4;
  output4.opt.merge_scope = 5;
  AscTensorAttr output5;
  output5.opt.merge_scope = 6;
  MemUtils::MergeScope(output1, output2, output3, output4, output5);
  EXPECT_EQ(output1.opt.merge_scope, output2.opt.merge_scope);
  EXPECT_EQ(output1.opt.merge_scope, output3.opt.merge_scope);
  EXPECT_EQ(output1.opt.merge_scope, output4.opt.merge_scope);
  EXPECT_EQ(output1.opt.merge_scope, output5.opt.merge_scope);
}
}
