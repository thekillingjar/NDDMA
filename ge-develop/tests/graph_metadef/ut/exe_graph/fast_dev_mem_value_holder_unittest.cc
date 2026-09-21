/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "checker/topo_checker.h"
#include "checker/bg_test.h"

namespace gert {
namespace bg {
class FastDevMemValueHolderUt : public BgTest {
};

TEST_F(FastDevMemValueHolderUt, SetGetLogicStreamOk) {
  int64_t logic_stream_id = 2;
  auto output0 = DevMemValueHolder::CreateSingleDataOutput("test", {}, logic_stream_id);
  output0->SetPlacement(1);
  EXPECT_EQ(output0->GetPlacement(), 1);
  EXPECT_EQ(output0->GetLogicStream(), 2);
}

TEST_F(FastDevMemValueHolderUt, DevMemCreateErrorOk) {
  auto holder = DevMemValueHolder::CreateError(0, "This is a test error information, int %d, %s", 10240, "Test msg");
  ASSERT_NE(holder, nullptr);
  EXPECT_FALSE(holder->IsOk());
}

TEST_F(FastDevMemValueHolderUt, DevMemCreateDataOutputOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<ValueHolderPtr> inputs = {data1, const1};
  auto holders = DevMemValueHolder::CreateDataOutput("TestNode", inputs, 3, 0);

  ASSERT_EQ(holders.size(), 3);
  ASSERT_TRUE(holders[0]->IsOk());
  ASSERT_TRUE(holders[1]->IsOk());
  ASSERT_TRUE(holders[2]->IsOk());
  EXPECT_EQ(holders[0]->GetType(), ValueHolder::ValueHolderType::kOutput);
  EXPECT_EQ(holders[1]->GetType(), ValueHolder::ValueHolderType::kOutput);
  EXPECT_EQ(holders[2]->GetType(), ValueHolder::ValueHolderType::kOutput);
}

TEST_F(FastDevMemValueHolderUt, DevMemCreateConstOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto holder = DevMemValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1), 0);
  ASSERT_NE(holder, nullptr);
  ASSERT_TRUE(holder->IsOk());
}

TEST_F(FastDevMemValueHolderUt, DevMemCreateMateFromNodeOk) {
  ge::ExecuteGraphPtr graph = std::make_shared<ge::ExecuteGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("FakeNode", "FakeNode");
  auto node = graph->AddNode(op_desc);
  auto holder = std::make_shared<DevMemValueHolder>(2);
  ASSERT_NE(holder, nullptr);
  auto value_holder = holder->CreateMateFromNode(node, 0, ValueHolder::ValueHolderType::kOutput);
  ASSERT_NE(value_holder, nullptr);
  ASSERT_TRUE(value_holder->IsOk());
  EXPECT_EQ(value_holder->GetType(), ValueHolder::ValueHolderType::kOutput);
  auto mem_holder = std::dynamic_pointer_cast<DevMemValueHolder>(value_holder);
  ASSERT_NE(mem_holder, nullptr);
  EXPECT_EQ(mem_holder->GetLogicStream(), 2);
}
}  // namespace bg
}  // namespace gert
