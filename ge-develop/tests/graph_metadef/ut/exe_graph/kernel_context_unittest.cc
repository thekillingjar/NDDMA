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
#include "exe_graph/runtime/kernel_context.h"
#include "faker/kernel_run_context_faker.h"

namespace gert {
class KernelContextUT : public testing::Test {};
namespace {
struct TestT {
  int64_t a;
  int32_t b;
};
}  // namespace
TEST_F(KernelContextUT, ChainGetInnerOk) {
  Chain c;
  EXPECT_EQ(c.GetPointer<uint64_t>(), reinterpret_cast<uint64_t *>(&c));
  EXPECT_EQ(c.GetPointer<uint8_t>(), reinterpret_cast<uint8_t *>(&c));

  const Chain &cc = c;
  EXPECT_EQ(cc.GetPointer<uint64_t>(), reinterpret_cast<uint64_t *>(&c));
  EXPECT_EQ(cc.GetPointer<uint8_t>(), reinterpret_cast<uint8_t *>(&c));
}

TEST_F(KernelContextUT, ChainGetAllocOk) {
  AsyncAnyValue av = {nullptr, nullptr};
  Chain *c = reinterpret_cast<Chain *>(&av);
  const Chain *cc = reinterpret_cast<Chain *>(&av);

  EXPECT_EQ(c->GetPointer<TestT>(), reinterpret_cast<TestT *>(av.data.pointer));
  EXPECT_EQ(c->GetPointer<TestT>(), reinterpret_cast<TestT *>(av.data.pointer));

  EXPECT_EQ(cc->GetPointer<TestT>(), reinterpret_cast<TestT *>(av.data.pointer));
  EXPECT_EQ(cc->GetPointer<TestT>(), reinterpret_cast<TestT *>(av.data.pointer));
}
TEST_F(KernelContextUT, ChainGetInnerValueOk) {
  AsyncAnyValue av = {nullptr, nullptr};
  av.data.pointer = reinterpret_cast<void *>(10);

  Chain *c = reinterpret_cast<Chain *>(&av);
  const Chain *cc = reinterpret_cast<Chain *>(&av);

  EXPECT_EQ(c->GetValue<int64_t>(), 10);
  EXPECT_EQ(cc->GetValue<int64_t>(), 10);

  c->GetValue<int64_t>() = 20;
  EXPECT_EQ(reinterpret_cast<int64_t>(av.data.pointer), 20);
}

TEST_F(KernelContextUT, ChainSetDeleterOk) {
  AsyncAnyValue av = {nullptr, nullptr};
  Chain *c = reinterpret_cast<Chain *>(&av);

  c->SetWithDefaultDeleter(new TestT());
  ASSERT_NE(av.data.pointer, nullptr);
  ASSERT_NE(av.deleter, nullptr);
  av.deleter(av.data.pointer);
  av.deleter = nullptr;

  c->SetWithDefaultDeleter(new std::vector<int64_t>());
  ASSERT_NE(av.data.pointer, nullptr);
  ASSERT_NE(av.deleter, nullptr);
  av.deleter(av.data.pointer);
}

TEST_F(KernelContextUT, ChainSetAndUseStructOk) {
  AsyncAnyValue av = {nullptr, nullptr};
  Chain *c = reinterpret_cast<Chain *>(&av);

  c->SetWithDefaultDeleter(new std::vector<int64_t>());

  auto vec = c->GetPointer<std::vector<int64_t>>();
  vec->push_back(10);
  vec->push_back(20);
  EXPECT_EQ(c->GetPointer<std::vector<int64_t>>()->size(), 2);
  EXPECT_EQ(*c->GetPointer<std::vector<int64_t>>(), std::vector<int64_t>({10, 20}));

  av.deleter(av.data.pointer);
}

TEST_F(KernelContextUT, GetInputNumOk) {
  auto context_holder = KernelRunContextFaker().KernelIONum(5, 6).Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(context->GetInputNum(), 5);
  EXPECT_EQ(context->GetOutputNum(), 6);
}

TEST_F(KernelContextUT, GetInnerInputOk) {
  auto context_holder = KernelRunContextFaker().KernelIONum(5, 6).Build();
  auto context = context_holder.GetContext<KernelContext>();

  EXPECT_EQ(context->GetInputPointer<int64_t>(0), context_holder.holder.value_holder_[0].GetPointer<int64_t>());
  EXPECT_EQ(context->GetInputPointer<int64_t>(1), context_holder.holder.value_holder_[1].GetPointer<int64_t>());
  EXPECT_EQ(context->GetInputPointer<int32_t>(2), context_holder.holder.value_holder_[2].GetPointer<int32_t>());

  EXPECT_EQ(context->GetInputValue<int64_t>(4), context_holder.holder.value_holder_[4].GetValue<int64_t>());

  EXPECT_EQ(context->GetInput(0), &context_holder.holder.value_holder_[0]);
  EXPECT_EQ(context->GetInput(4), &context_holder.holder.value_holder_[4]);
}

TEST_F(KernelContextUT, GetAllocInputOk) {
  auto context_holder = KernelRunContextFaker().KernelIONum(5, 6).Build();
  std::vector<TestT> t_holder(11);
  for (size_t i = 0; i < 11; ++i) {
    context_holder.holder.value_holder_[i].any_value_.data.pointer = &t_holder[i];
  }
  auto context = context_holder.GetContext<KernelContext>();

  EXPECT_EQ(context->GetInputPointer<TestT>(0),
            reinterpret_cast<const TestT *>((context_holder.holder.value_holder_[0].any_value_.data.pointer)));
  EXPECT_EQ(context->GetInputPointer<TestT>(1),
            reinterpret_cast<const TestT *>((context_holder.holder.value_holder_[1].any_value_.data.pointer)));
  EXPECT_EQ(context->GetInputPointer<TestT>(4),
            reinterpret_cast<const TestT *>((context_holder.holder.value_holder_[4].any_value_.data.pointer)));

  EXPECT_EQ(context->GetInput(0), &context_holder.holder.value_holder_[0]);
  EXPECT_EQ(context->GetInput(4), &context_holder.holder.value_holder_[4]);
}

TEST_F(KernelContextUT, GetInnerOutputOk) {
  auto context_holder = KernelRunContextFaker().KernelIONum(5, 6).Build();
  auto context = context_holder.GetContext<KernelContext>();

  EXPECT_EQ(context->GetOutputPointer<int64_t>(0),
            reinterpret_cast<const int64_t *>(&(context_holder.holder.value_holder_[5].any_value_.data)));
  EXPECT_EQ(context->GetOutputPointer<int64_t>(1),
            reinterpret_cast<const int64_t *>(&(context_holder.holder.value_holder_[6].any_value_.data)));
  EXPECT_EQ(context->GetOutputPointer<int32_t>(2),
            reinterpret_cast<const int32_t *>(&(context_holder.holder.value_holder_[7].any_value_.data)));

  EXPECT_EQ(context->GetOutput(0), &context_holder.holder.value_holder_[5]);
  EXPECT_EQ(context->GetOutput(5), &context_holder.holder.value_holder_[10]);
}

TEST_F(KernelContextUT, GetAllocOutputOk) {
  auto context_holder = KernelRunContextFaker().KernelIONum(5, 6).Build();
  std::vector<TestT> t_holder(11);
  for (size_t i = 0; i < 11; ++i) {
    context_holder.holder.value_holder_[i].any_value_.data.pointer = &t_holder[i];
  }
  auto context = context_holder.GetContext<KernelContext>();

  EXPECT_EQ(context->GetOutputPointer<TestT>(0),
            reinterpret_cast<const TestT *>((context_holder.holder.value_holder_[5].any_value_.data.pointer)));
  EXPECT_EQ(context->GetOutputPointer<TestT>(1),
            reinterpret_cast<const TestT *>((context_holder.holder.value_holder_[6].any_value_.data.pointer)));
  EXPECT_EQ(context->GetOutputPointer<TestT>(4),
            reinterpret_cast<const TestT *>((context_holder.holder.value_holder_[9].any_value_.data.pointer)));

  EXPECT_EQ(context->GetOutput(0), &context_holder.holder.value_holder_[5]);
  EXPECT_EQ(context->GetOutput(4), &context_holder.holder.value_holder_[9]);
}
}  // namespace gert
