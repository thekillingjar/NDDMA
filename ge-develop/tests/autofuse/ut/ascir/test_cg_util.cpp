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
#include "ascir_ops.h"
#include "cg_utils.h"
namespace ascir {
namespace cg {
TEST(CgUtils, LoopGuardContextOk) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  int64_t count = 0;
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
  LOOP(a, b, c) {
    ++count;
    ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
    ASSERT_NE(CgContext::GetSharedThreadLocalContext(), nullptr);
  }
  ASSERT_EQ(count, 1);
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
}
TEST(CgUtils, OptionLoopGuardContextOk) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  int64_t count = 0;
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
  LOOP(Axes{a, b, c}, LoopOption{}) {
    ++count;
    ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
    ASSERT_EQ(CgContext::GetThreadLocalContext()->GetOption().pad_tensor_axes_to_loop, false);
    ASSERT_NE(CgContext::GetSharedThreadLocalContext(), nullptr);
  }
  ASSERT_EQ(count, 1);
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);

  LOOP(Axes{a, b, c}, LoopOption{.pad_tensor_axes_to_loop = true}) {
    ++count;
    ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
    ASSERT_EQ(CgContext::GetThreadLocalContext()->GetOption().pad_tensor_axes_to_loop, true);
    ASSERT_NE(CgContext::GetSharedThreadLocalContext(), nullptr);
  }
  ASSERT_EQ(count, 2);
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
}
TEST(CgUtils, NestedLoopGuardContextOk) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("D");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
  LOOP(a) {
    ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 1UL);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);

    LOOP(b, c) {
      ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 3UL);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].name, b.name);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].id, b.id);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].name, c.name);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].id, c.id);

      LOOP(d) {
        ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 4UL);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].name, b.name);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].id, b.id);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].name, c.name);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].id, c.id);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[3].name, d.name);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[3].id, d.id);
      }

      ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 3UL);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].name, b.name);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].id, b.id);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].name, c.name);
      EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].id, c.id);
    }

    ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 1UL);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);
  }
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
}
TEST(CgUtils, LoopGuardAxisOk) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  LOOP(a, b, c) {
    ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 3UL);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].name, b.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].id, b.id);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].name, c.name);
    EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].id, c.id);
  }
}
TEST(CgUtils, LoopGuard_SchedAxis_Ok) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  LOOP(a, b, c) {
    auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT, {a, b});
    auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT, {b, c});
    auto mm = MatMul("mm", data0.y, data1.y);
  }

  auto data0 = graph.Find("data0");
  auto data1 = graph.Find("data1");
  auto mm = graph.Find("mm");
  ASSERT_EQ(std::vector<AxisId>(data0.attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
  ASSERT_EQ(std::vector<AxisId>(data1.attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
  ASSERT_EQ(std::vector<AxisId>(mm.attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
}

TEST(PadTensorAxisToSched, NoContext_DoNotPad) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  ops::Data data("data");
  data.y.axis = {a.id};
  data.y.repeats = {A};
  data.y.strides = {1};

  ASSERT_TRUE(PadOutputViewToSched(data.y));
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats), std::vector<SizeExpr>({A}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides), std::vector<SizeExpr>({SizeExpr{1}}));
}
TEST(PadTensorAxisToSched, NotConfigPad_DoNotPad) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  ops::Data data("data");
  data.y.axis = {a.id};
  data.y.repeats = {A};
  data.y.strides = {1};

  LOOP(a, b, c) {
    ASSERT_TRUE(PadOutputViewToSched(data.y));
  }
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats), std::vector<SizeExpr>({A}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides), std::vector<SizeExpr>({SizeExpr{1}}));
}
TEST(PadTensorAxisToSched, NoNeedPad_Ok) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  ops::Data data("data");
  data.y.axis = {a.id, b.id, c.id};
  data.y.repeats = {A, B, C};
  data.y.strides = {C, 0, 1};

  OPTION_LOOP(Axes({a, b, c}), LoopOption{true}) {
    ASSERT_TRUE(PadOutputViewToSched(data.y));
  }
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id, b.id, c.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats), std::vector<SizeExpr>({A, B, C}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides),
            std::vector<SizeExpr>({C, 0, 1}));
}
TEST(PadTensorAxisToSched, PadHead) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.SetContiguousView({c, d});

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    ASSERT_TRUE(PadOutputViewToSched(data.y));
  }
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats),
            std::vector<SizeExpr>({1, 1, C, D}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides),
            std::vector<SizeExpr>({0, 0, D, 1}));
}
TEST(PadTensorAxisToSched, PadTail) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.SetContiguousView({a, b});

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    ASSERT_TRUE(PadOutputViewToSched(data.y));
  }
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats),
            std::vector<SizeExpr>({A, B, 1, 1}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides),
            std::vector<SizeExpr>({B, 1, 0, 0}));
}
TEST(PadTensorAxisToSched, PadTail_NotContiguous) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.axis = {a.id, b.id, c.id};
  data.y.repeats = {A, 1, C};
  data.y.strides = {C, 0, 1};

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    ASSERT_TRUE(PadOutputViewToSched(data.y));
  }
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats),
            std::vector<SizeExpr>({A, 1, C, 1}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides),
            std::vector<SizeExpr>({C, 0, 1, 0}));
}
TEST(PadTensorAxisToSched, PadMiddle) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.SetContiguousView({a, d});

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    ASSERT_TRUE(PadOutputViewToSched(data.y));
  }
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats),
            std::vector<SizeExpr>({A, 1, 1, D}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides),
            std::vector<SizeExpr>({D, 0, 0, 1}));
}
TEST(PadTensorAxisToSched, PadMultiple) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.SetContiguousView({b, d});

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    ASSERT_TRUE(PadOutputViewToSched(data.y));
  }
  EXPECT_EQ(static_cast<std::vector<AxisId>>(data.y.axis), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.repeats),
            std::vector<SizeExpr>({1, B, 1, D}));
  EXPECT_EQ(static_cast<std::vector<SizeExpr>>(data.y.strides),
            std::vector<SizeExpr>({0, D, 0, 1}));
}
TEST(PadTensorAxisToSched, SameAxisNumButNotMatch_Failed) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.SetContiguousView({b, a, c, d});

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    ASSERT_FALSE(PadOutputViewToSched(data.y));
  }
}
TEST(PadTensorAxisToSched, DiffAxisNumAndNotMatch1_Failed) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.SetContiguousView({a, b, c, d});

  OPTION_LOOP(Axes({a, b, c}), LoopOption{true}) {
    ASSERT_FALSE(PadOutputViewToSched(data.y));
  }
}
TEST(PadTensorAxisToSched, DiffAxisNumAndNotMatch2_Failed) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ops::Data data("data");
  data.y.SetContiguousView({a, c, b});

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    ASSERT_FALSE(PadOutputViewToSched(data.y));
  }
}
TEST(AutoPadAxis, Ok) {
  Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  OPTION_LOOP(Axes({a, b, c, d}), LoopOption{true}) {
    auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
    auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, c, d});
    auto mm = MatMul("mm", data0.y, data1.y);
    mm.y.SetContiguousView({a, b, c});
    PadOutputViewToSched(mm.y);
  }

  auto d0 = graph.Find("data0");
  EXPECT_EQ(d0.outputs[0].axis(), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(d0.outputs[0].repeats(), std::vector<SizeExpr>({A, B, 1, D}));
  EXPECT_EQ(d0.outputs[0].strides(), std::vector<SizeExpr>({B * D, D, 0, 1}));

  auto d1 = graph.Find("data1");
  EXPECT_EQ(d1.outputs[0].axis(), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(d1.outputs[0].repeats(), std::vector<SizeExpr>({A, 1, C, D}));
  EXPECT_EQ(d1.outputs[0].strides(), std::vector<SizeExpr>({C * D, 0, D, 1}));

  auto mm = graph.Find("mm");
  EXPECT_EQ(mm.outputs[0].axis(), std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(mm.outputs[0].repeats(), std::vector<SizeExpr>({A, B, C, 1}));
  EXPECT_EQ(mm.outputs[0].strides(), std::vector<SizeExpr>({B * C, C, 1, 0}));
}
}  // namespace cg
}  // namespace ascir
