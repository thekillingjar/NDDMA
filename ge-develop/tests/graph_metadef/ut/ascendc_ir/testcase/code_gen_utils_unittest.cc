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
#include "ascir_ops.h"
#include "graph/utils/cg_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "expression/const_values.h"

#define EXPECT_VIEW_PTR_EQ(tensor0, tensor1) \
          EXPECT_EQ(*tensor0.axis, *tensor1.axis);\
          EXPECT_EQ(*tensor0.strides, *tensor1.strides);\
          EXPECT_EQ(*tensor0.repeats, *tensor1.repeats);

#define EXPECT_VIEW_EQ(tensor0, tensor1) \
          EXPECT_EQ(tensor0.axis, tensor1.axis); \
          EXPECT_EQ(tensor0.strides, tensor1.strides); \
          EXPECT_EQ(tensor0.repeats, tensor1.repeats);

#define EXPECT_VIEW_AND_DTYPE_EQ(tensor0, tensor1) \
          EXPECT_VIEW_EQ(tensor0, tensor1) \
          EXPECT_EQ(tensor0.dtype, tensor1.dtype)

namespace ge {
namespace ascir {
namespace cg {
using Graph = ge::AscGraph;
using ge::Expression;
using ge::Symbol;
Graph ConstructTestGraph(const std::string &graph_name) {
  Graph graph(graph_name.c_str());
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          // data0(GM1)-------------------------------|
          //            |->load0(TQue1)-->mm(TQue3)--------->y(TQue4)
          // data1(GM2)--->load1(TQue2)-|                |
          //             |_______________________________|
          // data2(TBuf1)________________________________|
          auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
          auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, c, d});
          AscendString name;
          data1.GetOwnerOp().GetName(name);
          EXPECT_EQ("data1", std::string(name.GetString()));
          auto load0 = LoadStub("load0", data0).TQue(Position::kPositionVecIn, 1, 2);
          auto load1 = LoadStub("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
          auto mm = MatMul("mm", load0, load1).TQue(Position::kPositionVecOut, 1, 1);
          auto data2 = ContiguousData("data2", graph, ge::DT_FLOAT, {a, c, d}).TBuf(Position::kPositionVecOut);
          auto y = CalcY("y", data0, data2, data1, mm).TQue(Position::kPositionVecOut, 1, 1);
          EXPECT_EQ(y.dtype, ge::DT_FLOAT);
        }
      }
    }
  }
  return graph;
}

TEST(CgUtils, SetGetContextOk) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto ctx = CgContext::GetSharedThreadLocalContext();
  ASSERT_EQ(ctx, nullptr);
  auto ctx_obj = std::make_shared<CgContext>();
  CgContext::SetThreadLocalContext(ctx_obj);
  ctx = CgContext::GetSharedThreadLocalContext();
  ASSERT_NE(ctx, nullptr);
  ctx->SetLoopAxes({a, b, c});
  ASSERT_EQ(ctx->GetLoopAxes().size(), 3);
  ctx->SetBlockLoopEnd(a.id);
  ASSERT_EQ(ctx->GetBlockLoopEnd(), a.id);
  ctx->SetVectorizedLoopEnd(c.id);
  ASSERT_EQ(ctx->GetVectorizedLoopEnd(), c.id);
  ctx->SetLoopEnd(c.id);
  ASSERT_EQ(ctx->GetLoopEnd(), c.id);
}

TEST(CgUtils, LoopGuardContextOk) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  int64_t count = 0;
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        ++count;
        ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
        ASSERT_NE(CgContext::GetSharedThreadLocalContext(), nullptr);
      }
    }
  }
  ASSERT_EQ(count, 1);
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
}
TEST(CgUtils, OptionLoopGuardContextOk) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  int64_t count = 0;
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
  LOOP(a) {
    LOOP(b) {
      OPTION_LOOP(c, LoopOption{}) {
        ++count;
        ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
        ASSERT_EQ(CgContext::GetThreadLocalContext()->GetOption().pad_tensor_axes_to_loop, false);
        ASSERT_NE(CgContext::GetSharedThreadLocalContext(), nullptr);
      }
    }
  }
  ASSERT_EQ(count, 1);
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);

  OPTION_LOOP(a, LoopOption{.pad_tensor_axes_to_loop = true}) {
    ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
    ASSERT_EQ(CgContext::GetThreadLocalContext()->GetOption().pad_tensor_axes_to_loop, true);
    ASSERT_NE(CgContext::GetSharedThreadLocalContext(), nullptr);
    LOOP(b) {
      LOOP(c) {
        ++count;
        ASSERT_NE(CgContext::GetThreadLocalContext(), nullptr);
        ASSERT_EQ(CgContext::GetThreadLocalContext()->GetOption().pad_tensor_axes_to_loop, false);
        ASSERT_NE(CgContext::GetSharedThreadLocalContext(), nullptr);
      }
    }
  }
  ASSERT_EQ(count, 2);
  ASSERT_EQ(CgContext::GetThreadLocalContext(), nullptr);
  ASSERT_EQ(CgContext::GetSharedThreadLocalContext(), nullptr);
}
TEST(CgUtils, NestedLoopGuardContextOk) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("D");
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

    LOOP(b) {
      LOOP(c) {
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
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        ASSERT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes().size(), 3UL);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].name, a.name);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[0].id, a.id);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].name, b.name);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[1].id, b.id);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].name, c.name);
        EXPECT_EQ(CgContext::GetThreadLocalContext()->GetLoopAxes()[2].id, c.id);
      }
    }
  }
}
TEST(CgUtils, LoopGuard_SchedAxis_Ok) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT, {a, b});
        auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT, {b, c});
        auto mm = MatMul("mm", data0, data1);
        (void) mm; // -Werror=unused-but-set-variable
      }
    }
  }

  auto data0 = graph.FindNode("data0");
  auto data1 = graph.FindNode("data1");
  auto mm = graph.FindNode("mm");
  ASSERT_EQ(std::vector<AxisId>(data0->attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
  ASSERT_EQ(std::vector<AxisId>(data1->attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
  ASSERT_EQ(std::vector<AxisId>(mm->attr.sched.axis), std::vector<AxisId>({a.id, b.id, c.id}));
}

TEST(PadTensorAxisToSched, NoContext_DoNotPad) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  ge::ascir_op::Data data("data", graph);
  *data.y.axis = {a.id};
  *data.y.repeats = {A};
  *data.y.strides = {sym::kSymbolOne};

  ASSERT_TRUE(PadOutputViewToSched(data.y));
  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == A);
  EXPECT_TRUE((*data.y.strides)[0] == 1);
}

TEST(PadTensorAxisToSched, NotConfigPad_DoNotPad) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  ge::ascir_op::Data data("data", graph);
  *data.y.axis = {a.id};
  *data.y.repeats = {A};
  *data.y.strides = {sym::kSymbolOne};

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        ASSERT_TRUE(PadOutputViewToSched(data.y));
      }
    }
  }
  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == A);
  EXPECT_TRUE((*data.y.strides)[0] == 1);
}

TEST(PadTensorAxisToSched, NoNeedPad_Ok) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  ge::ascir_op::Data data("data", graph);
  *data.y.axis = {a.id, b.id, c.id};
  *data.y.repeats = {A, B, C};
  *data.y.strides = {C, sym::kSymbolZero, sym::kSymbolOne};
  LOOP(a) {
    LOOP(b) {
      OPTION_LOOP(c, LoopOption{true}) {
        ASSERT_TRUE(PadOutputViewToSched(data.y));
      }
    }
  }

  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id, b.id, c.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == A);
  EXPECT_TRUE((*data.y.repeats)[0] == A);
  EXPECT_TRUE((*data.y.repeats)[1] == B);
  EXPECT_TRUE((*data.y.repeats)[2] == C);
  EXPECT_TRUE((*data.y.strides)[0] == C);
  EXPECT_TRUE((*data.y.strides)[1] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[2] == sym::kSymbolOne);
}

TEST(PadTensorAxisToSched, PadHead) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  data.y.SetContiguousView({c, d});
  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          ASSERT_TRUE(PadOutputViewToSched(data.y));
        }
      }
    }
  }
  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[1] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[2] == C);
  EXPECT_TRUE((*data.y.repeats)[3] == D);
  std::cout << "strides 0:" << (*data.y.strides)[0] << std::endl;
  std::cout << "strides 1:" << (*data.y.strides)[1] << std::endl;
  EXPECT_TRUE((*data.y.strides)[0] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[1] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[2] == D);
  EXPECT_TRUE((*data.y.strides)[3] == sym::kSymbolOne);
}
TEST(PadTensorAxisToSched, PadTail) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  data.y.SetContiguousView({a, b});
  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          ASSERT_TRUE(PadOutputViewToSched(data.y));
        }
      }
    }
  }

  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == A);
  EXPECT_TRUE((*data.y.repeats)[1] == B);
  EXPECT_TRUE((*data.y.repeats)[2] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[3] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.strides)[0] == B);
  EXPECT_TRUE((*data.y.strides)[1] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.strides)[2] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[3] == sym::kSymbolZero);
}

TEST(PadTensorAxisToSched, PadTail_NotContiguous) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  *data.y.axis = {a.id, b.id, c.id};
  *data.y.repeats = {A, sym::kSymbolOne, C};
  *data.y.strides = {C, sym::kSymbolZero, sym::kSymbolOne};

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          ASSERT_TRUE(PadOutputViewToSched(data.y));
        }
      }
    }
  }
  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == A);
  EXPECT_TRUE((*data.y.repeats)[1] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[2] == C);
  EXPECT_TRUE((*data.y.repeats)[3] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.strides)[0] == C);
  EXPECT_TRUE((*data.y.strides)[1] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[2] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.strides)[3] == sym::kSymbolZero);
}
TEST(PadTensorAxisToSched, PadMiddle) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  data.y.SetContiguousView({a, d});

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          ASSERT_TRUE(PadOutputViewToSched(data.y));
        }
      }
    }
  }
  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == A);
  EXPECT_TRUE((*data.y.repeats)[1] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[2] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[3] == D);
  EXPECT_TRUE((*data.y.strides)[0] == D);
  EXPECT_TRUE((*data.y.strides)[1] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[2] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[3] == sym::kSymbolOne);
}
TEST(PadTensorAxisToSched, PadMultiple) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  data.y.SetContiguousView({b, d});

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          ASSERT_TRUE(PadOutputViewToSched(data.y));
        }
      }
    }
  }
  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(*data.y.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE((*data.y.repeats)[0] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[1] == B);
  EXPECT_TRUE((*data.y.repeats)[2] == sym::kSymbolOne);
  EXPECT_TRUE((*data.y.repeats)[3] == D);
  EXPECT_TRUE((*data.y.strides)[0] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[1] == D);
  EXPECT_TRUE((*data.y.strides)[2] == sym::kSymbolZero);
  EXPECT_TRUE((*data.y.strides)[3] == sym::kSymbolOne);
}

TEST(PadTensorAxisToSched, SameAxisNumButNotMatch_Failed) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  data.y.SetContiguousView({b, a, c, d});

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          ASSERT_FALSE(PadOutputViewToSched(data.y));
        }
      }
    }
  }
}
TEST(PadTensorAxisToSched, DiffAxisNumAndNotMatch1_Failed) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  data.y.SetContiguousView({a, b, c, d});

  LOOP(a) {
    LOOP(b) {
      OPTION_LOOP(c, LoopOption{true}) {
        ASSERT_FALSE(PadOutputViewToSched(data.y));
      }
    }
  }
}
TEST(PadTensorAxisToSched, DiffAxisNumAndNotMatch2_Failed) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  ge::ascir_op::Data data("data", graph);
  data.y.SetContiguousView({a, c, b});

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          ASSERT_FALSE(PadOutputViewToSched(data.y));
        }
      }
    }
  }
}
TEST(AutoPadAxis, Ok) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
          auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, c, d});
          AscendString name;
          data1.GetOwnerOp().GetName(name);
          EXPECT_EQ("data1", std::string(name.GetString()));
          auto load0 = LoadStub("load0", data0);
          auto load1 = LoadStub("load1", data0);
          auto mm = MatMul("mm", load0, load1);
          mm.SetContiguousView({a, b, c});
          PadOutputViewToSched(mm);
          auto data2 = ContiguousData("data1", graph, ge::DT_FLOAT, {a, c, d});
          auto y = CalcY("y", data0, data2, data1, mm);
          EXPECT_EQ(y.dtype, ge::DT_FLOAT);
        }
      }
    }
  }

  auto d0 = graph.FindNode("data0");
  EXPECT_EQ(d0->outputs[0].attr.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE(d0->outputs[0].attr.repeats[0] == A);
  EXPECT_TRUE(d0->outputs[0].attr.repeats[1] == B);
  EXPECT_TRUE(d0->outputs[0].attr.repeats[2] == sym::kSymbolOne);
  EXPECT_TRUE(d0->outputs[0].attr.repeats[3] == D);
  EXPECT_TRUE(d0->outputs[0].attr.strides[0] == (B*D));
  EXPECT_TRUE(d0->outputs[0].attr.strides[1] == D);
  EXPECT_TRUE(d0->outputs[0].attr.strides[2] == sym::kSymbolZero);
  EXPECT_TRUE(d0->outputs[0].attr.strides[3] == sym::kSymbolOne);

  auto d1 = graph.FindNode("data1");

  EXPECT_EQ(d0->outputs[0].attr.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE(d1->outputs[0].attr.repeats[0] == A);
  EXPECT_TRUE(d1->outputs[0].attr.repeats[1] == sym::kSymbolOne);
  EXPECT_TRUE(d1->outputs[0].attr.repeats[2] == C);
  EXPECT_TRUE(d1->outputs[0].attr.repeats[3] == D);
  EXPECT_TRUE(d1->outputs[0].attr.strides[0] == (C*D));
  EXPECT_TRUE(d1->outputs[0].attr.strides[1] == sym::kSymbolZero);
  EXPECT_TRUE(d1->outputs[0].attr.strides[2] == D);
  EXPECT_TRUE(d1->outputs[0].attr.strides[3] == sym::kSymbolOne);

  auto mm = graph.FindNode("mm");
  EXPECT_EQ(mm->outputs[0].attr.axis, std::vector<AxisId>({a.id, b.id, c.id, d.id}));
  EXPECT_TRUE(mm->outputs[0].attr.repeats[0] == A);
  EXPECT_TRUE(mm->outputs[0].attr.repeats[1] == B);
  EXPECT_TRUE(mm->outputs[0].attr.repeats[2] == C);
  EXPECT_TRUE(mm->outputs[0].attr.repeats[3] == sym::kSymbolOne);
  EXPECT_TRUE(mm->outputs[0].attr.strides[0] == (B*C));
  EXPECT_TRUE(mm->outputs[0].attr.strides[1] == C);
  EXPECT_TRUE(mm->outputs[0].attr.strides[2] == sym::kSymbolOne);
  EXPECT_TRUE(mm->outputs[0].attr.strides[3] == sym::kSymbolZero);
}


TEST(CgApi, VectorizedTensor_move_assign) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("D");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  LOOP(a) {
    AscOpOutput v({b.id, c.id, d.id});
    LOOP(b) {
      LOOP(c) {
        auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, c, d});
        auto load0 = LoadStub("load0", data0);
        v.AutoOffset() = LoadStub("load1", data0);
        // make sure compile error
//        v = LoadStub("load1", data0); // 异常场景
        auto abs0 = AbsStub("abs0", load0);
        auto abs1 = AbsStub("abs1", static_cast<AscOpOutput>(v));
        (void) abs0;
        (void) abs1;
      }
    }
  }
  EXPECT_EQ(graph.FindNode("load1")->outputs[0U].attr.vectorized_axis, std::vector<int64_t>({b.id, c.id, d.id}));
  // dtype推导api接口还没切换到新的dtype注册机制，暂时不校验dtype
  EXPECT_VIEW_EQ(graph.FindNode("load1")->outputs[0U].attr, graph.FindNode("data0")->outputs[0U].attr);
  EXPECT_VIEW_EQ(graph.FindNode("abs1")->inputs[0U].attr, graph.FindNode("load1")->outputs[0U].attr);
  EXPECT_VIEW_EQ(graph.FindNode("abs1")->outputs[0U].attr, graph.FindNode("load1")->outputs[0U].attr);
  EXPECT_VIEW_EQ(graph.FindNode("load0")->outputs[0U].attr, graph.FindNode("data0")->outputs[0U].attr);
}

TEST(CgApi, ViewInfer_ok) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("D");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, c, d});
        auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, b, c, d});
        auto data2 = ContiguousData("data2", graph, ge::DT_FLOAT16, {d});
        auto load0 = LoadStub("load0", data0);
        auto load1 = LoadStub("load1", data1);
        auto load2 = LoadStub("load2", data2);
        auto[out0, out1, out2, out3] = CalcMeanStub("CalcMeanStub", load0, load1, load2, d.id);
        // out0 is reduced axis_d
        EXPECT_EQ(*out0.axis, *load0.axis);
        EXPECT_EQ(*out0.repeats, *load0.repeats);
        EXPECT_NE(*out0.strides, *load0.strides);
        std::vector<ge::Expression> strides_expect;
        strides_expect.emplace_back(B * C * D / D);
        strides_expect.emplace_back(C * D / D);
        strides_expect.emplace_back(D / D);
        strides_expect.emplace_back(sym::kSymbolZero);
        EXPECT_EQ(out0.strides->size(), strides_expect.size());
        size_t index = 0U;
        for (const auto &se : strides_expect) {
          EXPECT_EQ((*out0.strides)[index++], se);
        }
        EXPECT_VIEW_PTR_EQ(out1, load0);
        EXPECT_VIEW_PTR_EQ(out2, load0);
        EXPECT_VIEW_PTR_EQ(out3, load0);
      }
    }
  }
}

TEST(CgApi, VectorizedAxisInfer_ok) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("D");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  LOOP(a) {
    AscOpOutput v({b.id, c.id, d.id});
    LOOP(b) {
      LOOP(c) {
        auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, c, d});
        auto load0 = LoadStub("load0", data0);
        v.AutoOffset() = LoadStub("load1", data0);
        auto abs0 = AbsStub("abs0", load0);
        auto abs1 = AbsStub("abs1", static_cast<AscOpOutput>(v));
        (void) abs0;
        (void) abs1;
      }
    }
  }
  EXPECT_EQ(graph.FindNode("load1")->outputs[0U].attr.vectorized_axis, std::vector<int64_t>({b.id, c.id, d.id}));
  EXPECT_EQ(graph.FindNode("abs1")->inputs[0U].attr.vectorized_axis, graph.FindNode("load1")->outputs[0U].attr.vectorized_axis);
  EXPECT_EQ(graph.FindNode("abs1")->outputs[0U].attr.vectorized_axis, std::vector<int64_t>({d.id}));
  EXPECT_EQ(graph.FindNode("load0")->attr.sched.loop_axis, c.id);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0U].attr.axis, std::vector<int64_t>({a.id, b.id, c.id, d.id}));
  EXPECT_EQ(graph.FindNode("load0")->outputs[0U].attr.vectorized_axis, std::vector<int64_t>({d.id}));
}

TEST(SetDataNodeAttr, Ok) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  std::vector<std::vector<int64_t>> vec;
  vec.emplace_back(1);
  vec.emplace_back(2);
  vec.emplace_back(3);
  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d}, 0);
          auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, c, d}, 1);
          AscendString name;
          data0.GetOwnerOp().GetName(name);
          EXPECT_EQ("data0", std::string(name.GetString()));
          data1.GetOwnerOp().GetName(name);
          EXPECT_EQ("data1", std::string(name.GetString()));
        }
      }
    }
  }

  auto d0 = graph.FindNode("data0");
  ge::GeAttrValue attr_value;
  int64_t index_value = -1;
  auto d1 = graph.FindNode("data1");
  index_value = -1;
  (void) d1->GetOpDesc()->GetAttr("index", attr_value);
  attr_value.GetValue<int64_t>(index_value);
  EXPECT_TRUE(index_value == -1);
}

TEST(TBufTQue, CreatTQueFailed){
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto d = graph.CreateAxis("d", D);
  auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
  EXPECT_EQ(LoadStub("load0", data0).TQue(Position::kPositionVecIn, -1, 1).que->id, kIdNone);
  EXPECT_EQ(LoadStub("load0", data0).TQue(Position::kPositionVecIn, 0, 1).que->id, kIdNone);
  EXPECT_EQ(LoadStub("load0", data0).TQue(Position::kPositionVecIn, 1, -1).que->id, kIdNone);
  EXPECT_EQ(LoadStub("load0", data0).TQue(Position::kPositionVecIn, 1, 0).que->id, kIdNone);
}

TEST(TBufTQue, CreateOk) {
  Graph graph = ConstructTestGraph("test_graph1");
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeGlobal);
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.mem.position, Position::kPositionGM);

  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeGlobal);
  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.mem.position, Position::kPositionGM);

  EXPECT_EQ(graph.FindNode("data2")->outputs[0].attr.que.id, kIdNone);
  EXPECT_NE(graph.FindNode("data2")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data2")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeBuffer);
  EXPECT_EQ(graph.FindNode("data2")->outputs[0].attr.mem.position, Position::kPositionVecOut);

  EXPECT_NE(graph.FindNode("load0")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.que.depth, 1);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.que.buf_num, 2);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeQueue);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.mem.position, Position::kPositionVecIn);

  EXPECT_NE(graph.FindNode("load1")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.que.depth, 1);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.que.buf_num, 2);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeQueue);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.mem.position, Position::kPositionVecIn);

  EXPECT_NE(graph.FindNode("mm")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.que.depth, 1);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.que.buf_num, 1);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeQueue);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.mem.position, Position::kPositionVecOut);
  Graph graph2 = ConstructTestGraph("test_graph2");
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.mem.tensor_id, 0);
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.mem.tensor_id, graph2.FindNode("data0")->outputs[0].attr.mem.tensor_id);
  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.mem.tensor_id, graph2.FindNode("data1")->outputs[0].attr.mem.tensor_id);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.mem.tensor_id, graph2.FindNode("load0")->outputs[0].attr.mem.tensor_id);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.mem.tensor_id, graph2.FindNode("load1")->outputs[0].attr.mem.tensor_id);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.mem.tensor_id, graph2.FindNode("mm")->outputs[0].attr.mem.tensor_id);
  EXPECT_EQ(graph.FindNode("data2")->outputs[0].attr.mem.tensor_id, graph2.FindNode("data2")->outputs[0].attr.mem.tensor_id);
}

TEST(TBufTQue, RepeatBindingFailed){
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto d = graph.CreateAxis("d", D);
  auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
  auto test1 = LoadStub("load0", data0).TQue(Position::kPositionVecIn, 1, 2);
  EXPECT_EQ(test1.mem->position, Position::kPositionVecIn);
  EXPECT_EQ(test1.TBuf(Position::kPositionVecOut).mem->position, Position::kPositionVecIn);

  auto test2 = LoadStub("load0", data0).TBuf(Position::kPositionVecIn);
  EXPECT_EQ(test2.TQue(Position::kPositionVecIn, 1, 1).mem->position, Position::kPositionVecIn);
  auto test3 = LoadStub("load0", data0).TBuf(Position::kPositionVecIn);
  EXPECT_EQ(test3.TBuf(Position::kPositionVecIn).mem->position, Position::kPositionVecIn);
  auto test4 = LoadStub("load0", data0).TQue(Position::kPositionVecIn, 1, 2);
  EXPECT_EQ(test4.TQue(Position::kPositionVecIn, 1, 2).mem->position, Position::kPositionVecIn);
}

TEST(ScopeUse, Ok) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  LOOP(a) {
    LOOP(b) {
      LOOP(c) {
        OPTION_LOOP(d, LoopOption{true}) {
          // data0(GM1)-------------------------------|
          //            |->load0(TQue1)-->mm(TQue3)--------->y(ScopeUse(data2))
          // data1(GM2)--->load1(TQue2)-|                |
          //             |_______________________________|
          // data2(TBuf1)________________________________|
          auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
          auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, c, d});
          AscendString name;
          data1.GetOwnerOp().GetName(name);
          EXPECT_EQ("data1", std::string(name.GetString()));
          auto load0 = LoadStub("load0", data0).TQue(Position::kPositionVecIn, 1, 2);
          auto load1 = LoadStub("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
          auto mm = MatMul("mm", load0, load1).TQue(Position::kPositionVecOut, 1, 1);
          auto data2 = ContiguousData("data2", graph, ge::DT_FLOAT, {a, c, d}).TBuf(Position::kPositionVecOut);
          auto [rstd0, rstd1] = CalcRstd("rstd", data2, data1, mm);
          EXPECT_EQ(rstd0.dtype, ge::DT_FLOAT);
          EXPECT_EQ(rstd1.dtype, ge::DT_FLOAT);
          rstd0.Use(load1);
          rstd1.Use(mm);
        }
      }
    }
  }
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeGlobal);
  EXPECT_EQ(graph.FindNode("data0")->outputs[0].attr.mem.position, Position::kPositionGM);

  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeGlobal);
  EXPECT_EQ(graph.FindNode("data1")->outputs[0].attr.mem.position, Position::kPositionGM);

  EXPECT_EQ(graph.FindNode("data2")->outputs[0].attr.que.id, kIdNone);
  EXPECT_NE(graph.FindNode("data2")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("data2")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeBuffer);
  EXPECT_EQ(graph.FindNode("data2")->outputs[0].attr.mem.position, Position::kPositionVecOut);

  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.que.id, 0);
  EXPECT_NE(graph.FindNode("load0")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.que.depth, 1);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.que.buf_num, 2);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeQueue);
  EXPECT_EQ(graph.FindNode("load0")->outputs[0].attr.mem.position, Position::kPositionVecIn);

  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.que.id, 1);
  EXPECT_NE(graph.FindNode("load1")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.que.depth, 1);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.que.buf_num, 2);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeQueue);
  EXPECT_EQ(graph.FindNode("load1")->outputs[0].attr.mem.position, Position::kPositionVecIn);

  EXPECT_NE(graph.FindNode("mm")->outputs[0].attr.que.id, kIdNone);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.buf.id, kIdNone);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.que.depth, 1);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.que.buf_num, 1);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.mem.alloc_type, AllocType::kAllocTypeQueue);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.mem.position, Position::kPositionVecOut);

  EXPECT_EQ(graph.FindNode("rstd")->outputs[0].attr.que.id, graph.FindNode("load1")->outputs[0].attr.que.id);
  EXPECT_NE(graph.FindNode("rstd")->outputs[0].attr.buf.id, graph.FindNode("load1")->outputs[0].attr.que.id);
  EXPECT_EQ(graph.FindNode("rstd")->outputs[0].attr.mem.alloc_type, graph.FindNode("load1")->outputs[0].attr.mem.alloc_type);
  EXPECT_EQ(graph.FindNode("rstd")->outputs[0].attr.mem.position, graph.FindNode("load1")->outputs[0].attr.mem.position);

  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.que.id, graph.FindNode("rstd")->outputs[1].attr.que.id);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.buf.id, graph.FindNode("rstd")->outputs[1].attr.buf.id);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.que.depth, graph.FindNode("rstd")->outputs[1].attr.que.depth);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.que.buf_num, graph.FindNode("rstd")->outputs[1].attr.que.buf_num);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.mem.alloc_type, graph.FindNode("rstd")->outputs[1].attr.mem.alloc_type);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.mem.position, graph.FindNode("rstd")->outputs[1].attr.mem.position);
  EXPECT_EQ(graph.FindNode("mm")->outputs[0].attr.opt.merge_scope, graph.FindNode("rstd")->outputs[1].attr.opt.merge_scope);
}

TEST(ScopeUse, AlreadyBindFailed) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
  auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, c, d});
  auto load0 = LoadStub("load0", data0).TQue(Position::kPositionVecIn, 1, 2);
  auto load0_id = load0.mem->reuse_id;
  auto load1 = LoadStub("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
  auto load1_id = load1.mem->reuse_id;
  EXPECT_NE(load0_id, load1_id);
  EXPECT_EQ(load1.Use(load0).mem->reuse_id, load1_id);
  EXPECT_NE(load0_id, load1_id);
}

TEST(ScopeUse, ReuseIdSame) {
  Graph graph("test_graph");
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = LoadStub("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
      auto load2 = LoadStub("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto load3 = LoadStub("load3", data3).TQue(Position::kPositionVecIn, 1, 2);
      auto relu = Cast("FakeRelu", load1).TBuf(Position::kPositionVecOut);
      auto mul1 = Mul("Mul1", relu, load2).Use(relu);
      auto sig_mod = Abs("FakeSigmod", load3).Use(mul1);
      auto mul2 = Mul("Mul2", mul1, sig_mod).TQue(Position::kPositionVecOut, 1, 2);
      auto store1 = Store("store1", relu);
      auto output1 = Output("output1", store1);
      EXPECT_EQ(relu.mem->reuse_id, sig_mod.mem->reuse_id);
      EXPECT_EQ(relu.mem->reuse_id, mul1.mem->reuse_id);
    }
  }
}

TEST(ScopeUse, UsedNotBindFailed) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto data0 = ContiguousData("data0", graph, ge::DT_FLOAT16, {a, b, d});
  auto data1 = ContiguousData("data1", graph, ge::DT_FLOAT16, {a, c, d});
  auto load0 = LoadStub("load0", data0);
  auto load1 = LoadStub("load1", data1);
  EXPECT_EQ(load1.Use(load0).mem->reuse_id, kIdNone);
  AscOpOutput asc_op_output;
  EXPECT_EQ(asc_op_output.Use(load0).output_index, UINT32_MAX);
}

TEST(CodeGenUtils, GenNextExecIdOk) {
  ge::AscGraph graph("test");
  EXPECT_EQ(CodeGenUtils::GenNextExecId(graph), 0L);
}

TEST(CodeGenUtils, PopBackLoopAxisFailed) {
  Graph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto ctx = CgContext::GetSharedThreadLocalContext();
  ASSERT_EQ(ctx, nullptr);
  auto ctx_obj = std::make_shared<CgContext>();
  CgContext::SetThreadLocalContext(ctx_obj);
  ctx = CgContext::GetSharedThreadLocalContext();
  ASSERT_NE(ctx, nullptr);
  // pop empty
  ctx->PopBackLoopAxis(a);
  ctx->PushLoopAxis(a);
  // pop order unmatch
  ctx->PopBackLoopAxis(b);
}

}  // namespace cg
}  // namespace ascir
}
