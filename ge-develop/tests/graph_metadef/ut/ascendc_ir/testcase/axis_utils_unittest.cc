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
#include <iostream>

#include "ascendc_ir/ascend_reg_ops.h"
#include "ascendc_ir/core/ascendc_ir_impl.h"
#include "ascir_ops.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/axis_utils.h"
#include "graph/expression/const_values.h"

class UtestAxisUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
namespace ge {
namespace ascir {
namespace cg {
using ge::Expression;
TEST_F(UtestAxisUtils, ReduceView_ok) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto[aBO, aBI] = graph.BlockSplit(a.id, "nbi", "nbo");
  (void) aBO;
  auto[aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");
  auto data = graph.CreateContiguousData("data0", DT_FLOAT, {a, b});

  LOOP(*aBO) {
    LOOP(*aBIO) {
      LOOP(*aBII) {
        auto load = LoadStub("load0", data);
        auto before_view = View{*load.axis, *load.repeats, *load.strides};
        auto after_view = AxisUtils::ReduceView(before_view, b.id);
        EXPECT_EQ(ge::ViewToString(before_view),
                  "{ axis: [2, 4, 5, 1], repeats: [(A / (nbo_size)), (nbo_size / (nio_size)), nio_size, B], strides: [(B * nbo_size), (B * nio_size), B, 1] }");
        EXPECT_EQ(ge::ViewToString(after_view),
                  "{ axis: [2, 4, 5, 1], repeats: [(A / (nbo_size)), (nbo_size / (nio_size)), nio_size, B], strides: [nbo_size, nio_size, 1, 0] }");
      }
    }
  }
}

TEST_F(UtestAxisUtils, GetDefaultVectorizedAxis_ok) {
  std::vector<int64_t> axis = {0, 1, 2, 3};
  EXPECT_EQ(AxisUtils::GetDefaultVectorizedAxis(axis, 0), std::vector<int64_t>({1, 2, 3}));
  EXPECT_EQ(AxisUtils::GetDefaultVectorizedAxis(axis, 1), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(AxisUtils::GetDefaultVectorizedAxis(axis, 2), std::vector<int64_t>({3}));
  EXPECT_EQ(AxisUtils::GetDefaultVectorizedAxis(axis, 3), std::vector<int64_t>({}));
  EXPECT_EQ(AxisUtils::GetDefaultVectorizedAxis(axis, 4), std::vector<int64_t>({0, 1, 2, 3}));
}

TEST_F(UtestAxisUtils, UpdateViewIfCrossLoop_No_need_update) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto[aBO, aBI] = graph.BlockSplit(a.id, "nbi", "nbo");
  (void) aBO;
  auto[aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");
  auto trans_infos = graph.GetAllAxisTransInfo();
  LOOP(*aBO) {
    LOOP(*aBIO) {
      LOOP(*aBII) {
        auto data = ContiguousData("data0", graph, DT_FLOAT, {a, b, c});
        auto load = LoadStub("load0", data);
        auto data_attr =
            OpDescUtils::GetOpDescFromOperator(data.GetOwnerOp())->GetOrCreateAttrsGroup<AscNodeAttr>();
        auto load_attr = CodeGenUtils::GetOwnerOpAscAttr(load.GetOwnerOp());
        EXPECT_EQ(data_attr->sched.axis, std::vector<int64_t>({aBO->id, aBIO->id, aBII->id}));
        EXPECT_EQ(data_attr->sched.axis, load_attr->sched.axis);
        EXPECT_FALSE(AxisUtils::UpdateViewIfCrossLoop(trans_infos,
                                                      data_attr->sched.axis,
                                                      load_attr->sched.axis,
                                                      {*load.axis, *load.repeats, *load.strides}).first);
      }
    }
  }
}

TEST_F(UtestAxisUtils, UpdateViewIfCrossLoop_Update_success1) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto[aBO, aBI] = graph.BlockSplit(a.id, "nbi", "nbo");
  (void) aBO;
  auto[aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");
  auto trans_infos = graph.GetAllAxisTransInfo();
  auto data = ContiguousData("data0", graph, DT_FLOAT, {a, b, c});
  auto data_sched_axis = graph.FindNode("data0")->attr.sched.axis;
  EXPECT_TRUE(data_sched_axis.empty());
  LOOP(*aBO) {
    LOOP(*aBIO) {
      LOOP(*aBII) {
        // Load接口内部会调用UpdateViewIfCrossLoop
        auto load = LoadStub("load0", data);
        auto load_attr = OpDescUtils::GetOpDescFromOperator(load.GetOwnerOp())->GetOrCreateAttrsGroup<AscNodeAttr>();
        EXPECT_EQ(load_attr->sched.axis, std::vector<int64_t>({aBO->id, aBIO->id, aBII->id}));
        EXPECT_NE(data_sched_axis, load_attr->sched.axis);
        EXPECT_EQ(*load.axis, std::vector<int64_t>({aBO->id, aBIO->id, aBII->id, b.id, c.id}));
        std::vector<ge::Expression> repeats_expect;
        repeats_expect.emplace_back(A / aBI->size);
        repeats_expect.emplace_back(aBI->size / aBII->size);
        repeats_expect.emplace_back(aBII->size);
        repeats_expect.emplace_back(B);
        repeats_expect.emplace_back(C);
        EXPECT_EQ(load.repeats->size(), repeats_expect.size());
        size_t index = 0U;
        for (const auto &re : repeats_expect) {
          EXPECT_EQ((*load.repeats)[index++], re);
        }
        std::vector<ge::Expression> strides_expect;
        strides_expect.emplace_back(B * C * aBI->size);
        strides_expect.emplace_back(B * C * aBII->size);
        strides_expect.emplace_back(B * C);
        strides_expect.emplace_back(C);
        strides_expect.emplace_back(sym::kSymbolOne);
        EXPECT_EQ(load.strides->size(), strides_expect.size());
        index = 0U;
        for (const auto &se : strides_expect) {
          EXPECT_EQ((*load.strides)[index++], se);
        }
      }
    }
  }
}

TEST_F(UtestAxisUtils, UpdateViewIfCrossLoop_Update_success2) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto[aBO, aBI] = graph.BlockSplit(a.id, "nbi", "nbo");
  (void) aBO;
  auto aBIB = graph.MergeAxis({aBI->id, b.id});
  auto trans_infos = graph.GetAllAxisTransInfo();
  EXPECT_EQ(trans_infos.size(), 2U);
  auto data = ContiguousData("data0", graph, DT_FLOAT, {a, b, c});
  auto data_sched_axis = graph.FindNode("data0")->attr.sched.axis;
  EXPECT_TRUE(data_sched_axis.empty());
  LOOP(*aBO) {
    LOOP(*aBIB) {
      // Load接口内部会调用UpdateViewIfCrossLoop
      auto load = LoadStub("load0", data);
      auto load_attr = OpDescUtils::GetOpDescFromOperator(load.GetOwnerOp())->GetOrCreateAttrsGroup<AscNodeAttr>();
      EXPECT_EQ(load_attr->sched.axis, std::vector<int64_t>({aBO->id, aBIB->id}));
      EXPECT_NE(data_sched_axis, load_attr->sched.axis);
      // 测试多次调用UpdateViewIfCrossLoop
      auto pair = AxisUtils::UpdateViewIfCrossLoop(trans_infos,
                                                   data_sched_axis,
                                                   load_attr->sched.axis,
                                                   {*load.axis, *load.repeats, *load.strides});
      EXPECT_TRUE(pair.first);
      View view{*load.axis, *load.repeats, *load.strides};
      view = pair.second;
      EXPECT_EQ(*load.axis, std::vector<int64_t>({aBO->id, aBIB->id, c.id}));
      std::vector<ge::Expression> repeats_expect;
      repeats_expect.emplace_back(A / aBI->size);
      repeats_expect.emplace_back(aBI->size * B);
      repeats_expect.emplace_back(C);
      EXPECT_EQ(load.repeats->size(), repeats_expect.size());
      size_t index = 0U;
      for (const auto &re : repeats_expect) {
        EXPECT_EQ((*load.repeats)[index++], re);
      }
      std::vector<ge::Expression> strides_expect;
      strides_expect.emplace_back(aBI->size * B * C);
      strides_expect.emplace_back(C);
      strides_expect.emplace_back(sym::kSymbolOne);
      EXPECT_EQ(load.strides->size(), strides_expect.size());
      index = 0U;
      for (const auto &se : strides_expect) {
        EXPECT_EQ((*load.strides)[index++], se);
      }
    }
  }
}

TEST_F(UtestAxisUtils, UpdateViewIfCrossLoop_DelSceduleAxes_success3) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto L = Symbol("L");
  auto R = Symbol("R");
  auto axis = graph.CreateAxis("axes", A); // 0
  auto loop = graph.CreateAxis("loop", L); // 1
  auto r = graph.CreateAxis("r", R); // 2
  auto[axisB, axisb] = graph.BlockSplit(axis.id, "axisb", "axisB"); // 3,4
  auto[loopT, loopt] = graph.TileSplit(loop.id, "loopt", "loopT"); // 5,6
  auto data0 = ContiguousData("data0", graph, DT_FLOAT, {axis, loop, r}); // 0, 1, 2
  auto data1 = ContiguousData("data1", graph, DT_FLOAT, {axis, loop, r});
  auto data2 = ContiguousData("data2", graph, DT_FLOAT, {axis, loop, r});
  auto data_sched_axis = graph.FindNode("data0")->attr.sched.axis;
  LOOP(*axisB) { // 3
    LOOP(*axisb) { // 4
      AscOpOutput y1({loop.id, r.id});
      auto x1 = LoadStub("load1", data0);
      auto x2 = LoadStub("load2", data1);
      auto x3 = LoadStub("load3", data2);
      LOOP(*loopT) { // 5
        auto out1 = CalcY("calc_y", x1, x2, x3, x3);
        EXPECT_EQ(*out1.axis, std::vector<int64_t>({axisB->id, axisb->id, loopT->id, loopt->id, r.id}));
        std::vector<ge::Expression> repeats_expect;
        repeats_expect.emplace_back(axis.size / axisb->size);
        repeats_expect.emplace_back(axisb->size);
        repeats_expect.emplace_back(loop.size / loopt->size);
        repeats_expect.emplace_back(loopt->size);
        repeats_expect.emplace_back(r.size);
        EXPECT_EQ(out1.repeats->size(), repeats_expect.size());
        size_t index = 0U;
        for (const auto &re : repeats_expect) {
          EXPECT_EQ((*out1.repeats)[index++], re) << " index=" << index;
        }
        std::vector<ge::Expression> strides_expect;
        strides_expect.emplace_back(axisb->size * loop.size * r.size);
        strides_expect.emplace_back(loop.size * r.size);
        strides_expect.emplace_back(r.size * loopt->size);
        strides_expect.emplace_back(r.size);
        strides_expect.emplace_back(sym::kSymbolOne);
        EXPECT_EQ(out1.strides->size(), strides_expect.size());
        index = 0U;
        for (const auto &se : strides_expect) {
          EXPECT_EQ((*out1.strides)[index++], se) << " index=" << index;
        }
        y1.AutoOffset() = out1;
        EXPECT_EQ(*y1.vectorized_axis, std::vector<int64_t>({loop.id, r.id}));
      }
      auto output = StoreStub("store", y1);
      EXPECT_EQ(*output.axis, std::vector<int64_t>({axisB->id, axisb->id, loop.id, r.id}));
      std::vector<ge::Expression> repeats_expect;
      repeats_expect.emplace_back(axis.size / axisb->size);
      repeats_expect.emplace_back(axisb->size);
      repeats_expect.emplace_back(loop.size);
      repeats_expect.emplace_back(r.size);
      EXPECT_EQ(output.repeats->size(), repeats_expect.size());
      size_t index = 0U;
      for (const auto &re : repeats_expect) {
        EXPECT_EQ((*output.repeats)[index++], re) << " index=" << index;
      }
      std::vector<ge::Expression> strides_expect;
      strides_expect.emplace_back(axisb->size * loop.size * r.size);
      strides_expect.emplace_back(loop.size * r.size);
      strides_expect.emplace_back(r.size);
      strides_expect.emplace_back(sym::kSymbolOne);
      EXPECT_EQ(output.strides->size(), strides_expect.size());
      index = 0U;
      for (const auto &se:strides_expect) {
        EXPECT_EQ((*output.strides)[index++], se) << " index=" << index;
      }
    }
  }
}

TEST_F(UtestAxisUtils, UpdateViewIfCrossLoop_AddDelSceduleAxes_success3) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto L = Symbol("L");
  auto R = Symbol("R");
  auto axis = graph.CreateAxis("axes", A); // 0
  auto loop = graph.CreateAxis("loop", L);          // 1
  auto r = graph.CreateAxis("r", R);                // 2
  auto [axisB, axisb] = graph.BlockSplit(axis.id);  // 3,4
  auto [loopT, loopt] = graph.TileSplit(loop.id); // 5,6
  auto [rT, rt] = graph.TileSplit(r.id); // 7,8
  auto data0 = ContiguousData("data0", graph, DT_FLOAT, {axis, loop, r});
  auto data1 = ContiguousData("data1", graph, DT_FLOAT, {axis, loop, r});
  auto data2 = ContiguousData("data2", graph, DT_FLOAT, {axis, loop, r});
  auto data_sched_axis = graph.FindNode("data0")->attr.sched.axis;
  LOOP(*axisB) { // 3
    LOOP(*axisb) { // 4
      AscOpOutput y1({loop.id, r.id});
      auto x1 = LoadStub("load1", data0);
      auto x2 = LoadStub("load2", data1);
      auto x3 = LoadStub("load3", data2);
      LOOP(*loopT) { // 5
        auto out1 = CalcY("calc_y", x1, x2, x3, x3);
        EXPECT_EQ(*out1.axis, std::vector<int64_t>({axisB->id, axisb->id, loopT->id, loopt->id, r.id}));
        std::vector<ge::Expression> repeats_expect;
        repeats_expect.emplace_back(axis.size / axisb->size);
        repeats_expect.emplace_back(axisb->size);
        repeats_expect.emplace_back(loop.size / loopt->size);
        repeats_expect.emplace_back(loopt->size);
        repeats_expect.emplace_back(r.size);
        EXPECT_EQ(out1.repeats->size(), repeats_expect.size());
        size_t index = 0U;
        for (const auto &re : repeats_expect) {
          EXPECT_EQ((*out1.repeats)[index++], re);
        }
        std::vector<ge::Expression> strides_expect;
        strides_expect.emplace_back(axisb->size * loop.size * r.size);
        strides_expect.emplace_back(loop.size * r.size);
        strides_expect.emplace_back(r.size * loopt->size);
        strides_expect.emplace_back(r.size);
        strides_expect.emplace_back(sym::kSymbolOne);
        EXPECT_EQ(out1.strides->size(), strides_expect.size());
        index = 0U;
        for (const auto &se : strides_expect) {
          EXPECT_EQ((*out1.strides)[index++], se);
        }
        y1.AutoOffset() = out1;
        EXPECT_EQ(*y1.vectorized_axis, std::vector<int64_t>({loop.id, r.id}));
      }
      LOOP(*rT) { // 7
        auto output = StoreStub("store", y1);
        // 3,4,1,7,8
        EXPECT_EQ(*output.axis, std::vector<int64_t>({axisB->id, axisb->id, rT->id, loop.id, rt->id}));
        std::vector<ge::Expression> repeats_expect;
        repeats_expect.emplace_back(axis.size / axisb->size);
        repeats_expect.emplace_back(axisb->size);
        repeats_expect.emplace_back(r.size / rt->size);
        repeats_expect.emplace_back(loop.size);
        repeats_expect.emplace_back(rt->size);
        EXPECT_EQ(output.repeats->size(), repeats_expect.size());
        size_t index = 0U;
        for (const auto &re : repeats_expect) {
           EXPECT_EQ((*output.repeats)[index++], re) << " index=" << index;
        }
        std::vector<ge::Expression> strides_expect;
        strides_expect.emplace_back(axisb->size * loop.size * r.size);
        strides_expect.emplace_back(loop.size * r.size);
        strides_expect.emplace_back(rt->size);
        strides_expect.emplace_back(r.size);
        strides_expect.emplace_back(sym::kSymbolOne);
        EXPECT_EQ(output.strides->size(), strides_expect.size());
        index = 0U;
        for (const auto &se:strides_expect) {
          EXPECT_EQ((*output.strides)[index++], se) << " index=" << index;
        }
      }
    }
  }
}

TEST_F(UtestAxisUtils, UpdateViewIfCrossLoop_ReorderViewSuccess) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto L = Symbol("L");
  auto R = Symbol("R");
  auto axis = graph.CreateAxis("axes", A);    // 0
  auto loop = graph.CreateAxis("loop", L);    // 1
  auto r = graph.CreateAxis("r", R);          // 2
  auto [axisB, axisb] = graph.BlockSplit(axis.id);  // 3,4
  auto [loopT, loopt] = graph.TileSplit(loop.id);   // 5,6
  auto [rT, rt] = graph.TileSplit(r.id);            // 7,8
  std::vector<int64_t> axes = {3, 4, 7, 8, 1};
  std::vector<ge::Expression> repeats_expect;
  repeats_expect.emplace_back(axis.size / axisb->size); // 3
  repeats_expect.emplace_back(axisb->size); // 4
  repeats_expect.emplace_back(r.size / rt->size); // 7
  repeats_expect.emplace_back(loop.size); // 8
  repeats_expect.emplace_back(rt->size); // 1

  std::vector<ge::Expression> strides_expect;
  strides_expect.emplace_back(axisb->size * loop.size * r.size);
  strides_expect.emplace_back(loop.size * r.size);
  strides_expect.emplace_back(rt->size);
  strides_expect.emplace_back(r.size);
  strides_expect.emplace_back(sym::kSymbolOne);

  View src_view{axes, repeats_expect, strides_expect};
  std::vector<int64_t> my_api_sched_axes = {3, 4, 7};
  auto dst_view = AxisUtils::ReorderView(src_view, my_api_sched_axes);
  auto [axes_res, repeats, strides] = dst_view;
  std::vector<int64_t> expect_axes = {3, 4, 7, 8, 1};
  EXPECT_EQ(axes_res, expect_axes);
  size_t index = 0U;
  for (const auto &re : repeats_expect) {
    EXPECT_EQ(repeats[index++], re) << " index=" << index;
  }
  index = 0U;
  for (const auto &re : strides_expect) {
    EXPECT_EQ(strides[index++], re) << " index=" << index;
  }
}

TEST_F(UtestAxisUtils, UpdateViewIfCrossLoop_ReorderViewSuccess2) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto L = Symbol("L");
  auto R = Symbol("R");
  auto axis = graph.CreateAxis("axes", A);    // 0
  auto loop = graph.CreateAxis("loop", L);    // 1
  auto r = graph.CreateAxis("r", R);          // 2
  auto [axisB, axisb] = graph.BlockSplit(axis.id);  // 3,4
  auto [loopT, loopt] = graph.TileSplit(loop.id);   // 5,6
  auto [rT, rt] = graph.TileSplit(r.id);            // 7,8
  std::vector<int64_t> axes = {8, 7, 1, 3, 4};
  std::vector<ge::Expression> repeats;
  repeats.emplace_back(loop.size); // 8
  repeats.emplace_back(r.size / rt->size); // 7
  repeats.emplace_back(rt->size); // 1
  repeats.emplace_back(axis.size / axisb->size); // 3
  repeats.emplace_back(axisb->size); // 4

  std::vector<ge::Expression> strides;
  strides.emplace_back(r.size); // 8
  strides.emplace_back(rt->size); // 7
  strides.emplace_back(sym::kSymbolOne); // 1
  strides.emplace_back(axisb->size * loop.size * r.size); // 3
  strides.emplace_back(loop.size * r.size); // 4

  View src_view{axes, repeats, strides};
  std::vector<int64_t> my_api_sched_axes = {3, 4, 7};
  auto dst_view = AxisUtils::ReorderView(src_view, my_api_sched_axes);
  auto [axes_res, repeats_res, strides_res] = dst_view;
  std::vector<int64_t> expect_axes = {3, 4, 7, 8, 1};

  std::vector<ge::Expression> repeats_expect;
  repeats_expect.emplace_back(axis.size / axisb->size); // 3
  repeats_expect.emplace_back(axisb->size); // 4
  repeats_expect.emplace_back(r.size / rt->size); // 7
  repeats_expect.emplace_back(loop.size); // 8
  repeats_expect.emplace_back(rt->size); // 1

  std::vector<ge::Expression> strides_expect;
  strides_expect.emplace_back(axisb->size * loop.size * r.size); // 3
  strides_expect.emplace_back(loop.size * r.size); // 4
  strides_expect.emplace_back(rt->size); // 7
  strides_expect.emplace_back(r.size); // 8
  strides_expect.emplace_back(sym::kSymbolOne); // 1
  EXPECT_EQ(axes_res, expect_axes);
  size_t index = 0U;
  for (const auto &re : repeats_expect) {
    EXPECT_EQ(repeats_res[index++], re) << " index=" << index;
  }
  index = 0U;
  for (const auto &re : strides_expect) {
    EXPECT_EQ(strides_res[index++], re) << " index=" << index;
  }
}
}
}
}