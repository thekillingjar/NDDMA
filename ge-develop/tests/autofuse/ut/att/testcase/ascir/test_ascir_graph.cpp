/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "base/base_types.h"

#include "node_utils_ex.h"
#include "graph_utils_ex.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
using namespace ge::ascir_op;
using namespace ge::ascir;
using namespace att;
namespace ge {
namespace {
void MakeGraph0Normal(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x("x", graph);
  x.attr.sched.exec_order = 0;
  x.attr.sched.axis = {z0.id, z1.id, z2.id};

  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id, z1.id, z2.id};
  *x.y.repeats = {s0, s1, s2};
  *x.y.strides = {s1 * s2, s2, ONE};

  Load load("load");
  load.x = x.y;
  load.attr.sched.exec_order = 1;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};

  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2, s2, ONE};
  Abs abs("abs");
  abs.x = load.y;
  abs.attr.sched.exec_order = 2;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};

  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2, s2, ONE};
  Store store("store");
  store.x = abs.y;
  store.attr.sched.exec_order = 3;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};

  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, ONE};
  Output y("y");
  y.x = store.y;
  y.attr.sched.exec_order = 4;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};

  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void MakeGraph0ByCg(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto axis = {z0, z1, z2};
  LOOP(z0) {
    LOOP(z0) {
      LOOP(z1) {
        LOOP(z2) {
          // 当前作用域内的所有的节点自动设置为sched.axis设置为{z0, z1, z2}
          // 执行序exec_order根据创建的节点顺序自动生成
          auto x =
              cg::ContiguousData("x", graph, ge::DT_FLOAT16, axis);  // 因为是连续tensor, 由axis推导出repeats, strides
          auto load = ascir::cg::Load(
              "load", x);  // 完成x->load的连边关系， load输出buf(dtype, axis, repeats, strides)完全继承x的输出buf
          auto abs = ascir::cg::Abs("abs", load);
          auto store = ascir::cg::Store("store", abs);
          auto y = ascir::cg::Output("y", store);  // 输出的dtype和view信息依赖用户自己设置
          y.dtype = ge::DT_FLOAT16;
          *y.axis = {z0.id, z1.id, z2.id};
        }
      }
    }
  }
}  // namespace

TEST(Ascir_Graph_Bg, CgApi_Ok) {
  ge::AscGraph graph0("test");
  ge::AscGraph graph1("test");
  MakeGraph0Normal(graph0);
  int32_t node_num = 0;
  for (const auto &node : graph0.GetAllNodes()) {
    node_num++;
  }
  ASSERT_TRUE(node_num > 0);
  MakeGraph0ByCg(graph1);
  node_num = 0;
  for (const auto &node : graph1.GetAllNodes()) {
    node_num++;
  }
  ASSERT_TRUE(node_num > 0);
//  ASSERT_EQ(ascir::utils::DebugStr(graph0), ascir::utils::DebugStr(graph1));
}
}
}
