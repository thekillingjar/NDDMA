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

#include "graph/operator_reg.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

namespace ge {
namespace ascir {
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcReduceTmpSizeV2(const ge::AscNode &node);

using namespace testing;
using namespace ge::ascir_op;

class CalcReduceTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};
template <ge::DataType T>
void CreateGraphReduceWithInputMultiRefs(ge::AscGraph &graph, Expression &s1, Expression &s2) {
  ge::Expression One = ge::Symbol(1);
  ge::Expression Zero = ge::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1", graph);
  Load load1("load1");
  ge::ascir_op::Max max0("max0");
  Store store("store");
  Output y("y");
  Store store1("store1");
  Output y1("y1");

  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = T;
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = T;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  *load1.y.vectorized_axis = {z1.id, z2.id};

  max0.x = load1.y;
  max0.attr.sched.axis = {z0.id, z1.id, z2.id};
  max0.attr.sched.loop_axis = {z0.id};
  max0.y.dtype = T;
  *max0.y.axis = {z0.id, z1.id, z2.id};
  *max0.y.repeats = {s0, s1, One};
  *max0.y.strides = {s2, One, Zero};
  *max0.y.vectorized_axis = {z1.id, z2.id};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = T;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = T;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};

  store1.x = load1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.y.dtype = T;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.y.dtype = T;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};
}

template <ge::DataType T>
void CreateGraphReduceAccWithInputMultiRefs(ge::AscGraph &graph, Expression &s1, Expression &s2) {
  ge::Expression One = ge::Symbol(1);
  ge::Expression Zero = ge::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1", graph);
  Load load1("load1");
  ge::ascir_op::Prod prod0("prod0");
  Store store("store");
  Output y("y");
  Store store1("store1");
  Output y1("y1");

  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = T;
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = T;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  *load1.y.vectorized_axis = {z1.id, z2.id};

  prod0.x = load1.y;
  prod0.attr.sched.axis = {z0.id, z1.id, z2.id};
  prod0.attr.sched.loop_axis = {z0.id};
  prod0.y.dtype = T;
  *prod0.y.axis = {z0.id, z1.id, z2.id};
  *prod0.y.repeats = {s0, s1, One};
  *prod0.y.strides = {s2, One, Zero};
  *prod0.y.vectorized_axis = {z1.id, z2.id};

  store.x = prod0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = T;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = T;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};

  store1.x = load1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.y.dtype = T;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.y.dtype = T;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};
}

TEST_F(CalcReduceTmpSizeV2Test, CalcReduceTmpSizeV2_test_0) {
  ge::AscGraph graph("testx");
  ge::Expression One = ge::Symbol(1);
  ge::Expression Zero = ge::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphReduceWithInputMultiRefs<ge::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<ge::AscNode> node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcReduceTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 2);
}

TEST_F(CalcReduceTmpSizeV2Test, CalcReduceTmpSize_test_1) {
  ge::AscGraph graph("testx");
  ge::Expression One = ge::Symbol(1);
  ge::Expression Zero = ge::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphReduceAccWithInputMultiRefs<ge::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<ge::AscNode> node = graph.FindNode("prod0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcReduceTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 2);
}

}  // namespace ascir
}  // namespace ge