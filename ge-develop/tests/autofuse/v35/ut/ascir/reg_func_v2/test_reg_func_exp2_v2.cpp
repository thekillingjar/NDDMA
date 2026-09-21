/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcExp2TmpSizeV2(const ge::AscNode &node);
const Expression MAX_TMP_BUFFER_SIZE = ge::Symbol(255 * 256 + 32);

using namespace testing;

class CalcExp2TmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CalcExp2TmpSizeV2Test, CalcExp2TmpSizeV2_ShouldReturnCorrectSize_Int32) {
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("zo", s1);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Exp2 exp2("exp2");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  const auto data_type = ge::DT_INT32;

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  exp2.x = load1.y;
  exp2.attr.sched.axis = {z0.id, z1.id};
  exp2.y.dtype = data_type;
  *exp2.y.axis = {z0.id, z1.id};
  *exp2.y.repeats = {s0, s1};
  *exp2.y.strides = {s1, Symbol(1)};

  store.x = exp2.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("exp2");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcExp2TmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, 0);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcExp2TmpSizeV2Test, CalcExp2TmpSizeV2_ShouldReturnCorrectSize_FLoat32) {
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("zo", s1);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Exp2 exp2("exp2");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  const auto data_type = ge::DT_FLOAT;

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  exp2.x = load1.y;
  exp2.attr.sched.axis = {z0.id, z1.id};
  exp2.y.dtype = data_type;
  *exp2.y.axis = {z0.id, z1.id};
  *exp2.y.repeats = {s0, s1};
  *exp2.y.strides = {s1, Symbol(1)};

  store.x = exp2.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("exp2");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcExp2TmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size,
            sym::Min(sym::Ceiling(ge::Symbol(4) * s1 * s0 / Symbol(32)) * ge::Symbol(64), MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcExp2TmpSizeV2Test, CalcExp2TmpSizeV2_ShouldReturnCorrectSize_Float16) {
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("zo", s1);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Exp2 exp2("exp2");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  const auto data_type = ge::DT_FLOAT16;

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  exp2.x = load1.y;
  exp2.attr.sched.axis = {z0.id, z1.id};
  exp2.y.dtype = data_type;
  *exp2.y.axis = {z0.id, z1.id};
  *exp2.y.repeats = {s0, s1};
  *exp2.y.strides = {s1, Symbol(1)};

  store.x = exp2.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("exp2");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcExp2TmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size,
            sym::Min(sym::Ceiling(ge::Symbol(2) * s1 * s0 / Symbol(32)) * ge::Symbol(128), MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace ge