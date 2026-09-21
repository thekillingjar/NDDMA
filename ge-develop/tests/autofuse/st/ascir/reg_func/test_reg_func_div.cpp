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

#include "../test_util.h"
namespace ge{
namespace ascir{
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcDivTmpSize(const ge::AscNode &node);

using namespace testing;

class CalcDivTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

TEST_F(CalcDivTmpSizeTest, CalcDivTmpSize_ShouldReturnCorrectSize_WhenNodeIsUbScalar)
{
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z0", s1);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Data x2("x2", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Load load2("load2");
  ge::ascir_op::Div div("div");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_FLOAT;
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {Symbol(1), Symbol(1)};
  *x2.y.strides = {Symbol(1), Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {Symbol(1), Symbol(1)};
  *load2.y.strides = {Symbol(1), Symbol(1)};
  *load2.y.vectorized_axis = {z0.id, z1.id};

  div.x1 = load1.y;
  div.x2 = load2.y;
  div.attr.sched.axis = {z0.id, z1.id};
  div.y.dtype = ge::DT_FLOAT;
  *div.y.axis = {z0.id, z1.id};
  *div.y.repeats = {s0, s1};
  *div.y.strides = {s1, Symbol(1)};
  std::shared_ptr<ge::AscNode> node = graph.FindNode("div");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcDivTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(8192));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcDivTmpSizeTest, CalcDivTmpSize_ShouldReturnCorrectSize_WhenNodeIsScalar)
{
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z0", s1);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Scalar x2("x2", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Div div("div");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = DT_FLOAT;
  *x2.y.axis = {};
  *x2.y.repeats = {};
  *x2.y.strides = {};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  div.x1 = load1.y;
  div.x2 = x2.y;
  div.attr.sched.axis = {z0.id, z1.id};
  div.y.dtype = ge::DT_FLOAT;
  *div.y.axis = {z0.id, z1.id};
  *div.y.repeats = {s0, s1};
  *div.y.strides = {s1, Symbol(1)};
  std::shared_ptr<ge::AscNode> node = graph.FindNode("div");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcDivTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(8192));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcDivTmpSizeTest, CalcDivTmpSize_ShouldReturnCorrectSize_WhenNodeIsTensor)
{
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Div div("div");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

  div.x1 = load1.y;
  div.x2 = load1.y;
  div.attr.sched.axis = {z0.id, zo_s_0.id};
  div.y.dtype = ge::DT_FLOAT;
  *div.y.axis = {z0.id, zo_s_0.id};
  *div.y.repeats = {s0, s1};
  *div.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("div");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcDivTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcDivTmpSizeTest, CalcDivTmpSize_ShouldReturnCorrectSize_WhenNodeIsNotUbScalar)
{
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z0", s1);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Data x2("x2", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Load load2("load2");
  ge::ascir_op::Div div("div");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_FLOAT;
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {Symbol(2), Symbol(1)};
  *x2.y.strides = {Symbol(2), Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {Symbol(2), Symbol(1)};
  *load2.y.strides = {Symbol(2), Symbol(1)};
  *load2.y.vectorized_axis = {z0.id, z1.id};

  div.x1 = load1.y;
  div.x2 = load2.y;
  div.attr.sched.axis = {z0.id, z1.id};
  div.y.dtype = ge::DT_FLOAT;
  *div.y.axis = {z0.id, z1.id};
  *div.y.repeats = {s0, s1};
  *div.y.strides = {s1, Symbol(1)};
  std::shared_ptr<ge::AscNode> node = graph.FindNode("div");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcDivTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge