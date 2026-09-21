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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcCastTmpSize(const ge::AscNode &node);

using namespace testing;

class CalcCastTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

TEST_F(CalcCastTmpSizeTest, CalcCastTmpSize_ShouldReturnCorrectSizeBetweenInt64ToU8_WhenNodelsValid)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z1", s2);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Cast cast("cast");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, z1.id, z2.id};
    x1.y.dtype = ge::DT_INT64;
    *x1.y.axis = {z0.id, z1.id, z2.id};
    *x1.y.repeats = {s0, s1, s2};
    *x1.y.strides = {s1*s2, s2, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id, z2.id};
    load1.y.dtype = ge::DT_INT64;
    *load1.y.axis = {z0.id, z1.id, z2.id};
    *load1.y.repeats = {s0, s1, s2};
    *load1.y.strides = {s1*s2, s2, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id, z2.id};

    cast.x = load1.y;
    cast.attr.sched.axis = {z0.id, z1.id, z2.id};
    cast.y.dtype = ge::DT_INT8;
    *cast.y.axis = {z0.id, z1.id, z2.id};
    *cast.y.repeats = {s0, s1, s2};
    *cast.y.strides = {s1*s2, s2, Symbol(1)};

    store.x = cast.y;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_INT8;
    *store.y.axis = {z0.id, z1.id, z2.id};
    *store.y.repeats = {s0, s1, s2};
    *store.y.strides = {s1*s2, s2, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_INT8;
    *y.y.axis = {z0.id, z1.id, z2.id};
    *y.y.repeats = {s0, s1, s2};
    *y.y.strides = {s1*s2, s2, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("cast");
    node->inputs[0].attr.vectorized_strides = {s1*s2, s2, Symbol(1)};;
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcCastTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, ge::Symbol(8192));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcCastTmpSizeTest, CalcCastTmpSize_ShouldReturnCorrectSizeBetweenU8ToFloat_WhenNodelsValid)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z1", s2);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Cast cast("cast");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, z1.id, z2.id};
    x1.y.dtype = ge::DT_UINT8;
    *x1.y.axis = {z0.id, z1.id, z2.id};
    *x1.y.repeats = {s0, s1, s2};
    *x1.y.strides = {s1*s2, s2, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id, z2.id};
    load1.y.dtype = ge::DT_UINT8;
    *load1.y.axis = {z0.id, z1.id, z2.id};
    *load1.y.repeats = {s0, s1, s2};
    *load1.y.strides = {s1*s2, s2, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id, z2.id};

    cast.x = load1.y;
    cast.attr.sched.axis = {z0.id, z1.id, z2.id};
    cast.y.dtype = ge::DT_FLOAT;
    *cast.y.axis = {z0.id, z1.id, z2.id};
    *cast.y.repeats = {s0, s1, s2};
    *cast.y.strides = {s1*s2, s2, Symbol(1)};

    store.x = cast.y;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, z1.id, z2.id};
    *store.y.repeats = {s0, s1, s2};
    *store.y.strides = {s1*s2, s2, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, z1.id, z2.id};
    *y.y.repeats = {s0, s1, s2};
    *y.y.strides = {s1*s2, s2, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("cast");
    node->inputs[0].attr.vectorized_strides = {s1*s2, s2, Symbol(1)};;
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcCastTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(32) * ge::sym::Ceiling((ge::sym::Rational(1 , 16) * s0 * s1 * s2)));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcCastTmpSizeTest, CalcCastTmpSize_ShouldReturnCorrectSizeBetweenU8ToInt32_WhenNodelsValid)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z1", s2);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Cast cast("cast");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, z1.id, z2.id};
    x1.y.dtype = ge::DT_UINT8;
    *x1.y.axis = {z0.id, z1.id, z2.id};
    *x1.y.repeats = {s0, s1, s2};
    *x1.y.strides = {s1*s2, s2, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id, z2.id};
    load1.y.dtype = ge::DT_UINT8;
    *load1.y.axis = {z0.id, z1.id, z2.id};
    *load1.y.repeats = {s0, s1, s2};
    *load1.y.strides = {s1*s2, s2, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id, z2.id};

    cast.x = load1.y;
    cast.attr.sched.axis = {z0.id, z1.id, z2.id};
    cast.y.dtype = ge::DT_INT32;
    *cast.y.axis = {z0.id, z1.id, z2.id};
    *cast.y.repeats = {s0, s1, s2};
    *cast.y.strides = {s1*s2, s2, Symbol(1)};

    store.x = cast.y;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_INT32;
    *store.y.axis = {z0.id, z1.id, z2.id};
    *store.y.repeats = {s0, s1, s2};
    *store.y.strides = {s1*s2, s2, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_INT32;
    *y.y.axis = {z0.id, z1.id, z2.id};
    *y.y.repeats = {s0, s1, s2};
    *y.y.strides = {s1*s2, s2, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("cast");
    node->inputs[0].attr.vectorized_strides = {s1*s2, s2, Symbol(1)};;
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcCastTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(32) * ge::sym::Ceiling((ge::sym::Rational(1 , 16) * s0 * s1 * s2)));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcCastTmpSizeTest, CalcCastTmpSize_ShouldReturnCorrectSizeBetweenU8ToInt4_WhenNodelsValid)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z1", s2);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Cast cast("cast");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, z1.id, z2.id};
    x1.y.dtype = ge::DT_UINT8;
    *x1.y.axis = {z0.id, z1.id, z2.id};
    *x1.y.repeats = {s0, s1, s2};
    *x1.y.strides = {s1*s2, s2, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id, z2.id};
    load1.y.dtype = ge::DT_UINT8;
    *load1.y.axis = {z0.id, z1.id, z2.id};
    *load1.y.repeats = {s0, s1, s2};
    *load1.y.strides = {s1*s2, s2, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id, z2.id};

    cast.x = load1.y;
    cast.attr.sched.axis = {z0.id, z1.id, z2.id};
    cast.y.dtype = ge::DT_INT4;
    *cast.y.axis = {z0.id, z1.id, z2.id};
    *cast.y.repeats = {s0, s1, s2};
    *cast.y.strides = {s1*s2, s2, Symbol(1)};

    store.x = cast.y;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_INT4;
    *store.y.axis = {z0.id, z1.id, z2.id};
    *store.y.repeats = {s0, s1, s2};
    *store.y.strides = {s1*s2, s2, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_INT4;
    *y.y.axis = {z0.id, z1.id, z2.id};
    *y.y.repeats = {s0, s1, s2};
    *y.y.strides = {s1*s2, s2, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("cast");
    node->inputs[0].attr.vectorized_strides = {s1*s2, s2, Symbol(1)};;
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcCastTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(32) * ge::sym::Ceiling((ge::sym::Rational(1 , 16) * s0 * s1 * s2)));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcCastTmpSizeTest, CalcCastTmpSize_ShouldReturnCorrectSizeBetweenInt64ToHalf_WhenNodelsValid)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z1", s2);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Cast cast("cast");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, z1.id, z2.id};
    x1.y.dtype = ge::DT_INT64;
    *x1.y.axis = {z0.id, z1.id, z2.id};
    *x1.y.repeats = {s0, s1, s2};
    *x1.y.strides = {s1*s2, s2, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id, z2.id};
    load1.y.dtype = ge::DT_INT64;
    *load1.y.axis = {z0.id, z1.id, z2.id};
    *load1.y.repeats = {s0, s1, s2};
    *load1.y.strides = {s1*s2, s2, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id, z2.id};

    cast.x = load1.y;
    cast.attr.sched.axis = {z0.id, z1.id, z2.id};
    cast.y.dtype = ge::DT_FLOAT16;
    *cast.y.axis = {z0.id, z1.id, z2.id};
    *cast.y.repeats = {s0, s1, s2};
    *cast.y.strides = {s1*s2, s2, Symbol(1)};

    store.x = cast.y;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_FLOAT16;
    *store.y.axis = {z0.id, z1.id, z2.id};
    *store.y.repeats = {s0, s1, s2};
    *store.y.strides = {s1*s2, s2, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_FLOAT16;
    *y.y.axis = {z0.id, z1.id, z2.id};
    *y.y.repeats = {s0, s1, s2};
    *y.y.strides = {s1*s2, s2, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("cast");
    node->inputs[0].attr.vectorized_strides = {s1*s2, s2, Symbol(1)};;
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcCastTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(32) * ge::sym::Ceiling((ge::sym::Rational(1 , 8) * s0 * s1 * s2)));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge