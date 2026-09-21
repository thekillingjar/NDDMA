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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGeTmpSize(const ge::AscNode &node);
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcEqTmpSize(const ge::AscNode &node);
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcNeTmpSize(const ge::AscNode &node);
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGtTmpSize(const ge::AscNode &node);
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLeTmpSize(const ge::AscNode &node);
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLtTmpSize(const ge::AscNode &node);

const Expression MAX_TMP_BUFFER_SIZE = ge::Symbol(255 * 256 + 32);

using namespace testing;

class CalcCompareTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

/**
 * @tc.name:CalcGeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcGeTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcGeTmpSize returns correct size
 */
TEST_F(CalcCompareTmpSizeTest, CalcGeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::Ge ge("compare");
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

    ge.x1 = load1.y;
    ge.x2 = load1.y;
    ge.attr.sched.axis = {z0.id, zo_s_0.id};
    ge.y.dtype = ge::DT_FLOAT;
    *ge.y.axis = {z0.id, zo_s_0.id};
    *ge.y.repeats = {s0, s1};
    *ge.y.strides = {s1, Symbol(1)};

    store.x = ge.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("compare");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGeTmpSize(*node);
    ge::Expression compareNormalTmpSize = ge::Symbol(8) * s0 + 
                                          ge::Symbol(128) * ((s0 * ge::Symbol(2) + ge::Symbol(3)) / ge::Symbol(4));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Min(sym::Max(ge::Symbol(288) * s1, compareNormalTmpSize), MAX_TMP_BUFFER_SIZE));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcEqTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcEqTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcEqTmpSize returns correct size
 */
TEST_F(CalcCompareTmpSizeTest, CalcEqTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::Eq eq("compare");
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

    eq.x1 = load1.y;
    eq.x2 = load1.y;
    eq.attr.sched.axis = {z0.id, zo_s_0.id};
    eq.y.dtype = ge::DT_FLOAT;
    *eq.y.axis = {z0.id, zo_s_0.id};
    *eq.y.repeats = {s0, s1};
    *eq.y.strides = {s1, Symbol(1)};

    store.x = eq.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("compare");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcEqTmpSize(*node);
    ge::Expression compareNormalTmpSize = ge::Symbol(8) * s0 + 
                                          ge::Symbol(128) * ((s0 * ge::Symbol(2) + ge::Symbol(3)) / ge::Symbol(4));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Min(sym::Max(ge::Symbol(288) * s1, compareNormalTmpSize), MAX_TMP_BUFFER_SIZE));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcNeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcNeTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcNeTmpSize returns correct size
 */
TEST_F(CalcCompareTmpSizeTest, CalcNeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::Ne ne("compare");
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

    ne.x1 = load1.y;
    ne.x2 = load1.y;
    ne.attr.sched.axis = {z0.id, zo_s_0.id};
    ne.y.dtype = ge::DT_FLOAT;
    *ne.y.axis = {z0.id, zo_s_0.id};
    *ne.y.repeats = {s0, s1};
    *ne.y.strides = {s1, Symbol(1)};

    store.x = ne.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("compare");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcNeTmpSize(*node);
    ge::Expression compareNormalTmpSize = ge::Symbol(8) * s0 + 
                                          ge::Symbol(128) * ((s0 * ge::Symbol(2) + ge::Symbol(3)) / ge::Symbol(4));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Min(sym::Max(ge::Symbol(288) * s1, compareNormalTmpSize), MAX_TMP_BUFFER_SIZE));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcGtTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcGtTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcGtTmpSize returns correct size
 */
TEST_F(CalcCompareTmpSizeTest, CalcGtTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::Gt gt("compare");
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

    gt.x1 = load1.y;
    gt.x2 = load1.y;
    gt.attr.sched.axis = {z0.id, zo_s_0.id};
    gt.y.dtype = ge::DT_FLOAT;
    *gt.y.axis = {z0.id, zo_s_0.id};
    *gt.y.repeats = {s0, s1};
    *gt.y.strides = {s1, Symbol(1)};

    store.x = gt.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("compare");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGtTmpSize(*node);
    ge::Expression compareNormalTmpSize = ge::Symbol(8) * s0 + 
                                          ge::Symbol(128) * ((s0 * ge::Symbol(2) + ge::Symbol(3)) / ge::Symbol(4));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Min(sym::Max(ge::Symbol(288) * s1, compareNormalTmpSize), MAX_TMP_BUFFER_SIZE));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcLeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcLeTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcLeTmpSize returns correct size
 */
TEST_F(CalcCompareTmpSizeTest, CalcLeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::Le le("compare");
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

    le.x1 = load1.y;
    le.x2 = load1.y;
    le.attr.sched.axis = {z0.id, zo_s_0.id};
    le.y.dtype = ge::DT_FLOAT;
    *le.y.axis = {z0.id, zo_s_0.id};
    *le.y.repeats = {s0, s1};
    *le.y.strides = {s1, Symbol(1)};

    store.x = le.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("compare");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcLeTmpSize(*node);
    ge::Expression compareNormalTmpSize = ge::Symbol(8) * s0 + 
                                          ge::Symbol(128) * ((s0 * ge::Symbol(2) + ge::Symbol(3)) / ge::Symbol(4));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Min(sym::Max(ge::Symbol(288) * s1, compareNormalTmpSize), MAX_TMP_BUFFER_SIZE));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcLtTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcLtTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcLtTmpSize returns correct size
 */
TEST_F(CalcCompareTmpSizeTest, CalcLtTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::Lt lt("compare");
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

    lt.x1 = load1.y;
    lt.x2 = load1.y;
    lt.attr.sched.axis = {z0.id, zo_s_0.id};
    lt.y.dtype = ge::DT_FLOAT;
    *lt.y.axis = {z0.id, zo_s_0.id};
    *lt.y.repeats = {s0, s1};
    *lt.y.strides = {s1, Symbol(1)};

    store.x = lt.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("compare");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcLtTmpSize(*node);
    ge::Expression compareNormalTmpSize = ge::Symbol(8) * s0 + 
                                          ge::Symbol(128) * ((s0 * ge::Symbol(2) + ge::Symbol(3)) / ge::Symbol(4));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Min(sym::Max(ge::Symbol(288) * s1, compareNormalTmpSize), MAX_TMP_BUFFER_SIZE));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcFloat16LtTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcFloat16LtTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcLtTmpSize returns correct size
 */
TEST_F(CalcCompareTmpSizeTest, CalcFloat16LtTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::Lt lt("compare");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = ge::DT_FLOAT16;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = ge::DT_FLOAT16;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    lt.x1 = load1.y;
    lt.x2 = load1.y;
    lt.attr.sched.axis = {z0.id, zo_s_0.id};
    lt.y.dtype = ge::DT_FLOAT16;
    *lt.y.axis = {z0.id, zo_s_0.id};
    *lt.y.repeats = {s0, s1};
    *lt.y.strides = {s1, Symbol(1)};

    store.x = lt.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT16;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT16;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("compare");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcLtTmpSize(*node);
    ge::Expression compareNormalTmpSize = ge::Symbol(16) * s0 + 
                                          ge::Symbol(128) * ((s0 * ge::Symbol(2) + ge::Symbol(1)) / ge::Symbol(2));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Min(sym::Max(ge::Symbol(288) * s1, compareNormalTmpSize), MAX_TMP_BUFFER_SIZE));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge