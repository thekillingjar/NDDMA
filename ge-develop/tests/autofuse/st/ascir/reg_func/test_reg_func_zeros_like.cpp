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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcZerosLikeTmpSize(const ge::AscNode &node);

using namespace testing;

class CalcZerosLikeTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

/**
 * @tc.name:CalcZerosLikeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcZerosLikeTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcZerosLikeTmpSize returns correct size
 */
TEST_F(CalcZerosLikeTmpSizeTest, CalcZerosLikeTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    ge::ascir_op::ZerosLike zeroslike("zeroslike");
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

    zeroslike.x = load1.y;
    zeroslike.attr.sched.axis = {z0.id, zo_s_0.id};
    zeroslike.y.dtype = ge::DT_FLOAT;
    *zeroslike.y.axis = {z0.id, zo_s_0.id};
    *zeroslike.y.repeats = {s0, s1};
    *zeroslike.y.strides = {s1, Symbol(1)};
    

    store.x = zeroslike.y;
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


    std::shared_ptr<ge::AscNode> node = graph.FindNode("zeroslike");
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcZerosLikeTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, sym::Max(ge::Symbol(4)*s1*s0, ge::Symbol(8192)));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge


