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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcBroadCastTmpSize(const ge::AscNode &node);

using namespace testing;
using namespace ge::ascir_op;

class CalcBroadCastTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
    template<ge::DataType T>
    void CreateGraph(ge::AscGraph &graph, Expression &s0, Expression &s1, Expression &s2) {
        ge::Expression One = ge::Symbol(1);
        s0 = graph.CreateSizeVar("s0");
        s1 = graph.CreateSizeVar("s1");
        s2 = graph.CreateSizeVar("s2");

        auto z0 = graph.CreateAxis("z0", s0);
        auto zo = graph.CreateAxis("zo", s1 + s2);
        auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
        auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);

        Data x1("x1", graph);
        Load load1("load1");
        ge::ascir_op::Broadcast broadcast("broadcast");
        Store store("store");
        Output y("y");

        x1.attr.sched.axis = {z0.id, zo_s_0.id};
        x1.y.dtype = T;
        *x1.y.axis = {z0.id, zo_s_0.id};
        *x1.y.repeats = {s0, s1};
        *x1.y.strides = {s1, One};

        load1.x = x1.y;
        load1.attr.sched.axis = {z0.id, zo_s_0.id};
        load1.y.dtype = T;
        *load1.y.axis = {z0.id, zo_s_0.id};
        *load1.y.repeats = {s0, s1};
        *load1.y.strides = {s1, One};
        *load1.y.vectorized_axis = {z0.id, zo.id};

        broadcast.x = {load1.y};
        broadcast.attr.sched.axis = {z0.id, zo.id};
        broadcast.y.dtype = T;
        *broadcast.y.axis = {z0.id, zo.id};
        *broadcast.y.repeats = {s0, s1 + s2};
        *broadcast.y.strides = {s1+s2, One};
        *broadcast.y.vectorized_axis = {z0.id, zo.id};

        store.x = broadcast.y;
        store.attr.sched.axis = {z0.id, zo.id};
        store.y.dtype = T;
        *store.y.axis = {z0.id, zo.id};
        *store.y.repeats = {s0, s1 + s2};
        *store.y.strides = {s1+s2, One};

        y.x = store.y;
        y.attr.sched.axis = {z0.id, zo.id};
        y.y.dtype = T;
        *y.y.axis = {z0.id, zo.id};
        *y.y.repeats = {s0, s1 + s2};
        *y.y.strides = {s1 + s2, One};
        ge::Expression Zero = ge::Symbol(0);
        for (auto node : graph.GetAllNodes()) {
            if (node->GetType() == "Data" || node->GetType() == "Output") {
                continue;
            }
            if (node->GetType() == "Load") {
                node->outputs[0].attr.vectorized_strides = {s1, Zero};
            } else {
                node->outputs[0].attr.vectorized_strides = {s1, One};
            }
        }
    }
};

/**
 * @tc.name:CalcBroadCastTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcBroadCastTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcBroadCastTmpSize returns correct size
 */
TEST_F(CalcBroadCastTmpSizeTest, CalcBroadCastTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
{
    ge::SizeVar s0(ge::Symbol("s0"));
    ge::SizeVar s1(ge::Symbol("s1"));
    ge::SizeVar s2(ge::Symbol("s2"));

    ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
    ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
    ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

    ge::AscGraph graph("test");
    ge::ascir_op::Data x("x", graph);
    std::shared_ptr<ge::AscNode> node = graph.FindNode("x");
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcBroadCastTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, ge::Symbol(8192));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcBroadCastTmpSize_NeedExtrabufU8
 * @tc.number: CalcBroadCastTmpSize_Test_002
 * @tc.desc: Test when node is valid then CalcBroadCastTmpSize returns correct size
 */
TEST_F(CalcBroadCastTmpSizeTest, CalcBroadCastTmpSize_NeedExtrabufU8)
{
    ge::AscGraph graph("test");
    Expression s0;
    Expression s1;
    Expression s2;
    CreateGraph<ge::DT_UINT8>(graph, s0, s1, s2);
    std::shared_ptr<ge::AscNode> node = graph.FindNode("broadcast");
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcBroadCastTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    Expression input_size = ge::Symbol(32) * sym::Ceiling(sym::Rational(1, 16) * s0);
    Expression output_size = ge::Symbol(32) * sym::Ceiling(ge::Symbol(2) * sym::Ceiling((s1+s2) * sym::Rational(1, 32)) * s0);
    Expression tmp_size = input_size + output_size + ge::Symbol(8192);
    ASSERT_EQ(result[0]->size, tmp_size);
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge