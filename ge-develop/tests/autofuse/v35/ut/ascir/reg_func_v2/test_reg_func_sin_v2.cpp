/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSinTmpSizeV2(const ge::AscNode &node);

using namespace testing;
using namespace ge::ascir_op;

class CalcSinTmpSizeV2Test:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

/**
 * @tc.name: CalcSinTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT
 * @tc.number: CalcSinTmpSizeV2_Test_001
 * @tc.desc: Test CalcSinTmpSizeV2 returns correct size when input is DT_FLOAT
 */
TEST_F(CalcSinTmpSizeV2Test, CalcSinTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT)
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
    ge::ascir_op::Sin sin("sin");
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

    sin.x = load1.y;
    sin.attr.sched.axis = {z0.id, zo_s_0.id};
    sin.y.dtype = ge::DT_FLOAT;
    *sin.y.axis = {z0.id, zo_s_0.id};
    *sin.y.repeats = {s0, s1};
    *sin.y.strides = {s1, Symbol(1)};

    store.x = sin.y;
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

    std::shared_ptr<ge::AscNode> node = graph.FindNode("sin");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcSinTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(768));  // SIN_ONE_REPEAT_BYTE_SIZE * SIN_FLOAT_CALC_PROC
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name: CalcSinTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT16
 * @tc.number: CalcSinTmpSizeV2_Test_002
 * @tc.desc: Test CalcSinTmpSizeV2 returns correct size when input is DT_FLOAT16
 */
TEST_F(CalcSinTmpSizeV2Test, CalcSinTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT16)
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
    ge::ascir_op::Sin sin("sin");
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

    sin.x = load1.y;
    sin.attr.sched.axis = {z0.id, zo_s_0.id};
    sin.y.dtype = ge::DT_FLOAT16;
    *sin.y.axis = {z0.id, zo_s_0.id};
    *sin.y.repeats = {s0, s1};
    *sin.y.strides = {s1, Symbol(1)};

    store.x = sin.y;
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

    std::shared_ptr<ge::AscNode> node = graph.FindNode("sin");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcSinTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(2048));  // SIN_ONE_REPEAT_BYTE_SIZE * SIN_HALF_CALC_PROC = 256 * 4
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

} // namespace ascir
} // namespace ge