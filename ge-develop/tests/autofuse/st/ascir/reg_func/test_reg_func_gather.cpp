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
constexpr int32_t MAX_TMP_SIZE = 133*1024;
constexpr uint32_t BASIC_TMP_SIZE = 80*1024;
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGatherTmpSize(const ge::AscNode &node);

using namespace testing;
using namespace ge::ascir_op;

class CalcGatherTmpSizeTest:public::testing::Test{
protected:
 void SetUp() override{}
 void TearDown() override{}
};

template<ge::DataType T1, ge::DataType T2>
void CreateGraph(ge::AscGraph &graph, ge::Expression &s0, ge::Expression &s2) {
 ge::Expression One = ge::Symbol(1);
 s0 = graph.CreateSizeVar("s0");
 auto s1 = graph.CreateSizeVar("s1");
 s2 = graph.CreateSizeVar("s2");

 auto z0 = graph.CreateAxis("z0", s0);
 auto z1 = graph.CreateAxis("z1", s1);
 auto z2 = graph.CreateAxis("z2", s2);

 ge::ascir_op::Data x1("x1", graph);
 ge::ascir_op::Data x2("x2", graph);
 ge::ascir_op::Gather gather("gather");
 ge::ascir_op::Store store("store");
 ge::ascir_op::Output y("y");

 x1.attr.sched.axis = {z0.id};
 x1.y.dtype = T1;
 *x1.y.axis = {z0.id};
 *x1.y.repeats = {s0};
 *x1.y.strides = {One};

 x2.attr.sched.axis = {z1.id, z2.id};
 x2.y.dtype = T2;
 *x2.y.axis = {z1.id, z2.id};
 *x2.y.repeats = {s1, s2};
 *x2.y.strides = {s2, One};

 gather.x1 = x1.y;
 gather.x2 = x2.y;
 gather.attr.sched.axis = {z1.id, z2.id};
 gather.y.dtype = T1;
 *gather.y.axis = {z1.id, z2.id};
 *gather.y.repeats = {s1, s2};
 *gather.y.strides = {s2, One};

 store.x = gather.y;
 store.attr.sched.axis = {z1.id, z2.id};
 store.y.dtype = T1;
 *store.y.axis = {z1.id, z2.id};
 *store.y.repeats = {s1, s2};
 *store.y.strides = {s2, One};

 y.x = store.y;
 y.attr.sched.axis = {z1.id, z2.id};
 y.y.dtype = T1;
 *y.y.axis = {z1.id, z2.id};
 *y.y.repeats = {s1, s2};
 *y.y.strides = {s2, One};
}
/**
 * @tc.name: CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size
 * @tc.number: CalcGatherTmpSizeTest_Test_001
 * @tc.desc: Test when node is valid then CalcGatherTmpSize returns correct size
*/
TEST_F(CalcGatherTmpSizeTest, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size)
{
 ge::AscGraph graph("test");
 Expression s0;
 Expression s2;
 CreateGraph<ge::DT_FLOAT, ge::DT_INT64>(graph, s0, s2);
 auto node = graph.FindNode("gather");
 std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSize(*node);
 ASSERT_EQ(result.size(), 1);
 Expression minTempSize = sym::Min(ge::Symbol(MAX_TMP_SIZE), sym::Max(ge::Symbol(BASIC_TMP_SIZE), ((ge::Symbol(12) * s2) + (ge::Symbol(4) * s0))));
 ASSERT_EQ(result[0]->size, minTempSize);
 ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
* @tc.name: CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size2
* @tc.number: CalcGatherTmpSizeTest_Test_002
* @tc.desc: Test when node is valid then CalcGatherTmpSize returns correct size
*/
TEST_F(CalcGatherTmpSizeTest, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size2)
{
 ge::AscGraph graph("test");
 Expression s0;
 Expression s2;
 CreateGraph<ge::DT_FLOAT, ge::DT_INT32>(graph, s0, s2);
 auto node = graph.FindNode("gather");
 std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSize(*node);
 ASSERT_EQ(result.size(), 1);
 Expression minTempSize = sym::Min(ge::Symbol(MAX_TMP_SIZE), sym::Max(ge::Symbol(BASIC_TMP_SIZE), ((ge::Symbol(4) * s0) + (ge::Symbol(4) * s2))));
 ASSERT_EQ(result[0]->size, minTempSize);
 ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

} // namespace ascir
} // namespace ge