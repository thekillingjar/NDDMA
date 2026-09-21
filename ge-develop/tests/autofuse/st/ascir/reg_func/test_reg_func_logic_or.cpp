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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLogicalOrTmpSize(const ge::AscNode &node);

using namespace testing;

class CalcLogicalOrTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

/**
 * @tc.name:CalcLogicalOrTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcDivTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcLogicalOrTmpSize returns correct size
 */
TEST_F(CalcLogicalOrTmpSizeTest, CalcLogicalOrTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
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
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcLogicalOrTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, ge::Symbol(8192));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge