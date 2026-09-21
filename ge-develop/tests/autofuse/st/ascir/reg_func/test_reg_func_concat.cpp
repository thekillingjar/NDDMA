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
 extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcConcatTmpSize(const ge::AscNode &node);
 
 using namespace testing;
 using namespace ge::ascir_op;
 
 class CalcConcatTmpSizeTest:public::testing::Test{
 protected:
     void SetUp() override{}
     void TearDown() override{}
 };
 template<ge::DataType T>
 void CreateGraph(ge::AscGraph &graph, Expression &s1, Expression &s2) {
     ge::Expression One = ge::Symbol(1);
     auto s0 = graph.CreateSizeVar("s0");
     s1 = graph.CreateSizeVar("s1");
     s2 = graph.CreateSizeVar("s2");
 
     auto z0 = graph.CreateAxis("z0", s0);
     auto zo = graph.CreateAxis("zo", s1 + s2);
     auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
     auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);
 
     Data x1("x1", graph);
     Data x2("x2", graph);
     Load load1("load1");
     Load load2("load2");
     ge::ascir_op::Concat concat("concat");
     Store store("store");
     Output y("y");
 
     x1.attr.sched.axis = {z0.id, zo_s_0.id};
     x1.y.dtype = T;
     *x1.y.axis = {z0.id, zo_s_0.id};
     *x1.y.repeats = {s0, s1};
     *x1.y.strides = {s1, One};
 
     x2.attr.sched.axis = {z0.id, zo_s_1.id};
     x2.y.dtype = T;
     *x2.y.axis = {z0.id, zo_s_1.id};
     *x2.y.repeats = {s0, s2};
     *x2.y.strides = {s2, One};
 
     load1.x = x1.y;
     load1.attr.sched.axis = {z0.id, zo_s_0.id};
     load1.y.dtype = T;
     *load1.y.axis = {z0.id, zo_s_0.id};
     *load1.y.repeats = {s0, s1};
     *load1.y.strides = {s1, One};
 
     load2.x = x2.y;
     load2.attr.sched.axis = {z0.id, zo_s_1.id};
     load2.y.dtype = T;
     *load2.y.axis = {z0.id, zo_s_1.id};
     *load2.y.repeats = {s0, s2};
     *load2.y.strides = {s2, One};
 
     concat.x = {load1.y, load2.y};
     concat.attr.sched.axis = {z0.id, zo.id};
     concat.y.dtype = T;
     *concat.y.axis = {z0.id, zo.id};
     *concat.y.repeats = {s0, s1 + s2};
     *concat.y.strides = {s1+s2, One};
 
     store.x = concat.y;
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
 }
 
 template<ge::DataType T>
 void CreateGraphNotLastAxis(ge::AscGraph &graph, Expression &s1, Expression &s2) {
     ge::Expression One = ge::Symbol(1);
     auto s0 = graph.CreateSizeVar("s0");
     s1 = graph.CreateSizeVar("s1");
     s2 = graph.CreateSizeVar("s2");
 
     auto z0 = graph.CreateAxis("z0", s0);
     auto zo = graph.CreateAxis("zo", s1 + s2);
     auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
     auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);
 
     Data x1("x1", graph);
     Data x2("x2", graph);
     Load load1("load1");
     Load load2("load2");
     ge::ascir_op::Concat concat("concat");
     Store store("store");
     Output y("y");
 
     x1.attr.sched.axis = {z0.id, zo_s_0.id};
     x1.y.dtype = T;
     *x1.y.axis = {z0.id, zo_s_0.id};
     *x1.y.repeats = {s0, s1};
     *x1.y.strides = {s1, One};
 
     x2.attr.sched.axis = {z0.id, zo_s_1.id};
     x2.y.dtype = T;
     *x2.y.axis = {z0.id, zo_s_1.id};
     *x2.y.repeats = {s2, s1};
     *x2.y.strides = {s1, One};
 
     load1.x = x1.y;
     load1.attr.sched.axis = {z0.id, zo_s_0.id};
     load1.y.dtype = T;
     *load1.y.axis = {z0.id, zo_s_0.id};
     *load1.y.repeats = {s0, s1};
     *load1.y.strides = {s1, One};
 
     load2.x = x2.y;
     load2.attr.sched.axis = {z0.id, zo_s_1.id};
     load2.y.dtype = T;
     *load2.y.axis = {z0.id, zo_s_1.id};
     *load2.y.repeats = {s2, s1};
     *load2.y.strides = {s1, One};
 
     concat.x = {load1.y, load2.y};
     concat.attr.sched.axis = {z0.id, zo.id};
     concat.y.dtype = T;
     *concat.y.axis = {z0.id, zo.id};
     *concat.y.repeats = {s0 + s2, s1};
     *concat.y.strides = {s1, One};
 
     store.x = concat.y;
     store.attr.sched.axis = {z0.id, zo.id};
     store.y.dtype = T;
 
     *store.y.axis = {z0.id, zo.id};
     *store.y.repeats = {s0 + s2, s1};
     *store.y.strides = {s1, One};
 
     y.x = store.y;
     y.attr.sched.axis = {z0.id, zo.id};
     y.y.dtype = T;
     *y.y.axis = {z0.id, zo.id};
     *y.y.repeats = {s0 + s2, s1};
     *y.y.strides = {s1, One};
 }
 
 /**
  * @tc.name: CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
  * @tc.number: CalcConcatTmpSize_Test_001
  * @tc.desc: Test when node is valid then CalcConcatTmpSize returns correct size
  */
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size4)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraph<ge::DT_FLOAT>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
      sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(2) * sym::Max(ge::Symbol(0), s2), 8) + ge::Symbol(29)) * ge::Symbol(16) * ge::Symbol(2) * ge::Symbol(4)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }
 
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_NotLastAxis_Size4)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraphNotLastAxis<ge::DT_FLOAT>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
      sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(2) * sym::Max(ge::Symbol(0), sym::Mul(s2, s1)), 8) + ge::Symbol(29)) * ge::Symbol(16) * ge::Symbol(2) * ge::Symbol(4)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }
 
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size2)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraph<ge::DT_FLOAT16>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
       sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(2) * sym::Max(ge::Symbol(0), s2), 16) + ge::Symbol(45)) * ge::Symbol(16) * ge::Symbol(2) * ge::Symbol(2)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }
 
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_NotLastAxis_Size2)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraphNotLastAxis<ge::DT_FLOAT16>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
       sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(2) * sym::Max(ge::Symbol(0), sym::Mul(s2, s1)), 16) + ge::Symbol(45)) * ge::Symbol(16) * ge::Symbol(2) * ge::Symbol(2)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }
 
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size1)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraph<ge::DT_INT8>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
       sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(2) * sym::Max(ge::Symbol(0), s2), 32) + ge::Symbol(93)) * ge::Symbol(16) * ge::Symbol(3) * ge::Symbol(1)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }
 
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_NotLastAxis_Size1)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraphNotLastAxis<ge::DT_INT8>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
       sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(2) * sym::Max(ge::Symbol(0), sym::Mul(s2, s1)), 32) + ge::Symbol(93)) * ge::Symbol(16) * ge::Symbol(3) * ge::Symbol(1)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }
 
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size8)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraph<ge::DT_INT64>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
       sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(4) * sym::Max(ge::Symbol(0), s2), 8) + ge::Symbol(29)) * ge::Symbol(16) * ge::Symbol(2) * ge::Symbol(4)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }
 
 TEST_F(CalcConcatTmpSizeTest, CalcConcatTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_NotLastAxis_Size8)
 {
     ge::AscGraph graph("test");
     Expression s1;
     Expression s2;
     CreateGraphNotLastAxis<ge::DT_INT64>(graph, s1, s2);
     std::shared_ptr<ge::AscNode> node = graph.FindNode("concat");
     std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcConcatTmpSize(*node);
     ASSERT_EQ(result.size(), 1);
     Expression minTempSize = sym::Min(ge::Symbol(65536),
       sym::Max(ge::Symbol(16384), (sym::Align(ge::Symbol(4) * sym::Max(ge::Symbol(0), sym::Mul(s2, s1)), 8) + ge::Symbol(29)) * ge::Symbol(16) * ge::Symbol(2) * ge::Symbol(4)));
     ASSERT_EQ(result[0]->size, minTempSize);
     ASSERT_EQ(result[0]->life_time_axis_id, -1);
 }

 } // namespace ascir
 } // namespace ge