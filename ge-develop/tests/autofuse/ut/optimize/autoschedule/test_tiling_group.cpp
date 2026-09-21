/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ascendc_ir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <iostream>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "graph_utils_ex.h"

#define private public
#include "autoschedule/autoschedule.h"
#undef private
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "autoschedule/tiling_group.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "schedule_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace optimize::autoschedule;

namespace optimize {
using namespace ge;
class TilingGroupUT : public testing::Test {
 protected:
  void SetUp() override {
  }
};
TEST_F(TilingGroupUT, get_group_type) {
  AxisGroup g0;
  EXPECT_EQ(TilingGroup::GetGroupType(g0), GroupType::GROUP_INVALID);

  AxisGroup g1;
  g1.x_group = {0};
  EXPECT_EQ(TilingGroup::GetGroupType(g1), GroupType::GROUP_X);

  AxisGroup g2;
  g2.y_group = {0, 1};
  EXPECT_EQ(TilingGroup::GetGroupType(g2), GroupType::GROUP_Y);

  AxisGroup g3;
  g3.r_group = {0, 2};
  EXPECT_EQ(TilingGroup::GetGroupType(g3), GroupType::GROUP_R);

  AxisGroup g4;
  g4.x_group = {0};
  g4.y_group = {1};
  EXPECT_EQ(TilingGroup::GetGroupType(g4), GroupType::GROUP_XY);

  AxisGroup g5;
  g5.y_group = {1};
  g5.r_group = {2};
  EXPECT_EQ(TilingGroup::GetGroupType(g5), GroupType::GROUP_YR);

  AxisGroup g6;
  g6.x_group = {0};
  g6.y_group = {1};
  g6.r_group = {2};
  EXPECT_EQ(TilingGroup::GetGroupType(g6), GroupType::GROUP_XYR);
}

TEST_F(TilingGroupUT, group_merge_y_y) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  // Y merge Y
  AxisGroup axes_group;
  axes_group.y_group = {0, 1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case1;
  y_case1.y_group = {0, 1};
  y_case1.axes_order = {0, 1};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group, y_case1));

  // Y merge Y different axis_order
  axes_group.y_group = {0, 1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.y_group = {1, 0};
  y_case2.axes_order = {1, 0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case2));

  // Y merge Y different axis_num
  axes_group.y_group = {0, 1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case3;
  y_case3.y_group = {0};
  y_case3.axes_order = {0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case3));
}

TEST_F(TilingGroupUT, group_merge_y_yr) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  // Y merge YR
  AxisGroup axes_group;
  axes_group.y_group = {0, 1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case1;
  y_case1.y_group = {0};
  y_case1.r_group = {1};
  y_case1.axes_order = {0, 1};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group, y_case1));

  // Y merge YR different axis_order
  axes_group.y_group = {0, 1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.y_group = {1};
  y_case2.r_group = {0};
  y_case2.axes_order = {1, 0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case2));

  // Y merge YR different axis_num
  axes_group.y_group = {0};
  axes_group.axes_order = {0};
  AxisGroup y_case3;
  y_case3.y_group = {0};
  y_case3.axes_order = {0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case3));
}

TEST_F(TilingGroupUT, group_merge_yr_y) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  // YR merge Y
  AxisGroup axes_group;
  axes_group.y_group = {0};
  axes_group.r_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case1;
  y_case1.y_group = {0, 1};
  y_case1.axes_order = {0, 1};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group, y_case1));

  // YR merge Y different axis_order
  axes_group.y_group = {0};
  axes_group.r_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.y_group = {1, 0};
  y_case2.axes_order = {1, 0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case2));

  // YR merge Y different axis_num
  axes_group.y_group = {0};
  axes_group.r_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case3;
  y_case3.y_group = {0};
  y_case3.axes_order = {0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case3));
}

TEST_F(TilingGroupUT, group_merge_y_xy) {
  // Y merge XY
  AxisGroup axes_group;
  axes_group.x_group = {0};
  axes_group.y_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case1;
  y_case1.y_group = {0, 1};
  y_case1.axes_order = {0, 1};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group, y_case1));

  AxisGroup axes_group2;
  axes_group2.x_group = {0};
  axes_group2.y_group = {1};
  axes_group2.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.y_group = {0, 1, 2};
  y_case2.axes_order = {0, 1, 2};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group2, y_case2));
}

TEST_F(TilingGroupUT, group_merge_xy_y) {
  // XY merge Y
  AxisGroup axes_group1;
  axes_group1.y_group = {0, 1};
  axes_group1.axes_order = {0, 1};
  AxisGroup y_case1;
  y_case1.x_group = {0};
  y_case1.y_group = {1};
  y_case1.axes_order = {0, 1};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group1, y_case1));

  AxisGroup axes_group2;
  axes_group2.y_group = {0, 1};
  axes_group2.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.x_group = {0};
  y_case2.y_group = {1, 2};
  y_case2.axes_order = {0, 1, 2};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group2, y_case2));
}
TEST_F(TilingGroupUT, group_merge_yr_yr) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  // YR merge YR
  AxisGroup axes_group;
  axes_group.y_group = {0};
  axes_group.r_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case1;
  y_case1.y_group = {0};
  y_case1.r_group = {1};
  y_case1.axes_order = {0, 1};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group, y_case1));

  // YR merge YR different axis_order
  axes_group.y_group = {0};
  axes_group.r_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.y_group = {0};
  y_case2.r_group = {1};
  y_case2.axes_order = {1, 0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case2));
  EXPECT_TRUE(!axes_group.ToString().empty());

  // YR merge Y different axis_num
  axes_group.y_group = {0};
  axes_group.r_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case3;
  y_case3.y_group = {0};
  y_case3.r_group = {1, 2};
  y_case3.axes_order = {0, 1, 2};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case3));
}

TEST_F(TilingGroupUT, group_merge_concat_like) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  AxisGroup axes_group;
  axes_group.y_group = {0};
  axes_group.n_group = {1, 2};
  axes_group.axes_order = {0};
  AxisGroup y_case1;
  y_case1.y_group = {0, 1, 2};
  y_case1.axes_order = {0, 1, 2};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group, y_case1));

  // no loop axis
  AxisGroup y_case2;
  y_case2.y_group = {2};
  y_case2.axes_order = {0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case2));
}

TEST_F(TilingGroupUT, axis_group_is_empty) {
  AxisGroup axis_group1;
  axis_group1.x_group = {0};
  EXPECT_FALSE(axis_group1.IsEmpty());

  AxisGroup axis_group2;
  axis_group2.y_group = {0};
  EXPECT_FALSE(axis_group2.IsEmpty());

  AxisGroup axis_group3;
  axis_group3.r_group = {0};
  EXPECT_FALSE(axis_group3.IsEmpty());

  AxisGroup axis_group4;
  axis_group4.n_group = {0};
  EXPECT_FALSE(axis_group4.IsEmpty());

  AxisGroup axis_group5;
  EXPECT_TRUE(axis_group5.IsEmpty());
}


TEST_F(TilingGroupUT, gen_non_first_axis_concat_axis_group) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto si = graph.CreateSizeVar("si");
  auto sj = graph.CreateSizeVar("sj");
  auto sk = graph.CreateSizeVar("sk");
  auto s2 = graph.CreateSizeVar("s2");

  auto so = graph.CreateSizeVar("so");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto zo = graph.CreateAxis("zo", so);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, zo.id, z2.id};

  ge::ascir_op::Data data_j("data_j", graph);
  data_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.y.dtype = ge::DT_FLOAT16;
  *data_j.y.axis = {z0.id, z1.id, zo.id, z2.id};

  ge::ascir_op::Data data_k("data_k", graph);
  data_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.y.dtype = ge::DT_FLOAT16;
  *data_k.y.axis = {z0.id, z1.id, zo.id, z2.id};

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.repeats = {s0, s1, si, s2};
  *load_i.y.strides = {s1 * si * s2, si * s2, s2, ge::ops::One};

  ge::ascir_op::Load load_j("load_j");
  load_j.x = data_j.y;
  load_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.repeats = {s0, s1, sj, s2};
  *load_j.y.strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One};

  ge::ascir_op::Load load_k("load_k");
  load_k.x = data_k.y;
  load_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.repeats = {s0, s1, sk, s2};
  *load_k.y.strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};

  ge::ascir_op::Concat concat("concat");

  concat.x = {load_i.y, load_j.y, load_k.y};
  concat.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *concat.y.repeats = {s0, s1, so, s2};
  *concat.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store.y.repeats = {s0, s1, so, s2};
  *store.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, zo.id, z2.id};

  auto concat_node = graph.FindNode("concat");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenConcatTilingGroup(*concat_node, res), 0);
  std::vector<int64_t> golden_y{0, 1};
  EXPECT_EQ(res.y_group, golden_y);
  std::vector<size_t> golden_order{0, 1};
  EXPECT_EQ(res.axes_order, golden_order);
  std::vector<int64_t> golden_n{3, 2, 3, 3, 3};
  EXPECT_EQ(res.n_group, golden_n);
}

TEST_F(TilingGroupUT, transpose_axis_group1) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id, z3.id};

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load_i.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  *load_i.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_i.y.repeats = {s0, s1, s2, s3};
  *load_i.y.strides = {s1 * s2 * s3, s2 * s3, s3, ge::ops::One};

  ge::ascir_op::Transpose transpose("transpose");

  transpose.x = {load_i.y};
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  transpose.y.dtype = ge::DT_FLOAT16;
  *transpose.y.axis = {z0.id, z3.id, z1.id, z2.id};
  *transpose.y.repeats = {s0, s3, s1, s2};
  *transpose.y.strides = {s1 * s2 * s3, s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Store store("store");
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z3.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z3.id, z1.id, z2.id};
  *store.y.repeats = {s0, s3, s1, s2};
  *store.y.strides = {s1 * s2 * s3, s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z3.id, z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z3.id, z1.id, z2.id};
  // (0,1,2,3) => (0,3,1,2) 分组 ((3),(0,1,2))
  auto transpose_node = graph.FindNode("transpose");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenTransposeTilingGroup(*transpose_node, res), ge::SUCCESS);
  std::vector<int64_t> golden_x{3};
  EXPECT_EQ(res.x_group, golden_x);
  std::vector<int64_t> golden_y{0, 1, 2};
  EXPECT_EQ(res.y_group, golden_y);
  std::vector<size_t> golden_order{3, 0, 1, 2};
  EXPECT_EQ(res.axes_order, golden_order);
}

TEST_F(TilingGroupUT, transpose_axis_group2) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id, z3.id};

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load_i.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  *load_i.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_i.y.repeats = {s0, s1, s2, s3};
  *load_i.y.strides = {s1 * s2 * s3, s2 * s3, s3, ge::ops::One};

  ge::ascir_op::Transpose transpose("transpose");
  transpose.x = {load_i.y};
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  transpose.y.dtype = ge::DT_FLOAT16;
  *transpose.y.axis = {z3.id, z2.id, z1.id, z0.id};
  *transpose.y.repeats = {s3, s2, s1, s0};
  *transpose.y.strides = {s2 * s1 * s0, s1 * s0, s0, ge::ops::One};

  ge::ascir_op::Store store("store");
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z3.id, z2.id, z1.id, z0.id};
  *store.y.repeats = {s3, s2, s1, s0};
  *store.y.strides = {s2 * s1 * s0, s1 * s0, s0, ge::ops::One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z3.id, z2.id, z1.id, z0.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z3.id, z2.id, z1.id, z0.id};
  // (0,1,2,3) => (3,2,1,0) 分组 ((2,3),(1,0))
  auto transpose_node = graph.FindNode("transpose");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenTransposeTilingGroup(*transpose_node, res), ge::SUCCESS);
  std::vector<int64_t> golden_x{2, 3};
  EXPECT_EQ(res.x_group, golden_x);
  std::vector<int64_t> golden_y{1, 0};
  EXPECT_EQ(res.y_group, golden_y);
  std::vector<size_t> golden_order{2, 3, 1, 0};
  EXPECT_EQ(res.axes_order, golden_order);
}

TEST_F(TilingGroupUT, transpose_axis_group3) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  load_i.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  *load_i.y.axis = {z0.id, z1.id, z2.id};
  *load_i.y.repeats = {s0, s1, s2};
  *load_i.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Transpose transpose("transpose");
  transpose.x = {load_i.y};
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose.y.dtype = ge::DT_FLOAT16;
  *transpose.y.axis = {z1.id, z0.id, z2.id};
  *transpose.y.repeats = {s1, s0, s2};
  *transpose.y.strides = {s2 * s0, s2, ge::ops::One};

  ge::ascir_op::Store store("store");
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, };
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z1.id, z0.id, z2.id};
  *store.y.repeats = {s1, s0, s2};
  *store.y.strides = {s2 * s0, s2, ge::ops::One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z1.id, z0.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z1.id, z0.id, z2.id};
  // (0,1,2) => (1,0,2) 分组 ((1),(0),(),(2))
  auto transpose_node = graph.FindNode("transpose");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenTransposeTilingGroup(*transpose_node, res), ge::SUCCESS);
  std::vector<int64_t> golden_x{1};
  EXPECT_EQ(res.x_group, golden_x);
  std::vector<int64_t> golden_y{0};
  EXPECT_EQ(res.y_group, golden_y);
  std::vector<int64_t> golden_n{2};
  EXPECT_EQ(res.n_group, golden_n);
}

TEST_F(TilingGroupUT, transpose_axis_group5) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("512");
  auto s1 = graph.CreateSizeVar("32");
  auto s2 = graph.CreateSizeVar("32");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  load_i.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  *load_i.y.axis = {z0.id, z1.id, z2.id};
  *load_i.y.repeats = {s0, s1, s2};
  *load_i.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Transpose transpose("transpose");
  transpose.x = {load_i.y};
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose.y.dtype = ge::DT_FLOAT16;
  *transpose.y.axis = {z0.id, z2.id, z1.id};
  *transpose.y.repeats = {s0, s2, s1};
  *transpose.y.strides = {s2 * s1, s1, ge::ops::One};

  ge::ascir_op::Store store("store");
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, };
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z2.id, z1.id};
  *store.y.repeats = {s0, s2, s1};
  *store.y.strides = {s2 * s1, s1, ge::ops::One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z1.id, z0.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z2.id, z1.id};
  // (0,1,2) => (0,2,1) 分组 ((2),(1),(),(0))
  auto transpose_node = graph.FindNode("transpose");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenTransposeTilingGroup(*transpose_node, res), ge::SUCCESS);
  std::vector<int64_t> golden_x{2};
  EXPECT_EQ(res.x_group, golden_x);
  std::vector<int64_t> golden_y{0, 1};
  EXPECT_EQ(res.y_group, golden_y);
  std::vector<int64_t> golden_n{};
  EXPECT_EQ(res.n_group, golden_n);
}

TEST_F(TilingGroupUT, gen_first_axis_concat_axis_group) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar(1);
  auto s1 = graph.CreateSizeVar(1);
  auto si = graph.CreateSizeVar("si");
  auto sj = graph.CreateSizeVar("sj");
  auto sk = graph.CreateSizeVar("sk");
  auto s2 = graph.CreateSizeVar("s2");

  auto so = graph.CreateSizeVar("so");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto zo = graph.CreateAxis("zo", so);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Data data_j("data_j", graph);
  data_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.y.dtype = ge::DT_FLOAT16;
  *data_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Data data_k("data_k", graph);
  data_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.y.dtype = ge::DT_FLOAT16;
  *data_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.repeats = {s0, s1, si, s2};
  *load_i.y.strides = {s1 * si * s2, si * s2, s2, ge::ops::One};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Load load_j("load_j");
  load_j.x = data_j.y;
  load_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.repeats = {s0, s1, sj, s2};
  *load_j.y.strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One};
  load_j.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Load load_k("load_k");
  load_k.x = data_k.y;
  load_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.repeats = {s0, s1, sk, s2};
  *load_k.y.strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};
  load_k.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Concat concat("concat");

  concat.x = {load_i.y, load_j.y, load_k.y};
  concat.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *concat.y.repeats = {s0, s1, so, s2};
  *concat.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  concat.attr.api.compute_type = ComputeType::kComputeConcat;

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store.y.repeats = {s0, s1, so, s2};
  *store.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;

  auto concat_node = graph.FindNode("concat");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenConcatTilingGroup(*concat_node, res), 0);
  std::vector<int64_t> golden_y{0, 1};
  std::vector<size_t> golden_order{0, 1};
  EXPECT_EQ(res.y_group, golden_y);
  EXPECT_EQ(res.axes_order, golden_order);
  EXPECT_TRUE(!res.n_group.empty());
}

TEST_F(TilingGroupUT, t_merge_tr) {
  AxisGroup axes_group;
  axes_group.y_group = {0};
  axes_group.r_group = {1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.y_group = {0};
  y_case2.r_group = {1};
  y_case2.axes_order = {1, 0};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case2));
  EXPECT_TRUE(!axes_group.ToString().empty());
}

TEST_F(TilingGroupUT, y_merge_xy) {
  AxisGroup axes_group;
  axes_group.y_group = {0, 1};
  axes_group.axes_order = {0, 1};
  AxisGroup y_case2;
  y_case2.x_group = {0};
  y_case2.y_group = {1};
  y_case2.axes_order = {0, 1};
  EXPECT_TRUE(TilingGroup::MergeAxesGroup(axes_group, y_case2, true));

  AxisGroup xy_1;
  xy_1.x_group = {0};
  xy_1.y_group = {1};
  xy_1.axes_order = {1, 0};
  AxisGroup y_1;
  y_1.y_group = {0, 1};
  y_1.axes_order = {0, 1};
  EXPECT_FALSE(TilingGroup::MergeAxesGroup(axes_group, y_case2, true));
}

TEST_F(TilingGroupUT, gencase_success) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar(1);
  auto s1 = graph.CreateSizeVar(1);
  auto si = graph.CreateSizeVar("si");
  auto sj = graph.CreateSizeVar("sj");
  auto sk = graph.CreateSizeVar("sk");
  auto s2 = graph.CreateSizeVar("s2");

  auto so = graph.CreateSizeVar("so");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto zo = graph.CreateAxis("zo", so);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Data data_j("data_j", graph);
  data_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.y.dtype = ge::DT_FLOAT16;
  *data_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Data data_k("data_k", graph);
  data_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.y.dtype = ge::DT_FLOAT16;
  *data_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.repeats = {s0, s1, si, s2};
  *load_i.y.strides = {s1 * si * s2, si * s2, s2, ge::ops::One};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Load load_j("load_j");
  load_j.x = data_j.y;
  load_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.repeats = {s0, s1, sj, s2};
  *load_j.y.strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One};
  load_j.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Load load_k("load_k");
  load_k.x = data_k.y;
  load_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.repeats = {s0, s1, sk, s2};
  *load_k.y.strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};
  load_k.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Concat concat("concat");

  concat.x = {load_i.y, load_j.y, load_k.y};
  concat.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *concat.y.repeats = {s0, s1, so, s2};
  *concat.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  concat.attr.api.compute_type = ComputeType::kComputeConcat;

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store.y.repeats = {s0, s1, so, s2};
  *store.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;

  optimize::autoschedule::AxisGroup group;
  int32_t res = GenAscGraphAxisGroup(graph, group);
  ASSERT_EQ(res, 0);
}

TEST_F(TilingGroupUT, hint_graph_gencase_success) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar(1);
  auto s1 = graph.CreateSizeVar(1);
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id, z2.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");

  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id, z2.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;

  optimize::autoschedule::AxisGroup group;
  ASSERT_EQ(graph.GetGraphType(), ge::AscGraphType::kHintGraph);
  int32_t res = GenAscGraphAxisGroup(graph, group);
  ASSERT_EQ(res, 0);
}

TEST_F(TilingGroupUT, hint_graph_1axis_gencase_success) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  ge::ascir_op::Max max("max");
  max.x = add.y;
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  max.y.dtype = ge::DT_FLOAT16;
  *max.y.axis = {z0.id};
  *max.y.strides = {ge::ops::One};

  ge::ascir_op::Store store("store");
  store.x = max.y;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;

  optimize::autoschedule::AxisGroup group;
  ASSERT_EQ(graph.GetGraphType(), ge::AscGraphType::kHintGraph);
  int32_t res = GenAscGraphAxisGroup(graph, group);
  ASSERT_EQ(res, 0);
}

TEST_F(TilingGroupUT, hint_graph_2axis_gencase_success) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  ge::ascir_op::Max max("max");
  max.x = add.y;
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  max.y.dtype = ge::DT_FLOAT16;
  *max.y.axis = {z0.id, z1.id};
  *max.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store("store");
  store.x = max.y;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;

  optimize::autoschedule::AxisGroup group;
  ASSERT_EQ(graph.GetGraphType(), ge::AscGraphType::kHintGraph);
  int32_t res = GenAscGraphAxisGroup(graph, group);
  ASSERT_EQ(res, 0);
}

TEST_F(TilingGroupUT, gencase_cannot_fuse) {
  AxisGroup lhs;
  lhs.y_group = {0, 1, 2, 3};
  lhs.axes_order = {0, 1, 2, 3};
  AxisGroup rhs;
  rhs.y_group = {0, 1, 2};
  rhs.axes_order = {0, 1, 2};
  AxisGroup res;
  ASSERT_FALSE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, canfuse_disorder_y_y) {
  AxisGroup lhs;
  lhs.y_group = {0, 1, 2, 3};
  lhs.axes_order = {0, 1, 2, 3};
  AxisGroup rhs;
  rhs.y_group = {0, 3, 1, 2};
  rhs.axes_order = {0, 1, 2};
  AxisGroup res;
  ASSERT_TRUE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, cannot_fuse_disorder_y_y) {
  AxisGroup lhs;
  lhs.y_group = {0, 1, 2, 3};
  lhs.axes_order = {0, 1, 2, 3};
  AxisGroup rhs;
  rhs.y_group = {0, 1, 2, 4};
  rhs.axes_order = {0, 1, 2, 3};
  AxisGroup res;
  ASSERT_FALSE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, canfuse_disorder_y_yr) {
  AxisGroup lhs;
  lhs.y_group = {0, 1, 2, 3};
  lhs.axes_order = {0, 1, 2, 3};
  AxisGroup rhs;
  rhs.y_group = {2, 1};
  rhs.r_group = {0, 3};
  rhs.axes_order = {2, 1, 0, 3};
  AxisGroup res;
  ASSERT_TRUE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, cannot_fuse_disorder_y_yr) {
  AxisGroup lhs;
  lhs.y_group = {0, 1, 2, 3};
  lhs.axes_order = {0, 1, 2, 3};
  AxisGroup rhs;
  rhs.y_group = {2, 1};
  rhs.r_group = {0, 4};
  rhs.axes_order = {2, 1, 0, 3};
  AxisGroup res;
  ASSERT_FALSE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, canfuse_disorder_yr_y) {
  AxisGroup lhs;
  lhs.y_group = {2, 1};
  lhs.r_group = {0, 3};
  lhs.axes_order = {2, 1, 0, 3};
  AxisGroup rhs;
  rhs.y_group = {0, 1, 2, 3};
  rhs.axes_order = {0, 1, 2, 3};
  AxisGroup res;
  ASSERT_TRUE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, cannot_fuse_disorder_yr_y) {
  AxisGroup lhs;
  lhs.y_group = {2, 1};
  lhs.r_group = {0, 3};
  lhs.axes_order = {2, 1, 0, 3};
  AxisGroup rhs;
  rhs.y_group = {0, 1, 2, 4};
  rhs.axes_order = {0, 1, 2, 3};
  AxisGroup res;
  ASSERT_FALSE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, canfuse_disorder_yr_yr) {
  AxisGroup lhs;
  lhs.y_group = {2, 1};
  lhs.r_group = {0, 3};
  lhs.axes_order = {2, 1, 0, 3};
  AxisGroup rhs;
  rhs.y_group = {1, 2};
  rhs.r_group = {3, 0};
  rhs.axes_order = {0, 1, 2, 3};
  AxisGroup res;
  ASSERT_TRUE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, cannot_fuse_disorder_yr_yr) {
  AxisGroup lhs;
  lhs.y_group = {0, 1, 3};
  lhs.r_group = {2};
  lhs.axes_order = {0, 1, 3, 2};
  AxisGroup rhs;
  rhs.y_group = {0, 1, 2};
  lhs.r_group = {3};
  rhs.axes_order = {0, 1, 2, 3};
  AxisGroup res;
  ASSERT_FALSE(CanMergeAxisGroup(lhs, rhs, res));
}

TEST_F(TilingGroupUT, gencase_canfuse) {
  AxisGroup lhs;
  lhs.y_group = {0, 1, 2, 3};
  lhs.axes_order = {0, 1, 2, 3};
  AxisGroup rhs;
  rhs.y_group = {0, 1};
  rhs.r_group = {2, 3};
  rhs.axes_order = {0, 1, 2, 3};
  AxisGroup res;
  ASSERT_TRUE(CanMergeAxisGroup(lhs, rhs, res));
  std::vector<int64_t> golden_y{0, 1};
  std::vector<int64_t> golden_r{2, 3};
  std::vector<size_t> golden_order{0, 1, 2, 3};
  ASSERT_EQ(rhs.y_group, golden_y);
  ASSERT_EQ(rhs.r_group, golden_r);
  ASSERT_EQ(rhs.axes_order, golden_order);
}

TEST_F(TilingGroupUT, get_loop_strides_0dim) {
  ge::AscGraph graph("test_graph");
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  ASSERT_TRUE(res_strides.empty());
}

TEST_F(TilingGroupUT, get_loop_strides_1dim_not1) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{ge::Symbol(1)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_1dim_is1) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{ge::Symbol(0)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_2dim_no1) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{s1, ge::Symbol(1)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_2dim_last1) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{s1, ge::Symbol(0)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_2dim_first1) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar(1);
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{ge::Symbol(0), ge::Symbol(1)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_2dim_both1) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar(1);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{ge::Symbol(0), ge::Symbol(0)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_case1) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id, z3.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{s2, ge::Symbol(0), ge::Symbol(1), ge::Symbol(0)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_case2) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar(1);
  auto s3 = graph.CreateSizeVar(1);
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar(1);
  auto s6 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{s1 * s4,       s4,           ge::Symbol(0), ge::Symbol(0), ge::Symbol(1),
                                      ge::Symbol(0), ge::Symbol(0)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, get_loop_strides_case3) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar(1);
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar(1);
  auto s4 = graph.CreateSizeVar(1);
  auto s5 = graph.CreateSizeVar(1);
  auto s6 = graph.CreateSizeVar("s6");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);
  ge::ascir_op::Data data_i("data_i", graph);
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  data_i.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  *load_i.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  load_i.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Add add("add");
  add.x1 = load_i.y;
  add.x2 = load_i.y;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  add.attr.api.compute_type = ComputeType::kComputeElewise;

  std::vector<Expression> res_strides;
  std::vector<Expression> exp_strides{ge::Symbol(0), s2 * s6,       s6,           ge::Symbol(0),
                                      ge::Symbol(0), ge::Symbol(0), ge::Symbol(1)};
  auto node = graph.FindNode("add");
  ASSERT_EQ(ScheduleUtils::GetReduceInputStrides(*node, res_strides), ge::SUCCESS);
  for (size_t i = 0; i < res_strides.size(); i++) {
    ASSERT_EQ(res_strides[i], exp_strides[i]);
  }
}

TEST_F(TilingGroupUT, gen_first_axis_split_axis_group) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar(1);
  auto s1 = graph.CreateSizeVar(1);
  auto si = graph.CreateSizeVar("si");
  auto sj = graph.CreateSizeVar("sj");
  auto sk = graph.CreateSizeVar("sk");
  auto s2 = graph.CreateSizeVar("s2");

  auto so = graph.CreateSizeVar("so");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto zo = graph.CreateAxis("zo", so);

  ge::ascir_op::Data data("data", graph);
  data.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load.y.repeats = {s0, s1, so, s2};
  *load.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Split split("split");
  split.InstanceOutputy(3U);
  split.x = load.y;
  split.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  split.attr.api.compute_type = ComputeType::kComputeSplit;

  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1.id, zo.id, z2.id};
  *split.y[0].repeats = {s0, s1, si, s2};
  *split.y[0].strides = {s1 * si * s2, si * s2, s2, ge::ops::One};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1.id, zo.id, z2.id};
  *split.y[1].repeats = {s0, s1, sj, s2};
  *split.y[1].strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One}; 

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1.id, zo.id, z2.id};
  *split.y[2].repeats = {s0, s1, sk, s2};
  *split.y[2].strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};   

  ge::ascir_op::Store store0("store0");
  store0.x = split.y[0];
  store0.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store0.y.repeats = {s0, s1, si, s2};
  *store0.y.strides = {s1 * si * s2, si * s2, s2, ge::ops::One};
  store0.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Store store1("store1");
  store1.x = split.y[1];
  store1.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store1.y.repeats = {s0, s1, sj, s2};
  *store1.y.strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  
  ge::ascir_op::Store store2("store2");
  store2.x = split.y[2];
  store2.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store2.y.repeats = {s0, s1, sk, s2};
  *store2.y.strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};
  store2.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y0("y0");
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  
  ge::ascir_op::Output y2("y2");
  y2.x = store2.y;
  y2.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y2.y.dtype = ge::DT_FLOAT16;
  *y2.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y2.attr.api.compute_type = ComputeType::kComputeInvalid;  

  auto split_node = graph.FindNode("split");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenSplitTilingGroup(*split_node, res), 0);
  std::vector<int64_t> golden_y{0, 1, 3, 2};
  std::vector<size_t> golden_order{0, 1, 2, 3};
  EXPECT_EQ(res.y_group, golden_y);
  EXPECT_EQ(res.axes_order, golden_order);
  EXPECT_TRUE(res.n_group.empty());
}


TEST_F(TilingGroupUT, gen_not_first_axis_split_axis_group) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto si = graph.CreateSizeVar("si");
  auto sj = graph.CreateSizeVar("sj");
  auto sk = graph.CreateSizeVar("sk");
  auto s2 = graph.CreateSizeVar("s2");

  auto so = graph.CreateSizeVar("so");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto zo = graph.CreateAxis("zo", so);

  ge::ascir_op::Data data("data", graph);
  data.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load.y.repeats = {s0, s1, so, s2};
  *load.y.strides = {s1 * so * s2, so * s2, s2, ge::ops::One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  ge::ascir_op::Split split("split");
  split.InstanceOutputy(3U);
  split.x = load.y;
  split.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  split.attr.api.compute_type = ComputeType::kComputeSplit;

  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1.id, zo.id, z2.id};
  *split.y[0].repeats = {s0, s1, si, s2};
  *split.y[0].strides = {s1 * si * s2, si * s2, s2, ge::ops::One};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1.id, zo.id, z2.id};
  *split.y[1].repeats = {s0, s1, sj, s2};
  *split.y[1].strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One}; 

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1.id, zo.id, z2.id};
  *split.y[2].repeats = {s0, s1, sk, s2};
  *split.y[2].strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};   

  ge::ascir_op::Store store0("store0");
  store0.x = split.y[0];
  store0.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store0.y.repeats = {s0, s1, si, s2};
  *store0.y.strides = {s1 * si * s2, si * s2, s2, ge::ops::One};
  store0.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Store store1("store1");
  store1.x = split.y[1];
  store1.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store1.y.repeats = {s0, s1, sj, s2};
  *store1.y.strides = {s1 * sj * s2, sj * s2, s2, ge::ops::One};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  
  ge::ascir_op::Store store2("store2");
  store2.x = split.y[2];
  store2.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store2.y.repeats = {s0, s1, sk, s2};
  *store2.y.strides = {s1 * sk * s2, sk * s2, s2, ge::ops::One};
  store2.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y0("y0");
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  
  ge::ascir_op::Output y2("y2");
  y2.x = store2.y;
  y2.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y2.y.dtype = ge::DT_FLOAT16;
  *y2.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y2.attr.api.compute_type = ComputeType::kComputeInvalid;  

  auto split_node = graph.FindNode("split");
  AxisGroup res;
  EXPECT_EQ(TilingGroup::GenSplitTilingGroup(*split_node, res), 0);
  std::vector<int64_t> golden_y{0, 1};
  EXPECT_EQ(res.y_group, golden_y);
  std::vector<size_t> golden_order{0, 1};
  EXPECT_EQ(res.axes_order, golden_order);
  std::vector<int64_t> golden_n{3, 2, 3, 3, 3};
  EXPECT_EQ(res.n_group, golden_n);
}

TEST_F(TilingGroupUT, ScalarToStoreGenSuccess) {
  ge::AscGraph graph("test_graph");
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Scalar scalar("scalar", graph);
  scalar.attr.api.compute_type = ComputeType::kComputeInvalid;
  scalar.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Store store0("store0");
  store0.x = scalar.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, ge::ops::One};
  store0.attr.api.compute_type = ComputeType::kComputeStore;

  ge::ascir_op::Output y0("y0");
  y0.x = store0.y;
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  AxisGroup group;
  EXPECT_EQ(GenAscGraphAxisGroup(graph, group), 0);
  ASSERT_TRUE(!group.IsEmpty());

  std::vector<int64_t> golden_y{0, 1};
  EXPECT_EQ(group.y_group, golden_y);
  std::vector<size_t> golden_order{0, 1};
  EXPECT_EQ(group.axes_order, golden_order);
}
}  // namespace optimize
