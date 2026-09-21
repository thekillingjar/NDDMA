/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "gtest/gtest.h"
#include "ascendc_ir_def.h"
#define private public
#include "autoschedule/alignment_handler.h"
#include "partition/vector_func_partitioner.h"
#undef private
#include "ascir_ops_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"
#include "optimize.h"
#include "template/brc_inline_template_v2.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace optimize::autoschedule;

namespace optimize {
using namespace ge;
class BrcInlineV2 : public testing::Test {
 protected:
  void SetUp() override {
    // setenv("DUMP_GE_GRAPH", "2", 1);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2>();
    RuntimeStub::SetInstance(stub_v2);
    GTEST_SKIP(); // 临时关闭 V2 的 brc inline 模板，启动模板后需要同步打开此用例
  }
  void TearDown() override {
    // setenv("DUMP_GE_GRAPH", "0", 1);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }

  Optimizer optimizer;
  BrcInlineV2() : optimizer(OptimizerOptions{}) {}
};

static void CheckVectorStrides(const ge::AscGraph &graph, const std::vector<std::string> &names, const bool exp_align) {
  for (const auto &name : names) {
    const auto &node = graph.FindNode(name.c_str());
    EXPECT_NE(node, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<ge::AscNode>(node), nullptr);
    const auto &asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
    for (const auto &tensor : asc_node->outputs()) {
      const auto &strides = tensor->attr.vectorized_strides;
      bool is_aligned = std::all_of(strides.begin(), strides.end(), [](const auto &stride) {
        return stride == One || stride == Zero || ascgen_utils::ExpressEq(ge::sym::Mod(stride, ge::Symbol(16)), Zero);
      });
      EXPECT_EQ(is_aligned, exp_align);
    }
  }
}

TEST_F(BrcInlineV2, brc_inline_v2_no_cascade) {
  ge::AscGraph graph("brc_inline_v2_no_cascade");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(31);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = load0.y;
  mul0.x2 = brc1.y;
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  ge::ascir_op::Sub sub0("sub0");
  sub0.x1 = add0.y;
  sub0.x2 = mul0.y;
  sub0.attr.sched.axis = {z0.id, z1.id};
  sub0.y.dtype = ge::DT_FLOAT16;
  *sub0.y.axis = {z0.id, z1.id};
  *sub0.y.repeats = {s0, s1};
  *sub0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = sub0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_NE(impl_graphs[0].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[2].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].FindNode("brc1"), nullptr);

  const auto &inline_graph = impl_graphs[3];
  CheckVectorStrides(inline_graph, {"load0", "load1", "brc_inline_v2_no_cascade_0_general_0_nil_0_nil_inline_VfNode_0"},
                     true);

  CheckVectorStrides(inline_graph,
                     {"brc_inline_v2_no_cascade_0_general_0_nil_0_nil_inline_VfNode_0_remove_pad_0", "store0"}, false);

  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(inline_graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 1UL);
}

TEST_F(BrcInlineV2, brc_inline_v2_cascade_all_inline) {
  ge::AscGraph graph("brc_inline_v2_cascade_all_inline");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(31);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = add0.y;
  mul0.x2 = brc1.y;
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  ge::ascir_op::Sub sub0("sub0");
  sub0.x1 = mul0.y;
  sub0.x2 = mul0.y;
  sub0.attr.sched.axis = {z0.id, z1.id};
  sub0.y.dtype = ge::DT_FLOAT16;
  *sub0.y.axis = {z0.id, z1.id};
  *sub0.y.repeats = {s0, s1};
  *sub0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = sub0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_NE(impl_graphs[0].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[2].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].FindNode("brc1"), nullptr);

  const auto &inline_graph = impl_graphs[3];
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(inline_graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 1UL);

  CheckVectorStrides(
      inline_graph, {"load0", "load1", "brc_inline_v2_cascade_all_inline_0_general_0_nil_0_nil_inline_VfNode_0"}, true);

  CheckVectorStrides(inline_graph,
                     {"brc_inline_v2_cascade_all_inline_0_general_0_nil_0_nil_inline_VfNode_0_remove_pad_0", "store0"},
                     false);
}

TEST_F(BrcInlineV2, brc_inline_v2_cascade_mix_inline) {
  ge::AscGraph graph("brc_inline_v2_cascade_mix_inline");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(31);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = add0.y;
  mul0.x2 = brc1.y;
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  ge::ascir_op::Sub sub0("sub0");
  sub0.x1 = add0.y;
  sub0.x2 = mul0.y;
  sub0.attr.sched.axis = {z0.id, z1.id};
  sub0.y.dtype = ge::DT_FLOAT16;
  *sub0.y.axis = {z0.id, z1.id};
  *sub0.y.repeats = {s0, s1};
  *sub0.y.strides = {s1, One};

  ge::ascir_op::Sub sub1("sub1");
  sub1.x1 = add0.y;
  sub1.x2 = add0.y;
  sub1.attr.sched.axis = {z0.id, z1.id};
  sub1.y.dtype = ge::DT_FLOAT16;
  *sub1.y.axis = {z0.id, z1.id};
  *sub1.y.repeats = {s0, s1};
  *sub1.y.strides = {s1, One};

  ge::ascir_op::Div div0("div0");
  div0.x1 = sub0.y;
  div0.x2 = sub1.y;
  div0.attr.sched.axis = {z0.id, z1.id};
  div0.y.dtype = ge::DT_FLOAT16;
  *div0.y.axis = {z0.id, z1.id};
  *div0.y.repeats = {s0, s1};
  *div0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = div0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_NE(impl_graphs[0].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[2].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].FindNode("brc1"), nullptr);

  const auto &inline_graph = impl_graphs[3];
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(inline_graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 1UL);

  CheckVectorStrides(
      inline_graph, {"load0", "load1", "brc_inline_v2_cascade_mix_inline_0_general_0_nil_0_nil_inline_VfNode_0"}, true);

  CheckVectorStrides(
      inline_graph,
      {"brc_inline_v2_cascade_mix_inline_0_general_0_nil_0_nil_inline_VfNode_0_remove_pad_1",
       "brc_inline_v2_cascade_mix_inline_0_general_0_nil_0_nil_inline_VfNode_0_remove_pad_0", "sub1", "div0", "store0"},
      false);
}

/**
 *
 *                            div0
 *                          /   \
 *                       sub0    \
 *                      /  \     |
 *                   exp0   \   sub1
 *                    |     |   \ /
 *                   mul0   |   /
 *                  /   \   |  /
 *                /      \ | /
 *               |       add0
 *               |     /      \
 *               \   /         \
 *               brc0          brc1
 *               /               \
 *             abs0             abs1
 *               \               /
 *              load0         load1
 *                 \          /
 *               data0      data1
 */
TEST_F(BrcInlineV2, brc_inline_v2_vf_merge_axes_AABB) {
  ge::AscGraph graph("brc_inline_v2_vf_merge_axes_AABB");
  auto s0 = ge::Symbol(7);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(9);
  auto s3 = ge::Symbol(10);
  auto s4 = ge::Symbol(11);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, s1, s2, One, s4};
  *load0.y.strides = {Zero, s2 * s4, s4, Zero, One};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, s1, s2, One, s4};
  *abs0.y.strides = {Zero, s2 * s4, s4, Zero, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, s1, s2, s3, s4};
  *brc0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.repeats = {One, s1, s2, s3, One};
  *load1.y.strides = {Zero, s2 * s3, s3, One, Zero};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs1.y.dtype = ge::DT_FLOAT16;
  *abs1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs1.y.repeats = {One, s1, s2, s3, One};
  *abs1.y.strides = {Zero, s2 * s3, s3, One, Zero};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = abs1.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, s4};
  *brc1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = add0.y;
  mul0.x2 = brc0.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *mul0.y.repeats = {s0, s1, s2, s3, s4};
  *mul0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Exp exp0("exp0");
  exp0.x = mul0.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Sub sub0("sub0");
  sub0.x1 = add0.y;
  sub0.x2 = exp0.y;
  sub0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub0.y.dtype = ge::DT_FLOAT16;
  *sub0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub0.y.repeats = {s0, s1, s2, s3, s4};
  *sub0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Sub sub1("sub1");
  sub1.x1 = add0.y;
  sub1.x2 = add0.y;
  sub1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub1.y.dtype = ge::DT_FLOAT16;
  *sub1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub1.y.repeats = {s0, s1, s2, s3, s4};
  *sub1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Div div0("div0");
  div0.x1 = sub0.y;
  div0.x2 = sub1.y;
  div0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  div0.y.dtype = ge::DT_FLOAT16;
  *div0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *div0.y.repeats = {s0, s1, s2, s3, s4};
  *div0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  store0.x = div0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
}

TEST_F(BrcInlineV2, brc_inline_v2_vf_merge_axes_BBAA) {
  ge::AscGraph graph("brc_inline_v2_vf_merge_axes_BBAA");
  auto s0 = ge::Symbol(7);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(9);
  auto s3 = ge::Symbol(10);
  auto s4 = ge::Symbol(11);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, One, s2, s3, s4};
  *load0.y.strides = {s2 * s3 * s4, Zero, s3 * s4, s4, One};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {s0, One, s2, s3, s4};
  *abs0.y.strides = {s2 * s3 * s4, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, s1, s2, s3, s4};
  *brc0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.repeats = {s0, s1, One, s3, s4};
  *load1.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs1.y.dtype = ge::DT_FLOAT16;
  *abs1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs1.y.repeats = {s0, s1, One, s3, s4};
  *abs1.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = abs1.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, s4};
  *brc1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = add0.y;
  mul0.x2 = brc0.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *mul0.y.repeats = {s0, s1, s2, s3, s4};
  *mul0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Exp exp0("exp0");
  exp0.x = mul0.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Sub sub0("sub0");
  sub0.x1 = add0.y;
  sub0.x2 = exp0.y;
  sub0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub0.y.dtype = ge::DT_FLOAT16;
  *sub0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub0.y.repeats = {s0, s1, s2, s3, s4};
  *sub0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Sub sub1("sub1");
  sub1.x1 = add0.y;
  sub1.x2 = add0.y;
  sub1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub1.y.dtype = ge::DT_FLOAT16;
  *sub1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub1.y.repeats = {s0, s1, s2, s3, s4};
  *sub1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Div div0("div0");
  div0.x1 = sub0.y;
  div0.x2 = sub1.y;
  div0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  div0.y.dtype = ge::DT_FLOAT16;
  *div0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *div0.y.repeats = {s0, s1, s2, s3, s4};
  *div0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  store0.x = div0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 7UL);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs[4].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[4].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[5].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[5].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[6].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[6].FindNode("brc1"), nullptr);

  const auto &inline0_graph = impl_graphs[4];
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(inline0_graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 1UL);
  CheckVectorStrides(inline0_graph,
                     {"load0", "load1", "brc_inline_v2_vf_merge_axes_BBAA_0_general_0_nil_0_nil_inline_VfNode_0"},
                     true);

  CheckVectorStrides(inline0_graph,
                     {"brc_inline_v2_vf_merge_axes_BBAA_0_general_0_nil_0_nil_inline_VfNode_0_remove_pad_0",
                      "brc_inline_v2_vf_merge_axes_BBAA_0_general_0_nil_0_nil_inline_VfNode_0_remove_pad_1", "sub0",
                      "sub1", "div0", "store0"},
                     false);
}

TEST_F(BrcInlineV2, brc_inline_v2_not_support_inline) {
  ge::AscGraph graph("brc_api_not_support");
  auto s0 = ge::Symbol(7);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(9);
  auto s3 = ge::Symbol(10);
  auto s4 = ge::Symbol(11);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, One, s2, s3, s4};
  *load0.y.strides = {s2 * s3 * s4, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, s1, s2, s3, s4};
  *brc0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.repeats = {s0, s1, One, s3, s4};
  *load1.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, s4};
  *brc1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Ge ge0("ge0");
  ge0.x1 = brc0.y;
  ge0.x2 = brc1.y;
  ge0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  ge0.y.dtype = ge::DT_FLOAT16;
  *ge0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *ge0.y.repeats = {s0, s1, s2, s3, s4};
  *ge0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  store0.x = ge0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 7UL);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  for (const auto &asc_graph : impl_graphs) {
    const auto &name = asc_graph.GetName();
    EXPECT_EQ(name.find("inline"), std::string::npos);
  }
}

TEST_F(BrcInlineV2, brc_inline_v2_not_support_5axes) {
  ge::AscGraph graph("brc_not_support_5axes");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(31);
  auto s2 = s0;
  auto s3 = s0;
  auto s4 = s0;
  auto s5 = s0;

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *load0.y.repeats = {s0, s1, s2, s3, s4, s5};
  *load0.y.strides = {s1 * s2 * s3 * s4 * s5, s2 * s3 * s4 * s5, s3 * s4 * s5, s4 * s5, s5, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *load1.y.repeats = {s0, One, s2, One, s4, One};
  *load1.y.strides = {s2 * s4, Zero, s4, Zero, One, Zero};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *brc1.y.repeats = {s0, s1, s2, s3, s4, s5};
  *brc1.y.strides = {s1 * s2 * s3 * s4 * s5, s2 * s3 * s4 * s5, s3 * s4 * s5, s4 * s5, s5, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4, s5};
  *add0.y.strides = {s1 * s2 * s3 * s4 * s5, s2 * s3 * s4 * s5, s3 * s4 * s5, s4 * s5, s5, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4, s5};
  *store0.y.strides = {s1 * s2 * s3 * s4 * s5, s2 * s3 * s4 * s5, s3 * s4 * s5, s4 * s5, s5, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 12UL);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  for (const auto &asc_graph : impl_graphs) {
    const auto &name = asc_graph.GetName();
    EXPECT_EQ(name.find("inline"), std::string::npos);
  }
}

TEST_F(BrcInlineV2, brc_inline_v2_with_removepad) {
  ge::AscGraph graph("brc_inline_v2_with_removepad");
  auto s0 = ge::Symbol(20);
  auto s1 = ge::Symbol(31);
  auto s1_align = ge::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1_align};
  *load0.y.strides = {s1_align, One};

  ge::ascir_op::RemovePad removepad0("removepad0");
  removepad0.x = load0.y;
  removepad0.attr.sched.axis = {z0.id, z1.id};
  removepad0.y.dtype = ge::DT_FLOAT16;
  *removepad0.y.axis = {z0.id, z1.id};
  *removepad0.y.repeats = {s0, s1};
  *removepad0.y.strides = {s1, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = removepad0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_NE(impl_graphs[0].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[2].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].FindNode("brc1"), nullptr);

  const auto &inline_graph = impl_graphs[3];
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(inline_graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 1UL);
  EXPECT_EQ(inline_graph.FindNode("removepad0"), nullptr);

  CheckVectorStrides(inline_graph,
                     {"load0", "load1", "brc_inline_v2_with_removepad_0_general_0_nil_0_nil_inline_VfNode_0"}, true);

  CheckVectorStrides(inline_graph,
                     {"brc_inline_v2_with_removepad_0_general_0_nil_0_nil_inline_VfNode_0_remove_pad_0", "store0"},
                     false);
}

}  // namespace optimize