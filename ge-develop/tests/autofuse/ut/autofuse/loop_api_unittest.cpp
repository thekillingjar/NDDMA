
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lowering/asc_lowerer/loop_api.h"
#include "utils/auto_fuse_config.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/debug/ge_op_types.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"

#include "compliant_op_desc_builder.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include <gtest/gtest.h>
using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;

class LoopApiUT : public testing::Test {
 public:
 protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("Hi Lowering graph"));
    setenv("ENABLE_LOWER_MATMUL", "true", 1);
  }
  void TearDown() override {}
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(LoopApiUT, SimpleLogAddExp) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto log_add_exp = es::LogAddExp(data0, data1);
    log_add_exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(log_add_exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto log_add_exp = cg->FindNode("LogAddExp_0");
  ASSERT_NE(log_add_exp, nullptr);

  auto box0 = loop::Load(log_add_exp->GetInDataAnchor(0));
  auto box1 = loop::Load(log_add_exp->GetInDataAnchor(1));
  auto log_box = loop::Log(box0);
  auto add_box = loop::Add(log_box, box1);
  auto exp_box = loop::Exp(add_box);
  auto result_box = loop::Store(log_add_exp->GetOutDataAnchor(0), exp_box);

  EXPECT_EQ(log_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Log(tmp0)\n");

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Log(tmp0)\n"
            "tmp3 = ops.Add(tmp2, tmp1)\n"
            "tmp4 = ops.Exp(tmp3)\n"
            "tmp5 = ops.Store(\"LogAddExp_0:0\", tmp4)\n");
}

TEST_F(LoopApiUT, SqueezePositiveDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s2"});
    auto squeeze = es::Squeeze(data0, {1});
    squeeze.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeeze = cg->FindNode("Squeeze_0");
  ASSERT_NE(squeeze, nullptr);

  auto box0 = loop::Load(squeeze->GetInDataAnchor(0));
  auto result_box = loop::Store(squeeze->GetOutDataAnchor(0), loop::Squeeze(box0, 1));

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 1)\n"
            "tmp2 = ops.Store(\"Squeeze_0:0\", tmp1)\n");
}

TEST_F(LoopApiUT, SqueezeNegativeDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s2"});
    auto squeeze = es::Squeeze(data0, {-2});
    squeeze.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeeze = cg->FindNode("Squeeze_0");
  ASSERT_NE(squeeze, nullptr);

  auto box0 = loop::Load(squeeze->GetInDataAnchor(0));
  auto result_box = loop::Store(squeeze->GetOutDataAnchor(0), loop::Squeeze(box0, -2));

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, -2)\n"
            "tmp2 = ops.Store(\"Squeeze_0:0\", tmp1)\n");
}

TEST_F(LoopApiUT, UnsqueezePositiveDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s2"});
    auto squeeze = es::Unsqueeze(data0, {1});
    squeeze.SetSymbolShape({"s0", "1", "s2"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeeze = cg->FindNode("Unsqueeze_0");
  ASSERT_NE(squeeze, nullptr);

  auto box0 = loop::Load(squeeze->GetInDataAnchor(0));
  auto result_box = loop::Store(squeeze->GetOutDataAnchor(0), loop::Unsqueeze(box0, 1));

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, 1)\n"
            "tmp2 = ops.Store(\"Unsqueeze_0:0\", tmp1)\n");
}

TEST_F(LoopApiUT, UnsqueezeNegativeDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s2"});
    auto squeeze = es::Unsqueeze(data0, {-2});
    squeeze.SetSymbolShape({"s0", "1", "s2"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeeze = cg->FindNode("Unsqueeze_0");
  ASSERT_NE(squeeze, nullptr);

  auto box0 = loop::Load(squeeze->GetInDataAnchor(0));
  auto result_box = loop::Store(squeeze->GetOutDataAnchor(0), loop::Unsqueeze(box0, -2));

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, -2)\n"
            "tmp2 = ops.Store(\"Unsqueeze_0:0\", tmp1)\n");
}

TEST_F(LoopApiUT, MultiUseOfSingleOut) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp0 = es::Exp(data0);
    exp0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Exp(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp0, 0);
    es_graph_->SetOutput(exp1, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp0 = cg->FindNode("Exp_0");
  ASSERT_NE(exp0, nullptr);
  auto exp1 = cg->FindNode("Exp_1");
  ASSERT_NE(exp1, nullptr);

  auto box0 = loop::Load(exp0->GetInDataAnchor(0));
  auto exp_box0 = loop::Exp(box0);
  auto result_box0 = loop::Store(exp0->GetOutDataAnchor(0), exp_box0);

  auto box1 = loop::Load(exp1->GetInDataAnchor(0));
  auto exp_box1 = loop::Exp(box1);
  auto result_box1 = loop::Store(exp1->GetOutDataAnchor(0), exp_box1);

  EXPECT_EQ(result_box0.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n");
  EXPECT_EQ(result_box1.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_1:0\", tmp1)\n");
}

TEST_F(LoopApiUT, LoopWithScalar) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reciprocal = es::Reciprocal(data0);
    reciprocal.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(reciprocal, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reciprocal = cg->FindNode("Reciprocal_0");
  ASSERT_NE(reciprocal, nullptr);

  auto box = loop::Load(reciprocal->GetInDataAnchor(0));
  auto div_box = loop::Div(box, loop::Scalar("1.0", ge::DT_FLOAT));
  auto result_box = loop::Store(reciprocal->GetOutDataAnchor(0), div_box);

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Scalar(\"DT_FLOAT(1.0)\")\n"
            "tmp2 = ops.Div(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Reciprocal_0:0\", tmp2)\n");
}

TEST_F(LoopApiUT, LoopLoadSeed) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reciprocal = es::Reciprocal(data0);
    reciprocal.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(reciprocal, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reciprocal = cg->FindNode("Reciprocal_0");
  ASSERT_NE(reciprocal, nullptr);

  auto box = loop::Load(reciprocal->GetInDataAnchor(0));
  auto loadseed_box = loop::LoadSeed("UnknownSeed", box);
  auto result_box = loop::Store(reciprocal->GetOutDataAnchor(0), loadseed_box);

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.LoadSeed(UnknownSeed, tmp0)\n"
            "tmp2 = ops.Store(\"Reciprocal_0:0\", tmp1)\n");
}

TEST_F(LoopApiUT, LoopToDtypeBitcast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reciprocal = es::Reciprocal(data0);
    reciprocal.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(reciprocal, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reciprocal = cg->FindNode("Reciprocal_0");
  ASSERT_NE(reciprocal, nullptr);

  auto box = loop::Load(reciprocal->GetInDataAnchor(0));
  auto todtype_box = loop::ToDtypeBitcast(box, DT_FLOAT, DT_INT64);
  auto result_box = loop::Store(reciprocal->GetOutDataAnchor(0), todtype_box);

  EXPECT_EQ(result_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.ToDtypeBitcast(tmp0, DT_FLOAT, DT_INT64)\n"
            "tmp2 = ops.Store(\"Reciprocal_0:0\", tmp1)\n");
}

TEST_F(LoopApiUT, MultiUseOfSingleOutButNoSymShape) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp0 = es::Exp(data0);
    auto exp1 = es::Exp(data0);
    es_graph_->SetOutput(exp0, 0);
    es_graph_->SetOutput(exp1, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp0 = cg->FindNode("Exp_0");
  ASSERT_NE(exp0, nullptr);
  auto exp1 = cg->FindNode("Exp_1");
  ASSERT_NE(exp1, nullptr);

  auto box0 = loop::Load(exp0->GetInDataAnchor(0));
  auto exp_box0 = loop::Exp(box0);
  auto result_box0 = loop::Store(exp0->GetOutDataAnchor(0), exp_box0);

  auto box1 = loop::Load(exp1->GetInDataAnchor(0));
  auto exp_box1 = loop::Exp(box1);
  auto result_box1 = loop::Store(exp1->GetOutDataAnchor(0), exp_box1);

  EXPECT_EQ(result_box0.Readable(), "ops.ExternKernel.Exp(\"Exp_0:0\")");
  EXPECT_EQ(result_box1.Readable(), "ops.ExternKernel.Exp(\"Exp_1:0\")");
}

TEST_F(LoopApiUT, BroadcastWithNoRealBroadcast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto broadcast = es::BroadcastToD(data0, {});  // Just stub here
    broadcast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(broadcast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto broadcast = cg->FindNode("BroadcastToD_0");
  ASSERT_NE(broadcast, nullptr);

  auto box = loop::Load(broadcast->GetInDataAnchor(0));

  std::vector<Expression> src_dims = {Symbol(2), Symbol(3), Symbol(4)};
  auto broadcast_box0 = loop::Broadcast(box, src_dims, src_dims);

  std::vector<loop::BroadcastOp::DimKind> status = {
      loop::BroadcastOp::DimKind::NORMAL, loop::BroadcastOp::DimKind::NORMAL, loop::BroadcastOp::DimKind::NORMAL};
  auto broadcast_box1 = loop::Broadcast(box, status);

  EXPECT_EQ(broadcast_box0.Readable(), "tmp0 = ops.Load(\"data0:0\")\n");
  EXPECT_EQ(broadcast_box1.Readable(), "tmp0 = ops.Load(\"data0:0\")\n");
}

TEST_F(LoopApiUT, BroadcastWithOnlyBroadcast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "s1", "s2"});
    auto broadcast = es::BroadcastToD(data0, {});  // Just stub here
    broadcast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(broadcast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto broadcast = cg->FindNode("BroadcastToD_0");
  ASSERT_NE(broadcast, nullptr);

  auto box = loop::Load(broadcast->GetInDataAnchor(0));

  std::vector<Expression> src_dims = {Symbol(1), Symbol(3), Symbol(4)};
  std::vector<Expression> dst_dims = {Symbol(2), Symbol(3), Symbol(4)};
  auto broadcast_box0 = loop::Broadcast(box, src_dims, dst_dims);

  std::vector<loop::BroadcastOp::DimKind> status = {
      loop::BroadcastOp::DimKind::BROADCAST, loop::BroadcastOp::DimKind::NORMAL, loop::BroadcastOp::DimKind::NORMAL};
  auto broadcast_box1 = loop::Broadcast(box, status);

  EXPECT_EQ(broadcast_box0.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1, d1, d2]->[d0, d1, d2]\")\n");
  EXPECT_EQ(broadcast_box1.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1, d1, d2]->[d0, d1, d2]\")\n");
}

TEST_F(LoopApiUT, BroadcastWithNewAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "s2"});
    auto broadcast = es::BroadcastToD(data0, {});  // Just stub here
    broadcast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(broadcast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto broadcast = cg->FindNode("BroadcastToD_0");
  ASSERT_NE(broadcast, nullptr);

  auto box = loop::Load(broadcast->GetInDataAnchor(0));

  std::vector<Expression> src_dims = {Symbol(3), Symbol(4)};
  std::vector<Expression> dst_dims = {Symbol(2), Symbol(3), Symbol(4)};
  auto broadcast_box0 = loop::Broadcast(box, src_dims, dst_dims);

  std::vector<loop::BroadcastOp::DimKind> status = {
      loop::BroadcastOp::DimKind::NEW_AXIS, loop::BroadcastOp::DimKind::NORMAL, loop::BroadcastOp::DimKind::NORMAL};
  auto broadcast_box1 = loop::Broadcast(box, status);

  EXPECT_EQ(broadcast_box0.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1, d2]->[d0, d1, d2]\")\n");
  EXPECT_EQ(broadcast_box1.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1, d2]->[d0, d1, d2]\")\n");
}

TEST_F(LoopApiUT, BroadcastWithAllKind) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "1"});
    auto broadcast = es::BroadcastToD(data0, {});  // Just stub here
    broadcast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(broadcast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto broadcast = cg->FindNode("BroadcastToD_0");
  ASSERT_NE(broadcast, nullptr);

  auto box = loop::Load(broadcast->GetInDataAnchor(0));

  std::vector<Expression> src_dims = {Symbol(3), Symbol(1)};
  std::vector<Expression> dst_dims = {Symbol(2), Symbol(3), Symbol(4)};
  auto broadcast_box0 = loop::Broadcast(box, src_dims, dst_dims);

  std::vector<loop::BroadcastOp::DimKind> status = {loop::BroadcastOp::DimKind::NEW_AXIS,
                                                    loop::BroadcastOp::DimKind::NORMAL,
                                                    loop::BroadcastOp::DimKind::BROADCAST};
  auto broadcast_box1 = loop::Broadcast(box, status);

  EXPECT_EQ(broadcast_box0.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1, 1]->[d0, d1, d2]\")\n");
  EXPECT_EQ(broadcast_box1.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1, 1]->[d0, d1, d2]\")\n");
}

TEST_F(LoopApiUT, BroadcastWithInvalid) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto broadcast = es::BroadcastToD(data0, {});  // Just stub here
    broadcast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(broadcast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto broadcast = cg->FindNode("BroadcastToD_0");
  ASSERT_NE(broadcast, nullptr);

  auto box = loop::Load(broadcast->GetInDataAnchor(0));

  std::vector<Expression> src_dims = {Symbol(3), Symbol(3)};
  std::vector<Expression> dst_dims = {Symbol(2), Symbol(3), Symbol(4)};
  EXPECT_TRUE(!loop::Broadcast(box, src_dims, dst_dims).IsValid());

  src_dims = {Symbol(3), Symbol(3)};
  dst_dims = {Symbol(2)};
  EXPECT_TRUE(!loop::Broadcast(box, src_dims, dst_dims).IsValid());

  src_dims = {Symbol(3), Symbol(3)};
  dst_dims = {Symbol(3), Symbol(4)};
  EXPECT_TRUE(!loop::Broadcast(box, src_dims, dst_dims).IsValid());
}

TEST_F(LoopApiUT, CallWithInvalidLoopVar) {
  auto var = loop::LoopVar();
  EXPECT_FALSE(loop::Add(var, var).IsValid());
  EXPECT_FALSE(loop::Log(var).IsValid());
  EXPECT_FALSE(loop::Div(var, var).IsValid());
  EXPECT_FALSE(loop::Exp(var).IsValid());
  EXPECT_FALSE(loop::Broadcast(var, {}).IsValid());
  EXPECT_FALSE(loop::LoadSeed("LoadSeed", var).IsValid());
  EXPECT_FALSE(loop::ToDtypeBitcast(var, DT_FLOAT, DT_INT64).IsValid());
}

TEST_F(LoopApiUT, StoreTwice) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Exp(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto abs = cg->FindNode("Exp_0");
  ASSERT_NE(abs, nullptr);

  auto box0 = loop::Load(abs->GetInDataAnchor(0));
  auto exp_box = loop::Exp(box0);
  auto result_box0 = loop::Store(abs->GetOutDataAnchor(0), exp_box);
  auto result_box1 = loop::Store(abs->GetOutDataAnchor(0), exp_box);

  EXPECT_TRUE(!result_box0.IsExternKernel());
  EXPECT_TRUE(!result_box1.IsExternKernel());
}

TEST_F(LoopApiUT, StoreReductionKeepDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {1}, true);
    reduce.SetSymbolShape({"s0", "1", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box = loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {1});

  EXPECT_TRUE(!reduce_box.IsExternKernel());
  EXPECT_EQ(reduce_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSumD_0:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, 1, d2]\"))\n");
}

TEST_F(LoopApiUT, StoreReductionNotKeepDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {1}, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box = loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {1});

  EXPECT_TRUE(!reduce_box.IsExternKernel());
  EXPECT_EQ(reduce_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSumD_0:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

TEST_F(LoopApiUT, StoreReductionDimEndNotKeepDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {2}, false);
    reduce.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box = loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {2});

  EXPECT_TRUE(!reduce_box.IsExternKernel());
  EXPECT_EQ(reduce_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSumD_0:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d1]\"))\n");
}

TEST_F(LoopApiUT, StoreReductionDimBeginNotKeepDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {0}, false);
    reduce.SetSymbolShape({"s1", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box = loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {0});

  EXPECT_TRUE(!reduce_box.IsExternKernel());
  EXPECT_EQ(reduce_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSumD_0:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d1, d2]\"))\n");
}

TEST_F(LoopApiUT, StoreReduction2Dim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {1, 2}, false);
    reduce.SetSymbolShape({"s0"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box = loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {1, 2});

  EXPECT_TRUE(!reduce_box.IsExternKernel());
  EXPECT_EQ(reduce_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSumD_0:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0]\"))\n");
}

TEST_F(LoopApiUT, StoreReduction2DimKeepDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {1, 2}, true);
    reduce.SetSymbolShape({"s0", "1", "1"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box = loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {1, 2});

  EXPECT_TRUE(!reduce_box.IsExternKernel());
  EXPECT_EQ(reduce_box.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSumD_0:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, 1, 1]\"))\n");
}

TEST_F(LoopApiUT, StoreReductionAllDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {0, 1, 2}, false);
    reduce.SetSymbolShape({});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box =
      loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {0, 1, 2});

  EXPECT_FALSE(reduce_box.IsExternKernel());
}

TEST_F(LoopApiUT, StoreReductionAllDimKeepDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {0, 1, 2}, false);
    reduce.SetSymbolShape({"1", "1", "1"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  auto box0 = loop::Load(reduce->GetInDataAnchor(0));
  std::vector<Expression> src_dims = {Symbol("s1"), Symbol("s2"), Symbol("s3")};
  auto reduce_box =
      loop::StoreReduction(loop::ReduceType::SUM, reduce->GetOutDataAnchor(0), box0, src_dims, {0, 1, 2});

  EXPECT_FALSE(reduce_box.IsExternKernel());
}

TEST_F(LoopApiUT, StorePackPositiveDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto pack = es::Pack({data0, data1}, 0, 2);
    pack.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {pack->GetInDataAnchor(0), pack->GetInDataAnchor(1)};
  auto kernel_box = loop::StorePack(pack->GetOutDataAnchor(0), inputs, 0);
  EXPECT_TRUE(!kernel_box.IsExternKernel());
  EXPECT_EQ(kernel_box.Readable(),R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.StoreConcat("Pack_0:0", [tmp0, tmp1], concat_dim=0)
)");
}

TEST_F(LoopApiUT, StorePackNegDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto pack = es::Pack({data0, data1}, -2, 2);
    pack.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {pack->GetInDataAnchor(0), pack->GetInDataAnchor(1)};
  auto kernel_box = loop::StorePack(pack->GetOutDataAnchor(0), inputs, -2);
  EXPECT_TRUE(!kernel_box.IsExternKernel());
  EXPECT_EQ(kernel_box.Readable(),R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.StoreConcat("Pack_0:0", [tmp0, tmp1], concat_dim=0)
)");
}

TEST_F(LoopApiUT, StorePackNegDimOverflow) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto pack = es::Pack({data0, data1}, 0, 2);
    pack.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {pack->GetInDataAnchor(0), pack->GetInDataAnchor(1)};
  auto kernel_box = loop::StorePack(pack->GetOutDataAnchor(0), inputs, -3);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StorePackPositiveDimOverflow) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto pack = es::Pack({data0, data1}, 0, 2);
    pack.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {pack->GetInDataAnchor(0), pack->GetInDataAnchor(1)};
  auto kernel_box = loop::StorePack(pack->GetOutDataAnchor(0), inputs, 2);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StorePackNoSymShape) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto pack = es::Pack({data0, data1}, 0, 2);
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {pack->GetInDataAnchor(0), pack->GetInDataAnchor(1)};
  auto kernel_box = loop::StorePack(pack->GetOutDataAnchor(0), inputs, 0);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StorePackNullInput) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto pack = es::Pack({data0, data1}, 0, 2);
    pack.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {pack->GetInDataAnchor(0), nullptr};
  auto kernel_box = loop::StorePack(pack->GetOutDataAnchor(0), inputs, 0);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StoreMatmulInputOverflowWithCubeWithOutMatmulEnv) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto mm = es::MatMul(data0, data1);
    mm.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMul_0");
  ASSERT_NE(mm, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {mm->GetInDataAnchor(0)};
  MatMulAttr matmul_attr;
  auto kernel_box = loop::StoreMatMul(mm->GetOutDataAnchor(0), inputs, matmul_attr);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StoreMatmulInputOverflow) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto mm = es::MatMul(data0, data1);
    mm.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMul_0");
  ASSERT_NE(mm, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {mm->GetInDataAnchor(0)};
  MatMulAttr matmul_attr;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_matmul = true;
  auto kernel_box = loop::StoreMatMul(mm->GetOutDataAnchor(0), inputs, matmul_attr);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StoreMatmulOutputNull) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto mm = es::MatMul(data0, data1);
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMul_0");
  ASSERT_NE(mm, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {mm->GetInDataAnchor(0), mm->GetInDataAnchor(1)};
  MatMulAttr matmul_attr;
  auto kernel_box = loop::StoreMatMul(mm->GetOutDataAnchor(0), inputs, matmul_attr);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StoreMatmulInputNull) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto mm = es::MatMul(data0, data1);
    mm.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMul_0");
  ASSERT_NE(mm, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {mm->GetInDataAnchor(0), mm->GetInDataAnchor(1)};
  MatMulAttr matmul_attr;
  auto kernel_box = loop::StoreMatMul(mm->GetOutDataAnchor(0), inputs, matmul_attr);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

TEST_F(LoopApiUT, StoreMatmulInputNull2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto mm = es::MatMul(data0, data1);
    mm.SetSymbolShape({"2", "s0"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMul_0");
  ASSERT_NE(mm, nullptr);
  std::vector<ge::InDataAnchorPtr> inputs = {nullptr, mm->GetInDataAnchor(1)};
  MatMulAttr matmul_attr;
  auto kernel_box = loop::StoreMatMul(mm->GetOutDataAnchor(0), inputs, matmul_attr);
  EXPECT_TRUE(kernel_box.IsExternKernel());
}

#define TEST_F_LOOP_INST1(LOOPOP)                                     \
  TEST_F(LoopApiUT, Loop##LOOPOP) {                                   \
    [this]() {                                                        \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);       \
      data0.SetSymbolShape({"s0", "s1", "s2"});                       \
      auto reciprocal = es::Reciprocal(data0);                        \
      reciprocal.SetSymbolShape({"s0", "s1", "s2"});                  \
      es_graph_->SetOutput(reciprocal, 0);                            \
    }();                                                              \
    auto graph = es_graph_->Build();                                  \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                  \
    auto tmp = cg->FindNode("Reciprocal_0");                          \
    ASSERT_NE(tmp, nullptr);                                          \
    auto box = loop::Load(tmp->GetInDataAnchor(0));                   \
    auto tmp_box = loop::LOOPOP(box);                                 \
    auto result_box = loop::Store(tmp->GetOutDataAnchor(0), tmp_box); \
    EXPECT_EQ(result_box.Readable(),                                  \
              "tmp0 = ops.Load(\"data0:0\")\n"                        \
              "tmp1 = ops." #LOOPOP                                   \
              "(tmp0)\n"                                              \
              "tmp2 = ops.Store(\"Reciprocal_0:0\", tmp1)\n");        \
  }

#define TEST_F_LOOP_INST2(LOOPOP)                                     \
  TEST_F(LoopApiUT, Loop##LOOPOP) {                                   \
    [this]() {                                                        \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);       \
      data0.SetSymbolShape({"s0", "s1", "s2"});                       \
      auto reciprocal = es::Reciprocal(data0);                        \
      reciprocal.SetSymbolShape({"s0", "s1", "s2"});                  \
      es_graph_->SetOutput(reciprocal, 0);                            \
    }();                                                              \
    auto graph = es_graph_->Build();                                  \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                  \
    auto tmp = cg->FindNode("Reciprocal_0");                          \
    ASSERT_NE(tmp, nullptr);                                          \
    auto box = loop::Load(tmp->GetInDataAnchor(0));                   \
    auto tmp_box = loop::LOOPOP(box, box);                            \
    auto result_box = loop::Store(tmp->GetOutDataAnchor(0), tmp_box); \
    EXPECT_EQ(result_box.Readable(),                                  \
              "tmp0 = ops.Load(\"data0:0\")\n"                        \
              "tmp1 = ops.Load(\"data0:0\")\n"                        \
              "tmp2 = ops." #LOOPOP                                   \
              "(tmp1, tmp1)\n"                                        \
              "tmp3 = ops.Store(\"Reciprocal_0:0\", tmp2)\n");        \
  }

#define TEST_F_LOOP_INST3(LOOPOP)                                     \
  TEST_F(LoopApiUT, Loop##LOOPOP) {                                   \
    [this]() {                                                        \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);       \
      data0.SetSymbolShape({"s0", "s1", "s2"});                       \
      auto reciprocal = es::Reciprocal(data0);                        \
      reciprocal.SetSymbolShape({"s0", "s1", "s2"});                  \
      es_graph_->SetOutput(reciprocal, 0);                            \
    }();                                                              \
    auto graph = es_graph_->Build();                                  \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                  \
    auto tmp = cg->FindNode("Reciprocal_0");                          \
    ASSERT_NE(tmp, nullptr);                                          \
    auto box = loop::Load(tmp->GetInDataAnchor(0));                   \
    auto tmp_box = loop::LOOPOP(box, box, box);                       \
    auto result_box = loop::Store(tmp->GetOutDataAnchor(0), tmp_box); \
    EXPECT_EQ(result_box.Readable(),                                  \
              "tmp0 = ops.Load(\"data0:0\")\n"                        \
              "tmp1 = ops.Load(\"data0:0\")\n"                        \
              "tmp2 = ops.Load(\"data0:0\")\n"                        \
              "tmp3 = ops." #LOOPOP                                   \
              "(tmp2, tmp2, tmp2)\n"                                  \
              "tmp4 = ops.Store(\"Reciprocal_0:0\", tmp3)\n");        \
  }

#define TEST_F_LOOP_INST4(LOOPOP)                                     \
  TEST_F(LoopApiUT, Loop##LOOPOP) {                                   \
    [this]() {                                                        \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);       \
      data0.SetSymbolShape({"s0", "s1", "s2"});                       \
      auto reciprocal = es::Reciprocal(data0);                        \
      reciprocal.SetSymbolShape({"s0", "s1", "s2"});                  \
      es_graph_->SetOutput(reciprocal, 0);                            \
    }();                                                              \
    auto graph = es_graph_->Build();                                  \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                  \
    auto tmp = cg->FindNode("Reciprocal_0");                          \
    ASSERT_NE(tmp, nullptr);                                          \
    auto box = loop::Load(tmp->GetInDataAnchor(0));                   \
    auto tmp_box = loop::LOOPOP(box, box, box, box);                  \
    auto result_box = loop::Store(tmp->GetOutDataAnchor(0), tmp_box); \
    EXPECT_EQ(result_box.Readable(),                                  \
              "tmp0 = ops.Load(\"data0:0\")\n"                        \
              "tmp1 = ops.Load(\"data0:0\")\n"                        \
              "tmp2 = ops.Load(\"data0:0\")\n"                        \
              "tmp3 = ops.Load(\"data0:0\")\n"                        \
              "tmp4 = ops." #LOOPOP                                   \
              "(tmp3, tmp3, tmp3, tmp3)\n"                            \
              "tmp5 = ops.Store(\"Reciprocal_0:0\", tmp4)\n");        \
  }

#define TEST_F_LOOP_DTYPE_INST1(LOOPOP)                               \
  TEST_F(LoopApiUT, Loop##LOOPOP) {                                   \
    [this]() {                                                        \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);       \
      data0.SetSymbolShape({"s0", "s1", "s2"});                       \
      auto reciprocal = es::Cast(data0, DT_FLOAT);                    \
      reciprocal.SetSymbolShape({"s0", "s1", "s2"});                  \
      es_graph_->SetOutput(reciprocal, 0);                            \
    }();                                                              \
    auto graph = es_graph_->Build();                                  \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                  \
    auto tmp = cg->FindNode("Cast_0");                                \
    ASSERT_NE(tmp, nullptr);                                          \
    auto box = loop::Load(tmp->GetInDataAnchor(0));                   \
    auto tmp_box = loop::LOOPOP(box, DT_FLOAT);                       \
    auto result_box = loop::Store(tmp->GetOutDataAnchor(0), tmp_box); \
    EXPECT_EQ(result_box.Readable(),                                  \
              "tmp0 = ops.Load(\"data0:0\")\n"                        \
              "tmp1 = ops." #LOOPOP                                   \
              "(tmp0, DT_FLOAT)\n"                                    \
              "tmp2 = ops.Store(\"Cast_0:0\", tmp1)\n");              \
  }

TEST_F_LOOP_INST1(Abs);
TEST_F_LOOP_INST1(Acos);
TEST_F_LOOP_INST1(Acosh);
TEST_F_LOOP_INST1(Asin);
TEST_F_LOOP_INST1(Asinh);
TEST_F_LOOP_INST1(Atan);
TEST_F_LOOP_INST1(Atanh);
TEST_F_LOOP_INST1(BesselJ0);
TEST_F_LOOP_INST1(BesselJ1);
TEST_F_LOOP_INST1(BesselY0);
TEST_F_LOOP_INST1(BesselY1);
TEST_F_LOOP_INST1(BitwiseNot);
TEST_F_LOOP_INST1(Ceil);
TEST_F_LOOP_INST1(Cos);
TEST_F_LOOP_INST1(Cosh);
TEST_F_LOOP_INST1(Erf);
TEST_F_LOOP_INST1(Erfc);
TEST_F_LOOP_INST1(Erfcx);
TEST_F_LOOP_INST1(Erfinv);
TEST_F_LOOP_INST1(Exp);
TEST_F_LOOP_INST1(Exp2);
TEST_F_LOOP_INST1(Expm1);
TEST_F_LOOP_INST1(Floor);
TEST_F_LOOP_INST1(Frexp);
TEST_F_LOOP_INST1(I0);
TEST_F_LOOP_INST1(I1);
TEST_F_LOOP_INST1(Identity);
TEST_F_LOOP_INST1(IsInf);
TEST_F_LOOP_INST1(IsNan);
TEST_F_LOOP_INST1(Lgamma);
TEST_F_LOOP_INST1(LibdeviceAbs);
TEST_F_LOOP_INST1(LibdeviceCos);
TEST_F_LOOP_INST1(LibdeviceExp);
TEST_F_LOOP_INST1(LibdeviceLog);
TEST_F_LOOP_INST1(LibdeviceSigmoid);
TEST_F_LOOP_INST1(LibdeviceSin);
TEST_F_LOOP_INST1(LibdeviceSqrt);
TEST_F_LOOP_INST1(Log);
TEST_F_LOOP_INST1(Log10);
TEST_F_LOOP_INST1(Log1p);
TEST_F_LOOP_INST1(Log2);
TEST_F_LOOP_INST1(LogicalNot);
TEST_F_LOOP_INST1(ModifiedBesselI0);
TEST_F_LOOP_INST1(ModifiedBesselI1);
TEST_F_LOOP_INST1(Neg);
TEST_F_LOOP_INST1(Reciprocal);
TEST_F_LOOP_INST1(Relu);
TEST_F_LOOP_INST1(Round);
TEST_F_LOOP_INST1(Rsqrt);
TEST_F_LOOP_INST1(Sigmoid);
TEST_F_LOOP_INST1(Sign);
TEST_F_LOOP_INST1(SignBit);
TEST_F_LOOP_INST1(Sin);
TEST_F_LOOP_INST1(Sinh);
TEST_F_LOOP_INST1(SpecialErf);
TEST_F_LOOP_INST1(Sqrt);
TEST_F_LOOP_INST1(Square);
TEST_F_LOOP_INST1(Tan);
TEST_F_LOOP_INST1(Tanh);
TEST_F_LOOP_INST1(Trunc);

TEST_F_LOOP_INST2(Atan2);
TEST_F_LOOP_INST2(BitwiseAnd);
TEST_F_LOOP_INST2(BitwiseLeftshift);
TEST_F_LOOP_INST2(BitwiseRightshift);
TEST_F_LOOP_INST2(BitwiseOr);
TEST_F_LOOP_INST2(BitwiseXor);
TEST_F_LOOP_INST2(ClampMax);
TEST_F_LOOP_INST2(ClampMin);
TEST_F_LOOP_INST2(CopySign);
TEST_F_LOOP_INST2(Div);
TEST_F_LOOP_INST2(Eq);
TEST_F_LOOP_INST2(FloorDiv);
TEST_F_LOOP_INST2(Fmod);
TEST_F_LOOP_INST2(Ge);
TEST_F_LOOP_INST2(Gt);
TEST_F_LOOP_INST2(Hypot);
TEST_F_LOOP_INST2(InlineAsmElementwise);
TEST_F_LOOP_INST2(Le);
TEST_F_LOOP_INST2(LogicalAnd);
TEST_F_LOOP_INST2(LogicalOr);
TEST_F_LOOP_INST2(LogicalXor);
TEST_F_LOOP_INST2(Lt);
TEST_F_LOOP_INST2(Maximum);
TEST_F_LOOP_INST2(Minimum);
TEST_F_LOOP_INST2(Mul);
TEST_F_LOOP_INST2(Ne);
TEST_F_LOOP_INST2(NextAfter);
TEST_F_LOOP_INST2(Pow);
TEST_F_LOOP_INST2(Rand);
TEST_F_LOOP_INST2(RandN);
TEST_F_LOOP_INST2(Remainder);
TEST_F_LOOP_INST2(Sub);
TEST_F_LOOP_INST2(TrueDiv);
TEST_F_LOOP_INST2(TruncDiv);

TEST_F_LOOP_INST3(ClipByValue);
TEST_F_LOOP_INST3(Fma);
TEST_F_LOOP_INST3(Masked);
TEST_F_LOOP_INST3(Where);

TEST_F_LOOP_INST4(RandInt64);

TEST_F_LOOP_DTYPE_INST1(Cast);
TEST_F_LOOP_DTYPE_INST1(Ceil2Int);
TEST_F_LOOP_DTYPE_INST1(Floor2Int);
TEST_F_LOOP_DTYPE_INST1(IndexExpr);
TEST_F_LOOP_DTYPE_INST1(Round2Int);
TEST_F_LOOP_DTYPE_INST1(Trunc2Int);
}  // namespace ge
