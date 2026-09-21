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
#include "ascir_utils.h"

#include "test_util.h"

using namespace ascir;

namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Data)
}

REG_OPS_WITH_ATTR(Data)
  OPS_ATTR_NAME_START()
    OPS_ATTR_NAME(v)
  OPS_ATTR_NAME_END()

  OPS_ATTR(v, int64_t)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Data)

TEST(TestAscir, AscirOperator_ShouldHas_Fields) {
  Data data("test_op");

  SizeVar s0{0}, s1{1}, s2{2}, s3{3}, s4{4}, s5{5}, s6{6}, s7{7}, s8{8}, s9{9};
  SizeVar s10{10}, s11{11}, s12{12};

  data.v = 1;
  data.attr.sched.axis = {0, 1, 2};

  data.y.axis = {0, 1, 2};
  data.y.repeats = {s1/s2, s3/s4};
  data.y.strides = {s5*s6/s7/s8, s9*s10/s11/s12};

  ascir::Graph graph("test_graph");
  graph.SetInputs({data});
  auto result_op = ge::GraphUtilsEx::GetComputeGraph(graph)->FindNode("test_op")->GetOpDesc();

  AttrEq(result_op, Data::ATTR_v, 1);
  AttrEq(result_op, NodeAttr::SCHED_EXEC_ORDER, 10);
  AttrEq(result_op, NodeAttr::SCHED_AXIS, {0, 1, 2});

  auto result_y = result_op->GetOutputDesc(0);
  AttrEq(result_y, data.y.AXIS, {0, 1, 2});
  vector<SizeExpr> result_repeats = data.y.repeats;
  EXPECT_EQ(result_repeats[0], s1/s2);
  EXPECT_EQ(result_repeats[1], s3/s4);

  auto result_node = graph.Find("test_op");

  int axis_id = 0;
  for (auto axis : result_node.attr.sched.axis()) {
    EXPECT_EQ(axis, axis_id);
    axis_id++;
  }
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(result_node.attr.sched.axis[i], i);
  }

  EXPECT_EQ(result_node.outputs[0].axis[0], 0);
  EXPECT_EQ(result_node.outputs[0].axis[1], 1);
  EXPECT_EQ(result_node.outputs[0].axis[2], 2);

  auto stride = result_node.outputs[0].strides();
  EXPECT_EQ(stride[0].monomials[0].var_nums[0], 5);
  EXPECT_EQ(stride[0].monomials[0].var_nums[1], 6);
  EXPECT_EQ(stride[0].monomials[0].var_dens[0], 7);
  EXPECT_EQ(stride[0].monomials[0].var_dens[1], 8);
  EXPECT_EQ(stride[1].monomials[0].var_nums[0], 9);
  EXPECT_EQ(stride[1].monomials[0].var_nums[1], 10);
  EXPECT_EQ(stride[1].monomials[0].var_dens[0], 11);
  EXPECT_EQ(stride[1].monomials[0].var_dens[1], 12);

  EXPECT_EQ(result_node.outputs[0].repeats[1].monomials[0].var_nums[0], 3);
}

TEST(TestAscir, SizeVar) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  EXPECT_EQ((ascir::SizeExpr(1) / s0).String(), "1 / s0");
}

TEST(TestAscir, ContiguousView_CreateOk) {
  ascir::Graph graph("test_graph");

  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto D = graph.CreateSizeVar("D");

  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);

  Data data("test_op1");
  data.y.SetContiguousView({a, b, c, d});

  graph.SetInputs({data});
  auto result_y = ge::GraphUtilsEx::GetComputeGraph(graph)->FindNode("test_op1")->GetOpDesc()->GetOutputDesc(0);

  AttrEq(result_y, data.y.AXIS, {a.id, b.id, c.id, d.id});
}

/** Create a graph ready for hold attributes */
inline ascir::Graph CreateTestGraph() {
  ascir::Graph graph("test_graph");
  ge::op::Data x("x");
  // Need using SetInputs to trigger initialization of graph
  // so that it can hold attributes
  graph.SetInputs({x});
  return std::move(graph);
}

TEST(MonomialSizeExpr, ConstructByIntrger) {
  MonomialSizeExpr expr(1);
  ASSERT_EQ(expr.const_nums.size(), 1);
  ASSERT_EQ(expr.const_nums[0], 1);
  ASSERT_EQ(expr.const_dens.size(), 1);
  ASSERT_EQ(expr.const_dens[0], 1);

  ASSERT_EQ(expr.var_dens.size(), 0);
  ASSERT_EQ(expr.var_nums.size(), 0);
}

TEST(MonomialSizeExpr, ConstructBySizeVar) {
  ascir::Graph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  MonomialSizeExpr expr({s0}, {s1});
  ASSERT_EQ(expr.const_nums.size(), 1);
  ASSERT_EQ(expr.const_nums[0], 1);
  ASSERT_EQ(expr.const_dens.size(), 1);
  ASSERT_EQ(expr.const_dens[0], 1);

  ASSERT_EQ(expr.var_nums.size(), 1);
  ASSERT_EQ(expr.var_nums[0], s0.id);
  ASSERT_EQ(expr.var_dens.size(), 1);
  ASSERT_EQ(expr.var_dens[0], s1.id);
}

MonomialSizeExpr MExpr(std::initializer_list<Interger>&& const_nums = {},
                       std::initializer_list<Interger>&& const_dens = {},
                       std::initializer_list<SizeVarId>&& var_nums = {},
                       std::initializer_list<SizeVarId>&& var_dens = {}) {
  MonomialSizeExpr result;
  for (auto n: const_nums) {
    result.const_nums.emplace_back(n);
  }
  for (auto d: const_dens) {
    result.const_dens.emplace_back(d);
  }
  for (auto n: var_nums) {
    result.var_nums.emplace_back(n);
  }
  for (auto d: var_dens) {
    result.var_dens.emplace_back(d);
  }
  return result;
}

TEST(MonomialSizeExpr, Simplify) {
  SizeVarId s0 = 0, s1 = 1, s2 = 2, s3 = 3;

  EXPECT_EQ(MExpr({0}, {1}).Simplify(),
            MExpr())
      << "Should convert zero to empty const_nums and empty const_dens";

  EXPECT_EQ(MExpr({2}, {4}).Simplify(),
            MExpr({1}, {2}))
      << "Should reduce the fraction to its simplest form";

  EXPECT_EQ(MExpr({1}, {1}, {s0, s1}, {s0, s2}).Simplify(),
            MExpr({1}, {1}, {s1}, {s2}))
      << "Should remove common items";

  EXPECT_EQ(MExpr({1}, {1}, {s0, s0, s1}, {s0, s2}).Simplify(),
            MExpr({1}, {1}, {s0, s1}, {s2}))
      << "Should remove common items only one times";

  EXPECT_EQ(MExpr({1}, {1}, {s1, s0}, {s3, s2}).Simplify(),
            MExpr({1}, {1}, {s0, s1}, {s2, s3}))
      << "Should sort var_nums and var_dens";
}

TEST(MonomialSizeExpr, Divide) {
  SizeVarId s0 = 0, s1 = 1;

  EXPECT_EQ(MExpr({1}, {1}) / 2, MExpr({1}, {2}));
  EXPECT_EQ(MExpr({4}, {1}) / 2, MExpr({2}, {1}));
  EXPECT_EQ(MExpr() / 2, MExpr());

  EXPECT_EQ(MExpr({1}, {1}) /= 2, MExpr({1}, {2}));
  EXPECT_EQ(MExpr({4}, {1}) /= 2, MExpr({2}, {1}));
  EXPECT_EQ(MExpr() /= 2, MExpr());

  EXPECT_EQ(MExpr({1}, {1}) / MExpr({2}, {1}),
            MExpr({1}, {2}))
      << "1 / 2 == 1/2";
  EXPECT_EQ(MExpr({2}, {1}) / MExpr({4}, {1}),
            MExpr({1}, {2}))
      << "2 / 4 == 1/2";
  EXPECT_EQ(MExpr({1}, {1}) / MExpr({1}, {4}),
            MExpr({4}, {1}))
      << "1 / 1/4 = 4";
  EXPECT_EQ(MExpr({2}, {5}) / MExpr({1}, {10}),
            MExpr({4}, {1}))
      << "2/5 / 1/10 = 4";
  EXPECT_EQ(MExpr() / MExpr({1}, {10}),
            MExpr())
      << "0 / 1/10 = 0";

  EXPECT_EQ(MExpr({1}, {1}) /= MExpr({2}, {1}),
            MExpr({1}, {2}))
      << "1 / 2 == 1/2";
  EXPECT_EQ(MExpr({2}, {1}) /= MExpr({4}, {1}),
            MExpr({1}, {2}))
      << "2 / 4 == 1/2";
  EXPECT_EQ(MExpr({1}, {1}) /= MExpr({1}, {4}),
            MExpr({4}, {1}))
      << "1 / 1/4 = 4";
  EXPECT_EQ(MExpr({2}, {5}) /= MExpr({1}, {10}),
            MExpr({4}, {1}))
      << "2/5 / 1/10 = 4";
  EXPECT_EQ(MExpr() /= MExpr({1}, {10}),
            MExpr())
      << "0 / 1/10 = 0";

  EXPECT_EQ(MExpr({1}, {1}, {s0}) / MExpr({1}, {1}, {s1}),
            MExpr({1}, {1}, {s0}, {s1}))
      << "s0 / s1 == s0/s1";
  EXPECT_EQ(MExpr({1}, {1}, {s0}) / MExpr({1}, {1}, {}, {s1}),
            MExpr({1}, {1}, {s0, s1}))
      << "s0 / 1/s1 == s0*s1";

  EXPECT_EQ(MExpr({1}, {1}, {s0}) /= MExpr({1}, {1}, {s1}),
            MExpr({1}, {1}, {s0}, {s1}))
      << "s0 / s1 == s0/s1";
  EXPECT_EQ(MExpr({1}, {1}, {s0}) /= MExpr({1}, {1}, {}, {s1}),
            MExpr({1}, {1}, {s0, s1}))
      << "s0 / 1/s1 == s0*s1";

  ASSERT_ANY_THROW(MonomialSizeExpr(1) / 0)
      << "Should assert Divide by zero";
  ASSERT_ANY_THROW(MonomialSizeExpr(1) /= 0)
      << "Should assert Divide by zero";
  ASSERT_ANY_THROW(MonomialSizeExpr(1) / MonomialSizeExpr(0))
      << "Should assert Divide by zero";
  ASSERT_ANY_THROW(MonomialSizeExpr(1) /= MonomialSizeExpr(0))
      << "Should assert Divide by zero";
}

TEST(MonomialSizeExpr, MultiplyBy_MonomialSizeExpr) {
  SizeVarId s0 = 0, s1 = 1;

  EXPECT_EQ(MExpr({1}, {1}) * 2, MExpr({2}, {1}))
      << "1 * 2 == 2";
  EXPECT_EQ(MExpr({1}, {1}) * 0, MExpr())
      << "1 * 0 == 0";
  EXPECT_EQ(MExpr() * 2, MExpr())
      << "0 * 2 == 0";

  EXPECT_EQ(MExpr({1}, {1}) *= 2, MExpr({2}, {1}))
      << "1 * 2 == 2";
  EXPECT_EQ(MExpr({1}, {1}) *= 0, MExpr())
      << "1 * 0 == 0";
  EXPECT_EQ(MExpr() *= 2, MExpr())
      << "0 * 2 == 0";

  EXPECT_EQ(MExpr({1}, {1}) * MExpr(), MExpr())
      << "1 * 0 == 0";
  EXPECT_EQ(MExpr({1}, {1}) * MExpr({2}, {1}),
          MExpr({2}, {1}))
      << "1 * 2 == 2";
  EXPECT_EQ(MExpr({1}, {1}) * MExpr({1}, {2}),
          MExpr({1}, {2}))
      << "1 * 1/2 == 1/2";
  EXPECT_EQ(MExpr({2}, {1}) * MExpr({1}, {2}),
          MExpr({1}, {1}))
      << "2 * 1/2 == 1";
  EXPECT_EQ(MExpr() * MExpr({2}, {1}),
          MExpr())
      << "0 * 2 == 0";

  EXPECT_EQ(MExpr({1}, {1}) *= MExpr(), MExpr())
      << "1 * 0 == 0";
  EXPECT_EQ(MExpr({1}, {1}) *= MExpr({2}, {1}),
          MExpr({2}, {1}))
      << "1 * 2 == 2";
  EXPECT_EQ(MExpr({1}, {1}) *= MExpr({1}, {2}),
          MExpr({1}, {2}))
      << "1 * 1/2 == 1/2";
  EXPECT_EQ(MExpr({2}, {1}) *= MExpr({1}, {2}),
          MExpr({1}, {1}))
      << "2 * 1/2 == 1";
  EXPECT_EQ(MExpr() *= MExpr({2}, {1}),
          MExpr())
      << "0 * 2 == 0";

  EXPECT_EQ(MExpr({1}, {1}) * MExpr({1}, {1}, {s0}),
          MExpr({1}, {1}, {s0}))
      << "1 * s0 == s0";
  EXPECT_EQ(MExpr({1}, {1}) * MExpr({1}, {1}, {}, {s0}),
          MExpr({1}, {1}, {}, {s0}))
      << "1 * 1/s0 == 1/s0";
  EXPECT_EQ(MExpr({1}, {1}, {s0}) * MExpr({1}, {1}, {}, {s0}),
          MExpr({1}, {1}))
      << "s0 * 1/s0 == 1";

  EXPECT_EQ(MExpr({1}, {1}) *= MExpr({1}, {1}, {s0}),
          MExpr({1}, {1}, {s0}))
      << "1 * s0 == s0";
  EXPECT_EQ(MExpr({1}, {1}) *= MExpr({1}, {1}, {}, {s0}),
          MExpr({1}, {1}, {}, {s0}))
      << "1 * 1/s0 == 1/s0";
  EXPECT_EQ(MExpr({1}, {1}, {s0}) *= MExpr({1}, {1}, {}, {s0}),
          MExpr({1}, {1}))
      << "s0 * 1/s0 == 1";
}

TEST(PolynomialSizeExpr, CombineLikeItems) {
  SizeVarId s0=0, s1=1;

  EXPECT_EQ(PolynomialSizeExpr({MExpr({3},{5}), MExpr({3},{10})}).CombineLikeItems(),
            PolynomialSizeExpr(MExpr({9},{10})))
      << "3/5 + 3/10 = 9/10";

  EXPECT_EQ(PolynomialSizeExpr({MExpr({1}, {1}, {s0}), MExpr({1},{1},{s0})}).CombineLikeItems(),
            PolynomialSizeExpr(MExpr({2},{1},{s0})))
      << "s0 + s0 == 2s0";

  EXPECT_EQ(PolynomialSizeExpr({MExpr({1}, {4}, {s0}), MExpr({1}, {4}, {s0})}).CombineLikeItems(),
            PolynomialSizeExpr({MExpr({1},{2},{s0})}))
      << "1/4*s0 +1/4*s0 = 1/2*s0";

  EXPECT_EQ(PolynomialSizeExpr({MExpr({1}, {1}, {s1}), MExpr({1}, {1}, {s0})}).CombineLikeItems(),
            PolynomialSizeExpr({MExpr({1}, {1}, {s0}), MExpr({1}, {1}, {s1})}))
      << "s1 + s0 will sort to s0 + s1";

  EXPECT_EQ(PolynomialSizeExpr({MExpr({1}, {1}, {s0}), MExpr({10}, {1})}).CombineLikeItems(),
            PolynomialSizeExpr({MExpr({10}, {1}), MExpr({1}, {1}, {s0})}))
      << "s0 + 10 will sort to 10 + s0";
}

TEST(PolynomialSizeExpr, Add) {
  using M = MonomialSizeExpr;
  using P = PolynomialSizeExpr;
  SizeVar s0{0}, s1{1};

  // PolynomialSizeExpr add Interger
  EXPECT_EQ(P(10) + 10, P(20));
  EXPECT_EQ(P(10) + 0, P(10));
  EXPECT_EQ(P(s0) + 10,
            P({M(10), M(s0)}));

  // PolynomialSizeExpr add MonomialSizeExpr
  EXPECT_EQ(P(10) + M(s0), P({M(10), M(s0)}));

  // PolynomialSizeExpr add PolynomialSizeExpr
  EXPECT_EQ((P({M{s0}, M(10)})
            + P({M(s1), M(20)})),
            P({M(30), M(s0), M(s1)})
            );

  // using operator
  EXPECT_EQ(s0 + 10, P({M(10), M(s0)}));
  EXPECT_EQ(10 + s0, P({M(10), M(s0)}));
  EXPECT_EQ(s0 + s1, P({M(s0), M(s1)}));
  EXPECT_EQ((s0 + s1) + 10, P({M(10), M(s0), M(s1)}));
  EXPECT_EQ(10 + (s0 + s1), P({M(10), M(s0), M(s1)}));
  EXPECT_EQ((10 + s0) + (10 + s1), P({M(20), M(s0), M(s1)}));
}

TEST(PolynomialSizeExpr, Divide) {
  SizeVar s0{0}, s1{1}, s2{2}, s3{3};

  EXPECT_EQ((s0 + s1) / 2, PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) / MonomialSizeExpr(2), PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) / PolynomialSizeExpr(2), PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) / MExpr({1},{2}), PolynomialSizeExpr({2*s0, 2*s1}));
  EXPECT_EQ((s0 + s1) / s1, 1 + s0/s1);
  EXPECT_EQ((s0 + s1) / (1 / s1), s0*s1 + s1*s1);

  EXPECT_EQ((s0 + s1) / (s2 + s3), (s0/s2 + s0/s3 + s1/s2 + s1/s3));
  EXPECT_EQ((s0 + s1) / (1/s2 + 1/s3), (s0*s2 + s0*s3 + s1*s2 + s1*s3));

  EXPECT_EQ(PolynomialSizeExpr() / 2, PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() / s0, PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() / MonomialSizeExpr(s0), PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() / PolynomialSizeExpr(s0), PolynomialSizeExpr());

  ASSERT_ANY_THROW((s0 + s1) / 0);
  ASSERT_ANY_THROW((s0 + s1) / MonomialSizeExpr());
  ASSERT_ANY_THROW((s0 + s1) / PolynomialSizeExpr());

  EXPECT_EQ((s0 + s1) /= 2, PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) /= MonomialSizeExpr(2), PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) /= PolynomialSizeExpr(2), PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) /= MExpr({1},{2}), PolynomialSizeExpr({2*s0, 2*s1}));
  EXPECT_EQ((s0 + s1) /= s1, 1 + s0/s1);
  EXPECT_EQ((s0 + s1) /= (1 / s1), s0*s1 + s1*s1);

  EXPECT_EQ((s0 + s1) /= (s2 + s3), (s0/s2 + s0/s3 + s1/s2 + s1/s3));
  EXPECT_EQ((s0 + s1) /= (1/s2 + 1/s3), (s0*s2 + s0*s3 + s1*s2 + s1*s3));

  EXPECT_EQ(PolynomialSizeExpr() /= 2, PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() /= s0, 0);
  EXPECT_EQ(PolynomialSizeExpr() /= MonomialSizeExpr(s0), 0);
  EXPECT_EQ(PolynomialSizeExpr() /= PolynomialSizeExpr(s0), 0);

  ASSERT_ANY_THROW((s0 + s1) /= 0);
  ASSERT_ANY_THROW((s0 + s1) /= MonomialSizeExpr());
  ASSERT_ANY_THROW((s0 + s1) /= PolynomialSizeExpr());
}

TEST(PolynomialSizeExpr, Multiply) {
  SizeVar s0{0}, s1{1}, s2{2}, s3{3};

  EXPECT_EQ((s0 + s1) * 2, PolynomialSizeExpr({s0*2, s1*2}));
  EXPECT_EQ((s0 + s1) * MonomialSizeExpr(2), PolynomialSizeExpr({s0*2, s1*2}));
  EXPECT_EQ((s0 + s1) * PolynomialSizeExpr(2), PolynomialSizeExpr({s0*2, s1*2}));
  EXPECT_EQ((s0 + s1) * MExpr({1},{2}), PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) * s1, s0*s1 + s1*s1);
  EXPECT_EQ((s0 + s1) * (1 / s1), 1 + s0/s1);

  EXPECT_EQ((s0 + s1) * (s2 + s3), (s0*s2 + s0*s3 + s1*s2 + s1*s3));
  EXPECT_EQ((s0 + s1) * (1/s2 + 1/s3), (s0/s2 + s0/s3 + s1/s2 + s1/s3));

  EXPECT_EQ(PolynomialSizeExpr() * 2, PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() * s0, PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() * MonomialSizeExpr(s0), PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() * PolynomialSizeExpr(s0), PolynomialSizeExpr());

  EXPECT_EQ((s0 + s1) * 0, PolynomialSizeExpr());
  EXPECT_EQ((s0 + s1) * MonomialSizeExpr(), PolynomialSizeExpr());
  EXPECT_EQ((s0 + s1) * PolynomialSizeExpr(), PolynomialSizeExpr());

  EXPECT_EQ((s0 + s1) *= 2, PolynomialSizeExpr({s0*2, s1*2}));
  EXPECT_EQ((s0 + s1) *= MonomialSizeExpr(2), PolynomialSizeExpr({s0*2, s1*2}));
  EXPECT_EQ((s0 + s1) *= PolynomialSizeExpr(2), PolynomialSizeExpr({s0*2, s1*2}));
  EXPECT_EQ((s0 + s1) *= MExpr({1},{2}), PolynomialSizeExpr({s0/2, s1/2}));
  EXPECT_EQ((s0 + s1) *= s1, s0*s1 + s1*s1);
  EXPECT_EQ((s0 + s1) *= (1 / s1), 1 + s0/s1);

  EXPECT_EQ((s0 + s1) *= (s2 + s3), (s0*s2 + s0*s3 + s1*s2 + s1*s3));
  EXPECT_EQ((s0 + s1) *= (1/s2 + 1/s3), (s0/s2 + s0/s3 + s1/s2 + s1/s3));

  EXPECT_EQ(PolynomialSizeExpr() *= 2, PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() *= s0, PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() *= MonomialSizeExpr(s0), PolynomialSizeExpr());
  EXPECT_EQ(PolynomialSizeExpr() *= PolynomialSizeExpr(s0), PolynomialSizeExpr());

  EXPECT_EQ((s0 + s1) *= 0, PolynomialSizeExpr());
  EXPECT_EQ((s0 + s1) *= MonomialSizeExpr(), PolynomialSizeExpr());
  EXPECT_EQ((s0 + s1) *= PolynomialSizeExpr(), PolynomialSizeExpr());
}

TEST(Ascir, CanGetPeer_FromInput) {
  Data x1("x1"), x2("x2");

  x2.x = x1;
  x2.y.axis = {0, 1, 2};

  ascir::Graph graph("test_graph");
  graph.SetInputs({x1});

  auto result_x1 = graph.Find("x1");
  auto result_x2 = graph.Find("x2");
  EXPECT_EQ(result_x2.inputs[0]->Owner(), result_x1);
}

TEST(Ascir, ListPolynomialSizeExpr_SaveToAttr) {
  static constexpr char TEST_ATTR[] = "test_attr";
  union {
    ge::OpDesc *holder;
    AttrField<ge::AttrHolder *, TEST_ATTR, std::vector<PolynomialSizeExpr>> exprs;
  } attrs;

  auto op = ge::OpDesc("test", "test");
  attrs.holder = &op;

  SizeVar s0{0}, s1{1}, s2{2}, s3{3};
  std::vector exprs = {
    PolynomialSizeExpr(),
    PolynomialSizeExpr(2),
    PolynomialSizeExpr(1) / 2,
    s0 + s1,
    1 + s0 + s1,
    s0 * s1 + s2, // 分子分母数量不同
    s0 * s1 / s2 + s3 / s0 * s1, // 分子分母数量不同
  };

  attrs.exprs = exprs;

  auto result = attrs.exprs();
  auto iter = result.begin();
  EXPECT_EQ(*iter++, PolynomialSizeExpr());
  EXPECT_EQ(*iter++, PolynomialSizeExpr(2));
  EXPECT_EQ(*iter++, PolynomialSizeExpr(1) / 2);
  EXPECT_EQ(*iter++, s0 + s1);
  EXPECT_EQ(*iter++, 1 + s0 + s1);
  EXPECT_EQ(*iter++, s0 * s1 + s2); // 分子分母数量不同
  EXPECT_EQ(*iter++, s0 * s1 / s2 + s3 / s0 * s1); // 分子分母数量不同
}

TEST(Ascir_AxisOperations, CreateSize_WillSetSizeTable_ToGraphAttr) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);

  EXPECT_EQ(s0.id, 0);
  EXPECT_EQ(s0.name, "s0");
  EXPECT_EQ(s0.type, ascir::SizeVar::SIZE_TYPE_VAR);
  EXPECT_EQ(s1.id, 1);
  EXPECT_EQ(s1.name, "s1");
  EXPECT_EQ(s1.type, ascir::SizeVar::SIZE_TYPE_CONST);
  EXPECT_EQ(s1.value, 100);

  auto result_graph = ge::GraphUtilsEx::GetComputeGraph(graph);

  vector<string> result_size_names;
  vector<int64_t> result_size_types;
  vector<int64_t> result_size_values;
  ge::AttrUtils::GetListStr(result_graph, graph.size_var.NAME, result_size_names);
  ge::AttrUtils::GetListInt(result_graph, graph.size_var.TYPE, result_size_types);
  ge::AttrUtils::GetListInt(result_graph, graph.size_var.VALUE, result_size_values);

  EXPECT_EQ(result_size_names[s0.id], "s0");
  EXPECT_EQ(result_size_types[s0.id], ascir::SizeVar::SIZE_TYPE_VAR);

  EXPECT_EQ(result_size_names[s1.id], "s1");
  EXPECT_EQ(result_size_types[s1.id], ascir::SizeVar::SIZE_TYPE_CONST);
  EXPECT_EQ(result_size_values[s1.id], 100);
}

TEST(Ascir_AxisOperations, CreateAxis_WillSetAxisTable_ToGraphAttr) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0 / s1);

  EXPECT_EQ(z0.id, 0);
  EXPECT_EQ(z0.name, "z0");
  EXPECT_EQ(z0.type, ascir::Axis::Type::kAxisTypeOriginal);
  EXPECT_EQ(z0.size, s0);
  EXPECT_EQ(z0.from, vector<ascir::AxisId>{});
  EXPECT_EQ(z0.split_peer, ID_NONE);

  EXPECT_EQ(z1.id, 1);
  EXPECT_EQ(z1.name, "z1");
  EXPECT_EQ(z1.type, ascir::Axis::Type::kAxisTypeOriginal);
  EXPECT_EQ(z1.size, s0 / s1);
  EXPECT_EQ(z1.from, vector<ascir::AxisId>{});
  EXPECT_EQ(z1.split_peer, ID_NONE);

  auto result_axis = graph.axis();
  EXPECT_EQ(z0.name, result_axis[z0.id].name);
  EXPECT_EQ(z0.type, result_axis[z0.id].type);
  EXPECT_EQ(z0.size, result_axis[z0.id].size);
  EXPECT_EQ(z0.from, result_axis[z0.id].from);
  EXPECT_EQ(z0.split_peer, result_axis[z0.id].split_peer);
  EXPECT_EQ(z1.name, result_axis[z1.id].name);
  EXPECT_EQ(z1.type, result_axis[z1.id].type);
  EXPECT_EQ(z1.size, result_axis[z1.id].size);
  EXPECT_EQ(z1.from, result_axis[z1.id].from);
  EXPECT_EQ(z1.split_peer, result_axis[z1.id].split_peer);
}

TEST(Ascir_AxisOperations, BlockSplit_WillCreate_BlockOutAndInAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  auto [z0_out, z0_in] = graph.BlockSplit(z0.id);
  auto block_size = *graph.size_var().rbegin();

  auto result_z0_out = graph.axis[z0_out.id];
  auto result_z0_in = graph.axis[z0_in.id];
  EXPECT_EQ(result_z0_in.name, "z0b");
  EXPECT_EQ(result_z0_in.type, ascir::Axis::Type::kAxisTypeBlockInner);
  ASSERT_EQ(result_z0_in.from.size(), 1);
  EXPECT_EQ(result_z0_in.from[0], z0.id);
  ASSERT_EQ(result_z0_in.size, block_size);
  EXPECT_EQ(result_z0_in.split_peer, result_z0_out.id);

  EXPECT_EQ(result_z0_out.name, "z0B");
  EXPECT_EQ(result_z0_out.type, ascir::Axis::Type::kAxisTypeBlockOuter);
  ASSERT_EQ(result_z0_out.from.size(), 1);
  EXPECT_EQ(result_z0_out.from[0], z0.id);
  EXPECT_EQ(result_z0_out.size, s0/block_size);
  EXPECT_EQ(result_z0_out.split_peer, result_z0_in.id);

  EXPECT_EQ(block_size.name, "z0b_size");
}

TEST(Ascir_AxisOperations, TileSplit_WillCreate_TileOutAndInAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  auto [z0_out, z0_in] = graph.TileSplit(z0.id);
  auto tile_size = *graph.size_var().rbegin();

  auto result_z0_out = graph.axis[z0_out.id];
  auto result_z0_in = graph.axis[z0_in.id];
  EXPECT_EQ(result_z0_in.name, "z0t");
  EXPECT_EQ(result_z0_in.type, ascir::Axis::Type::kAxisTypeTileInner);
  ASSERT_EQ(result_z0_in.from.size(), 1);
  EXPECT_EQ(result_z0_in.from[0], z0.id);
  EXPECT_EQ(result_z0_in.size, tile_size);
  EXPECT_EQ(result_z0_in.split_peer, result_z0_out.id);

  EXPECT_EQ(result_z0_out.name, "z0T");
  EXPECT_EQ(result_z0_out.type, ascir::Axis::Type::kAxisTypeTileOuter);
  ASSERT_EQ(result_z0_out.from.size(), 1);
  EXPECT_EQ(result_z0_out.from[0], z0.id);
  EXPECT_EQ(result_z0_out.size, s0/tile_size);
  EXPECT_EQ(result_z0_out.split_peer, result_z0_in.id);

  EXPECT_EQ(tile_size.name, "z0t_size");
}

TEST(Ascir_AxisOperations, MergeAxis_WillCreate_MergedAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0/s1);
  auto z1 = graph.CreateAxis("z1", s2/s3);

  auto z3 = graph.MergeAxis({z0.id, z1.id});

  auto result_z3 = graph.axis[z3.id];
  EXPECT_EQ(result_z3.name, "z0z1");
  EXPECT_EQ(result_z3.type, ascir::Axis::Type::kAxisTypeMerged);
  EXPECT_EQ(result_z3.from.size(), 2);
  EXPECT_EQ(result_z3.from[0], z0.id);
  EXPECT_EQ(result_z3.from[1], z1.id);
  EXPECT_EQ(result_z3.size, s0/s1*s2/s3);
}

TEST(Ascir_AxisOperations, ApplySplit_OnNode_WillSplitNodeAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data data("test_op");

  data.attr.sched.axis = {z0.id};
  data.y.axis = {z0.id};
  data.y.repeats = {s0};
  data.y.strides = {1};
  graph.SetInputs({data});

  auto result_op = graph.Find("test_op");

  auto [z0_out, z0_in] = graph.TileSplit(z0.id);
  graph.ApplySplit(result_op, z0_out.id, z0_in.id, z0.id);

  auto tile_size = z0_in.size;
  EXPECT_EQ(result_op.attr.sched.axis[0], z0_out.id);
  EXPECT_EQ(result_op.attr.sched.axis[1], z0_in.id);
  EXPECT_EQ(result_op.outputs[0].axis[0], z0_out.id);
  EXPECT_EQ(result_op.outputs[0].axis[1], z0_in.id);
  EXPECT_EQ(result_op.outputs[0].repeats[0], z0.size / tile_size);
  EXPECT_EQ(result_op.outputs[0].repeats[1], tile_size);
  EXPECT_EQ(result_op.outputs[0].strides[0], tile_size);
  EXPECT_EQ(result_op.outputs[0].strides[1], 1);
}

TEST(Ascir_AxisOperations, ApplySplitWithoutOriginal_OnNode_WillSplitNodeAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data data("test_op");

  data.attr.sched.axis = {z0.id};
  data.y.axis = {z0.id};
  data.y.repeats = {s0};
  data.y.strides = {1};
  graph.SetInputs({data});

  auto result_op = graph.Find("test_op");

  auto [z0_out, z0_in] = graph.TileSplit(z0.id);
  graph.ApplySplit(result_op, z0_out.id, z0_in.id);

  auto tile_size = z0_in.size;
  EXPECT_EQ(result_op.attr.sched.axis[0], z0_out.id);
  EXPECT_EQ(result_op.attr.sched.axis[1], z0_in.id);
  EXPECT_EQ(result_op.outputs[0].axis[0], z0_out.id);
  EXPECT_EQ(result_op.outputs[0].axis[1], z0_in.id);
  EXPECT_EQ(result_op.outputs[0].repeats[0], z0.size / tile_size);
  EXPECT_EQ(result_op.outputs[0].repeats[1], tile_size);
  EXPECT_EQ(result_op.outputs[0].strides[0], tile_size);
  EXPECT_EQ(result_op.outputs[0].strides[1], 1);
}

TEST(Ascir_AxisOperations, ApplyMerge_OnNode_WillMergeNodeAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("test_op");

  data.attr.sched.axis = {z0.id, z1.id};
  data.y.axis = {z0.id, z1.id};
  data.y.repeats = {s0, s1};
  data.y.strides = {s1, 1};
  graph.SetInputs({data});

  auto result_op = graph.Find("test_op");

  auto z_new = graph.MergeAxis({z0.id, z1.id});
  graph.ApplySchedAxisMerge(result_op, z_new.id, {z0.id, z1.id});
  graph.ApplyTensorAxisMerge(result_op, z_new.id, {z0.id, z1.id});

  EXPECT_EQ(result_op.attr.sched.axis[0], z_new.id);
  EXPECT_EQ(result_op.outputs[0].axis[0], z_new.id);
  EXPECT_EQ(result_op.outputs[0].repeats[0], z0.size * z1.size);
  EXPECT_EQ(result_op.outputs[0].strides[0], 1);
}

TEST(Ascir_AxisOperations, ApplyMergeWithoutOriginal_OnNode_WillMergeNodeAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("test_op");

  data.attr.sched.axis = {z0.id, z1.id};
  data.y.axis = {z0.id, z1.id};
  data.y.repeats = {s0, s1};
  data.y.strides = {s1, 1};
  graph.SetInputs({data});

  auto result_op = graph.Find("test_op");

  auto z_new = graph.MergeAxis({z0.id, z1.id});
  graph.ApplyMerge(result_op, z_new.id);

  EXPECT_EQ(result_op.attr.sched.axis[0], z_new.id);
  EXPECT_EQ(result_op.outputs[0].axis[0], z_new.id);
  EXPECT_EQ(result_op.outputs[0].repeats[0], z0.size * z1.size);
  EXPECT_EQ(result_op.outputs[0].strides[0], 1);
}

TEST(Ascir_AxisOperations, ApplyReorder_OnNode_WillChangeAxisAndStrideOrder) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("test_op");

  data.attr.sched.axis = {z0.id, z1.id, z2.id};
  data.y.axis = {z0.id, z1.id, z2.id};
  data.y.repeats = {s0, s1, s2};
  data.y.strides = {s1*s2, s2, 1};
  graph.SetInputs({data});

  auto result_op = graph.Find("test_op");

  graph.ApplyReorder(result_op, {z2.id, z0.id, z1.id});

  EXPECT_EQ(result_op.attr.sched.axis[0], z2.id);
  EXPECT_EQ(result_op.attr.sched.axis[1], z0.id);
  EXPECT_EQ(result_op.attr.sched.axis[2], z1.id);

  EXPECT_EQ(result_op.outputs[0].axis[0], z2.id);
  EXPECT_EQ(result_op.outputs[0].repeats[0], z2.size);
  EXPECT_EQ(result_op.outputs[0].strides[0], 1);
  EXPECT_EQ(result_op.outputs[0].axis[1], z0.id);
  EXPECT_EQ(result_op.outputs[0].repeats[1], z0.size);
  EXPECT_EQ(result_op.outputs[0].strides[1], s1*s2);
  EXPECT_EQ(result_op.outputs[0].axis[2], z1.id);
  EXPECT_EQ(result_op.outputs[0].repeats[2], z1.size);
  EXPECT_EQ(result_op.outputs[0].strides[2], s2);
}

TEST(Ascir_AxisOperations, AxisUpdate) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  EXPECT_EQ(z0.align, ge::Symbol(1));
  EXPECT_EQ(z0.allow_oversize_axis, false);
  EXPECT_EQ(z0.allow_unaligned_tail, true);
  graph.UpdateAxisAlign(z0.id, ge::Symbol(10));
  graph.UpdateAxisAllowOversizeAxis(z0.id, true);
  graph.UpdateAxisAllowUnalignedTail(z0.id, false);
  auto z0_new = graph.axis[z0.id];
  EXPECT_EQ(z0_new.align, ge::Symbol(10));
  EXPECT_EQ(z0_new.allow_oversize_axis, true);
  EXPECT_EQ(z0_new.allow_unaligned_tail, false);
}

TEST(Ascir_SparseOperations, CreateSparse) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  auto s1 = graph.CreateSizeVar("s1");
  auto z1 = graph.CreateAxis("z1", s1);

  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  Sparse::Value val(Sparse::SPARSE_TYPE_OBLIQUE_BAND);
  val.ob.pre.id = z0.id;
  val.ob.pre.position = s2;
  val.ob.next.id = z1.id;
  val.ob.next.position = s3;
  auto sparse = graph.CreateSparse(Sparse::SPARSE_TYPE_OBLIQUE_BAND, val);
  EXPECT_EQ(sparse.type, Sparse::SPARSE_TYPE_OBLIQUE_BAND);
  EXPECT_EQ(sparse.value.ob.pre.id, z0.id);
  bool res = (sparse.value.ob.pre.position == s2);
  EXPECT_TRUE(res);
  EXPECT_EQ(sparse.value.ob.next.id, z1.id);
  res = (sparse.value.ob.next.position == s3);
  EXPECT_TRUE(res);
  auto get_sparse = graph.sparse[sparse.id];
  EXPECT_EQ(get_sparse.type, Sparse::SPARSE_TYPE_OBLIQUE_BAND);
  EXPECT_EQ(get_sparse.value.ob.pre.id, z0.id);
  res = (get_sparse.value.ob.pre.position == s2);
  EXPECT_TRUE(res);
  EXPECT_EQ(get_sparse.value.ob.next.id, z1.id);
  res = (get_sparse.value.ob.next.position == s3);
  EXPECT_TRUE(res);
}

TEST(Ascir_SparseOperations, CreateObliqueBandSparse) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  auto s1 = graph.CreateSizeVar("s1");
  auto z1 = graph.CreateAxis("z1", s1);

  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  Sparse::AxisInfo pre_axis;
  pre_axis.id = z0.id;
  pre_axis.position = s2;
  Sparse::AxisInfo next_axis;
  next_axis.id = z1.id;
  next_axis.position = s3;
  auto sparse = graph.CreateObliqueBandSparse(pre_axis, next_axis);
  EXPECT_EQ(sparse.type, Sparse::SPARSE_TYPE_OBLIQUE_BAND);
  EXPECT_EQ(sparse.value.ob.pre.id, z0.id);
  bool res = (sparse.value.ob.pre.position == s2);
  EXPECT_TRUE(res);
  EXPECT_EQ(sparse.value.ob.next.id, z1.id);
  res = (sparse.value.ob.next.position == s3);
  EXPECT_TRUE(res);
  auto get_sparse = graph.sparse[sparse.id];
  EXPECT_EQ(get_sparse.type, Sparse::SPARSE_TYPE_OBLIQUE_BAND);
  EXPECT_EQ(get_sparse.value.ob.pre.id, z0.id);
  res = (get_sparse.value.ob.pre.position == s2);
  EXPECT_TRUE(res);
  EXPECT_EQ(get_sparse.value.ob.next.id, z1.id);
  res = (get_sparse.value.ob.next.position == s3);
  EXPECT_TRUE(res);
}

TEST(Ascir_Utils, DebugHintGraphStr_WillShowAxisInfo) {
  Data data("test_op");
  ascir::Graph graph("test_graph");
  graph.SetInputs({data});

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);
  auto s0_block = graph.CreateSizeVar("s0_block", 100);

  auto z0_out = graph.CreateAxis("z0_out", s0/s0_block);
  auto z0_in = graph.CreateAxis("z0_in", s0_block);
  auto z1 = graph.CreateAxis("z1", s1);

  Sparse::AxisInfo pre;
  pre.id = z0_out.id;
  pre.position = s0;
  Sparse::AxisInfo next;
  next.id = z0_in.id;
  next.position = s1;
  (void)graph.CreateObliqueBandSparse(pre, next);

  data.attr.sched.axis = {z0_out.id, z0_in.id, z1.id};

  data.y.axis = {z0_out.id, z0_in.id, z1.id};
  data.y.repeats = {s0 / s0_block, s0_block, s1};
  data.y.strides = {s0_block * s1, s1, 1};

  auto result_str = ascir::utils::DebugHintGraphStr(graph);
  EXPECT_EQ(result_str, string{"Graph: test_graph\n"
                               "Sizes:\n"
                               "  s0: VAR\n"
                               "  s1: CONST(100)\n"
                               "  s0_block: CONST(100)\n"
                               "Axis:\n"
                               "  z0_out: s0 / s0_block, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n"
                               "  z0_in: s0_block, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n"
                               "  z1: s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n"
                               "Sparse:\n"
                               "  type: OBLIQUE_BAND, value: {pre: {z0_out, s0}, next: {z0_in, s1}}\n"
                               "Nodes:\n"
                               "  test_op: Data (0)\n"
                               "    .axis = {z0_out, z0_in, z1, }\n"
                               "    .hint:\n"
                               "      .compute_type = data\n"
                               "    .x = nil\n"
                               "    .y.dtype = float32\n"
                               "    .y.axis = {z0_out, z0_in, z1, }\n"
                               "    .y.repeats = {s0 / s0_block, s0_block, s1, }\n"
                               "    .y.strides = {s1 * s0_block, s1, 1, }\n"
                               "    .y.vectorized_axis = {}\n"});
}

TEST(Ascir_Utils, DebugImplGraphStr) {
  Data data("test_op");
  ascir::Graph graph("test_graph");
  graph.SetInputs({data});

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);
  auto s0_block = graph.CreateSizeVar("s0_block", 100);

  auto z0_out = graph.CreateAxis("z0_out", s0/s0_block);
  auto z0_in = graph.CreateAxis("z0_in", s0_block);
  auto z1 = graph.CreateAxis("z1", s1);

  auto data_op = graph.Find(data.GetName().c_str());
  data_op.attr.api.type = ascir::API_TYPE_BUFFER;
  data_op.attr.api.type = ascir::UNIT_NONE;
  data_op.attr.sched.axis = {z0_out.id, z0_in.id, z1.id};

  auto data_y = data_op.outputs[0];
  data_y.axis = {z0_out.id, z0_in.id, z1.id};
  data.y.repeats = {s0 / s0_block, s0_block, s1};
  data.y.strides = {s0_block * s1, s1, 1};
  data_y.mem.tensor_id = 0;
  data_y.mem.alloc_type = ALLOC_TYPE_QUEUE;
  data_y.mem.hardware = MEM_HARDWARE_UB;
  data_y.mem.position = POSITION_VECIN;
  data_y.buf.id = ID_NONE;
  data_y.que.id = 0;
  data_y.que.depth = 2;
  data_y.que.buf_num = 2;
  data_y.opt.reuse_id = ID_NONE;
  data_y.opt.ref_tensor = ID_NONE;
  data_y.opt.merge_scope = ID_NONE;

  auto result_str = ascir::utils::DebugImplGraphStr(graph);

  EXPECT_EQ(result_str, string{
                            "Graph: test_graph\n"
                            "Sizes:\n"
                            "  s0: VAR\n"
                            "  s1: CONST(100)\n"
                            "  s0_block: CONST(100)\n"
                            "Axis:\n"
                            "  z0_out: s0 / s0_block, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n"
                            "  z0_in: s0_block, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n"
                            "  z1: s1, ORIGINAL, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1\n"
                            "Nodes:\n"
                            "  test_op: Data (0)\n"
                            "    .axis = {z0_out, z0_in, z1, }\n"
                            "    .loop_axis = z0_out\n"
                            "    .hint:\n"
                            "      .compute_type = data\n"
                            "    .api:\n"
                            "      .type = Buffer\n"
                            "      .unit = None\n"
                            "    .x = nil\n"
                            "    .y.dtype = float32\n"
                            "    .y.axis = {z0_out, z0_in, z1, }\n"
                            "    .y.repeats = {s0 / s0_block, s0_block, s1, }\n"
                            "    .y.strides = {s1 * s0_block, s1, 1, }\n"
                            "    .y.vectorized_axis = {}\n"
                            "    .y.mem:\n"
                            "      .tensor_id = 0\n"
                            "      .alloc_type = Queue\n"
                            "      .hardware = UB\n"
                            "      .position = TPosition::VECIN\n"
                            "      .buf_ids = {}\n"
                            "    .y.que:\n"
                            "      .id = 0\n"
                            "      .depth = 2\n"
                            "      .buf_num = 2\n"
                            "    .y.opt:\n"
                            "      .reuse_id = nil\n"
                            "      .ref_tensor = nil\n"
                            "      .merge_scope = nil\n"
                            });
}
