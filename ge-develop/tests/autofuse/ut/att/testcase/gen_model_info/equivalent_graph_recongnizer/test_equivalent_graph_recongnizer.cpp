/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "gen_tiling_impl.h"
#include "graph_construct_utils.h"
#define private public
#include "reuse_group_utils/equivalent_graph_recognizer.h"
#undef private
#include "reuse_group_utils/reuse_group_utils.h"
using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1;
using namespace ge;
using HintGraph=AscGraph;
}
using namespace att;

class EquivalentGraphRecongnizerUnitTest : public ::testing::Test {
public:
 void SetUp() override {
   // Code here will be called immediately after the constructor (right
   // before each test).
//   dlog_setlevel(GE, 0, 1);
 }

 void TearDown() override {
   // Code here will be called immediately after each test (right
   // before the destructor).
 }
};

namespace ge {
namespace ascir {
namespace cg {
Status BuildAscendGraphTest1(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", nd.id, ndB->id, ndb->id, ndbT->id, ndbt->id);
  GELOGD("axis id = [%d, %d]", ndB->id, ndbT->id);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest2(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto abs = Abs("abs", load2_perm).TQue(Position::kPositionVecCalc, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", abs);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, abs, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest3(ge::AscGraph &graph) {
  auto A = ge::Symbol("A");
  auto ND = ge::Symbol("ND");
  auto a = graph.CreateAxis("a", A);
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest4(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto exp = Exp("exp", load2_perm).TQue(Position::kPositionVecCalc, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", exp);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, exp, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest5(ge::AscGraph &graph) {
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT16, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  auto node = graph.FindNode("input1");
  GE_ASSERT_NOTNULL(node);
  node->outputs[0].attr.dtype = DT_FLOAT16;
  return ge::SUCCESS;
}

Status BuildAscendGraphTest6(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndT, ndt] = graph.TileSplit(nd.id);
  auto [ndTB, ndTb] = graph.BlockSplit(ndt->id);
  std::swap(ndTB->id, ndT->id);
  std::swap(ndTb->id, ndt->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", nd.id, ndT->id, ndt->id, ndTB->id, ndTb->id);
  GELOGD("axis id = [%d, %d]", ndTB->id, ndT->id);
  LOOP(*ndTB) {
    LOOP(*ndT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndTB, *ndT, *ndTb, *ndt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_Equal(ge::AscGraph &graph) {
  // create default axis
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [ndB, ndb] = graph.BlockSplit(z0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", z0.id, ndB->id, ndb->id, ndbT->id, ndbt->id);
  GELOGD("axis id = [%d, %d]", ndB->id, ndbT->id);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest7_AddDefaultInput(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [ndB, ndb] = graph.BlockSplit(z0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", z0.id, ndB->id, ndb->id, ndbT->id, ndbt->id);
  GELOGD("axis id = [%d, %d]", ndB->id, ndbT->id);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto add = Add("add", load1, load2_perm).TQue(Position::kPositionVecCalc, 1, 2);
      auto store1 = Store("store1", load1);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt},
                                                                     {load1, load2_perm, add, store1}, 1));
      auto output1 = Output("output1", store1);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest7_AddSwapInput(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto S1 = ge::Symbol("S1");
  auto z0 = graph.CreateAxis("z0", S0);
  auto z1 = graph.CreateAxis("z1", S1);
  auto [ndB, ndb] = graph.BlockSplit(z0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z1});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", z0.id, ndB->id, ndb->id, ndbT->id, ndbt->id);
  GELOGD("axis id = [%d, %d]", ndB->id, ndbT->id);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto add = Add("add", load2_perm, load1).TQue(Position::kPositionVecCalc, 1, 2);
      auto store1 = Store("store1", load1);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt},
                                                                     {load1, load2_perm, add, store1}, 1));
      auto output1 = Output("output1", store1);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_ConstND(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol(20);
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", nd.id, ndB->id, ndb->id, ndbT->id, ndbt->id);
  GELOGD("axis id = [%d, %d]", ndB->id, ndbT->id);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt},
                                                                     {load1, load2_perm, store1, store2}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_EqualValue(ge::AscGraph &graph) {
  // create default axis
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [ndB, ndb] = graph.BlockSplit(z0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", z0.id, ndB->id, ndb->id, ndbT->id, ndbt->id);
  GELOGD("axis id = [%d, %d]", ndB->id, ndbT->id);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_NotEqualValue(ge::AscGraph &graph) {
  // create default axis
  auto S0 = ge::Symbol(40) * ge::Symbol("ND");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [ndB, ndb] = graph.BlockSplit(z0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  GELOGD("axis id = [%d, %d, %d, %d, %d]", z0.id, ndB->id, ndb->id, ndbT->id, ndbt->id);
  GELOGD("axis id = [%d, %d]", ndB->id, ndbT->id);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt}, {load1, load2_perm, store1, store2}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_NotEqualValue2(ge::AscGraph &graph) {
  GE_ASSERT_SUCCESS(BuildAscendGraphTest1_NotEqualValue(graph));
  auto node = graph.FindNode("input1");
  GE_ASSERT_NOTNULL(node);
  node->outputs[0].attr.repeats.emplace_back(ge::Symbol(21));
  node->outputs[0].attr.repeats.emplace_back(ge::Symbol(22));
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_NotEqualValue3(ge::AscGraph &graph) {
  GE_ASSERT_SUCCESS(BuildAscendGraphTest1_NotEqualValue(graph));
  auto node = graph.FindNode("input1");
  GE_ASSERT_NOTNULL(node);
  node->outputs[0].attr.repeats.emplace_back(ge::Symbol(222));
  node->outputs[0].attr.repeats.emplace_back(ge::Symbol(222));
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_InputAxes1(ge::AscGraph &graph) {
  // create default axis
  auto S0 = ge::Symbol("S0");
  auto S1 = ge::Symbol("S1");
  auto S2 = ge::Symbol("S2");
  auto z0 = graph.CreateAxis("z0", S0); // 0
  auto z1 = graph.CreateAxis("z1", S1); // 1
  auto z2 = graph.CreateAxis("z2", S2); // 2
  auto [ndB, ndb] = graph.BlockSplit(z0.id); // 3,4
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id); // 5,6
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z1});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt}, {load1, load2_perm, store1, store2}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildAscendGraphTest1_InputAxes2(ge::AscGraph &graph) {
  // create default axis
  auto S0 = ge::Symbol("S1");
  auto S1 = ge::Symbol("S0");
  auto S2 = ge::Symbol("S2");
  auto z0 = graph.CreateAxis("z0", S0); // 0
  auto z1 = graph.CreateAxis("z1", S1); // 1
  auto z2 = graph.CreateAxis("z2", S2); // 2
  auto [ndB, ndb] = graph.BlockSplit(z0.id); // 3,4
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id); // 5,6
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z1});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt}, {load1, load2_perm, store1, store2}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}
}
}
}
TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphGraphSizeNotEqual) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest2(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph1);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphSizeNotEqual) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest2(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphAxesNumNotEqual) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest3(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphApiTypeNotEqual) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest2(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest4(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphTensorDTypeNotEqual) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest5(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphAxisSplitNotEqual) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest6(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphAxisSplitNotEqual2) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph2), ge::SUCCESS);
  auto axis1_all = graph2.GetAllAxis();
  for (auto &axis : axis1_all) {
    if (axis->type == ge::Axis::Type::kAxisTypeOriginal) {
      axis->type = ge::Axis::Type::kAxisTypeMerged;
    }
  }
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphScheduleAxisNotEqual3) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph2), ge::SUCCESS);
  const auto input1 = graph2.FindNode("input1");
  ASSERT_NE(input1, nullptr);
  input1->attr.sched.axis = {1, 2, 3, 4, 5};
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphEqual1) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph1_equal");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_Equal(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("ND");
  group_info1.reuse_search_axes.emplace_back("ndb_size");
  group_info1.reuse_search_axes.emplace_back("ndbt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  EXPECT_TRUE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
  // init all_graphs_lists
  std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> all_graphs_lists; // schedule result->group->tiling cases
  all_graphs_lists.resize(1);
  std::vector<std::vector<ge::AscGraph>> groups1_graph;
  groups1_graph.emplace_back(group_graphs1);
  groups1_graph.emplace_back(group_graphs2);
  all_graphs_lists[0].emplace_back(groups1_graph);
  // init parsed_model_info
  FusedParsedScheduleResult all_model_infos;
  ParsedScheduleResult parsed_schedule_result;
  TilingModelInfo tiling_model_info1;
  tiling_model_info1.resize(1);
  TilingModelInfo tiling_model_info2;
  tiling_model_info2.resize(1);
  EXPECT_TRUE(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, tiling_model_info1) == ge::SUCCESS);
  EXPECT_TRUE(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 1UL}, tiling_model_info2) == ge::SUCCESS);
  all_model_infos[0][0].groups_tiling_model_info[0] = tiling_model_info1;
  all_model_infos[0][0].groups_tiling_model_info[1] = tiling_model_info2;
  all_model_infos[0][0].impl_graph_id = 0;
  EXPECT_TRUE(ReuseGroupUtils::MergeAllReusableGroups(all_graphs_lists, all_model_infos) == ge::SUCCESS);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphEqual1_with_AutofusePGOenv) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph1_equal");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_Equal(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("ND");
  group_info1.reuse_search_axes.emplace_back("ndb_size");
  group_info1.reuse_search_axes.emplace_back("ndbt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  EXPECT_TRUE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
  // init all_graphs_lists
  std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> all_graphs_lists; // schedule result->group->tiling cases
  all_graphs_lists.resize(1);
  std::vector<std::vector<ge::AscGraph>> groups1_graph;
  groups1_graph.emplace_back(group_graphs1);
  groups1_graph.emplace_back(group_graphs2);
  all_graphs_lists[0].emplace_back(groups1_graph);
  // init parsed_model_info
  FusedParsedScheduleResult all_model_infos;
  ParsedScheduleResult parsed_schedule_result;
  TilingModelInfo tiling_model_info1;
  tiling_model_info1.resize(1);
  TilingModelInfo tiling_model_info2;
  tiling_model_info2.resize(1);
  EXPECT_TRUE(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, tiling_model_info1) == ge::SUCCESS);
  EXPECT_TRUE(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 1UL}, tiling_model_info2) == ge::SUCCESS);
  all_model_infos[0][0].groups_tiling_model_info[0] = tiling_model_info1;
  all_model_infos[0][0].groups_tiling_model_info[1] = tiling_model_info2;
  all_model_infos[0][0].impl_graph_id = 0;
  EXPECT_TRUE(ReuseGroupUtils::MergeAllReusableGroups(all_graphs_lists, all_model_infos) == ge::SUCCESS);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphEqual2) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes2(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("S0");
  group_info1.reuse_input_axes.emplace_back("S1");
  group_info1.reuse_input_axes.emplace_back("S2");
  group_info1.reuse_search_axes.emplace_back("z0b_size");
  group_info1.reuse_search_axes.emplace_back("z0bt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_input_axes.emplace_back("S1");
  group_info2.reuse_input_axes.emplace_back("S2");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  // TTODO 暂时不支持轴映射不一致的场景，待后续TilingDataCopy整改后支持
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphNotEqualDuplicateMappedToDiffAxes) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes2(graph2), ge::SUCCESS);
  auto input1 = graph2.FindNode("input1");
  ASSERT_NE(input1, nullptr);
  input1->outputs[0].attr.repeats[0] = ge::Symbol("S2");
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("S0");
  group_info1.reuse_input_axes.emplace_back("S1");
  group_info1.reuse_input_axes.emplace_back("S2");
  group_info1.reuse_search_axes.emplace_back("z0b_size");
  group_info1.reuse_search_axes.emplace_back("z0bt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_input_axes.emplace_back("S1");
  group_info2.reuse_input_axes.emplace_back("S2");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphNotEqualDuplicateMappedFromDiffAxes) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes2(graph2), ge::SUCCESS);
  auto input1 = graph1.FindNode("input1");
  ASSERT_NE(input1, nullptr);
  input1->outputs[0].attr.repeats[0] = ge::Symbol("S2");
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("S0");
  group_info1.reuse_input_axes.emplace_back("S1");
  group_info1.reuse_input_axes.emplace_back("S2");
  group_info1.reuse_search_axes.emplace_back("z0b_size");
  group_info1.reuse_search_axes.emplace_back("z0bt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_input_axes.emplace_back("S1");
  group_info2.reuse_input_axes.emplace_back("S2");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphNotEqualDiffInputAxesSize) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes2(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("S0");
  group_info1.reuse_input_axes.emplace_back("S1");
  group_info1.reuse_search_axes.emplace_back("z0b_size");
  group_info1.reuse_search_axes.emplace_back("z0bt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_input_axes.emplace_back("S1");
  group_info2.reuse_input_axes.emplace_back("S2");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphNotEqualDiffNodeScheduleAxesSize) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph2");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_InputAxes2(graph2), ge::SUCCESS);
  auto input1 = graph1.FindNode("input1");
  ASSERT_NE(input1, nullptr);
  input1->attr.sched.axis = {1,2,3};
  auto input1_2 = graph2.FindNode("input1");
  ASSERT_NE(input1_2, nullptr);
  input1_2->attr.sched.axis = {1,2};
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("S0");
  group_info1.reuse_input_axes.emplace_back("S1");
  group_info1.reuse_input_axes.emplace_back("S2");
  group_info1.reuse_search_axes.emplace_back("z0b_size");
  group_info1.reuse_search_axes.emplace_back("z0bt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_input_axes.emplace_back("S1");
  group_info2.reuse_input_axes.emplace_back("S2");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphDifferentInput) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest7_AddDefaultInput(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph1_equal");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest7_AddSwapInput(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphDifferentExprType) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph1_equal");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_ConstND(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphSameExprValue) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph1_equal");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_EqualValue(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  group_info1.reuse_input_axes.emplace_back("ND");
  group_info1.reuse_search_axes.emplace_back("ndb_size");
  group_info1.reuse_search_axes.emplace_back("ndbt_size");
  ReuseScheduleGroupInfo group_info2;
  group_info2.reuse_input_axes.emplace_back("S0");
  group_info2.reuse_search_axes.emplace_back("z0b_size");
  group_info2.reuse_search_axes.emplace_back("z0bt_size");
  EXPECT_TRUE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, GroupGraphDifferentExprValue) {
  std::vector<ge::AscGraph> group_graphs1;
  ge::AscGraph graph1("graph");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_NotEqualValue2(graph1), ge::SUCCESS);
  group_graphs1.emplace_back(graph1);
  std::vector<ge::AscGraph> group_graphs2;
  ge::AscGraph graph2("graph1_equal");
  ASSERT_EQ(ge::ascir::cg::BuildAscendGraphTest1_NotEqualValue3(graph2), ge::SUCCESS);
  group_graphs2.emplace_back(graph2);
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EXPECT_FALSE(ReuseGroupUtils::IsGroupGraphsEquivalent(group_graphs1, group_graphs2, group_info1, group_info2));
}

TEST_F(EquivalentGraphRecongnizerUnitTest, IsTensorViewEquivalentViewNotSame) {
  ge::AscGraph graph1("graph");
  ge::AscGraph graph2("graph1_equal");
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EquivalentGraphRecognizer recognizer(graph1, graph2, group_info1, group_info2);
  ge::AscTensorAttr tensor1;
  tensor1.repeats.emplace_back(ge::Symbol(1));
  ge::AscTensorAttr tensor2;
  tensor2.repeats.emplace_back(ge::Symbol(1));
  tensor2.repeats.emplace_back(ge::Symbol(1));
  EXPECT_EQ(recognizer.IsTensorViewEquivalent(tensor1, tensor2), false);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, IsTensorViewEquivalentViewNotSame2) {
  ge::AscGraph graph1("graph");
  ge::AscGraph graph2("graph1_equal");
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EquivalentGraphRecognizer recognizer(graph1, graph2, group_info1, group_info2);
  ge::AscTensorAttr tensor1;
  tensor1.strides.emplace_back(ge::Symbol(1));
  ge::AscTensorAttr tensor2;
  tensor2.strides.emplace_back(ge::Symbol(1));
  tensor2.strides.emplace_back(ge::Symbol(1));
  EXPECT_EQ(recognizer.IsTensorViewEquivalent(tensor1, tensor2), false);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, IsTensorViewEquivalentViewNotSame3) {
  ge::AscGraph graph1("graph");
  ge::AscGraph graph2("graph1_equal");
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EquivalentGraphRecognizer recognizer(graph1, graph2, group_info1, group_info2);
  ge::AscTensorAttr tensor1;
  tensor1.vectorized_strides.emplace_back(ge::Symbol(1));
  ge::AscTensorAttr tensor2;
  tensor2.vectorized_strides.emplace_back(ge::Symbol(1));
  tensor2.vectorized_strides.emplace_back(ge::Symbol(1));
  EXPECT_EQ(recognizer.IsTensorViewEquivalent(tensor1, tensor2), false);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, IsTensorViewEquivalentStrideExprNotSame) {
  ge::AscGraph graph1("graph");
  ge::AscGraph graph2("graph1_equal");
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EquivalentGraphRecognizer recognizer(graph1, graph2, group_info1, group_info2);
  ge::AscTensorAttr tensor1;
  tensor1.repeats.emplace_back(ge::Symbol(10));
  tensor1.repeats.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(1));
  ge::AscTensorAttr tensor2;
  tensor2.repeats.emplace_back(ge::Symbol(10));
  tensor2.repeats.emplace_back(ge::Symbol(5));
  tensor2.strides.emplace_back(ge::Symbol(8));
  tensor2.strides.emplace_back(ge::Symbol(1));
  EXPECT_EQ(recognizer.IsTensorViewEquivalent(tensor1, tensor2), false);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, IsTensorViewEquivalentVectorizedStrideExprNotSame) {
  ge::AscGraph graph1("graph");
  ge::AscGraph graph2("graph1_equal");
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EquivalentGraphRecognizer recognizer(graph1, graph2, group_info1, group_info2);
  ge::AscTensorAttr tensor1;
  tensor1.repeats.emplace_back(ge::Symbol(10));
  tensor1.repeats.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(1));
  tensor1.vectorized_strides.emplace_back(ge::Symbol(5));
  tensor1.vectorized_strides.emplace_back(ge::Symbol(1));
  ge::AscTensorAttr tensor2;
  tensor2.repeats.emplace_back(ge::Symbol(10));
  tensor2.repeats.emplace_back(ge::Symbol(5));
  tensor2.strides.emplace_back(ge::Symbol(5));
  tensor2.strides.emplace_back(ge::Symbol(1));
  tensor2.vectorized_strides.emplace_back(ge::Symbol(8));
  tensor2.vectorized_strides.emplace_back(ge::Symbol(1));
  EXPECT_EQ(recognizer.IsTensorViewEquivalent(tensor1, tensor2), false);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, IsTensorViewEquivalentFreeSymbolsNotSameSize) {
  ge::AscGraph graph1("graph");
  ge::AscGraph graph2("graph1_equal");
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EquivalentGraphRecognizer recognizer(graph1, graph2, group_info1, group_info2);
  ge::AscTensorAttr tensor1;
  tensor1.repeats.emplace_back(ge::Symbol(10) * ge::Symbol("ND") + ge::Symbol("A") * ge::Symbol("B"));
  tensor1.repeats.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(1));
  tensor1.vectorized_strides.emplace_back(ge::Symbol(5));
  tensor1.vectorized_strides.emplace_back(ge::Symbol(1));
  ge::AscTensorAttr tensor2;
  tensor2.repeats.emplace_back(ge::Symbol(10) / ge::Symbol("B"));
  tensor2.repeats.emplace_back(ge::Symbol(5));
  tensor2.strides.emplace_back(ge::Symbol(5));
  tensor2.strides.emplace_back(ge::Symbol(1));
  tensor2.vectorized_strides.emplace_back(ge::Symbol(5));
  tensor2.vectorized_strides.emplace_back(ge::Symbol(1));
  EXPECT_EQ(recognizer.IsTensorViewEquivalent(tensor1, tensor2), false);
}

TEST_F(EquivalentGraphRecongnizerUnitTest, IsTensorViewEquivalentFreeSymbolsNotSameType) {
  ge::AscGraph graph1("graph");
  ge::AscGraph graph2("graph1_equal");
  ReuseScheduleGroupInfo group_info1;
  ReuseScheduleGroupInfo group_info2;
  EquivalentGraphRecognizer recognizer(graph1, graph2, group_info1, group_info2);
  ge::AscTensorAttr tensor1;
  tensor1.repeats.emplace_back(ge::Symbol(100) + ge::Symbol("A") * ge::Symbol("B"));
  tensor1.repeats.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(5));
  tensor1.strides.emplace_back(ge::Symbol(1));
  tensor1.vectorized_strides.emplace_back(ge::Symbol(5));
  tensor1.vectorized_strides.emplace_back(ge::Symbol(1));
  ge::AscTensorAttr tensor2;
  tensor2.repeats.emplace_back(ge::Symbol(10) + ge::Symbol(10) / ge::Symbol("D") +
                               ge::Symbol(10) / ge::Symbol("E"));
  tensor2.repeats.emplace_back(ge::Symbol(5));
  tensor2.strides.emplace_back(ge::Symbol(5));
  tensor2.strides.emplace_back(ge::Symbol(1));
  tensor2.vectorized_strides.emplace_back(ge::Symbol(8));
  tensor2.vectorized_strides.emplace_back(ge::Symbol(1));
  EXPECT_EQ(recognizer.IsTensorViewEquivalent(tensor1, tensor2), false);
}