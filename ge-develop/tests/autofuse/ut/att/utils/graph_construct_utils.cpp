/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph_construct_utils.h"
#include <string>
#include <vector>
#include "gen_model_info.h"

namespace ge {
namespace ascir {
namespace cg {
Status ConstructSimpleLoadStoreOp(ge::AscGraph &graph) {
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto abs1 = Abs("Abs", load1).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store", abs1);
      GE_ASSERT_SUCCESS(
          att::GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt}, {load1, abs1, store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  auto load_asc_node = graph.FindNode("load");
  GE_ASSERT_NOTNULL(load_asc_node);
  load_asc_node->inputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  auto store_asc_node = graph.FindNode("store");
  GE_ASSERT_NOTNULL(store_asc_node);
  store_asc_node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  return ge::SUCCESS;
}

// 公共构图函数：Reduce分核惩罚场景的Concat图
// 用于V1和V35测试，避免符号冲突
Status BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto S1 = ge::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto [s0T, s0t] = graph.TileSplit(s0.id);
  auto [s1T, s1t] = graph.TileSplit(s1.id);
  auto s1Ts0T = *graph.MergeAxis({s1T->id, s0T->id});
  auto [s1Ts0TB, s1Ts0Tb] = graph.BlockSplit(s1Ts0T.id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s0, s1});
  LOOP(*s1Ts0TB) {
    LOOP(*s1Ts0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto mean = Mean("mean1", load1).TQue(Position::kPositionVecOut, 1, 1);
      auto store1 = Store("store1", mean);
      GE_ASSERT_SUCCESS(
          att::GraphConstructUtils::UpdateOutputTensorAxes({*s1Ts0TB, *s1Ts0Tb, *s1t, *s0t}, {load1, store1}, 1));
      *load1.axis = {s1Ts0Tb->id, s1t->id, s0t->id};
      *load1.repeats = {s1Ts0Tb->size, s1t->size, s0t->size};
      *load1.strides = {s0t->size * s1t->size, s1t->size, att::CreateExpr(1)};
      *load1.vectorized_axis = {s1t->id, s0t->id};

      *mean.axis = {s1Ts0Tb->id, s1t->id, s0t->id};
      *mean.repeats = {s1Ts0Tb->size, s1t->size, att::CreateExpr(1)};
      *mean.strides = {s0t->size * s1t->size, s0t->size, att::CreateExpr(0)};
      *mean.vectorized_axis = {s1t->id, s0t->id};

      *store1.axis = {s1Ts0Tb->id, s1t->id, s0t->id};
      *store1.repeats = {s1Ts0Tb->size, s1t->size, att::CreateExpr(1)};
      *store1.strides = {s0t->size * s1t->size, s0t->size, att::CreateExpr(0)};
      *store1.vectorized_axis = {s1t->id, s0t->id};
      auto output1 = Output("output1", store1);
    }
  }
  for (auto node : graph.GetAllNodes()) {
    if (node->outputs().empty()) {
      continue;
    }
    auto last_dim_name = att::GetVecString(node->outputs()[0]->attr.repeats);
    GELOGD("Found Tile split axis %s in load/store node", last_dim_name.c_str());
  }
  return ge::SUCCESS;
}
}
}
}  // namespace ge
namespace att {
ge::AscNodePtr GraphConstructUtils::ConstructSingleOp(const std::string &op_type, int32_t in_cnt, int32_t out_cnt) {
  GraphBuilder graph_builder("test");
  graph_builder.AddNode("test_node", op_type, in_cnt, out_cnt);
  ge::AscGraph asc_graph("test");
  GE_ASSERT_SUCCESS(ge::AscGraphUtils::ConvertComputeGraphToAscGraph(graph_builder.GetGraph(), asc_graph));
  ge::AscNodePtr node_ptr = asc_graph.FindNode("test_node");
  return node_ptr;
}

ge::Status GraphConstructUtils::CreateSimpleLoadStoreOp(ge::AscGraph &graph) {
  return ge::ascir::cg::ConstructSimpleLoadStoreOp(graph);
}

ge::Status GraphConstructUtils::BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(ge::AscGraph &graph) {
  return ge::ascir::cg::BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(graph);
}

void GraphConstructUtils::UpdateVectorizedStride(const std::vector<int64_t> &axis,
                                                 const std::vector<ge::Expression> &strides,
                                                 const std::vector<int64_t> &vectorized_axis,
                                                 std::vector<ge::Expression> &vectorized_strides) {
  for (auto axis_id : vectorized_axis) {
    int idx = 0;
    for (auto a : axis) {
      if (a == axis_id) {
        vectorized_strides.emplace_back(strides[idx]);
        break;
      }
      idx += 1;
    }
  }
}

void GraphConstructUtils::UpdateGraphVectorizedStride(ge::AscGraph &graph) {
  for (auto x : graph.GetAllNodes()) {
    for (size_t i = 0; i < x->GetAllOutDataAnchorsSize(); i++) {
      UpdateVectorizedStride(x->outputs[i].attr.axis, x->outputs[i].attr.strides, x->outputs[i].attr.vectorized_axis,
                             x->outputs[i].attr.vectorized_strides);
    }
  }
}

void GraphConstructUtils::UpdateGraphsVectorizedStride(std::vector<ge::AscGraph> &impl_graphs) {
  for (auto &graph : impl_graphs) {
    for (auto x : graph.GetAllNodes()) {
      for (size_t i = 0; i < x->GetAllOutDataAnchorsSize(); i++) {
        UpdateVectorizedStride(x->outputs[i].attr.axis, x->outputs[i].attr.strides, x->outputs[i].attr.vectorized_axis,
                               x->outputs[i].attr.vectorized_strides);
      }
    }
  }
}

ge::Status GraphConstructUtils::UpdateTensorAxes(const std::vector<ge::Axis> &axes, ge::AscOpOutput &output,
                                                 const int32_t loop_id) {
  GE_ASSERT_TRUE(loop_id < static_cast<int32_t>(axes.size()));
  ge::Expression stride = att::CreateExpr(1);
  // {z0TB->id, z0Tb->id, z0T->id, z0t->id, z2.id, z3.id};
  // axes size = 6, loop axis id = 2, vectorized_axis size = 3
  const auto vectorized_axis_size = static_cast<int32_t>((loop_id >= 0) ? (axes.size() - loop_id - 1) : axes.size());
  GE_ASSERT_TRUE(vectorized_axis_size >= 0);
  output.axis->resize(axes.size());
  output.vectorized_axis->resize(vectorized_axis_size);
  output.repeats->resize(axes.size());
  output.strides->resize(axes.size());
  // id = 5, 4, 3, 2, 1, 0
  for (auto id = static_cast<int32_t>(axes.size() - 1); id >= 0; id--) {
    if (id - loop_id - 1 >= 0) {
      // vectorized_axis id = (5 - 2 - 1), (4 - 2 - 1), (3 - 2 - 1)
      (*output.vectorized_axis)[id - loop_id - 1] = (axes[id].id);
    }
    // axis id = 5, 4, 3, 2, 1, 0
    (*output.axis)[id] = (axes[id].id);
    (*output.repeats)[id] = (axes[id].size);
    (*output.strides)[id] = (stride);
    stride = stride * axes[id].size;
  }
  return ge::SUCCESS;
}

ge::Status GraphConstructUtils::UpdateOutputTensorAxes(const std::vector<ge::Axis> &axes,
                                                       std::vector<ge::AscOpOutput> &&outputs, const int32_t loop_id) {
  for (auto &output : outputs) {
    GE_ASSERT_SUCCESS(UpdateTensorAxes(axes, output, loop_id));
  }
  return ge::SUCCESS;
}
}  // namespace att