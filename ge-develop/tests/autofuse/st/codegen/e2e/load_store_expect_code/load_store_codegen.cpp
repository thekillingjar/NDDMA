/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <fstream>
#include <gtest/gtest.h>

#include "codegen.h"
#include "e2e_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"

using namespace ge::ascir_op;
using namespace ge::ops;

std::vector<std::string> splitString(const std::string& input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
      result.push_back(token);
  }

  return result;
}

void ShapeOneLoadBroadcastStore(std::vector<ge::AscGraph> &impl_graphs) {
  ge::AscGraph graph("load_store_general_0_nil_0_nil");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x");
  graph.AddNode(x_op);
  x_op.attr.sched.axis = {z0.id, z1.id};
  x_op.y.dtype = ge::DT_FLOAT16;
  *x_op.y.axis = {z0.id, z1.id};

  Load load_op("load");
  graph.AddNode(load_op);
  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  load_op.y.dtype = ge::DT_FLOAT16;
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, One};
  *load_op.y.strides = {One, Zero};

  Broadcast broadcast_op("broadcast");
  graph.AddNode(broadcast_op);
  broadcast_op.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, ge::MemAttr(), 0}};
  broadcast_op.x = load_op.y;
  broadcast_op.attr.sched.axis = {z0.id, z1.id};
  broadcast_op.y.dtype = ge::DT_FLOAT16;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  Store store_op("store");
  graph.AddNode(store_op);
  store_op.x = load_op.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, One};
  *store_op.y.strides = {One, Zero};

  Store store2_op("store2");
  graph.AddNode(store2_op);
  store2_op.x = broadcast_op.y;
  store2_op.attr.sched.axis = {z0.id, z1.id};
  store2_op.y.dtype = ge::DT_FLOAT16;
  *store2_op.y.axis = {z0.id, z1.id};
  *store2_op.y.repeats = {s0, s1};
  *store2_op.y.strides = {s1, One};

  Output y_op("y");
  graph.AddNode(y_op);
  y_op.x = store_op.y;
  y_op.attr.sched.axis = {z0.id, z1.id};
  y_op.y.dtype = ge::DT_FLOAT16;

  Output y2_op("y2");
  graph.AddNode(y2_op);
  y2_op.x = store2_op.y;
  y2_op.attr.sched.axis = {z0.id, z1.id};
  y2_op.y.dtype = ge::DT_FLOAT16;

  int tensor_id = 0;
  auto x = graph.FindNode("x");
  x->attr.api.type = ge::ApiType::kAPITypeBuffer;
  x->attr.api.unit = ge::ComputeUnit::kUnitNone;
  x->attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  x->outputs[0].attr.mem.tensor_id = tensor_id++;
  x->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto load = graph.FindNode("load");
  // Scheduler
  load->attr.sched.loop_axis = z0.id;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  // QueBufAlloc
  load->outputs[0].attr.mem.tensor_id = tensor_id++;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = ge::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 1;
  load->outputs[0].attr.que.buf_num = 1;
  load->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  // Scheduler
  store->attr.sched.loop_axis = z0.id;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  store->attr.api.compute_type = ge::ComputeType::kComputeStore;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  // QueBufAlloc
  store->outputs[0].attr.mem.tensor_id = tensor_id++;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto broadcast = graph.FindNode("broadcast");
  // Scheduler
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->attr.api.type = ge::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = ge::ComputeUnit::kUnitVector;
  broadcast->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  broadcast->outputs[0].attr.vectorized_axis = {z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {One};
  // QueBufAlloc
  broadcast->outputs[0].attr.mem.tensor_id = tensor_id++;
  broadcast->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  broadcast->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  broadcast->outputs[0].attr.buf.id = ge::kIdNone;
  broadcast->outputs[0].attr.que.id = 1;
  broadcast->outputs[0].attr.mem.reuse_id = 1;
  broadcast->outputs[0].attr.que.depth = 1;
  broadcast->outputs[0].attr.que.buf_num = 1;
  broadcast->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  broadcast->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store2 = graph.FindNode("store2");
  // Scheduler
  store2->attr.sched.loop_axis = z0.id;
  store2->attr.api.type = ge::ApiType::kAPITypeCompute;
  store2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  store2->attr.api.compute_type = ge::ComputeType::kComputeStore;
  store2->outputs[0].attr.vectorized_axis = {z1.id};
  store2->outputs[0].attr.vectorized_strides = {One};
  // QueBufAlloc
  store2->outputs[0].attr.mem.tensor_id = tensor_id++;
  store2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto y = graph.FindNode("y");
  y->attr.api.type = ge::ApiType::kAPITypeBuffer;
  y->attr.api.unit = ge::ComputeUnit::kUnitNone;
  y->attr.api.compute_type = ge::ComputeType::kComputeInvalid;

  auto y2 = graph.FindNode("y2");
  y2->attr.api.type = ge::ApiType::kAPITypeBuffer;
  y2->attr.api.unit = ge::ComputeUnit::kUnitNone;
  y2->attr.api.compute_type = ge::ComputeType::kComputeInvalid;

  impl_graphs.push_back(graph);
}

class LoadBroadcastShapeOneStoreTest : public testing::Test {
};

TEST_F(LoadBroadcastShapeOneStoreTest, LoadBroadcastShapeOneStoreCodegen) {
  ge::AscGraph test_graph("load_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<ge::AscGraph> test_impl_graphs;
  ShapeOneLoadBroadcastStore(test_impl_graphs);

  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];      // load_sub_store_kernel.cpp
  std::string tiling_src_file_name = parts[1];      // load_sub_store_tiling.cpp
  std::string tiling_data_src_file_name = parts[2]; // autofuse_tiling_data.h

  bool gen_success = true;
  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    ascir::ScheduledResult schedule_result;
    std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = ge::AscendString("load_store");
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result), 0);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }
  
  EXPECT_EQ(gen_success, true);
}
