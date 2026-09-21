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
#include <exception>
#include <filesystem>
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "backend_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "ascgen_log.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class TestBackendMatmulE2e : public testing::Test, public codegen::TilingLib {
 public:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
 protected:
  TestBackendMatmulE2e() : codegen::TilingLib("test", "test") {}
};

void CreateMatmulGraph(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  ge::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {ge::ops::One, ge::ops::One};
  *data1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *load1.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  ge::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

TEST_F(TestBackendMatmulE2e, MatmulE2eCodegen) {
  bool gen_success= true;
  ge::AscGraph graph("matmul_elemwise_pro");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, ge::ops::One};
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.attr.sched.axis = {z0.id, z1.id};
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = ge::DT_FLOAT;
  *scalar0.y.axis = {z0.id, z1.id};
  *scalar0.y.repeats = {ge::ops::One, ge::ops::One};
  *scalar0.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = scalar0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {ge::ops::One, s1};
  *broadcast0.y.strides = {ge::ops::Zero, ge::ops::One};

  ge::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {ge::ops::One, ge::ops::One};
  *data1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *load1.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = load1.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  broadcast2.y.dtype = ge::DT_FLOAT;
  *broadcast2.y.repeats = {ge::ops::One, s1};
  *broadcast2.y.strides = {ge::ops::Zero, ge::ops::One};

  ge::ascir_op::Broadcast broadcast3("broadcast3");
  broadcast3.x = broadcast2.y;
  broadcast3.attr.sched.axis = {z0.id, z1.id};
  *broadcast3.y.axis = {z0.id, z1.id};
  broadcast3.y.dtype = ge::DT_FLOAT;
  *broadcast3.y.repeats = {s0, s1};
  *broadcast3.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = add_op.y;
  mul.x2 = broadcast3.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;

  ge::AscGraph mm_graph("mutmul");
  CreateMatmulGraph(mm_graph);
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

    ascir::ScheduleGroup schedule_group2;
    schedule_group2.impl_graphs.push_back(mm_graph);
    fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
    fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;
    const std::map<std::string, std::string> shape_info;
    auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

    std::fstream tiling_func("Mutmul_fuse_tiling_func.cpp", std::ios::out);
    tiling_func << res["tiling_def_and_tiling_const"];

    std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
    std::cout << res["tiling_def_and_tiling_const"] << std::endl;
    std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
    std::string tiling_data_src_file_name = parts[0];  // autofuse_tiling_data.h
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);
    tiling_data_file << "";

    auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t GenCVFusionTilingKey(char* config_file, int aiv_num, int ub_size)");
    ASSERT_NE(pos, std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
