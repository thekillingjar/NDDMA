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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "reg_nddma_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, NddmaApiCall_ThreeDimTensor) {
 ge::AscGraph graph("test_graph");

 ge::Expression Two = ge::Symbol(2);
 ge::Expression Three = ge::Symbol(3);
 ge::Expression Four = ge::Symbol(4);

 auto s0 = ge::Symbol(16);
 auto s1 = ge::Symbol(8);
 auto s2 = ge::Symbol(4);
 auto s3 = ge::Symbol(2);
 auto z0 = graph.CreateAxis("z0", s0);
 auto z1 = graph.CreateAxis("z1", s1);
 auto z2 = graph.CreateAxis("z2", s2);
 auto z3 = graph.CreateAxis("z3", s3);

 Data data0("data0", graph);
 Nddma nddma_op("nddma");
 ge::ascir_op::Store store_op("store");
 graph.AddNode(nddma_op);
 graph.AddNode(store_op);

 nddma_op.x = data0.y;
 nddma_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
 *nddma_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
 *nddma_op.y.repeats = {s0, s1, s2, s3}; // {16, 8, 4, 2}
 *nddma_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One}; // {256, 24, 4, 1}
 store_op.x = nddma_op.y;
 store_op.ir_attr.SetOffset(ge::Symbol(0));
 *store_op.y.axis = {z0.id, z1.id, z2.id};
 *store_op.y.repeats = {s0, s1, s2, s3};
 *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

 auto nddma = graph.FindNode("nddma");
 nddma->attr.api.compute_type = ge::ComputeType::kComputeLoad;
 nddma->attr.api.type = ge::ApiType::kAPITypeCompute;
 nddma->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
 nddma->attr.sched.loop_axis = z0.id;
 nddma->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
 nddma->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
 nddma->outputs[0].attr.dtype = ge::DT_FLOAT;
 nddma->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
 nddma->outputs[0].attr.mem.tensor_id = 0;
 nddma->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
 nddma->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
 nddma->outputs[0].attr.que.id = 1;
 nddma->outputs[0].attr.opt.merge_scope = ge::kIdNone;

 codegen::Tiler tiler;
 codegen::TPipe tpipe("tpipe", tiler);
 tpipe.AddTensor(nddma->outputs[0]);

 tiler.AddAxis(z0);
 tiler.AddAxis(z1);
 tiler.AddAxis(z2);
 tiler.AddAxis(z3);
 tiler.AddSizeVar(ge::SizeVar(s0));
 tiler.AddSizeVar(ge::SizeVar(s1));
 tiler.AddSizeVar(ge::SizeVar(s2));
 tiler.AddSizeVar(ge::SizeVar(s3));

 codegen::ApiTensor x;
 x.id = nddma->outputs[0].attr.mem.tensor_id;

 codegen::NddmaApiCall call_0("DataCopyNddma");
 EXPECT_EQ(call_0.Init(nddma), 0);
 call_0.inputs.push_back(&x);

 std::string result;
 call_0.Generate(tpipe, vector<ge::AxisId>{}, result);
 EXPECT_EQ(result,
           std::string{"const int64_t output_dims_0[5] = {1, 1, 8, 4, 2};\nconst int64_t output_stride_0[5] = "
                       "{1, 1, 8, 2, 1};\nconst int64_t input_stride_0[5] = {1, 1, 24, 4, 1};\n"
                       "DataCopyNddma(local_0, local_0[0 + 0], output_dims_0, output_stride_0, input_stride_0);\n"});
}

TEST(CodegenKernel, NddmaApiCall_SevenDimTensor) {
  ge::AscGraph graph("test_graph");

  ge::Expression Two = ge::Symbol(2);
  ge::Expression Three = ge::Symbol(3);
  ge::Expression Four = ge::Symbol(4);
  ge::Expression Five = ge::Symbol(5);
  ge::Expression Six = ge::Symbol(6);
  ge::Expression Seven = ge::Symbol(7);
  ge::Expression Eight = ge::Symbol(8);

  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(4);
  auto s3 = ge::Symbol(2);
  auto s4 = ge::Symbol(2);
  auto s5 = ge::Symbol(2);
  auto s6 = ge::Symbol(2);
  auto s7 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);
  auto z7 = graph.CreateAxis("z7", s7);

  Data data0("data0", graph);
  Nddma nddma_op("nddma");
  ge::ascir_op::Store store_op("store");
  graph.AddNode(nddma_op);
  graph.AddNode(store_op);

  nddma_op.x = data0.y;
  nddma_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id};
  *nddma_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id};
  *nddma_op.y.repeats = {s0, s1, s2, s3, s4, s5, s6, s7}; // {16, 8, 4, 2, 2, 2, 2, 2}
  *nddma_op.y.strides = {
      s1 * s2 * s3 * s4 * s5 * s6 * s7 * Eight, s2 * s3 * s4 * s5 * s6 * s7 * Seven, s3 * s4 * s5 * s6 * s7 * Six,
      s4 * s5 * s6 * s7 * Five, s5 * s6 * s7 * Four, s6 * s7 * Three, s7 * Two, One}; // {8192, 896, 192, 80, 32, 12, 4, 1}
  store_op.x = nddma_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id};
  *store_op.y.repeats = {s0, s1, s2, s3, s4, s5, s6, s7};
  *store_op.y.strides = {
      s1 * s2 * s3 * s4 * s5 * s6 * s7 * Eight, s2 * s3 * s4 * s5 * s6 * s7 * Seven, s3 * s4 * s5 * s6 * s7 * Six,
      s4 * s5 * s6 * s7 * Five, s5 * s6 * s7 * Four, s6 * s7 * Three, s7 * Two, One};

  auto nddma = graph.FindNode("nddma");
  nddma->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  nddma->attr.api.type = ge::ApiType::kAPITypeCompute;
  nddma->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  nddma->attr.sched.loop_axis = z0.id;
  nddma->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id};
  nddma->outputs[0].attr.vectorized_strides = {ge::Symbol(8192), ge::Symbol(256), ge::Symbol(32), ge::Symbol(8),
                                               ge::Symbol(4), ge::Symbol(2), One};
  nddma->outputs[0].attr.dtype = ge::DT_FLOAT;
  nddma->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  nddma->outputs[0].attr.mem.tensor_id = 0;
  nddma->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  nddma->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  nddma->outputs[0].attr.que.id = 1;
  nddma->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(nddma->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);
  tiler.AddAxis(z5);
  tiler.AddAxis(z6);
  tiler.AddAxis(z7);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddSizeVar(ge::SizeVar(s3));
  tiler.AddSizeVar(ge::SizeVar(s4));
  tiler.AddSizeVar(ge::SizeVar(s5));
  tiler.AddSizeVar(ge::SizeVar(s6));
  tiler.AddSizeVar(ge::SizeVar(s7));
  codegen::ApiTensor x;
  x.id = nddma->outputs[0].attr.mem.tensor_id;

  codegen::NddmaApiCall call_0("DataCopyNddma");
  EXPECT_EQ(call_0.Init(nddma), 0);
  call_0.inputs.push_back(&x);

  std::string result;
  call_0.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "const int64_t output_dims_0[5] = {2, 2, 2, 2, 2};\nconst int64_t output_stride_0[5] = "
          "{32, 8, 4, 2, 1};\nconst int64_t input_stride_0[5] = {80, 32, 12, 4, 1};\n"
          "for(int outer_for_0 = 0; outer_for_0 < 8; outer_for_0++) {\nfor(int outer_for_1 = 0; outer_for_1 < 4;"
          " outer_for_1++) {\nDataCopyNddma"
          "(local_0[outer_for_0 * 8192 + outer_for_1 * 256], local_0[0 + 0 + outer_for_0 * 896 + outer_for_1 * 192], "
          "output_dims_0, output_stride_0, input_stride_0);\n\n}\n}\n"});
}