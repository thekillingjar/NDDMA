/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "elewise/compare_v2_api_call.h"

#include "runtime_stub.h"
#include "platform_context.h"

using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace testing;

class NeCompareV2ApiCallTest : public ::testing::Test {
protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

// Test Ne with INT8 type
TEST(NeCompareV2ApiCallTest, Ne_INT8) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  ge::ascir_op::Ne ne_op("ne");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(ne_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1*s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1*s2, s2, One};

  ne_op.x1 = load_op1.y;
  ne_op.x2 = load_op2.y;
  *ne_op.y.axis = {z0.id, z1.id, z2.id};
  *ne_op.y.repeats = {s0, s1, s2};
  *ne_op.y.strides = {s1*s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1->attr.api.type = ge::ApiType::kAPITypeCompute;
  load1->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = ge::DT_INT8;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2->attr.api.type = ge::ApiType::kAPITypeCompute;
  load2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2+s2, One};
  load2->outputs[0].attr.dtype = ge::DT_INT8;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto ne = graph.FindNode("ne");
  ne->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  ne->attr.api.type = ge::ApiType::kAPITypeCompute;
  ne->attr.api.unit = ge::ComputeUnit::kUnitVector;
  ne->attr.sched.loop_axis = z0.id;
  ne->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  ne->outputs[0].attr.vectorized_strides = {s2, One};
  ne->outputs[0].attr.dtype = ge::DT_UINT8;
  ne->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  ne->outputs[0].attr.mem.tensor_id = 3;
  ne->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  ne->outputs[0].attr.que.id = 2;
  ne->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(ne->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareV2ApiCall call("ne");
  EXPECT_EQ(call.Init(ne), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{
    "CompareExtend<int8_t, 2, CMPMODE::ne>(local_3[0], local_0[0], local_1[0], {static_cast<uint16_t>(t->s1), "
    "static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, "
    "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"
  });
}

// Test Ne with INT16 type
TEST(NeCompareV2ApiCallTest, Ne_INT16) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  ge::ascir_op::Ne ne_op("ne");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(ne_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1*s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1*s2, s2, One};

  ne_op.x1 = load_op1.y;
  ne_op.x2 = load_op2.y;
  *ne_op.y.axis = {z0.id, z1.id, z2.id};
  *ne_op.y.repeats = {s0, s1, s2};
  *ne_op.y.strides = {s1*s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1->attr.api.type = ge::ApiType::kAPITypeCompute;
  load1->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = ge::DT_INT16;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2->attr.api.type = ge::ApiType::kAPITypeCompute;
  load2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2+s2, One};
  load2->outputs[0].attr.dtype = ge::DT_INT16;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto ne = graph.FindNode("ne");
  ne->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  ne->attr.api.type = ge::ApiType::kAPITypeCompute;
  ne->attr.api.unit = ge::ComputeUnit::kUnitVector;
  ne->attr.sched.loop_axis = z0.id;
  ne->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  ne->outputs[0].attr.vectorized_strides = {s2, One};
  ne->outputs[0].attr.dtype = ge::DT_UINT8;
  ne->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  ne->outputs[0].attr.mem.tensor_id = 3;
  ne->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  ne->outputs[0].attr.que.id = 2;
  ne->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(ne->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareV2ApiCall call("ne");
  EXPECT_EQ(call.Init(ne), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{
    "CompareExtend<int16_t, 2, CMPMODE::ne>(local_3[0], local_0[0], local_1[0], {static_cast<uint16_t>(t->s1), "
    "static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, "
    "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"
  });
}

// Test Ne with BF16 type
TEST(NeCompareV2ApiCallTest, Ne_BF16) {
  ge::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  ge::ascir_op::Ne ne_op("ne");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(ne_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1*s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1*s2, s2, One};

  ne_op.x1 = load_op1.y;
  ne_op.x2 = load_op2.y;
  *ne_op.y.axis = {z0.id, z1.id, z2.id};
  *ne_op.y.repeats = {s0, s1, s2};
  *ne_op.y.strides = {s1*s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1->attr.api.type = ge::ApiType::kAPITypeCompute;
  load1->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = ge::DT_BF16;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2->attr.api.type = ge::ApiType::kAPITypeCompute;
  load2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2+s2, One};
  load2->outputs[0].attr.dtype = ge::DT_BF16;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto ne = graph.FindNode("ne");
  ne->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  ne->attr.api.type = ge::ApiType::kAPITypeCompute;
  ne->attr.api.unit = ge::ComputeUnit::kUnitVector;
  ne->attr.sched.loop_axis = z0.id;
  ne->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  ne->outputs[0].attr.vectorized_strides = {s2, One};
  ne->outputs[0].attr.dtype = ge::DT_UINT8;
  ne->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  ne->outputs[0].attr.mem.tensor_id = 3;
  ne->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  ne->outputs[0].attr.que.id = 2;
  ne->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(ne->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareV2ApiCall call("ne");
  EXPECT_EQ(call.Init(ne), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{
    "CompareExtend<bfloat16_t, 2, CMPMODE::ne>(local_3[0], local_0[0], local_1[0], {static_cast<uint16_t>(t->s1), " \
    "static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, " \
    "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"
  });
}
