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
#include "micro_api_call.h"
#include "micro_scalar_broadcast_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, MicroScalarBroadCastStore) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  Scalar constant_op("constant", graph);
  constant_op.ir_attr.SetValue("1.0");
  Broadcast broadcast("broadcast");
  broadcast.x = constant_op.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};

  Store store("store");
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  auto broadcast_node = graph.FindNode("broadcast");
  broadcast_node->attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
  broadcast_node->attr.api.type = ge::ApiType::kAPITypeCompute;
  broadcast_node->attr.api.unit = ge::ComputeUnit::kUnitVector;
  broadcast_node->attr.sched.loop_axis = z0.id;
  broadcast_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  broadcast_node->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  broadcast_node->outputs[0].attr.mem.tensor_id = 1;
  broadcast_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  broadcast_node->outputs[0].attr.que.id = 1;
  broadcast_node->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store_node = graph.FindNode("store");
  store_node->attr.api.compute_type = ge::ComputeType::kComputeStore;
  store_node->attr.api.type = ge::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = ge::ComputeUnit::kUnitMTE3;
  store_node->attr.sched.loop_axis = z0.id;
  store_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store_node->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store_node->outputs[0].attr.que.id = 1;
  store_node->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 0;
  constant_node->outputs[0].attr.mem.position = ge::Position::kPositionInvalid;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(broadcast_node->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::ApiTensor y1;
  y1.id = store_node->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto scalar_val = constant_node->GetName() + "_" + constant_node->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(constant_node->outputs[0], scalar_val);
  auto tensor_store = store_node->GetName() + "_" + store_node->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store_node->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);

  codegen::MicroScalarBroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast_node), 0);
  call.AddInput(x1.id, codegen::TensorType::UB_TENSOR);
  call.AddOutput(y1.id);
  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Duplicate(vreg_2, scalar_0, p_reg);\n"});
}