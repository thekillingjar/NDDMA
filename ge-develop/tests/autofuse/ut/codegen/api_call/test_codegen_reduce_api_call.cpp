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
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "reduce/reduce_api_call.h"

using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace testing;

class ReduceApicallTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ReduceApicallTest, ReduceApi_Test_001) {
  std::string api_name = "Max";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  ApiAttr api_attr;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size*z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<ge::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  vector<ge::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Max";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_002) {
  std::string api_name = "Max";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size*z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<ge::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  vector<ge::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Max";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_003) {
  std::string api_name = "Max";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));
  ge::SizeVar s3(ge::Symbol("s3"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s2.expr};
  ge::Axis z3{.id = 3, .name = "z3", .type = ge::Axis::Type::kAxisTypeOriginal, .size = s3.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddSizeVar(ge::SizeVar(s3));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id, z3.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id, z3.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size, z3.size};
  tensorx.attr.strides = {z1.size*z2.size*z3.size, z2.size*z3.size, z3.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<ge::Expression> vectorized_stride_x{One, One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id, z3.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id, z3.id};
  tensory.attr.repeats = {z0.size, One, One, One};
  tensory.attr.strides = {One, Zero, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  vector<ge::Expression> vectorized_stride_y{One, Zero, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Max";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_004) {
  std::string api_name = "Max";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeBlockInner, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size*z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<ge::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  vector<ge::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Max";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_All_Reduce) {
  std::string api_name = "Max";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  ApiAttr api_attr;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  nodex->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, ge::MemAttr(), 0}};
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size*z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<ge::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {One, z1.size, One};
  tensory.attr.strides = {Zero, One, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  vector<ge::Expression> vectorized_stride_y{Zero, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Max";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"{\nuint32_t last = 1 * t->s0 * KernelUtils::SizeAlign(t->s2, 32/sizeof(float));\nuint32_t last_actual = 1 * z0_actual_size * KernelUtils::SizeAlign(z2_actual_size, 32/sizeof(float));\nuint32_t first = 1;\nuint32_t first_actual = 1;\nReduceInit<float, 1, false>(local_1, first_actual, last, last_actual, z2_actual_size);\nAscendC::PipeBarrier<PIPE_V>();\nuint32_t tmp_reduce_shape[] = {first_actual, last};\nReduceMax<float, AscendC::Pattern::Reduce::AR, false>(local_3[0], local_1[0], tmp_buf_0, tmp_reduce_shape, true);\n}\n"});
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_Int32_Inner) {
  std::string api_name = "Sum";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeBlockInner, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size*z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_INT32;
  vector<ge::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_INT32;
  vector<ge::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Sum";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_Int32_Outer) {
  std::string api_name = "Sum";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size*z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_INT32;
  vector<ge::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_INT32;
  vector<ge::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Sum";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_ReduceMean_NoNeed_MultiReduce_Int32) {
  std::string api_name = "Mean";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z1.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_INT32;
  vector<ge::Expression> vectorized_stride_x{z2.size, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z1.id, z2.id};
  tensory.attr.repeats = {z0.size, z1.size, One};
  tensory.attr.strides = {z1.size, One, Zero};
  tensory.attr.mem.tensor_id = 2;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_INT32;
  vector<ge::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Mean";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_ReduceMean_NoNeed_MultiReduce_Float) {
  std::string api_name = "Mean";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  ge::SizeVar s0(ge::Symbol("s0"));
  ge::SizeVar s1(ge::Symbol("s1"));
  ge::SizeVar s2(ge::Symbol("s2"));

  ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  ge::AscGraph graph("test");
  ge::ascir_op::Data x("x", graph);
  ge::ascir_op::Data y("y", graph);

  auto nodex = graph.FindNode("x");
  ge::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  ge::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z1.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = ge::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = ge::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_FLOAT;
  vector<ge::Expression> vectorized_stride_x{z2.size, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z1.id, z2.id};
  tensory.attr.repeats = {z0.size, z1.size, One};
  tensory.attr.strides = {z1.size, One, Zero};
  tensory.attr.mem.tensor_id = 2;
  tensory.attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = ge::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = ge::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_FLOAT;
  vector<ge::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
// Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = ge::ComputeUnit::kUnitVector;
  call.type = "Mean";
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, ge::SUCCESS);
}