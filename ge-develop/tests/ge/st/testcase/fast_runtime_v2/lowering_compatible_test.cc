/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "faker/fake_value.h"
#include "faker/global_data_faker.h"
#include "runtime/dev.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "faker/ge_model_builder.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "lowering/model_converter.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "mmpa/mmpa_api.h"
#include "securec.h"
#include "graph/operator_reg.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "register/op_tiling/op_compile_info_manager.h"
#include "register/op_tiling_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "faker/aicpu_taskdef_faker.h"
#include "check/executor_statistician.h"
#include "graph_builder/bg_infer_shape_range.h"
#include "common/executor_tracer_on.h"

using namespace ge;

namespace gert {
class LowerCompatibleUnitTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
};
namespace {
const std::string AddStubName = "AddStubBin";
// stub infershape for add. current not support element wise
ge::graphStatus StubAddInferShapeV1(ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_input_0 = op_desc->MutableInputDesc(0);
  auto tensordesc_input_1 = op_desc->MutableInputDesc(1);
  auto tensordesc_output = op_desc->MutableOutputDesc(0);

  auto input_shape_0 = tensordesc_input_0->GetShape();
  auto input_shape_1 = tensordesc_input_1->GetShape();
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    return ge::GRAPH_FAILED;
  }
  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape.emplace_back(input_shape_0.GetDim(i) + input_shape_1.GetDim(i));
  }
  // set output
  tensordesc_output->SetShape(ge::GeShape(output_shape));
  ge::DataType input_dtype = tensordesc_input_0->GetDataType();
  tensordesc_output->SetDataType(input_dtype);
  return ge::GRAPH_SUCCESS;
}

// stub infershape for expanddims. current not support element wise
ge::graphStatus StubExpandDimsInferShapeV1(ge::Operator &op) {
  std::vector<string> dep_inputs = {"axis"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc("x");
  auto axis_desc = op_desc->MutableInputDesc("axis");
  auto y_desc = op_desc->MutableOutputDesc("y");

  op_desc->SetOpInferDepends(dep_inputs);
  auto axis_type = axis_desc->GetDataType();
  auto x_type = x_desc->GetDataType();

  if (axis_type != DT_INT32 && axis_type != DT_INT64) {
    return GRAPH_PARAM_INVALID;
  }

  bool is_x_unknonwn_rank = x_desc->MutableShape().GetDims() == UNKNOWN_RANK ? true : false;
  if (is_x_unknonwn_rank) {
    y_desc->SetUnknownDimNumShape();
    y_desc->SetDataType(x_type);
    y_desc->SetOriginDataType(x_type);
    return GRAPH_SUCCESS;
  }

  int64_t axis_nums = axis_desc->MutableShape().GetShapeSize();

  if (axis_nums != 1) {
    // Shape::GetDims().size() == 0, means it's a scalar, its shape is [].
    if (!(axis_nums == 0 && axis_desc->MutableShape().GetDims().size() == 0)) {
      string reason = "axis input must be a tensor with a single value, actually rank = " + std::to_string(axis_nums);
      REPORT_INNER_ERR_MSG("E19999", "[Node:%s] Check input axis shape failed, as %s", op.GetName().c_str(),
                         reason.c_str());
      return GRAPH_PARAM_INVALID;
    }
  }

  auto axis_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName("axis"));
  auto tensor_axis = OpDescUtils::GetInputConstData(op, axis_idx);
  if (tensor_axis == nullptr) {
    auto x_shape_size = x_desc->MutableShape().GetDims().size();
    std::vector<int64_t> out_dims(x_shape_size + 1, UNKNOWN_DIM);
    y_desc->SetShape(GeShape(out_dims));
    y_desc->SetOriginShape(GeShape(out_dims));
    y_desc->SetDataType(x_type);
    y_desc->SetOriginDataType(x_type);
    // infer shape range
    std::vector<std::pair<int64_t, int64_t>> x_range;
    (void)x_desc->GetShapeRange(x_range);
    if (x_range.empty()) {
      return GRAPH_SUCCESS;
    }
    if (x_range.size() != x_shape_size) {
      string reason = "input shape range rank should be same with input shape rank, actually shape_rank=" +
                      std::to_string(x_shape_size) + ", shape_range_rank=" + std::to_string(x_range.size());
      REPORT_INNER_ERR_MSG("E19999", "[Node:%s] Check input x shape range failed, as %s", op.GetName().c_str(),
                         reason.c_str());
      return GRAPH_FAILED;
    }
    int64_t max_range_value = 1;
    for (const auto &ele : x_range) {
      if (ele.second > max_range_value) {
        max_range_value = ele.second;
      }
    }
    std::vector<std::pair<int64_t, int64_t>> y_range(x_shape_size + 1,
                                                     std::pair<int64_t, int64_t>({0, max_range_value}));
    y_desc->SetShapeRange(y_range);
    return GRAPH_SUCCESS;
  }

  auto pbuff = tensor_axis->GetData().GetData();
  if (pbuff == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "[Node:%s] Get attr value from axis input failed, as data buff is null",
                       op.GetName().c_str());
    return GRAPH_FAILED;
  }
  int64_t axis = 0;
  if (axis_type == DT_INT32) {
    axis = *const_cast<int32_t*>(reinterpret_cast<const int32_t*>(pbuff));
  } else if (axis_type == DT_INT64) {
    axis = *const_cast<int64_t*>(reinterpret_cast<const int64_t*>(pbuff));
  }

  std::vector<int64_t> vec_dim;
  int32_t dim_num = x_desc->MutableShape().GetDimNum();
  if (axis < -1 - dim_num || axis > dim_num) {
    string reason = "axis[" + std::to_string(axis) + "] is not in [" + std::to_string(-1 - dim_num) + " , " +
                    std::to_string(dim_num) + "]";
    REPORT_INNER_ERR_MSG("E19999", "[Node:%s] Check input axis failed, as %s", op.GetName().c_str(), reason.c_str());
    return GRAPH_PARAM_INVALID;
  }

  if (axis < 0) {
    axis += dim_num + 1;
  }
  for (int i = 0; i < dim_num; i++) {
    vec_dim.push_back(x_desc->MutableShape().GetDim(i));
  }
  vec_dim.emplace(vec_dim.begin() + axis, 1);
  y_desc->SetShape(GeShape(vec_dim));
  y_desc->SetOriginShape(GeShape(vec_dim));
  y_desc->SetDataType(x_type);
  y_desc->SetOriginDataType(x_type);
  // infer shape range
  auto x_shape_size = x_desc->MutableShape().GetDims().size();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  (void)x_desc->GetShapeRange(x_range);
  if (x_range.empty()) {
    return GRAPH_SUCCESS;
  }
  if (x_range.size() != x_shape_size) {
    REPORT_INNER_ERR_MSG("E19999", "[Node:%s] Check input x shape range failed, as input shape range size num should be "
                       "same with input shape size, actually shape_rank=%zu, shape_range_rank=%zu",
                       op.GetName().c_str(), x_shape_size, x_range.size());
    return GRAPH_FAILED;
  }
  x_range.emplace(x_range.begin() + axis, std::pair<int64_t, int64_t>{1, 1});
  y_desc->SetShapeRange(x_range);
  return GRAPH_SUCCESS;
}

class StubCompileInfoJson : public optiling::CompileInfoBase {
 public:
  StubCompileInfoJson(const std::string &json) : json_str_(json) {}
  ~StubCompileInfoJson() {}
  std::string GetJsonStr() {
    return json_str_;
  };

 private:
  std::string json_str_;
};

optiling::CompileInfoPtr StubOpParseFuncV4(const ge::Operator &op, const ge::AscendString &compileinfo) {
  optiling::CompileInfoPtr info = std::make_shared<StubCompileInfoJson>("testStubOpParseFuncV4");
  return info;
}

bool StubOpTilingFuncV4OnRun(const ge::Operator &op, const optiling::CompileInfoPtr compile_info, optiling::OpRunInfoV2 &op_run_info) {
  op_run_info.SetTilingKey(11);
  op_run_info.SetBlockDim(2);
  op_run_info.SetClearAtomic(true);
  op_run_info.SetWorkspaces({1,2});
  op_run_info.SetTilingCond(1);
  int64_t data1 = 123;
  int64_t data2 = 456;
  op_run_info.AddTilingData(data1);
  op_run_info.AddTilingData(data2);
  // check compile info
  std::string compile_json = dynamic_cast<StubCompileInfoJson*>(compile_info.get())->GetJsonStr();
  if (compile_json != "testStubOpParseFuncV4") {
    return false;
  }
  return true;
}

void *StubOpParseFuncV3(const ge::Operator &op, const ge::AscendString &compileinfo) {
  static void *a = new int(3);
  return a;
}

bool StubOpTilingFuncV3(const ge::Operator &op, const void *compile_info, optiling::OpRunInfoV2 &op_run_info) {
  op_run_info.SetTilingKey(12);
  return true;
}
} // namespace

TEST_F(LowerCompatibleUnitTest, Lowering_Execute_Model_On_Compatible_InferShape_On_Input_Dependency) {
  // mock infer_func
  auto expanddims_infer_func = [](ge::Operator &v) { return StubExpandDimsInferShapeV1(v);}; // simulate operator_reg
  ge::InferShapeFuncRegister(EXPANDDIMS, expanddims_infer_func);

  auto graph = ShareGraph::BuildDataDependencyNodeGraph();
  graph->TopologicalSorting();

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  
  // start lowering
  bg::ValueHolder::PopGraphFrame(); // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2E_Expanddims_Graph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(24*4);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({1, 2, 3, 4}, 1);
  int32_t data_i1[1] = {2}; // axis value

  auto i0 = FakeValue<Tensor>(Tensor{{{1, 2, 3, 4}, {1, 2, 3, 4}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_INT32, mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{{{}, {}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnHost, ge::DT_INT32, &data_i1});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get()}).data(), 2,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()), ge::GRAPH_SUCCESS);
                                    
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get()}).data(), 2,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()), ge::GRAPH_SUCCESS);
                                    
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  // input x shape [1, 2, 3, 4]  input_axis value [2]
  // expect output shape[1,2,1,3,4]
  ASSERT_EQ(outputs.at(0).GetShape().GetStorageShape(), gert::Shape({1,2,1,3,4})); // check output shape
  rtStreamDestroy(stream);
  mem_block->Free();
}


TEST_F(LowerCompatibleUnitTest, Lowering_Execute_Model_On_Compatible_infer_tiling_v4) {
  // mock infer_func & tiling _func for Add
  auto add_infer_func = [](ge::Operator &v) { return StubAddInferShapeV1(v);}; // simulate operator_reg
  auto node_type = "AddTestCompatibleV4";
  ge::InferShapeFuncRegister(node_type, add_infer_func);
  // mock tiling func
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV4OnRun, StubOpParseFuncV4);

  auto graph = ShareGraph::BuildSingleNodeGraph(node_type);
  graph->TopologicalSorting();
  auto add_node = graph->FindFirstNodeMatchType(node_type);
  ge::AttrUtils::SetStr(add_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}"); // mock for tiling parse

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef(node_type, AiCoreTaskDefFaker(AddStubName).WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame(); // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EReshapeGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048*4);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({4096}, 1);

  auto i0 = FakeValue<Tensor>(Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get()}).data(), 2,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()), ge::GRAPH_SUCCESS);
                                    
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get()}).data(), 2,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()), ge::GRAPH_SUCCESS);
                                    
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  // input_0 shape [2048]  input_1 shape [2048]
  // expect output shape[4096]
  ASSERT_EQ(outputs.at(0).GetShape().GetStorageShape().GetDim(0), 2048 + 2048); // check output shape
  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(LowerCompatibleUnitTest, Lowering_Execute_Model_On_Compatible_infer_tiling_v3) {
  // mock infer_func & tiling _func for Add
  auto add_infer_func = [](ge::Operator &v) { return StubAddInferShapeV1(v);}; // simulate operator_reg
  auto node_type = "AddTestCompatibleV3";
  ge::InferShapeFuncRegister(node_type, add_infer_func);
  // mock tiling func v3
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV3, StubOpParseFuncV3);

  auto graph = ShareGraph::BuildSingleNodeGraph(node_type);
  graph->TopologicalSorting();
  auto add_node = graph->FindFirstNodeMatchType(node_type);
  ge::AttrUtils::SetStr(add_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}"); // mock for tiling parse

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef(node_type, AiCoreTaskDefFaker(AddStubName).WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame(); // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(1024UL);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({4096}, 1);

  auto i0 = FakeValue<Tensor>(Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get()}).data(), 2,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get()}).data(), 2,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  // input_0 shape [2048]  input_1 shape [2048]
  // expect output shape[4096]
  ASSERT_EQ(outputs.at(0).GetShape().GetStorageShape().GetDim(0), 2048 + 2048); // check output shape

  rtStreamDestroy(stream);
  mem_block->Free();
}

//TEST_F(LowerCompatibleUnitTest, ExecuteModel_BinaryKernel_Optiling) {
  // todo: 当前这个ST的构图是错误的，会把一个TensorData中的数据指向一个BlockDim传递给NetOutput，这会导致在将NetOutput的数据考出时失败。
  //   因此这个用例需要重新构造，这里先删掉了
//}

TEST_F(LowerCompatibleUnitTest, InferShapeRange_ExecuteSuccess) {
  auto graph = ShareGraph::BuildCompatibleInferShapeRangeGraph();
  graph->FindNode("test_no_infer")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model =
      builder.AddTaskDef("TestNoInferShapeRange", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  ge::DumpGraph(exe_graph.get(), "E2ETestNoInferShapeRangeGraph");
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ExecutorTracerOn executor_tracer_on;
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  void *device_block = (void *)0x01;
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT16)
      .Format(ge::FORMAT_ND)
      .Shape({2048, 2, 3, 4})
      .Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  auto i0 = FakeValue<Tensor>(Tensor{{{64, 2, 3, 4}, {64, 2, 3, 4}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     device_block});
  auto i1 = FakeValue<Tensor>(Tensor{{{64, 2, 3, 4}, {64, 2, 3, 4}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     device_block});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("TestNoInferShapeRange", "CompatibleInferShapeRange"), 1);
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("TestNoInferShapeRange", "CreateTensorRangesAndShapeRanges"), 1);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  bg::ShapeRangeInferenceResult::ErrorResult();
  rtStreamDestroy(stream);
}
} // namespace gert