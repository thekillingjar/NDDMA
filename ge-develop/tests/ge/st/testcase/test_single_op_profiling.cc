/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "macro_utils/dt_public_scope.h"
#include "single_op/single_op.h"
#include "macro_utils/dt_public_unscope.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "framework/common/profiling_definitions.h"
#include "framework/common/ge_types.h"
#include "framework/executor/ge_executor.h"
#include "single_op/task/tbe_task_builder.h"
#include "common/profiling/profiling_manager.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "utils/taskdef_builder.h"
#include "ge/st/stubs/utils/bench_env.h"
#include "common/global_variables/diagnose_switch.h"
#include "ge/st/stubs/utils/model_data_builder.h"
#include "ge/st/stubs/utils/graph_factory.h"
#include "ge/st/stubs/utils/tensor_descs.h"
#include "ge/st/stubs/utils/data_buffers.h"
#include "ge/st/stubs/utils/profiling_test_data.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "utils/mock_runtime.h"
#include "framework/generator/ge_generator.h"
#include "utils/mock_ops_kernel_builder.h"
#include "framework/ge_runtime_stub/include/faker/fake_allocator.h"
#include <gtest/gtest.h>
#include "depends/profiler/src/profiling_auto_checker.h"

namespace ge {
using namespace profiling;
constexpr int64_t kMemtypeHostCompileIndependent = 2;
class SingleOpProfilingSt : public testing::Test {
 protected:
  void TearDown() override {
    profiling::ProfilingContext::GetInstance().Reset();
    profiling::ProfilingContext::GetInstance().SetDisable();
    ProfilingProperties::Instance().ClearProperties();
  }
};

void FillDataForHostMemInput(const GeTensorDescPtr &tensor_desc, DataBuffer &data_buffer) {
  int64_t mem_type = 0;
  AttrUtils::GetInt(tensor_desc, ge::ATTR_NAME_PLACEMENT, mem_type);
  if (mem_type == kMemtypeHostCompileIndependent) {
    data_buffer.placement = kHostMemType;
    uint64_t *data_ptr = PtrToPtr<void, uint64_t>(data_buffer.data);
    for (size_t i = 0; i < data_buffer.length / sizeof(uint64_t); i++) {
      data_ptr[i] = kHostMemInputValue;
    }
  }
}

Status GenerateTaskForAiCore(const Node &node,
                             RunContext &run_context,
                             std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != RELU) {
    return SUCCESS;
  }

  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask());
  return SUCCESS;
}

void CreateInputsAndOutputs(OpDescPtr &op_desc,
                            std::vector<GeTensorDesc> &input_desc,
                            std::vector<DataBuffer> &inputs,
                            std::vector<GeTensorDesc> &output_desc,
                            std::vector<DataBuffer> &outputs) {
  std::vector<std::unique_ptr<uint8_t[]>> input_buffers;
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers;
  for (const auto &tensor_desc : op_desc->GetAllInputsDescPtr()) {
    bool is_const = false;
    AttrUtils::GetBool(tensor_desc, CONST_ATTR_NAME_INPUT, is_const);
    if (is_const) {
      continue;
    }
    input_desc.emplace_back(*tensor_desc);
    int64_t tensor_size = -1;
    TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size);
    EXPECT_GE(tensor_size, 0);
    input_buffers.emplace_back(MakeUnique<uint8_t[]>(tensor_size));
    DataBuffer data_buffer;
    data_buffer.data = input_buffers.back().get();
    data_buffer.length = tensor_size;
    FillDataForHostMemInput(tensor_desc, data_buffer);
    inputs.emplace_back(data_buffer);
  }
  for (const auto &tensor_desc : op_desc->GetAllOutputsDescPtr()) {
    output_desc.emplace_back(*tensor_desc);
    int64_t tensor_size = -1;
    TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size);
    EXPECT_GE(tensor_size, 0);
    output_buffers.emplace_back(MakeUnique<uint8_t[]>(tensor_size));
    DataBuffer data_buffer;
    data_buffer.data = output_buffers.back().get();
    data_buffer.length = tensor_size;
    outputs.emplace_back(data_buffer);
  }
}

Status BuildSingleOp(OpDescPtr &op_desc, ModelBufferData &model_buffer) {
  vector<GeTensor> inputs;
  vector<GeTensor> outputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<GeTensorDesc> output_desc;
  for (const auto &tensor_desc : op_desc->GetAllInputsDescPtr()) {
    inputs.emplace_back(GeTensor(*tensor_desc));
    input_desc.emplace_back(*tensor_desc);
  }
  for (const auto &tensor_desc : op_desc->GetAllOutputsDescPtr()) {
    outputs.emplace_back(GeTensor(*tensor_desc));
    output_desc.emplace_back(*tensor_desc);
  }
  GeGenerator generator;
  generator.Initialize({});
  auto ret = generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_SYS, false, model_buffer);
  return ret;
}

Status RunStaticTestCast(OpDescPtr &op_desc) {
  ModelBufferData model_buffer;
  EXPECT_EQ(BuildSingleOp(op_desc, model_buffer), SUCCESS);
  ModelData model_data;
  model_data.model_data = model_buffer.data.get();
  model_data.model_len = model_buffer.length;
  SingleOp *single_op = nullptr;
  GeExecutor ge_executor;
  rtStream_t stream = nullptr;
  GE_MAKE_GUARD(stream_destroy, [&stream](){
    rtStreamDestroy(stream);
    stream = nullptr;
  });
  rtStreamCreate(&stream, 0);
  EXPECT_EQ(ge_executor.LoadSingleOpV2("aicore_op", model_data, stream, &single_op, 4), SUCCESS);

  std::vector<DataBuffer> inputs;
  std::vector<DataBuffer> outputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<GeTensorDesc> output_desc;
  CreateInputsAndOutputs(op_desc, input_desc, inputs, output_desc, outputs);
  single_op->impl_->model_param_->runtime_param.mem_size = 1000;
  FakeAllocator fake_allocator{};
  single_op->impl_->stream_resource_->SetAllocator(&fake_allocator);
  return GeExecutor::ExecuteAsync(single_op, inputs, outputs);
}

OpDescPtr CreateOp(const std::string &op_type) {
  GeShape shape({2, 8});
  GeTensorDesc tensor_desc(shape);
  auto op_desc = std::make_shared<OpDesc>(op_type, op_type);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  return op_desc;
}

UINT32 StubTilingPro(gert::TilingContext *) {
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParsePro(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreatorPro() {
  auto tmp =  ge::MakeUnique<char>();
  return tmp.get();
}

TEST_F(SingleOpProfilingSt, DynamicSingleOpProfiling) {
  profiling::ProfilingContext::GetInstance().SetEnable();

  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::SingeOpGraph()).AddTask(2, 2).Build();

  auto input_desc = TensorDescs(1).Value();
  auto input_buffers = DataBuffers(1).Value();
  auto output_desc = TensorDescs(1).Value();
  auto output_buffers = DataBuffers(1).Value();
  ge::DynamicSingleOp *single_op = nullptr;
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/transdata", std::move(kernelBin));
  auto holder = std::unique_ptr<KernelHolder>(new (std::nothrow) KernelHolder("0/_tvmbin", tbe_kernel));
  KernelBinRegistry::GetInstance().AddKernel("0/_tvmbin", std::move(holder));
  ASSERT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op2", modelData, nullptr, &single_op, 0), SUCCESS);
  ASSERT_EQ(single_op->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), SUCCESS);

  ProfilingData().HasRecord(kUpdateShape).HasRecord(kTiling);
}

TEST_F(SingleOpProfilingSt, HybridSingeOpGraphProfiling) {
  profiling::ProfilingContext::GetInstance().SetEnable();

  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::HybridSingeOpGraph()).AddTask(2, 2)
  .AddTask(2, 4)
  .AddTask(2, 4)
  .AddTask(2, 5)
  .AddTask(2, 5)
  .Build();

  auto input_desc = TensorDescs(2).Value();
  auto input_buffers = DataBuffers(2).Value();
  auto output_desc = TensorDescs(1).Value();
  auto output_buffers = DataBuffers(1).Value();

  ge::DynamicSingleOp *singleOp = nullptr;
  (void) singleOp;
  (void) modelData;
  (void) input_desc;
  (void) input_buffers;
  (void) output_desc;
  (void) output_buffers;
//  ASSERT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op", modelData, nullptr, &singleOp, 4), SUCCESS);
//  ASSERT_EQ(ge::GeExecutor::ExecuteAsync(singleOp, input_desc, input_buffers, output_desc, output_buffers), SUCCESS);
//  ProfilingData().HasRecord(kInferShape).HasRecord(kUpdateShape).HasRecord(kTiling).HasRecord(kRtKernelLaunch);
}

TEST_F(SingleOpProfilingSt, TestStaticAiCoreMemory) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kMemory});
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);

  auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTilingPro;
  funcs->tiling_parse = StubTilingParsePro;
  funcs->compile_info_creator = CompileInfoCreatorPro;

  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(SingleOpProfilingSt, testSingleopExecuteAsync2WithMemory) {
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kMemory});
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  GE_MAKE_GUARD(stream_destroy, [&stream](){
    rtStreamDestroy(stream);
    stream = nullptr;
  });
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  data_buffer.placement = 1;
  input_buffers.emplace_back(data_buffer);
  vector<DataBuffer> output_buffers;

  single_op.input_sizes_.emplace_back(4);
  SingleOpModelParam model_params;
  single_op.model_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  single_op.args_.resize(1);

  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NHWC, DT_UINT64);
  single_op.inputs_desc_.emplace_back(tensor_desc);
  std::shared_ptr<ge::GeRootModel> root_model = ge::MakeShared<ge::GeRootModel>();
  EXPECT_EQ(single_op.model_param_->runtime_param.mem_base, 0U);
  EXPECT_EQ(single_op.tasks_.size(), 0);
  single_op.model_param_->runtime_param.mem_size = 1024;
  FakeAllocator fake_allocator{};
  single_op.stream_resource_->SetAllocator(&fake_allocator);
  GeTensorDesc tensor;
  int64_t storage_format_val = static_cast<Format>(FORMAT_NCHW);
  AttrUtils::SetInt(tensor, "storage_format", storage_format_val);
  std::vector<int64_t> storage_shape{1, 1, 1, 1};
  AttrUtils::SetListInt(tensor, "storage_shape", storage_shape);
  single_op.inputs_desc_.emplace_back(tensor);
  EXPECT_EQ(single_op.ExecuteAsync(input_buffers, output_buffers), SUCCESS);
  ge::diagnoseSwitch::DisableProfiling();
  for (auto &buff : input_buffers) {
    if (buff.data != nullptr) {
      delete[] reinterpret_cast<char *>(buff.data);
    }
  }
  delete res;
}

void TestFunc() {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);

  auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTilingPro;
  funcs->tiling_parse = StubTilingParsePro;
  funcs->compile_info_creator = CompileInfoCreatorPro;
  RunStaticTestCast(op_desc);
}


TEST_F(SingleOpProfilingSt, TestStaticAiCoreProfiling_ReportApi_TaskTimeL0) {
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime});
  EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestFunc, 1,0,0,0);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(SingleOpProfilingSt, TestStaticAiCoreProfiling_ReportApiAndInfo_TaskTimeL1) {
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice});
  EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestFunc, 1,1,0,1);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(SingleOpProfilingSt, TestStaticAiCoreProfiling_ReportApi_FwkScheduleL0) {
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kCannHost});
  EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestFunc, 1,0,0,0);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(SingleOpProfilingSt, TestStaticAiCoreProfiling_ReportApi_FwkScheduleL1) {
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kCannHostL1, gert::ProfilingType::kCannHost});
  EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestFunc, 1,0,0,0);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(SingleOpProfilingSt, Test_mix_op_prof) {
  ge::ProfilingTestUtil::Instance().Clear();
  ge::diagnoseSwitch::EnableDeviceProfiling();
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  uint32_t task_ratio = 10U;
  ge::AttrUtils::SetBool(op_desc, "_is_fftsplus_task", true);
  ge::AttrUtils::SetBool(op_desc, "_mix_is_aiv", true);
  (void)AttrUtils::SetInt(op_desc, "_task_ratio", task_ratio);

  bool has_context_id = false;
  auto check_func = [&task_ratio, &has_context_id](uint32_t moduleId, uint32_t type, void *data,
                                                   uint32_t len) -> int32_t {
    if ((moduleId == 0) && (type == InfoType::kInfo)) {
      const auto *context_info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      if (context_info->type == MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE) {
        const auto context_data = reinterpret_cast<const MsprofContextIdInfo *>(context_info->data);
        EXPECT_EQ(context_data->ctxIdNum, 1U);
        EXPECT_EQ(context_data->ctxIds[0], 0U);
        has_context_id = true;
      }
      return 0;
    }

    if ((moduleId == 0) && (type == InfoType::kCompactInfo)) {
      const auto &node_basic_info = (reinterpret_cast<MsprofCompactInfo *>(data))->data.nodeBasicInfo;
      EXPECT_EQ(node_basic_info.taskType, MSPROF_GE_TASK_TYPE_MIX_AIV);
      EXPECT_EQ(node_basic_info.blockDim >> 16U, task_ratio);
      return 0;
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);

  auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTilingPro;
  funcs->tiling_parse = StubTilingParsePro;
  funcs->compile_info_creator = CompileInfoCreatorPro;

  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  EXPECT_TRUE(has_context_id == true);
  ge::ProfilingTestUtil::Instance().Clear();
  ge::diagnoseSwitch::DisableProfiling();
}
}  // namespace ge
