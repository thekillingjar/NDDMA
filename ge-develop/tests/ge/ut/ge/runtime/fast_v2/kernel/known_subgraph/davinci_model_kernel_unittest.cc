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
#include <kernel/memory/mem_block.h>
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "register/node_converter_registry.h"
#include "graph/load/model_manager/davinci_model.h"
#include "kernel/known_subgraph/davinci_model_kernel.h"
#include "common/ge_inner_attrs.h"
#include "kernel/known_subgraph/davinci_model_tracing.h"
#include "graph/manager/graph_var_manager.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph_builder/bg_reusable_stream_allocator.h"

using namespace ge;
namespace gert {
namespace {
const void *kExpectGetRunAddress = reinterpret_cast<void *>(0x20220803);
const uintptr_t kFakeMemoryBase = static_cast<uintptr_t>(0x202208031049);
constexpr uint64_t kSessionScopeMemory = 0x100000000U;

const void *kWeightBase = reinterpret_cast<void *>(0x202208122104);
const void *kVarBase = reinterpret_cast<void *>(0x202208122105);
const void *kFileConstantBase = reinterpret_cast<void *>(0x202208122106);
const void *kOffset = reinterpret_cast<void *>(0x996246);

const void *kInputAddressBase = reinterpret_cast<void *>(0x110);
constexpr const char* kInputFrozenIndex = "0,1\0";

enum InputsCommon {
  kDavinciModel,
  kInputsCommonEnd
};

enum class InputsSpecial {
  kDavinciModel,
  kStreamId,
  kInputsCommonEnd
};

enum ModelExecute {
  kRtStream = InputsCommon::kInputsCommonEnd,
  kInputNum,
  kOutputNum,
  kModelExecuteEnd
};

enum UpdateWorkspaces {
  kWorkspacesNum = InputsCommon::kInputsCommonEnd,
  kWorkspaceMemory,
  kUpdateWorkspacesEnd
};
}

class DavinciModelUt : public testing::Test {
 public:
  void SetUp() override {
    mmSetEnv("GE_USE_STATIC_MEMORY", "4", 1);
    auto space_registry_array = SpaceRegistryFaker().BuildRegistryArray();
    ASSERT_NE(space_registry_array, nullptr);
    context_holder_with_davinci_model_ = BuildContextForCreateDavinciModel(space_registry_array);
    auto ret = registry.FindKernelFuncs("DavinciModelCreate")->run_func(context_holder_with_davinci_model_);
    EXPECT_EQ(ret, ge::SUCCESS);
    davinci_model_ = context_holder_with_davinci_model_.value_holder[static_cast<size_t>(kernel::DavinciModelCreateInput::kDavinciModelCreateInputEnd)].GetPointer<ge::DavinciModel>();
    EXPECT_NE(davinci_model_, nullptr);
    auto ret2 = registry.FindKernelFuncs("DavinciModelCreate")->trace_printer(context_holder_with_davinci_model_);
    EXPECT_FALSE(ret2.empty());
  }

  void TearDown() override {
    mmSetEnv("GE_USE_STATIC_MEMORY", "0", 1);
    SetAllDavinciModelExMemoryNull();
    SetAllStreamNull();
  }
  void SetAllDavinciModelExMemoryNull() {
    ge::VarManager::Instance(davinci_model_->GetSessionId())->FreeVarMemory();
    for (auto &mem_info : davinci_model_->GetRuntimeParam().memory_infos) {
      const_cast<MemInfo&>(mem_info.second).memory_base = nullptr;
    }
  }
  void SetAllStreamNull() {
    auto list = davinci_model_->GetStreamList();
    const_cast<std::vector<rtStream_t> &>(list).clear();
  }
  void FillFileConstantVec(KernelRunContextHolder &context) {
    std::vector<kernel::FileConstantNameAndMem> file_constant_weight_mems;
    {
      kernel::FileConstantNameAndMem file1{};
      auto ret = strcpy_s(file1.name, sizeof(file1.name), "file1");
      (void)ret;
      file1.mem = &file_constant_mem;
      file1.size = 100;
      file_constant_weight_mems.emplace_back(file1);
    }
    {
      kernel::FileConstantNameAndMem file1{};
      auto ret = strcpy_s(file1.name, sizeof(file1.name), "file2");
      (void)ret;
      file1.mem = &file_constant_mem;
      file1.size = 100;
      file_constant_weight_mems.emplace_back(file1);
    }
    size_t total_size = 0U;
    size_t vec_size = file_constant_weight_mems.size();
    file_constant_mem_data_ = ContinuousVector::Create<kernel::FileConstantNameAndMem>(vec_size, total_size);
    auto vec = reinterpret_cast<ContinuousVector *>(file_constant_mem_data_.get());
    vec->SetSize(vec_size);
    for (size_t i = 0U; i < vec_size; ++i) {
      *(reinterpret_cast<kernel::FileConstantNameAndMem *>(vec->MutableData()) + i) = file_constant_weight_mems[i];
    }
    context.value_holder[15].Set((void *)file_constant_mem_data_.get(), nullptr);
  }
  KernelRunContextHolder BuildContextForCreateDavinciModel(
      const shared_ptr<OpImplSpaceRegistryV2Array> &space_registry_array) {
    (void)bg::ValueHolder::PushGraphFrame();
    auto graph = ShareGraph::BuildWithKnownSubgraph();
    auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
    faker_ = GlobalDataFaker(root_model);
    auto global_data = faker_.FakeWithoutHandleAiCore("Conv2d", false).Build();
    LowerInput lower_input = {{}, {}, &global_data};
    auto ge_model = reinterpret_cast<ge::GeModel *>(
      lower_input.global_data->GetGraphStaticCompiledModel(parent_node->GetOpDesc()->GetSubgraphInstanceName(0)));
    ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_STREAM_NUM, 1);
    // construct outer weight
    size_t weight_size = ge_model->GetWeightSize();
    weight_data_ = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[weight_size]);
    weight_info_ = {weight_data_.get(), weight_size, kTensorPlacementEnd, -1};

    KernelRunContextHolder context = BuildKernelRunContext(
        static_cast<size_t>(kernel::DavinciModelCreateInput::kDavinciModelCreateInputEnd), 1);
    context.value_holder[0].Set(ge_model, nullptr);
    context.value_holder[1].Set((void *)session_id_, nullptr);
    context.value_holder[2].Set(&weight_info_, nullptr);
    context.value_holder[3].Set((void *)1, nullptr);
    context.value_holder[4].Set((void *)1, nullptr);
    context.value_holder[5].Set(space_registry_array.get(), nullptr);
    std::string file_constant_weight_dir;
    auto file_constant_weight_dir_data =
        static_cast<void *>(const_cast<char *>(file_constant_weight_dir.c_str()));
    context.value_holder[6].Set(file_constant_weight_dir_data, nullptr);
    reusable_stream_allocator_.emplace_back(ge::ReusableStreamAllocator::Create());
    context.value_holder[7].Set((void *)reusable_stream_allocator_.back().get(), nullptr);
    context.value_holder[8].Set((void *)1, nullptr);
    context.value_holder[9].Set((void *)1, nullptr);
    context.value_holder[10].Set((void *)1, nullptr);
    context.value_holder[11].Set((void *)1, nullptr);
    context.value_holder[12].Set((void *)1, nullptr);
    context.value_holder[13].Set((void *)1, nullptr);
    auto frozen_inputs_data = static_cast<void *>(const_cast<char *>(kInputFrozenIndex));
    context.value_holder[14].Set(frozen_inputs_data, nullptr);
    FillFileConstantVec(context);
    (void)bg::ValueHolder::PopGraphFrame();
    return context;
  }

  KernelContext *BuildContextForUpdateWorkspaces() {
    ConstructFakeRuntimeParam();
    const size_t workspace_num = runtime_param_.memory_infos.size();
    context_ = BuildKernelRunContext(workspace_num * 2U + 2U, 1U);
    context_.value_holder[kDavinciModel].Set(davinci_model_, nullptr);
    context_.value_holder[kWorkspacesNum].Set(reinterpret_cast<void *>(workspace_num * 2), nullptr);

    size_t index = kWorkspaceMemory;
    for (const auto &mem_info : runtime_param_.memory_infos) {
      const auto type = mem_info.second.memory_type;
      const auto address = mem_info.second.memory_base;

      context_.value_holder[index].Set(reinterpret_cast<void *>(type), nullptr);
      inputs_tensor_data[index] = {reinterpret_cast<void *>(address), 0U, kTensorPlacementEnd, -1};
      context_.value_holder[index + 1U].Set(&inputs_tensor_data[index], nullptr);
      index += 2;
    }
    return context_;
  }

  void ConstructFakeRuntimeParam() {
    runtime_param_.memory_infos.clear();
    std::vector<uint64_t> memory_types{RT_MEMORY_HBM, kSessionScopeMemory | RT_MEMORY_HBM, RT_MEMORY_P2P_DDR};
    for (const auto type : memory_types) {
      MemInfo mem_info;
      mem_info.memory_type = type;
      mem_info.memory_base = reinterpret_cast<uint8_t *>(kFakeMemoryBase);
      runtime_param_.memory_infos[type] = mem_info;
    }
  }

  KernelContext *BuildContextForUpdateWorkspacesInvalidType() {
    runtime_param_.memory_infos.clear();
    std::vector<uint64_t> memory_types{0x888};
    for (const auto type : memory_types) {
      MemInfo mem_info;
      mem_info.memory_type = type;
      mem_info.memory_base = reinterpret_cast<uint8_t *>(kFakeMemoryBase);
      runtime_param_.memory_infos[type] = mem_info;
    }
    const size_t workspace_num = runtime_param_.memory_infos.size();
    context_ = BuildKernelRunContext(workspace_num * 2U + 2U, 1U);
    context_.value_holder[kDavinciModel].Set(davinci_model_, nullptr);
    context_.value_holder[kWorkspacesNum].Set(reinterpret_cast<void *>(workspace_num), nullptr);

    size_t index = kWorkspaceMemory;
    for (const auto &mem_info : runtime_param_.memory_infos) {
      const auto type = mem_info.second.memory_type;
      const auto address = mem_info.second.memory_base;

      context_.value_holder[index].Set(reinterpret_cast<void *>(type), nullptr);
      inputs_tensor_data[index] = {reinterpret_cast<void *>(address), 0U, kTensorPlacementEnd, -1};
      context_.value_holder[index + 1U].Set(&inputs_tensor_data[index], nullptr);
      index += 2;
    }
    return context_;
  }

  KernelContext *BuildContextForDavinciModelExecute() {
    const size_t inputs_num = 1U;
    const size_t outputs_num = 3U;
    context_ = BuildKernelRunContext(kModelExecuteEnd + inputs_num + outputs_num, 1U);
    context_.value_holder[kDavinciModel].Set(davinci_model_, nullptr);
    stream_ = reinterpret_cast<void *>(0x123);
    context_.value_holder[kRtStream].Set(stream_, nullptr);
    context_.value_holder[kInputNum].Set(reinterpret_cast<void *>(inputs_num), nullptr);
    context_.value_holder[kOutputNum].Set(reinterpret_cast<void *>(outputs_num), nullptr);

    for (size_t i = 0U; i < inputs_num; ++i) {
      inputs_tensor_data[i] = {reinterpret_cast<void *>(PtrAdd<uint32_t>((uint32_t *)kInputAddressBase, 100, i)), 0U, kTensorPlacementEnd, -1};
      context_.value_holder[kModelExecuteEnd + i].Set(&inputs_tensor_data[i], nullptr);
    }
    for (size_t i = 0U; i < outputs_num; ++i) {
      outputs_tensor_data[i] = {0, 16, kTensorPlacementEnd, -1};
      context_.value_holder[kModelExecuteEnd + inputs_num + i].Set(&outputs_tensor_data[i], nullptr);
    }

    return context_;
  }

  KernelContext *BuildContextForGetRunAddress() {
    const size_t inputs_num = static_cast<size_t>(kernel::MemoryBaseType::kMemoryBaseTypeEnd);
    const size_t outputs_num = static_cast<size_t>(kernel::MemoryBaseType::kMemoryBaseTypeEnd);
    context_ = BuildKernelRunContext(inputs_num + 2U, outputs_num);
    context_.value_holder[static_cast<size_t>(InputsSpecial::kDavinciModel)].Set(davinci_model_, nullptr);
    context_.value_holder[static_cast<size_t>(InputsSpecial::kStreamId)].Set(&logic_stream_id_, nullptr);

    RuntimeParam &runtime_param = const_cast<RuntimeParam &>(davinci_model_->GetRuntimeParam());
    runtime_param.weight_base = reinterpret_cast<uintptr_t>(kWeightBase);
    runtime_param.mem_base = reinterpret_cast<uintptr_t>(kExpectGetRunAddress);
    runtime_param.var_base = reinterpret_cast<uintptr_t>(kVarBase);
    runtime_param.fileconstant_addr_mapping[reinterpret_cast<uintptr_t>(kOffset)] =
        reinterpret_cast<uintptr_t>(kFileConstantBase);

    for (size_t i = 0U; i < inputs_num; ++i) {
      memory_type_offset_[i].offset = reinterpret_cast<uintptr_t>(kOffset);
      memory_type_offset_[i].base_type = static_cast<kernel::MemoryBaseType>(i);
      context_.value_holder[static_cast<size_t>(InputsSpecial::kInputsCommonEnd) + i].Set(&memory_type_offset_[i],
                                                                                          nullptr);
    }
    for (size_t i = 0U; i < outputs_num; ++i) {
      outputs_tensor_data[i] = {reinterpret_cast<void *>(0x110), 0U, kTensorPlacementEnd, -1};
      context_.value_holder[static_cast<size_t>(InputsSpecial::kInputsCommonEnd) + inputs_num + i].Set(
          &outputs_tensor_data[i], nullptr);
    }

    return context_;
  }

  KernelContext *BuildContextForGetRunAddressWithInvalidType() {
    const size_t inputs_num = 1;
    const size_t outputs_num = 1;
    context_ = BuildKernelRunContext(inputs_num + 2U, outputs_num);
    context_.value_holder[static_cast<size_t>(InputsSpecial::kDavinciModel)].Set(davinci_model_, nullptr);
    context_.value_holder[static_cast<size_t>(InputsSpecial::kStreamId)].Set(&logic_stream_id_, nullptr);

    for (size_t i = 0U; i < inputs_num; ++i) {
      memory_type_offset_[i].offset = 0;
      memory_type_offset_[i].base_type = kernel::MemoryBaseType::kMemoryBaseTypeEnd;
      context_.value_holder[static_cast<size_t>(InputsSpecial::kInputsCommonEnd) + i].Set(&memory_type_offset_[i], nullptr);
    }
    for (size_t i = 0U; i < outputs_num; ++i) {
      outputs_tensor_data[i] = {reinterpret_cast<void *>(0x110), 0U, kTensorPlacementEnd, -1};
      context_.value_holder[static_cast<size_t>(InputsSpecial::kInputsCommonEnd) + inputs_num + i].Set(&outputs_tensor_data[i], nullptr);
    }

    return context_;
  }

  KernelContext *BuildContextForDavinciModelExecuteNullStream() {
    BuildContextForDavinciModelExecute();
    stream_ = nullptr;
    context_.value_holder[kRtStream].Set(stream_, nullptr);
    return context_;
  }
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
  std::deque<unique_ptr<ge::ReusableStreamAllocator>> reusable_stream_allocator_;
  ge::DavinciModel *davinci_model_;
  KernelRunContextHolder context_holder_with_davinci_model_;
  int64_t logic_stream_id_{0};

  GertTensorData inputs_tensor_data[10];
  GertTensorData outputs_tensor_data[10];
  KernelRunContextHolder context_;
  GlobalDataFaker faker_;
  kernel::MemoryBaseTypeOffset memory_type_offset_[static_cast<int32_t>(kernel::MemoryBaseType::kMemoryBaseTypeEnd)];
  rtStream_t stream_;

  RuntimeParam runtime_param_;
  uint64_t session_id_ = 1U;
  std::unique_ptr<uint8_t[]> weight_data_;
  GertTensorData weight_info_;
  uint32_t file_constant_mem[256];
  std::unique_ptr<uint8_t[]> file_constant_mem_data_;
};

TEST_F(DavinciModelUt, UpdateWorkspacesSuccess) {
  KernelContext *context = BuildContextForUpdateWorkspaces();
  auto ret = registry.FindKernelFuncs("DavinciModelUpdateWorkspaces")->run_func(context);
  ASSERT_EQ(ret, ge::SUCCESS);
  for (const auto &mem_info : runtime_param_.memory_infos) {
    const auto type = mem_info.first;
    const auto base = mem_info.second.memory_base;
    if (type == RT_MEMORY_HBM) {
      EXPECT_EQ(davinci_model_->GetRuntimeParam().mem_base, reinterpret_cast<uintptr_t>(base));
      continue;
    }
    auto iter = davinci_model_->GetRuntimeParam().memory_infos.find(type);
    ASSERT_NE(iter, davinci_model_->GetRuntimeParam().memory_infos.end());
    EXPECT_EQ(iter->second.memory_base, base);
  }
  auto ret2 = registry.FindKernelFuncs("DavinciModelUpdateWorkspaces")->trace_printer(context);
  EXPECT_FALSE(ret2.empty());
}

TEST_F(DavinciModelUt, UpdateWorkspacesWithInvalidTypeFailed) {
  KernelContext *context = BuildContextForUpdateWorkspacesInvalidType();
  auto ret = registry.FindKernelFuncs("DavinciModelUpdateWorkspaces")->run_func(context);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(DavinciModelUt, GetRunAddressSuccess) {
  KernelContext *context = BuildContextForGetRunAddress();
  auto ret = registry.FindKernelFuncs("DavinciModelGetRunAddress")->run_func(context);
  ASSERT_EQ(ret, ge::SUCCESS);
  auto ret2 = registry.FindKernelFuncs("DavinciModelGetRunAddress")->trace_printer(context);
  EXPECT_FALSE(ret2.empty());

  size_t outputs_num = context->GetOutputNum();
  ASSERT_EQ(outputs_num, 2U);

  auto tensor_data = context->GetOutputPointer<GertTensorData>(0);
  auto expect_addr = reinterpret_cast<uintptr_t>(kWeightBase) + reinterpret_cast<uintptr_t>(kOffset);
  EXPECT_EQ(tensor_data->GetAddr(), reinterpret_cast<void *>(expect_addr));

  tensor_data = context->GetOutputPointer<GertTensorData>(1);
  EXPECT_EQ(tensor_data->GetAddr(), kFileConstantBase);

  RuntimeParam &runtime_param = const_cast<RuntimeParam &>(davinci_model_->GetRuntimeParam());
  runtime_param.fileconstant_addr_mapping.clear();
}

TEST_F(DavinciModelUt, GetRunAddressMemoryBaseTypeInvalidFailed) {
  KernelContext *context = BuildContextForGetRunAddressWithInvalidType();
  auto ret = registry.FindKernelFuncs("DavinciModelGetRunAddress")->run_func(context);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(DavinciModelUt, DavinciModelExecuteNullStreamFailed) {
  KernelContext *context = BuildContextForDavinciModelExecuteNullStream();
  kernel::PrintModelExecute(context);
  auto ret = registry.FindKernelFuncs("DavinciModelExecute")->run_func(context);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(DavinciModelUt, DavinciModelExecuteWithStream) {
  KernelContext *context = BuildContextForDavinciModelExecute();
  kernel::PrintModelExecute(context);
  auto ret = registry.FindKernelFuncs("DavinciModelExecute")->run_func(context);
  EXPECT_EQ(ret, ge::SUCCESS);
}
} // namespace gert
