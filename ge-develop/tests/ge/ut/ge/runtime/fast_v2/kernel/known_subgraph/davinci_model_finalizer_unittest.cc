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

#include "common/share_graph.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "faker/global_data_faker.h"
#include "faker/kernel_run_context_facker.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph_builder/bg_reusable_stream_allocator.h"
#include "register/node_converter_registry.h"
#include "register/kernel_registry.h"
#include "kernel/known_subgraph/davinci_model_kernel.h"

using namespace ge;
namespace gert {
class DavinciModelFinalizerUt : public testing::Test {
 public:
  void SetUp() override {
    auto space_registry_array = SpaceRegistryFaker().BuildRegistryArray();
    ASSERT_NE(space_registry_array, nullptr);
    for (int32_t i = 0; i < 2; ++i) {
      context_holder_with_davinci_model_.emplace_back(BuildContextForCreateDavinciModel(space_registry_array));
      auto ret = registry.FindKernelFuncs("DavinciModelCreate")->run_func(context_holder_with_davinci_model_.back());
      EXPECT_EQ(ret, ge::SUCCESS);
      auto davinci_model = context_holder_with_davinci_model_.back().value_holder[static_cast<size_t>(kernel::DavinciModelCreateInput::kDavinciModelCreateInputEnd)].GetPointer<ge::DavinciModel>();
      EXPECT_NE(davinci_model, nullptr);
      auto ret2 =
          registry.FindKernelFuncs("DavinciModelCreate")->trace_printer(context_holder_with_davinci_model_.back());
      EXPECT_FALSE(ret2.empty());
      davinci_model_.emplace_back(davinci_model);
    }
  }

  void FillFileConstantVec(KernelRunContextHolder &context) {
    size_t total_size = 0U;
    size_t vec_size = 0;
    file_constant_mem_data_ = ContinuousVector::Create<kernel::FileConstantNameAndMem>(vec_size, total_size);
    auto vec = reinterpret_cast<ContinuousVector *>(file_constant_mem_data_.get());
    if (vec == nullptr) {
      std::cerr << "file_constant_mem_data_ is nullptr" << std::endl;
    }
    vec->SetSize(vec_size);
    context.value_holder[15].Set((void *)file_constant_mem_data_.get(), nullptr);
  }

  KernelRunContextHolder BuildContextForCreateDavinciModel(std::shared_ptr<OpImplSpaceRegistryV2Array> space_registry_array) {
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

    KernelRunContextHolder context = BuildKernelRunContext(static_cast<size_t>(kernel::DavinciModelCreateInput::kDavinciModelCreateInputEnd), 1);
    context.value_holder[0].Set(ge_model, nullptr);
    context.value_holder[1].Set((void *)session_id_, nullptr);
    context.value_holder[2].Set(&weight_info_, nullptr);
    context.value_holder[3].Set((void *)1, nullptr);
    context.value_holder[4].Set((void *)1, nullptr);
    context.value_holder[5].Set(space_registry_array.get(), nullptr);
    std::string file_constant_weight_dir;
    auto file_constant_weight_dir_data = static_cast<void *>(const_cast<char *>(file_constant_weight_dir.c_str()));
    context.value_holder[6].Set(file_constant_weight_dir_data, nullptr);
    reusable_stream_allocator_.emplace_back(ge::ReusableStreamAllocator::Create());
    context.value_holder[7].Set((void *)reusable_stream_allocator_.back().get(), nullptr);
    context.value_holder[8].Set((void *)1, nullptr);
    context.value_holder[9].Set((void *)1, nullptr);
    context.value_holder[10].Set((void *)1, nullptr);
    context.value_holder[11].Set((void *)1, nullptr);
    context.value_holder[12].Set((void *)1, nullptr);
    context.value_holder[13].Set((void *)1, nullptr);
    std::string frozen_inputs;
    auto frozen_inputs_data = static_cast<void *>(const_cast<char *>(frozen_inputs.c_str()));
    context.value_holder[14].Set(frozen_inputs_data, nullptr);
    FillFileConstantVec(context);
    (void)bg::ValueHolder::PopGraphFrame();
    return context;
  }

 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
  std::deque<unique_ptr<ge::ReusableStreamAllocator>> reusable_stream_allocator_;
  std::vector<ge::DavinciModel *> davinci_model_;
  std::vector<KernelRunContextHolder> context_holder_with_davinci_model_;
  GlobalDataFaker faker_;
  uint64_t session_id_ = 1U;
  std::unique_ptr<uint8_t[]> weight_data_;
  GertTensorData weight_info_;
  std::unique_ptr<uint8_t[]> file_constant_mem_data_;
};

TEST_F(DavinciModelFinalizerUt, test_multi_finalize_davinci_model_succ) {
  std::vector<void *> davinci_model_list{this->davinci_model_.begin(), this->davinci_model_.end()};
  auto alloc_context_holder = KernelRunContextFaker().KernelIONum(2, 0).Inputs(davinci_model_list).Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(this->davinci_model_[0]->has_finalized_, false);
  ASSERT_EQ(this->davinci_model_[1]->has_finalized_, false);
  ASSERT_EQ(registry.FindKernelFuncs("DavinciModelFinalizer")->run_func(alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(this->davinci_model_[0]->has_finalized_, true);
  ASSERT_EQ(this->davinci_model_[1]->has_finalized_, true);
}
}  // namespace gert