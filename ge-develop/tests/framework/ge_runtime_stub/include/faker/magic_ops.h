/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_NODES_FAKER_MAGIC_OPS_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_NODES_FAKER_MAGIC_OPS_H_
#include "base_node_exe_faker.h"
#include "fake_kernel_definitions.h"
#include "common/checker.h"
#include "nodes_faker_for_exe.h"
#include "register/kernel_registry.h"
#include "stub/gert_runtime_stub.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
class MagicOpFaker : public BaseNodeExeFaker {
 public:
  MagicOpFaker(TensorPlacement placement, KernelRegistry::KernelFunc kernel,
               std::map<size_t, size_t> index_to_output_size)
      : placement_(placement),
        kernel_(kernel),
        index_to_output_size_(index_to_output_size) {}
  ~MagicOpFaker() {
    for (auto buffer : memory_holders) {
      free(buffer);
    }
  }
  TensorPlacement GetOutPlacement() const override {
    return placement_;
  }
  ge::graphStatus RunFunc(KernelContext *context) override {
    return kernel_(context);
  }

  ge::graphStatus OutputCreator(const ge::FastNode *node, KernelContext *context) const override {
    auto output_num = context->GetOutputNum();

    size_t output_index = 0;
    for (; output_index < output_num / 2; ++output_index) {
      context->GetOutput(output_index)->SetWithDefaultDeleter(new StorageShape());
    }
    for (; output_index < output_num; ++output_index) {
      auto iter = index_to_output_size_.find(output_index - output_num / 2);
      size_t output_size = (iter != index_to_output_size_.end()) ? iter->second : 128U;
      memory_holders.emplace_back(malloc(output_size));
      memset(memory_holders.back(), 0, output_size);
      auto gert_tensor_data = new GertTensorData();
      *gert_tensor_data = {memory_holders.back(), 0U, kTensorPlacementEnd, -1};
      context->GetOutput(output_index)->SetWithDefaultDeleter(gert_tensor_data);
    }
    return ge::GRAPH_SUCCESS;
  }

 private:
  TensorPlacement placement_;
  KernelRegistry::KernelFunc kernel_;

  std::map<size_t, size_t> index_to_output_size_;
  mutable std::vector<void*> memory_holders;
};

class MagicOpFakerBuilder {
 public:
  explicit MagicOpFakerBuilder(std::string type) : type_(std::move(type)) {}
  MagicOpFakerBuilder &RequireInputPlacement(int32_t placement) {
    require_input_placement_ = placement;
    return *this;
  }

  MagicOpFakerBuilder &OutputPlacement(TensorPlacement placement) {
    placement_ = placement;
    return *this;
  }
  MagicOpFakerBuilder &KernelFunc(KernelRegistry::KernelFunc kernel) {
    kernel_ = kernel;
    return *this;
  }
  MagicOpFakerBuilder &OutputSize(size_t index, size_t max_size) {
    index_to_output_size_[index] = max_size;
    return *this;
  }
  void Build(GertRuntimeStub &holder) {
    auto faker = std::make_shared<MagicOpFaker>(placement_, kernel_, index_to_output_size_);
    holder.StubByNodeTypes({type_}, {require_input_placement_}).AddCtxFaker(type_, faker);
  }

 private:
  std::string type_;

  TensorPlacement placement_ = kOnDeviceHbm;
  int32_t require_input_placement_ = -1;
  KernelRegistry::KernelFunc kernel_{};
  std::map<size_t, size_t> index_to_output_size_;
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_NODES_FAKER_MAGIC_OPS_H_
