/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __LLT_HCCL_STUB_KERNEL_RUN_CTX_FAKER_H__
#define __LLT_HCCL_STUB_KERNEL_RUN_CTX_FAKER_H__

#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/lowering/shape_utils.h"
#include "exe_graph/runtime/tensor_data_utils.h"

namespace gert{
struct FakeKernelContextHolder {
  template<typename T>
  T *GetContext() {
    return reinterpret_cast<T*>(holder.context_);
  }

  size_t kernel_input_num;
  size_t kernel_output_num;
  KernelContextHolder holder;
};

class KernelRunContextFaker {
 public:
  KernelRunContextFaker() = default;
  KernelRunContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  KernelRunContextFaker &KernelIONum(size_t input_num, size_t output_num);
  FakeKernelContextHolder Build() const;

  KernelRunContextFaker &Inputs(std::vector<void *> inputs) {
    inputs_ = std::move(inputs);
    return *this;
  }

  KernelRunContextFaker &Outputs(std::vector<void *> outputs) {
    outputs_ = std::move(outputs);
    return *this;
  }

 private:
  ge::OpDescPtr FakeOp() const;

 private:
  size_t kernel_input_num_;
  size_t kernel_output_num_;
  size_t node_input_num_;
  size_t node_output_num_;
  std::vector<uint32_t> ir_instance_num_;
  std::vector<uint32_t> ir_output_instance_num_{};
  std::vector<CompileTimeTensorDesc> node_input_tds_;
  std::vector<CompileTimeTensorDesc> node_output_tds_;
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  std::vector<std::pair<std::string, ge::AnyValue>> attrs_;
};

class GertMemBlockFaker : public GertMemBlock {
  public:
  explicit GertMemBlockFaker(void *addr) : addr_(addr) {}
  ~GertMemBlockFaker() override = default;
  void Free(int64_t stream_id) override {}
  void *GetAddr() override { return addr_; }
  private:
  void *addr_;
};
class AllocatorFaker : public GertAllocator {
  GertMemBlock *Malloc(size_t size) override;
  void Free(GertMemBlock *block) override;
  GertTensorData MallocTensorData(size_t size) override {
    return GertTensorData();
  }
  TensorData MallocTensorDataFromL1(size_t size) override {
    return TensorData();
  }
  ge::graphStatus ShareFromTensorData(const TensorData &td, GertTensorData &gtd) override {
    return 0;
  }
  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override {
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus FreeAt(int64_t stream_id, GertMemBlock *block) override {
    return ge::GRAPH_SUCCESS;
  }
  int64_t GetStreamNum() override {
    return 0;
  }
};

}
#endif  // __LLT_HCCL_STUB_KERNEL_RUN_CTX_FAKER_H__
