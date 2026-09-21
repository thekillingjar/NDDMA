/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_RUNTIME_KERNEL_CONTEXT_BUILDER_H_
#define METADEF_CXX_RUNTIME_KERNEL_CONTEXT_BUILDER_H_

#include "graph/node.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/lowering/buffer_pool.h"

namespace gert {
class KernelContextHolder {
public:
  KernelContextHolder() = default;
  KernelContextHolder(KernelContextHolder &&holder) {
    context_holder_ = std::move(holder.context_holder_);
    value_holder_ = std::move(holder.value_holder_);
    compute_node_extend_holder_ = std::move(holder.compute_node_extend_holder_);
    buffer_pool_ = holder.buffer_pool_;
    context_ = holder.context_;
    holder.context_ = nullptr;
  }

  KernelContextHolder &operator=(KernelContextHolder &&holder) {
    context_holder_ = std::move(holder.context_holder_);
    value_holder_ = std::move(holder.value_holder_);
    compute_node_extend_holder_ = std::move(holder.compute_node_extend_holder_);
    buffer_pool_ = holder.buffer_pool_;
    context_ = holder.context_;
    holder.context_ = nullptr;
    return *this;
  }

  ~KernelContextHolder() {
    for (auto &value : value_holder_) {
      value.Set(nullptr, nullptr);
    }
  }

  KernelContext *GetKernelContext() {
    return context_;
  }

  std::unique_ptr<uint8_t[]> context_holder_;
  std::vector<Chain> value_holder_;
  std::unique_ptr<uint8_t[]> compute_node_extend_holder_;
  bg::BufferPool buffer_pool_;
  KernelContext *context_;
};
class KernelRunContextBuilder {
public:
  KernelRunContextBuilder() = default;
  KernelRunContextBuilder &Inputs(std::vector<std::pair<void *, Chain::Deleter>> inputs) {
    inputs_ = std::move(inputs);
    return *this;
  }

  KernelRunContextBuilder &Inputs(std::vector<void *> inputs) {
    for (auto &input : inputs) {
      inputs_.emplace_back(input, nullptr);
    }
    return *this;
  }

  KernelRunContextBuilder &Outputs(std::vector<void *> outputs) {
    for (auto &output : outputs) {
      outputs_.emplace_back(output, nullptr);
    }
    return *this;
  }

  KernelRunContextBuilder &Outputs(std::vector<std::pair<void *, Chain::Deleter>> outputs) {
    outputs_ = std::move(outputs);
    return *this;
  }

  // deprecated when air use Build interface with ret_status
  KernelContextHolder Build(const ge::OpDescPtr &op_desc);
  KernelContextHolder Build(const ge::OpDescPtr &op_desc, ge::graphStatus &ret);

private:
  ge::NodePtr MakeNode(const ge::OpDescPtr &op_desc);
private:
  ge::ComputeGraphPtr graph_;
  std::vector<std::pair<void *, Chain::Deleter>> inputs_;
  std::vector<std::pair<void *, Chain::Deleter>> outputs_;
};
}  // namespace gert
#endif
