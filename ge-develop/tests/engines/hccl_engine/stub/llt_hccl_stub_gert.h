/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __LLT_HCCL_STUB_GERT_H__
#define __LLT_HCCL_STUB_GERT_H__

#include "register/node_converter_registry.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/compiler_def.h"
#include "register/op_impl_kernel_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include <initializer_list>
#include <string>
#include <map>

#include "exe_graph/runtime/tensor_data_utils.h"
// #include "inc/common/checker.h"

namespace gert{
namespace bg{
class GenerateExeGraph {
 public:
  struct ExeGraphGenerator {
    using InferShapeFunc = std::vector<ValueHolderPtr> (*)(const ge::NodePtr &node,
                                                           const std::vector<ValueHolderPtr> &shapes);
    using AllocOutputMemoryFunc = std::vector<DevMemValueHolderPtr> (*)(TensorPlacement placement, const ge::NodePtr &node,
                                                                        const std::vector<ValueHolderPtr> &output_sizes,
                                                                        LoweringGlobalData &global_data);
    using CalcTensorSizeFunc = std::vector<ValueHolderPtr> (*)(const ge::NodePtr &node,
                                                               const std::vector<ValueHolderPtr> &output_shapes);

    InferShapeFunc infer_shape;
    AllocOutputMemoryFunc alloc_output_memory;
    CalcTensorSizeFunc calc_tensor_size;
  };

 public:
class DevMemValueHolder : public ValueHolder {
   static std::vector<DevMemValueHolderPtr> CreateDataOutput(const char *node_type,
                                                            const std::vector<ValueHolderPtr> &inputs,
                                                            size_t out_count, int64_t logic_stream_id);
};
  static std::vector<ValueHolderPtr> InferShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &shapes) {
    if (generator_.infer_shape == nullptr) {
      return {};
    }
    return generator_.infer_shape(node, shapes);
  }
  static std::vector<DevMemValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                             const std::vector<ValueHolderPtr> &output_sizes,
                                                             LoweringGlobalData &global_data) {
    if (generator_.alloc_output_memory == nullptr) {
      return {};
    }
    return generator_.alloc_output_memory(placement, node, output_sizes, global_data);
  }
  static std::vector<ValueHolderPtr> CalcTensorSize(const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &output_shapes) {
    std::vector<ValueHolderPtr> holders;
    size_t outputSize = output_shapes.size();
    ValueHolderPtr outputHolder;
    for (size_t i = 0; i < outputSize; i++) {
        holders.push_back(outputHolder);
    }
    return holders;
  }

  static void AddBuilderImplement(ExeGraphGenerator generator) {
    generator_ = generator;
  }

 private:
  static ExeGraphGenerator generator_;
};
}

class KernelContextHolder {
public:
  KernelContextHolder() = default;
  KernelContextHolder(KernelContextHolder &&holder) {
    context_holder_ = std::move(holder.context_holder_);
    value_holder_ = std::move(holder.value_holder_);
    compute_node_extend_holder_ = std::move(holder.compute_node_extend_holder_);
    buffer_pool_ = holder.buffer_pool_;
    context_ = holder.context_;
  }

  KernelContextHolder &operator=(KernelContextHolder &&holder) {
    context_holder_ = std::move(holder.context_holder_);
    value_holder_ = std::move(holder.value_holder_);
    compute_node_extend_holder_ = std::move(holder.compute_node_extend_holder_);
    buffer_pool_ = holder.buffer_pool_;
    context_ = holder.context_;
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

  KernelContextHolder Build(const ge::OpDescPtr &op_desc);

private:
  ge::NodePtr MakeNode(const ge::OpDescPtr &op_desc);
private:
  ge::ComputeGraphPtr graph_;
  std::vector<std::pair<void *, Chain::Deleter>> inputs_;
  std::vector<std::pair<void *, Chain::Deleter>> outputs_;
};
}
#endif  // __LLT_HCCL_STUB_GERT_H__
