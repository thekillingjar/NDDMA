/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/device_tiling_context_builder.h"

#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/def_types.h"
#include "common/checker.h"

namespace gert {
namespace {
constexpr size_t kChainMemAlignedSize = 256UL;

inline static size_t MemoryAligned(const size_t bytes, const size_t aligns = 128U) {
  const size_t aligned_size = (aligns == 0UL) ? sizeof(uintptr_t) : aligns;
  return ((bytes + aligned_size - 1UL) / aligned_size) * aligned_size;
}

static void TiledPointerOffset(const size_t offset_size, uint8_t *&host_addr, uint64_t &dev_addr,
                               size_t &max_mem_size) {
  max_mem_size -= offset_size;
  host_addr += offset_size;
  dev_addr += offset_size;
}

void GetStorageShape(const ge::GeTensorDesc &tensor_desc, gert::StorageShape &storage_shape) {
  const auto &storage_dims = tensor_desc.GetShape().GetDims();
  for (const auto &dim : storage_dims) {
    (void) storage_shape.MutableStorageShape().AppendDim(dim);
  }
  const auto &origin_dims = tensor_desc.GetOriginShape().GetDims();
  for (const auto &dim : origin_dims) {
    (void) storage_shape.MutableOriginShape().AppendDim(dim);
  }
}
}  // namespace

size_t DeviceTilingContextBuilder::CalcTotalTiledSize(const ge::OpDescPtr &op_desc) {
  // op infos
  size_t total_size{op_desc->GetName().size() + 1UL};  // \0
  total_size += op_desc->GetType().size() + 1UL;       // \0

  // gert::tensor size
  const size_t io_num = op_desc->GetInputsSize() + op_desc->GetOutputsSize();
  total_size += io_num * sizeof(gert::Tensor);

  // kernel context_size
  const size_t chain_num = io_num + static_cast<size_t>(TilingContext::kOutputNum) + 4UL;  // default input ptr nums
  const size_t context_size = sizeof(KernelRunContext) + sizeof(Chain *) * chain_num;
  const size_t chain_size = (sizeof(Chain) + kChainMemAlignedSize) * chain_num;
  total_size += context_size;
  total_size += chain_size;
  return total_size;
}

DeviceTilingContextBuilder &DeviceTilingContextBuilder::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  return *this;
}
DeviceTilingContextBuilder &DeviceTilingContextBuilder::PlatformInfo(void *platform_info) {
  platform_info_ = platform_info;
  return *this;
}
DeviceTilingContextBuilder &DeviceTilingContextBuilder::Deterministic(int32_t deterministic) {
  deterministic_ = deterministic;
  return *this;
}

DeviceTilingContextBuilder &DeviceTilingContextBuilder::DeterministicLevel(int32_t deterministic_level) {
  deterministic_level_ = deterministic_level;
  return *this;
}

DeviceTilingContextBuilder &DeviceTilingContextBuilder::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  return *this;
}

DeviceTilingContextBuilder &DeviceTilingContextBuilder::AddrRefreshedInputTensor(
    const std::map<size_t, AddrRefreshedTensor> &index_to_tensor) {
  index_to_tensor_ = index_to_tensor;
  return *this;
}

DeviceTilingContextBuilder &DeviceTilingContextBuilder::Workspace(void *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  return *this;
}

DeviceTilingContextBuilder &DeviceTilingContextBuilder::TiledHolder(uint8_t *host_addr, uint64_t dev_addr,
                                                                    size_t max_mem_size) {
  host_begin_ = host_addr;
  dev_begin_ = dev_addr;
  max_mem_size_ = max_mem_size;
  return *this;
}

ge::graphStatus DeviceTilingContextBuilder::BuildRtTensor(const ge::GeTensorDesc &tensor_desc,
                                                          ConstTensorAddressPtr address) {
  gert::StorageShape storage_shape;
  GetStorageShape(tensor_desc, storage_shape);
  const size_t rt_tensor_size = sizeof(gert::Tensor);
  GE_ASSERT(max_mem_size_ >= rt_tensor_size);
  GE_ASSERT_NOTNULL(host_begin_);
  auto rt_tensor = new (host_begin_)(gert::Tensor);
  GE_ASSERT_NOTNULL(rt_tensor);
  rt_tensor->SetDataType(tensor_desc.GetDataType());
  rt_tensor->MutableStorageShape() = storage_shape.GetStorageShape();
  rt_tensor->MutableOriginShape() = storage_shape.GetOriginShape();
  rt_tensor->MutableFormat().SetStorageFormat(tensor_desc.GetFormat());
  rt_tensor->MutableFormat().SetOriginFormat(tensor_desc.GetOriginFormat());
  (void) rt_tensor->MutableTensorData().SetAddr(address, nullptr);
  rt_tensor->MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
  // dev_value
  inputs_.push_back(ge::ValueToPtr(dev_begin_));
  dev_begin_ += rt_tensor_size;
  max_mem_size_ -= rt_tensor_size;
  host_begin_ += rt_tensor_size;
  GELOGD("Build rt tensor from device addr %lx.", dev_begin_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeviceTilingContextBuilder::BuildPlacementRtTensor(const ge::GeTensorDesc &tensor_desc,
                                                                   Tensor *rt_tensor) const {
  GE_ASSERT_NOTNULL(rt_tensor);
  gert::StorageShape storage_shape;
  GetStorageShape(tensor_desc, storage_shape);
  rt_tensor->SetDataType(tensor_desc.GetDataType());
  rt_tensor->MutableStorageShape() = storage_shape.GetStorageShape();
  rt_tensor->MutableOriginShape() = storage_shape.GetOriginShape();
  rt_tensor->MutableFormat().SetStorageFormat(tensor_desc.GetFormat());
  rt_tensor->MutableFormat().SetOriginFormat(tensor_desc.GetOriginFormat());
  rt_tensor->MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeviceTilingContextBuilder::BuildIOTensors(const ge::OpDesc *const op_desc) {
  GE_ASSERT_NOTNULL(op_desc);
  size_t valid_inputs{0UL};
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const ge::GeTensorDesc &input_desc = op_desc->GetInputDesc(i);
    if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
      continue;
    }
    const auto iter = index_to_tensor_.find(valid_inputs);
    if (iter != index_to_tensor_.end()) {
      GE_ASSERT_GRAPH_SUCCESS(BuildPlacementRtTensor(input_desc, iter->second.host_addr));
      // dev_value
      inputs_.push_back(ge::ValueToPtr(iter->second.device_addr));
    } else {
      GE_ASSERT_GRAPH_SUCCESS(BuildRtTensor(input_desc, nullptr));
    }
    ++valid_inputs;
  }

  for (size_t i = 0UL; i < op_desc->GetOutputsSize(); ++i) {
    GE_ASSERT_GRAPH_SUCCESS(BuildRtTensor(op_desc->GetOutputDesc(i), nullptr));
  }
  return ge::GRAPH_SUCCESS;
}

// 0-n input tensors
// n-m output shapes
// m + 1 compile info
// m + 2 tiling func
// 其中 n为输入个数总和，m为输入输出个数总和
ge::graphStatus DeviceTilingContextBuilder::Build(const ge::NodePtr &node, TiledKernelContextHolder &holder) {
  GE_ASSERT_NOTNULL(platform_info_, " Device platform info addr is nullptr.");
  GE_ASSERT_EOK(memset_s(host_begin_, max_mem_size_, 0, max_mem_size_), "Failed to memset host context buffer.");

  inputs_.clear();
  GE_ASSERT_GRAPH_SUCCESS(BuildIOTensors(node->GetOpDescBarePtr()));

  inputs_.emplace_back(compile_info_);
  inputs_.emplace_back(platform_info_);
  inputs_.emplace_back(nullptr);
  inputs_.emplace_back(reinterpret_cast<void *>(deterministic_));
  inputs_.emplace_back(reinterpret_cast<void *>(deterministic_level_));

  return TiledBuild(node, holder);
}

ge::graphStatus DeviceTilingContextBuilder::TiledBuild(const ge::NodePtr &node, TiledKernelContextHolder &holder) {
  // op_type
  const size_t op_type_len = node->GetType().length() + 1UL;  // '\0'
  GE_ASSERT_TRUE(max_mem_size_ >= op_type_len);
  GE_ASSERT_EOK(memcpy_s(host_begin_, max_mem_size_, node->GetTypePtr(), op_type_len));
  holder.dev_op_type_addr_ = dev_begin_;
  TiledPointerOffset(op_type_len, host_begin_, dev_begin_, max_mem_size_);

  // op_name
  const size_t op_name_len = node->GetName().length() + 1UL;  // '\0'
  GE_ASSERT_TRUE(max_mem_size_ >= op_name_len);
  GE_ASSERT_EOK(memcpy_s(host_begin_, max_mem_size_, node->GetNamePtr(), op_name_len));
  holder.dev_op_name_addr_ = dev_begin_;
  TiledPointerOffset(op_name_len, host_begin_, dev_begin_, max_mem_size_);

  // compute node info
  auto host_compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(holder.host_compute_node_info_);
  GE_ASSERT_NOTNULL(host_compute_node_info);
  host_compute_node_info->SetNodeName(ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(holder.dev_op_name_addr_)));
  host_compute_node_info->SetNodeType(ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(holder.dev_op_type_addr_)));

  GE_ASSERT_TRUE(max_mem_size_ >= holder.compute_node_info_size_);
  const uint64_t dev_compute_node_info = dev_begin_;
  GE_ASSERT_EOK(memcpy_s(host_begin_, max_mem_size_, holder.host_compute_node_info_, holder.compute_node_info_size_));
  TiledPointerOffset(holder.compute_node_info_size_, host_begin_, dev_begin_, max_mem_size_);

  size_t context_size = sizeof(KernelRunContext) + sizeof(Chain *) * (inputs_.size() + outputs_.size());
  GE_ASSERT_TRUE(max_mem_size_ >= context_size);
  KernelContext *kernel_context = ge::PtrToPtr<uint8_t, KernelContext>(host_begin_);
  GE_ASSERT_NOTNULL(kernel_context);
  holder.host_context_ = kernel_context;
  holder.dev_context_addr_ = dev_begin_;
  TiledPointerOffset(context_size, host_begin_, dev_begin_, max_mem_size_);

  // kernel run context
  auto kernel_run_context = holder.host_context_->GetContext();
  kernel_run_context->input_size = inputs_.size();
  kernel_run_context->output_size = outputs_.size();
  kernel_run_context->compute_node_info = ge::ValueToPtr(dev_compute_node_info);
  // set output_start with dev_begin_
  kernel_run_context->output_start = reinterpret_cast<AsyncAnyValue **>(
      holder.dev_context_addr_ + ge::PtrToValue(&(kernel_run_context->values[kernel_run_context->input_size])) -
      ge::PtrToValue(holder.host_context_));

  // aligned dev for ts
  const size_t aligned_dev_addr = MemoryAligned(dev_begin_);
  const size_t aligned_offset = static_cast<size_t>(aligned_dev_addr - dev_begin_);
  TiledPointerOffset(aligned_offset, host_begin_, dev_begin_, max_mem_size_);

  // dev
  const size_t aligned_chain_size = MemoryAligned(sizeof(Chain));
  const size_t total_chain_size = aligned_chain_size * (inputs_.size() + outputs_.size());
  GE_ASSERT_TRUE(max_mem_size_ >= total_chain_size);

  // input output chain
  size_t chain_index{0UL};
  for (auto &input : inputs_) {
    Chain *host_chain = ge::PtrToPtr<uint8_t, Chain>(host_begin_);
    GE_ASSERT_NOTNULL(host_chain);
    host_chain->Set(input, nullptr);
    kernel_run_context->values[chain_index] = ge::PtrToPtr<void, AsyncAnyValue>(ge::ValueToPtr(dev_begin_));
    host_begin_ += aligned_chain_size;
    dev_begin_ += aligned_chain_size;
    ++chain_index;
  }
  for (auto &output : outputs_) {
    Chain *host_chain = ge::PtrToPtr<uint8_t, Chain>(host_begin_);
    GE_ASSERT_NOTNULL(host_chain);
    host_chain->Set(output, nullptr);
    kernel_run_context->values[chain_index] = ge::PtrToPtr<void, AsyncAnyValue>(ge::ValueToPtr(dev_begin_));
    holder.output_addrs_.push_back(dev_begin_);
    host_begin_ += aligned_chain_size;
    dev_begin_ += aligned_chain_size;
    ++chain_index;
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
