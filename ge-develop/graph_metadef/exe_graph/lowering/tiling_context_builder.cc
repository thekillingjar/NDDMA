/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/tiling_context_builder.h"

#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "lowering/data_dependent_interpreter.h"
#include "graph/compute_graph.h"
#include "graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/def_types.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace gert {
namespace {
void GetStorageShape(const ge::GeTensorDesc &tensor_desc, gert::StorageShape &storage_shape) {
  const auto &storage_dims = tensor_desc.GetShape().GetDims();
  for (const auto &dim : storage_dims) {
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
  const auto &origin_dims = tensor_desc.GetOriginShape().GetDims();
  for (const auto &dim : origin_dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
  }
}
} // namespace

TilingContextBuilder &TilingContextBuilder::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  return *this;
}
TilingContextBuilder &TilingContextBuilder::PlatformInfo(void *platform_info) {
  platform_info_ = platform_info;
  return *this;
}
TilingContextBuilder &TilingContextBuilder::Deterministic(int32_t deterministic) {
  deterministic_ = deterministic;
  return *this;
}

TilingContextBuilder &TilingContextBuilder::DeterministicLevel(int32_t deterministic_level) {
  deterministic_level_ = deterministic_level;
  return *this;
}

TilingContextBuilder &TilingContextBuilder::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  return *this;
}
TilingContextBuilder &TilingContextBuilder::Workspace(ContinuousVector *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  return *this;
}

TilingContextBuilder &TilingContextBuilder::SetSpaceRegistryV2(const OpImplSpaceRegistryV2Ptr &space_registry, OppImplVersionTag version_tag) {
  if (version_tag >= OppImplVersionTag::kVersionEnd) {
    GELOGE(ge::PARAM_INVALID, "version_tag %d is invalid", static_cast<int>(version_tag));
    return *this;
  }
  space_registries_v2_[static_cast<size_t>(version_tag)] = space_registry;
  return *this;
}

ge::graphStatus TilingContextBuilder::GetDependInputTensorAddr(const ge::Operator &op, const size_t input_idx,
                                                               TensorAddress &address) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  auto depend_tensor = ge::ComGraphMakeUnique<ge::Tensor>();
  depend_ge_tensor_holders_.emplace_back(std::move(depend_tensor));
  GE_ASSERT_NOTNULL(depend_ge_tensor_holders_.back());
  auto input_name = op_desc->GetValidInputNameByIndex(static_cast<uint32_t>(input_idx));
  if (op.GetInputConstData(input_name.c_str(), *(depend_ge_tensor_holders_.back().get())) == ge::GRAPH_SUCCESS) {
    address = depend_ge_tensor_holders_.back()->GetData();
  } else {
    address = nullptr;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingContextBuilder::BuildRtTensor(const ge::GeTensorDesc &tensor_desc,
                                                    ConstTensorAddressPtr address,
                                                    std::unique_ptr<uint8_t[]> &rt_tensor_holder) const {
  gert::StorageShape storage_shape;
  GetStorageShape(tensor_desc, storage_shape);

  rt_tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
  GE_ASSERT_NOTNULL(rt_tensor_holder, "Create context holder inputs failed.");
  auto rt_tensor = ge::PtrToPtr<uint8_t, gert::Tensor>(rt_tensor_holder.get());
  rt_tensor->SetDataType(tensor_desc.GetDataType());
  rt_tensor->MutableStorageShape() = storage_shape.GetStorageShape();
  rt_tensor->MutableOriginShape() = storage_shape.GetOriginShape();
  rt_tensor->MutableFormat().SetStorageFormat(tensor_desc.GetFormat());
  rt_tensor->MutableFormat().SetOriginFormat(tensor_desc.GetOriginFormat());
  (void)rt_tensor->MutableTensorData().SetAddr(address, nullptr);
  rt_tensor->MutableTensorData().SetPlacement(gert::kOnHost);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingContextBuilder::BuildRTInputTensors(const ge::Operator &op) {
  const auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  auto shared_node = const_cast<ge::Node *>(node.get())->shared_from_this();
  std::shared_ptr<DataDependentInterpreter> ddi = nullptr;
  ddi = ge::ComGraphMakeShared<DataDependentInterpreter>(shared_node->GetOpDesc(), space_registries_v2_);

  GE_ASSERT_NOTNULL(ddi);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  size_t valid_input_idx = 0U;
  const auto &all_in_data_anchors = node->GetAllInDataAnchorsPtr();
  for (size_t i = 0U; i < all_in_data_anchors.size(); ++i) {
    GE_ASSERT_NOTNULL(all_in_data_anchors.at(i));
    if (all_in_data_anchors.at(i)->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    TensorAddress address = nullptr;
    bool is_data_dependent = false;
    GE_ASSERT_SUCCESS(ddi->IsDataDependent(static_cast<int32_t>(valid_input_idx), is_data_dependent));
    bool is_tiling_dependent = false;
    if (!is_data_dependent) {
      GE_ASSERT_SUCCESS(ddi->IsTilingInputDataDependent(static_cast<int32_t>(valid_input_idx), is_tiling_dependent));
    }
    GELOGD("Node: %s input: %zu data/tiling depend flag: %d/%d", node->GetNamePtr(), valid_input_idx, is_data_dependent,
           is_tiling_dependent);
    is_data_dependent = is_data_dependent || is_tiling_dependent;
    if (is_data_dependent) {
      GE_ASSERT_GRAPH_SUCCESS(GetDependInputTensorAddr(op, valid_input_idx, address));
    }
    std::unique_ptr<uint8_t[]> tensor_holder;
    const auto &valid_op_desc = op_desc->GetInputDescPtr(i);
    GE_ASSERT_NOTNULL(valid_op_desc);
    GE_ASSERT_GRAPH_SUCCESS(BuildRtTensor(*valid_op_desc, address, tensor_holder));
    rt_tensor_holders_.emplace_back(std::move(tensor_holder));
    ++valid_input_idx;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingContextBuilder::BuildRTOutputShapes(const ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    gert::StorageShape storage_shape;
    GetStorageShape(op_desc->GetOutputDesc(i), storage_shape);
    std::unique_ptr<uint8_t[]> tensor_holder;
    GE_ASSERT_GRAPH_SUCCESS(BuildRtTensor(op_desc->GetOutputDesc(i), nullptr, tensor_holder));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder outputs failed, op[%s]", op_desc->GetNamePtr());
    rt_tensor_holders_.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}
KernelContextHolder TilingContextBuilder::Build(const ge::Operator &op, ge::graphStatus &ret) {
  ret = ge::GRAPH_FAILED;
  KernelContextHolder holder{};
  if (compile_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Please give tiling context builder compile info.");
    return holder;
  }
  if (platform_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Please give tiling context builder platform info.");
    return holder;
  }
  auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  std::vector<void *> context_inputs;
  auto build_ret = BuildRTInputTensors(op);
  if (build_ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Fail to BuildRTInputTensors.");
    return holder;
  }
  build_ret = BuildRTOutputShapes(op);
  if (build_ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Fail to BuildRTOutputShapes.");
    return holder;
  }
  for (const auto &input_holder : rt_tensor_holders_) {
    context_inputs.emplace_back(input_holder.get());
  }
  context_inputs.emplace_back(compile_info_);
  context_inputs.emplace_back(platform_info_);
  context_inputs.emplace_back(nullptr);
  context_inputs.emplace_back(reinterpret_cast<void *>(deterministic_));
  context_inputs.emplace_back(reinterpret_cast<void *>(deterministic_level_));
  return base_builder_.Inputs(context_inputs).Outputs(outputs_).Build(node->GetOpDesc(), ret);
}
// 0-n input tensors
// n-m output shapes
// m + 1 compile info
// m + 2 tiling func
// 其中 n为输入个数总和，m为输入输出个数总和
KernelContextHolder TilingContextBuilder::Build(const ge::Operator &op) {
  ge::Status ret = ge::GRAPH_FAILED;
  return Build(op, ret);
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  return *this;
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::CleanWorkspaceSizes(ContinuousVector *workspace_sizes) {
  worksapce_sizes_ = reinterpret_cast<void *>(workspace_sizes);
  return *this;
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::CleanOutputSizes(const std::vector<int64_t> &output_sizes) {
  clean_output_sizes_ = output_sizes;
  return *this;
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  return *this;
}
AtomicTilingContextBuilder &AtomicTilingContextBuilder::Workspace(ContinuousVector *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  return *this;
}
KernelContextHolder AtomicTilingContextBuilder::Build(const ge::Operator &op, ge::graphStatus &ret) {
  ret = ge::GRAPH_FAILED;
  KernelContextHolder holder{};
  if (compile_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Please give tiling context builder compile info.");
    return holder;
  }
  std::vector<void *> context_inputs;
  context_inputs.emplace_back(worksapce_sizes_);
  for (const int64_t out_size : clean_output_sizes_) {
    context_inputs.emplace_back(reinterpret_cast<void *>(out_size));
  }
  context_inputs.emplace_back(compile_info_);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  return base_builder_.Inputs(context_inputs).Outputs(outputs_).Build(op_desc, ret);
}
// 0 atomic op workspace
// 1~n  待清零的output size
// n+1  compile info
// n+2  atomic tiling func
// 其中 n 为待清零的输出个数，
KernelContextHolder AtomicTilingContextBuilder::Build(const ge::Operator &op) {
  ge::graphStatus ret = ge::GRAPH_FAILED;
  return Build(op, ret);
}
}  // namespace gert
