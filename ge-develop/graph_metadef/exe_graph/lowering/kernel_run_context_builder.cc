/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/def_types.h"

namespace gert {
KernelContextHolder KernelRunContextBuilder::Build(const ge::OpDescPtr &op_desc) {
  ge::Status ret = ge::GRAPH_FAILED;
  return Build(op_desc, ret);
}

KernelContextHolder KernelRunContextBuilder::Build(const ge::OpDescPtr &op_desc, ge::graphStatus &ret) {
  ret = ge::GRAPH_FAILED;
  KernelContextHolder holder;
  size_t size = sizeof(KernelRunContext) + sizeof(Chain *) * (inputs_.size() + outputs_.size());
  holder.context_holder_ = ge::ComGraphMakeUnique<uint8_t[]>(size);
  if (holder.context_holder_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Create context holder failed.");
    return holder;
  }
  const auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
    .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()));
  OpImplRegisterV2::PrivateAttrList private_attrs;
  if (space_registry != nullptr) {
    const auto op_impl_func_v2 = space_registry->GetOpImpl(op_desc->GetType().c_str());
    if (op_impl_func_v2 != nullptr) {
      private_attrs = op_impl_func_v2->private_attrs;
    }
  }
  GELOGD("Default space registry is %s. Op:%s(%s) has %zu private attrs.",
         space_registry == nullptr ? "nullptr" : "not nullptr", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
         private_attrs.size());
  size_t extend_info_size;
  holder.compute_node_extend_holder_ =
      bg::CreateComputeNodeInfo(MakeNode(op_desc), holder.buffer_pool_, private_attrs, extend_info_size);

  if (holder.compute_node_extend_holder_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED,
           "Failed to create compute node info for node %s", op_desc->GetName().c_str());
    return holder;
  }
  auto compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(holder.compute_node_extend_holder_.get());
  compute_node_info->SetNodeName(
      holder.buffer_pool_.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeName())));
  compute_node_info->SetNodeType(
      holder.buffer_pool_.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeType())));
  holder.context_ = ge::PtrToPtr<uint8_t, KernelContext>(holder.context_holder_.get());
  auto kernel_run_context = holder.context_->GetContext();
  kernel_run_context->input_size = inputs_.size();
  kernel_run_context->output_size = outputs_.size();
  kernel_run_context->compute_node_info = compute_node_info;
  kernel_run_context->output_start = &(kernel_run_context->values[kernel_run_context->input_size]);
  holder.value_holder_.resize(inputs_.size() + outputs_.size());
  for (size_t i = 0UL; i < holder.value_holder_.size(); ++i) {
    kernel_run_context->values[i] = ge::PtrToPtr<Chain, AsyncAnyValue>(&holder.value_holder_[i]);
  }
  for (size_t i = 0UL; i < inputs_.size(); ++i) {
    holder.value_holder_[i].Set(inputs_[i].first, inputs_[i].second);
  }
  for (size_t i = 0UL; i < outputs_.size(); ++i) {
    holder.value_holder_[inputs_.size() + i].Set(outputs_[i].first, outputs_[i].second);
  }
  ret = ge::GRAPH_SUCCESS;
  return holder;
}

ge::NodePtr KernelRunContextBuilder::MakeNode(const ge::OpDescPtr &op_desc) {
  const auto node_id = op_desc->GetId();
  graph_ = std::make_shared<ge::ComputeGraph>("tmp");
  auto fake_node = graph_->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(fake_node, return nullptr);
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const auto input_desc = op_desc->GetInputDesc(i);
    if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
      GELOGD("Node: %s, input: %zu, is invalid, skip add edge.", op_desc->GetNamePtr(), i);
      continue;
    }
    auto op_data = ge::OpDescBuilder(std::to_string(i), "Data").AddInput("x").AddOutput("y").Build();
    auto data_node = graph_->AddNode(op_data);
    if (data_node == nullptr) {
      return nullptr;
    }
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), fake_node->GetInDataAnchor(i));
  }
  // AddNode operation may change node id to 0, which need to be recovered
  op_desc->SetId(node_id);
  return fake_node;
}
}  // namespace gert
