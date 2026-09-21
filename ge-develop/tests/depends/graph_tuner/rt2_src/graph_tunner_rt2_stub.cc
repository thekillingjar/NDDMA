/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_tunner_rt2_stub.h"
#include "framework/common/ge_types.h"
#include "ge/ge_api_error_codes.h"
#include "graph/compute_graph.h"
#include "common/sgt_slice_type.h"
#include "register/kernel_registry_impl.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "framework/common/debug/ge_log.h"

namespace tune {
ge::graphStatus FFTSNodeThreadRT2(gert::KernelContext *context) {
  (void) context;
  GELOGD("FFTSNodeThreadRT2");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildNodeThreadOutputs(const ge::FastNode *node, gert::KernelContext *context) {
  (void) node;
  auto extend_context = reinterpret_cast<gert::ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  auto in_num = compute_node_info->GetInputsNum();
  auto out_num = compute_node_info->GetOutputsNum();
  GELOGD("Node thread in_num:%zu, out_num:%zu.", in_num, out_num);
  gert::Shape shape = {1, 2, 3};
  for (size_t i = 0; i < 4; ++i) {
    auto in_vec_holder = gert::ContinuousVector::Create<gert::Shape>(in_num);
    auto in_vec_p = reinterpret_cast<gert::ContinuousVector *>(in_vec_holder.get());
    auto shape_vec = reinterpret_cast<gert::Shape *>(const_cast<void*>(in_vec_p->GetData()));
    in_vec_p->SetSize(in_num);
    for (size_t j = 0; j < in_num; ++j) {
      shape_vec[j] = shape;
    }
    auto output = context->GetOutput(i);
    output->SetWithDefaultDeleter<uint8_t[]>(in_vec_holder.release());
  }
  for (size_t i = 0; i < 4; ++i) {
    auto in_vec_holder = gert::ContinuousVector::Create<gert::Shape>(out_num);
    auto in_vec_p = reinterpret_cast<gert::ContinuousVector *>(in_vec_holder.get());
    auto shape_vec = reinterpret_cast<gert::Shape *>(const_cast<void*>(in_vec_p->GetData()));
    in_vec_p->SetSize(out_num);
    for (size_t j = 0; j < out_num; ++j) {
      shape_vec[j] = shape;
    }
    auto output = context->GetOutput(4 + i);
    output->SetWithDefaultDeleter<uint8_t[]>(in_vec_holder.release());
  }
  GELOGD("BuildNodeThreadOutputs in num:%u, out num:%u", in_num, out_num);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FFTSNodeThread).RunFunc(FFTSNodeThreadRT2).OutputsCreator(BuildNodeThreadOutputs);


ge::graphStatus FFTSGraphPreThreadRT2(gert::KernelContext *context) {
  auto thread_dim = context->GetOutputPointer<uint32_t>(0);
  auto window_size = context->GetOutputPointer<uint32_t>(1);
  *thread_dim = 2;
  *window_size = 4;
  GELOGD("FFTSGraphPreThreadRT2");
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FFTSGraphPreThread).RunFunc(FFTSGraphPreThreadRT2);

using namespace gert;

constexpr char const *kParamK = "param_k";
constexpr char const *kParamB = "param_b";
constexpr char const *kUnknownAxisIndex = "unknown_axis_index";
ge::graphStatus FFTSGraphPreThreadV2(const ge::ComputeGraphPtr sub_graph,
    const std::vector<bg::ValueHolderPtr> &input_shapes, std::vector<bg::ValueHolderPtr> &output) {
  GELOGD("FFTSGraphPreThreadV2 begin.");
  ge::NodePtr data_node = nullptr;
  int64_t param_k = 0;
  int64_t param_b = 0;
  std::vector<int32_t> unknown_axis;
  for (const auto &node : sub_graph->GetInputNodes()) {
    data_node = node;
    const auto out_anchor = node->GetOutDataAnchor(0);
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      const auto next_node = peer_in_anchor->GetOwnerNode();
      const auto &next_desc = next_node->GetOpDesc();
      if (ge::AttrUtils::GetListInt(next_desc, kUnknownAxisIndex, unknown_axis)) {
        (void)ge::AttrUtils::GetInt(next_desc, kParamK, param_k);
        (void)ge::AttrUtils::GetInt(next_desc, kParamB, param_b);
        break;
      }
    }
  }
  if (data_node == nullptr) {
    GELOGE(ge::FAILED, "Sub graph[%s] not get pre thread param.", sub_graph->GetName().c_str());
    return ge::GRAPH_FAILED;
  }
  std::vector<bg::ValueHolderPtr> inputs;
  auto ids_holder = bg::CreateContVecHolder(unknown_axis);
  inputs.emplace_back(ids_holder);

  uint32_t index = 0U;
  (void)ge::AttrUtils::GetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, index);
  inputs.emplace_back(input_shapes[index]);

  auto param_k_holder = bg::ValueHolder::CreateConst(&param_k, sizeof(param_k));
  auto param_b_holder = bg::ValueHolder::CreateConst(&param_b, sizeof(param_b));
  inputs.emplace_back(param_k_holder);
  inputs.emplace_back(param_b_holder);
  output = bg::ValueHolder::CreateDataOutput("FFTSGraphPreThread", inputs, 2);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFTSGraphPreThreadV2New(const ge::ComputeGraphPtr &sub_graph,
    const std::vector<bg::ValueHolderPtr> &input_shapes, const std::vector<bg::ValueHolderPtr> &input_addrs,
    std::vector<bg::ValueHolderPtr> &output) {
  return FFTSGraphPreThreadV2(sub_graph, input_shapes, output);
}

ge::graphStatus FFTSNodeThreadV2(const ge::NodePtr &node, const std::vector<bg::ValueHolderPtr> &input_shapes,
    const std::vector<bg::ValueHolderPtr> &output_shapes, const bg::ValueHolderPtr thread_dim,
    std::vector<bg::ValueHolderPtr> &output) {
  GELOGD("FFTSNodeThreadV2 BEGIN.");
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(thread_dim);
  // input
  std::vector<uint32_t> input_tensor_indexes;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::kInputTensorIndexs, input_tensor_indexes);
  auto ids_holder = bg::CreateContVecHolder(input_tensor_indexes);
  inputs.emplace_back(ids_holder);
  uint32_t input_num = static_cast<uint32_t>(input_shapes.size());
  inputs.emplace_back(bg::ValueHolder::CreateConst(&input_num, sizeof(input_num)));
  inputs.insert(inputs.cend(), input_shapes.cbegin(), input_shapes.cend());
  // output
  std::vector<uint32_t> output_tensor_indexes;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::kOutputTensorIndexs, output_tensor_indexes);
  ids_holder = bg::CreateContVecHolder(output_tensor_indexes);
  inputs.emplace_back(ids_holder);
  uint32_t output_num = output_shapes.size();
  inputs.emplace_back(bg::ValueHolder::CreateConst(&output_num, sizeof(output_num)));
  inputs.insert(inputs.cend(), output_shapes.cbegin(), output_shapes.cend());
  output = bg::ValueHolder::CreateDataOutput("FFTSNodeThread", inputs, 8);
  return ge::GRAPH_SUCCESS;
}

} // namespace tune
