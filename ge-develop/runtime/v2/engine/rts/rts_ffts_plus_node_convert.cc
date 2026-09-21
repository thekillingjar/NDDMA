/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "engine/node_converter_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/value_holder_generator.h"
#include "framework/common/ge_types.h"
#include "common/hyper_status.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "framework/common/types.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"

namespace gert {
const size_t kThreadOutputSize = 2UL;
bg::DevMemValueHolderPtr AllocOutputMem(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                        bg::ValueHolderPtr &output_shape, bg::ValueHolderPtr &l2_alloc_size,
                                        uint32_t &output_mem_type) {
  auto *subgraph_outputs_idx = node->GetOpDesc()->GetExtAttr<std::set<int32_t>>("_ffts_subgraph_outputs_index");
  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  if ((subgraph_outputs_idx == nullptr) || (subgraph_outputs_idx->count(0UL) == 0UL)) {
    // use l2_allocator
    output_addrs = bg::AllocateFftsMems(lower_input.ffts_mem_allocator, node->GetOpDesc()->GetStreamId(),
                                        {l2_alloc_size});
    output_mem_type = 1U;
  } else {
    // use l1_allocator
    auto output_size = CalcOutTensorSize(node, 0, output_shape);
    output_addrs = bg::AllocMemories(kOnDeviceHbm, {output_size}, *(lower_input.global_data),
                                     node->GetOpDesc()->GetStreamId());
    output_mem_type = 0U;
  }
  std::vector<uint32_t> mem_pool_types{output_mem_type};
  (void)node->GetOpDesc()->SetExtAttr("_ffts_memory_pool_type", mem_pool_types);
  GE_ASSERT_TRUE(!output_addrs.empty());
  return output_addrs[0];
}

std::vector<bg::ValueHolderPtr> CalcThreadOutputMemSize(const ge::NodePtr &node, const FFTSLowerInput &input,
                                                        const std::vector<bg::ValueHolderPtr> &thread_ret,
                                                        bool slice_flag) {
  auto data_type = node->GetOpDesc()->GetOutputDesc(0U).GetDataType();
  std::vector<bg::ValueHolderPtr> inputs{thread_ret[static_cast<size_t>(kernel::ThreadOutKey::OUT_SHAPES)],
                                         thread_ret[static_cast<size_t>(kernel::ThreadOutKey::LAST_OUT_SHAPES)],
                                         bg::ValueHolder::CreateConst(&data_type, sizeof(data_type)),
                                         bg::ValueHolder::CreateConst(&slice_flag, sizeof(slice_flag)),
                                         input.thread_dim};
  return bg::ValueHolder::CreateDataOutput("CalcFftsThreadDataLen", inputs, kThreadOutputSize);
}

LowerResult LoweringIdentityLikeNode(const ge::NodePtr &node, const FFTSLowerInput &input) {
  auto ret = CheckFFTSLowerInput(input);
  if (!ret.IsSuccess()) {
    GELOGE(ge::FAILED, "[LoweringIdentityLikeNode]Input is invalid.");
    return {ret, {}, {}, {}};
  }
  GELOGD("Lowering rts node [%s] type [%s].", node->GetName().c_str(), node->GetType().c_str());
  if (input.input_addrs.size() != 1UL || input.mem_pool_types.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op] Input addr size is invalid.");
    return {HyperStatus::ErrorStatus("Input addr size is invalid"), {}, {}, {}};
  }

  auto out_shapes = bg::InferStorageShape(node, input.input_shapes, *(input.global_data));
  LOWER_REQUIRE(out_shapes.size() == input.input_addrs.size(), "Output shape size [%zu] is invalid", out_shapes.size());

  std::vector<bg::ValueHolderPtr> thread_ret;
  (void)input.ffts_thread_fun(node, input.input_shapes, out_shapes, input.thread_dim, thread_ret);
  CONVERTER_CHECK_HOLDERS_ALL_OK(thread_ret, static_cast<size_t>(kernel::ThreadOutKey::kNUM));

  // slice_flag
  std::vector<uint32_t> output_tensor_idxes;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::kOutputTensorIndexs, output_tensor_idxes);
  bool slice_flag = !output_tensor_idxes.empty();

  auto thread_len_res = CalcThreadOutputMemSize(node, input, thread_ret, slice_flag);
  CONVERTER_CHECK_HOLDERS_ALL_OK(thread_len_res, kThreadOutputSize);
  uint32_t output_mem_type{0U};
  auto output_addr = AllocOutputMem(node, input, out_shapes[0UL], thread_ret[0UL], output_mem_type);
  // get context
  std::vector<uint32_t> ctx_id_vec;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "_context_id_list", ctx_id_vec);
  auto ctx_holder = bg::CreateContVecHolder(ctx_id_vec);

  // Update sdma ctx
  std::vector<bg::ValueHolderPtr> inputs{ctx_holder, input.thread_dim, input.window_size, thread_len_res[1UL]};
  inputs.emplace_back(bg::ValueHolder::CreateConst(&input.mem_pool_types[0UL], sizeof(input.mem_pool_types[0UL])));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&output_mem_type, sizeof(output_mem_type)));
  inputs.emplace_back(input.input_addrs[0UL]);
  inputs.emplace_back(output_addr);

  auto sdma_holder = bg::ValueHolder::CreateSingleDataOutput("SdmaUpdateContext", inputs);
  GE_ASSERT_NOTNULL(sdma_holder);
  sdma_holder->RefFrom(input.task_info);

  std::vector<bg::ValueHolderPtr> alloc_holder_vec{output_addr};
  LOWER_REQUIRE(node->GetOpDesc()->SetExtAttr(kFftsAllocVecHolder, alloc_holder_vec), "Set free addr failed.");
  return {HyperStatus::Success(), {sdma_holder}, out_shapes, {output_addr}};
}

FFTS_REGISTER_NODE_CONVERTER_PLACEMENT("Identity", kOnDeviceHbm, LoweringIdentityLikeNode);
FFTS_REGISTER_NODE_CONVERTER_PLACEMENT("MemcpyAsync", kOnDeviceHbm, LoweringIdentityLikeNode);
}  // namespace gert
