/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ffts_node_converter.h"

#include "aicpu_callback.h"
#include "common/hyper_status.h"
#include "common/math/math_util.h"
#include "common/sgt_slice_type.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "engine/aicpu/kernel/ffts_plus/aicpu_ffts_args.h"
#include "engine/aicpu/kernel/ffts_plus/aicpu_update_kernel.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "framework/common/ge_types.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_rt_session.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/value_holder_generator.h"
#include "kernel/memory/ffts_mem_allocator.h"

using namespace ge;
namespace gert {
namespace {
const std::string kEngineNameAicpuFfts = "ffts_plus_aicpu_ascend";
const std::string kEngineNameAicpuTfFfts = "ffts_plus_aicpu_tf";
constexpr const ge::char_t *kFtfsMemoryPoolType = "_ffts_memory_pool_type";
constexpr const ge::char_t *kFtfsSubGraphOutputsIndex = "_ffts_subgraph_outputs_index";

bg::ValueHolderPtr CalAutoThreadParam(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                      std::vector<bg::ValueHolderPtr> &thread_ret) {
  std::vector<uint32_t> input_tensor_indexes;
  (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), ge::kInputTensorIndexs, input_tensor_indexes);
  auto in_idx_holder = bg::CreateContVecHolder(input_tensor_indexes);

  std::vector<uint32_t> output_tensor_indexes;
  (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), ge::kOutputTensorIndexs, output_tensor_indexes);
  auto out_idx_holder = bg::CreateContVecHolder(output_tensor_indexes);

  std::vector<bg::ValueHolderPtr> inputs;
  // thread_dim
  inputs.emplace_back(lower_input.thread_dim);
  inputs.emplace_back(in_idx_holder);
  inputs.emplace_back(out_idx_holder);
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kNotLastInShapes)]);
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kNotLastOutShapes)]);
  return bg::ValueHolder::CreateSingleDataOutput("AICpuGetAutoThreadParam", inputs);
}


}  // namespace

bg::ValueHolderPtr UpdateAicpuContext(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                      bg::ValueHolderPtr flush_data) {
  std::vector<uint32_t> ctx_id_vec;
  (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), "_context_id_list", ctx_id_vec);
  auto ctx_holder = bg::CreateContVecHolder(ctx_id_vec);
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(flush_data);
  inputs.emplace_back(ctx_holder);
  // thread_dim
  inputs.emplace_back(lower_input.thread_dim);
  return bg::ValueHolder::CreateSingleDataOutput("AICpuUpdateContext", inputs);
}

bg::ValueHolderPtr AicpuCalcFftsOutputAllocMem(const std::vector<bg::ValueHolderPtr> &thread_ret, const size_t index) {
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(bg::ValueHolder::CreateConst(&index, sizeof(index)));
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kNotLastOutShapes)]);
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kLastOutShapes)]);
  // The maximum sub thread size is used to alloc secondary memory
  return bg::ValueHolder::CreateSingleDataOutput("AicpuCalcOutputMaxThreadSize", inputs);
}

std::vector<bg::ValueHolderPtr> AicpuCalcFftsOutputAllocMemVec(const ge::NodePtr &node,
                                                               const std::vector<bg::ValueHolderPtr> &thread_ret) {
  size_t output_num = static_cast<size_t>(node->GetOpDescBarePtr()->GetAllOutputsDescSize());
  if (output_num == 0) {
    GELOGE(ge::PARAM_INVALID, "Aicpu calculate output block size vector failed result of output num is zero.");
    return {};
  }

  std::vector<bg::ValueHolderPtr> outputs;
  for (size_t i = 0; i < output_num; ++i) {
    outputs.emplace_back(AicpuCalcFftsOutputAllocMem(thread_ret, i));
  }
  return outputs;
}

bool IsGetRefNodeOutAddr(const RefNodeInfo &ref_node_info, const FFTSLowerInput &lower_input,
                         std::vector<bg::DevMemValueHolderPtr> &output_addrs,
                         std::vector<bg::DevMemValueHolderPtr> &level_one_mem_addrs,
                         std::vector<uint32_t> &mem_pool_types) {
  GE_ASSERT_EQ(lower_input.mem_pool_types.size(), lower_input.input_addrs.size());
  auto iter = ref_node_info.ref_map.find(ref_node_info.out_index);
  if (iter == ref_node_info.ref_map.end()) {
    GELOGD("Node %s if not ref node", ref_node_info.node->GetName().c_str());
    return false;
  }
  auto ref_input_index = iter->second;
  if (ref_input_index >= lower_input.input_addrs.size()) {
    GELOGE(ge::FAILED, "Node %s output %u ref from input %zu exceed input addrs num %zu",
           ref_node_info.node->GetName().c_str(), ref_node_info.out_index, ref_input_index,
           lower_input.input_addrs.size());
    return false;
  }
  output_addrs.emplace_back(lower_input.input_addrs[ref_input_index]);
  // 0 indicates the first memory pool type, and 1 indicates the secondary memory pool type
  auto mem_pool_type = lower_input.mem_pool_types[ref_input_index];
  bool is_secondary_memory_pool = static_cast<bool>(mem_pool_type);
  if (!is_secondary_memory_pool) {
    level_one_mem_addrs.emplace_back(lower_input.input_addrs[ref_input_index]);
  }
  mem_pool_types.emplace_back(mem_pool_type);
  GELOGD("Node %s Output index[%u] ref input index[%zu]", ref_node_info.node->GetName().c_str(),
         ref_node_info.out_index, ref_input_index);
  return true;
}

std::vector<bg::DevMemValueHolderPtr> AicpuAllocOutputMem(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                                          const std::vector<bg::ValueHolderPtr> &output_sizes,
                                                          const std::vector<bg::ValueHolderPtr> &output_shapes,
                                                          std::vector<bg::DevMemValueHolderPtr> &level_one_mem_addrs) {
  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  std::vector<uint32_t> mem_pool_types;
  std::map<size_t, size_t> ref_map;
  if (bg::GetNodeRefMap(node, ref_map) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Node %s get ref map failed", node->GetName().c_str());
    return {};
  }

  auto output_num = node->GetAllOutDataAnchorsSize();
  GE_ASSERT_EQ(output_sizes.size(), output_shapes.size());
  GE_ASSERT_EQ(output_shapes.size(), output_num);
  auto *sub_graph_netoutput_index = node->GetOpDescBarePtr()->GetExtAttr<std::set<int32_t>>(kFtfsSubGraphOutputsIndex);
  level_one_mem_addrs.reserve(output_num);
  output_addrs.reserve(output_num);
  mem_pool_types.reserve(output_num);
  RefNodeInfo ref_node_info;
  ref_node_info.ref_map = ref_map;
  ref_node_info.node = node;
  const int64_t stream_id = node->GetOpDescBarePtr()->GetStreamId();

  if (sub_graph_netoutput_index != nullptr && !sub_graph_netoutput_index->empty()) {
    for (uint32_t out_index = 0U; out_index < output_num; ++out_index) {
      ref_node_info.out_index = out_index;
      if (IsGetRefNodeOutAddr(ref_node_info, lower_input, output_addrs, level_one_mem_addrs, mem_pool_types)) {
        continue;
      }
      bg::DevMemValueHolderPtr output_addr;
      if (sub_graph_netoutput_index->find(out_index) == sub_graph_netoutput_index->end()) {
        // not found netoutput node, use secondary memory pool to alloc memory
        output_addr = bg::AllocateFftsMems(lower_input.ffts_mem_allocator, stream_id, {output_sizes[out_index]})[0];
        mem_pool_types.emplace_back(static_cast<uint32_t>(MemPoolType::kSecondaryMemPool));
      } else {
        // found netoutput node, use first memory pool to alloc memory
        auto output_size = CalcOutTensorSize(node, static_cast<int32_t>(out_index), output_shapes[out_index]);
        output_addr = bg::AllocMemories(kOnDeviceHbm, {output_size}, *(lower_input.global_data), stream_id)[0];
        level_one_mem_addrs.emplace_back(output_addr);
        mem_pool_types.emplace_back(static_cast<uint32_t>(MemPoolType::kFirstMemPool));
      }
      output_addrs.emplace_back(std::move(output_addr));
    }
  } else {
    // completely not found netoutput node, use secondary memory pool to alloc memory
    for (uint32_t out_index = 0U; out_index < output_num; ++out_index) {
      ref_node_info.out_index = out_index;
      if (IsGetRefNodeOutAddr(ref_node_info, lower_input, output_addrs, level_one_mem_addrs, mem_pool_types)) {
        continue;
      }
      auto output_addr = bg::AllocateFftsMems(lower_input.ffts_mem_allocator, stream_id, {output_sizes[out_index]})[0];
      mem_pool_types.emplace_back(1);
      output_addrs.emplace_back(std::move(output_addr));
    }
  }
  node->GetOpDescBarePtr()->SetExtAttr(kFtfsMemoryPoolType, mem_pool_types);
  return output_addrs;
}

ge::graphStatus InitAicpuCtxUserData(const domi::FftsPlusTaskDef &task_def, const ge::NodePtr &node,
                                     const FFTSLowerInput &lower_input,
                                     std::vector<bg::ValueHolderPtr> &aicpu_free_holder,
                                     bg::ValueHolderPtr update_task_info) {
  // initialize aicpu ctx user data of so_name and kernel_name
  GELOGD("Begin InitAicpuCtxUserData");
  std::vector<uint32_t> ctx_id_vec;
  (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), "_context_id_list", ctx_id_vec);
  auto ctx_holder = bg::CreateContVecHolder(ctx_id_vec);
  uint32_t ctx_num = static_cast<uint32_t>(task_def.ffts_plus_ctx_size());

  for (size_t i = 0; i < ctx_id_vec.size(); ++i) {
    GE_ASSERT_TRUE(
        ctx_id_vec[i] < ctx_num,
        "Out-of-bounds index[%u] access to task_def.ffts_plus_ctx, the size of task_def.ffts_plus_ctx is:[%u].",
        ctx_id_vec[i], ctx_num);
    const domi::FftsPlusCtxDef &ffts_plus_task_def = task_def.ffts_plus_ctx(ctx_id_vec[i]);
    const domi::FftsPlusAicpuCtxDef &ctx_def = ffts_plus_task_def.aicpu_ctx();
    const auto &kernel = ctx_def.kernel();
    const auto &so_name = kernel.so_name();
    const size_t so_name_len = so_name.size() + 1U;  // Need copy terminate byte '\0' to device.
    auto so_name_len_holder = bg::ValueHolder::CreateConst(&so_name_len, sizeof(so_name_len));

    const auto &kernel_name = kernel.kernel_name();
    const size_t kernel_name_len = kernel_name.size() + 1U;  // Need copy terminate byte '\0' to device.
    auto kernel_name_len_holder = bg::ValueHolder::CreateConst(&kernel_name_len, sizeof(kernel_name_len));

    const int64_t stream_id = node->GetOpDescBarePtr()->GetStreamId();
    auto so_name_dev = bg::AllocMem(kOnDeviceHbm, so_name_len_holder, *(lower_input.global_data), stream_id);
    auto kernel_name_dev = bg::AllocMem(kOnDeviceHbm, kernel_name_len_holder, *(lower_input.global_data), stream_id);
    GELOGD("Ctx:%u so_name:%s, length:%zu, kernel_name:%s, length:%zu", ctx_id_vec[i], so_name.c_str(), so_name_len,
           kernel_name.c_str(), kernel_name_len);
    aicpu_free_holder.emplace_back(so_name_dev);
    aicpu_free_holder.emplace_back(kernel_name_dev);
    std::vector<bg::ValueHolderPtr> inputs;
    inputs.emplace_back(ctx_holder);
    inputs.emplace_back(bg::ValueHolder::CreateConst(&i, sizeof(i)));
    inputs.emplace_back(so_name_dev);
    inputs.emplace_back(bg::ValueHolder::CreateConst(&so_name_len, sizeof(so_name_len)));
    inputs.emplace_back(bg::ValueHolder::CreateConst(so_name.c_str(), so_name_len, true));
    inputs.emplace_back(kernel_name_dev);
    inputs.emplace_back(bg::ValueHolder::CreateConst(&kernel_name_len, sizeof(kernel_name_len)));
    inputs.emplace_back(bg::ValueHolder::CreateConst(kernel_name.c_str(), kernel_name_len, true));
    inputs.emplace_back(lower_input.global_data->GetStream());
    auto res_holder = bg::ValueHolder::CreateSingleDataOutput("FFTSInitAicpuCtxUserData", inputs);
    // Ensure InitAicpuCtxUserData before Fe H2D Copy
    bg::ValueHolder::AddDependency(res_holder, update_task_info);
    if (res_holder != nullptr) {
      res_holder->RefFrom(lower_input.task_info);
    }
  }
  return ge::GRAPH_SUCCESS;
}

inline bool ConstructMemTypeInput(const ge::NodePtr &node, std::vector<bg::ValueHolderPtr> &inputs,
                                  const FFTSLowerInput &lower_input) {
  GELOGI("Begin ConstructMemTypeInput");
  auto in_mem_type_holder = bg::CreateContVecHolder(lower_input.mem_pool_types);
  if (in_mem_type_holder == nullptr) {
    GELOGE(ge::FAILED, "Node[%s] create input mem type holder failed.", node->GetName().c_str());
    return false;
  }
  inputs.emplace_back(in_mem_type_holder);

  const auto *mem_pool_types = node->GetOpDescBarePtr()->GetExtAttr<std::vector<uint32_t>>(kFtfsMemoryPoolType);
  if (mem_pool_types == nullptr) {
    GELOGE(ge::FAILED, "Node[%s] do not have mem pool type attr.", node->GetName().c_str());
    return false;
  }
  auto out_mem_type_holder = bg::CreateContVecHolder(*mem_pool_types);
  if (out_mem_type_holder == nullptr) {
    GELOGE(ge::FAILED, "Node[%s] create output mem type holder failed.", node->GetName().c_str());
    return false;
  }
  inputs.emplace_back(out_mem_type_holder);
  return true;
}

bool ConstructAicpuArgsInput(const ge::NodePtr &node, const FFTSLowerInput &lower_input, const ThreadInfo thread_infos,
                             const std::vector<bg::DevMemValueHolderPtr> &output_addrs,
                             std::vector<bg::ValueHolderPtr> &inputs) {
  GELOGI("Begin ConstructAicpuArgsInput");
  const auto &session_id = bg::GetSessionId(*lower_input.global_data);
  const std::string *ext_info = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), "_aicpu_ffts_ext_info");
  if ((ext_info == nullptr) || (ext_info->empty())) {
    GELOGE(ge::PARAM_INVALID, "Aicpu update args failed result of empty ext_info.");
    return false;
  }
  const size_t ext_size = ext_info->size();
  const std::string *args_info = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), "_aicpu_ffts_args");
  if ((args_info == nullptr) || (args_info->empty())) {
    GELOGE(ge::PARAM_INVALID, "Aicpu update args failed result of empty args_info.");
    return false;
  }
  const size_t arg_size = args_info->size();
  const auto &node_name = node->GetName();
  int32_t unknown_shape_type_val = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  inputs.emplace_back(session_id);
  inputs.emplace_back(bg::ValueHolder::CreateConst(ext_info->c_str(), ext_info->size(), true));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&ext_size, sizeof(ext_size)));
  inputs.emplace_back(bg::ValueHolder::CreateConst(args_info->c_str(), args_info->size(), true));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&arg_size, sizeof(arg_size)));
  inputs.emplace_back(bg::ValueHolder::CreateConst(node_name.c_str(), node_name.size() + 1, true));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&unknown_shape_type_val, sizeof(unknown_shape_type_val)));
  inputs.emplace_back(thread_infos.thread_para);
  auto thread_ret = thread_infos.sgt_thread_info;
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kNotLastInShapes)]);
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kNotLastOutShapes)]);
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kLastInShapes)]);
  inputs.emplace_back(thread_ret[static_cast<size_t>(SliceShapeIndex::kLastOutShapes)]);

  std::vector<uint32_t> ctx_id_vec;
  (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), "_context_id_list", ctx_id_vec);
  auto ctx_holder = bg::CreateContVecHolder(ctx_id_vec);
  inputs.emplace_back(ctx_holder);

  inputs.insert(inputs.cend(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());
  return true;
}

std::vector<bg::ValueHolderPtr> AicpuCCUpdateArgs(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                                  const ThreadInfo thread_infos,
                                                  const std::vector<bg::DevMemValueHolderPtr> &output_addrs) {
  GELOGI("Begin AicpuCCUpdateArgs");
  std::vector<bg::ValueHolderPtr> args_inputs;
  if (!ConstructMemTypeInput(node, args_inputs, lower_input)) {
    GELOGE(ge::PARAM_INVALID, "ConstructMemTypeInput failed.");
    return {};
  }

  if (!ConstructAicpuArgsInput(node, lower_input, thread_infos, output_addrs, args_inputs)) {
    GELOGE(ge::PARAM_INVALID, "ConstructArgsInput failed.");
    return {};
  }
  return bg::ValueHolder::CreateDataOutput("FFTSUpdateAICpuCCArgs", args_inputs,
                                           static_cast<size_t>(FFTSAicpuArgsOutKey::kNum));
}

std::vector<bg::ValueHolderPtr> AicpuTfUpdateArgs(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                                  const ThreadInfo thread_infos,
                                                  const std::vector<bg::DevMemValueHolderPtr> &output_addrs) {
  const auto &session_id = bg::GetSessionId(*lower_input.global_data);
  const auto &step_id = GetStepId(*lower_input.global_data);
  bg::ValueHolder::CreateSingleDataOutput("EnsureCreateTfSession", {session_id});

  std::vector<bg::ValueHolderPtr> args_inputs;
  if (!ConstructMemTypeInput(node, args_inputs, lower_input)) {
    GELOGE(ge::PARAM_INVALID, "ConstructMemTypeInput failed.");
    return {};
  }

  if (!ConstructAicpuArgsInput(node, lower_input, thread_infos, output_addrs, args_inputs)) {
    GELOGE(ge::PARAM_INVALID, "ConstructArgsInput failed.");
    return {};
  }

  args_inputs.emplace_back(step_id);
  return bg::ValueHolder::CreateDataOutput("FFTSUpdateAICpuTfArgs", args_inputs,
                                           static_cast<size_t>(FFTSAicpuArgsOutKey::kNum));
}

ge::graphStatus CalcAICpuCommonArgsMem(const ge::NodePtr &node, size_t &ext_total_byte) {
  const std::string *ext_info = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), "_aicpu_ffts_ext_info");
  if (ext_info == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Aicpu calc failed result of null ext_info.");
    return ge::GRAPH_FAILED;
  }
  const size_t ext_size = ext_info->size();
  if (ext_size == 0UL) {
    GELOGE(ge::PARAM_INVALID, "Aicpu calc failed result of empty ext_info.");
    return ge::GRAPH_FAILED;
  }
  FMK_SIZET_MULCHECK(ge::MemSizeAlign(ext_size * MAX_THREAD_DIM), 1);
  ext_total_byte = ge::MemSizeAlign(ext_size * MAX_THREAD_DIM);
  GELOGI("ext_size is %zu, ext_total_byte is %zu.", ext_size, ext_total_byte);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcAICpuCCArgsMem(const ge::NodePtr &node, const LoweringGlobalData *global_data, size_t &total_size,
                                   size_t &pre_data_size, std::unique_ptr<uint8_t[]> &pre_data_ptr) {
  (void)global_data;
  (void)pre_data_size;
  (void)pre_data_ptr;
  const std::string *args_info = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), "_aicpu_ffts_args");
  if (args_info == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Aicpu update args failed result of null args_info.");
    return ge::GRAPH_FAILED;
  }
  if (args_info->empty()) {
    GELOGE(ge::PARAM_INVALID, "Aicpu update args failed result of empty args_info.");
    return ge::GRAPH_FAILED;
  }
  const size_t arg_size = args_info->size();
  FMK_SIZET_MULCHECK(arg_size, MAX_THREAD_DIM);
  const size_t arg_total_byte = arg_size * MAX_THREAD_DIM;

  size_t ext_total_byte = 0UL;
  auto ret = CalcAICpuCommonArgsMem(node, ext_total_byte);
  if (ret != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  FMK_SIZET_ADDCHECK(ext_total_byte, arg_total_byte);

  total_size = ext_total_byte + arg_total_byte;
  GELOGI("ext_total_byte is %zu, arg_size is %zu, arg_total_byte is %zu, total_size is %zu.", ext_total_byte, arg_size,
         arg_total_byte, total_size);
  return ge::GRAPH_SUCCESS;
}

FFTS_REGISTER_NODE_CALCULATER(ge::kEngineNameAiCpu, CalcAICpuCCArgsMem);

ge::graphStatus CalcAICpuTfArgsMem(const ge::NodePtr &node, const LoweringGlobalData *global_data, size_t &total_size,
                                   size_t &pre_data_size, std::unique_ptr<uint8_t[]> &pre_data_ptr) {
  (void)global_data;
  (void)pre_data_size;
  (void)pre_data_ptr;

  const std::string *args_info = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), "_aicpu_ffts_args");
  if ((args_info == nullptr) || (args_info->empty())) {
    GELOGE(ge::PARAM_INVALID, "Aicpu update args failed result of empty args_info.");
    return ge::GRAPH_FAILED;
  }
  const size_t arg_size = args_info->size();
  FMK_SIZET_MULCHECK(ge::MemSizeAlign(arg_size * MAX_THREAD_DIM), 1);
  const size_t arg_total_byte = ge::MemSizeAlign(arg_size * MAX_THREAD_DIM);

  size_t ext_total_byte = 0UL;
  auto ret = CalcAICpuCommonArgsMem(node, ext_total_byte);
  if (ret != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  FMK_SIZET_ADDCHECK(ext_total_byte, arg_total_byte);

  const size_t total_ext_args_byte = ext_total_byte + arg_total_byte;
  const size_t io_num = node->GetInDataNodesAndAnchors().size() + node->GetAllOutDataAnchorsSize();
  FMK_SIZET_MULCHECK(io_num, MAX_THREAD_DIM);
  const size_t io_num_dim = io_num * MAX_THREAD_DIM;
  FMK_SIZET_MULCHECK(io_num_dim, sizeof(uintptr_t));
  const size_t io_total_byte = io_num_dim * sizeof(uintptr_t);
  FMK_SIZET_ADDCHECK(total_ext_args_byte, io_total_byte);
  total_size = total_ext_args_byte + io_total_byte;
  GELOGI(
      "ext_total_byte is %zu, arg_size is %zu, arg_total_byte is %zu, io_num is %zu, io_total_byte is %zu, total_size "
      "is %zu.",
      ext_total_byte, arg_size, arg_total_byte, io_num, io_total_byte, total_size);
  return ge::GRAPH_SUCCESS;
}

FFTS_REGISTER_NODE_CALCULATER(ge::kEngineNameAiCpuTf, CalcAICpuTfArgsMem);

ge::graphStatus GetFftsPlusTaskDef(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                   domi::FftsPlusTaskDef &ffts_plus_task_def) {
  auto part_node = node->GetOwnerComputeGraphBarePtr()->GetParentNode();
  auto compile_result = lower_input.global_data->FindCompiledResult(part_node);
  if (compile_result == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Compile result is nullptr.");
    return ge::GRAPH_FAILED;
  }
  if (compile_result->task_defs.empty()) {
    GELOGE(ge::PARAM_INVALID, "Task defs is empty.");
    return ge::GRAPH_FAILED;
  }
  const domi::TaskDef &task_def = compile_result->task_defs.at(0U);
  ffts_plus_task_def = task_def.ffts_plus_task();
  return ge::GRAPH_SUCCESS;
}

std::vector<bg::DevMemValueHolderPtr> AicpuFftsCalAndAllocMem(
    const ge::NodePtr &node, const std::vector<bg::ValueHolderPtr> &thread_ret,
    const std::vector<bg::ValueHolderPtr> &output_shapes, const FFTSLowerInput &lower_input,
    std::vector<bg::DevMemValueHolderPtr> &level_one_mem_addrs) {
  auto output_sizes = AicpuCalcFftsOutputAllocMemVec(node, thread_ret);
  if (output_sizes.empty()) {
    GELOGE(ge::FAILED, "Failed to compute the memory allocated for ffts output.");
    return {};
  }
  auto output_addrs = AicpuAllocOutputMem(node, lower_input, output_sizes, output_shapes, level_one_mem_addrs);
  if (output_addrs.empty()) {
    GELOGE(ge::FAILED, "Failed to get output addrs.");
    return {};
  }

  if (level_one_mem_addrs.empty()) {
    GELOGW("Level one memory addrs is empty.");
  }
  return output_addrs;
}

LowerResult LoweringFFTSAiCpuCCNode(const ge::NodePtr &node, const FFTSLowerInput &lower_input) {
  GELOGD("Lowering AICPU FFTS Plus node[%s]", node->GetName().c_str());
  domi::FftsPlusTaskDef ffts_plus_task_def;
  RET_ERR_RET_IF((GetFftsPlusTaskDef(node, lower_input, ffts_plus_task_def) != ge::GRAPH_SUCCESS),
                 "Not find AI cpu CC ffts plus taskdef.");

  // infer shape
  auto output_shapes = bg::GetMemAllocShape(node, lower_input.input_shapes, *(lower_input.global_data));
  RET_ERR_RET_IF(output_shapes.empty(), "Infer shape failed.");

  std::vector<bg::ValueHolderPtr> thread_ret;
  (void)lower_input.ffts_thread_fun(node, lower_input.input_shapes, output_shapes, lower_input.thread_dim, thread_ret);
  CONVERTER_CHECK_HOLDERS_ALL_OK(thread_ret, static_cast<size_t>(SliceShapeIndex::kTotalNum));

  auto thread_para = CalAutoThreadParam(node, lower_input, thread_ret);
  RET_ERR_RET_IF((thread_para == nullptr), "Cal auto thread param failed.");

  // alloc output memory
  std::vector<bg::DevMemValueHolderPtr> level_one_mem_addrs;
  auto output_addrs = AicpuFftsCalAndAllocMem(node, thread_ret, output_shapes, lower_input, level_one_mem_addrs);
  RET_ERR_RET_IF(output_addrs.empty(), "Failed to get output addrs.");

  // update args and extinfo && ioaddr
  ThreadInfo res = {thread_para, thread_ret};
  auto args_ret = AicpuCCUpdateArgs(node, lower_input, res, output_addrs);
  CONVERTER_CHECK_HOLDERS_ALL_OK(args_ret, static_cast<size_t>(FFTSAicpuArgsOutKey::kNum));
  args_ret[static_cast<size_t>(FFTSAicpuArgsOutKey::kArgAddr)]->RefFrom(lower_input.args_para);

  auto update_task_info =
      UpdateAicpuContext(node, lower_input, args_ret[static_cast<size_t>(FFTSAicpuArgsOutKey::kFlushData)]);

  RET_ERR_RET_IF((update_task_info == nullptr), "Update node context failed.");
  update_task_info->RefFrom(lower_input.task_info);

  std::vector<bg::ValueHolderPtr> aicpu_free_holder;
  RET_ERR_RET_IF(InitAicpuCtxUserData(ffts_plus_task_def, node, lower_input, aicpu_free_holder, update_task_info) !=
                     ge::GRAPH_SUCCESS,
                 "InitAicpuCtxUserData failed.");
  aicpu_free_holder.insert(aicpu_free_holder.cend(), level_one_mem_addrs.cbegin(), level_one_mem_addrs.cend());
  if (!node->GetOpDescBarePtr()->SetExtAttr("_ffts_alloc_vec_holder", std::move(aicpu_free_holder))) {
    GELOGD("Set free attr failed.");
    return {};
  }

  return {HyperStatus::Success(), {update_task_info}, output_shapes, output_addrs};
}

LowerResult LoweringFFTSAiCpuTfNode(const ge::NodePtr &node, const FFTSLowerInput &lower_input) {
  GELOGI("Lowering AICPU TF FFTS Plus node[%s]", node->GetName().c_str());
  domi::FftsPlusTaskDef ffts_plus_task_def;
  RET_ERR_RET_IF((GetFftsPlusTaskDef(node, lower_input, ffts_plus_task_def) != ge::GRAPH_SUCCESS),
                 "Not find AI cpu Tf ffts plus taskdef.");

  // infer shape
  auto output_shapes = bg::GetMemAllocShape(node, lower_input.input_shapes, *(lower_input.global_data));
  RET_ERR_RET_IF(output_shapes.empty(), "Infer shape failed.");

  std::vector<bg::ValueHolderPtr> thread_ret;
  (void)lower_input.ffts_thread_fun(node, lower_input.input_shapes, output_shapes, lower_input.thread_dim, thread_ret);
  CONVERTER_CHECK_HOLDERS_ALL_OK(thread_ret, static_cast<size_t>(SliceShapeIndex::kTotalNum));

  auto thread_para = CalAutoThreadParam(node, lower_input, thread_ret);
  RET_ERR_RET_IF((thread_para == nullptr), "Cal auto thread param failed.");

  // alloc output memory
  std::vector<bg::DevMemValueHolderPtr> level_one_mem_addrs;
  auto output_addrs = AicpuFftsCalAndAllocMem(node, thread_ret, output_shapes, lower_input, level_one_mem_addrs);
  RET_ERR_RET_IF(output_addrs.empty(), "Failed to get output addrs.");

  // update args and extinfo && ioaddr
  ThreadInfo res = {thread_para, thread_ret};
  auto args_ret = AicpuTfUpdateArgs(node, lower_input, res, output_addrs);
  CONVERTER_CHECK_HOLDERS_ALL_OK(args_ret, static_cast<size_t>(FFTSAicpuArgsOutKey::kNum));
  args_ret[static_cast<size_t>(FFTSAicpuArgsOutKey::kArgAddr)]->RefFrom(lower_input.args_para);

  auto update_task_info =
      UpdateAicpuContext(node, lower_input, args_ret[static_cast<size_t>(FFTSAicpuArgsOutKey::kFlushData)]);

  RET_ERR_RET_IF((update_task_info == nullptr), "Update node context failed.");
  update_task_info->RefFrom(lower_input.task_info);

  std::vector<bg::ValueHolderPtr> aicpu_free_holder;
  RET_ERR_RET_IF(InitAicpuCtxUserData(ffts_plus_task_def, node, lower_input, aicpu_free_holder, update_task_info) !=
                     ge::GRAPH_SUCCESS,
                 "InitAicpuCtxUserData failed.");
  aicpu_free_holder.insert(aicpu_free_holder.cend(), level_one_mem_addrs.cbegin(), level_one_mem_addrs.cend());
  if (!node->GetOpDescBarePtr()->SetExtAttr("_ffts_alloc_vec_holder", std::move(aicpu_free_holder))) {
    GELOGD("Set free attr failed");
    return {};
  }

  return {HyperStatus::Success(), {update_task_info}, output_shapes, output_addrs};
}

LowerResult LoweringFFTSAiCpuNode(const ge::NodePtr &node, const FFTSLowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E19999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckFFTSLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E19999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                       ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }
  if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpuTf) {
    GELOGI("Op %s type %s in FFTS tf_aicpu lowering", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringFFTSAiCpuTfNode(node, lower_input);
  } else {
    GELOGI("Op %s type %s in FFTS cc_aicpu lowering", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringFFTSAiCpuCCNode(node, lower_input);
  }
}
FFTS_REGISTER_NODE_CONVERTER_PLACEMENT(kEngineNameAicpuFfts.c_str(), kOnDeviceHbm, LoweringFFTSAiCpuNode);
FFTS_REGISTER_NODE_CONVERTER_PLACEMENT(kEngineNameAicpuTfFfts.c_str(), kOnDeviceHbm, LoweringFFTSAiCpuNode);
}  // namespace gert
