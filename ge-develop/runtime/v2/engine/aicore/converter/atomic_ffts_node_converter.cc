/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore_node_converter.h"
#include "aicore_ffts_node_converter.h"
#include "engine/node_converter_utils.h"
#include "common/hyper_status.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_tiling.h"
#include "graph_builder/bg_platform.h"
#include "graph_builder/bg_model_desc.h"
#include "graph_builder/value_holder_generator.h"
#include "engine/ffts_plus/converter/ffts_plus_common.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "framework/common/ge_types.h"
#include "aicore_compile_results.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "graph/utils/node_utils.h"
#include "register/op_tiling_info.h"
namespace gert {
namespace {
struct AtomProcArg {
  std::vector<bg::ValueHolderPtr> output_clean_sizes;
  std::vector<bg::DevMemValueHolderPtr> output_clean_addrs;
  std::vector<bg::ValueHolderPtr> need_ctr_edge;
  bg::ValueHolderPtr blk_dim = nullptr;
  bg::ValueHolderPtr tail_blk_dim = nullptr;
  uint32_t proc_type = static_cast<uint32_t>(kernel::AtomProcType::DY_SLICE_OP);
};

bg::ValueHolderPtr UpdateAtomicContext(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
    bg::ValueHolderPtr &flush_data, AtomProcArg &pro_arg) {
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(flush_data);
  std::vector<uint32_t> ctx_id_vec;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), kAtomicCtxIdList, ctx_id_vec);
  auto ctx_holder = bg::CreateContVecHolder(ctx_id_vec);
  inputs.emplace_back(ctx_holder);
  inputs.emplace_back(pro_arg.blk_dim);
  inputs.emplace_back(pro_arg.tail_blk_dim);
  inputs.emplace_back(lower_input.task_info);
  return bg::ValueHolder::CreateVoid<bg::ValueHolder>("AtomicUpdateContext", inputs);
}

void ConstructArgsInputs(const ge::OpDescPtr &atomic_op_desc, std::vector<bg::ValueHolderPtr> &inputs,
    const FFTSLowerInput &lower_input, bg::ValueHolderPtr atomic_sink_ret, bg::ValueHolderPtr proc_type) {
  std::vector<int64_t> work_clear_indexes;
  ge::AttrUtils::GetListInt(atomic_op_desc, "WorkspaceIndexes", work_clear_indexes);
  std::vector<int64_t> output_clean_indexes;
  ge::AttrUtils::GetListInt(atomic_op_desc, "ClearOutIndexes", output_clean_indexes);
  inputs.emplace_back(lower_input.thread_dim);
  inputs.emplace_back(lower_input.window_size);
  inputs.emplace_back(proc_type);
  inputs.emplace_back(atomic_sink_ret);
  inputs.emplace_back(lower_input.args_para);
  inputs.emplace_back(bg::CreateContVecHolder(work_clear_indexes));
  inputs.emplace_back(bg::CreateContVecHolder(output_clean_indexes));
}

bg::ValueHolderPtr AtomicUpdateArgs(std::vector<bg::ValueHolderPtr> &inputs,
                                    const std::vector<bg::DevMemValueHolderPtr> &output_clean_addrs,
                                    const AtomicFFTSLowerArg &atomic_lowering_arg) {
  inputs.emplace_back(atomic_lowering_arg.workspaces_addrs);
  inputs.emplace_back(atomic_lowering_arg.out_mem_type);
  inputs.emplace_back(atomic_lowering_arg.thread_para);
  inputs.insert(inputs.cend(), output_clean_addrs.cbegin(), output_clean_addrs.cend());
  return bg::ValueHolder::CreateSingleDataOutput("FFTSUpdateAtomicArgs", inputs);
}

ge::graphStatus CalculateAtomicOutputSize(const ge::NodePtr &ori_node, const ge::NodePtr &clean_node,
    const AtomicFFTSLowerArg &lower_args, std::vector<bg::ValueHolderPtr> &shape_size,
    std::vector<bg::ValueHolderPtr> &tail_shape_size) {
  auto holder = bg::ValueHolder::SetScopedCurrentComputeNode(ori_node);
  std::vector<int64_t> output_clean_indexes;
  ge::AttrUtils::GetListInt(clean_node->GetOpDesc(), "ClearOutIndexes", output_clean_indexes);
  auto out_clean_indexes = bg::CreateContVecHolder(output_clean_indexes);
  FE_ASSERT_NOTNULL(out_clean_indexes);
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(out_clean_indexes);
  inputs.emplace_back(lower_args.thread_ret[static_cast<size_t>(kernel::ThreadOutKey::OUT_SHAPES)]);
  shape_size = bg::ValueHolder::CreateDataOutput("FFTSCalcAtomicOutputShapeSize", inputs, output_clean_indexes.size());

  inputs[1] = lower_args.thread_ret[static_cast<size_t>(kernel::ThreadOutKey::LAST_OUT_SHAPES)];
  tail_shape_size = bg::ValueHolder::CreateDataOutput("FFTSCalcAtomicOutputShapeSize", inputs,
                                                      output_clean_indexes.size());
  return ge::GRAPH_SUCCESS;
}

std::vector<bg::ValueHolderPtr> FFTSAtomicTiling(const ge::NodePtr &ori_node, const ge::NodePtr &clean_node,
    const FFTSLowerInput &lower_input, const AtomicFFTSLowerArg &lower_args) {
  std::vector<bg::ValueHolderPtr> shape_size, tail_shape_size;
  auto ret = CalculateAtomicOutputSize(ori_node, clean_node, lower_args, shape_size, tail_shape_size);
  FE_ASSERT_TRUE(ret == ge::GRAPH_SUCCESS);
  auto work_size = lower_args.tiling_ret[TilingContext::kOutputWorkspace];
  std::vector<bg::ValueHolderPtr> tiling_ret =
      bg::TilingForAtomicLegacy(clean_node, work_size, shape_size, *(lower_input.global_data));
  CHECK_HOLDERS_ALL_OK_RET(tiling_ret, static_cast<size_t>(TilingContext::kOutputNum), return {});
  tiling_ret[TilingContext::kOutputTilingData]->RefFrom(
      lower_args.launch_arg[static_cast<size_t>(AllocFFTSArgOutputs::kAtomTilingDataBase)]);
  work_size = lower_args.tiling_ret[static_cast<size_t>(TilingContext::kOutputNum) +
                                    static_cast<size_t>(TilingContext::kOutputWorkspace)];
  std::vector<bg::ValueHolderPtr> tail_tiling_ret =
      bg::TilingForAtomicLegacy(clean_node, work_size, tail_shape_size, *(lower_input.global_data));
  tail_tiling_ret[TilingContext::kOutputTilingData]->RefFrom(
      lower_args.launch_arg[static_cast<size_t>(AllocFFTSArgOutputs::kAtomTailTilingDataBase)]);
  CHECK_HOLDERS_ALL_OK_RET(tail_tiling_ret, static_cast<size_t>(TilingContext::kOutputNum), return {});
  tiling_ret.insert(tiling_ret.cend(), tail_tiling_ret.cbegin(), tail_tiling_ret.cend());
  return tiling_ret;
}

bg::ValueHolderPtr CopyAtomicKernelTilingdata(const FFTSLowerInput &lower_input,
                                              std::shared_ptr<optiling::utils::OpRunInfo> &atomic_tiling_info) {
  GELOGD("Begin to copy atomic kernel tiling data.");
  std::vector<bg::ValueHolderPtr> inputs;
  auto &tiling_data = atomic_tiling_info->GetAllTilingData();
  inputs.emplace_back(lower_input.args_para);
  size_t size = tiling_data.str().size();
  inputs.emplace_back(bg::ValueHolder::CreateConst(tiling_data.str().c_str(), size + 1, true));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&size, sizeof(size)));
  return bg::ValueHolder::CreateVoid<bg::ValueHolder>("CopyAtomicTilingdata", inputs);
}

ge::graphStatus AtomicNodeTiling(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                 const AtomicFFTSLowerArg &lower_args,
                                 const ge::NodePtr &clean_node, AtomProcArg &pro_arg) {
  GELOGD("Node [%s] needs to perform an atomic clean with tiling size [%zu].", node->GetNamePtr(), lower_args.tiling_ret.size());
  auto atomic_op_desc = clean_node->GetOpDesc();
  std::shared_ptr<optiling::utils::OpRunInfo> atomic_tiling_info = nullptr;
  atomic_tiling_info = node->GetOpDesc()->TryGetExtAttr(ge::ATTR_NAME_ATOMIC_OP_RUN_INFO, atomic_tiling_info);
  if (!lower_args.tiling_ret.empty()) {
    auto compile_info_json = ge::AttrUtils::GetStr(node->GetOpDesc(), optiling::ATOMIC_COMPILE_INFO_JSON);
    if (compile_info_json == nullptr) {
      GELOGE(ge::FAILED, "Op[%s] does not have attr[%s].", node->GetNamePtr(),
             optiling::ATOMIC_COMPILE_INFO_JSON.c_str());
      return ge::GRAPH_FAILED;
    }
    if (!ge::AttrUtils::SetStr(atomic_op_desc, optiling::ATOMIC_COMPILE_INFO_JSON, *compile_info_json)) {
      GELOGE(ge::FAILED, "Set attr[%s] for Op[%s] failed.", optiling::ATOMIC_COMPILE_INFO_JSON.c_str(),
             atomic_op_desc->GetNamePtr());
      return ge::GRAPH_FAILED;
    }
    // 1. tiling
    if (lower_args.tiling_ret.size() == ((static_cast<size_t>(TilingContext::kOutputNum) << 1U) + 1U)) {
      // FFTS+ dynamic node, include tail && nontail tiling
      auto tiling_ret = FFTSAtomicTiling(node, clean_node, lower_input, lower_args);
      CHECK_HOLDERS_ALL_OK_RET(tiling_ret, (static_cast<size_t>(TilingContext::kOutputNum) << 1U),
                               return ge::GRAPH_FAILED);
      pro_arg.blk_dim = tiling_ret[TilingContext::kOutputBlockDim];
      pro_arg.tail_blk_dim = tiling_ret[TilingContext::kOutputNum + TilingContext::kOutputBlockDim];
      pro_arg.need_ctr_edge.emplace_back(tiling_ret[TilingContext::kOutputTilingData]);
      pro_arg.need_ctr_edge.emplace_back(tiling_ret[TilingContext::kOutputNum + TilingContext::kOutputTilingData]);
    } else if (lower_args.tiling_ret.size() == static_cast<size_t>(TilingContext::kOutputNum)) {
      // dynamic not slice (mixl2)
      auto tiling_ret = bg::TilingForAtomicLegacy(clean_node, lower_args.tiling_ret[TilingContext::kOutputWorkspace],
                                                  pro_arg.output_clean_sizes, *(lower_input.global_data));
      CHECK_HOLDERS_ALL_OK_RET(tiling_ret, static_cast<size_t>(TilingContext::kOutputNum), return ge::GRAPH_FAILED);
      tiling_ret[TilingContext::kOutputTilingData]->RefFrom(
          lower_args.launch_arg[static_cast<size_t>(AllocFFTSArgOutputs::kAtomTilingDataBase)]);
      pro_arg.blk_dim = tiling_ret[TilingContext::kOutputBlockDim];
      pro_arg.tail_blk_dim = pro_arg.blk_dim;
      pro_arg.need_ctr_edge.emplace_back(tiling_ret[TilingContext::kOutputTilingData]);
      pro_arg.proc_type = static_cast<uint32_t>(kernel::AtomProcType::DY_OP);
    }
  } else if (atomic_tiling_info != nullptr) {
      GELOGD("Static and reuse binary situation.");
      bg::ValueHolderPtr copy_ret;
      int32_t blk_dim = atomic_tiling_info->GetBlockDim();
      pro_arg.blk_dim = bg::ValueHolder::CreateConst(&blk_dim, sizeof(blk_dim));
      pro_arg.tail_blk_dim = pro_arg.blk_dim;
      copy_ret = CopyAtomicKernelTilingdata(lower_input, atomic_tiling_info);
      pro_arg.need_ctr_edge.emplace_back(copy_ret);
      pro_arg.proc_type = static_cast<uint32_t>(kernel::AtomProcType::DY_OP);
  } else {
    uint32_t tmp_val = 1U;
    auto one_val = bg::ValueHolder::CreateConst(&tmp_val, sizeof(tmp_val));
    FE_ASSERT_NOTNULL(one_val);
    pro_arg.blk_dim = one_val;
    pro_arg.tail_blk_dim = one_val;
    pro_arg.proc_type = static_cast<uint32_t>(kernel::AtomProcType::STATIC_OP);
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

bg::ValueHolderPtr LaunchFFTSAtomicClean(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
    const AtomicFFTSLowerArg &lower_args) {
  if (!node->GetOpDesc()->HasAttr(kAtomicCtxIdList)) {
    return nullptr;
  }
  ge::ComputeGraphPtr tmp_graph = nullptr;
  GE_MAKE_SHARED(tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph"), return nullptr);
  FE_ASSERT_NOTNULL(tmp_graph);
  AtomProcArg pro_arg;
  auto clean_node = BuildAtomicNode(node, {nullptr, lower_args.workspaces_addrs, lower_args.output_sizes,
      lower_args.output_addrs}, pro_arg.output_clean_sizes, pro_arg.output_clean_addrs, tmp_graph);
  FE_ASSERT_NOTNULL(clean_node);

  auto current_node_guarder = bg::ValueHolder::SetScopedCurrentComputeNode(clean_node);
  auto ret = AtomicNodeTiling(node, lower_input, lower_args, clean_node, pro_arg);
  FE_ASSERT_TRUE(ret == ge::GRAPH_SUCCESS);

  auto proc_type_holder = bg::ValueHolder::CreateConst(&(pro_arg.proc_type), sizeof(pro_arg.proc_type));
  // 2. sink bin
  auto atomic_sink_ret = SinkFFTSAtomicBin(node);
  FE_RET_NULL_RET_IF((atomic_sink_ret == nullptr), "Sink atomic FFTS node bin failed.");

  // 3. update args
  std::vector<bg::ValueHolderPtr> inputs;
  ConstructArgsInputs(clean_node->GetOpDesc(), inputs, lower_input, atomic_sink_ret, proc_type_holder);
  auto args_ret = AtomicUpdateArgs(inputs, pro_arg.output_clean_addrs, lower_args);
  FE_RET_NULL_RET_IF((args_ret == nullptr), "Update atomic args failed.");
  for (const auto &src_node : pro_arg.need_ctr_edge) {
    bg::ValueHolder::AddDependency(src_node, args_ret);
  }
  // 4. update context
  auto update_ret = UpdateAtomicContext(node, lower_input, args_ret, pro_arg);
  FE_RET_NULL_RET_IF((update_ret == nullptr), "Update atomic context failed.");
  return update_ret;
}
}  // namespace gert
