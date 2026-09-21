/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_TILING_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_TILING_H_
#include <vector>
#include "graph/node.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/lowering/value_holder.h"
#include "engine/aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "common/checker.h"
#include "proto/task.pb.h"
#include "platform/platform_info.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"

namespace gert {
namespace bg {
using ArgsInfo = ArgsInfosDesc::ArgsInfo;
constexpr char const *kMaxTilingSize = "op_para_size";
constexpr char const *kMaxAtomicCleanTilingSize = "atomic_op_para_size";
const std::string kGlobalWorkspaceSize = "globalworkspace_size";
constexpr int kRequiredIoNum = 1U;

struct TilingLowerInput {
  ValueHolderPtr platform_info;
  LoweringGlobalData &global_data;
  ValueHolderPtr launch_arg;
};

void DebugForArgsInfo(const ge::NodePtr &compute_node, const std::vector<ArgsInfo> &args_infos,
                      size_t received_args_info_num);
ge::Status ParseIoNumFromArgsInfo(const std::vector<ArgsInfo> &args_infos, size_t &input_args_info_num,
                                  size_t &output_args_info_num, size_t &input_valid_num, size_t &output_valid_num);
ge::Status CheckArgsInfo(const std::vector<ArgsInfo> &args_infos, const ge::NodePtr &compute_node);
template <typename T>
ge::Status GetOrCreateArgsInfos(std::vector<ArgsInfo> &args_infos, size_t received_args_info_num, const T &kernel_def,
                                const ge::NodePtr &compute_node) {
  if (received_args_info_num != args_infos.size()) {
    GELOGW(
        "Received args info num[%zu], while compute node io num[%zu]. No ArgsInfo received. "
        "Create args info by compute node automatically.",
        received_args_info_num, args_infos.size());
    auto node_input_num = compute_node->GetInDataNodesAndAnchors().size();
    for (size_t idx = 0U; idx < args_infos.size(); ++idx) {
      auto arg_type = idx < node_input_num ? ArgsInfo::ArgsInfoType::INPUT : ArgsInfo::ArgsInfoType::OUTPUT;
      auto start_idx = (idx < node_input_num) ? idx : idx - node_input_num;
      args_infos[idx].Init(arg_type, ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, static_cast<int32_t>(start_idx),
                           kRequiredIoNum);
    }
    return ge::SUCCESS;
  }
  for (size_t idx = 0U; idx < received_args_info_num; ++idx) {
    const ::domi::ArgsInfo &args_info = kernel_def.args_info(idx);
    auto arg_type = static_cast<ArgsInfo::ArgsInfoType>(static_cast<int32_t>(args_info.arg_type()));
    auto arg_fmt = static_cast<ArgsInfo::ArgsInfoFormat>(static_cast<int32_t>(args_info.arg_format()));
    auto start_idx = args_info.start_index();
    auto arg_size = args_info.size();
    args_infos[idx].Init(arg_type, arg_fmt, start_idx, arg_size);
  }
  GE_ASSERT_SUCCESS(CheckArgsInfo(args_infos, compute_node));
  DebugForArgsInfo(compute_node, args_infos, received_args_info_num);
  return ge::SUCCESS;
}

std::vector<ValueHolderPtr> Tiling(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                   const std::vector<ValueHolderPtr> &output_shapes,
                                   const TilingLowerInput &lower_inputs);
std::vector<ValueHolderPtr> FallibleTiling(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                           const std::vector<ValueHolderPtr> &output_shapes,
                                           const TilingLowerInput &lower_inputs);

/**
 * generate exe graph for AtomicClean node, the atomic clean node should have no outputs, inputs like:
 *       DynamicAtomicAddrClean(Attr: WorkspaceIndexes)
 *          /all-workspace      \outputs-to-clean
 *  Data(workspace)            Data(outputs-to-clean)
 *
 * Warning: The lifetime of the input/output nodes for parameter 'node' should be the same as itself.
 *
 * @param clean_node DynamicAtomicAddrClean compute-node instance
 * @param workspaces_size workspace holder in exe-graph
 * @param output_clean_sizes all output clean size holders in exe-graph
 * @return see TilingContext::TilingOutputIndex
 */
std::vector<ValueHolderPtr> TilingForAtomic(const ge::NodePtr &atomic_node, const ValueHolderPtr &workspaces_size,
                                            const std::vector<ValueHolderPtr> &output_clean_sizes,
                                            const ValueHolderPtr &launch_arg, LoweringGlobalData &global_data);

// Legacy Tiling lowering function for ffts, which will be deprecated in the future
std::vector<ValueHolderPtr> TilingLegacy(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                         const std::vector<ValueHolderPtr> &output_shapes,
                                         const ValueHolderPtr &platform_info, LoweringGlobalData &global_data);
std::vector<ValueHolderPtr> TilingForAtomicLegacy(const ge::NodePtr &atomic_node, const ValueHolderPtr &workspaces_size,
                                                  const std::vector<ValueHolderPtr> &output_clean_sizes,
                                                  LoweringGlobalData &global_data);

template <typename T>
ValueHolderPtr CreateComputeNodeDescNode(const ge::NodePtr &compute_node, const T &kernel_def,
                                         uint64_t aligned_max_size) {
  auto &context = kernel_def.context();
  if (context.args_offset().size() < sizeof(uint16_t)) {
    GELOGE(ge::PARAM_INVALID,
           "[Check][Size]Invalid args_offset, size:%zu is smaller"
           "than size of uint16_t. ",
           context.args_offset().size());
    return nullptr;
  }
  auto offset = static_cast<size_t>(*(ge::PtrToPtr<const void, const uint16_t>(context.args_offset().data())));
  auto args_size_without_tiling = static_cast<size_t>(kernel_def.args_size());
  if ((offset > args_size_without_tiling) || (kernel_def.args().size() < args_size_without_tiling)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Offset] Arg offset out of range. offset = %zu, arg_size = %zu, "
           "kernel def arg size = %zu", offset, args_size_without_tiling, kernel_def.args().size());
    return nullptr;
  }

  size_t total_size;
  auto node_desc_holder = RtKernelLaunchArgsEx::ComputeNodeDesc::Create(offset, total_size);
  GE_ASSERT_NOTNULL(node_desc_holder);
  auto node_desc = reinterpret_cast<RtKernelLaunchArgsEx::ComputeNodeDesc *>(node_desc_holder.get());
  if (offset > 0) {
    GE_ASSERT_EOK(
        memcpy_s(node_desc->compiled_args, node_desc->GetCompiledArgsCap(), kernel_def.args().data(), offset));
  }
  const auto op_desc = compute_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  // 这里在有占位的场景下已经不准确了，但是不影响，已经统一到argsformat里了
  node_desc->input_num = compute_node->GetInDataNodesAndAnchors().size();
  node_desc->output_num = compute_node->GetAllOutDataAnchorsSize();
  node_desc->workspace_cap = op_desc->GetWorkspaceBytes().size();
  node_desc->max_tiling_data = aligned_max_size;
  node_desc->need_shape_buffer = false;
  int32_t unknown_shape_type_val = 0;
  if (ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val)) {
    node_desc->need_shape_buffer = static_cast<ge::UnknowShapeOpType>(unknown_shape_type_val) == ge::DEPEND_SHAPE_RANGE;
  }
  node_desc->need_overflow = ge::AttrUtils::HasAttr(op_desc, ge::GLOBALWORKSPACE_TYPE);
  auto node_desc_input = bg::ValueHolder::CreateConst(node_desc_holder.get(), total_size);
  return node_desc_input;
}
template <typename T>
ValueHolderPtr CreateArgsInfoDesc(const ge::NodePtr &compute_node, const T &kernel_def) {
  auto received_args_info_num = static_cast<size_t>(kernel_def.args_info_size());
  size_t node_io_num = compute_node->GetInDataNodesAndAnchors().size() + compute_node->GetAllOutDataAnchorsSize();
  const size_t args_info_num = (received_args_info_num == 0U) ? node_io_num : received_args_info_num;
  std::vector<ArgsInfo> args_infos(args_info_num);
  GE_ASSERT_SUCCESS(GetOrCreateArgsInfos(args_infos, received_args_info_num, kernel_def, compute_node));
  const std::string *dy_mode_ptr = ge::AttrUtils::GetStr(compute_node->GetOpDescBarePtr(), kAttrDynamicParamMode);
  bool is_folded_desc = (dy_mode_ptr != nullptr) && (*dy_mode_ptr == kFoldedWithDesc);
  if (is_folded_desc) {
    GE_ASSERT_GRAPH_SUCCESS(ProcFoldedDescArgs(compute_node, args_infos));
  }

  size_t input_args_info_num = 0U;
  size_t output_args_info_num = 0U;
  size_t input_valid_num = 0U;
  size_t output_valid_num = 0U;
  GE_ASSERT_SUCCESS(ParseIoNumFromArgsInfo(args_infos, input_args_info_num, output_args_info_num, input_valid_num,
                                           output_valid_num));

  size_t total_size = 0U;
  const size_t args_info_size = args_infos.size() * sizeof(ArgsInfo);
  auto args_info_desc_holder = ArgsInfosDesc::Create(args_info_size, total_size);
  GE_ASSERT_NOTNULL(args_info_desc_holder);

  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  args_info_desc->Init(input_args_info_num, output_args_info_num, input_valid_num, output_valid_num);
  if (args_info_size > 0U) {
    GELOGD("Copy args info, copy num:%zu, mem size:%zu", args_infos.size(), args_info_size);
    GE_ASSERT_EOK(memcpy_s(args_info_desc->MutableArgsInfoBase(), args_info_desc->GetArgsInfoSize(), args_infos.data(),
                           args_info_size));
  }

  auto args_info_desc_input = bg::ValueHolder::CreateConst(args_info_desc_holder.get(), total_size);
  return args_info_desc_input;
}
template <typename T>
std::vector<bg::ValueHolderPtr> AllocRtArg(const ge::NodePtr &compute_node, const T &kernel_def,
                                           const char *max_tiling_size_key) {
  int64_t max_size = -1;
  if (!ge::AttrUtils::GetInt(compute_node->GetOpDescBarePtr(), max_tiling_size_key, max_size) || max_size < 0) {
    GELOGE(ge::PARAM_INVALID, "[Check][Size][%s(%s)] Invalid max tiling size: %" PRId64 ", key name %s.",
           compute_node->GetName().c_str(), compute_node->GetType().c_str(), max_size, max_tiling_size_key);
    return {static_cast<size_t>(AllocLaunchArgOutputs::kNum), nullptr};
  }
  auto aligned_max_size = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));

  auto node_desc_input = CreateComputeNodeDescNode(compute_node, kernel_def, aligned_max_size);
  GE_ASSERT_NOTNULL(node_desc_input);
  auto args_info_desc_input = CreateArgsInfoDesc(compute_node, kernel_def);
  GE_ASSERT_NOTNULL(args_info_desc_input);

  return bg::ValueHolder::CreateDataOutput("AllocLaunchArg", {node_desc_input, args_info_desc_input},
                                           static_cast<size_t>(AllocLaunchArgOutputs::kNum));
}
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_TILING_H_
