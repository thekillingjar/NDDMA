/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_node_converter.h"

#include "graph_metadef/common/ge_common/util.h"
#include "common/ge_common/debug/ge_log.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_tensor.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_platform.h"
#include "graph/anchor.h"
#include "lowering/placement/placed_lowering_result.h"
#include "engine/aicore/fe_rt2_common.h"
#include "register/op_tiling/op_tiling_constants.h"

namespace gert {
namespace {
constexpr char const *kAclnnLoweringFunc = "AclnnLoweringFunc";
constexpr char const *kAttrPrecisionModeEnum = "_precision_mode_enum";

const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImplFunctions(const LowerInput &lower_input,
                                                                const ge::NodePtr &node) {
  FE_ASSERT_NOTNULL(node);
  FE_ASSERT_NOTNULL(node->GetOpDesc());
  const OpImplSpaceRegistryV2Ptr &space_registry =
    lower_input.global_data->GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(node->GetOpDesc()->GetOppImplVersion()));
  if (space_registry == nullptr) {
    GELOGW("Op impl registry is nullptr");
    return nullptr;
  }
  return space_registry->GetOpImpl(node->GetType().c_str());
}

bool BuildOpInputTensors(const ge::NodePtr &node, const LowerInput &lower_input,
                         const OpImplKernelRegistry::OpImplFunctions *functions,
                         std::vector<bg::ValueHolderPtr> &op_exe_tensors,
                         std::vector<bg::ValueHolderPtr> &op_exe_input_output_addrs) {
  const ge::OpDescPtr op_desc = node->GetOpDesc();
  size_t index = 0UL;
  for (const ge::InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    if (in_data_anchor == nullptr || in_data_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    ge::OutDataAnchorPtr out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    ge::NodePtr peer_node = out_data_anchor->GetOwnerNode();
    if (peer_node == nullptr) {
      continue;
    }
    const auto *const_lower_result = peer_node->GetOpDesc()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    if (const_lower_result == nullptr) {
      GELOGE(ge::FAILED, "Lowering result of node [%s, %s] is not found.", peer_node->GetNamePtr(),
             peer_node->GetTypePtr());
      return false;
    }
    auto *lower_result = const_cast<PlacedLoweringResult *>(const_lower_result);
    size_t ir_input_index = 0;
    if (ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, index, ir_input_index)) {
      GELOGE(ge::FAILED, "Failed to get ir input index by node [%s, %s]'s input[%zu].", op_desc->GetNamePtr(),
             op_desc->GetTypePtr(), index);
      return false;
    }
    GELOGD("ir input index of node [%s, %s]'s input[%zu] is [%zu].", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           index, ir_input_index);
    int32_t input_placement = functions->IsHostInput(ir_input_index) ? kOnHost : kOnDeviceHbm;
    // remove last true
    const OutputLowerResult *result = lower_result->GetOutputTensorResult(
        *lower_input.global_data, out_data_anchor->GetIdx(), {input_placement, node->GetOpDesc()->GetStreamId()});
    if (result == nullptr || result->shape == nullptr) {
      GELOGE(ge::FAILED, "Lowering result or its shape of node [%s, %s] output[%d] is not null.",
             peer_node->GetNamePtr(), peer_node->GetTypePtr(), out_data_anchor->GetIdx());
      return false;
    }
    op_exe_tensors.emplace_back(result->shape);
    op_exe_input_output_addrs.emplace_back(result->address);
    index++;
  }
  if (lower_input.input_shapes.size() != op_exe_tensors.size()) {
    GELOGE(ge::FAILED, "Size[%zu] of input shapes and size[%zu] of input tensor is not same.",
           lower_input.input_shapes.size(), op_exe_tensors.size());
    return false;
  }
  return true;
}

bool BuildOpOutputTensors(const ge::NodePtr &node, const std::vector<bg::ValueHolderPtr> &output_shapes,
                          const std::vector<bg::DevMemValueHolderPtr> &output_addrs,
                          std::vector<bg::ValueHolderPtr> &op_exe_tensors,
                          std::vector<bg::ValueHolderPtr> &op_exe_input_output_addrs) {
  // BuildOutputTensor
  for (size_t i = 0UL; i < output_shapes.size(); ++i) {
    auto output_tensor =
        bg::BuildRefTensor(node, static_cast<int32_t>(i), static_cast<TensorPlacement>(output_addrs[i]->GetPlacement()),
                           output_shapes[i], output_addrs[i]);
    if (output_tensor == nullptr) {
      GELOGE(ge::FAILED, "Output tensor[%zu] is null.", i);
      return false;
    }
    op_exe_tensors.emplace_back(output_tensor);
    op_exe_input_output_addrs.emplace_back(output_addrs[i]);
  }
  return true;
}

bg::ValueHolderPtr FindOpExecuteFunc(const ge::NodePtr &node, const LowerInput &lower_input) {
  ge::OppImplVersion opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
  auto builder = [&node, &opp_impl_version, &lower_input]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto node_type = bg::ValueHolder::CreateConst(node->GetTypePtr(), node->GetType().size() + 1, true);
      auto space_registry_addr = lower_input.global_data->GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(opp_impl_version)).get();
      auto space_registry = bg::ValueHolder::CreateConst(&space_registry_addr, sizeof(void *), false);
      return {bg::ValueHolder::CreateSingleDataOutput("FindOpExeFunc", {node_type, space_registry})};
    });
  };
  return lower_input.global_data->GetOrCreateUniqueValueHolder(
      node->GetType() + "_FindOpExeFunc_" + std::to_string(static_cast<int32_t>(opp_impl_version)), builder)[0];
}

std::vector<bg::ValueHolderPtr> FindOpExecute2PhaseFunc(const ge::NodePtr &node, const LowerInput &lower_input) {
  ge::OppImplVersion opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
  auto builder = [&node, &opp_impl_version, &lower_input]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto node_type = bg::ValueHolder::CreateConst(node->GetTypePtr(), node->GetType().size() + 1, true);
      auto space_registry_addr = lower_input.global_data->GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(opp_impl_version)).get();
      auto space_registry = bg::ValueHolder::CreateConst(&space_registry_addr, sizeof(void *), false);
      return bg::ValueHolder::CreateDataOutput("FindOpExe2PhaseFunc", {node_type, space_registry},
                                               2U);  // prepare & launch func
    });
  };
  return lower_input.global_data->GetOrCreateUniqueValueHolder(
      node->GetType() + "_FindOpExe2PhaseFunc_" + std::to_string(static_cast<int32_t>(opp_impl_version)), builder);
}

bg::ValueHolderPtr CreateOpExecuteOption(const ge::NodePtr &node) {
  OpExecuteOptions execute_option = {};
  // precision_mode
  execute_option.precision_mode = 0;
  if (!ge::AttrUtils::GetInt(node->GetOpDesc(), kAttrPrecisionModeEnum, execute_option.precision_mode)) {
    GELOGD("Do not get attr precision_mode_enum from node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
  }
  GELOGD("Precision mode is [%d]", execute_option.precision_mode);

  std::string attr_value;
  // deterministic
  execute_option.deterministic = 0;
  if (ge::AttrUtils::GetStr(node->GetOpDesc(), ge::DETERMINISTIC, attr_value)) {
    GELOGD("Attr value of [%s] is [%s].", ge::DETERMINISTIC.c_str(), attr_value.c_str());
    execute_option.deterministic = attr_value == "1" ? 1 : 0;
    attr_value.clear();
  }
  GELOGD("Deterministic is [%d]!", execute_option.deterministic);

  // allow_hf32
  if (ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ALLOW_HF32, attr_value) && !attr_value.empty()) {
    GELOGD("Attr value of [%s] is [%s].", ge::ALLOW_HF32.c_str(), attr_value.c_str());
    const size_t allow_hf32_size = 3;
    if (attr_value.size() < allow_hf32_size) {
      (void)strcpy_s(execute_option.allow_hf32, allow_hf32_size, attr_value.c_str());
    }
  }

  return bg::ValueHolder::CreateConst(&execute_option, sizeof(execute_option));
}

bg::ValueHolderPtr OpExecute(const ge::NodePtr &node, const LowerInput &lower_input,
                             const std::vector<bg::ValueHolderPtr> &op_exe_tensors) {
  GELOGI("Begin to do lowering for aclnn node[%s, %s], enter OpExecute.", node->GetNamePtr(), node->GetTypePtr());
  // Allocate
  auto allocator_holder =
      lower_input.global_data->GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeWorkspace});
  // OpExecuteOptions
  auto execute_option_holder = CreateOpExecuteOption(node);

  // op exe func
  auto op_exe_func = FindOpExecuteFunc(node, lower_input);

  std::vector<bg::ValueHolderPtr> op_exe_inputs;
  (void)op_exe_inputs.insert(op_exe_inputs.end(), op_exe_tensors.begin(), op_exe_tensors.end());

  // Create op executeFunc
  op_exe_inputs.emplace_back(allocator_holder);
  op_exe_inputs.emplace_back(lower_input.global_data->GetStream());
  op_exe_inputs.emplace_back(execute_option_holder);
  op_exe_inputs.emplace_back(op_exe_func);

  auto assembled_platform_info_holders = bg::AppendCoreTypeToPlatform(node, lower_input.global_data);
  GE_ASSERT(assembled_platform_info_holders.size() == static_cast<size_t>(bg::AssemblePlatformInfoIndex::kNums));

  auto aclnn_op_fwk_data_holder = bg::ValueHolder::CreateSingleDataOutput(
      "BuildSingleStageAclnnOpFwkData", {
          assembled_platform_info_holders[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kCoreNumInfos)]
      });
  op_exe_inputs.emplace_back(aclnn_op_fwk_data_holder);
  return bg::ValueHolder::CreateSingleDataOutput("ExecuteOpFunc", op_exe_inputs);
}

bg::ValueHolderPtr Op2PhaseExecute(const ge::NodePtr &node, const LowerInput &lower_input,
                                   const std::vector<bg::ValueHolderPtr> &op_exe_tensors) {
  GELOGI("Begin to do lowering for aclnn node[%s, %s], enter Op2PhaseExecute.", node->GetNamePtr(), node->GetTypePtr());
  // Allocate
  auto allocator_holder =
      lower_input.global_data->GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeWorkspace});
  // OpExecuteOptions
  auto execute_option_holder = CreateOpExecuteOption(node);

  // op exe func
  auto op_exe_funcs = FindOpExecute2PhaseFunc(node, lower_input);
  GE_ASSERT_EQ(op_exe_funcs.size(), 2U);  // Prepare & Launch func

  // platform info
  auto assembled_platform_info_holders = bg::AppendCoreTypeToPlatform(node, lower_input.global_data);
  GE_ASSERT(assembled_platform_info_holders.size() == static_cast<size_t>(bg::AssemblePlatformInfoIndex::kNums));
  auto aclnn_op_fwk_data_holder = bg::ValueHolder::CreateSingleDataOutput(
      "BuildDualStageAclnnOpFwkData", {
          op_exe_funcs[0],
          op_exe_funcs[1],
          assembled_platform_info_holders[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)],
          assembled_platform_info_holders[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kCoreNumInfos)]
      });

  // stream，不能放在fwk_data中，否则会导致CEM失效
  const auto stream = lower_input.global_data->GetStream();

  // Create op execute prepare
  std::vector<bg::ValueHolderPtr> op_exe_prepare_inputs;
  (void)op_exe_prepare_inputs.insert(op_exe_prepare_inputs.end(), op_exe_tensors.begin(), op_exe_tensors.end());
  op_exe_prepare_inputs.emplace_back(execute_option_holder);
  op_exe_prepare_inputs.emplace_back(aclnn_op_fwk_data_holder);
  op_exe_prepare_inputs.emplace_back(stream);
  auto execute_op_prepare = bg::ValueHolder::CreateDataOutput("ExecuteOpPrepare", op_exe_prepare_inputs,
                                                              2U);  // op param & workspace size

  // Malloc op execute workspace size
  auto workspace_addr =
      bg::ValueHolder::CreateSingleDataOutput("AllocBatchHbm", {allocator_holder, execute_op_prepare[1U]});
  bg::ValueHolder::CreateVoidGuarder("FreeBatchHbm", workspace_addr, {});

  // Create op execute launch
  std::vector<bg::ValueHolderPtr> op_exe_launch_inputs;
  (void)op_exe_launch_inputs.insert(op_exe_launch_inputs.end(), op_exe_tensors.begin(), op_exe_tensors.end());
  op_exe_launch_inputs.emplace_back(execute_op_prepare[0U]);
  op_exe_launch_inputs.emplace_back(workspace_addr);
  op_exe_launch_inputs.emplace_back(execute_op_prepare[1U]);
  op_exe_launch_inputs.emplace_back(lower_input.global_data->GetStream());
  op_exe_launch_inputs.emplace_back(aclnn_op_fwk_data_holder);
  return bg::ValueHolder::CreateVoid<bg::ValueHolder>("ExecuteOpLaunch", op_exe_launch_inputs);
}
}  // namespace

LowerResult LoweringAclnnNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGI("Begin to do lowering for aclnn node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
  HyperStatus ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    return {ret, {}, {}, {}};
  }

  // infershape
  auto output_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  // MallocOutputMemory
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs =
      bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes, lower_input.input_addrs, *(lower_input.global_data));

  // op tensors
  const OpImplKernelRegistry::OpImplFunctionsV2 *functions = GetOpImplFunctions(lower_input, node);
  GE_ASSERT_NOTNULL(functions, "Do not find op impl functions for node[%s, %s].", node->GetNamePtr(),
                    node->GetTypePtr());
  std::vector<bg::ValueHolderPtr> op_exe_tensors;
  std::vector<bg::ValueHolderPtr> op_exe_input_output_addrs;
  if (!BuildOpInputTensors(node, lower_input, functions, op_exe_tensors, op_exe_input_output_addrs)) {
    GELOGE(ge::FAILED, "Failed to build input tensors for node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Failed to build op input tensor.")), {}, {}, {}};
  }
  if (!BuildOpOutputTensors(node, output_shapes, output_addrs, op_exe_tensors, op_exe_input_output_addrs)) {
    GELOGE(ge::FAILED, "Failed to build output tensors for node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Failed to build op output tensor.")), {}, {}, {}};
  }

  if (functions->op_execute_func != nullptr) {
    // execute op
    auto execute_op = OpExecute(node, lower_input, op_exe_tensors);
    LOWER_REQUIRE_NOTNULL(execute_op, "Failed to execute op.");
    for (size_t i = 0; i < lower_input.input_addrs.size(); ++i) {
      auto guarder = lower_input.input_addrs[i]->GetGuarder();
      if (guarder != nullptr) {
        GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(execute_op, guarder));
      }
    }
    for (auto &addr : op_exe_input_output_addrs) {
      auto guarder = addr->GetGuarder();
      if (guarder != nullptr) {
        GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(execute_op, guarder));
      }
    }
    return {HyperStatus::Success(), {execute_op}, output_shapes, output_addrs};
  } else if (functions->op_execute_prepare_func != nullptr && functions->op_execute_launch_func != nullptr) {
    auto execute_2_phase_op = Op2PhaseExecute(node, lower_input, op_exe_tensors);
    LOWER_REQUIRE_NOTNULL(execute_2_phase_op, "Failed to execute 2 phase op.");
    for (auto &exe_tensor : op_exe_tensors) {
      auto guarder = exe_tensor->GetGuarder();
      if (guarder != nullptr) {
        GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(execute_2_phase_op, guarder));
      }
    }
    for (auto &addr : op_exe_input_output_addrs) {
      auto guarder = addr->GetGuarder();
      if (guarder != nullptr) {
        GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(execute_2_phase_op, guarder));
      }
    }
    return {HyperStatus::Success(), {execute_2_phase_op}, output_shapes, output_addrs};
  } else {
    GELOGE(ge::FAILED, "Do not find op execute function of node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Do not find op execute function.")), {}, {}, {}};
  }
}

REGISTER_NODE_CONVERTER_PLACEMENT(kAclnnLoweringFunc, kOnDeviceHbm, LoweringAclnnNode);
}  // namespace gert
