/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dsacore_node_converter.h"
#include <cinttypes>
#include "common/checker.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_tiling.h"
#include "engine/aicore/converter/bg_kernel_launch.h"
#include "framework/common/ge_types.h"
#include "exe_graph/runtime/tiling_context.h"
#include "common/hyper_status.h"
#include "runtime/rt_model.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "graph/def_types.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "engine/node_converter_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
constexpr size_t kDSAOutputAddrSize = 1U;
constexpr size_t kDSAStatelessWorkspaceAddrSize = 1U;
constexpr size_t kDSAWorkspaceAddrSize = 2U;
constexpr size_t kDSAInputAddrSize = 3U;
constexpr size_t kDSAStatelessAddrSize = 5U;
constexpr size_t kKernelCredit = 100U;

static bg::ValueHolderPtr CreateVectorHolder(const std::vector<int64_t> &workspaces) {
  size_t total_size = 0U;
  auto vec_holder = ContinuousVector::Create<int64_t>(workspaces.size(), total_size);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  FE_ASSERT_NOTNULL(vec);
  vec->SetSize(workspaces.size());
  for (size_t i = 0U; i < workspaces.size(); ++i) {
    *(reinterpret_cast<int64_t *>(vec->MutableData()) + i) = workspaces[i];
  }
  return bg::ValueHolder::CreateConst(vec, total_size);
}

void LoweringDsaNodeFillSqeSolidInfo(std::unique_ptr<rtStarsDsaSqe_t> &dsa_sqe,
                                     const domi::DSATaskDef &dsa_task) {
  dsa_sqe->sqeHeader.type = static_cast<uint8_t>(dsa_task.sqe_type());
  dsa_sqe->start = dsa_task.start();
  dsa_sqe->functionType = dsa_task.distribution_type();
  dsa_sqe->dataType = dsa_task.data_type();
  dsa_sqe->algoType = dsa_task.alg_type();
  dsa_sqe->paramVldBitmap = dsa_task.input_vld();
  dsa_sqe->paramAddrValBitmap = dsa_task.input_value_addr_flag();
  dsa_sqe->kernelCredit = kKernelCredit;
}

void LoweringDsaNodeFillTaskInfo(std::vector<bg::ValueHolderPtr> &inputs,
                                 const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                                 const std::vector<bg::DevMemValueHolderPtr> &output_addrs,
                                 const domi::DSATaskDef &dsa_task) {
  auto seed_type = dsa_task.seed_value_or_ptr();
  auto random_count_type = dsa_task.random_count_value_or_ptr();
  inputs.emplace_back(bg::ValueHolder::CreateConst(&seed_type, sizeof(seed_type)));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&random_count_type, sizeof(random_count_type)));

  auto &seed_type_string = dsa_task.args().seed_value_or_addr();
  auto &random_count_type_string = dsa_task.args().random_count_value_or_addr();
  inputs.emplace_back(bg::ValueHolder::CreateConst(seed_type_string.c_str(), seed_type_string.size() + 1U, true));
  inputs.emplace_back(bg::ValueHolder::CreateConst(random_count_type_string.c_str(),
                                                   random_count_type_string.size() + 1U, true));

  auto input1_type = dsa_task.input1_value_or_ptr();
  inputs.emplace_back(bg::ValueHolder::CreateConst(&input1_type, sizeof(input1_type)));

  auto &input1_value_string = dsa_task.args().input1_value_or_addr();
  auto &input2_value_string = dsa_task.args().input2_value_or_addr();
  inputs.emplace_back(bg::ValueHolder::CreateConst(input1_value_string.c_str(), input1_value_string.size() + 1U, true));
  inputs.emplace_back(bg::ValueHolder::CreateConst(input2_value_string.c_str(), input2_value_string.size() + 1U, true));

  size_t input_num = input_addrs.size();
  size_t output_num = output_addrs.size();
  auto input_num_holder = bg::ValueHolder::CreateConst(&input_num, sizeof(input_num));
  auto output_num_holder = bg::ValueHolder::CreateConst(&output_num, sizeof(output_num));
  inputs.emplace_back(input_num_holder);
  inputs.emplace_back(output_num_holder);
  inputs.insert(inputs.cend(), input_addrs.cbegin(), input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());
  GELOGD("LoweringDsaNodeFillTaskInfo dump. seed_type:%u, random_count_type:%u, seed_type_string:%s,"
         "random_count_type_string:%s, input1_type:%u, input1_value_string:%s, input2_value_string:%s",
         seed_type, random_count_type, seed_type_string.c_str(), random_count_type_string.c_str(), input1_type,
         input1_value_string.c_str(), input2_value_string.c_str());
}

static LowerResult LoweringDsaCoreNodeCommon(const ge::NodePtr &node, const LowerInput &lower_input,
                                      const domi::TaskDef *task_def) {
  const domi::DSATaskDef &dsa_task = task_def->dsa_task();
  GELOGD("LoweringDsaCoreNodeCommon Enter, node:[%s, %s]", node->GetName().c_str(), node->GetType().c_str());

  const auto &op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Op desc is null")), {}, {}, {}};
  }
  const auto workspaces_size = op_desc->GetWorkspaceBytes();
  const auto &op_type = node->GetType();
  size_t is_dsa_stateless_op = kDSAStatelessOps.count(op_type);
  size_t workspace_addr_size = kDSAStatelessWorkspaceAddrSize;
  if (is_dsa_stateless_op == 0U) {
    workspace_addr_size = kDSAWorkspaceAddrSize;
  }
  if (workspaces_size.size() != workspace_addr_size) { // dsa workspace size
    GELOGE(ge::INTERNAL_ERROR, "Node %s workspace addr size %zu is wrong, expected %zu", node->GetName().c_str(),
           workspaces_size.size(), workspace_addr_size);
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Workspace size is wrong.")), {}, {}, {}};
  }

  const auto &input_addrs = lower_input.input_addrs;
  if ((input_addrs.size() < kDSAInputAddrSize) || (input_addrs.size() > kDSAStatelessAddrSize)) {
    GELOGE(ge::INTERNAL_ERROR, "Node %s input addr size %zu is wrong, expected between %zu and %zu",
           node->GetName().c_str(), input_addrs.size(), kDSAInputAddrSize, kDSAStatelessAddrSize);
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Input size is wrong.")), {}, {}, {}};
  }

  std::unique_ptr<rtStarsDsaSqe_t> dsa_sqe = ge::MakeUnique<rtStarsDsaSqe_t>();
  if (dsa_sqe == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Alloc dsa sqe buff fail")), {}, {}, {}};
  }
  LoweringDsaNodeFillSqeSolidInfo(dsa_sqe, dsa_task);
  auto sqe_data_holder = bg::ValueHolder::CreateConst(dsa_sqe.get(), sizeof(rtStarsDsaSqe_t));
  size_t sqe_len = sizeof(rtStarsDsaSqe_t);
  auto sqe_len_holder = bg::ValueHolder::CreateConst(&sqe_len, sizeof(sqe_len));

  // 1. infer shape
  std::vector<bg::ValueHolderPtr> output_shapes;
  if (bg::IsUnkownShape(op_desc)) {
    output_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  } else {
    output_shapes = CreateOutputShapes(op_desc);
  }
  const auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  // 2. alloc outout mem
  const auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes, *(lower_input.global_data));
  if (output_addrs.size() != kDSAOutputAddrSize) {
    GELOGE(ge::INTERNAL_ERROR, "Node %s output addr size %zu is wrong, expected %zu", node->GetName().c_str(),
           output_addrs.size(), kDSAOutputAddrSize);
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Output size is wrong.")), {}, {}, {}};
  }
  // 3. alloc workspace mem
  auto addr_vec = bg::FrameSelector::OnInitRoot([&workspaces_size, &lower_input]() -> std::vector<bg::ValueHolderPtr> {
    const auto workspaces_size_holder = CreateVectorHolder(workspaces_size);
    return {bg::AllocWorkspaceMem(kOnDeviceHbm, workspaces_size_holder, *(lower_input.global_data))};
  });
  if (addr_vec.size() != 1) {
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Addr size is wrong.")), {}, {}, {}};
  }

  auto workspaces_addr_holder = addr_vec[0];
  const auto workspaces_size_holder = CreateVectorHolder(workspaces_size);

  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(sqe_data_holder);
  inputs.emplace_back(workspaces_addr_holder);
  inputs.emplace_back(workspaces_size_holder);
  inputs.emplace_back(lower_input.global_data->GetStream());
  LoweringDsaNodeFillTaskInfo(inputs, input_addrs, output_addrs, dsa_task);

  // 4. SQE update
  auto update_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("UpdateSqeArg", inputs);
  // 5. launch dsa sqe
  auto launch_arg = bg::LaunchStarsKernel(sqe_data_holder, sqe_len_holder, lower_input.global_data->GetStream());
  // free
  bg::ValueHolder::AddDependency(update_holder, launch_arg);
  for (const auto &addr : lower_input.input_addrs) {
    addr->ReleaseAfter(launch_arg);
  }
  for (const auto &addr : output_addrs) {
    addr->ReleaseAfter(launch_arg);
  }
  return {HyperStatus::Success(), {launch_arg}, output_shapes, output_addrs};
}

LowerResult LoweringDsaCoreNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGD("LoweringDsaCoreNode Enter, node: %s, %s.", node->GetName().c_str(), node->GetType().c_str());
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    return {ret, {}, {}, {}};
  }
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  if (compile_result == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Can not find compile result for node %s type %s", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Can not find compile result")), {}, {}, {}};
  }
  if (compile_result->task_defs.empty() || compile_result->task_defs.size() != 1) {
    GELOGE(ge::PARAM_INVALID, "Unexpected task defs count %zu", compile_result->task_defs.size());
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Unexpected task defs count")), {}, {}, {}};
  }
  const domi::TaskDef *task_def = &compile_result->task_defs[0];
  if (static_cast<ge::ModelTaskType>(task_def->type()) == ge::ModelTaskType::MODEL_TASK_DSA) {
    return LoweringDsaCoreNodeCommon(node, lower_input, task_def);
  }
  return {ret, {}, {}, {}};
}

REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameDsa.c_str(), gert::kOnDeviceHbm, gert::LoweringDsaCoreNode);
} // namespace gert
