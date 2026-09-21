/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/dsa_task_builder.h"

#include <securec.h>
#include <string>
#include <functional>
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/fe_error_code.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_constants.h"
#include "common/configuration.h"
#include "adapter/common/task_builder_adapter_factory.h"
#include "ops_kernel_builder/task_builder/ffts_task_builder.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "rt_error_codes.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "common/fe_graph_common.h"

namespace fe {
static const std::string kConstantOp = "Constant";
static const std::string kConstant = "Const";
static const size_t kNeededWorkspaceCount = 2;

DsaTaskBuilder::DsaTaskBuilder() {}

DsaTaskBuilder::~DsaTaskBuilder() {}

uint32_t DsaTaskBuilder::DSAFlags::CalAddrOrValueFlag() const {
  uint32_t addr_or_value_flag = 0U;
  switch (distribution_type) {
    case DistributionType::DIS_BITMASK:
      addr_or_value_flag = static_cast<uint32_t>(input1_type);
      break;
    case DistributionType::DIS_UNIFORM:
      addr_or_value_flag = static_cast<uint32_t>(input1_type) << 1;
      addr_or_value_flag |= static_cast<uint32_t>(input2_type) << 2;
      break;
    case DistributionType::DIS_NORMAL:
      addr_or_value_flag = static_cast<uint32_t>(input1_type) << 3;
      addr_or_value_flag |= static_cast<uint32_t>(input1_type) << 4;
      break;
    case DistributionType::DIS_TRUNCATED_NORMAL:
      addr_or_value_flag = static_cast<uint32_t>(input1_type) << 3;
      addr_or_value_flag |= static_cast<uint32_t>(input1_type) << 4;
      break;
    default:
      break;
  }

  addr_or_value_flag |= static_cast<uint32_t>(seed_type) << 5;
  addr_or_value_flag |= static_cast<uint32_t>(rand_count_type) << 6;
  return addr_or_value_flag;
}

Status DsaTaskBuilder::GenerateTask(const ge::Node &node, const ge::RunContext &context,
                                    std::vector<domi::TaskDef> &task_defs) {
  FE_LOGD("DsaTaskBuilder::GenerateTask begin, node name:%s, node type:%s.", node.GetName().c_str(),
          node.GetType().c_str());
  auto opDesc = node.GetOpDesc();
  FE_CHECK_NOTNULL(opDesc);
  bool ffts_flag = false;
  (void)ge::AttrUtils::GetBool(opDesc, kTypeFFTSPlus, ffts_flag);
  uint32_t thread_mode = 0U;
  (void)ge::AttrUtils::GetInt(opDesc, kThreadMode, thread_mode);

  if ((ffts_flag && thread_mode == kManualMode) || opDesc->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
    FftsTaskBuilder ffts_task_builder;
    return ffts_task_builder.GenerateFFTSPlusCtx(node, context, kDsaCoreEngineName);
  }
  int64_t begin_timestamps = GetMicroSecondTime();

  FE_CHECK_NOTNULL(context.dataMemBase);
  context_.dataMemBase = context.dataMemBase;

  domi::TaskDef task_def;
  task_def.set_type(RT_MODEL_TASK_DSA_TASK);
  auto dsa_task_def = task_def.mutable_dsa_task();
  FE_CHECK_NOTNULL(dsa_task_def);
  dsa_task_def->set_op_index(opDesc->GetId());
  dsa_task_def->set_start(dsa_start_);
  dsa_task_def->set_sqe_type(dsa_type_);

  DSAFlags dsa_flags;
  FE_CHECK(GetDsaValueOrAddrFlags(node, dsa_flags) != SUCCESS, FE_LOGD("GetDsaValueOrAddrFlags unsuccessful"), return FAILED);
  dsa_task_def->set_distribution_type(static_cast<uint32_t>(dsa_flags.distribution_type));
  dsa_task_def->set_input_vld(static_cast<uint32_t>(dsa_flags.input_vld));
  dsa_task_def->set_input1_value_or_ptr(static_cast<uint32_t>(dsa_flags.input1_type));
  dsa_task_def->set_input2_value_or_ptr(static_cast<uint32_t>(dsa_flags.input2_type));
  dsa_task_def->set_seed_value_or_ptr(static_cast<uint32_t>(dsa_flags.seed_type));
  dsa_task_def->set_random_count_value_or_ptr(static_cast<uint32_t>(dsa_flags.rand_count_type));
  dsa_task_def->set_input_value_addr_flag(dsa_flags.CalAddrOrValueFlag());

  uint32_t data_type = 0;
  FE_CHECK(GetDataType(node, data_type) != SUCCESS, FE_LOGD("GetDataType unsuccessful"), return FAILED);
  dsa_task_def->set_data_type(data_type);
  dsa_task_def->set_alg_type(dsa_philox_type_);

  DsaWorkspace workspace;
  FE_CHECK(GetWorkspaceInfo(node, workspace) != SUCCESS, FE_LOGD("GetWorkspaceInfo unsuccessful"), return FAILED);
  auto args_ptr = dsa_task_def->mutable_args();
  FE_CHECK_NOTNULL(args_ptr);
  args_ptr->set_workspace_input_addr(workspace.input_addr);
  args_ptr->set_workspace_philox_count_addr(workspace.philox_count_addr);

  uint64_t output_addr = 0;
  FE_CHECK(GetOutputAddr(node, output_addr) != SUCCESS, FE_LOGD("GetOutputAddr unsuccessful"), return FAILED);
  args_ptr->set_output_addr(output_addr);

  DsaInput input;
  FE_CHECK(GetInputs(node, dsa_flags, input) != SUCCESS, FE_LOGD("GetInputs unsuccessful"), return FAILED);
  args_ptr->set_random_count_value_or_addr(input.random_count);
  args_ptr->set_input1_value_or_addr(input.input1);
  args_ptr->set_input2_value_or_addr(input.input2);
  args_ptr->set_seed_value_or_addr(input.seed);

  task_defs.push_back(task_def);
  FE_LOGI("DsaTaskBuilder::GenerateTask end, node name:%s, node type:%s, dsa_task_def:%s.", node.GetName().c_str(),
          node.GetType().c_str(), task_def.DebugString().c_str());

  int64_t end_timestamps = GetMicroSecondTime();
  FE_LOGI("[FE_PERFORMANCE]The time cost of DsaTaskBuilder::GenerateTask is [%ld] micro second.",
          (end_timestamps - begin_timestamps));
  return SUCCESS;
}

bool DsaTaskBuilder::IsConstInput(const ge::Node &node, uint32_t input_idx) {
  auto in_anchor = node.GetInDataAnchor(input_idx);
  if (in_anchor == nullptr) {
    return false;
  }
  auto peer_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_anchor == nullptr) {
    return false;
  }
  auto in_node = peer_anchor->GetOwnerNode();
  if (in_node == nullptr) {
    return false;
  }
  std::string type = in_node->GetType();
  return (type == kConstant || type == kConstantOp);
}

bool DsaTaskBuilder::GetConstInputValue(const ge::Node &node, uint32_t input_idx, std::string &value) {
  std::vector<std::string> value_list;
  if (ge::AttrUtils::GetListStr(node.GetOpDesc(), kOpConstValueList, value_list)) {
    if (input_idx >= value_list.size()) {
      return false;
    }
    value = value_list[input_idx];
    FE_LOGD("DsaTaskBuilder::GetConstInputValue input:%u success.", input_idx);
    return true;
  }
  return false;
}

bool DsaTaskBuilder::GetInputDataType(const ge::Node &node, uint32_t input_idx, ge::DataType &data_type) {
  auto desc_ptr = node.GetOpDesc();
  if (desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GenTask][GetInputDataType][node_name %s] get desc failed.", node.GetName().c_str());
    return false;
  }

  auto input_desc_ptr = desc_ptr->MutableInputDesc(input_idx);
  if (input_desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GenTask][GetInputDataType][node_name: %s] Failed to get input desc.", node.GetName().c_str());
    return false;
  }

  data_type = input_desc_ptr->GetDataType();
  return true;
}

template <typename T>
static std::string ToValueString(const std::string &value) {
  const T *value_ptr = reinterpret_cast<const T *>(value.data());
  return std::to_string(*value_ptr);
}

std::string DsaTaskBuilder::GetValueDebugString(const ge::Node &node, uint32_t input_idx) {
  std::string value;
  if (!GetConstInputValue(node, input_idx, value)) {
    return "";
  }

  ge::DataType data_type = ge::DT_FLOAT;
  if (!GetInputDataType(node, input_idx, data_type)) {
    return value;
  }

  using namespace std::placeholders;
  static const std::map<ge::DataType, std::function<std::string(const std::string &)>> type_funcs = {
      {ge::DT_INT64, std::bind(ToValueString<int64_t>, _1)},
      {ge::DT_UINT64, std::bind(ToValueString<uint64_t>, _1)},
      {ge::DT_FLOAT, std::bind(ToValueString<float>, _1)},
      {ge::DT_FLOAT16, std::bind(ToValueString<float>, _1)},
  };

  auto found = type_funcs.find(data_type);
  if (found != type_funcs.end()) {
    return found->second(value);
  }

  return value;
}

Status DsaTaskBuilder::GetDsaValueOrAddrFlags(const ge::Node &node, DSAFlags &flags) const {
  const auto op_type = node.GetType();
  auto iter = dsa_opname_values_.find(op_type);
  if (iter == dsa_opname_values_.end()) {
    REPORT_FE_ERROR("[GenTask][GetDsaValueOrAddrFlags][type %s] not supported.", op_type.c_str());
    return FAILED;
  }

  auto flags_indexes = dsa_op_flags_idx_.find(op_type);
  if (flags_indexes == dsa_op_flags_idx_.end() || flags_indexes->second.empty()) {
    REPORT_FE_ERROR("[GenTask][GetDsaValueOrAddrFlags][type %s] not supported.", op_type.c_str());
    return FAILED;
  }

  flags = iter->second;

  vector<DsaValueType *> input_dsa_value_types = {&(flags.rand_count_type), &(flags.seed_type), &(flags.input1_type),
                                                  &(flags.input2_type), &(flags.counter_type)};

  for (auto flag_index : flags_indexes->second) {
    if (flag_index >= input_dsa_value_types.size()) {
      REPORT_FE_ERROR("[GenTask][GetDsaValueOrAddrFlags][type %s] not supported.", op_type.c_str());
      return FAILED;
    }
    DsaValueType &value = *input_dsa_value_types[flag_index];

    if (IsConstInput(node, flag_index) && !FeGraphCommon::IsNodeOfUnknownRootGraph(node)) {
      value = DsaValueType::DSA_DATA_VALUE;
      FE_LOGD("DsaTaskBuilder::GenerateTask input:%u is value.", flag_index);
    } else {
      value = DsaValueType::DSA_DATA_ADDR;
      FE_LOGD("DsaTaskBuilder::GenerateTask input: %u is address.", flag_index);
    }
  }

  return SUCCESS;
}

Status DsaTaskBuilder::GetDataType(const ge::Node &node, uint32_t &type) const {
  const uint32_t input_value_idx = 2;
  ge::DataType data_type = ge::DT_FLOAT;
  if (!GetInputDataType(node, input_value_idx, data_type)) {
    return FAILED;
  }

  FE_LOGD("DsaTaskBuilder::GenerateTask input data type is %s.",
          ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
  auto iter = dsa_datatype_values_.find(data_type);
  if (iter == dsa_datatype_values_.end()) {
    REPORT_FE_ERROR("[GenTask][GetDataType][node_name %s] dsa unsupported data_type[%s].", node.GetName().c_str(),
                    ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return FAILED;
  }

  type = iter->second;
  return SUCCESS;
}

Status DsaTaskBuilder::GetWorkspaceInfo(const ge::Node &node, DsaWorkspace &workspace) const {
  auto desc_ptr = node.GetOpDesc();
  if (desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GenTask][GetWorkspaceInfo][node_name %s] get desc failed.", node.GetName().c_str());
    return FAILED;
  }

  auto workspaces = desc_ptr->GetWorkspace();
  if (workspaces.size() < kNeededWorkspaceCount) {
    for (size_t i = workspaces.size(); i < kNeededWorkspaceCount; ++i) {
      workspaces.emplace_back(0);
    }
  }

  workspace.philox_count_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context_.dataMemBase)) +
                                workspaces[0];
  workspace.input_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context_.dataMemBase)) + workspaces[1];
  return SUCCESS;
}

Status DsaTaskBuilder::GetOutputAddr(const ge::Node &node, uint64_t &addr) const {
  auto desc_ptr = node.GetOpDesc();
  if (desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GenTask][GetOutputAddr][node_name %s] Failed to get description.", node.GetName().c_str());
    return FAILED;
  }

  auto offsets = desc_ptr->GetOutputOffset();
  if (offsets.empty()) {
    REPORT_FE_ERROR("[GenTask][GetOutputAddr][node_name %s] output offset is empty.", node.GetName().c_str());
    return FAILED;
  }

  addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context_.dataMemBase)) + offsets[0];
  return SUCCESS;
}

Status DsaTaskBuilder::GetInputs(const ge::Node &node, const DSAFlags &flags, DsaInput &inputs) const {
  auto desc_ptr = node.GetOpDesc();
  if (desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GenTask][GetInputs][node_name: %s] Failed to get description.", node.GetName().c_str());
    return FAILED;
  }

  const vector<std::pair<std::string &, DsaValueType>> input_map = {{inputs.random_count, flags.rand_count_type},
                                                                    {inputs.seed, flags.seed_type},
                                                                    {inputs.input1, flags.input1_type},
                                                                    {inputs.input2, flags.input2_type},
                                                                    {inputs.counter, flags.counter_type}};

  auto offsets = desc_ptr->GetInputOffset();
  auto op_type = node.GetType();
  auto flag_indexes = dsa_op_flags_idx_.find(op_type);
  if (flag_indexes == dsa_op_flags_idx_.end() || flag_indexes->second.empty()) {
    REPORT_FE_ERROR("[GenTask][GetInputs][type %s] not supported.", op_type.c_str());
    return FAILED;
  }

  uint32_t input_index = 0;
  for (auto flag_index : flag_indexes->second) {
    if (flag_index >= input_map.size()) {
      REPORT_FE_ERROR("[GenTask][GetInputs][type %s] not supported.", op_type.c_str());
      return FAILED;
    }
    auto content = input_map[flag_index];
    if (content.second == DsaValueType::DSA_DATA_VALUE) {
      string input_value;
      if (!GetConstInputValue(node, input_index, input_value)) {
        REPORT_FE_ERROR("[GenTask][GetInputs][name %s] Failed to get const value of %u.", node.GetName().c_str(),
                        input_index);
        return FAILED;
      }
      content.first = input_value;
    } else {
      if (offsets.size() != flag_indexes->second.size()) {
        REPORT_FE_ERROR("[GenTask][GetInputs][node_name %s] requires %zu inputs, but only %zu were provided.",
                        node.GetName().c_str(), flag_indexes->second.size(), offsets.size());
        return FAILED;
      }
      auto input_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context_.dataMemBase)) + offsets[input_index];
      content.first.assign(reinterpret_cast<char *>(&input_addr), sizeof(input_addr));
      FE_LOGD("DsaTaskBuilder::GenerateTask input: %u is address.[%lu].", input_index, input_addr);
    }
    input_index++;
  }

  return SUCCESS;
}
}  // namespace fe
