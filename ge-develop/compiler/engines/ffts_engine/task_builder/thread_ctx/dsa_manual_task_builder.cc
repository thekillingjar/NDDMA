/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dsa_manual_task_builder.h"
#include "inc/ffts_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "rt_error_codes.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
namespace ffts {
static const std::string kOpConstValueList = "_const_value_list";
static const std::string kConstantOp = "Constant";
static const std::string kConstant = "Const";
static constexpr uint32_t k1Bit = 1U;
static constexpr uint32_t k2Bit = 2U;
static constexpr uint32_t k3Bit = 3U;
static constexpr uint32_t k4Bit = 4U;
static constexpr uint32_t k5Bit = 5U;
static constexpr uint32_t k6Bit = 6U;
DSAManualTaskBuilder::DSAManualTaskBuilder() {}
DSAManualTaskBuilder::~DSAManualTaskBuilder() {}
Status DSAManualTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("DSA task builder genContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr(kAttrDsaCtxDef, ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusDsaCtxDef *dsacore_ctx_def = ctx_def_ptr->mutable_dsa_ctx();
  FFTS_CHECK_NOTNULL(dsacore_ctx_def);
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_DSA);
  ffts_plus_ctx_def->set_op_index(op_desc->GetId());

  uint32_t context_id = 0;
  if (ge::AttrUtils::GetInt(op_desc, kContextId, context_id)) {
    ffts_plus_ctx_def->set_context_id(context_id);
  }
  FFTS_LOGD("GenContextDef dsa context_id:%u, nodetype:%s, name:%s, context_type:%u, op_index:%u op_index:%lld",
            context_id, node->GetType().c_str(), node->GetName().c_str(), ffts_plus_ctx_def->context_type(),
            ffts_plus_ctx_def->op_index(), op_desc->GetId());
  domi::FftsPlusDsaCtxDef *dsa_ctx_def = ffts_plus_ctx_def->mutable_dsa_ctx();
  FFTS_CHECK_NOTNULL(dsa_ctx_def);
  if (FillDsaContextData(*node, dsacore_ctx_def, dsa_ctx_def) != SUCCESS) {
    return FAILED;
  }
  dsa_ctx_def->set_pred_cnt(sub_ffts_plus_context[0].pred_cnt);
  dsa_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[0].pred_cnt);
  dsa_ctx_def->set_aten(0);
  dsa_ctx_def->set_successor_num(0);

  (void)ge::AttrUtils::SetListInt(op_desc, kSuccList, sub_ffts_plus_context[0].succ_list);
  FFTS_LOGD("GenContextDef nodetype:%s, name:%s, total_addr_size:%u, pred_cnt:%u", node->GetType().c_str(),
            node->GetName().c_str(), ffts_plus_task_def->addr_size(),
            sub_ffts_plus_context[0].pred_cnt);
  return SUCCESS;
}

Status DSAManualTaskBuilder::FillDsaContextDef(const ge::Node &node, const DSAFlags &dsa_flags,
                                               domi::FftsPlusDsaCtxDef *dsa_ctx_def) const {
  dsa_ctx_def->set_start(dsa_start_);
  dsa_ctx_def->set_distribution_type(static_cast<uint32_t>(dsa_flags.distribution_type));
  dsa_ctx_def->set_input_vld(static_cast<uint32_t>(dsa_flags.input_vld));
  dsa_ctx_def->set_input1_value_or_ptr(static_cast<uint32_t>(dsa_flags.input1_type));
  dsa_ctx_def->set_input2_value_or_ptr(static_cast<uint32_t>(dsa_flags.input2_type));
  dsa_ctx_def->set_seed_value_or_ptr(static_cast<uint32_t>(dsa_flags.seed_type));
  dsa_ctx_def->set_random_count_value_or_ptr(static_cast<uint32_t>(dsa_flags.rand_count_type));
  dsa_ctx_def->set_input_value_addr_flag(dsa_flags.CalAddrOrValueFlag());
  FFTS_LOGD("Distribution_type:%u, input_vld:%u, input1_value_or_ptr:%u, input2_value_or_ptr:%u, seed_value_or_ptr:%u,"
            " random_count_value_or_ptr:%u, input_value_addr_flag:%u.",
            static_cast<uint32_t>(dsa_flags.distribution_type), static_cast<uint32_t>(dsa_flags.input_vld),
            static_cast<uint32_t>(dsa_flags.input1_type), static_cast<uint32_t>(dsa_flags.input2_type),
            static_cast<uint32_t>(dsa_flags.seed_type), static_cast<uint32_t>(dsa_flags.rand_count_type),
            dsa_flags.CalAddrOrValueFlag());
  uint32_t data_type = 0;
  FFTS_CHECK(GetDataType(node, data_type) != SUCCESS, FFTS_LOGD("GetDataType unsuccessful"), return FAILED);
  dsa_ctx_def->set_data_type(data_type);
  dsa_ctx_def->set_alg_type(dsa_philox_type_);
  FFTS_LOGD("Data_type: %u.", data_type);
  return SUCCESS;
}

Status DSAManualTaskBuilder::FillDsaContextData(const ge::Node &node,
                                                const domi::FftsPlusDsaCtxDef *dsacore_ctx_def,
                                                domi::FftsPlusDsaCtxDef *dsa_ctx_def) const {
  if (dsacore_ctx_def == nullptr || dsa_ctx_def == nullptr) {
    return FAILED;
  }
  std::shared_ptr<ge::RunContext> contextptr = nullptr;
  contextptr = node.GetOpDesc()->TryGetExtAttr(kRuntimeContentx, contextptr);
  if (contextptr == nullptr || contextptr->dataMemBase == nullptr) {
     FFTS_LOGD("DsaOpTaskBuilder: contextPtr is null");
     return FAILED;
  }
  DSAFlags dsa_flags;
  if (GetDsaValueOrAddrFlags(node, dsa_flags) != SUCCESS) {
    FFTS_LOGD("GetDsaValueOrAddrFlags unsuccessful");
    return FAILED;
  }
  if (FillDsaContextDef(node, dsa_flags, dsa_ctx_def) != SUCCESS) {
    FFTS_LOGD("Failed to fill DSA context definition");
    return FAILED;
  }

  DsaWorkspace workspace;
  FFTS_CHECK(GetWorkspaceInfo(node, workspace, contextptr) != SUCCESS, FFTS_LOGD("GetWorkspaceInfo unsuccessful"),
             return FAILED);
  auto args_ptr = dsa_ctx_def->mutable_args();
  FFTS_CHECK_NOTNULL(args_ptr);
  args_ptr->set_workspace_input_addr(workspace.input_addr);
  args_ptr->set_workspace_philox_count_addr(workspace.philox_count_addr);

  uint64_t output_addr = 0;
  FFTS_CHECK(GetOutputAddr(node, output_addr, contextptr) != SUCCESS, FFTS_LOGD("GetOutputAddr failed"),
             return FAILED);
  args_ptr->set_output_addr(output_addr);

  DsaInput input;
  FFTS_CHECK(GetInputs(node, dsa_flags, input, contextptr) != SUCCESS, FFTS_LOGD("GetInputs failed"),
  return FAILED);
  args_ptr->set_random_count_value_or_addr(input.random_count);
  args_ptr->set_input1_value_or_addr(input.input1);
  args_ptr->set_input2_value_or_addr(input.input2);
  args_ptr->set_seed_value_or_addr(input.seed);
  FFTS_LOGD("Alg_type:%u, workspace_input_addr:0x%lx, _philox_count_addr:0x%lx, output_addr:0x%lx, "
            "random_count_value_or_addr:%s, input1_value_or_addr:%s, input2_value_or_addr:%s, seed_value_or_addr:%s.",
            dsa_philox_type_, workspace.input_addr, workspace.philox_count_addr, output_addr,
            input.random_count.c_str(), input.input1.c_str(), input.input2.c_str(), input.seed.c_str());
  return SUCCESS;
}

uint32_t DSAManualTaskBuilder::DSAFlags::CalAddrOrValueFlag() const {
  uint32_t addr_or_value_flag = 0U;
  switch (distribution_type) {
    case DistributionType::DIS_BITMASK:
      addr_or_value_flag = static_cast<uint32_t>(input1_type);
      FFTS_LOGD("ffts CalAddrOrValueFlag::DIS_BITMASK input1_type: %u.", static_cast<uint32_t>(input1_type));
      break;
    case DistributionType::DIS_UNIFORM:
      addr_or_value_flag = static_cast<uint32_t>(input1_type) << k1Bit;
      addr_or_value_flag |= static_cast<uint32_t>(input2_type) << k2Bit;
      FFTS_LOGD("ffts CalAddrOrValueFlag::DIS_UNIFORM input1_type:%u, input2_type:%u",
                static_cast<uint32_t>(input1_type),  static_cast<uint32_t>(input2_type));
      break;
    case DistributionType::DIS_NORMAL:
      addr_or_value_flag = static_cast<uint32_t>(input1_type) << k3Bit;
      addr_or_value_flag |= static_cast<uint32_t>(input1_type) << k4Bit;
      FFTS_LOGD("ffts CalAddrOrValueFlag::DIS_NORMAL input1_type: %u",
                static_cast<uint32_t>(input1_type));
      break;
    case DistributionType::DIS_TRUNCATED_NORMAL:
      addr_or_value_flag = static_cast<uint32_t>(input1_type) << k3Bit;
      addr_or_value_flag |= static_cast<uint32_t>(input1_type) << k4Bit;
      FFTS_LOGD("ffts CalAddrOrValueFlag::DIS_TRUNCATED_NORMAL input1_type:%u",
                static_cast<uint32_t>(input1_type));
      break;
    default:
      break;
  }
  addr_or_value_flag |= static_cast<uint32_t>(seed_type) << k5Bit;
  addr_or_value_flag |= static_cast<uint32_t>(rand_count_type) << k6Bit;
  FFTS_LOGD("ffts CalAddrOrValueFlag::DIS_UNIFORM seed_type:%u, rand_count_type:%u",
            static_cast<uint32_t>(seed_type),  static_cast<uint32_t>(rand_count_type));
  return addr_or_value_flag;
}

bool DSAManualTaskBuilder::IsConstInput(const ge::Node &node, uint32_t input_idx) {
  bool ret = false;
  const auto &in_data_anchors = node.GetAllInDataAnchors();
  FFTS_LOGD("DSAManualTaskBuilder::IsConstInput input_idx: %u, in_data_anchors_size: %zu.",
            input_idx, in_data_anchors.size());
  if (input_idx >= static_cast<size_t>(in_data_anchors.size())) {
    return ret;
  }
  auto anchor = node.GetInDataAnchor(input_idx);
  if (anchor == nullptr) {
    return ret;
  }
  auto peer_anchor = anchor->GetPeerOutAnchor();
  if (peer_anchor == nullptr) {
    return ret;
  }
  auto owner_node = peer_anchor->GetOwnerNode();
  if (owner_node == nullptr) {
    return ret;
  }
  const auto type = owner_node->GetType();
  ret = (type == kConstant || type == kConstantOp);
  FFTS_LOGD("DSAManualTaskBuilder::IsConstInput owner_node:%s type:%s",
            owner_node->GetName().c_str(), owner_node->GetType().c_str());
  return ret;
}

bool DSAManualTaskBuilder::GetConstInputValue(const ge::Node &node, uint32_t input_idx, std::string &value) {
  std::vector<std::string> value_list;
  if (ge::AttrUtils::GetListStr(node.GetOpDesc(), kOpConstValueList, value_list)) {
    if (input_idx < value_list.size()) {
      value = value_list[input_idx];
      FFTS_LOGD("DSAManualTaskBuilder::GetConstInputValue input:%u success.", input_idx);
      return true;
    }
  }
  return false;
}

bool DSAManualTaskBuilder::GetInputDataType(const ge::Node &node, uint32_t input_idx, ge::DataType &data_type) {
  auto desc_ptr = node.GetOpDesc();
  auto input_desc_ptr = desc_ptr->MutableInputDesc(input_idx);
  if (input_desc_ptr == nullptr) {
    REPORT_FFTS_ERROR("[GenTask][GetInputDataType][node_name %s] get output desc failed.", node.GetName().c_str());
    return false;
  }
  data_type = input_desc_ptr->GetDataType();
  return true;
}


Status DSAManualTaskBuilder::GetDsaValueOrAddrFlags(const ge::Node &node, DSAFlags &flags) const {
  const auto op_type = node.GetType();
  auto iter = dsa_opname_values_.find(op_type);
  if (iter == dsa_opname_values_.end()) {
    REPORT_FFTS_ERROR("[GenTask][GetDsaValueOrAddrFlags] [type %s] is not supported.", op_type.c_str());
    return FAILED;
  }

  auto flags_indexes = dsa_op_flags_idx_.find(op_type);
  if (flags_indexes == dsa_op_flags_idx_.end() || flags_indexes->second.empty()) {
    REPORT_FFTS_ERROR("[GenTask][GetDsaValueOrAddrFlags] [type %s] is not supported.", op_type.c_str());
    return FAILED;
  }
  flags = iter->second;
  vector<DsaValueType *> input_dsa_value_types = {&(flags.rand_count_type), &(flags.seed_type), &(flags.input1_type),
                                                  &(flags.input2_type), &(flags.counter_type)};
  uint32_t input_idx = 0;
  for (auto flag_index : flags_indexes->second) {
    if (flag_index >= input_dsa_value_types.size()) {
      REPORT_FFTS_ERROR("[GenTask][GetDsaValueOrAddrFlags] [type %s] is not supported.", op_type.c_str());
      return FAILED;
    }
    DsaValueType &value = *input_dsa_value_types[flag_index];
    if (IsConstInput(node, input_idx)) {
      value = DsaValueType::DSA_DATA_VALUE;
      FFTS_LOGD("DSAManualTaskBuilder::GenerateTask input: %u is the value.", input_idx);
    } else {
      value = DsaValueType::DSA_DATA_ADDR;
      FFTS_LOGD("DSAManualTaskBuilder::GenerateTask input: %u is address.", input_idx);
    }
    input_idx++;
  }
  return SUCCESS;
}

Status DSAManualTaskBuilder::GetDataType(const ge::Node &node, uint32_t &type) const {
  const uint32_t input_value_idx = 2;
  ge::DataType data_type = ge::DT_FLOAT;
  if (!GetInputDataType(node, input_value_idx, data_type)) {
    return FAILED;
  }
  FFTS_LOGD("DSAManualTaskBuilder::GenerateTask input data type is %s.",
            ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
  auto iter = dsa_datatype_values_.find(data_type);
  if (iter == dsa_datatype_values_.end()) {
    REPORT_FFTS_ERROR("[GenTask][GetDataType][node_name %s] DSA unsupported data_type [%s].", node.GetName().c_str(),
                      ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return FAILED;
  }
  type = iter->second;
  return SUCCESS;
}

Status DSAManualTaskBuilder::GetWorkspaceInfo(const ge::Node &node, DsaWorkspace &workspace,
                                              const std::shared_ptr<ge::RunContext> &contextptr) const {
  auto desc_ptr = node.GetOpDesc();
  auto workspaces = desc_ptr->GetWorkspace();
  if (workspaces.empty()) {
    workspaces = {0, 0};
  }
  workspace.philox_count_addr = reinterpret_cast<uint64_t>(contextptr->dataMemBase) + workspaces[0];
  workspace.input_addr = reinterpret_cast<uint64_t>(contextptr->dataMemBase) + workspaces[1];
  return SUCCESS;
}

Status DSAManualTaskBuilder::GetOutputAddr(const ge::Node &node, uint64_t &addr,
                                           const std::shared_ptr<ge::RunContext> &contextptr) const {
  auto desc_ptr = node.GetOpDesc();
  auto offsets = desc_ptr->GetOutputOffset();
  if (offsets.empty()) {
    REPORT_FFTS_ERROR("[GenTask][GetOutputAddr][node_name %s] output offset is empty.", node.GetName().c_str());
    return FAILED;
  }
  addr = reinterpret_cast<uint64_t>(contextptr->dataMemBase) + offsets[0];
  return SUCCESS;
}

Status DSAManualTaskBuilder::GetInputs(const ge::Node &node, const DSAFlags &flags, DsaInput &inputs,
                                       const std::shared_ptr<ge::RunContext> &contextptr) const {
  auto desc_ptr = node.GetOpDesc();
  const vector<std::pair<std::string &, DsaValueType>> input_map = {{inputs.random_count, flags.rand_count_type},
                                                                    {inputs.seed, flags.seed_type},
                                                                    {inputs.input1, flags.input1_type},
                                                                    {inputs.input2, flags.input2_type},
                                                                    {inputs.counter, flags.counter_type}};

  auto offsets = desc_ptr->GetInputOffset();
  auto op_type = node.GetType();
  auto flag_indexes = dsa_op_flags_idx_.find(op_type);
  if (flag_indexes == dsa_op_flags_idx_.end() || flag_indexes->second.empty()) {
    REPORT_FFTS_ERROR("[GenTask][GetInputs][type %s] not supported.", op_type.c_str());
    return FAILED;
  }
  uint32_t input_index = 0;
  for (auto flag_index : flag_indexes->second) {
    if (flag_index >= input_map.size()) {
      REPORT_FFTS_ERROR("[GenTask][GetInputs]type [%s] is not supported.", op_type.c_str());
      return FAILED;
    }
    auto content = input_map[flag_index];
    if (content.second == DsaValueType::DSA_DATA_VALUE) {
      string input_value;
      if (!GetConstInputValue(node, input_index, input_value)) {
        REPORT_FFTS_ERROR("[GenTask][GetInputs][name %s] get const value of %u failed.", node.GetName().c_str(),
                          input_index);
        return FAILED;
      }
      content.first = input_value;
    } else {
      if (offsets.size() != flag_indexes->second.size()) {
        REPORT_FFTS_ERROR("[GenTask][GetInputs][node_name %s] requires %zu inputs, but only %zu were provided.",
                          node.GetName().c_str(), flag_indexes->second.size(), offsets.size());
        return FAILED;
      }
      uint64_t input_addr = reinterpret_cast<uint64_t>(contextptr->dataMemBase) + offsets[input_index];
      content.first.assign(reinterpret_cast<const char*>(&input_addr), sizeof(input_addr));
      FFTS_LOGD("DSAManualTaskBuilder::GenerateTask input: %u is at address[%lu].", input_index, input_addr);
    }
    input_index++;
  }
  return SUCCESS;
}
}  // namespace ffts
