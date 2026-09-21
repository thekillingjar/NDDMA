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
#include "op_def_impl.h"
#include "framework/common/debug/ge_log.h"
#include "register/op_def.h"
#include "register/op_config_registry.h"

namespace ops {
OpDef::OpDef(const char *type) : impl_(new(std::nothrow) OpDefImpl) {
  this->impl_->op_type = type;
  auto regConfigs = GetOpAllAICoreConfig(type);
  GELOGD("Aicore op[%s] configs size: %zu", type, regConfigs.size());
  for (auto it = regConfigs.cbegin(); it!= regConfigs.cend(); ++it) {
    GELOGD("Found aicore op[%s] at socVersion[%s] registerd by REGISTER_OP_AICORE_CONFIG.",
      type, it->first.GetString());
    if (it->second == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Aicore func of op[%s] at socVersion[%s] registerd by REGISTER_OP_AICORE_CONFIG is nullptr.");
      return;
    }
    auto config = it->second();
    this->impl_->op_aicore.AddConfig(it->first.GetString(), config);
  }
}

OpDef::OpDef(const OpDef &op_def) : impl_(new(std::nothrow) OpDefImpl) {
  this->impl_->op_type = op_def.impl_->op_type;
  this->impl_->op_params = op_def.impl_->op_params;
  this->impl_->attrs = op_def.impl_->attrs;
  this->impl_->op_aicore = op_def.impl_->op_aicore;
  this->impl_->op_aicpu = op_def.impl_->op_aicpu;
  this->impl_->op_hostcpu = op_def.impl_->op_hostcpu;
  this->impl_->has_workspace = op_def.impl_->has_workspace;
  this->impl_->infer_shape = op_def.impl_->infer_shape;
  this->impl_->infer_shape_range = op_def.impl_->infer_shape_range;
  this->impl_->infer_data_type = op_def.impl_->infer_data_type;
  this->impl_->op_mc2 = op_def.impl_->op_mc2;
  this->impl_->non_list_len = op_def.impl_->non_list_len;
  this->impl_->category = op_def.impl_->category;
  this->impl_->comment_map = op_def.impl_->comment_map;
  this->impl_->format_mode = op_def.impl_->format_mode;
  this->impl_->enable_fall_back = op_def.impl_->enable_fall_back;
}

OpDef::~OpDef() = default;

OpDef &OpDef::operator=(const OpDef &op_def) {
  if (this != &op_def) {
    *this->impl_ = *op_def.impl_;
  }
  return *this;
}

OpParamDef &OpDef::Input(const char *name) {
  return this->impl_->op_params.Input(name);
}

OpParamDef &OpDef::Output(const char *name) {
  return this->impl_->op_params.Output(name);
}

OpAttrDef &OpDef::Attr(const char *name) {
  return this->GetOrCreateAttr(name);
}
OpDef &OpDef::Comment(CommentSection section, const char *comment) {
  if (section >= CommentSection::SECTION_MAX) {
    GELOGE(ge::PARAM_INVALID, "Ops %s : Comment Section is Invalid", this->GetOpType().GetString());
    return *this;
  }
  if (comment == nullptr || strlen(comment) == 0) {
    GELOGE(ge::PARAM_INVALID, "Ops %s : Comment content cannot be empty", this->GetOpType().GetString());
    return *this;
  }
  if (section == CommentSection::CATEGORY) {
    if (strchr(comment, ' ') != nullptr) {
      GELOGE(ge::PARAM_INVALID, "Ops %s : category names cannot be split by spaces", this->GetOpType().GetString());
      return *this;
    }
    this->impl_->category = comment;
    return *this;
  }
  this->impl_->comment_map[section].emplace_back(comment);
  return *this;
}
ItemFindStatus OpDef::FindAttr(const char *name, OpAttrDef **attr) {
  std::vector<OpAttrDef> *attrList = &this->impl_->attrs;
  for (auto it = attrList->begin(); it != attrList->end(); it++) {
    if (ge::AscendString(it->GetName()) == ge::AscendString(name)) {
      *attr = &(*it);
      return ItemFindStatus::ITEM_FIND;
    }
  }
  return ItemFindStatus::ITEM_NOEXIST;
}

OpAttrDef &OpDef::AddAttr(OpAttrDef &attr) {
  this->impl_->attrs.emplace_back(attr);
  return this->impl_->attrs.back();
}

OpAttrDef &OpDef::GetOrCreateAttr(const char *name) {
  OpAttrDef *pAttr;
  if (this->FindAttr(name, &pAttr) == ItemFindStatus::ITEM_FIND) {
    return *pAttr;
  } else {
    OpAttrDef attr(name);
    return this->AddAttr(attr);
  }
}

std::vector<OpAttrDef> &OpDef::GetAttrs(void) {
  return this->impl_->attrs;
}

OpDef &OpDef::SetInferShape(gert::OpImplRegisterV2::InferShapeKernelFunc func) {
  this->impl_->infer_shape = func;
  return *this;
}

OpDef &OpDef::SetInferShapeRange(gert::OpImplRegisterV2::InferShapeRangeKernelFunc func) {
  this->impl_->infer_shape_range = func;
  return *this;
}

OpDef &OpDef::SetInferDataType(gert::OpImplRegisterV2::InferDataTypeKernelFunc func) {
  this->impl_->infer_data_type = func;
  return *this;
}

gert::OpImplRegisterV2::InferShapeKernelFunc &OpDef::GetInferShape(void) {
  return this->impl_->infer_shape;
}
gert::OpImplRegisterV2::InferShapeRangeKernelFunc &OpDef::GetInferShapeRange(void) {
  return this->impl_->infer_shape_range;
}
gert::OpImplRegisterV2::InferDataTypeKernelFunc &OpDef::GetInferDataType(void) {
  return this->impl_->infer_data_type;
}
ge::AscendString &OpDef::GetOpType(void) {
  return this->impl_->op_type;
}
ge::AscendString &OpDef::GetCateGory(void) const {
  return this->impl_->category;
}
std::vector<ge::AscendString> &OpDef::GetBrief(void) const {
  return this->impl_->comment_map[ops::CommentSection::BRIEF];
}
std::vector<ge::AscendString> &OpDef::GetConstraints(void) const {
  return this->impl_->comment_map[ops::CommentSection::CONSTRAINTS];
}
std::vector<ge::AscendString> &OpDef::GetRestrictions(void) const {
  return this->impl_->comment_map[ops::CommentSection::RESTRICTIONS];
}
std::vector<ge::AscendString> &OpDef::GetSee(void) const {
  return this->impl_->comment_map[ops::CommentSection::SEE];
}
std::vector<ge::AscendString> &OpDef::GetThirdPartyFwkCopat(void) const {
  return this->impl_->comment_map[ops::CommentSection::THIRDPARTYFWKCOMPAT];
}
std::vector<OpParamDef> &OpDef::GetInputs(void) {
  return this->impl_->op_params.GetInputs();
}

std::vector<OpParamDef> &OpDef::GetOutputs(void) {
  return this->impl_->op_params.GetOutputs();
}

void OpDef::MergeParam(std::vector<OpParamDef> &merge, std::vector<OpParamDef> &aicore_params) const {
  for (auto &aicoreParam : aicore_params) {
    bool find = false;
    for (auto &mergeParam : merge) {
      if (mergeParam == aicoreParam) {
        mergeParam.MergeParam(aicoreParam);
        find = true;
        break;
      }
    }
    if (!find) {
      merge.emplace_back(aicoreParam);
    }
  }
}

void OpDef::DfsDataType(DfsParam &dfs_param, const std::vector<OpParamDef> &all_param,
                        uint32_t list_idx, uint32_t non_list_idx) const {
  constexpr uint32_t two = 2;
  const OpParamDef &def = all_param[list_idx / two];
  if (def.IsScalarOrScalarList() && (def.IsScalarTypeSet() || def.IsScalarNameSet())) {
    dfs_param.types.push_back(OpDef::ArrParam(static_cast<uint32_t>(def.GetScalarType()), false));
    DfsFullPermutation(dfs_param, all_param, list_idx + 1, non_list_idx);
    dfs_param.types.pop_back();
  } else if (def.IsDtypeList()) {
    for (uint32_t i = 0; i < def.impl_->types_list.size(); ++i) {
      dfs_param.types.push_back(OpDef::ArrParam(i, true));
      DfsFullPermutation(dfs_param, all_param, list_idx + 1, non_list_idx);
      dfs_param.types.pop_back();
    }
  } else {
    dfs_param.types.push_back(OpDef::ArrParam(non_list_idx, true));
    DfsFullPermutation(dfs_param, all_param, list_idx + 1, non_list_idx);
    dfs_param.types.pop_back();
  }
}

void OpDef::DfsFormat(DfsParam &dfs_param, const std::vector<OpParamDef> &all_param,
                      uint32_t list_idx, uint32_t non_list_idx) const {
  constexpr uint32_t two = 2;
  const OpParamDef &def = all_param[list_idx / two];
  if ((def.IsScalarOrScalarList() || def.IsValueDepend())) {
    dfs_param.formats.push_back(OpDef::ArrParam(static_cast<uint32_t>(ge::FORMAT_ND), false));
    DfsFullPermutation(dfs_param, all_param, list_idx + 1, non_list_idx);
    dfs_param.formats.pop_back();
  } else if (def.IsFormatList()) {
    for (uint32_t i = 0; i < def.impl_->formats_list.size(); ++i) {
      dfs_param.formats.push_back(OpDef::ArrParam(i, true));
      DfsFullPermutation(dfs_param, all_param, list_idx + 1, non_list_idx);
      dfs_param.formats.pop_back();
    }
  } else {
    dfs_param.formats.push_back(OpDef::ArrParam(non_list_idx, true));
    DfsFullPermutation(dfs_param, all_param, list_idx + 1, non_list_idx);
    dfs_param.formats.pop_back();
  }
}

void OpDef::DfsFullPermutation(DfsParam &dfs_param, const std::vector<OpParamDef> &all_param,
                               uint32_t list_idx, uint32_t non_list_idx) const {
  constexpr uint32_t two = 2;
  if (list_idx == all_param.size() * two) {
    dfs_param.full_types.push_back(dfs_param.types);
    dfs_param.full_formats.push_back(dfs_param.formats);
    return;
  }
  // process types while list_idx is even; process formats while list_idx is odd
  if (list_idx % two == 0) {
    DfsDataType(dfs_param, all_param, list_idx, non_list_idx);
  } else {
    DfsFormat(dfs_param, all_param, list_idx, non_list_idx);
  }
}

bool OpDef::IsNonListTypes(const OpParamDef &def) const {
  return (!def.IsScalarOrScalarList() && def.IsDtype()) ||
    (def.IsScalarOrScalarList() && (!def.IsScalarTypeSet() && !def.IsScalarNameSet()) && def.IsDtype());
}

bool OpDef::IsNonListFormats(const OpParamDef &def) const {
  return (!def.IsScalarOrScalarList() && !def.IsValueDepend() && def.IsFormat());
}

uint32_t OpDef::GetNonListLen(std::vector<OpParamDef> &input_param, std::vector<OpParamDef> &output_param) const {
  std::unordered_set<uint32_t> non_list_lens;
  auto set_non_list_len = [this, &non_list_lens](const std::vector<OpParamDef> &params) {
    for (auto &def : params) {
      if (this->IsNonListTypes(def)) {
        non_list_lens.insert(def.impl_->types.size());
      }
      if (this->IsNonListFormats(def)) {
        non_list_lens.insert(def.impl_->formats.size());
      }
    }
  };
  set_non_list_len(input_param);
  set_non_list_len(output_param);

  if (non_list_lens.empty()) {
    return 1;
  }
  if (non_list_lens.size() > 1) {
    GELOGE(ge::PARAM_INVALID, "Element num of DataType and Format is not aligned.");
    return 0;
  }
  if (*non_list_lens.begin() == 0) {
    GELOGE(ge::PARAM_INVALID, "DataType or Format cannot be empty.");
    return 0;
  }
  return *non_list_lens.begin();
}

void OpDef::UpdateDtypeImpl(const DfsParam &dfs_param, OpParamDef &param, const uint32_t &param_idx) {
  uint32_t param_type = dfs_param.full_types[0][param_idx].first;
  bool have_scalar_param = !(dfs_param.full_types[0][param_idx].second);
  if (have_scalar_param && static_cast<ge::DataType>(param_type) != ge::DT_UNDEFINED) {
    if (param.IsSetDtypeForBin()) {
      GELOGW("DataTypeForBinQuery is incompatible with To Type.");
      param.impl_->set_type_for_bin = false;
    }
    param.impl_->types = std::vector<ge::DataType>(dfs_param.full_types.size(), static_cast<ge::DataType>(param_type));
    return;
  }
  if (have_scalar_param && static_cast<ge::DataType>(param_type) == ge::DT_UNDEFINED) {
    return;
  }
  uint32_t num = 0;
  bool is_idx = false;
  std::vector<ge::DataType> data_types;
  std::vector<ge::DataType> data_types_for_bin;
  bool is_follow_list =
      (param.impl_->follow_type == FollowType::ALL || param.impl_->follow_type == FollowType::DTYPE) &&
      param.IsDtypeList();
  for (uint32_t type_idx = 0; type_idx < dfs_param.full_types.size(); ++type_idx) {
    std::tie(num, is_idx) = dfs_param.full_types[type_idx][param_idx];
    if (param.IsSetDtypeForBin() && is_idx && !is_follow_list) {
      data_types_for_bin.emplace_back(param.impl_->types_for_bin[num]);
    }
    if (param.IsDtype()) {
      data_types.emplace_back(param.impl_->types[num]);
    }
    if (param.IsDtypeList()) {
      data_types.emplace_back(param.impl_->types_list[num]);
    }
  }
  if (!data_types_for_bin.empty()) {
    param.impl_->types_for_bin = data_types_for_bin;
  }
  param.impl_->types = data_types;
}

void OpDef::UpdateFormatImpl(const DfsParam &dfs_param, OpParamDef &param, const uint32_t &param_idx) {
  uint32_t param_format = dfs_param.full_formats[0][param_idx].first;
  bool have_scalar_param = !(dfs_param.full_formats[0][param_idx].second);
  if (have_scalar_param) {
    if (param.IsSetFormatForBin()) {
      GELOGW("FormatForBinQuery is incompatible with Scalar/ScalarList or ValueDepend.");
      param.impl_->set_format_for_bin = false;
    }
    param.impl_->formats =
        std::vector<ge::Format>(dfs_param.full_formats.size(), static_cast<ge::Format>(param_format));
    return;
  }
  uint32_t num = 0;
  bool is_idx = false;
  std::vector<ge::Format> data_formats;
  std::vector<ge::Format> data_formats_for_bin;
  bool is_follow_list =
      (param.impl_->follow_type == FollowType::ALL || param.impl_->follow_type == FollowType::FORMAT) &&
      param.IsFormatList();
  for (uint32_t type_idx = 0; type_idx < dfs_param.full_formats.size(); ++type_idx) {
    std::tie(num, is_idx) = dfs_param.full_formats[type_idx][param_idx];
    if (param.IsSetFormatForBin() && is_idx && !is_follow_list) {
      data_formats_for_bin.emplace_back(param.impl_->formats_for_bin[num]);
    }
    if (param.IsFormat()) {
      data_formats.emplace_back(param.impl_->formats[num]);
    }
    if (param.IsFormatList()) {
      data_formats.emplace_back(param.impl_->formats_list[num]);
    }
  }
  if (!data_formats_for_bin.empty()) {
    param.impl_->formats_for_bin = data_formats_for_bin;
  }
  param.impl_->formats = data_formats;
}

void OpDef::UpdateInput(const DfsParam &dfs_param, std::vector<OpParamDef> &input) {
  std::vector<std::pair<uint32_t, ge::AscendString>> to_list;
  for (uint32_t param_idx = 0; param_idx < input.size(); ++param_idx) {
    if (input[param_idx].IsScalarNameSet()) {
      to_list.emplace_back(param_idx, input[param_idx].GetScalarName());
    }
    this->UpdateDtypeImpl(dfs_param, input[param_idx], param_idx);
    this->UpdateFormatImpl(dfs_param, input[param_idx], param_idx);
  }
  auto follow_map = this->GetFollowMap();
  uint32_t input_idx = 0;
  ge::AscendString to_name = "";
  for (const auto &to : to_list) {
    std::tie(input_idx, to_name) = to;
    if (follow_map.find(to_name) == follow_map.end()) {
      GELOGE(ge::PARAM_INVALID, "Param %s : Cannot find param to be set To.",
             input[input_idx].GetParamName().GetString());
      continue;
    }
    const PortFollowInfo &to_param = follow_map.at(to_name);
    if (to_param.port_stat == OpDef::PortStat::OUT) {
      GELOGE(ge::PARAM_INVALID, "Param %s : Cannot set To to output param.",
             input[input_idx].GetParamName().GetString());
      continue;
    }
    if (input[to_param.index_in].IsScalarNameSet()) {
      GELOGE(ge::PARAM_INVALID, "Param %s : Chained parameter setting is not supported in To with name.",
             input[input_idx].GetParamName().GetString());
      continue;
    }
    input[input_idx].impl_->types =  input[to_param.index_in].impl_->types;
    if (input[input_idx].IsSetDtypeForBin()) {
      std::vector<ge::DataType> data_types_for_bin;
      for (uint32_t type_idx = 0; type_idx < dfs_param.full_types.size(); ++type_idx) {
        uint32_t idx = dfs_param.full_types[type_idx][to_param.index_in].first;
        data_types_for_bin.emplace_back(input[input_idx].impl_->types_for_bin[idx]);
      }
      input[input_idx].impl_->types_for_bin = data_types_for_bin;
    }
  }
}

void OpDef::UpdateOutput(const DfsParam &dfs_param, std::vector<OpParamDef> &output) {
  for (uint32_t param_idx = 0; param_idx < output.size(); ++param_idx) {
    if (output[param_idx].IsScalarOrScalarList()) {
      GELOGE(ge::PARAM_INVALID, "Output %s : output cannot be set to Scalar or ScalarList.",
             output[param_idx].GetParamName().GetString());
      continue;
    }
    uint32_t dfs_full_idx = dfs_param.full_types[0].size() - output.size() + param_idx;
    this->UpdateDtypeImpl(dfs_param, output[param_idx], dfs_full_idx);
    this->UpdateFormatImpl(dfs_param, output[param_idx], dfs_full_idx);
  }
}

void OpDef::SetPermutedParam(const DfsParam &dfs_param,
                             std::vector<OpParamDef> &input,
                             std::vector<OpParamDef> &output) {
  this->UpdateInput(dfs_param, input);
  this->UpdateOutput(dfs_param, output);
  this->FollowListImpl(dfs_param, input, output);
}

void OpDef::CheckIncompatible(const std::vector<OpParamDef>& all) const {
  bool is_unknown_shape_format = false;
  for (auto &def : all) {
    if (!def.impl_->unknown_shape_formats.empty()) {
      is_unknown_shape_format = true;
      break;
    }
  }
  if (is_unknown_shape_format) {
    for (auto &def : all) {
      if (def.impl_->formats_list.size() > 1 || def.impl_->types_list.size() > 1) {
        GELOGW("UnknownShapeFormat is incompatible with FormatList/DataTypeList.");
        return;
      }
    }
  }
}

void OpDef::FullPermutation(std::vector<OpParamDef> &input_param,
                            std::vector<OpParamDef> &output_param) {
  this->impl_->non_list_len = GetNonListLen(input_param, output_param);
  std::vector<OpParamDef> all_param = input_param;
  all_param.insert(all_param.end(), output_param.begin(), output_param.end());
  CheckIncompatible(all_param);
  struct DfsParam dfs_param;
  for (uint32_t i = 0; i < this->impl_->non_list_len; ++i) {
    DfsFullPermutation(dfs_param, all_param, 0, i);
  }
  if (dfs_param.full_types.empty() || dfs_param.full_formats.empty()) {
    for (auto &def : input_param) {
      def.impl_->types.clear();
      def.impl_->formats.clear();
    }
    for (auto &def : output_param) {
      def.impl_->types.clear();
      def.impl_->formats.clear();
    }
    return;
  }
  SetPermutedParam(dfs_param, input_param, output_param);
}

void OpDef::SetDefaultND(std::vector<OpParamDef> &defs) const {
  for (auto &def : defs) {
    if (def.impl_->formats.empty() && def.impl_->formats_list.empty()) {
      def.impl_->formats_status = LIST;
      def.impl_->formats_list = {ge::FORMAT_ND};
    }
  }
}

std::vector<std::vector<OpParamDef>> OpDef::GetMergeInputsOutputs(const OpAICoreConfig &aicore_config) {
  this->FollowImpl();
  std::vector<OpParamDef> inputs = this->GetInputs();
  std::vector<OpParamDef> outputs = this->GetOutputs();
  MergeParam(inputs, aicore_config.GetInputs());
  MergeParam(outputs, aicore_config.GetOutputs());
  SetDefaultND(inputs);
  SetDefaultND(outputs);
  this->FullPermutation(inputs, outputs);
  std::vector<std::vector<OpParamDef>> inputs_outputs;
  inputs_outputs.push_back(inputs);
  inputs_outputs.push_back(outputs);
  return inputs_outputs;
}

std::vector<OpParamDef> OpDef::GetMergeInputs(OpAICoreConfig &aicore_config) {
  std::vector<std::vector<OpParamDef>> inputs_outputs = GetMergeInputsOutputs(aicore_config);
  return inputs_outputs[0];
}

std::vector<OpParamDef> OpDef::GetMergeOutputs(OpAICoreConfig &aicore_config) {
  std::vector<std::vector<OpParamDef>> inputs_outputs = GetMergeInputsOutputs(aicore_config);
  return inputs_outputs[1];
}

OpAICoreDef &OpDef::AICore(void) {
  return this->impl_->op_aicore;
}

OpAICPUDef &OpDef::AICPU(void) {
  return this->impl_->op_aicpu;
}

OpHostCPUDef &OpDef::HostCPU(void) {
  return this->impl_->op_hostcpu;
}

OpMC2Def &OpDef::MC2(void) {
  return this->impl_->op_mc2;
}
 
void OpDef::FollowImpl(void) {
  this->impl_->op_params.FollowDataImpl();
  return;
}
 
void OpDef::FollowListImpl(const DfsParam &dfs_param, std::vector<OpParamDef>& input, std::vector<OpParamDef>& output) {
  this->impl_->op_params.FollowListDataImpl(dfs_param, input, output);
  return;
}

std::map<ge::AscendString, OpDef::PortFollowInfo> OpDef::GetFollowMap(void) {
  return this->impl_->op_params.GetFollowMap();
}
std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> OpDef::GetFollowShapeMap(void) {
  return this->impl_->op_params.GetShapeMap();
}
std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> OpDef::GetFollowTypeMap(void) {
  return this->impl_->op_params.GetDtypeMap();
}
OpParamDef OpDef::GetParamDef(const ge::AscendString& name, OpDef::PortStat stat) {
  return this->impl_->op_params.GetParamDef(name, stat);
}

OpDef &OpDef::FormatMatchMode(FormatCheckOption option) {
  this->impl_->format_mode = option;
  return *this;
}

FormatCheckOption OpDef::GetFormatMatchMode(void) {
  return this->impl_->format_mode;
}

OpDef &OpDef::EnableFallBack(void) {
  this->impl_->enable_fall_back = true;
  return *this;
}

bool OpDef::IsEnableFallBack(void) {
  return this->impl_->enable_fall_back;
}

}  // namespace ops
