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
#include "register/op_def.h"
#include "op_def_impl.h"
#include "framework/common/debug/ge_log.h"

namespace ops {
OpParamDef::OpParamDef(const char *name) : impl_(new(std::nothrow) OpParamDefImpl) {
  this->impl_->name = name;
}

OpParamDef::OpParamDef(const OpParamDef &def) : impl_(new(std::nothrow) OpParamDefImpl) {
  this->impl_->name = def.impl_->name;
  this->impl_->param_type = def.impl_->param_type;
  this->impl_->types = def.impl_->types;
  this->impl_->origin_types = def.impl_->origin_types;
  this->impl_->formats = def.impl_->formats;
  this->impl_->formats_list = def.impl_->formats_list;
  this->impl_->types_list = def.impl_->types_list;
  this->impl_->need_compile = def.impl_->need_compile;
  this->impl_->reshape_type = def.impl_->reshape_type;
  this->impl_->value_depend = def.impl_->value_depend;
  this->impl_->depend_scope = def.impl_->depend_scope;
  this->impl_->unknown_shape_formats = def.impl_->unknown_shape_formats;
  this->impl_->ignore_contiguous = def.impl_->ignore_contiguous;
  this->impl_->auto_contiguous = def.impl_->auto_contiguous;
  this->impl_->is_scalar = def.impl_->is_scalar;
  this->impl_->is_scalar_list = def.impl_->is_scalar_list;
  this->impl_->types_status = def.impl_->types_status;
  this->impl_->formats_status = def.impl_->formats_status;
  this->impl_->scalar_name = def.impl_->scalar_name;
  this->impl_->scalar_type = def.impl_->scalar_type;
  this->impl_->version = def.impl_->version;
  this->impl_->init_value_type = def.impl_->init_value_type;
  this->impl_->init_value = def.impl_->init_value;
  this->impl_->init_value_list = def.impl_->init_value_list;
  this->impl_->is_output_shape_depend_on_compute = def.impl_->is_output_shape_depend_on_compute;
  this->impl_->follow_port_name = def.impl_->follow_port_name;
  this->impl_->follow_type = def.impl_->follow_type;
  this->impl_->comment = def.impl_->comment;
  this->impl_->types_for_bin = def.impl_->types_for_bin;
  this->impl_->formats_for_bin = def.impl_->formats_for_bin;
  this->impl_->set_type_for_bin = def.impl_->set_type_for_bin;
  this->impl_->set_format_for_bin = def.impl_->set_format_for_bin;
}


OpParamDef &OpParamDef::operator=(const OpParamDef &def) {
  if (this != &def) {
    *this->impl_ = *def.impl_;
  }
  return *this;
}

void OpParamDef::MergeParam(const OpParamDef &def) {
  this->impl_->param_type = def.impl_->param_type;
  if (!def.impl_->types.empty()) {
    this->impl_->types = def.impl_->types;
    this->impl_->origin_types = def.impl_->origin_types;
  }
  if (!def.impl_->types_list.empty()) {
    this->impl_->types_list = def.impl_->types_list;
  }
  if (!def.impl_->formats.empty()) {
    this->impl_->formats = def.impl_->formats;
  }
  if (!def.impl_->formats_list.empty()) {
    this->impl_->formats_list = def.impl_->formats_list;
  }
  if (def.impl_->need_compile.GetLength() > 0) {
    this->impl_->need_compile = def.impl_->need_compile;
  }
  if (def.impl_->reshape_type.GetLength() > 0) {
    this->impl_->reshape_type = def.impl_->reshape_type;
  }
  if (def.impl_->value_depend.GetLength() > 0) {
    this->impl_->value_depend = def.impl_->value_depend;
  }
  if (!def.impl_->unknown_shape_formats.empty()) {
    this->impl_->unknown_shape_formats = def.impl_->unknown_shape_formats;
  }
  if (!def.impl_->types_for_bin.empty()) {
    this->impl_->types_for_bin = def.impl_->types_for_bin;
    this->impl_->set_type_for_bin = def.impl_->set_type_for_bin;
  }
  if (!def.impl_->formats_for_bin.empty()) {
    this->impl_->formats_for_bin = def.impl_->formats_for_bin;
    this->impl_->set_format_for_bin = def.impl_->set_format_for_bin;
  }
  this->impl_->init_value_type = def.impl_->init_value_type;
  this->impl_->init_value = def.impl_->init_value;
  this->impl_->init_value_list = def.impl_->init_value_list;
  this->impl_->ignore_contiguous = def.impl_->ignore_contiguous;
  this->impl_->auto_contiguous = def.impl_->auto_contiguous;
  this->impl_->is_scalar = def.impl_->is_scalar;
  this->impl_->is_scalar_list = def.impl_->is_scalar_list;
  this->impl_->types_status = def.impl_->types_status;
  this->impl_->formats_status = def.impl_->formats_status;
  this->impl_->scalar_name = def.impl_->scalar_name;
  this->impl_->scalar_type = def.impl_->scalar_type;
  this->impl_->version = def.impl_->version;
  this->impl_->is_output_shape_depend_on_compute = def.impl_->is_output_shape_depend_on_compute;
  this->impl_->depend_scope = def.impl_->depend_scope;
  this->impl_->follow_port_name = def.impl_->follow_port_name;
  this->impl_->follow_type = def.impl_->follow_type;
  this->impl_->comment = def.impl_->comment;
}

OpParamDef::~OpParamDef() = default;

bool OpParamDef::operator==(const OpParamDef &def) const {
  if (this->impl_->name == def.impl_->name) {
    return true;
  }
  return false;
}

OpParamDef &OpParamDef::ParamType(Option param_type) {
  this->impl_->param_type = param_type;
  return *this;
}

bool OpParamDef::IsDtype(void) const {
  return this->impl_->types_status == NON_LIST;
}

bool OpParamDef::IsDtypeList(void) const {
  return this->impl_->types_status == LIST;
}

bool OpParamDef::IsFormat(void) const {
  return this->impl_->formats_status == NON_LIST;
}

bool OpParamDef::IsFormatList(void) const {
  return this->impl_->formats_status == LIST;
}

bool OpParamDef::IsScalarOrScalarList(void) const {
  return this->IsScalar() || this->IsScalarList();
}

bool OpParamDef::IsScalarTypeSet(void) const {
  return this->impl_->scalar_type != ge::DT_UNDEFINED;
}

bool OpParamDef::IsScalarNameSet(void) const {
  return std::strcmp(this->impl_->scalar_name.GetString(), "") != 0;
}

bool OpParamDef::IsValueDepend(void) const {
  return std::strcmp(this->impl_->value_depend.GetString(), "") != 0;
}

OpParamDef &OpParamDef::DataType(std::vector<ge::DataType> types) {
  if (this->IsDtypeList()) {
    GELOGE(ge::PARAM_INVALID, "DataTypeList and DataType can not be called at the same time!");
    return *this;
  }
  if (types.empty()) {
    GELOGE(ge::PARAM_INVALID, "DataType can not be empty");
    return *this;
  }
  if (this->impl_->set_type_for_bin && types.size() != this->impl_->types_for_bin.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : DataType size is not equal to DataTypeForBinQuery size",
        this->impl_->name.GetString());
    return *this;
  }
  this->impl_->types_status = NON_LIST;
  this->impl_->types = types;
  this->impl_->origin_types = types;
  return *this;
}

OpParamDef &OpParamDef::DataTypeList(std::vector<ge::DataType> types) {
  if (this->IsDtype()) {
    GELOGE(ge::PARAM_INVALID, "DataTypeList and DataType can not be called at the same time!");
    return *this;
  }
  if (types.empty()) {
    GELOGE(ge::PARAM_INVALID, "DataTypeList can not be empty");
    return *this;
  }
  std::unordered_set<uint32_t> dtype_set(types.begin(), types.end());
  if (dtype_set.size() < types.size()) {
    GELOGE(ge::PARAM_INVALID, "Element of DataTypeList must be unique!");
    return *this;
  }
  if (this->impl_->set_type_for_bin && types.size() != this->impl_->types_for_bin.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : DataTypeList size is not equal to DataTypeForBinQuery size",
        this->impl_->name.GetString());
    return *this;
  }
  this->impl_->types_status = LIST;
  this->impl_->types_list = types;
  return *this;
}

OpParamDef &OpParamDef::Format(std::vector<ge::Format> formats) {
  if (this->IsFormatList()) {
    GELOGE(ge::PARAM_INVALID, "FormatList and Format can not be called at the same time!");
    return *this;
  }
  if (formats.empty()) {
    GELOGE(ge::PARAM_INVALID, "Format can not be empty");
    return *this;
  }
  if (this->impl_->set_format_for_bin && formats.size() != this->impl_->formats_for_bin.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : Format size is not equal to FormatForBinQuery size",
        this->impl_->name.GetString());
    return *this;
  }
  this->impl_->formats_status = NON_LIST;
  this->impl_->formats = formats;
  return *this;
}

OpParamDef &OpParamDef::FormatList(std::vector<ge::Format> formats) {
  if (this->IsFormat()) {
    GELOGE(ge::PARAM_INVALID, "FormatList and Format can not be called at the same time!");
    return *this;
  }
  if (formats.empty()) {
    GELOGE(ge::PARAM_INVALID, "Format can not be empty");
    return *this;
  }
  std::unordered_set<uint32_t> format_set(formats.begin(), formats.end());
  if (format_set.size() < formats.size()) {
    GELOGE(ge::PARAM_INVALID, "Element of FormatList must be unique!");
    return *this;
  }
  if (this->impl_->set_format_for_bin && formats.size() != this->impl_->formats_for_bin.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : FormatList size is not equal to FormatForBinQuery size",
        this->impl_->name.GetString());
    return *this;
  }
  this->impl_->formats_status = LIST;
  this->impl_->formats_list = formats;
  return *this;
}

OpParamDef &OpParamDef::DataTypeForBinQuery(std::vector<ge::DataType> types) {
  if (types.empty()) {
    GELOGE(ge::PARAM_INVALID, "DataTypeForBinList can not be empty!");
    return *this;
  }
  if (this->impl_->types_status == NON_LIST && this->impl_->types.size() != types.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : DataTypeForBinQuery size is not equal to DataType size",
        this->impl_->name.GetString());
    return *this;
  }
  if (this->impl_->types_status == LIST && this->impl_->types_list.size() != types.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : DataTypeForBinQuery size is not equal to DataTypeList size",
        this->impl_->name.GetString());
    return *this;
  }
  this->impl_->types_for_bin = types;
  this->impl_->set_type_for_bin = true;
  return *this;
}

OpParamDef &OpParamDef::FormatForBinQuery(std::vector<ge::Format> formats) {
  if (formats.empty()) {
    GELOGE(ge::PARAM_INVALID, "FormatForBinList can not be empty!");
    return *this;
  }
  if (this->impl_->formats_status == NON_LIST && this->impl_->formats.size() != formats.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : FormatForBinQuery size is not equal to Format size",
        this->impl_->name.GetString());
    return *this;
  }
  if (this->impl_->formats_status == LIST && this->impl_->formats_list.size() != formats.size()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : FormatForBinQuery size is not equal to FormatList size",
        this->impl_->name.GetString());
    return *this;
  }
  this->impl_->formats_for_bin = formats;
  this->impl_->set_format_for_bin = true;
  return *this;
}

OpParamDef &OpParamDef::UnknownShapeFormat(std::vector<ge::Format> formats) {
  this->impl_->unknown_shape_formats = formats;
  return *this;
}

OpParamDef &OpParamDef::ValueDepend(Option value_depend) {
  if (value_depend == Option::REQUIRED) {
    this->impl_->value_depend = "required";
  } else if (value_depend == Option::OPTIONAL) {
    this->impl_->value_depend = "optional";
  } else {
    this->impl_->value_depend = "";
    GELOGW("Param %s : ValueDepend Option is Invalid", this->impl_->name.GetString());
    return *this;
  }
  this->impl_->depend_scope = DependScope::ALL;
  return *this;
}

OpParamDef &OpParamDef::ValueDepend(Option value_depend, DependScope scope) {
  if (scope >= DependScope::INVALID_SCOPE) {
    GELOGE(ge::PARAM_INVALID, "Param %s : ValueDepend DependScope is Invalid", this->impl_->name.GetString());
    return *this;
  }
  if (this->ValueDepend(value_depend).impl_->value_depend.GetLength() > 0) {
    this->impl_->depend_scope = scope;
  }
  return *this;
}

OpParamDef &OpParamDef::IgnoreContiguous(void) {
  this->impl_->ignore_contiguous = true;
  return *this;
}

OpParamDef &OpParamDef::AutoContiguous() {
  this->impl_->auto_contiguous = true;
  return *this;
}

OpParamDef &OpParamDef::Scalar() {
  this->impl_->is_scalar = true;
  return *this;
}

OpParamDef &OpParamDef::ScalarList() {
  this->impl_->is_scalar_list = true;
  return *this;
}

OpParamDef &OpParamDef::To(const ge::DataType type) {
  if (!this->impl_->is_scalar && !this->impl_->is_scalar_list) {
    GELOGE(ge::PARAM_INVALID, "Param %s : To must be set on the Scalar/ScalarList parameter.",
           this->impl_->name.GetString());
    return *this;
  }
  if (this->impl_->follow_type != FollowType::INVALID_TYPE) {
    GELOGE(ge::PARAM_INVALID, "Param %s : To is incompatible with Follow", this->impl_->name.GetString());
    return *this;
  }

  this->impl_->scalar_type = type;
  return *this;
}

OpParamDef &OpParamDef::To(const char *name) {
  if (!this->impl_->is_scalar && !this->impl_->is_scalar_list) {
    GELOGE(ge::PARAM_INVALID, "Param %s : To must be set on the Scalar/ScalarList parameter.",
           this->impl_->name.GetString());
    return *this;
  }
  if (this->impl_->follow_type != FollowType::INVALID_TYPE) {
    GELOGE(ge::PARAM_INVALID, "Param %s : To is incompatible with Follow", this->impl_->name.GetString());
    return *this;
  }
  this->impl_->scalar_name = name;
  return *this;
}

OpParamDef &OpParamDef::Version(uint32_t version) {
  this->impl_->version = version;
  return *this;
}

OpParamDef &OpParamDef::InitValue(uint64_t value) {
  this->impl_->init_value.value_u64 = value;
  this->impl_->init_value_type = InitValueType::INIT_VALUE_UINT64_T;
  return *this;
}

OpParamDef &OpParamDef::InitValue(const ScalarVar &value) {
  if (!this->impl_->init_value_list.empty()) {
    GELOGW("InitValue has been set, %s InitValue will be reset, please check whether it is correct.",
        this->impl_->name.GetString());
    this->impl_->init_value_list.clear();
  }
  this->impl_->init_value_list.emplace_back(value);
  return *this;
}

OpParamDef &OpParamDef::InitValue(const std::vector<ScalarVar> &value) {
  if (!this->impl_->init_value_list.empty()) {
    GELOGW("InitValue has been set, %s InitValue will be reset, please check whether it is correct.",
        this->impl_->name.GetString());
    this->impl_->init_value_list.clear();
  }
  this->impl_->init_value_list.assign(value.begin(), value.end());
  return *this;
}

OpParamDef &OpParamDef::OutputShapeDependOnCompute() {
  this->impl_->is_output_shape_depend_on_compute = true;
  return *this;
}

OpParamDef &OpParamDef::Follow(const char *paramName)
{
  if (this->IsScalarTypeSet() || this->IsScalarNameSet()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : Follow is incompatible with To", this->impl_->name.GetString());
    return *this;
  }
  this->impl_->follow_port_name = paramName;
  this->impl_->follow_type = FollowType::ALL;
  return *this;
}
 
OpParamDef &OpParamDef::Follow(const char *paramName, FollowType ftype)
{
  if (this->IsScalarTypeSet() || this->IsScalarNameSet()) {
    GELOGE(ge::PARAM_INVALID, "Param %s : Follow is incompatible with To", this->impl_->name.GetString());
    return *this;
  }
  if (ftype >= FollowType::INVALID_TYPE) {
    GELOGE(ge::PARAM_INVALID, "Port %s : FollowType is Invalid", this->impl_->name.GetString());
    return *this;
  }
  this->impl_->follow_port_name = paramName;
  this->impl_->follow_type = ftype;
  return *this;
}

OpParamDef &OpParamDef::Comment(const char *comment) {
  if (comment == nullptr || strlen(comment) == 0) {
    GELOGE(ge::PARAM_INVALID, "Param %s : Comment content cannot be empty", this->GetParamName().GetString());
    return *this;
  }
  this->impl_->comment = comment;
  return *this;
}

bool OpParamDef::IsOutputShapeDependOnCompute(void) const {
  return this->impl_->is_output_shape_depend_on_compute;
}

ge::AscendString &OpParamDef::GetParamName(void) const {
  return this->impl_->name;
}
Option OpParamDef::GetParamType(void) {
  return this->impl_->param_type;
}
std::vector<ge::DataType> &OpParamDef::GetDataTypes(void) {
  if (this->impl_->types.empty()) {
    GELOGW("GetDataTypes returns types_list because types is empty!");
    return this->impl_->types_list;
  }
  return this->impl_->types;
}

std::vector<ge::DataType> &OpParamDef::GetOriginDataTypes(void) {
  if (this->impl_->origin_types.empty()) {
    GELOGE(ge::PARAM_INVALID, "origin types is empty, please check!");
    return this->impl_->origin_types;
  }
  return this->impl_->origin_types;
}

std::vector<ge::DataType> &OpParamDef::GetDataTypesList(void) {
  return this->impl_->types_list;
}
std::vector<ge::DataType> &OpParamDef::GetDataTypesForBin(void) const {
  return this->impl_->types_for_bin;
}
bool OpParamDef::IsSetDtypeForBin(void) const {
  return this->impl_->set_type_for_bin;
}
std::vector<ge::Format> &OpParamDef::GetFormats(void) {
  return this->impl_->formats;
}
std::vector<ge::Format> &OpParamDef::GetFormatsList(void) {
  return this->impl_->formats_list;
}
std::vector<ge::Format> &OpParamDef::GetFormatsForBin(void) const {
  return this->impl_->formats_for_bin;
}
bool OpParamDef::IsSetFormatForBin(void) const {
  return this->impl_->set_format_for_bin;
}
std::vector<ge::Format> &OpParamDef::GetUnknownShapeFormats(void) {
  return this->impl_->unknown_shape_formats;
}
ge::AscendString &OpParamDef::GetValueDepend(void) const {
  return this->impl_->value_depend;
}
DependScope &OpParamDef::GetDependScope(void) const {
  return this->impl_->depend_scope;
}
ge::AscendString &OpParamDef::GetFollowName(void) const {
  return this->impl_->follow_port_name;
}
FollowType &OpParamDef::GetFollowType(void) const {
  return this->impl_->follow_type;
}
ge::AscendString &OpParamDef::GetComment(void) const {
  return this->impl_->comment;
}
bool OpParamDef::GetIgnoreContiguous(void) {
  return this->impl_->ignore_contiguous;
}
bool OpParamDef::GetAutoContiguous(void) {
  return this->impl_->auto_contiguous;
}
bool OpParamDef::IsScalar(void) const {
  return this->impl_->is_scalar;
}
bool OpParamDef::IsScalarList(void) const {
  return this->impl_->is_scalar_list;
}
ge::AscendString &OpParamDef::GetScalarName(void) const {
  return this->impl_->scalar_name;
}
ge::DataType OpParamDef::GetScalarType(void) const {
  return this->impl_->scalar_type;
}

uint32_t OpParamDef::GetVersion(void) {
  return this->impl_->version;
}

InitValueType &OpParamDef::GetInitValueType(void) {
  return this->impl_->init_value_type;
}

InitValueNum &OpParamDef::GetInitValue(void) {
  return this->impl_->init_value;
}

std::vector<ScalarVar> &OpParamDef::GetInitValueList(void) {
  return this->impl_->init_value_list;
}

OpParamDef &OpParamTrunk::Input(const char *name) {
  return this->ParamGetOrCreate(name, false);
}

OpParamDef &OpParamTrunk::Output(const char *name) {
  return this->ParamGetOrCreate(name, true);
}

OpParamDef &OpParamTrunk::ParamGetOrCreate(const char *name, bool is_output) {
  OpParamDef *param;
  if (this->ParamFind(name, is_output, &param) == ItemFindStatus::ITEM_FIND) {
    return *param;
  } else {
    OpParamDef addParam(name);
    return this->ParamAdd(addParam, is_output);
  }
}

ItemFindStatus OpParamTrunk::ParamFind(const char *name, bool is_output, OpParamDef **param) {
  std::vector<OpParamDef> *paramList;

  if (is_output) {
    paramList = &(this->outputs_);
  } else {
    paramList = &(this->inputs_);
  }
  for (auto it = paramList->begin(); it != paramList->end(); it++) {
    if (it->GetParamName() == name) {
      *param = &(*it);
      return ItemFindStatus::ITEM_FIND;
    }
  }
  return ItemFindStatus::ITEM_NOEXIST;
}

OpParamDef &OpParamTrunk::ParamAdd(OpParamDef &param, bool is_output) {
  FollowMapUpdate(param, is_output);
  if (is_output) {
    this->outputs_.emplace_back(param);
    return this->outputs_.back();
  } else {
    this->inputs_.emplace_back(param);
    return this->inputs_.back();
  }
}

std::vector<OpParamDef> &OpParamTrunk::GetInputs(void) {
  return this->inputs_;
}

std::vector<OpParamDef> &OpParamTrunk::GetOutputs(void) {
  return this->outputs_;
}
void OpParamTrunk::FollowMapUpdate(OpParamDef &param, bool is_output) {
  ge::AscendString& cur_name = param.GetParamName();
  if (this->follow_map.find(cur_name) != this->follow_map.end()) {
    OpDef::PortFollowInfo& follow_info = this->follow_map[param.GetParamName()];
    follow_info.port_stat = OpDef::PortStat::INOUT;
    if (is_output) {
      follow_info.index_out = this->outputs_.size();
    } else {
      follow_info.index_in = this->inputs_.size();
    }
    return;
  }
  OpDef::PortFollowInfo follow_info;
  if (is_output) {
    follow_info.port_stat = OpDef::PortStat::OUT;
    follow_info.index_out = this->outputs_.size();
  } else {
    follow_info.port_stat = OpDef::PortStat::IN;
    follow_info.index_in = this->inputs_.size();
  }
  this->follow_map.emplace(cur_name, follow_info);
  return;
}
 
OpParamDef &OpParamTrunk::GetParamDef(const ge::AscendString& name, OpDef::PortStat stat) {
  OpDef::PortFollowInfo& follow_info = this->follow_map[name];
  if (stat == OpDef::PortStat::OUT) {
    return this->outputs_[follow_info.index_out];
  } else {
    return this->inputs_[follow_info.index_in];
  }
}
 
void OpParamTrunk::FollowDataImpl(void) {
  if (this->follow_isimpl == true) {
    return;
  }
  for (auto& op_param_def : this->inputs_) {
    this->DfsFollow(op_param_def, OpDef::PortStat::IN);
  }
  for (auto& op_param_def : this->outputs_) {
    this->DfsFollow(op_param_def, OpDef::PortStat::OUT);
  }
  this->follow_isimpl = true;
  return;
}
void OpParamTrunk::ParamFollow(OpParamDef &op_param_def, OpParamDef &target_param, OpDef::PortStat stat) {
  ge::AscendString cur_name = op_param_def.GetParamName();
  FollowType ftype = op_param_def.GetFollowType();
  ge::AscendString follow_name = target_param.GetParamName();
  if (ftype == FollowType::ALL || ftype == FollowType::DTYPE) {
    this->follow_dtype_map[follow_name].emplace_back(std::make_pair(cur_name, stat));
    if (target_param.IsDtype()) {
      op_param_def.impl_->types_status = target_param.impl_->types_status;
      op_param_def.impl_->types = target_param.impl_->types;
      op_param_def.impl_->origin_types = target_param.impl_->origin_types;
    }
    if (target_param.IsDtypeList()) {
      op_param_def.impl_->types_status = target_param.impl_->types_status;
      op_param_def.impl_->types_list = std::vector<ge::DataType>(1, target_param.impl_->types_list[0]);
      this->follow_dtypelist.emplace_back(std::make_pair(cur_name, stat));
    }
  }
  if (ftype == FollowType::ALL || ftype == FollowType::FORMAT) {
    if (target_param.IsFormat()) {
      op_param_def.impl_->formats_status = target_param.impl_->formats_status;
      op_param_def.impl_->formats = target_param.impl_->formats;
    }
    if (target_param.IsFormatList()) {
      op_param_def.impl_->formats_status = target_param.impl_->formats_status;
      op_param_def.impl_->formats_list = std::vector<ge::Format>(1, target_param.impl_->formats_list[0]);
      this->follow_formatlist.emplace_back(std::make_pair(cur_name, stat));
    }
  }
  if (ftype == FollowType::ALL || ftype == FollowType::SHAPE) {
    this->follow_shape_map[follow_name].emplace_back(std::make_pair(cur_name, stat));
  }
}

void OpParamTrunk::DfsFollow(OpParamDef& op_param_def, OpDef::PortStat stat) {
  if (op_param_def.GetFollowType() >= FollowType::INVALID_TYPE ||
      op_param_def.GetFollowName() == ge::AscendString("")) {
    return;
  }
  ge::AscendString cur_name = op_param_def.GetParamName();
  ge::AscendString follow_name = op_param_def.GetFollowName();
  FollowType ftype = op_param_def.GetFollowType();
  std::map<ge::AscendString, OpDef::PortFollowInfo>& flw_mp = this->follow_map;
  OpDef::PortFollowInfo& follow_info = flw_mp[cur_name];
  if (flw_mp.find(follow_name) == flw_mp.end()) {
    GELOGE(ge::PARAM_INVALID, "PortName %s : FollowPort is Not Exist", cur_name.GetString());
    return;
  }
  if (cur_name == follow_name && flw_mp[cur_name].port_stat != OpDef::PortStat::INOUT) {
    GELOGE(ge::PARAM_INVALID, "PortName %s : FollowPort ParamData is Not Found", cur_name.GetString());
    return;
  }
  if (cur_name != follow_name) {
    std::map<ge::AscendString, int> ring_check_map;
    while (flw_mp.find(follow_name) != flw_mp.end()) {
      if (ring_check_map.find(follow_name) != ring_check_map.end()) {
        GELOGE(ge::PARAM_INVALID, "Port %s : FollowData not Found", cur_name.GetString());
        return;
      }
      if (flw_mp[follow_name].follow_port_name == ge::AscendString("")) {
        break;
      }
      if (flw_mp[follow_name].follow_type != ftype) {
        GELOGE(ge::PARAM_INVALID, "Port %s : FollowType cannot be changed.", cur_name.GetString());
        return;
      }
      ring_check_map.emplace(follow_name, 1);
      follow_name = flw_mp[follow_name].follow_port_name;
    }
  }
  OpDef::PortFollowInfo& target_follow_info = flw_mp[follow_name];
  if (target_follow_info.port_stat == OpDef::PortStat::OUT) {
    GELOGE(ge::PARAM_INVALID, "Port %s : FollowData not Found", cur_name.GetString());
    return;
  }
  follow_info.follow_port_name = follow_name;
  follow_info.follow_type = ftype;
  op_param_def.impl_->follow_port_name = follow_name;
  OpParamDef& target_param = this->inputs_[target_follow_info.index_in];
  this->ParamFollow(op_param_def, target_param, stat);
}

void OpParamTrunk::FollowListDataImpl(const OpDef::DfsParam &dfs_param, std::vector<OpParamDef> &input,
                                      std::vector<OpParamDef> &output) {
  ge::AscendString name;
  OpDef::PortStat stat;
  uint32_t idx = 0;
  auto get_param_ref = [this, &input, &output](const ge::AscendString &port_name,
                                               OpDef::PortStat port_stat) -> OpParamDef & {
    if (port_stat == OpDef::PortStat::OUT) {
      return output[this->follow_map[port_name].index_out];
    } else {
      return input[this->follow_map[port_name].index_in];
    }
  };
  for (const auto& param_pair : this->follow_dtypelist) {
    std::tie(name, stat) = param_pair;
    uint32_t target_index = this->follow_map[this->follow_map[name].follow_port_name].index_in;
    OpParamDef& target_param = input[target_index];
    OpParamDef& op_param_def = get_param_ref(name, stat);
    op_param_def.impl_->types = target_param.impl_->types;
    if (op_param_def.IsSetDtypeForBin()) {
      std::vector<ge::DataType> data_types_for_bin;
      for (uint32_t type_idx = 0; type_idx < dfs_param.full_types.size(); ++type_idx) {
        idx = dfs_param.full_types[type_idx][target_index].first;
        data_types_for_bin.emplace_back(op_param_def.impl_->types_for_bin[idx]);
      }
      op_param_def.impl_->types_for_bin = data_types_for_bin;
    }
  }
  for (const auto& param_pair : this->follow_formatlist) {
    std::tie(name, stat) = param_pair;
    uint32_t target_index = this->follow_map[this->follow_map[name].follow_port_name].index_in;
    OpParamDef& target_param = input[target_index];
    OpParamDef& op_param_def = get_param_ref(name, stat);
    op_param_def.impl_->formats = target_param.impl_->formats;
    if (op_param_def.IsSetFormatForBin()) {
      std::vector<ge::Format> data_formats_for_bin;
      for (uint32_t format_idx = 0; format_idx < dfs_param.full_formats.size(); ++format_idx) {
        idx = dfs_param.full_formats[format_idx][input.size() + target_index].first;
        data_formats_for_bin.emplace_back(op_param_def.impl_->formats_for_bin[idx]);
      }
      op_param_def.impl_->formats_for_bin = data_formats_for_bin;
    }
  }
}
std::map<ge::AscendString, OpDef::PortFollowInfo> OpParamTrunk::GetFollowMap(void) {
  return this->follow_map;
}
std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> OpParamTrunk::GetShapeMap() {
  return this->follow_shape_map;
}
std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> OpParamTrunk::GetDtypeMap() {
  return this->follow_dtype_map;
}
}  // namespace ops
