/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_store/op_kernel_info.h"
#include <algorithm>
#include "common/fe_log.h"
#include "ops_store/ops_kernel_error_codes.h"
#include "ops_store/ops_kernel_constants.h"
#include "graph/utils/type_utils.h"

namespace fe {
InputOrOutputInfo::InputOrOutputInfo(std::string name)
    : is_input_(true),
      name_(std::move(name)),
      index_(),
      reshape_type_(""),
      propagate_reshape_type_(""),
      op_param_type_(RESERVED),
      op_tune_format_switch_(false),
      op_const_value_depend_(CONST_IGNORE),
      supported_formats_(),
      supported_sub_formats_(),
      supported_dtypes_(),
      supported_unknown_shape_formats_(),
      supported_unknown_shape_dtypes_() {}

uint32_t InputOrOutputInfo::GetIndex() const { return index_; }

InputOrOutputInfo::~InputOrOutputInfo() {}

const std::string& InputOrOutputInfo::GetName() const { return name_; }

const bool& InputOrOutputInfo::GetIsInput() const { return is_input_; }

const vector<ge::DataType>& InputOrOutputInfo::GetDataType() const { return supported_dtypes_; }

const vector<ge::Format> &InputOrOutputInfo::GetFormat() const { return supported_formats_; }

const vector<uint32_t> &InputOrOutputInfo::GetSubformat() const { return supported_sub_formats_; }

const vector<ge::DataType>& InputOrOutputInfo::GetUnknownShapeDataType() const {
  return supported_unknown_shape_dtypes_;
}

const vector<ge::Format>& InputOrOutputInfo::GetUnknownShapeFormat() const {
  return supported_unknown_shape_formats_;
}

OpParamType InputOrOutputInfo::GetParamType() const { return op_param_type_; }

bool InputOrOutputInfo::GetTuneFormatSwitch() const {return op_tune_format_switch_;}

OpConstValueDepend InputOrOutputInfo::GetConstValueDepend() const { return op_const_value_depend_; }

const std::string& InputOrOutputInfo::GetReshapeType() const { return reshape_type_; }

void InputOrOutputInfo::SetReshapeType(const std::string &reshape_type) { reshape_type_ = reshape_type; }

const std::string& InputOrOutputInfo::GetUniqueName() const { return unique_name_; }

/*---------------------------------------Below is definition of AttrInfo-----------------------------------------*/
AttrInfo::AttrInfo(const string &attr_name)
    : attr_name_(attr_name),
      is_required_(false),
      dtype_(),
      is_support_all_value_(false),
      supported_values_(),
      is_default_value_defined_(false),
      default_value_() {}

AttrInfo::~AttrInfo() {}

const ge::GeAttrValue::ValueType& AttrInfo::GetAttrDType() const { return dtype_; }

const bool& AttrInfo::GetSupportAllValue() const { return is_support_all_value_; }

bool AttrInfo::GetIsRequired() const { return is_required_; }

const std::vector<ge::GeAttrValue>& AttrInfo::GetSupportedAttrValueVector() const { return supported_values_; }

const bool& AttrInfo::GetDefaultValueDefinedFlag() const { return is_default_value_defined_; }

const ge::GeAttrValue& AttrInfo::GetDefaultValue() const { return default_value_; }

const std::string& AttrInfo::GetAttrName() const { return attr_name_; }

/*-------------------------------Below is definition of OpKernelInfo--------------------------------------------------*/
OpKernelInfo::OpKernelInfo(const std::string &op_type)
    : init_flag_(false),
      op_type_(op_type),
      impl_type_(EN_RESERVED),
      op_info_(),
      input_infos_(),
      output_infos_(),
      attrs_info_() {
}

OpKernelInfo::OpKernelInfo(const std::string &op_type, const OpImplType &impl_type)
    : init_flag_(false),
      op_type_(op_type),
      impl_type_(impl_type),
      op_info_(),
      input_infos_(),
      output_infos_(),
      attrs_info_() {
}

OpKernelInfo::~OpKernelInfo() {}

const vector<InputOrOutputInfoPtr>& OpKernelInfo::GetAllInputInfo() const { return input_infos_; }

const vector<InputOrOutputInfoPtr>& OpKernelInfo::GetAllOutputInfo() const { return output_infos_; }

Status OpKernelInfo::GetTensorInfoByName(const bool &isinput, const std::string &tensor_name,
                                         InputOrOutputInfoPtr &tensor_info_ptr) const {
  if (isinput) {
    return GetInputInfoByName(tensor_name, tensor_info_ptr);
  } else {
    return GetOutputInfoByName(tensor_name, tensor_info_ptr);
  }
}

Status OpKernelInfo::GetInputInfoByName(const std::string &input_name, InputOrOutputInfoPtr &input_info_ptr) const {
  for (auto& input_iter : input_infos_) {
    if (input_iter->GetName() == input_name) {
      input_info_ptr = input_iter;
      return SUCCESS;
    }
  }
  return OP_INPUT_NOT_FOUND_IN_OP_KERNEL_INFO;
}

Status OpKernelInfo::GetOutputInfoByName(const std::string &output_name, InputOrOutputInfoPtr &output_info_ptr) const {
  for (auto& output_iter : output_infos_) {
    if (output_iter->GetName() == output_name) {
      output_info_ptr = output_iter;
      return SUCCESS;
    }
  }
  return OP_OUTPUT_NOT_FOUND_IN_OP_KERNEL_INFO;
}

const std::string& OpKernelInfo::GetOpType() const { return op_type_; }

OpImplType OpKernelInfo::GetOpStoreImplType() const { return impl_type_; }

const std::string& OpKernelInfo::GetOpsPathNamePrefix() const { return ops_path_name_prefix_; }

Status OpKernelInfo::GetAttrTypeByAttrName(const std::string &attr_name,
                                                 ge::GeAttrValue::ValueType &dtype) const {
  if (attrs_info_.empty()) {
    return OP_ATTR_EMPTY_IN_OP_KERNEL_INFO;
  }
  for (auto attr_info_iter : attrs_info_) {
    if (attr_info_iter->GetAttrName() == attr_name) {
      dtype = attr_info_iter->GetAttrDType();
      return SUCCESS;
    }
  }
  return OP_ATTR_NOT_FOUND_IN_OP_KERNEL_INFO;
}

Status OpKernelInfo::GetAttrValueByAttrName(const std::string &attr_name,
                                                  std::vector<ge::GeAttrValue> &attr_value) const {
  for (auto attr_info_iter : attrs_info_) {
    if (attr_info_iter->GetAttrName() == attr_name) {
      attr_value = attr_info_iter->GetSupportedAttrValueVector();
      return SUCCESS;
    }
  }
  return OP_ATTR_NOT_FOUND_IN_OP_KERNEL_INFO;
}

const std::vector<AttrInfoPtr>& OpKernelInfo::GetVecAttrInfo() const { return attrs_info_; }

const ge::OpInfo& OpKernelInfo::GetOpInfo() const { return op_info_; }

bool OpKernelInfo::IsDynamicCompileStatic() const {
  return GetDynamicCompileStatic() != DynamicCompileStatic::NOT_SUPPORT;
}

bool OpKernelInfo::IsSupportDynamicRank() const {
  return GetDynamicRankType() != DynamicRankType::NOT_SUPPORT;
}

bool OpKernelInfo::IsOpFileNull() const {
  return (op_info_.opFileName == kNullOpFile || op_info_.opFileName == kAicoreNull);
}

void OpKernelInfo::SetOpImpPath(const std::string &op_imp_path) {
  op_str_param_vec_[static_cast<size_t>(OP_KERNEL_STR_PARAM::OpImpPath)] = op_imp_path;
}

const std::vector<std::string>& OpKernelInfo::GetReferTensorNameVec() const { return refer_tensor_name_vec_; }

void OpKernelInfo::UpdatePrebuildPattern(const std::string &prebuild_pattern) {
  op_str_param_vec_[static_cast<size_t>(OP_KERNEL_STR_PARAM::PrebuildPattern)] = prebuild_pattern;
}

bool OpKernelInfo::IsMultiKernelSupport() const {
  return (GetMultiKernelSupportType() == MultiKernelSupportType::MULTI_SUPPORT);
}

void OpKernelInfo::SetPromoteInfo(const PromoteTypeVal &promote_val) {
  promote_type_ = promote_val;
}

void OpKernelInfo::GetPromoteInfo(PromoteTypeVal &promote_val) const {
  promote_val = promote_type_;
}

const FallbackFlags &OpKernelInfo::GetFallbackFlags() const {
  return fallback_flags_;
}

void OpKernelInfo::SetOutputIplaceInfo(std::vector<std::vector<int64_t>> &output_inplace_ability) {
  output_inplace_ability_ = output_inplace_ability;
}

std::vector<std::vector<int64_t>> OpKernelInfo::GetOutputIplaceInfo() const {
  return output_inplace_ability_;
}
}  // namespace fe
