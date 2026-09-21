/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/common/format_dtype_selector_base.h"

namespace fe {
FormatDtypeSelectorBase::FormatDtypeSelectorBase() {}
FormatDtypeSelectorBase::~FormatDtypeSelectorBase() {}

Status FormatDtypeSelectorBase::GetAllFormatsFromOpDesc(const ge::OpDescPtr &op_desc,
                                                        map<string, vector<ge::Format>> &result) {
  result = op_desc->TryGetExtAttr(EXT_DYNAMIC_FORMAT, result);
  if (result.empty()) {
    FE_LOGD("Op[name=%s,type=%s]: the %s attribute is empty.", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
            EXT_DYNAMIC_FORMAT.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FormatDtypeSelectorBase::GetAllFallbackFromOpDesc(const ge::OpDescPtr &op_desc,
                                                         vector<bool> &fallback_flags) {
  fallback_flags = op_desc->TryGetExtAttr(EXT_DYNAMIC_FALLBACK, fallback_flags);
  if (fallback_flags.empty()) {
    FE_LOGD("Op[name=%s,type=%s]: the %s attribute is empty.", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
            EXT_DYNAMIC_FALLBACK.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FormatDtypeSelectorBase::GetAllSubFormatsFromOpDesc(const ge::OpDescPtr &op_desc,
                                                           map<string, vector<uint32_t>> &result) {
  result = op_desc->TryGetExtAttr(EXT_DYNAMIC_SUB_FORMAT, result);
  if (result.empty()) {
    FE_LOGD("Op[name=%s,type=%s]: the %s attribute is empty.", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
            EXT_DYNAMIC_SUB_FORMAT.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FormatDtypeSelectorBase::GetAllDataTypesFromOpDesc(const ge::OpDescPtr &op_desc,
                                                          map<string, vector<ge::DataType>> &result) {
  result = op_desc->TryGetExtAttr(EXT_DYNAMIC_DATATYPE, result);
  if (result.empty()) {
    FE_LOGD("Op[name=%s,type=%s]: the %s attribute is empty.", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
            EXT_DYNAMIC_DATATYPE.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FormatDtypeSelectorBase::GetFormatFromOpDescByKey(const ge::OpDescPtr &op_desc, const string &key,
                                                         vector<ge::Format> &result) {
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();

  // 1. get all formats from the op_desc
  map<string, vector<ge::Format>> format_map;
  if (GetAllFormatsFromOpDesc(op_desc, format_map) != SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: GetAllFormatsFromOpDesc by the key [%s] unsuccessful.", op_name.c_str(), op_type.c_str(),
            key.c_str());
    return FAILED;
  }

  // 2. find the key from format_map
  if (format_map.find(key) == format_map.end()) {
    FE_LOGD("Op[name=%s,type=%s]: find the formats from the formatMap by the key [%s] unsuccessful.", op_name.c_str(),
            op_type.c_str(), key.c_str());
    return FAILED;
  }
  result = format_map[key];
  return SUCCESS;
}

Status FormatDtypeSelectorBase::GetSubFormatFromOpDescByKey(const ge::OpDescPtr &op_desc, const string &key,
                                                            vector<uint32_t> &result) {
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();

  // 1. get all sub_formats from the op_desc
  map<string, vector<uint32_t>> sub_format_map;
  if (GetAllSubFormatsFromOpDesc(op_desc, sub_format_map) != SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: GetAllSubFormatsFromOpDesc by sub_format [%s] unsuccessful.", op_name.c_str(),
            op_type.c_str(), key.c_str());
    return FAILED;
  }

  // 2. find the key from format_map
  auto it = sub_format_map.find(key);
  if (it == sub_format_map.end()) {
    FE_LOGD("Op[name=%s,type=%s]: find the sub_formats from the subFormatMap by sub_format [%s] unsuccessful.",
            op_name.c_str(), op_type.c_str(), key.c_str());
    return FAILED;
  }
  result = it->second;
  return SUCCESS;
}

Status FormatDtypeSelectorBase::GetDataTypeFromOpDescByKey(const ge::OpDescPtr &op_desc, const string &key,
                                                           vector<ge::DataType> &result) {
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();

  // 1. get all data_types from the op_desc
  std::map<std::string, vector<ge::DataType>> data_type_map;
  if (GetAllDataTypesFromOpDesc(op_desc, data_type_map) != SUCCESS) {
    FE_LOGD("Op[name=%s,type=%s]: GetAllDataTypesFromOpDesc by the key [%s] unsuccessful.", op_name.c_str(), op_type.c_str(),
            key.c_str());
    return FAILED;
  }

  // 2. find the key from the data_type_map
  if (data_type_map.find(key) == data_type_map.end()) {
    FE_LOGD("Op[name=%s,type=%s]: find the data_types from the dataTypeMap by the key [%s] unsuccessful.", op_name.c_str(),
            op_type.c_str(), key.c_str());
    return FAILED;
  }
  result = data_type_map[key];
  return SUCCESS;
}

Status FormatDtypeSelectorBase::SetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                      const HeavyFormatInfo &heavy_format_info,
                                                      const ge::NodePtr &node,
                                                      const bool &is_dynamic_check) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  FormatDtypeInfo format_dtype_info;
  FE_LOGD("Op[name=%s,type=%s]: propagat_primary_format=%s, propagat_sub_format=%d.", op_name.c_str(), op_type.c_str(),
          ge::TypeUtils::FormatToSerialString(heavy_format_info.expected_heavy_format).c_str(),
          heavy_format_info.sub_format);

  if (GetDynamicFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, heavy_format_info,
                            format_dtype_info, static_cast<uint32_t>(GROUPS_DEFAULT_VALUE)) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SetSptFmtDtype] Op[name=%s,type=%s]: failed to GetDynamicFormatDtype.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  if (SaveDynamicFormatDtype(format_dtype_info, op_desc) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SetSptFmtDtype] Op[name=%s,type=%s]: failed to SaveFormatAndDtype.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status FormatDtypeSelectorBase::SaveDynamicFormatDtype(const FormatDtypeInfo &format_dtype_info,
                                                       const ge::OpDescPtr &op_desc) const {
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();

  if (!format_dtype_info.format_map.empty()) {
    FE_LOGI("Op[name=%s, type=%s]: The format_map is not empty; setting it to the attribute [%s].", op_name.c_str(),
            op_type.c_str(), EXT_DYNAMIC_FORMAT.c_str());
    if (!op_desc->SetExtAttr(EXT_DYNAMIC_FORMAT, format_dtype_info.format_map)) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][SvDymcFmtDty][name %s, type %s]: Failed to set the format_map to the attribute [%s].",
          op_name.c_str(), op_type.c_str(), EXT_DYNAMIC_FORMAT.c_str());
      return FAILED;
    }
  }

  if (!format_dtype_info.sub_format_map.empty()) {
    FE_LOGD("Op[name=%s, type=%s]: The sub_format_map is not empty; setting it to the attribute [%s].", op_name.c_str(),
            op_type.c_str(), EXT_DYNAMIC_SUB_FORMAT.c_str());
    if (!op_desc->SetExtAttr(EXT_DYNAMIC_SUB_FORMAT, format_dtype_info.sub_format_map)) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][SvDymcFmtDty][name %s, type %s]: Failed to set the sub_format_map to the attribute [%s].",
          op_name.c_str(), op_type.c_str(), EXT_DYNAMIC_SUB_FORMAT.c_str());
      return FAILED;
    }
  }

  if (!format_dtype_info.data_type_map.empty()) {
    FE_LOGI("Op[name=%s, type=%s]: The data_type_map is not empty; setting it to the attribute [%s].", op_name.c_str(),
            op_type.c_str(), EXT_DYNAMIC_DATATYPE.c_str());
    if (!op_desc->SetExtAttr(EXT_DYNAMIC_DATATYPE, format_dtype_info.data_type_map)) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][SvDymcFmtDty][name: %s, type: %s] Failed to set the data_type_map for attribute [%s].",
          op_name.c_str(), op_type.c_str(), EXT_DYNAMIC_DATATYPE.c_str());
      return FAILED;
    }
  }

  if (!format_dtype_info.fallback_flags.empty()) {
    FE_LOGI("Op[name=%s, type=%s]: fallback_flags is not empty; setting it to attribute [%s].", op_name.c_str(),
            op_type.c_str(), EXT_DYNAMIC_FALLBACK.c_str());
    if (!op_desc->SetExtAttr(EXT_DYNAMIC_FALLBACK, format_dtype_info.fallback_flags)) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][SvDymcFmtDty][name: %s, type: %s]: failed to set fallback_flag for attribute [%s].",
          op_name.c_str(), op_type.c_str(), EXT_DYNAMIC_FALLBACK.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

}  // namespace fe
