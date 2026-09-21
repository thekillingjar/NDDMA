/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/op_customize/format_dtype_op_customize_selector.h"
#include "common/configuration.h"
#include "common/fe_type_utils.h"
#include "common/fe_op_info_common.h"
#include "adapter/common/op_store_adapter_manager.h"

namespace fe {
namespace {
uint32_t GetSubFormatFromStr(const std::string &str) {
  uint32_t val;
  try {
    val = std::stol(str);
  } catch (const std::exception &e) {
    FE_LOGE("[GetSubFormatFromStr] Sub_format: %s transformation failed, using default value.", str.c_str());
    return DEFAULT_SUB_FORMAT;
  }

  return val;
}

Status CheckJsonValid(const nlohmann::json &j, const string &input_or_output_key) {
  if (input_or_output_key.empty()) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][ChkJsonValid] The input_or_output_key is empty.");
    return FAILED;
  }
  if (!j[input_or_output_key].is_object()) {
    REPORT_FE_ERROR(
        "[GraphOpt][Setcheck][ChkJsonValid] The input_or_output is not an object; the input_or_output_key is %s.",
        input_or_output_key.c_str());
    return FAILED;
  }

  if (input_or_output_key == STR_FALLBACK) {
    if (j[input_or_output_key].find(STR_ENABLE) == j[input_or_output_key].end() &&
        j[input_or_output_key].find(STR_UNKNOWN_SHAPE_ENABLE) == j[input_or_output_key].end()) {
      REPORT_FE_ERROR(
          "[GraphOpt][Setcheck][ChkJsonValid] The enable or unknownshape_enable of fallback cannot be found.");
      return FAILED;
    }
    return SUCCESS;
  }

  if (j[input_or_output_key].find(STR_FORMAT) == j[input_or_output_key].end() &&
      j[input_or_output_key].find(STR_UNKNOWN_SHAPE_FORMAT) == j[input_or_output_key].end()) {
    REPORT_FE_ERROR(
        "[GraphOpt][Setcheck][ChkJsonValid] Cannot find the format or unknown_format for input_or_output_key %s.",
        input_or_output_key.c_str());
    return FAILED;
  }
  if (j[input_or_output_key].find(STR_DTYPE) == j[input_or_output_key].end()) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][ChkJsonValid] The data_type of input_or_output_key %s cannot be found.",
                    input_or_output_key.c_str());
    return FAILED;
  }
  if (j[input_or_output_key].find(STR_NAME) == j[input_or_output_key].end()) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][ChkJsonValid] The name of the input_or_output_key %s cannot be found.",
                    input_or_output_key.c_str());
    return FAILED;
  }
  return SUCCESS;
}
} // namespace

FormatDtypeOpCustomizeSelector::FormatDtypeOpCustomizeSelector(const std::string &engine_name)
  : FormatDtypeSelectorBase(), engine_name_(engine_name) {}

FormatDtypeOpCustomizeSelector::~FormatDtypeOpCustomizeSelector() {}

Status FormatDtypeOpCustomizeSelector::GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const ge::NodePtr &node,
                                               const bool &is_dynamic_check,
                                               map<string, vector<ge::Format>> &format_map) {
  Status status = GetAllFormatsFromOpDesc(node->GetOpDesc(), format_map);
  if (status != SUCCESS) {
    HeavyFormatInfo heavy_format_info;
    FormatDtypeInfo format_dtype_info;
    if (GetDynamicFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, heavy_format_info,
        format_dtype_info) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptedFmt] Op[name=%s, type=%s]: failed to GetDynamicFormatAndDtype.",
                      node->GetNamePtr(), node->GetTypePtr());
      return FAILED;
    }
    if (!format_dtype_info.format_map.empty()) {
      format_map = std::move(format_dtype_info.format_map);
    }
  }
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::GetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                             const ge::NodePtr &node,
                                                             const bool &is_dynamic_check,
                                                             FormatDtypeInfo &format_dtype_info) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  const string &op_name = op_desc->GetName();
  const string &op_type = op_desc->GetType();
  FE_LOGD("Start to obtain custom format and data type from operation [%s, %s].", op_name.c_str(), op_type.c_str());

  // 1. get all  formats and data_types from the op_desc
  Status get_format_status = GetAllFormatsFromOpDesc(op_desc, format_dtype_info.format_map);
  Status get_data_type_status = GetAllDataTypesFromOpDesc(op_desc, format_dtype_info.data_type_map);
  Status get_subformat_status = GetAllSubFormatsFromOpDesc(op_desc, format_dtype_info.sub_format_map);
  // 2. if failed, GetDynamicFormatDtype
  if (get_format_status != SUCCESS || get_data_type_status != SUCCESS || get_subformat_status != SUCCESS) {
    HeavyFormatInfo heavy_format_info;
    if (GetDynamicFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, heavy_format_info,
                              format_dtype_info) != SUCCESS) {
      FE_LOGW("[GraphOpt][Setcheck][GetSptFmtDty][op: %s, type: %s] Failed to retrieve dynamic format and data type.",
              op_name.c_str(), op_type.c_str());
      return FAILED;
    }
  }
  if (IsDebugLogLevel) {
    LogFormatMap(format_dtype_info.format_map);
    LogDataTypeMap(format_dtype_info.data_type_map);
    LogSubformatMap(format_dtype_info.sub_format_map);
  }
  FE_LOGD("Finish obtaining custom format and data type from op[%s, %s].", op_name.c_str(), op_type.c_str());
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                         const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                         const ge::NodePtr &node, vector<ge::Format> &result) {
  (void)op_kernel_info_ptr;
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  string unique_name = input_or_output_info_ptr->GetUniqueName();
  if (GetFormatFromOpDescByKey(op_desc, unique_name, result) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Setcheck][GetSptFmt][op %s, type %s]: failed to get supported formats for input or output %s.",
        op_name.c_str(), op_type.c_str(), unique_name.c_str());
    return FAILED;
  }

  FE_LOGD("Op[name=%s, type=%s]: supports formats [ %s ] for the input_or_output [ %s ].", op_name.c_str(),
          op_type.c_str(), GetStrByFormatVec(result).c_str(), unique_name.c_str());
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::GetSupportFormatSubFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                                 const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                                 const ge::NodePtr &node,
                                                                 vector<ge::Format> &format_res,
                                                                 vector<uint32_t> &sub_format_res,
                                                                 uint32_t sub_format) {
  (void)op_kernel_info_ptr;
  (void)sub_format;
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  string unique_name = input_or_output_info_ptr->GetUniqueName();
  if (GetFormatFromOpDescByKey(op_desc, unique_name, format_res) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Setcheck][GetSptFmt][op %s,type %s] get format from op_desc for the input_or_output [%s] failed.",
        op_name.c_str(), op_type.c_str(), unique_name.c_str());
    return FAILED;
  }
  if (GetSubFormatFromOpDescByKey(op_desc, unique_name, sub_format_res) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Setcheck][GetSptFmt] Node[%s, %s]: failed to obtain supported formats for input or output %s.",
        op_name.c_str(), op_type.c_str(), unique_name.c_str());
    return FAILED;
  }

  FE_LOGD("Op[name=%s, type=%s]: supports subformats [%s] for the input_or_output [%s].", op_name.c_str(),
          op_type.c_str(), GetStrBySubFormatVec(sub_format_res).c_str(), unique_name.c_str());
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::GetSupportDataTypes(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                           const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                           const ge::NodePtr &node, vector<ge::DataType> &result) {
  (void)op_kernel_info_ptr;
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  string unique_name = input_or_output_info_ptr->GetUniqueName();
  if (GetDataTypeFromOpDescByKey(op_desc, unique_name, result) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][DtypeJdg][GetDtype] Op[%s], type=%s: failed to GetSupportDataTypesFromOpDesc for [%s].",
                    op_name.c_str(), op_type.c_str(), unique_name.c_str());

    return FAILED;
  }

  FE_LOGD("Op[name=%s, type=%s]: the supported data types are [%s] for the [%s].", op_name.c_str(), op_type.c_str(),
          GetStrByDataTypeVec(result).c_str(), unique_name.c_str());
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::GetDynamicFormatDtype(
    const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node, const bool &is_dynamic_check,
    const HeavyFormatInfo &heavy_format_info, FormatDtypeInfo &format_dtype_info, uint32_t sub_format) {
  (void)sub_format;
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  const string &op_name = op_desc->GetName();
  const string &op_type = op_desc->GetType();
  FE_LOGD("Op[name=%s,type=%s]: start to get the dynamic format_map and dataTypeMap.", op_name.c_str(),
          op_type.c_str());

  // 1. get the op_store_adapter
  OpStoreAdapterPtr op_store_adapter = nullptr;
  OpImplType impl_type = op_kernel_info_ptr->GetOpStoreImplType();
  if (OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(impl_type, op_store_adapter) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Setcheck][GetDymcFmtDty][op %s,type %s]: fail to get op_store_adapter by op impl type[%ld].",
        op_name.c_str(), op_type.c_str(), impl_type);
    return FAILED;
  }

  // 2. call SelectOpFormat
  string op_format_dtype_str;
  if (op_store_adapter->SelectOpFormat(node, op_kernel_info_ptr, is_dynamic_check, heavy_format_info,
                                       op_format_dtype_str) != SUCCESS) {
    FE_LOGW("[GenFormat][SelectOpFormat][Op name=%s, type=%s]: Failed to select formats and data_types.",
            op_name.c_str(), op_type.c_str());
    return FAILED;
  }
  if (op_format_dtype_str.empty()) {
    FE_LOGW("[GraphOpt][Setcheck][GetDymcFmtDty][op %s, type %s] The op_format_dtype_str is empty.",
            op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  // 3. parse the op_format_dtype_str
  if (ParseJsonStr(node, op_format_dtype_str, is_dynamic_check, format_dtype_info) != SUCCESS) {
    REPORT_FE_ERROR("[GenFormat][ParseFmtJson][Op %s, type=%s]: failed to parse the op_format_dtype_str %s",
                    op_name.c_str(), op_type.c_str(), op_format_dtype_str.c_str());
    return FAILED;
  }

  FE_LOGD("Op[name=%s,type=%s]: end to get the dynamic format_map and data_type_map.", op_name.c_str(),
          op_type.c_str());
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::GetFallbackFlags(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                        const ge::NodePtr &node, bool is_dynamic_check,
                                                        std::vector<bool> &fallback_res) {
  (void)op_kernel_info_ptr;
  (void)is_dynamic_check;
  Status status = GetAllFallbackFromOpDesc(node->GetOpDesc(), fallback_res);
  if (status != SUCCESS) {
    FE_LOGW("[GraphOpt][FmtJdg][GetSptedFmt] Op[name=%s, type=%s]: failed to obtain fallback flags from select_format result.",
            node->GetNamePtr(), node->GetTypePtr());
  }
  return SUCCESS;
}

/**
 * {
 *   "input0":{
 *       "name": "x",
 *       "format": "NCHW",
 *       "dtype": "float"
 *   },
 *   "output0":{
 *       "name": "y",
 *       "format": "NCHW",
 *       "dtype": "float"
 *   }
 * }
 */
Status FormatDtypeOpCustomizeSelector::ParseFallbackJsonInfo(const ge::NodePtr &node, bool is_dynamic_check,
    const nlohmann::json &fallback_json, FormatDtypeInfo &format_dtype_info) const {
  string enable_str;
  if (is_dynamic_check && fallback_json.find(STR_UNKNOWN_SHAPE_ENABLE) != fallback_json.end()) {
    enable_str = static_cast<string>(fallback_json.at(STR_UNKNOWN_SHAPE_ENABLE));
    FE_LOGD("Op[name:%s,type:%s] has unknown shape fallback config.",
            node->GetNamePtr(), node->GetTypePtr());
  } else if (fallback_json.find(STR_ENABLE) != fallback_json.end()) {
    enable_str = static_cast<string>(fallback_json.at(STR_ENABLE));
    FE_LOGD("Op[name:%s,type:%s] has fallback config.",
            node->GetNamePtr(), node->GetTypePtr());
  } else {
    REPORT_FE_ERROR("[SelectFormat][ParseJsonStr] parse json fallback failed, 'enable' is required key!");
    return FAILED;
  }
  vector<string> fallback_str_vec;
  StringUtils::SplitWithTrim(enable_str, ',', fallback_str_vec);
  size_t fallback_size = fallback_str_vec.size();
  for (size_t i = 0; i < fallback_size; ++i) {
    bool fallback_flag;
    if (String2Bool(fallback_str_vec[i], fallback_flag) != SUCCESS) {
      REPORT_FE_ERROR("[SelectFormat][ParseJsonStr] String2Bool failed: %s. Fallback flag is invalid!",
                      fallback_str_vec[i].c_str());
      return FAILED;
    }
    format_dtype_info.fallback_flags.emplace_back(fallback_flag);
  }
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::ParseJsonStr(const ge::NodePtr &node, const string &json_str,
    const bool &is_dynamic_check, FormatDtypeInfo &format_dtype_info) {
  try {
    auto op_desc = node->GetOpDesc();
    const nlohmann::json &j = nlohmann::json::parse(json_str);
    if (!j.is_object()) {
      REPORT_FE_ERROR("[GraphOpt][Setcheck][PasJsonStr] The json_str %s is not an object.", json_str.c_str());
      return ILLEGAL_JSON;
    }
    uint32_t format_size_of_first_input_or_output = INVALID_DTYPE_AND_FORMAT_SIZE;
    for (auto &input_or_output : j.items()) {
      string input_or_output_key = input_or_output.key();
      if (CheckJsonValid(j, input_or_output_key) != SUCCESS) {
        return FAILED;
      }
      if (input_or_output_key == STR_FALLBACK) {
        if (ParseFallbackJsonInfo(node, is_dynamic_check, j[input_or_output_key], format_dtype_info) != SUCCESS) {
          return FAILED;
        }
        continue;
      }

      string format_vec_str;
      if (j[input_or_output_key].find(STR_FORMAT) != j[input_or_output_key].end()) {
        format_vec_str = static_cast<string>(j[input_or_output_key].at(STR_FORMAT));
      }
      string sub_format_vec_str;
      if (j[input_or_output_key].find(STR_SUB_FORMAT) != j[input_or_output_key].end()) {
        sub_format_vec_str = static_cast<string>(j[input_or_output_key].at(STR_SUB_FORMAT));
      }

      if (is_dynamic_check) {
        if (j[input_or_output_key].find(STR_UNKNOWN_SHAPE_FORMAT) != j[input_or_output_key].end()) {
          FE_LOGD("Unknown shape op[name:%s,type:%s] supports dynamic shape.", op_desc->GetName().c_str(),
                  op_desc->GetType().c_str());
          format_vec_str = static_cast<string>(j[input_or_output_key].at(STR_UNKNOWN_SHAPE_FORMAT));
        }
      }
      const string &data_type_vec_str = static_cast<string>(j[input_or_output_key].at(STR_DTYPE));
      const string &name_key = input_or_output_key + "." + static_cast<string>(j[input_or_output_key].at(STR_NAME));
      vector<ge::Format> format_vec;
      vector<uint32_t> sub_format_vec;
      vector<ge::DataType> data_type_vec;
      if (ConvertFormatDtype(format_vec_str, sub_format_vec_str, data_type_vec_str,
          format_size_of_first_input_or_output, format_vec, sub_format_vec, data_type_vec) != SUCCESS) {
        REPORT_FE_ERROR("[GenFormat][ParseFmtJson][Op %s], tensor_name %s: failed to convert from JSON [%s].",
                        op_desc->GetName().c_str(), name_key.c_str(), json_str.c_str());
        return FAILED;
      }
      format_dtype_info.format_map.insert(pair<string, vector<ge::Format>>(name_key, format_vec));
      format_dtype_info.sub_format_map.insert(pair<string, vector<uint32_t>>(name_key, sub_format_vec));
      format_dtype_info.data_type_map.insert(pair<string, vector<ge::DataType>>(name_key, data_type_vec));
    }
    return SUCCESS;
  } catch (std::exception &e) {
    REPORT_FE_ERROR("[GraphOpt][ParseFmtJson][Exception] The JSON string is %s and the error reason is %s", json_str.c_str(),
                    e.what());
    return FAILED;
  }
}

Status FormatDtypeOpCustomizeSelector::ConvertFormatDtype(const string &format_vec_str,
                                                          const string &sub_format_vec_str,
                                                          const string &data_type_vec_str,
                                                          uint32_t &format_size_of_first_input_or_output,
                                                          vector<ge::Format> &format_vec,
                                                          vector<uint32_t> &sub_format_vec,
                                                          vector<ge::DataType> &data_type_vec) {
  vector<string> format_str_vec = StringUtils::Split(format_vec_str, ",");
  vector<string> sub_format_str_vec = StringUtils::Split(sub_format_vec_str, ",");
  vector<string> data_type_str_vec = StringUtils::Split(data_type_vec_str, ",");

  // 1. check whether size of dtype is the same as size of format
  uint32_t format_size = format_str_vec.size();
  uint32_t sub_format_size = sub_format_str_vec.size();
  uint32_t dtype_size = data_type_str_vec.size();
  bool use_default_sub_format = sub_format_size == 0 ? true : false;
  if (dtype_size != format_size) {
    REPORT_FE_ERROR("[GraphOpt][SetFmt][ConvertFmt]: The format size [%u] and dtype size [%u] is not equal!",
                    format_size, dtype_size);
    return FAILED;
  }
  // 2. check the format size whether is same with other input_or_outputs
  if (format_size_of_first_input_or_output == INVALID_DTYPE_AND_FORMAT_SIZE) {
    format_size_of_first_input_or_output = format_size;
  }
  if (format_size != format_size_of_first_input_or_output) {
    REPORT_FE_ERROR("[GraphOpt][SetFmt][ConvertFmt]: format size %u is invalid; it should be within the range of %u.", format_size,
                    format_size_of_first_input_or_output);
    return FAILED;
  }

  // 3. get the ge::Format and ge::DataType
  for (uint32_t i = 0; i != format_size; i++) {
    string &format_str = format_str_vec[i];
    ge::Format format = ge::TypeUtils::SerialStringToFormat(StringUtils::Trim(format_str));
    if (format == ge::FORMAT_RESERVED) {
      REPORT_FE_ERROR("[GraphOpt][SetFmt][ConvertFmt]: Invalid format: %s cannot be converted to ge::Format enum.",
                      format_str.c_str());
      return FAILED;
    }

    ge::DataType dtype;
    string &dtype_str = data_type_str_vec[i];
    if (String2DataType(StringUtils::Trim(dtype_str), dtype) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][SetFmt][ConvertFmt]: Invalid data type: %s cannot be converted.", dtype_str.c_str());
      return FAILED;
    }
    format_vec.emplace_back(format);
    data_type_vec.emplace_back(dtype);
  }
  if (use_default_sub_format) {
    sub_format_vec.emplace_back(DEFAULT_SUB_FORMAT);
  } else {
    for (auto &sub_format_str : sub_format_str_vec) {
      sub_format_vec.emplace_back(GetSubFormatFromStr(sub_format_str));
    }
  }
  return SUCCESS;
}

Status FormatDtypeOpCustomizeSelector::UpdateDTypeAndFormat(
    set<uint32_t> &remain_index_set, std::map<std::string, vector<ge::Format>> &format_map,
    std::map<std::string, vector<uint32_t>> &sub_format_map,
    std::map<std::string, vector<ge::DataType>> &data_type_map) const {
  // after filter format, if remind_index_set is empty, do nothing.
  if (remain_index_set.empty()) {
    FE_LOGD("The remaining format is empty. The format remains unchanged.");
    return SUCCESS;
  }

  set<uint32_t>::iterator index_iter;
  std::map<std::string, vector<ge::Format>>::iterator format_iter;
  for (format_iter = format_map.begin(); format_iter != format_map.end(); ++format_iter) {
    if (format_iter->second.empty()) {
      continue;
    }
    vector<ge::Format> remain_format_vec;
    for (index_iter = remain_index_set.begin(); index_iter != remain_index_set.end(); ++index_iter) {
      if (*index_iter < format_iter->second.size()) {
        remain_format_vec.push_back(format_iter->second[*index_iter]);
      }
    }
    format_iter->second.swap(remain_format_vec);
  }

  std::map<std::string, vector<uint32_t>>::iterator sub_format_iter;
  for (sub_format_iter = sub_format_map.begin(); sub_format_iter != sub_format_map.end(); ++sub_format_iter) {
    if (sub_format_iter->second.empty()) {
      continue;
    }
    vector<uint32_t> remain_sub_format_vec;
    for (index_iter = remain_index_set.begin(); index_iter != remain_index_set.end(); ++index_iter) {
      if (*index_iter < sub_format_iter->second.size()) {
        remain_sub_format_vec.emplace_back(sub_format_iter->second[*index_iter]);
      }
    }
    sub_format_iter->second.swap(remain_sub_format_vec);
  }

  std::map<std::string, vector<ge::DataType>>::iterator date_type_iter;
  for (date_type_iter = data_type_map.begin(); date_type_iter != data_type_map.end(); ++date_type_iter) {
    if (date_type_iter->second.empty()) {
      continue;
    }
    vector<ge::DataType> remain_data_type_vec;
    for (index_iter = remain_index_set.begin(); index_iter != remain_index_set.end(); ++index_iter) {
      if (*index_iter < date_type_iter->second.size()) {
        remain_data_type_vec.push_back(date_type_iter->second[*index_iter]);
      }
    }
    date_type_iter->second.swap(remain_data_type_vec);
  }
  return SUCCESS;
}
}  // namespace fe
