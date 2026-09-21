/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_store/ops_kernel_utils.h"
#include <fstream>
#include <sstream>
#include "common/fe_log.h"
#include "common/string_utils.h"
#include "common/fe_type_utils.h"
#include "common/fe_error_code.h"
#include "common/fe_report_error.h"
#include "common/aicore_util_constants.h"
#include "ops_store/ops_kernel_constants.h"
#include "base/err_msg.h"

namespace fe {
static const int BASE = 10;

Status ReadJsonObject(const std::string &file, nlohmann::json &json_obj) {
  std::string real_path = GetRealPath(file);
  if (real_path.empty()) {
    FE_LOGW("file path %s is invalid", file.c_str());
    ErrorMessageDetail error_msg(EM_OPEN_FILE_FAILED, {file});
    ReportErrorMessage(error_msg);
    return FAILED;
  }
  std::ifstream ifs(real_path);
  try {
    if (!ifs.is_open()) {
      FE_LOGW("Opening %s failed, file is already in use.", file.c_str());
      ErrorMessageDetail error_msg(EM_OPEN_FILE_FAILED, {file});
      ReportErrorMessage(error_msg);
      return FAILED;
    }
    ifs >> json_obj;
    ifs.close();
  } catch (const std::exception &er) {
    FE_LOGW("Unable to convert file [%s] to JSON. Message: %s", real_path.c_str(), er.what());
    ifs.close();
    ErrorMessageDetail error_msg(EM_READ_FILE_FAILED, {file, er.what()});
    ReportErrorMessage(error_msg);
    return FAILED;
  }

  return SUCCESS;
}

std::string GetJsonObjectType(const nlohmann::json &json_object) {
  std::string json_to_type;

  switch (json_object.type()) {
    case nlohmann::json::value_t::null:
      json_to_type = "null";
      break;
    case nlohmann::json::value_t::object:
      json_to_type = "object";
      break;
    case nlohmann::json::value_t::array:
      json_to_type = "array";
      break;
    case nlohmann::json::value_t::string:
      json_to_type = "string";
      break;
    case nlohmann::json::value_t::boolean:
      json_to_type = "boolean";
      break;
    case nlohmann::json::value_t::number_integer:
      json_to_type = "number_integer";
      break;
    case nlohmann::json::value_t::number_unsigned:
      json_to_type = "number_unsigned";
      break;
    case nlohmann::json::value_t::number_float:
      json_to_type = "number_float";
      break;
    case nlohmann::json::value_t::discarded:
      json_to_type = "discarded";
      break;
    default:
      break;
  }

  return json_to_type;
}

bool CmpInputsNum(std::string input1, std::string input2) {
  size_t strlen = std::strlen(STR_INPUT_LOWERCASE);
  return std::strtol(input1.substr(strlen).c_str(), nullptr, BASE) <
         std::strtol(input2.substr(strlen).c_str(), nullptr, BASE);
}

bool CmpOutputsNum(std::string output1, std::string output2) {
  size_t strlen = std::strlen(STR_OUTPUT_LOWERCASE);
  return std::strtol(output1.substr(strlen).c_str(), nullptr, BASE) <
         std::strtol(output2.substr(strlen).c_str(), nullptr, BASE);
}
bool CheckInputSubStr(const std::string& op_desc_input_name, const std::string& info_input_name) {
  auto length_of_info_input_to_name = static_cast<uint32_t>(info_input_name.length());
  auto length_of_op_desc_input_name = static_cast<uint32_t>(op_desc_input_name.length());
  if (length_of_info_input_to_name > length_of_op_desc_input_name) {
    return false;
  } else {
    /* LengthOfInfoInputName less than length_of_op_desc_input_name */
    if (op_desc_input_name.substr(0, length_of_info_input_to_name) == info_input_name) {
      /* Get from the first char after "infoInputName"
       * to the end of op_desc_input_name */
      std::string result = op_desc_input_name.substr(length_of_info_input_to_name);
      if (result.empty()) {
        return true;
      }
      if (StringUtils::IsInteger(result)) {
        return true;
      } else {
        /* In other cases, we consider this input name of op_desc is illegal.
         * Digits should only appears at the end of name
         * as index. */
        FE_LOGW("Illegal input name [%s] in opdesc during comparison with inputname [%s].",
                op_desc_input_name.c_str(), info_input_name.c_str());
        return false;
      }
    } else {
      return false;
    }
  }
}

Status GenerateUnionFormatAndDtype(const vector<ge::Format>& old_formats, const vector<ge::DataType>& old_data_types,
                                   vector<ge::Format>& new_formats, vector<ge::DataType>& new_data_types) {
  size_t old_formats_size = old_formats.size();
  size_t old_dtypes_size = old_data_types.size();
  if (old_formats.empty() || old_data_types.empty()) {
    FE_LOGI("The old_formats_size [%zu] is 0, or the old_dtypes_size [%zu] is 0.", old_formats_size, old_dtypes_size);
    if (!old_formats.empty()) {
      new_formats = old_formats;
    }
    if (!old_data_types.empty()) {
      new_data_types = old_data_types;
    }
    return SUCCESS;
  }

  for (size_t i = 0; i < old_formats_size; i++) {
    new_formats.insert(new_formats.cend(), old_dtypes_size, old_formats[i]);
    new_data_types.insert(new_data_types.cend(), old_data_types.cbegin(), old_data_types.cend());
  }
  size_t new_formats_size = new_formats.size();
  size_t new_dtypes_size = new_data_types.size();
  if (new_formats_size != new_dtypes_size) {
    REPORT_FE_ERROR("[Init][InitOpsKernel] The new format size (%zu) is not equal to the new dtype size (%zu)",
                    new_formats_size, new_dtypes_size);
    return FAILED;
  }
  return SUCCESS;
}

std::string StringVecToString(const std::vector<std::string> &str_vec) {
  std::ostringstream oss;
  for (size_t i = 0; i < str_vec.size(); i++) {
    oss << str_vec.at(i);
    if (i < str_vec.size() - 1) {
      oss << ", ";
    }
  }
  return oss.str();
}
}  // namespace fe
