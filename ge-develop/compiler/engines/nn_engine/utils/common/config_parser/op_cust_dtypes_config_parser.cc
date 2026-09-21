/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config_parser/op_cust_dtypes_config_parser.h"
#include <fstream>
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/string_utils.h"
#include "common/fe_report_error.h"
#include "ge/ge_api_types.h"

namespace fe {
namespace {
const size_t kHeadSize = 8;
const size_t kInputDtypeStrSize = 11;
const size_t kOutputDtypeStrSize = 12;
const uint32_t kEachLineMaxSize = 1024;
constexpr char const *kHeadOpType = "OpType::";
constexpr char const *kInputDType = "InputDtype";
constexpr char const *kOutputDType = "OutputDtype";
const char kSplitPattern = ':';
const char kCommasPattern = ',';
}

OpCustDtypesConfigParser::OpCustDtypesConfigParser() : BaseConfigParser() {}
OpCustDtypesConfigParser::~OpCustDtypesConfigParser() {}

Status OpCustDtypesConfigParser::InitializeFromOptions(const std::map<string, string> &options) {
  const std::map<string, string>::const_iterator iter = options.find(ge::CUSTOMIZE_DTYPES);
  if (iter == options.end() || iter->second.empty()) {
    FE_LOGD("The value of param[ge.customizeDtypes] is not configured in ge.option.");
    return SUCCESS;
  }
  FE_LOGD("The value of param[ge.customizedTypes] is [%s].", iter->second.c_str());
  if (cust_dtypes_path_ == iter->second) {
    FE_LOGD("The path for the custom dtypes file is the same as it was last time.");
    return SUCCESS;
  }
  cust_dtypes_path_ = iter->second;
  return ParseCustomDtypeContent();
}

Status OpCustDtypesConfigParser::InitializeFromContext(const std::string &combined_params_key) {
  if (combined_params_key.empty()) {
    FE_LOGD("The modified mixlist file path is empty.");
    return SUCCESS;
  }
  FE_LOGD("The value of param[ge.customizedTypes] is [%s]", combined_params_key.c_str());
  if (cust_dtypes_path_ == combined_params_key) {
    FE_LOGD("The custom dtypes file path is the same as it was last time.");
    return SUCCESS;
  }
  cust_dtypes_path_ = combined_params_key;
  return ParseCustomDtypeContent();
}

Status OpCustDtypesConfigParser::ParseCustomDtypeContent() {
  string real_file_path = GetRealPath(cust_dtypes_path_);
  if (real_file_path.empty()) {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
        {cust_dtypes_path_, ge::CUSTOMIZE_DTYPES,
        "The customize file does not exist or its access permission is denied"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[OpCustDtypesConfigParser][Parse] The customize file path[%s] is invalid.", cust_dtypes_path_.c_str());
    return FAILED;
  }

  if (!ParseFileContent(real_file_path)) {
    FE_LOGW("[OpCustDtypesConfigParser][Parse] Invalid file path [%s].", real_file_path.c_str());
    return FAILED;
  }
  FE_LOGD("[OpCustDtypesConfigParser][Parse] Parse optimize file[%s] successfully.", real_file_path.c_str());
  return SUCCESS;
}

bool OpCustDtypesConfigParser::ParseFileContent(const string &custom_file_path) {
  std::ifstream in_file_stream;
  in_file_stream.open(custom_file_path);
  if (!in_file_stream.is_open()) {
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {custom_file_path});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[OpCustDtypesConfigParser][Parse] File [%s] doesn't exist or has been opened.",
                    custom_file_path.c_str());
    return false;
  }
  char *line_c_str = new (std::nothrow) char[kEachLineMaxSize];
  if (line_c_str == nullptr) {
    REPORT_FE_ERROR("[OpCustDtypesConfigParser][Parse] line_c_str is a nullptr.");
    in_file_stream.close();
    return false;
  }

  if (memset_s(line_c_str, kEachLineMaxSize, 0, kEachLineMaxSize) != EOK) {
    REPORT_FE_ERROR("[OpCustDtypesConfigParser][Parse] Initialize lineStr buffer failed.");
    in_file_stream.close();
    delete[] line_c_str;
    return false;
  }

  bool is_type;
  while (in_file_stream.getline(line_c_str, kEachLineMaxSize)) {
    string line_str = line_c_str;
    is_type = false;
    if (line_str.find(kHeadOpType) == 0) {
      line_str = line_str.substr(kHeadSize, line_str.size() - kHeadSize + 1);
      is_type = true;
    }
    (void)StringUtils::Trim(line_str);
    if (!ParseDTypeFromLine(line_str, is_type)) {
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
                                 {custom_file_path, ge::CUSTOMIZE_DTYPES,
                                  "The content format of the customization file is incorrect"});
      ReportErrorMessage(err_msg);
      FE_LOGE("[OpCustDtypesConfigParser][Parse] Parse line content[%s] failed, break.", line_str.c_str());
      in_file_stream.close();
      delete[] line_c_str;
      return false;
    }
  }
  in_file_stream.close();
  delete[] line_c_str;
  return true;
}

bool OpCustDtypesConfigParser::ParseDTypeFromLine(const string &line_str, const bool &is_type) {
  string name_or_type = "";
  std::vector<string> input_dtype;
  std::vector<string> output_dtype;
  bool bres_0 = SplitNameOrType(line_str, name_or_type);
  bool bres_1 = SplitInoutDtype(line_str, input_dtype, output_dtype);
  bool bres_2 = FeedOpCustomizeDtype(name_or_type, is_type, input_dtype, output_dtype);
  if (!(bres_0 && bres_1 && bres_2)) {
    FE_LOGW("[OpCustDtypesConfigParser][Parse] Parse line_content[%s] failed.", line_str.c_str());
    return false;
  }
  return true;
}

bool OpCustDtypesConfigParser::SplitNameOrType(const string &line_str, string &name_or_type) const {
  size_t pos = line_str.find(kSplitPattern);
  if (pos == string::npos || pos == 0) {
    FE_LOGW("[Configuration][Parse] Invalid line content: [%s].", line_str.c_str());
    return false;
  }
  name_or_type = line_str.substr(0, pos);
  FE_LOGD("[Configuration][Parse] Parsing op_name_or_type[%s].", name_or_type.c_str());
  return true;
}

bool OpCustDtypesConfigParser::SplitInoutDtype(const string &line_str, std::vector<string> &input_dtype,
                                               std::vector<string> &output_dtype) const {
  size_t pos_i = line_str.find(kInputDType);
  size_t pos_o = line_str.find(kOutputDType);
  string input_str = "";
  string output_str = "";
  bool has_input = (pos_i != string::npos);
  bool has_output = (pos_o != string::npos);
  size_t str_len = line_str.size();
  if (has_input && has_output) {
    if (pos_i > pos_o) {
      REPORT_FE_ERROR("[Configuration][Parse] Invalid customize file, inputDtype must on the left of outputDtype.");
      return false;
    }
    input_str = line_str.substr(pos_i + kInputDtypeStrSize, pos_o - pos_i - kInputDtypeStrSize);
    output_str = line_str.substr(pos_o + kOutputDtypeStrSize, str_len - pos_i - kOutputDtypeStrSize);
  } else if (has_input && !has_output) {
    input_str = line_str.substr(pos_i + kInputDtypeStrSize, str_len - pos_i - kInputDtypeStrSize);
  } else if (has_output && !has_input) {
    output_str = line_str.substr(pos_o + kOutputDtypeStrSize, str_len - pos_i - kOutputDtypeStrSize);
  } else {
    FE_LOGW("[Configuration][Parse] Cannot find input or output keywords on line content[%s].", line_str.c_str());
    return false;
  }
  input_dtype = StringUtils::Split(input_str, kCommasPattern);
  output_dtype = StringUtils::Split(output_str, kCommasPattern);
  return true;
}

bool OpCustDtypesConfigParser::FeedOpCustomizeDtype(const string &op_type, const bool &is_type,
                                                    const std::vector<string> &input_dtype,
                                                    const std::vector<string> &output_dtype) {
  OpCustomizeDtype custom_dtype_vec;
  ge::DataType ge_dtype;
  for (auto dtype_str : input_dtype) {
    (void)StringUtils::Trim(dtype_str);
    if (dtype_str.empty()) {
      dtype_str = "DT_UNDEFINED";
      custom_dtype_vec.input_dtypes.emplace_back(ge::DataType::DT_UNDEFINED);
    } else {
      if (!CheckValidAndTrans(dtype_str, ge_dtype)) {
        return false;
      }
      custom_dtype_vec.input_dtypes.emplace_back(ge_dtype);
    }
    FE_LOGD("[Configuration][Parse] Set input_type to %s.", dtype_str.c_str());
  }

  for (auto dtype_str : output_dtype) {
    (void)StringUtils::Trim(dtype_str);
    if (dtype_str.empty()) {
      dtype_str = "DT_UNDEFINED";
      custom_dtype_vec.output_dtypes.emplace_back(ge::DataType::DT_UNDEFINED);
    } else {
      if (!CheckValidAndTrans(dtype_str, ge_dtype)) {
        return false;
      }
      custom_dtype_vec.output_dtypes.emplace_back(ge_dtype);
    }
    FE_LOGD("[Configuration][Parse] Set output_type[%s].", dtype_str.c_str());
  }
  CheckAndSetCustomizeDtype(is_type, op_type, custom_dtype_vec);
  FE_LOGD("[Configuration][Parse] Node[%s]: finished feeding cust_dtypes_vec.", op_type.c_str());
  return true;
}

bool OpCustDtypesConfigParser::CheckValidAndTrans(string &dtype_str, ge::DataType &ge_dtype) const {
  string str_tmp = dtype_str;
  std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
  dtype_str = "DT_" + dtype_str;
  ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  if (ge_dtype == ge::DataType::DT_UNDEFINED) {
    ErrorMessageDetail err_msg(EM_INVALID_CONTENT,
                               {str_tmp, ge::CUSTOMIZE_DTYPES, "" + str_tmp + " is an invalid dtype"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[Configuration][Parse] Invalid custom dtype[%s].", str_tmp.c_str());
    return false;
  }
  return true;
}

void OpCustDtypesConfigParser::CheckAndSetCustomizeDtype(const bool &is_type, const string &op_type,
                                                         const OpCustomizeDtype &custom_dtype_vec) {
  if (is_type) {
    if (op_type_cust_dtypes_.find(op_type) != op_type_cust_dtypes_.end()) {
      FE_LOGW("[Configuration][Parse] Repeated attempt to set custom data type for opType[%s] will not be available.",
              op_type.c_str());
      return;
    }
    op_type_cust_dtypes_.insert(std::make_pair(op_type, custom_dtype_vec));
  } else {
    if (op_name_cust_dtypes_.find(op_type) != op_name_cust_dtypes_.end()) {
      FE_LOGW("[Configuration][Parse] Repeated setting of custom data type for opName[%s] is ignored and will not be available.",
              op_type.c_str());
      return;
    }
    op_name_cust_dtypes_.insert(std::make_pair(op_type, custom_dtype_vec));
  }
  return;
}

bool OpCustDtypesConfigParser::GetCustomizeDtypeByOpType(const string &op_type, OpCustomizeDtype &custom_dtype) const {
  std::map<string, OpCustomizeDtype>::const_iterator iter = op_type_cust_dtypes_.find(op_type);
  if (iter != op_type_cust_dtypes_.end()) {
    custom_dtype = iter->second;
    return true;
  }
  FE_LOGD("[OpCustDtypesConfigParser][Parse] Node[%s]: customize_dtype has not been found.", op_type.c_str());
  return false;
}

bool OpCustDtypesConfigParser::GetCustomizeDtypeByOpName(const string &op_name, OpCustomizeDtype &custom_dtype) const {
  std::map<string, OpCustomizeDtype>::const_iterator iter = op_name_cust_dtypes_.find(op_name);
  if (iter != op_name_cust_dtypes_.end()) {
    custom_dtype = iter->second;
    return true;
  }
  FE_LOGD("[OpCustDtypesConfigParser][Parse] Node[%s]: customize_dtype has not been found.", op_name.c_str());
  return false;
}
}
