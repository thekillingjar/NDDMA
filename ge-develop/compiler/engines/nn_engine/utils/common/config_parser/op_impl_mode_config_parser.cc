/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config_parser/op_impl_mode_config_parser.h"
#include <fstream>
#include <sstream>
#include "common/fe_log.h"
#include "common/string_utils.h"
#include "common/comm_error_codes.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "common/fe_report_error.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"

namespace fe {
namespace {
constexpr char const *kByNodeName = "[ByNodeName]";
constexpr char const *kByOpType = "[ByOpType]";
constexpr char const *kConfigFileSuffix = ".ini";
constexpr char const *kOpImplModeRelativePath = "built-in/op_impl/ai_core/tbe/impl_mode/";
constexpr char const *kOpSelectImplModeDefaultValue = "high_performance";
constexpr char const *kAllowHF32AllTrueForAtc = "true";
constexpr char const *kAllowHF32AllFalseForAtc = "false";
constexpr char const *kAllowHF32AllTrue = "1";
constexpr char const *kAllowHF32AllFalse = "0";
constexpr char const *kAllowHF32MatmulFConvF = "00";  // matmul(bit0):false,conv(bit1):false
constexpr char const *kAllowHF32MatmulTConvF = "01";  // matmul(bit0):true,conv(bit1):false
constexpr char const *kAllowHF32DefaultMode = "10";   // matmul(bit0):false,conv(bit1):true
constexpr char const *kAllowHF32MatmulTConvT = "11";  // matmul(bit0):true,conv(bit1):true
const std::vector<std::string> kOpSelectImplModeVec = {"high_precision", "high_performance"};
const std::vector<std::string> kOpSelectImplModeAllVec = {"high_precision_for_all", "high_performance_for_all"};
const std::unordered_map<std::string, std::string> kAllowHF32ModeMap = {
        {kAllowHF32MatmulFConvF, "allow_hf32_matmul_f_conv_f"},
        {kAllowHF32MatmulTConvF, "allow_hf32_matmul_t_conv_f"},
        {kAllowHF32DefaultMode, "allow_hf32_matmul_f_conv_t"},
        {kAllowHF32MatmulTConvT, "allow_hf32_matmul_t_conv_t"},
        {kAllowHF32AllTrue, "allow_hf32_matmul_t_conv_t"},
        {kAllowHF32AllFalse, "allow_hf32_matmul_f_conv_f"},
        {kAllowHF32AllTrueForAtc, "allow_hf32_matmul_t_conv_t"},
        {kAllowHF32AllFalseForAtc, "allow_hf32_matmul_f_conv_f"}
};
const std::map<std::string, std::string> kAllowHF32ForAclnnFallbackMap = {
        {kAllowHF32AllTrue, "11"},
        {kAllowHF32AllFalse, "00"},
        {kAllowHF32AllTrueForAtc, "11"},
        {kAllowHF32AllFalseForAtc, "00"}
};
const std::set<std::string> kSupportImpyType = {
    "high_performance", "enable_float_32_execution", "enable_hi_float_32_execution",
    "high_precision", "support_out_of_bound_index", "super_performance", "norm_class", "keep_fp16"};
}  // namespace
OpImplModeConfigParser::OpImplModeConfigParser(const std::string &ascend_opp_path)
  : BaseConfigParser(), ascend_opp_path_(ascend_opp_path) {}

OpImplModeConfigParser::~OpImplModeConfigParser() {}

Status OpImplModeConfigParser::InitializeFromOptions(const std::map<std::string, std::string> &options) {
  std::string op_precision_mode;
  std::map<std::string, std::string>::const_iterator iter = options.find(ge::OP_PRECISION_MODE);
  if (iter != options.cend() && !iter->second.empty()) {
    op_precision_mode = iter->second;
  }
  std::string op_select_impl_mode;
  iter = options.find(ge::OP_SELECT_IMPL_MODE);
  if (iter != options.cend() && !iter->second.empty()) {
    op_select_impl_mode = iter->second;
  }
  std::string op_type_list_str;
  iter = options.find(ge::OPTYPELIST_FOR_IMPLMODE);
  if (iter != options.cend() && !iter->second.empty()) {
    op_type_list_str = iter->second;
    (void)StringUtils::Trim(op_type_list_str);
  }
  std::string allow_hf32;
  iter = options.find(ge::ALLOW_HF32);
  if (iter != options.end() && !iter->second.empty()) {
    allow_hf32 = iter->second;
    FE_LOGI("[Init][AllowHF32] Parameter ge.exec.allow_hf32 is set to %s.", allow_hf32.c_str());
  }
  UpDateDefaultValue(op_precision_mode, op_select_impl_mode, allow_hf32);
  return Initialize(op_precision_mode, op_select_impl_mode, op_type_list_str, allow_hf32);
}

Status OpImplModeConfigParser::InitializeFromContext() {
  std::string op_precision_mode;
  (void)ge::GetContext().GetOption(ge::OP_PRECISION_MODE, op_precision_mode);
  std::string op_select_impl_mode;
  (void)ge::GetContext().GetOption(ge::OP_SELECT_IMPL_MODE, op_select_impl_mode);
  std::string op_type_list_str;
  (void)ge::GetContext().GetOption(ge::OPTYPELIST_FOR_IMPLMODE, op_type_list_str);
  std::string allow_hf32;
  (void)ge::GetContext().GetOption(ge::ALLOW_HF32, allow_hf32);
  FE_LOGD("[Refresh][AllowHF32] Parameter [ge.exec.allow_hf32] is set to [%s].", allow_hf32.c_str());

  UpDateDefaultValue(op_precision_mode, op_select_impl_mode, allow_hf32);
  return Initialize(op_precision_mode, op_select_impl_mode, op_type_list_str, allow_hf32);
}

/*
 * priority: allow_hf32 > op_precision_mode > op_select_impl_mode
 * To maintain compatibility, allow_hf32 is not mutual exclusion with others.
 * so, we need to consider default value of allow_hf32
 */
void OpImplModeConfigParser::UpDateDefaultValue(const std::string &op_precision_mode,
                                                std::string &op_select_impl_mode,
                                                std::string &allow_hf32) {
  if (op_select_impl_mode.empty()) {
    FE_LOGD("The value of param[%s] is empty. Using default value[%s].",
            ge::OP_SELECT_IMPL_MODE.c_str(), kOpSelectImplModeDefaultValue);
    op_select_impl_mode = kOpSelectImplModeDefaultValue;
    if (op_precision_mode.empty() && allow_hf32.empty()) {
      FE_LOGD("[Update][AllowHF32] Parameter ge.exec.allow_hf32 uses the default value %s.",
              kAllowHF32DefaultMode);
      allow_hf32 = kAllowHF32DefaultMode;
    }
  }
}

// PRECISIONMODE
Status OpImplModeConfigParser::Initialize(const std::string &op_precision_mode,
                                          const std::string &op_select_impl_mode,
                                          const std::string &op_type_list_for_impl_mode,
                                          const std::string &allow_hf32) {
  bool enable_allow_hf32 = PlatformUtils::Instance().IsEnableAllowHF32();
  std::string tmp_allow_hf32;
  if (enable_allow_hf32) {
    tmp_allow_hf32 = allow_hf32.empty() ? kAllowHF32DefaultMode : allow_hf32;
  }
  std::lock_guard<std::mutex> lock_guard(op_impl_mode_mutex_);
  bool not_change = op_precision_mode == op_precision_mode_ && op_select_impl_mode == op_select_impl_mode_ &&
                    op_type_list_for_impl_mode == op_type_list_for_impl_mode_ && tmp_allow_hf32 == allow_hf32_;
  if (not_change) {
    FE_LOGD("The parameters of op_impl_mode are the same as last time.");
    return SUCCESS;
  }
  op_precision_mode_ = op_precision_mode;
  op_select_impl_mode_ = op_select_impl_mode;
  op_type_list_for_impl_mode_ = op_type_list_for_impl_mode;
  allow_hf32_ = tmp_allow_hf32;
  op_name_select_impl_mode_map_.clear();
  op_type_select_impl_mode_map_.clear();
  if (enable_allow_hf32 && !allow_hf32.empty()) {
    if (InitAllowHF32Mode(allow_hf32) != SUCCESS) {
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {allow_hf32, ge::ALLOW_HF32,
                                 "The current value is not within the valid range"});
      ReportErrorMessage(err_msg);
      REPORT_FE_ERROR(
          "[GraphOpt][Init][InitAllowHF32] ge.exec.allow_hf32[%s] is invalid, only support [0,00,01,10,11,1].",
          allow_hf32.c_str());
      return FAILED;
    }
  }
  Status status = InitOpPrecisionMode(op_precision_mode, op_select_impl_mode, op_type_list_for_impl_mode);
  if (status != SUCCESS) {
    return status;
  }
  if (enable_allow_hf32 && allow_hf32.empty()) {
    if (InitAllowHF32Mode(kAllowHF32DefaultMode) != SUCCESS) {
      REPORT_FE_ERROR("[Init][allow_hf32] Init default allow_hf32 mode failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitOpPrecisionMode(const std::string &op_precision_mode,
                                                   const std::string &op_select_impl_mode,
                                                   const std::string &op_type_list_str) {
  // 1. Check op precision mode
  if (!op_precision_mode.empty()) {
    FE_LOGD("The value of parameter [op_precision_mode] is [%s].", op_precision_mode.c_str());
    return InitOpPrecisionModeByPrecisionMode(op_precision_mode);
  }
  FE_LOGD("The parameter [op_precision_mode] is not found or its value is empty.");

  if (op_select_impl_mode.empty()) {
    FE_LOGD("The parameter [op_select_impl_mode] is not found or its value is empty.");
    return SUCCESS;
  }
  FE_LOGD("The value of parameter[op_select_impl_mode] is [%s].", op_select_impl_mode.c_str());
  if (std::find(kOpSelectImplModeAllVec.begin(), kOpSelectImplModeAllVec.end(), op_select_impl_mode) !=
      kOpSelectImplModeAllVec.end()) {
    return InitOpPrecisionModeByImplModeAll(op_select_impl_mode);
  }
  if (std::find(kOpSelectImplModeVec.begin(), kOpSelectImplModeVec.end(), op_select_impl_mode) !=
      kOpSelectImplModeVec.end()) {
    if (!op_type_list_str.empty()) {
      FE_LOGD("The value of parameter [optypelist_for_implmode] is [%s].", op_type_list_str.c_str());
    }
    return InitOpPrecisionModeByImplMode(op_select_impl_mode, op_type_list_str);
  }

  FE_LOGE("[GraphOpt][Init][InitOpPrecisionMode] Para:op_select_impl_mode[%s] is invalid.",
          op_select_impl_mode.c_str());
  ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {op_select_impl_mode, ge::OP_SELECT_IMPL_MODE,
                             "The current value is not within the valid range"});
  ReportErrorMessage(err_msg);
  return FAILED;
}

Status OpImplModeConfigParser::InitOpPrecisionModeByPrecisionMode(const std::string &op_precision_mode) {
  if (op_precision_mode.empty()) {
    return SUCCESS;
  }
  // check whether file is existed
  std::string file_path = GetRealPath(op_precision_mode);
  if (file_path.empty()) {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
                               {op_precision_mode, ge::OP_PRECISION_MODE,
                                "The file does not exist or its access permission is denied"});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[GraphOpt][Init][InitOpPrecisionMode] The op precision mode configuration file [%s] does not exist.",
                    op_precision_mode.c_str());
    return FAILED;
  }
  Status ret = GetOpPrecisonModeStrFromConfigFile(file_path);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][InitOpPrecisionMode] Failed to retrieve op_precision_mode string from file [%s].",
                    op_precision_mode.c_str());
    return ret;
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitOpPrecisionModeByImplModeAll(const std::string &op_select_impl_mode_all) {
  std::string file_path = ascend_opp_path_ + kOpImplModeRelativePath + op_select_impl_mode_all + kConfigFileSuffix;
  std::string real_file_path = GetRealPath(file_path);
  if (real_file_path.empty()) {
    FE_LOGW("[%s] file [%s] does not exist.", op_select_impl_mode_all.c_str(), file_path.c_str());
    return SUCCESS;
  }
  Status ret = GetOpPrecisonModeStrFromConfigFile(real_file_path);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][InitOpPrecisionMode] Failed to retrieve op_precision_mode string from file [%s].",
                    op_select_impl_mode_all.c_str());
    return ret;
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitOpPrecisionModeByImplMode(const std::string &op_select_impl_mode,
                                                             const std::string &op_type_list_str) {
  if (!op_type_list_str.empty()) {
    std::vector<std::string> op_type_list = StringUtils::Split(op_type_list_str, ',');
    for (std::string &op_type : op_type_list) {
      op_type = StringUtils::Trim(op_type);
      op_type_select_impl_mode_map_.emplace(op_type, op_select_impl_mode);
    }
  } else {
    string file_path = ascend_opp_path_ + kOpImplModeRelativePath + op_select_impl_mode + kConfigFileSuffix;
    std::string real_file_path = GetRealPath(file_path);
    if (real_file_path.empty()) {
      FE_LOGW("[%s] file [%s] does not exist.", op_select_impl_mode.c_str(), file_path.c_str());
      return SUCCESS;
    }
    Status status = GetOpPrecisonModeStrFromConfigFile(real_file_path);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Init][InitOpPrecisionMode] Failed to retrieve op_precision_mode string from file [%s].",
                      op_select_impl_mode.c_str());
      return status;
    }
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitAllowHF32Mode(const std::string &allow_hf32) {
  auto iter = kAllowHF32ModeMap.find(allow_hf32);
  if (iter == kAllowHF32ModeMap.end()) {
    return FAILED;
  }
  std::string allow_hf32_mode = iter->second;
  std::string file_path = ascend_opp_path_ + kOpImplModeRelativePath + allow_hf32_mode + kConfigFileSuffix;
  std::string real_file_path = GetRealPath(file_path);
  if (real_file_path.empty()) {
    FE_LOGW("Allow hf32 file [%s] does not exist.", file_path.c_str());
    return SUCCESS;
  }
  Status status = GetOpPrecisonModeStrFromConfigFile(real_file_path);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][InitAllowHF32Mode] Failed to retrieve allow_hf32 string from [%s] file.",
                    allow_hf32_mode.c_str());
    return status;
  }
  return SUCCESS;
}
bool OpImplModeConfigParser::CheckConfigImplType(const std::string &impl_mode) const {
  std::stringstream ss;
  for (auto impl : kSupportImpyType) {
    ss << impl << "|";
  }
  ss.flush();
  auto iter = kSupportImpyType.find(impl_mode);
  if (iter == kSupportImpyType.end()) {
    FE_LOGW("impl mode %s is not in current support list: %s.", impl_mode.c_str(), ss.str().c_str());
    return false;
  }
  return true;
}

void OpImplModeConfigParser::ParseLineContentWithMode(const std::string &line_content, bool parse_by_op_type,
                                                      const size_t &pos_of_equal) {
  std::string op_type_or_name = line_content.substr(0, pos_of_equal);
  std::string impl_mode = line_content.substr(pos_of_equal + 1);
  op_type_or_name = StringUtils::Trim(op_type_or_name);
  impl_mode = StringUtils::Trim(impl_mode);
  if (op_type_or_name.empty() || impl_mode.empty()) {
    FE_LOGW("in op_precision_mode config file current line %s optype or opname or op_impl_mode is empty",
            line_content.c_str());
    return;
  }
  if (!CheckConfigImplType(impl_mode)) {
    FE_LOGW("Op_type_or_name %s with op_impl_mode %s is invalid according to the configuration.", op_type_or_name.c_str(),
            impl_mode.c_str());
    return;
  }
  if (parse_by_op_type) {
    op_type_select_impl_mode_map_.emplace(op_type_or_name, impl_mode);
  } else {
    op_name_select_impl_mode_map_.emplace(op_type_or_name, impl_mode);
  }
  return;
}

Status OpImplModeConfigParser::GetOpPrecisonModeStrFromConfigFile(const std::string &file_path) {
  FE_LOGD("Begin to load op select implementation mode file [%s].", file_path.c_str());
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {file_path});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[GraphOpt][InitOpPrecisionMode] Failed to open config file [%s].", file_path.c_str());
    return INVALID_FILE_PATH;
  }

  std::string line;
  bool parse_by_op_type = true;
  while (std::getline(ifs, line)) {
    if (line.empty() || line.find('#') == 0) {
      continue;
    }
    size_t pos_of_equal = line.find('=');
    if (pos_of_equal == std::string::npos) {
      std::string line_tmp = StringUtils::Trim(line);
      if (line_tmp == kByNodeName) {
        parse_by_op_type = false;
      } else if (line_tmp == kByOpType) {
        parse_by_op_type = true;
      }
      continue;
    }
    ParseLineContentWithMode(line, parse_by_op_type, pos_of_equal);
  }
  ifs.close();
  FE_LOGD("Finish parsing select implementation mode file [%s].", file_path.c_str());
  return SUCCESS;
}

bool OpImplModeConfigParser::GetOpImplModeByOpType(const std::string &op_type, std::string &op_impl_mode) const {
  std::lock_guard<std::mutex> lock_guard(op_impl_mode_mutex_);
  auto iter = op_type_select_impl_mode_map_.find(op_type);
  if (iter == op_type_select_impl_mode_map_.cend()) {
    return false;
  }
  op_impl_mode = iter->second;
  return true;
}

bool OpImplModeConfigParser::GetOpImplModeByOpName(const std::string &op_name, std::string &op_impl_mode) const {
  std::lock_guard<std::mutex> lock_guard(op_impl_mode_mutex_);
  auto iter = op_name_select_impl_mode_map_.find(op_name);
  if (iter == op_name_select_impl_mode_map_.cend()) {
    return false;
  }
  op_impl_mode = iter->second;
  return true;
}

bool OpImplModeConfigParser::GetOpImplMode(const std::string &op_name, const std::string &op_type,
                                           std::string &op_impl_mode) const {
  if (GetOpImplModeByOpName(op_name, op_impl_mode)) {
    return true;
  }
  return GetOpImplModeByOpType(op_type, op_impl_mode);
}

bool OpImplModeConfigParser::IsEnableCustomImplMode() const {
  return !op_precision_mode_.empty();
}

std::string OpImplModeConfigParser::EmplaceHf32ModeForAclnn(const std::string &hf32_mode) const {
  const auto it_hf32 = kAllowHF32ForAclnnFallbackMap.find(hf32_mode);
  if (it_hf32 != kAllowHF32ForAclnnFallbackMap.cend()) {
    FE_LOGD("Origin hf32 mode %s, new hf32 mode %s", hf32_mode.c_str(), it_hf32->second.c_str());
    return it_hf32->second;
  }
  return hf32_mode;
}
}
