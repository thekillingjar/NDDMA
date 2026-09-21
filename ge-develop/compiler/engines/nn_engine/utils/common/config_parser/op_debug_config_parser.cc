/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config_parser/op_debug_config_parser.h"
#include <fstream>
#include <sstream>
#include <string>
#include "common/fe_log.h"
#include "common/string_utils.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "common/fe_report_error.h"
#include "ge/ge_api_types.h"
#include "common/aicore_util_constants.h"
#include "framework/common/ge_types.h"
#include "graph/ge_context.h"


namespace fe {
namespace {
constexpr char const *kMemoryCheckKey = "oom";
const int64_t OFFSET_2 = 2;
};
OpDebugConfigParser::OpDebugConfigParser() : BaseConfigParser() {}
OpDebugConfigParser::OpDebugConfigParser(const string &npu_collect_path) : BaseConfigParser() {
  npu_collect_path_ = npu_collect_path;
}
OpDebugConfigParser::~OpDebugConfigParser() {}
Status OpDebugConfigParser::InitializeFromOptions(const std::map<std::string, std::string> &options)
{
  std::map<std::string, std::string>::const_iterator iter = options.find(kOpDebugConfig);
  if (iter == options.end() || iter->second.empty()) {
    FE_LOGD("Parameter[op_debug_config] is not found in options.");
    return SUCCESS;
  }

  string file_path = iter->second;
  std::string real_path = GetRealPath(file_path);
  if (real_path.empty()) {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {file_path, kOpDebugConfig,
                               "The file does not exist or its access permission is denied"});
    ReportErrorMessage(err_msg);
    FE_LOGE("op_debug_configs real_path is null.");
    return FAILED;
  }
  FE_LOGI("The real path of config is %s.", real_path.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {iter->second, kOpDebugConfig,
                               "The file does not exist or its access permission is denied"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[Configuration][GetConfigValueByKey] Config file %s does not exist.", iter->second.c_str());
    return FAILED;
  }

  std::string line;
  std::string op_debug_list{};
  while (std::getline(ifs, line)) {
    line = StringUtils::Trim(line);
    if (line.empty() || line.find('#') == 0) {
      continue;
    }
    if (line.find(kOpDebugConfig) != string::npos) {
      op_debug_config_ = line;
      FE_LOGD("Config line value is %s.", line.c_str());
    } else if (line.find(kOpDebugList) != string::npos) {
      op_debug_list = line;
      FE_LOGD("Configuration list is %s.", line.c_str());
    }
  }
  ifs.close();
  bool set_op_config_status = SetOpdebugConfig(file_path);
  if (set_op_config_status) {
    (void)SetOpDebugList(op_debug_list, file_path);
  }
  return SUCCESS;
}

Status OpDebugConfigParser::InitializeFromContext() {
  string op_debug_option;
  if (ge::GetContext().GetOption(ge::OP_DEBUG_OPTION, op_debug_option) != ge::GRAPH_SUCCESS) {
    FE_LOGD("The value of op_debug_option is not found or the value is empty from context.");
    return SUCCESS;
  }
  if ((!npu_collect_path_.empty() || op_debug_config_.empty())) {
    std::vector<std::string> op_debug_option_list = StringUtils::Split(op_debug_option, ",");
    AssembleOpDebugConfigInfo(op_debug_option_list);
  }
  return SUCCESS;
}

bool OpDebugConfigParser::GetOpdebugValue(const std::string &line, std::vector<std::string> &res,
                                          const std::string &file_path)
{
  size_t pos_of_equal = line.find('=');
  if (pos_of_equal == string::npos) {
    ErrorMessageDetail err_msg(EM_INVALID_CONTENT,
        {kOpDebugConfig, file_path, "Line:\"" + line + "\" not contain \"=\"."});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[Configuration][ParseOpDebugConfig]Config [%s] format is error.", line.c_str());
    return false;
  }
  std::string value = line.substr(pos_of_equal + 1);
  if (value.empty()) {
    ErrorMessageDetail err_msg(EM_INVALID_CONTENT,
        {kOpDebugConfig, file_path, "Line:\"" + line + "\", value is empty."});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[Configuration][ParseOpDebugConfig]Config value [%s] is empty.", line.c_str());
    return false;
  }
  res = StringUtils::Split(value, ',');
  return true;
}

void OpDebugConfigParser::AssembleOpDebugConfigInfo(std::vector<std::string> &value_vec) {
  bool comma = false;
  std::stringstream ss;
  for (auto &it : value_vec) {
    it = StringUtils::Trim(it);
    if (it.empty()) {
      continue;
    }
    if (it == kMemoryCheckKey) {
      enable_op_memory_check_ = true;
    }
    if (comma) {
      ss << ",";
    }
    ss << it;
    comma = true;
  }
  op_debug_config_.clear();
  op_debug_config_ = ss.str();
}

void OpDebugConfigParser::AssembleOpDebugListInfo(std::vector<std::string> &value_list_vec)
{
  for (auto &it : value_list_vec) {
    it = StringUtils::Trim(it);
    if (it.empty()) {
      continue;
    }
    if (it.find("::") != string::npos) {
      op_debug_list_type_.insert(it.substr(it.find("::") + OFFSET_2));
    } else {
      op_debug_list_name_.insert(it);
    }
  }
}

bool OpDebugConfigParser::SetOpDebugList(const std::string &op_debug_list, const std::string &file_path)
{
  if (op_debug_list.empty()) {
    FE_LOGD("without op_debug_list");
    return true;
  }
  std::vector<std::string> op_debug_list_vec;
  bool op_debug_list_status = GetOpdebugValue(op_debug_list, op_debug_list_vec, file_path);
  if (op_debug_list_status) {
    AssembleOpDebugListInfo(op_debug_list_vec);
  }
  return true;
}

bool OpDebugConfigParser::SetOpdebugConfig(const std::string &file_path)
{
  if (op_debug_config_.empty()) {
    FE_LOGD("with out op_debug_config");
    return false;
  }
  std::vector<std::string> op_debug_config_vec;
  bool get_op_bebug_status = GetOpdebugValue(op_debug_config_, op_debug_config_vec, file_path);
  if (get_op_bebug_status) {
    AssembleOpDebugConfigInfo(op_debug_config_vec);
  }
  return get_op_bebug_status;
}

std::string OpDebugConfigParser::GetOpDebugConfig() const
{
  return op_debug_config_;
}

bool OpDebugConfigParser::IsNeedMemoryCheck() const
{
  return enable_op_memory_check_;
}

void OpDebugConfigParser::SetOpDebugConfigEnv(const std::string &env)
{
  if (op_debug_config_.empty() && !env.empty()) {
      op_debug_config_ = "dump_bin,dump_cce,dump_loc,ccec_g";
      FE_LOGD("The op_debug_config is empty, but env[%s] is not empty; setting op_debug_config_ value to [%s].",
              env.c_str(), op_debug_config_.c_str());
  }
}

bool OpDebugConfigParser::IsOpDebugListOp(const ge::OpDescPtr &op_desc_ptr) const {
  string op_type = op_desc_ptr->GetType().c_str();
  string op_name = op_desc_ptr->GetName().c_str();
  if (!op_debug_config_.empty() && op_debug_list_name_.empty() && op_debug_list_type_.empty()) {
    FE_LOGD("Tbe op_debug_config is null. Setting all ops to use op_debug_compile.");
    return true;
  }
  if (op_debug_list_name_.count(op_name) || op_debug_list_type_.count(op_type)) {
    return true;
  }
  return false;
}
}
