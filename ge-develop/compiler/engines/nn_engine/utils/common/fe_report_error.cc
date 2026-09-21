/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fe_report_error.h"
#include <sys/time.h>
#include <algorithm>
#include <cstring>
#include "common/fe_log.h"
#include "common/fe_error_code.h"
#include "graph/ge_context.h"
#include "base/err_msg.h"

namespace fe {
std::map<std::string, std::vector<std::string>> kFeErrorParamMap = {
  {EM_COMPLIE_FAILED, {"pass_name", "pass_type"}},
  {EM_ENVIRONMENT_VARIABLE_FAILED, {"value", "env", "reason"}},
  {EM_INVALID_CONTENT, {"parameter", "filepath", "reason"}},
  {EM_RUN_PASS_FAILED, {"pass_name", "pass_type"}},
  {EM_INPUT_OPTION_INVALID, {"value", "parameter", "reason"}},
  {EM_AICORENUM_OUT_OF_RANGE, {"value", "parameter", "range"}},
  {EM_OPEN_FILE_FAILED, {"file_name"}},
  {EM_READ_FILE_FAILED, {"file_name", "reason"}},
};

void ErrorMessageDetail::ModifyArgsByErrorCode() {
  if (error_code != EM_INPUT_OPTION_INVALID && error_code != EM_AICORENUM_OUT_OF_RANGE) {
    return;
  }
  auto param_list = kFeErrorParamMap[EM_INPUT_OPTION_INVALID];
  auto iter = std::find(param_list.begin(), param_list.end(), "parameter");
  size_t index = iter - param_list.begin();
  if (arg_list.size() <= index) {
    return;
  }
  std::string arg = ge::GetContext().GetReadableName(arg_list[index]);
  if (arg.empty()) {
    FE_LOGW("Get readable name by option [%s] resulted in an empty value", arg_list[index].c_str());
    return;
  }
  FE_LOGD("Original name: %s, readable name: %s, for error code: %s.", arg_list[index].c_str(), arg.c_str(),
          error_code.c_str());
  arg_list[index] = arg;
}

void ErrorMessageDetail::ToParamMap(std::map<std::string, std::string> &args_map) {
  auto iter = kFeErrorParamMap.find(error_code);
  if (iter == kFeErrorParamMap.end()) {
    FE_LOGW("Code %s is not configured in error_code.json", error_code.c_str());
    return;
  }
  size_t arg_size = (iter->second.size() > arg_list.size()) ? arg_list.size() : iter->second.size();
  
  for (size_t i = 0; i < arg_size; ++i) {
    args_map[iter->second[i]] = arg_list[i];
    FE_LOGD("error_code: %s, add key: %s, args: %s to args_map.", error_code.c_str(), iter->second[i].c_str(),
            arg_list[i].c_str());
  }
  return;
}

void ReportErrorMessage(ErrorMessageDetail &error_detail) {
  error_detail.ModifyArgsByErrorCode();
  std::map<std::string, std::string> args_map;
  error_detail.ToParamMap(args_map);
  std::vector<const char*> args_keys;
  std::vector<const char*> args_values;
  for (const auto &item : args_map) {
    args_keys.push_back(item.first.c_str());
    args_values.push_back(item.second.c_str());
  }
  int result = REPORT_PREDEFINED_ERR_MSG(error_detail.GetErrorCode().c_str(), args_keys, args_values);
  if (result != 0) {
    FE_LOGE("Faild to call ReportErrMessage.");
  }
  return;
}
}  // namespace fe
