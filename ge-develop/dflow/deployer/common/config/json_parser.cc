/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "json_parser.h"
#include <string>
#include <fstream>
#include <exception>
#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

namespace ge {
namespace {
constexpr size_t kMaxErrorStringLen = 128U;
}
bool JsonParser::CheckFilePath(const std::string &file_path) {
  char_t trusted_path[MMPA_MAX_PATH] = {'\0'};
  int32_t ret = mmRealPath(file_path.c_str(), trusted_path, sizeof(trusted_path));
  if (ret != EN_OK) {
    GELOGE(FAILED, "[Check][Path]The file path %s is not like a realpath,mmRealPath return %d, errcode is %d",
           file_path.c_str(), ret, mmGetErrorCode());
    char_t err_buf[kMaxErrorStringLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    REPORT_PREDEFINED_ERR_MSG("E13000", std::vector<const char_t *>({"path", "errmsg"}),
                       std::vector<const char_t *>({file_path.c_str(), reason.c_str()}));
    return false;
  }

  mmStat_t stat = {};
  ret = mmStatGet(trusted_path, &stat);
  if (ret != EN_OK) {
    GELOGE(FAILED,
           "[Check][File]Cannot get config file status,which path is %s, maybe does not exist, return %d, errcode %d",
           trusted_path, ret, mmGetErrorCode());
    char_t err_buf[kMaxErrorStringLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({trusted_path, reason.c_str()}));
    return false;
  }
  if ((stat.st_mode & S_IFMT) != S_IFREG) {
    GELOGE(FAILED, "[Check][File]Config file is not a common file,which path is %s, mode is %u", trusted_path,
           stat.st_mode);
    REPORT_PREDEFINED_ERR_MSG("E13006", std::vector<const char *>({"file", "current_type", "expect_type"}),
                              std::vector<const char *>({trusted_path, std::to_string(stat.st_mode).c_str(),
                                                         std::to_string(S_IFREG).c_str()}));
    return false;
  }
  return true;
}

Status JsonParser::ReadConfigFile(const std::string &file_path, nlohmann::json &js) {
  GE_CHK_BOOL_RET_STATUS(CheckFilePath(file_path), ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Path]Invalid config file path[%s]", file_path.c_str());

  std::ifstream fin(file_path);
  if (!fin.is_open()) {
    GELOGE(FAILED, "[Read][File]Read file %s failed", file_path.c_str());
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char_t *>({"file", "errmsg"}),
                       std::vector<const char_t *>({file_path.c_str(), "Open file failed"}));
    return FAILED;
  }

  try {
    fin >> js;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][File]Invalid json file, exception:%s", e.what());
    REPORT_PREDEFINED_ERR_MSG("WF0001", std::vector<const char_t *>({"filepath"}),
                       std::vector<const char_t *>({file_path.c_str()}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GELOGI("Parse json from file [%s] successfully", file_path.c_str());
  return SUCCESS;
}
}  // namespace ge
