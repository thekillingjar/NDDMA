/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/base/utils/process_utils.h"
#include <unistd.h>
#include <cstdlib>
#include <regex>
#include "mmpa/mmpa_api.h"
#include "common/debug/ge_log.h"
#include "common/debug/log.h"
#include "common/checker.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr char_t kValidChars[] = "[A-Za-z\\d/_.-]+";
constexpr char_t kInvalidChars[] = "[^A-Za-z\\d/_.-]+";
}
pid_t ProcessUtils::Fork() {
  return fork();
}

int32_t ProcessUtils::Execute(const std::string &path, char *const *argv) {
  (void)path;
  (void)argv;
  char *argv_stub[] = {const_cast<char*>("echo"),
    const_cast<char*>("stub exec"), NULL};
  return execvp("echo", argv_stub);
}

Status ProcessUtils::System(const std::string &cmd, bool print_err) {
  return SUCCESS;
}

Status ProcessUtils::GetIpaddr(const std::string &name, std::string &ipaddr) {
  ipaddr = "127.0.0.1";
  return SUCCESS;
}

pid_t ProcessUtils::WaitPid(pid_t pid, int *wstatus, int options) {
  return mmWaitPid(pid, wstatus, options);
}

Status ProcessUtils::CreateDir(const std::string &directory_path) {
  return SUCCESS;
}

Status ProcessUtils::CreateDir(const std::string &directory_path, uint32_t mode) {
  GE_ASSERT_TRUE((ge::CreateDir(directory_path, mode) == EOK), "Create direct failed, path: %s.",
                 directory_path.c_str());
  return SUCCESS;
}

Status ProcessUtils::IsValidPath(const std::string &path) {
  GE_CHK_BOOL_RET_STATUS((path.find("..") == std::string::npos),
                         FAILED,
                         "File path[%s] is invalid, include relative path.",
                         path.c_str());
  std::regex e(kValidChars);
  std::smatch sm;
  GE_CHK_BOOL_RET_STATUS(std::regex_match(path, sm, e), FAILED,
                         "Path[%s] is invalid, please use a-z, A-Z, 0-9, _, - and /",
                         path.c_str());
  return SUCCESS;
}

std::string ProcessUtils::NormalizePath(const string &path) {
  std::regex e(kInvalidChars);
  return std::regex_replace(path, e, "_");
}
}  // namespace ge
