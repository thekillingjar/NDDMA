/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_DFLOW_BASE_UTILS_PROCESS_UTILS_H_
#define AIR_DFLOW_BASE_UTILS_PROCESS_UTILS_H_

#include <sys/types.h>
#include "mmpa/mmpa_api.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class ProcessUtils {
 public:
  ProcessUtils() = delete;
  ~ProcessUtils() = delete;

  static pid_t Fork();
  static int32_t Execute(const std::string &path, char_t *const argv[]);
  static Status System(const std::string &cmd, bool print_err = true);
  static Status GetIpaddr(const std::string &name, std::string &ipaddr);
  static pid_t WaitPid(pid_t pid, int32_t *wstatus, int options);
  static Status CreateDir(const std::string &directory_path);
  static Status CreateDir(const std::string &directory_path, uint32_t mode);
  static Status IsValidPath(const std::string &path);
  static std::string NormalizePath(const std::string &path);

 private:
  static Status DoMkdir(const char_t *tmp_dir_path, mmMode_t mode);
};
}  // namespace ge

#endif  // AIR_DFLOW_BASE_UTILS_PROCESS_UTILS_H_
