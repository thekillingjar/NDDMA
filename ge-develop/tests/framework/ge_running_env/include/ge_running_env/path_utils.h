/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_PATH_UTILS_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_PATH_UTILS_H_
#include "fake_ns.h"
#include <string>
#include <sstream>
#include <fstream>
#include "mmpa/mmpa_api.h"

FAKE_NS_BEGIN
inline bool IsDir(const char *path) {
  return mmIsDir(path) == EN_OK;
}
inline bool IsFile(const char *path) {
  std::ifstream f(path);
  return f.good();
}
inline void RemoveFile(const char *path) {
  remove(path);
}
inline std::string BaseName(const char *path) {
  std::stringstream ss;
  for (size_t i = strlen(path); i > 0; --i) {
    if (path[i-1] == '/') {
      break;
    } else {
      ss << path[i-1];
    }
  }
  auto reverse_name = ss.str();
  return std::string{reverse_name.rbegin(), reverse_name.rend()};
}
inline std::string DirName(const char *path) {
  size_t i;
  for (i = strlen(path); i > 0; --i) {
    if (path[i-1] == '/') {
      break;
    }
  }
  return {path, i};
}
inline std::string PathJoin(const char *path1, const char *path2) {
  std::stringstream ss;
  ss << path1 << '/' << path2;
  return ss.str();
}
inline int Mkdir(const char *path) {
  if (!IsDir(path)) {
    auto ret = mmMkdir(path,
                       S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IXOTH); // 775
    if (ret != EN_OK) {
      return -1;
    }
  }
  return 0;
}
inline const std::string &GetRunPath() {
  static std::string path;
  if (!path.empty()) {
    return path;
  }
  char buf[2048] = {0};
  if (readlink("/proc/self/exe", buf, sizeof(buf) - 1) < 0) {
    return path;
  }
  path = PathJoin(DirName(buf).c_str(), "st_run_data");
  if (Mkdir(path.c_str()) != 0) {
    path = "";
  }
  return path;
}

const std::string GetAirPath();

FAKE_NS_END
#endif //AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_PATH_UTILS_H_
