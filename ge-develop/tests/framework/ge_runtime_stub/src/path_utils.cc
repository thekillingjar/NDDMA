/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/path_utils.h"
#include <fstream>
#include "ge_common/string_util.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

#ifdef __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#endif

namespace ge {
std::string ge::PathUtils::Join(const std::vector<std::string> &names) {
  return StringUtils::Join(names.begin(), names.end(), "/");
}
bool PathUtils::CopyFile(const std::string &src, const std::string &dst) {
  std::ifstream src_file(src, std::ios::binary);
  std::ofstream dst_file(dst, std::ios::binary);
  dst_file << src_file.rdbuf();
}

// 使用run包里的so，下面符号会优先使用libmetadef.so里的，导致打桩失效，这里需要重新打桩编译到ut/st中
std::string RealPath(const char_t *path) {
  std::string res;
  char_t resolved_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(path, &(resolved_path[0U]), MMPA_MAX_PATH) == EN_OK) {
    res = &(resolved_path[0]);
  }
  return res;
}

int32_t CheckAndMkdir(const char_t *tmp_dir_path, mmMode_t mode) {
  if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
    const int32_t ret = mmMkdir(tmp_dir_path, mode);
    if (ret != 0 && errno != EEXIST) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Can not create directory %s. Make sure the directory "
                           "exists and writable. errmsg:%s",
                           tmp_dir_path, strerror(errno));
      GELOGW("[Util][mkdir] Create directory %s failed, reason:%s. Make sure the "
             "directory exists and writable.",
             tmp_dir_path, strerror(errno));
      return ret;
    }
  }
  return 0;
}

int32_t CreateDir(const std::string &directory_path, uint32_t mode) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(), REPORT_INNER_ERR_MSG("E18888", "directory path is empty, check invalid");
  return -1, "[Check][Param] directory path is empty.");
  const auto dir_path_len = directory_path.length();
  if (dir_path_len >= static_cast<size_t>(MMPA_MAX_PATH)) {
    REPORT_PREDEFINED_ERR_MSG(
        "E13002", std::vector<const char *>({"filepath", "size"}),
        std::vector<const char *>({directory_path.c_str(), std::to_string(MMPA_MAX_PATH).c_str()}));
    GELOGW("[Util][mkdir] Path %s len is too long, it must be less than %d", directory_path.c_str(), MMPA_MAX_PATH);
    return -1;
  }
  char_t tmp_dir_path[MMPA_MAX_PATH] = {};
  const auto mkdir_mode = static_cast<mmMode_t>(mode);
  for (size_t i = 0U; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      const int32_t ret = CheckAndMkdir(&(tmp_dir_path[0U]), mkdir_mode);
      if (ret != 0) {
        return ret;
      }
    }
  }
  return CheckAndMkdir(directory_path.c_str(), mkdir_mode);
}

int32_t CreateDir(const std::string &directory_path) {
  constexpr auto mkdir_mode = static_cast<uint32_t>(M_IRUSR | M_IWUSR | M_IXUSR);
  return CreateDir(directory_path, mkdir_mode);
}

#ifdef __GNUC__
bool IsFile(const std::string &filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool IsDirectory(const std::string &file_folder) {
  struct stat buffer;
  return (stat(file_folder.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}
int64_t PathUtils::RemoveDirectories(const std::string &path) {
  int result = 0;
  DIR *p_dir;
  struct dirent *p_dirent;
  if (IsDirectory(path)) {
    if ((p_dir = opendir(path.c_str())) == nullptr) {
      return -1;
    }

    while ((p_dirent = readdir(p_dir)) != nullptr) {
      std::string file_name = path + "/" + p_dirent->d_name;
      // It is a directory
      if (IsDirectory(file_name) && (0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
        result = RemoveDirectories(file_name);
        if (result < 0) {
          return result;
        }
      }
      // It is a file
      else if ((0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
        result = remove(file_name.c_str());
        if (result < 0) {
          return result;
        }
      }
    }
    closedir(p_dir);
    result = rmdir(path.c_str());
  } else if (IsFile(path)) {
    result = remove(path.c_str());
  }
  return result;
}
#else
#error "not support yet"
#endif
}  // namespace ge
