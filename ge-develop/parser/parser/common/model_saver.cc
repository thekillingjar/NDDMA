/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/model_saver.h"

#include "base/err_msg.h"

#include <sys/stat.h>
#include <fcntl.h>

#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "base/err_msg.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"

namespace {
const size_t kMaxErrStrLen = 128U;
const int kFileOpSuccess = 0;
}  //  namespace

namespace ge {
namespace parser {
const uint32_t kInteval = 2;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelSaver::SaveJsonToFile(const char *file_path,
                                                                                   const Json &model) {
  Status ret = SUCCESS;
  if ((file_path == nullptr) || (CheckPath(file_path) != SUCCESS)) {
    REPORT_INNER_ERR_MSG("E19999", "param file_path is nullptr or checkpath not return success");
    GELOGE(FAILED, "[Check][Param]Check output file failed.");
    return FAILED;
  }
  std::string model_str;
  try {
    model_str = model.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to convert JSON to string, reason: %s, savefile:%s.", e.what(), file_path);
    GELOGE(FAILED, "[Invoke][Dump] Failed to convert JSON to string, reason: %s, savefile:%s.", e.what(), file_path);
    return FAILED;
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to convert JSON to string, savefile:%s.", file_path);
    GELOGE(FAILED, "[Invoke][Dump] Failed to convert JSON to string, savefile:%s.", file_path);
    return FAILED;
  }

  char real_path[PATH_MAX] = {0};
  if (strlen(file_path) >= PATH_MAX) {
    REPORT_PREDEFINED_ERR_MSG("E13002", std::vector<const char *>({"filepath", "size"}),
                              std::vector<const char *>({file_path, std::to_string(PATH_MAX).c_str()}));
    GELOGE(FAILED, "[Check][Param] file path %s is too long!", file_path);
    return FAILED;
  }
  if (realpath(file_path, real_path) == nullptr) {
    GELOGI("File %s does not exit, it will be created.", file_path);
  }

  // Open file
  mode_t mode = S_IRUSR | S_IWUSR;
  int32_t fd = mmOpen2(real_path, O_RDWR | O_CREAT | O_TRUNC, mode);
  if (fd == EN_ERROR || fd == EN_INVALID_PARAM) {
    char_t err_buf[kMaxErrStrLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({file_path, reason.c_str()}));
    GELOGE(FAILED, "[Open][File] [%s] failed. %s", file_path, reason.c_str());
    return FAILED;
  }
  uint32_t len = static_cast<uint32_t>(model_str.length());
  // Write data to file
  const auto mmpa_ret = WriteBinToFile(fd, model_str.c_str(), len);
  if (mmpa_ret != GRAPH_SUCCESS) {
    char_t err_buf[kMaxErrStrLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    REPORT_PREDEFINED_ERR_MSG("E13004", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({file_path, reason.c_str()}));
    // Need to both print the error info of mmWrite and mmClose, so return ret after mmClose
    GELOGE(FAILED, "[WriteTo][File] %s failed. errno = %ld, %s", file_path, mmpa_ret, reason.c_str());
    ret = FAILED;
  }
  // Close file
  if (mmClose(fd) != EN_OK) {
    char_t err_buf[kMaxErrStrLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    REPORT_INNER_ERR_MSG("E19999", "close file:%s failed. errmsg:%s", file_path, err_msg);
    GELOGE(FAILED, "[Close][File] %s failed. errmsg:%s", file_path, err_msg);
    ret = FAILED;
  }
  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelSaver::CheckPath(const std::string &file_path) {
  // Determine file path length
  if (file_path.size() >= PATH_MAX) {
    REPORT_INNER_ERR_MSG("E19999", "Path is too long:%zu", file_path.size());
    GELOGE(FAILED, "[Check][Param] Path is too long:%zu", file_path.size());
    return FAILED;
  }

  // Find the last separator
  int path_split_pos = static_cast<int>(file_path.size() - 1);
  for (; path_split_pos >= 0; path_split_pos--) {
    if (file_path[path_split_pos] == '\\' || file_path[path_split_pos] == '/') {
      break;
    }
  }

  if (path_split_pos == 0) {
    return SUCCESS;
  }

  // If there is a path before the file name, create the path
  if (path_split_pos != -1) {
    if (CreateDirectory(std::string(file_path).substr(0, static_cast<size_t>(path_split_pos))) != kFileOpSuccess) {
      GELOGE(FAILED, "[Create][Directory] failed, file path:%s.", file_path.c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY int ModelSaver::CreateDirectory(const std::string &directory_path) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(), return -1, "directory path is empty.");
  auto dir_path_len = directory_path.length();
  if (dir_path_len >= PATH_MAX) {
    REPORT_PREDEFINED_ERR_MSG("E13002", std::vector<const char *>({"filepath", "size"}),
                              std::vector<const char *>({directory_path.c_str(), std::to_string(PATH_MAX).c_str()}));
    GELOGW("Path[%s] len is too long, it must be less than %d", directory_path.c_str(), PATH_MAX);
    return -1;
  }
  char tmp_dir_path[PATH_MAX] = {0};
  for (size_t i = 0; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      if (access(tmp_dir_path, F_OK) != 0) {
        int32_t ret = mmMkdir(tmp_dir_path, S_IRUSR | S_IWUSR | S_IXUSR);  // 700
        if (ret != 0) {
          if (errno != EEXIST) {
            char_t err_buf[kMaxErrStrLen + 1U] = {};
            const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
            REPORT_INNER_ERR_MSG("E19999",
                              "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                              directory_path.c_str(), err_msg);
            GELOGW("Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                   directory_path.c_str(), err_msg);
            return ret;
          }
        }
      }
    }
  }
  int32_t ret = mmMkdir(const_cast<char *>(directory_path.c_str()), S_IRUSR | S_IWUSR | S_IXUSR);  // 700
  if (ret != 0) {
    if (errno != EEXIST) {
      char_t err_buf[kMaxErrStrLen + 1U] = {};
      const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
      REPORT_INNER_ERR_MSG("E19999",
                        "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                        directory_path.c_str(), err_msg);
      GELOGW("Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
             directory_path.c_str(), err_msg);
      return ret;
    }
  }
  return 0;
}

}  // namespace parser
}  // namespace ge
