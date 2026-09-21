/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/util/json_util.h"
#include <climits>
#include <fstream>
#include <thread>
#include <ext/stdio_filebuf.h>
#include <fcntl.h>
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/aicore_util_constants.h"
#include "common/fe_error_code.h"
#include "common/fe_report_error.h"

namespace fe {
const uint32_t kFlockRecursiveIntvl = 10; // 10 milliseconds
const uint32_t kFlockRecursiveCntMax = 6000; // 1mins 6000*10 milliseconds

std::string RealPath(const std::string &path) {
  if (path.empty()) {
    FE_LOGI("Path string is nullptr.");
    return "";
  }
  if (path.size() >= PATH_MAX) {
    FE_LOGI("File path %s is too long!", path.c_str());
    return "";
  }

  // PATH_MAX is the system marco，indicate the maximum length for file path
  // pclint check，one param in stack can not exceed 1K bytes
  char *resoved_path = new(std::nothrow) char[PATH_MAX];
  if (resoved_path == nullptr) {
    FE_LOGI("New resoved_path not successfully.");
    return "";
  }
  (void)memset_s(resoved_path, PATH_MAX, 0, PATH_MAX);

  std::string res = "";

  // path not exists or not allowed to read，return nullptr
  // path exists and readable, return the resoved path
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    res = resoved_path;
  } else {
    FE_LOGI("Path %s does not exist.", path.c_str());
  }

  delete[] resoved_path;
  return res;
}

Status FcntlLockFile(const std::string &file, int fd, int type, uint32_t recursive_cnt) {
  struct flock lock_arg;
  lock_arg.l_whence = SEEK_SET;
  lock_arg.l_start = 0;
  lock_arg.l_len = 0;
  lock_arg.l_type = type;

  if (fcntl(fd, F_SETLK, &lock_arg) != 0) {
    if (recursive_cnt == 0) {
      if (type == F_UNLCK) {
        FE_LOGW("Realse lock file(%s) failed.", file.c_str());
      } else {
        FE_LOGD("File(%s) is locked by %d.", file.c_str(), lock_arg.l_pid);
      }
    }
    return FAILED;
  }
  return SUCCESS;
}

void LogOpenFileErrMsg(const std::string &file)
{
  ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {file});
  ReportErrorMessage(err_msg);
}

Status ReadJsonFile(const std::string &file, nlohmann::json &json_obj) {
  std::string path = RealPath(file);
  if (path.empty()) {
    FE_LOGW("File path [%s] does not exist.", file.c_str());
    return FAILED;
  }
  std::ifstream if_stream(path);
  try {
    if (!if_stream.is_open()) {
      FE_LOGW("Failed to open %s, file is already opened.", file.c_str());

      LogOpenFileErrMsg(file);
      return FAILED;
    }

    if_stream >> json_obj;
    if_stream.close();
  } catch (const std::exception &e) {
    FE_LOGW("Failed to convert file[%s] to Json. Exception message is %s", path.c_str(), e.what());
    if_stream.close();
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

Status ReadJsonFileByLock(const std::string &file, nlohmann::json &json_obj) {
  std::string path = RealPath(file);
  if (path.empty()) {
    FE_LOGW("File path [%s] is not valid.", file.c_str());

    LogOpenFileErrMsg(file);
    return FAILED;
  }
  std::ifstream if_stream(path);
  try {
    if (!if_stream.is_open()) {
      FE_LOGW("Failed to open %s, file is already opened.", file.c_str());

      LogOpenFileErrMsg(file);
      return FAILED;
    }

    uint32_t recursive_cnt = 0;
    int ifs_fd = static_cast<__gnu_cxx::stdio_filebuf<char> *>(if_stream.rdbuf())->fd();

    do {
      if (FcntlLockFile(file, ifs_fd, F_RDLCK, recursive_cnt) == FAILED) {
        std::this_thread::sleep_for(std::chrono::microseconds(kFlockRecursiveIntvl));
      } else {
        FE_LOGD("Lock file(%s).", file.c_str());
        break;
      }
      if (recursive_cnt == kFlockRecursiveCntMax) {
        FE_LOGE("Lock file(%s) failed, try %u times.", file.c_str(), kFlockRecursiveCntMax);
        LogOpenFileErrMsg(file);
        if_stream.close();
        return FAILED;
      }
      recursive_cnt++;
    } while (true);
    if_stream >> json_obj;
    (void)FcntlLockFile(file, ifs_fd, F_UNLCK, 0);
    FE_LOGD("Release lock file(%s).", file.c_str());
    if_stream.close();
  } catch (const std::exception &e) {
    FE_LOGW("Failed to convert file [%s] to json. Message is %s", path.c_str(), e.what());
    if_stream.close();

    LogOpenFileErrMsg(file);
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

std::string GetSuffixJsonFile(const std::string &json_file_path, const std::string &suffix) {
  if (json_file_path.empty()) {
    FE_LOGW("Json file path is empty.");
    return "";
  }
  string tmp_str = json_file_path;
  size_t find_pos = json_file_path.find_last_of('.');
  if (find_pos != std::string::npos) {
    return tmp_str.insert(find_pos, suffix);
  } else {
    return "";
  }
}

std::string GetJsonType(const nlohmann::json &json_object) {
  std::string json_type;

  switch (json_object.type()) {
    case nlohmann::json::value_t::null:
      json_type = "null";
      break;
    case nlohmann::json::value_t::object:
      json_type = "object";
      break;
    case nlohmann::json::value_t::array:
      json_type = "array";
      break;
    case nlohmann::json::value_t::string:
      json_type = "string";
      break;
    case nlohmann::json::value_t::boolean:
      json_type = "boolean";
      break;
    case nlohmann::json::value_t::number_integer:
      json_type = "number_integer";
      break;
    case nlohmann::json::value_t::number_unsigned:
      json_type = "number_unsigned";
      break;
    case nlohmann::json::value_t::number_float:
      json_type = "number_float";
      break;
    case nlohmann::json::value_t::discarded:
      json_type = "discarded";
      break;
    default:
      json_type = "discarded";
      break;
  }

  return json_type;
}
}  // namespace fe
