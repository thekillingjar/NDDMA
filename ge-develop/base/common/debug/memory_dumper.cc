/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/debug/memory_dumper.h"

#include <string>
#include <climits>
#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/types.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/def_types.h"
#include "graph_metadef/common/ge_common/util.h"
#include "base/err_msg.h"

namespace {
constexpr size_t kMaxErrorStringLength = 128U;
}  // namespace

namespace ge {
MemoryDumper::~MemoryDumper() { Close(); }

// Dump the data to the file
Status MemoryDumper::DumpToFile(const char_t *const filename, const void *const data, const uint64_t len) {
#ifdef FMK_SUPPORT_DUMP
  GE_CHECK_NOTNULL(filename);
  GE_CHECK_NOTNULL(data);
  if (len == 0) {
    GELOGE(FAILED, "[Check][Param]Failed, data length is 0.");
    REPORT_INNER_ERR_MSG("E19999", "Check param failed, data length is 0.");
    return PARAM_INVALID;
  }

  // Open the file
  const int32_t fd = OpenFile(filename);
  if (fd == kInvalidFd) {
    GELOGE(FAILED, "[Open][File]Failed, filename:%s.", filename);
    REPORT_INNER_ERR_MSG("E19999", "Opne file failed, filename:%s.", filename);
    return FAILED;
  }

  // Write the data to the file
  Status ret = SUCCESS;
  const auto graph_status = WriteBinToFile(fd, PtrToPtr<void, char_t>(data), static_cast<size_t>(len));
  if (graph_status != GRAPH_SUCCESS) {
    // already print error in WriteBinToFile
    ret = FAILED;
  }

  // Close the file
  if (mmClose(fd) != EN_OK) {  // mmClose return 0: success
    char_t err_buf[kMaxErrorStringLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u, filename:%s errmsg:%s.", ret, filename, err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Close file failed, error_code:%u, filename:%s errmsg:%s.",
                       ret, filename, err_msg);
    ret = FAILED;
  }

  return ret;
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump op input and output.");
  return SUCCESS;
#endif
}

// Close file
void MemoryDumper::Close() noexcept {
  // Close file
  if ((fd_ != kInvalidFd) && (mmClose(fd_) != EN_OK)) {
    char_t err_buf[kMaxErrorStringLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
    GELOGW("Close file failed, errmsg:%s.", err_msg);
  }
  fd_ = kInvalidFd;
}

void MemoryDumper::PrintErrorMsg(const std::string &error_msg) {
  size_t i = 0UL;
  const constexpr size_t print_size = 512UL;
  while (i < error_msg.length()) {
    const size_t split_size = ((error_msg.length() - i) > print_size) ? print_size : (error_msg.length() - i);
    std::string temp_str = error_msg.substr(i, split_size);
    GELOGE(FAILED, "%s", temp_str.c_str());
    i += split_size;
  }
}
// Open file
int32_t MemoryDumper::OpenFile(const std::string &filename) {
  // Find the last separator
  int32_t path_split_pos = static_cast<int32_t>(filename.size());
  path_split_pos--;
  for (; path_split_pos >= 0; path_split_pos--) {
    GE_IF_BOOL_EXEC((filename[static_cast<size_t>(path_split_pos)] == '\\') ||
                    (filename[static_cast<size_t>(path_split_pos)] == '/'), break;)
  }
  // Get the absolute path
  std::string real_path;
  char_t tmp_path[MMPA_MAX_PATH] = {};
  GE_IF_BOOL_EXEC(
    path_split_pos != -1, const std::string prefix_path = filename.substr(0U, static_cast<size_t>(path_split_pos));
    const std::string last_path = filename.substr(static_cast<size_t>(path_split_pos), filename.size() - 1U);
    if (prefix_path.length() >= static_cast<size_t>(MMPA_MAX_PATH)) {
      GELOGE(FAILED, "Prefix path is too long!");
      return kInvalidFd;
    }
    if (mmRealPath(prefix_path.c_str(), &tmp_path[0], MMPA_MAX_PATH) != EN_OK) {
      char_t err_buf[kMaxErrorStringLength + 1U] = {};
      const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
      GELOGE(ge::FAILED, "Dir %s does not exit, errmsg:%s.", prefix_path.c_str(), err_msg);
      return kInvalidFd;
    }
    real_path = std::string(tmp_path) + last_path;)
  GE_IF_BOOL_EXEC(
    (path_split_pos == -1) || (path_split_pos == 0),
    if (filename.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
      GELOGE(FAILED, "Prefix path is too long!");
      return kInvalidFd;
    }

    GE_IF_BOOL_EXEC(mmRealPath(filename.c_str(), &tmp_path[0], MMPA_MAX_PATH) != EN_OK,
                    GELOGI("File %s does not exit, it will be created.", filename.c_str()));
    real_path = std::string(tmp_path));

  // Open file, only the current user can read and write, to avoid malicious application access
  // Using the O_EXCL, if the file already exists,return failed to avoid privilege escalation vulnerability.
  constexpr mmMode_t open_mode = static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR);
  constexpr int32_t flag = static_cast<int32_t>(static_cast<uint32_t>(M_RDWR) | static_cast<uint32_t>(M_CREAT) |
                                                static_cast<uint32_t>(M_APPEND));
  
  std::string dir_path;
  std::string file_name;
  SplitFilePath(real_path, dir_path, file_name);
  std::string file_path = real_path;
  if (file_name.length() > NAME_MAX) {
    const constexpr size_t file_name_cut_length = 20UL;
    file_name = file_name.substr(0, file_name_cut_length);
    file_name += "_";
    file_name += CurrentTimeInStr();
    file_path = real_path.substr(0, dir_path.length() + 1UL) + file_name;
    std::string error_msg = "Cause file name is too long, change name from " +
        real_path + " to " + file_path;
    PrintErrorMsg(error_msg);
  }

  const int32_t fd = mmOpen2(file_path.c_str(), flag, open_mode);
  if ((fd == EN_ERROR) || (fd == EN_INVALID_PARAM)) {
    char_t err_buf[kMaxErrorStringLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
    GELOGE(static_cast<uint32_t>(kInvalidFd), "[Open][File]Failed. errno:%d, errmsg:%s, filename:%s.",
           fd, err_msg, filename.c_str());
    return kInvalidFd;
  }
  return fd;
}
}  // namespace ge
