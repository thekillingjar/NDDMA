/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <array>

#include "common/llm_file_saver.h"
#include "def_types.h"
#include "common/llm_checker.h"
#include "common/llm_utils.h"
#include "common/llm_log.h"
#include "common/def_types.h"

namespace llm {
constexpr int32_t kFileOpSuccess = 0;
constexpr size_t kMaxErrStrLen = 128U;

inline int32_t CheckAndMkdir(const char_t *tmp_dir_path, mmMode_t mode) {
  if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
    const int32_t ret = mmMkdir(tmp_dir_path, mode);
    if (ret != 0) {
      REPORT_INNER_ERR_MSG("E18888",
                        "Can not create directory %s. Make sure the directory "
                        "exists and writable. errmsg:%s",
                        tmp_dir_path, strerror(errno));
      LLMLOGW(
          "[Util][mkdir] Create directory %s failed, reason:%s. Make sure the "
          "directory exists and writable.",
          tmp_dir_path, strerror(errno));
      return ret;
    }
  }
  return 0;
}
namespace {
/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDir(const std::string &directory_path, uint32_t mode) {
  LLM_CHK_BOOL_EXEC(!directory_path.empty(),
                   REPORT_INNER_ERR_MSG("E18888", "directory path is empty, check invalid");
                       return -1, "[Check][Param] directory path is empty.");
  const auto dir_path_len = directory_path.length();
  if (dir_path_len >= static_cast<size_t>(MMPA_MAX_PATH)) {
    LLMLOGW("[Util][mkdir] Path %s len is too long, it must be less than %d", directory_path.c_str(), MMPA_MAX_PATH);
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

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDir(const std::string &directory_path) {
  constexpr auto mkdir_mode = static_cast<uint32_t>(M_IRUSR | M_IWUSR | M_IXUSR);
  return CreateDir(directory_path, mkdir_mode);
}
} // namespace
/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDirectory(const std::string &directory_path) {
  return CreateDir(directory_path);
}

ge::Status LLMFileSaver::CheckPathValid(const std::string &file_path) {
  // Determine file path length
  if (file_path.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
    LLMLOGE(ge::FAILED, "[Check][FilePath]Failed, file path's length:%zu >= mmpa_max_path:%d",
           file_path.size(), MMPA_MAX_PATH);
    REPORT_INNER_ERR_MSG("E19999", "Check file path failed, file path's length:%zu >= "
                       "mmpa_max_path:%d", file_path.size(), MMPA_MAX_PATH);
    return ge::FAILED;
  }

  // Find the last separator
  int32_t path_split_pos = static_cast<int32_t>(file_path.size() - 1U);
  for (; path_split_pos >= 0; path_split_pos--) {
    if ((file_path[static_cast<size_t>(path_split_pos)] == '\\') ||
        (file_path[static_cast<size_t>(path_split_pos)] == '/')) {
      break;
    }
  }

  if (path_split_pos == 0) {
    return ge::SUCCESS;
  }

  // If there is a path before the file name, create the path
  if (path_split_pos != -1) {
    if (CreateDirectory(std::string(file_path).substr(0U, static_cast<size_t>(path_split_pos))) != kFileOpSuccess) {
      LLMLOGE(ge::FAILED, "[Create][Directory]Failed, file path:%s.", file_path.c_str());
      return ge::FAILED;
    }
  }

  return ge::SUCCESS;
}

ge::Status LLMFileSaver::OpenFile(int32_t &fd, const std::string &file_path, const bool append) {
  if (CheckPathValid(file_path) != ge::SUCCESS) {
    LLMLOGE(ge::FAILED, "[Check][FilePath]Check output file failed, file_path:%s.",
           file_path.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Check output file failed, file_path:%s.",
                      file_path.c_str());
    return ge::FAILED;
  }

  std::array<char_t, MMPA_MAX_PATH> real_path = {};
  LLM_IF_BOOL_EXEC(mmRealPath(file_path.c_str(), &real_path[0], MMPA_MAX_PATH) != EN_OK,
                  LLMLOGI("File %s does not exist, it will be created.", file_path.c_str()));
  // Open file
  constexpr mmMode_t mode = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR));
  uint32_t open_flag = static_cast<uint32_t>(M_RDWR) | static_cast<uint32_t>(M_CREAT);
  if (append) {
    open_flag |= static_cast<uint32_t>(O_APPEND);
  } else {
    open_flag |= static_cast<uint32_t>(O_TRUNC);
  }

  fd = mmOpen2(&real_path[0], static_cast<int32_t>(open_flag), mode);
  if ((fd == EN_INVALID_PARAM) || (fd == EN_ERROR)) {
    // -1: Failed to open file; - 2: Illegal parameter
    std::array<char_t, kMaxErrStrLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    std::string reason = "[Errno " + std::to_string(mmGetErrorCode()) + "] " + err_msg + ".";
    LLMLOGE(ge::FAILED, "[Open][File]Failed. errno:%d, errmsg:%s", fd, err_msg);
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"value", "parameter", "reason"}),
                              std::vector<const char *>({file_path.c_str(), "file path", reason.c_str()}));
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status LLMFileSaver::WriteData(const void * const data, uint64_t size, const int32_t fd) {
  if ((size == 0U) || (data == nullptr)) {
    return ge::LLM_PARAM_INVALID;
  }
  int64_t write_count;
  constexpr uint64_t kMaxWriteSize = 1 * 1024 * 1024 * 1024UL; // 1G
  auto seek = PtrToPtr<void, uint8_t>(const_cast<void *>(data));
  while (size > 0U) {
    const uint64_t expect_write_size = std::min(size, kMaxWriteSize);
    write_count = mmWrite(fd, reinterpret_cast<void *>(seek), static_cast<uint32_t>(expect_write_size));
    LLM_ASSERT_TRUE(((write_count != EN_INVALID_PARAM) && (write_count != EN_ERROR)),
        "Write data failed, errno: %lld", write_count);
    seek = PtrAdd<uint8_t>(seek, static_cast<size_t>(size), write_count);
    LLM_ASSERT_TRUE(size >= static_cast<uint64_t>(write_count),
        "Write data failed, errno: %lld, size: %u", write_count, size);
    size -= write_count;
  }

  return ge::SUCCESS;
}

ge::Status LLMFileSaver::SaveToFile(const std::string &file_path, const void *const data, const uint64_t len,
                             const bool append) {
  if ((data == nullptr) || (len <= 0)) {
    LLMLOGE(ge::FAILED, "[Check][Param]Failed, model_data is null or the "
           "length[%lu] is less than 1.", len);
    REPORT_INNER_ERR_MSG("E19999", "Save file failed, the model_data is null or "
                       "its length:%" PRIu64 " is less than 1.", len);
    return ge::FAILED;
  }

  // Open file
  int32_t fd = 0;
  if (OpenFile(fd, file_path, append) != ge::SUCCESS) {
    LLMLOGE(ge::FAILED, "OpenFile ge::FAILED");
    return ge::FAILED;
  }

  ge::Status ret = ge::SUCCESS;

  // write data
  LLM_CHK_BOOL_EXEC(WriteData(data, len, fd) == ge::SUCCESS, ret = ge::FAILED, "WriteData ge::FAILED");

  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    std::array<char_t, kMaxErrStrLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    LLMLOGE(ge::FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Close file failed, error_code:%u errmsg:%s",
                      ret, err_msg);
    ret = ge::FAILED;
  }
  return ret;
}
}  //  namespace llt
