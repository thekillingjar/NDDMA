/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/util.h"
#include "base/err_msg.h"
#include "common/checker.h"
#ifdef __GNUC__
#include <regex.h>
#else
#include <regex>
#endif
#include <algorithm>
#include <climits>
#include <ctime>
#include <fstream>

#include "ge/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "mmpa/mmpa_api.h"
#include "graph/types.h"

namespace ge {
namespace {
constexpr int32_t kFileSizeOutLimitedOrOpenFailed = -1;

/// The maximum length of the file.
constexpr int32_t kMaxBuffSize = 256;
constexpr size_t kMaxErrorStrLength = 128U;
const std::string kPathValidReason =
    "The path can only contain 'a-z' 'A-Z' '0-9' '-' '.' '_' and Chinese characters.";

void PathValidErrReport(const std::string &file_path, const std::string &atc_param, const std::string &reason) {
  if (!atc_param.empty()) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({atc_param.c_str(), file_path.c_str(), reason.c_str()}));
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Path[%s] invalid, reason:%s", file_path.c_str(), reason.c_str());
  }
}
}  // namespace

// Get file length
int64_t GetFileLength(const std::string &input_file) {
  if (input_file.empty()) {
    GELOGE(FAILED, "input_file path is null.");
    return -1;
  }

  const std::string real_path = RealPath(input_file.c_str());
  if (real_path.empty()) {
    GELOGE(FAILED, "input_file path '%s' not valid", input_file.c_str());
    return -1;
  }
  ULONGLONG file_length = 0U;
  if (mmGetFileSize(input_file.c_str(), &file_length) != EN_OK) {
    GELOGE(static_cast<uint32_t>(kFileSizeOutLimitedOrOpenFailed), "Open file[%s] failed.", input_file.c_str());
  }
  if (file_length == 0U) {
    GELOGE(FAILED, "File[%s] size is 0, not valid.", input_file.c_str());
    return -1;
  }

  return static_cast<int64_t>(file_length);
}

/** @ingroup domi_common
 *  @brief Read all data from binary file
 *  @param [in] file_name  File path
 *  @param [out] buffer  The address of the output memory, which needs to be released by the caller
 *  @param [out] length  Output memory size
 *  @return false fail
 *  @return true success
 */
bool ReadBytesFromBinaryFile(const char_t *const file_name, char_t **const buffer, int32_t &length) {
  if (file_name == nullptr) {
    GELOGE(FAILED, "incorrect parameter. file is nullptr");
    return false;
  }
  if (buffer == nullptr) {
    GELOGE(FAILED, "incorrect parameter. buffer is nullptr");
    return false;
  }

  const std::string real_path = RealPath(file_name);
  if (real_path.empty()) {
    GELOGE(FAILED, "file path '%s' not valid", file_name);
    return false;
  }

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    GELOGE(ge::FAILED, "[Read][File]Failed, file %s", file_name);
    REPORT_INNER_ERR_MSG("E19999", "Read file %s failed", file_name);
    return false;
  }

  length = static_cast<int32_t>(file.tellg());
  if (length <= 0) {
    file.close();
    GELOGE(FAILED, "file length <= 0");
    return false;
  }

  (void)file.seekg(0, std::ios::beg);
  
  *buffer = new (std::nothrow) char[static_cast<size_t>(length)]();
  if (*buffer == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "new an object failed.");
    GELOGE(FAILED, "new an object failed.");
    file.close();
    return false;
  }

  (void)file.read(*buffer, static_cast<int64_t>(length));
  file.close();
  return true;
}

std::string CurrentTimeInStr() {
  const std::time_t now = std::time(nullptr);
  const std::tm *const ptm = std::localtime(&now);
  if (ptm == nullptr) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    GELOGE(ge::FAILED, "[Check][Param]Localtime incorrect, errmsg %s", err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Localtime incorrect, errmsg %s", err_msg);
    return "";
  }

  constexpr int32_t kTimeBufferLen = 32;
  char_t buffer[kTimeBufferLen + 1] = {};
  // format: 20171122042550
  (void)std::strftime(&buffer[0], static_cast<size_t>(kTimeBufferLen), "%Y%m%d%H%M%S", ptm);
  return std::string(buffer);
}

uint64_t GetCurrentTimestamp() {
  mmTimeval tv{};
  const int32_t ret = mmGetTimeOfDay(&tv, nullptr);
  if (ret != EN_OK) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    GELOGE(ge::FAILED, "Func gettimeofday may failed, ret:%d, errmsg:%s", ret, err_msg);
  }
  const int64_t total_use_time = tv.tv_usec + (tv.tv_sec * 1000000);  // 1000000: seconds to microseconds
  return static_cast<uint64_t>(total_use_time);
}

uint32_t GetCurrentSecondTimestap() {
  mmTimeval tv{};
  const int32_t ret = mmGetTimeOfDay(&tv, nullptr);
  if ((ret != 0)) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    GELOGE(ge::FAILED, "Func gettimeofday may failed, ret:%d, errmsg:%s", ret, err_msg);
  }
  const int64_t total_use_time = tv.tv_sec;  // seconds
  return static_cast<uint32_t>(total_use_time);
}

static void RemoveDoubleHyphen(std::string &str) {
  if (!str.empty()) {
    constexpr size_t pos = 2UL;
    if (str.substr(0, pos) == "--") {
      (void)str.erase(0, pos);
    }
  }
}

bool CheckInputPathValid(const std::string &file_path, const std::string &atc_param) {
  // The specified path is empty
  if (file_path.empty()) {
    if (!atc_param.empty()) {
      std::string para = atc_param;
      RemoveDoubleHyphen(para);
      (void)REPORT_PREDEFINED_ERR_MSG("E10004", std::vector<const char *>({"parameter"}), std::vector<const char *>({para.c_str()}));
    } else {
      char_t err_buf[kMaxErrorStrLength + 1U] = {};
      const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
      std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
      (void)REPORT_PREDEFINED_ERR_MSG("E13000", std::vector<const char *>({"path", "errmsg"}),
                                std::vector<const char *>({file_path.c_str(), reason.c_str()}));
    }
    GELOGW("Input parameter %s is empty.", file_path.c_str());
    return false;
  }
  const std::string real_path = RealPath(file_path.c_str());
  // Unable to get absolute path (does not exist or does not have permission to access)
  if (real_path.empty()) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    const std::string reason = "realpath error, errmsg:" + std::string(err_msg);
    PathValidErrReport(file_path, atc_param, reason);
    GELOGW("Path[%s]'s realpath is empty, errmsg[%s]", file_path.c_str(), err_msg);
    return false;
  }

  // A regular matching expression to verify the validity of the input file path
  // Path section: Support upper and lower case letters, numbers dots(.) chinese and underscores
  // File name section: Support upper and lower case letters, numbers, underscores chinese and dots(.)
#ifdef __GNUC__
  const std::string mode = "^[\u4e00-\u9fa5A-Za-z0-9./_-]+$";
#else
  std::string mode = "^[a-zA-Z]:([\\\\/][^\\s\\\\/:*?<>\"|][^\\\\/:*?<>\"|]*)*([/\\\\][^\\s\\\\/:*?<>\"|])?$";
#endif

  if (!ValidateStr(real_path, mode)) {
    PathValidErrReport(file_path, atc_param, kPathValidReason);
    GELOGE(FAILED, "Invalid value for %s[%s], %s.", atc_param.c_str(), real_path.c_str(), kPathValidReason.c_str());
    return false;
  }

  // The absolute path points to a file that is not readable
  if (mmAccess2(real_path.c_str(), M_R_OK) != EN_OK) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto reason = "[Errno " + std::to_string(mmGetErrorCode()) + "] " +
                        mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    PathValidErrReport(file_path, atc_param, "cat not access, errmsg:" + reason);
    GELOGW("Read file[%s] unsuccessful, errmsg[%s]", file_path.c_str(), reason.c_str());
    return false;
  }

  return true;
}

bool CheckOutputPathValid(const std::string &file_path, const std::string &atc_param) {
  // The specified path is empty
  if (file_path.empty()) {
    if (!atc_param.empty()) {
      (void)REPORT_PREDEFINED_ERR_MSG("E10004", std::vector<const char_t *>({"parameter"}), std::vector<const char_t *>({atc_param.c_str()}));
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Param file_path is empty, check invalid.");
    }
    (void)REPORT_PREDEFINED_ERR_MSG("E10004", std::vector<const char *>({"parameter"}),
                              std::vector<const char *>({atc_param.c_str()}));
    GELOGW("Input parameter's value is empty.");
    return false;
  }

  if (file_path.length() >= static_cast<size_t>(MMPA_MAX_PATH)) {
    const std::string reason = "Path length is too long. It must be less than " + std::to_string(MMPA_MAX_PATH) + ".";
    PathValidErrReport(file_path, atc_param, reason);
    GELOGE(FAILED, "Path len is too long, it must be less than %d, path: [%s]", MMPA_MAX_PATH, file_path.c_str());
    return false;
  }

  // A regular matching expression to verify the validity of the input file path
  // Path section: Support upper and lower case letters, numbers dots(.) chinese and underscores
  // File name section: Support upper and lower case letters, numbers, underscores chinese and dots(.)
#ifdef __GNUC__
  const std::string mode = "^[\u4e00-\u9fa5A-Za-z0-9./_-]+$";
#else
  std::string mode = "^[a-zA-Z]:([\\\\/][^\\s\\\\/:*?<>\"|][^\\\\/:*?<>\"|]*)*([/\\\\][^\\s\\\\/:*?<>\"|])?$";
#endif

  if (!ValidateStr(file_path, mode)) {
    PathValidErrReport(file_path, atc_param, kPathValidReason);
    GELOGE(FAILED, "Invalid value for %s[%s], %s.", atc_param.c_str(), file_path.c_str(), kPathValidReason.c_str());
    return false;
  }

  const std::string real_path = RealPath(file_path.c_str());
  // Can get absolute path (file exists)
  if (!real_path.empty()) {
    // File is not readable or writable
    if (mmAccess2(real_path.c_str(),
        static_cast<int32_t>(static_cast<uint32_t>(M_W_OK) | static_cast<uint32_t>(M_F_OK))) != EN_OK) {
      char_t err_buf[kMaxErrorStrLength + 1U] = {};
      const auto reason = "[Errno " + std::to_string(mmGetErrorCode()) + "] " +
                          mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
      PathValidErrReport(file_path, atc_param, "cat not access, errmsg:" + std::string(reason));
      GELOGW("Write file[%s] unsuccessful, errmsg[%s]", real_path.c_str(), reason.c_str());
      return false;
    }
  } else {
    // Find the last separator
    int32_t path_split_pos = static_cast<int32_t>(file_path.size() - 1U);
    for (; path_split_pos >= 0; path_split_pos--) {
      if ((file_path[static_cast<uint64_t>(path_split_pos)] == '\\') ||
          (file_path[static_cast<uint64_t>(path_split_pos)] == '/')) {
        break;
      }
    }
    if (path_split_pos == 0) {
      return true;
    }
    if (path_split_pos != -1) {
      const std::string prefix_path = std::string(file_path).substr(0U, static_cast<size_t>(path_split_pos));
      // Determine whether the specified path is valid by creating the path
      if (CreateDirectory(prefix_path) != 0) {
        char_t err_buf[kMaxErrorStrLength + 1U] = {};
        const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
        std::string reason =
            "Directory creation failed. [Errno " + std::to_string(mmGetErrorCode()) + "] " + err_msg + ".";
        PathValidErrReport(file_path, atc_param, "Can not create directory");
        GELOGW("Can not create directory[%s].", file_path.c_str());
        return false;
      }
    }
  }

  return true;
}

FMK_FUNC_HOST_VISIBILITY bool ValidateStr(const std::string &file_path, const std::string &mode) {
#ifdef __GNUC__
  char_t ebuff[kMaxBuffSize];
  regex_t reg;
  constexpr int32_t cflags =
      static_cast<int32_t>(static_cast<uint32_t>(REG_EXTENDED) | static_cast<uint32_t>(REG_NOSUB));
  int32_t ret = regcomp(&reg, mode.c_str(), cflags);
  if (static_cast<bool>(ret)) {
    (void)regerror(ret, &reg, &ebuff[0U], static_cast<size_t>(kMaxBuffSize));
    GELOGW("regcomp unsuccessful, reason: %s", &ebuff[0U]);
    regfree(&reg);
    return true;
  }

  ret = regexec(&reg, file_path.c_str(), 0U, nullptr, 0);
  if (static_cast<bool>(ret)) {
    (void)regerror(ret, &reg, &ebuff[0], static_cast<size_t>(kMaxBuffSize));
    GELOGE(ge::PARAM_INVALID, "[Rgexec][Param]Failed, reason %s", &ebuff[0]);
    REPORT_INNER_ERR_MSG("E19999", "Rgexec failed, reason %s", &ebuff[0]);
    regfree(&reg);
    return false;
  }

  regfree(&reg);
  return true;
#else
  std::wstring wstr(file_path.begin(), file_path.end());
  std::wstring wmode(mode.begin(), mode.end());
  std::wsmatch match;
  bool res = false;

  try {
    std::wregex reg(wmode, std::regex::icase);
    // Matching std::string part
    res = regex_match(wstr, match, reg);
    res = regex_search(file_path, std::regex("[`!@#$%^&*()|{}';',<>?]"));
  } catch (std::exception &ex) {
    GELOGW("The directory %s is invalid, error: %s.", file_path.c_str(), ex.what());
    return false;
  }
  return !(res) && (file_path.size() == match.str().size());
#endif
}

Status ConvertToInt32(const std::string &str, int32_t &val) {
  try {
    val = std::stoi(str);
  } catch (std::invalid_argument &) {
    GELOGE(FAILED, "[Parse][Param]Failed, digit str:%s is invalid", str.c_str());
    REPORT_INNER_ERR_MSG("E18888", "Parse param failed, digit str:%s is invalid", str.c_str());
    return FAILED;
  } catch (std::out_of_range &) {
    GELOGE(FAILED, "[Parse][Param]Failed, digit str:%s cannot change to int", str.c_str());
    REPORT_INNER_ERR_MSG("E18888", "Parse param failed, digit str:%s cannot change to int", str.c_str());
    return FAILED;
  } catch (...) {
    GELOGE(FAILED, "[Parse][Param]Failed, digit str:%s cannot change to int", str.c_str());
    REPORT_INNER_ERR_MSG("E18888", "Parse param failed, digit str:%s cannot change to int", str.c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status ConvertToInt64(const std::string &str, int64_t &val) {
  try {
    val = std::stoll(str);
  } catch (std::invalid_argument &) {
    GELOGE(FAILED, "[Parse][Param]Failed, digit str:%s is invalid", str.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Parse param failed, digit str:%s is invalid", str.c_str());
    return FAILED;
  } catch (std::out_of_range &) {
    GELOGE(FAILED, "[Parse][Param]Failed, digit str:%s cannot change to int", str.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Parse param failed, digit str:%s cannot change to int", str.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ConvertToUint64(const std::string &str, uint64_t &val) {
  try {
    val = std::stoull(str);
  } catch (std::invalid_argument &) {
    GELOGE(FAILED, "[Parse][Param]Failed, digit str:%s is invalid", str.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Parse param failed, digit str:%s is invalid", str.c_str());
    return FAILED;
  } catch (std::out_of_range &) {
    GELOGE(FAILED, "[Parse][Param]Failed, digit str:%s cannot change to uint64 result of out of range", str.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Parse param failed, digit str:%s cannot change to uint64", str.c_str());
    return FAILED;
  }
  return SUCCESS;
}

std::string GetErrorNumStr(const int32_t errorNum) {
  char_t err_buf[kMaxErrorStrLength + 1U] = {};
  const auto err_msg = mmGetErrorFormatMessage(errorNum, &err_buf[0], kMaxErrorStrLength);
  return (err_msg == nullptr) ? "" : err_msg;
}

void SplitStringByComma(const std::string &str, std::vector<std::string> &sub_str_vec) {
  std::string tmp_string = str + ",";
  std::string::size_type start_pos = 0;
  std::string::size_type cur_pos = tmp_string.find(',', 0);
  while (cur_pos != std::string::npos) {
    std::string sub_str = tmp_string.substr(start_pos, cur_pos - start_pos);
    if (!sub_str.empty()) {
      std::vector<std::string>::iterator ret = std::find(sub_str_vec.begin(), sub_str_vec.end(), sub_str);
      if (ret == sub_str_vec.end()) {
        sub_str_vec.push_back(sub_str);
      }
    }
    start_pos = cur_pos + 1;
    cur_pos = tmp_string.find(',', start_pos);
  }
}

void ParseOutputReuseInputMemIndexes(const std::string &reuse_indexes_str,
                                     std::vector<std::pair<size_t, size_t>> &io_same_addr_pairs) {
  if (reuse_indexes_str.empty()) {
    return;
  }

  std::string::size_type start = 0;
  while (start < reuse_indexes_str.length()) {
    const std::string::size_type end = reuse_indexes_str.find('|', start);
    std::string pair_str;
    // 1. 优先计算当前子串并更新下一轮的 start
    if (end == std::string::npos) {
      pair_str = reuse_indexes_str.substr(start);
      start = reuse_indexes_str.length();
    } else {
      pair_str = reuse_indexes_str.substr(start, end - start);
      start = end + 1;
    }
    // 2. 过滤空字符串
    if (pair_str.empty()) {
      continue;
    }
    // 3. 校验格式（必须包含逗号）
    const size_t comma_pos = pair_str.find(',');
    if (comma_pos == std::string::npos) {
      continue;
    }
    const std::string input_idx_str = pair_str.substr(0, comma_pos);
    const std::string output_idx_str = pair_str.substr(comma_pos + 1);
    // 4. 校验负数
    if (input_idx_str.find('-') != std::string::npos ||
        output_idx_str.find('-') != std::string::npos) {
      GELOGW("[Parse][Option] Invalid negative index in ge.exec.outputReuseInputMemIndexes: '%s'. "
             "Indexes must be non-negative.", reuse_indexes_str.c_str());
      continue;
    }
    try {
      const size_t input_idx = static_cast<size_t>(std::stoul(input_idx_str));
      const size_t output_idx = static_cast<size_t>(std::stoul(output_idx_str));
      io_same_addr_pairs.emplace_back(input_idx, output_idx);
    } catch (...) {
      GELOGW("[Parse][Option] Invalid index in ge.exec.outputReuseInputMemIndexes: '%s'. "
             "Indexes must be non-negative integers.", reuse_indexes_str.c_str());
    }
  }
}

Status CheckIoReuseAddrPairs(const std::vector<std::pair<size_t, size_t>> &io_same_addr_pairs,
                             const AddrGetter& get_input_addr, size_t input_num,
                             const AddrGetter& get_output_addr, size_t output_num) {
  GELOGD("Start to check io reuse addr pairs, io_same_addr_pairs count: %zu, input num: %zu, output num: %zu",
         io_same_addr_pairs.size(), input_num, output_num);
  for (const auto &pair : io_same_addr_pairs) {
    const size_t input_idx = pair.first;
    const size_t output_idx = pair.second;

    if (input_idx >= input_num) {
      GELOGE(ge::PARAM_INVALID, "[Check][Param] Input index %zu out of range, input_num is %zu, addr reuse check not pass",
             input_idx, input_num);
      return ge::PARAM_INVALID;
    }
    if (output_idx >= output_num) {
      GELOGE(ge::PARAM_INVALID, "[Check][Param] Output index %zu out of range, output_num is %zu, addr reuse check not pass",
             output_idx, output_num);
      return ge::PARAM_INVALID;
    }

    const void *input_addr = get_input_addr(input_idx);
    const void *output_addr = get_output_addr(output_idx);

    if (output_addr != input_addr) {
      GELOGE(ge::PARAM_INVALID, "[Check][Param] Output[%zu] address(%p) must be same as Input[%zu] address(%p) because the option "
             "'ge.exec.outputReuseInputMemIndexes' is set with value containing '%zu,%zu', addr reuse check not pass",
             output_idx, output_addr, input_idx, input_addr, input_idx, output_idx);
      return ge::PARAM_INVALID;
    }
    GELOGD("[Check][Param] Output[%zu] address(%p) is same as Input[%zu] address(%p), addr reuse check pass.",
           output_idx, output_addr, input_idx, input_addr);
  }
  return SUCCESS;
}
}  //  namespace ge
