/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/model_saver.h"

#include "base/err_msg.h"

#include <securec.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstring>
#include <array>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"

namespace {
constexpr size_t kMaxErrStrLength = 128U;
}  //  namespace

namespace ge {
constexpr int32_t kInteval = 2;

Status ModelSaver::SaveJsonToFile(const char_t *const file_path, const Json &model) {
  Status ret = SUCCESS;
  if ((file_path == nullptr) || (CheckPathValid(file_path) != SUCCESS)) {
    GELOGE(FAILED, "[Check][OutputFile]Failed, file %s", file_path);
    REPORT_INNER_ERR_MSG("E19999", "Output file %s check invalid", file_path);
    return FAILED;
  }
  std::string model_str;
  try {
    model_str = model.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to convert JSON to string, reason: %s, savefile:%s.", e.what(), file_path);
    GELOGE(FAILED, "[Convert][File]Failed to convert JSON to string, file %s, reason %s",
           file_path, e.what());
    return FAILED;
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to convert JSON to string, savefile:%s.", file_path);
    GELOGE(FAILED, "[Convert][File]Failed to convert JSON to string, file %s", file_path);
    return FAILED;
  }

  std::array<char_t, MMPA_MAX_PATH> file_real_path = {};
  GE_IF_BOOL_EXEC(mmRealPath(file_path, &file_real_path[0], MMPA_MAX_PATH) != EN_OK,
                  GELOGI("File %s does not exit, it will be created.", file_path));

  // Open file
  constexpr mmMode_t open_mode =
      static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR));
  std::array<char_t, kMaxErrStrLength + 1UL> err_buf = {};
  const int32_t fd = mmOpen2(&file_real_path[0],
                             static_cast<int32_t>(static_cast<uint32_t>(M_RDWR) |
                                                  static_cast<uint32_t>(M_CREAT) |
                                                  static_cast<uint32_t>(O_TRUNC)), open_mode);
  if ((fd == EN_ERROR) || (fd == EN_INVALID_PARAM)) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLength);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    (void)REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({file_path, reason.c_str()}));
    GELOGE(FAILED, "[Open][File]Failed, file %s, errmsg %s", file_path, reason.c_str());
    return FAILED;
  }
  const char_t *const model_char = model_str.c_str();
  const uint32_t len = static_cast<uint32_t>(model_str.length());
  // Write data to file
  const mmSsize_t mmpa_ret = mmWrite(fd, const_cast<char_t *>(model_char), len);
  if ((mmpa_ret == EN_ERROR) || (mmpa_ret == EN_INVALID_PARAM)) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLength);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    (void)REPORT_PREDEFINED_ERR_MSG("E13004", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({file_path, reason.c_str()}));
    // Need to both print the error info of mmWrite and mmClose, so return ret after mmClose
    GELOGE(FAILED, "[Write][Data]To file %s failed. errno %ld, errmsg %s",
           file_path, mmpa_ret, reason.c_str());
    ret = FAILED;
  }
  // Close file
  if (mmClose(fd) != EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLength);
    GELOGE(FAILED, "[Close][File]Failed, file %s, errmsg %s", file_path, err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Close file %s failed, errmsg %s", file_path, err_msg);
    ret = FAILED;
  }
  return ret;
}
}  // namespace ge
