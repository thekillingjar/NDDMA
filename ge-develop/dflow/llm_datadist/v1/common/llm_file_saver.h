/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_COMMON_LLM_FILE_SAVER_H_
#define AIR_RUNTIME_LLM_ENGINE_COMMON_LLM_FILE_SAVER_H_

#include <string>
#include <vector>

#include "llm_inner_types.h"
#include "ge/ge_api.h"
#include "mmpa/mmpa_api.h"

namespace llm {
int32_t CreateDirectory(const std::string &directory_path);

class LLMFileSaver {
 public:
  static ge::Status SaveToFile(const std::string &file_path, const void *const data, const uint64_t len,
                           const bool append = false);
 protected:
  /// @ingroup domi_common
  /// @brief Check validity of the file path
  /// @return ge::Status  result
  static ge::Status CheckPathValid(const std::string &file_path);

  static ge::Status OpenFile(int32_t &fd, const std::string &file_path, const bool append = false);

  static ge::Status WriteData(const void * const data, uint64_t size, const int32_t fd);
};
}  // namespace llm
#endif  // AIR_RUNTIME_LLM_ENGINE_COMMON_LLM_FILE_SAVER_H_
