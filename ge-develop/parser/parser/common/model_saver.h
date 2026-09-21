/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_FILE_SAVER_H_
#define PARSER_COMMON_FILE_SAVER_H_

#include <string>

#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "register/register_types.h"
#include "nlohmann/json.hpp"

namespace ge {
namespace parser {
using Json = nlohmann::json;
using std::string;

class ModelSaver {
public:
  /**
   * @ingroup domi_common
   * @brief Save JSON object to file
   * @param [in] file_path File output path
   * @param [in] model json object
   * @return Status result
   */
  static Status SaveJsonToFile(const char *file_path, const Json &model);

private:
  /// @ingroup domi_common
  /// @brief Check validity of the file path
  /// @return Status  result
  static Status CheckPath(const string &file_path);

  static int CreateDirectory(const std::string &directory_path);
};
}  // namespace parser
}  // namespace ge

#endif // PARSER_COMMON_FILE_SAVER_H_
