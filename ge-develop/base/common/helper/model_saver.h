/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_MODEL_SAVER_H_
#define GE_COMMON_MODEL_SAVER_H_

#include "common/helper/file_saver.h"
#include "nlohmann/json.hpp"
#include "framework/common/types.h"

/**
* Provide read and write operations for offline model files
*/
namespace ge {
using Json = nlohmann::json;

class ModelSaver : public FileSaver {
 public:
  /**
   * @ingroup domi_common
   * @brief Save JSON object to file
   * @param [in] file_path File output path
   * @param [in] model json object
   * @return Status result
   */
  static Status SaveJsonToFile(const char_t *const file_path, const Json &model);
};
}  //  namespace ge

#endif  //  GE_COMMON_MODEL_SAVER_H_
