/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_JSON_UTIL_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_JSON_UTIL_H_

#include <nlohmann/json.hpp>
#include <string>
#include "graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
std::string RealPath(const std::string& path);
Status FcntlLockFile(const std::string &file, int fd, int type, uint32_t recursive_cnt);
void LogOpenFileErrMsg(const std::string &file);
Status ReadJsonFile(const std::string& file, nlohmann::json& json_obj);
Status ReadJsonFileByLock(const std::string &file, nlohmann::json &json_obj);
std::string GetSuffixJsonFile(const std::string &json_file_path, const std::string &suffix);
std::string GetJsonType(const nlohmann::json &json_object);
}
#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_JSON_UTIL_H_

