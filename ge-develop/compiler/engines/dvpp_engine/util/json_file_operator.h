/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_UTIL_JSON_FILE_OPERATOR_H_
#define DVPP_ENGINE_UTIL_JSON_FILE_OPERATOR_H_

#include <nlohmann/json.hpp>
#include "common/dvpp_ops.h"
#include "util/dvpp_constexpr.h"

namespace dvpp {
class JsonFileOperator {
 public:
  /**
   * @brief Read json file in path
   * @param file_path_new new json file path
   * @param file_path json file path
   * @param json_file read json file handle
   * @return whether read file success
   */
  static bool ReadJsonFile(const std::string& file_path_new,
      const std::string& file_path, nlohmann::json& json_file);

 private:
  /**
   * @brief convert json file format
   * @param json_file original format json
   * @return whether convert format sucess
   */
  static bool ConvertJsonFormat(nlohmann::json& json_file);

  /**
   * @brief convert json file format which value type is string
   * @param dvpp_op original format json
   * @param key convert key
   * @return whether convert format sucess
   */
  static bool ConvertJsonFormatString(
      nlohmann::json& dvpp_op, const std::string& key);

  /**
   * @brief convert json file format which value type is int32_t
   * @param dvpp_op original format json
   * @param key convert key
   * @return whether convert format sucess
   */
  static bool ConvertJsonFormatInt32(
      nlohmann::json& dvpp_op, const std::string& key);

  /**
   * @brief convert json file format which value type is int64_t
   * @param dvpp_op original format json
   * @param key convert key
   * @return whether convert format sucess
   */
  static bool ConvertJsonFormatInt64(
      nlohmann::json& dvpp_op, const std::string& key);

  /**
   * @brief convert json file format which value type is bool
   * @param dvpp_op original format json
   * @param key convert key
   * @return whether convert format sucess
   */
  static bool ConvertJsonFormatBool(
      nlohmann::json& dvpp_op, const std::string& key);

  /**
   * @brief convert json file format which is inputs & outputs
   * @param dvpp_op original format json
   * @return whether convert format sucess
   */
  static bool ConvertJsonFormatInputOutput(nlohmann::json& dvpp_op);

  /**
   * @brief convert json file format which is attrs
   * @param dvpp_op original format json
   * @return whether convert format sucess
   */
  static bool ConvertJsonFormatAttr(nlohmann::json& dvpp_op);

  /**
   * @brief print json_file context
   * @param json_file context file
   */
  static void PrintJsonFileContext(nlohmann::json& json_file);

  /**
   * @brief print dvpp op input and output context
   * @param dvpp_op dvpp op context
   */
  static void PrintDvppOpInfoInputOutput(nlohmann::json& dvpp_op);

  /**
   * @brief print dvpp op attr context
   * @param dvpp_op dvpp op context
   */
  static void PrintDvppOpInfoAttr(nlohmann::json& dvpp_op);
}; // class JsonFileOperator

/**
 * @brief DvppOpsInfoLib json to struct object funtion
 * @param json_file read json handle
 * @param dvpp_ops_info_lib convert struct
 */
void from_json(const nlohmann::json& json_file,
               DvppOpsInfoLib& dvpp_ops_info_lib);

/**
 * @brief DvppOp json to struct object funtion
 * @param json_file read json handle
 * @param dvpp_op convert struct
 */
void from_json(const nlohmann::json& json_file, DvppOp& dvpp_op);

/**
 * @brief DvppOpInfo json to struct object funtion
 * @param json_file read json handle
 * @param dvpp_ops_info_lib convert struct
 */
void from_json(const nlohmann::json& json_file, DvppOpInfo& dvpp_op_info);
} // namespace dvpp

#endif // DVPP_ENGINE_UTIL_JSON_FILE_OPERATOR_H_
