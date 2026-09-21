/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_CONFIG_FILE_H_
#define AICPU_CONFIG_FILE_H_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "error_code/error_code.h"

namespace aicpu {
class ConfigFile {
 public:
  /**
   * Get instance
   * @return ConfigFile reference
   */
  static ConfigFile &GetInstance();

  /**
   * Destructor
   */
  virtual ~ConfigFile() = default;

  /**
   * Load config file in specific path
   * @param file_name, file path
   * @return whether read file successfully
   */
  aicpu::State LoadFile(const std::string &file_name);
  /**
   * split string
   * @param str, src string
   * @param result, vector used to store result
   * @param blacklist, engine prohibit
   */
  void SplitValue(
      const std::string &str, std::vector<std::string> &result,
      const std::set<std::string> &blacklist = std::set<std::string>()) const; //lint !e562

  /**
   * Get value by specific key
   * @param key, config key name
   * @param value, config key's value
   * @return config file's value of specific key
   */
  bool GetValue(const std::string &key, std::string &value) const;

  /**
   * Set value by specific key
   * @param key, config key name
   * @param value, config key's value
   */
  void SetValue(const std::string &key, std::string &value);

  // Copy prohibit
  ConfigFile(const ConfigFile &) = delete;
  // Copy prohibit
  ConfigFile &operator=(const ConfigFile &) = delete;
  // Move prohibit
  ConfigFile(ConfigFile &&) = delete;
  // Move prohibit
  ConfigFile &operator=(ConfigFile &&) = delete;

 private:
  // Constructor
  ConfigFile() : is_load_file_(false) {}
  /**
   * Trim string
   * @param source, source string to be processed
   * @param delims, delimit exist in source string e.g. space
   * @return trim source string result
   */
  std::string Trim(const std::string &source, const char delims = ' ') const;
  // store config item by key-value pair
  std::map<std::string, std::string> data_map_;

  bool is_load_file_;
};
}  // namespace aicpu

#endif  // AICPU_CONFIG_FILE_H_
