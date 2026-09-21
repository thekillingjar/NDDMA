/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_GE_TBE_PLUGIN_MANAGER_H_
#define GE_COMMON_GE_TBE_PLUGIN_MANAGER_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ge/ge_api_error_codes.h"
#include "graph_metadef/register/register.h"

namespace ge {
using SoHandlesVec = std::vector<void *>;

class TBEPluginManager {
 public:
  Status Finalize();

  // Get TBEPluginManager singleton instance
  static TBEPluginManager& Instance();

  static std::string GetPath();

  static void InitPreparation(const std::map<std::string, std::string> &options);

  void LoadPluginSo(const std::map< std::string, std::string> &options);

 private:
  TBEPluginManager() = default;
  ~TBEPluginManager() = default;
  Status ClearHandles_();

  static void ProcessSoFullName(std::vector<std::string> &file_list, std::string &caffe_parser_path,
                                const std::string &full_name, const std::string &caffe_parser_so_suff);
  static void FindParserUsedSo(const std::string &path, std::vector<std::string> &file_list,
                               std::string &caffe_parser_path, const uint32_t recursive_depth = 0U);
  static void GetOpPluginSoFileList(const std::string &path, std::vector<std::string> &file_list,
                                    std::string &caffe_parser_path);
  static void GetCustomOpPath(std::string &customop_path);
  void LoadCustomOpLib();

  SoHandlesVec handles_vec_;
  static std::map<std::string, std::string> options_;
};
}  // namespace ge

#endif  // GE_COMMON_GE_TBE_PLUGIN_MANAGER_H_
