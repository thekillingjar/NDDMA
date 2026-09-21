/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_PATH_HELPER_H
#define AIR_CXX_PATH_HELPER_H
#include <string>
#include "path_utils.h"
namespace ge {
class EnvPath {
 public:
  EnvPath();
  std::string GetTestCaseTmpPath(const std::string &case_name) const;
  std::string GetOrCreateCaseTmpPath(const std::string &case_name) const;

  void RemoveRfCaseTmpPath(const std::string &case_name) const;
  const std::string &GetBinRootPath() const;
  const std::string &GetAirBasePath() const;
  const std::string &GetAscendInstallPath() const;

 private:
  const std::string &GetOrCreateTestRootPath() const;

 private:
  std::string root_path_;
  std::string temp_path_;
  std::string air_base_path_;
  std::string ascend_install_path_;
};
}  // namespace ge
#endif  // AIR_CXX_PATH_HELPER_H
