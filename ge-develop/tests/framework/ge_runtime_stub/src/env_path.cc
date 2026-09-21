/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/env_path.h"
#include "common/path_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
EnvPath::EnvPath() {
  root_path_ = CMAKE_BINARY_DIR;
  temp_path_ = CMAKE_BINARY_DIR "/test_temp_path";
  air_base_path_ = AIR_CODE_DIR;
  ascend_install_path_ = ASCEND_INSTALL_PATH;
}
std::string EnvPath::GetTestCaseTmpPath(const std::string &case_name) const {
  return PathUtils::Join({temp_path_, case_name});
}
std::string EnvPath::GetOrCreateCaseTmpPath(const std::string &case_name) const {
  auto path = GetTestCaseTmpPath(case_name);
  CreateDirectory(path);
  return path;
}
void EnvPath::RemoveRfCaseTmpPath(const std::string &case_name) const {
  auto path = GetTestCaseTmpPath(case_name);
  PathUtils::RemoveDirectories(path);
}
const std::string &EnvPath::GetBinRootPath() const {
  return root_path_;
}

const std::string &EnvPath::GetAirBasePath() const {
  return air_base_path_;
}

const std::string &EnvPath::GetAscendInstallPath() const {
  return ascend_install_path_;
}

const std::string &EnvPath::GetOrCreateTestRootPath() const {
  CreateDirectory(temp_path_);
  return temp_path_;
}
}  // namespace ge
