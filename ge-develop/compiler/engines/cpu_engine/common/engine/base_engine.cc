/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_engine.h"

#include "base/err_msg.h"
#include "config/config_file.h"
#include "error_code/error_code.h"
#include "util/log.h"
#include "util/util.h"
#include "util/constant.h"

using namespace std;
using namespace ge;

namespace {
const string kInitConfigFileRelativePath = "config/init.conf";
}

namespace aicpu {
Status BaseEngine::LoadConfigFile(const void *instance) const {
  string real_config_file_path = GetSoPath(instance);
  real_config_file_path += kInitConfigFileRelativePath;
  AICPUE_LOGI("Configuration file path is %s.", real_config_file_path.c_str());

  aicpu::State ret = ConfigFile::GetInstance().LoadFile(real_config_file_path);
  if (ret.state != ge::SUCCESS) {
    map<string, string> err_map;
    err_map["filename"] = real_config_file_path;
    err_map["reason"] = real_config_file_path;
    string report_err_code_str = GetViewErrorCodeStr(LOAD_INIT_CONFIG_FILE_FAILED);
    REPORT_PREDEFINED_ERR_MSG(report_err_code_str.c_str(),
                              std::vector<const char *>({"filename", "reason"}),
                              std::vector<const char *>({real_config_file_path.c_str(), ret.msg.c_str()}));
    AICPUE_LOGE("Configuration file load failed, %s.", ret.msg.c_str());
    return LOAD_INIT_CONFIGURE_FILE_FAILED;
  }
  return SUCCESS;
}
}  // namespace aicpu
