/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/ffts_plugin_manager.h"
#include <fstream>
#include <iostream>
#include <memory>
#include "common/util/json_util.h"
#include "inc/ffts_error_codes.h"
#include "mmpa/mmpa_api.h"

namespace ffts {
Status PluginManager::CloseHandle() {
  FFTS_CHECK(handle == nullptr, FFTS_LOGW("Handle of %s is nullptr.", so_name.c_str()), return FAILED);
  if (mmDlclose(handle) != 0) {
    FFTS_LOGW("Failed to close handle of %s.", so_name.c_str());
    return FAILED;
  }

  return SUCCESS;
}

std::string MockerRealPath(const std::string &path) {
  if (path.empty()) {
    FFTS_LOGI("path string is nullptr.");
    return "";
  }
  if (path.size() >= PATH_MAX) {
    FFTS_LOGI("file path %s is too long! ", path.c_str());
    return "";
  }

  // PATH_MAX is the system marco，indicate the maximum length for file path
  // pclint check，one param in stack can not exceed 1K bytes
  char *resoved_path = new(std::nothrow) char[PATH_MAX];
  if (resoved_path == nullptr) {
    FFTS_LOGI("New resoved_path failed. ");
    return "";
  }
  (void)memset_s(resoved_path, PATH_MAX, 0, PATH_MAX);

  std::string res = "";

  // path not exists or not allowed to read，return nullptr
  // path exists and readable, return the resoved path
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    res = resoved_path;
  } else {
    FFTS_LOGI("Path %s is not exist.", path.c_str());
  }

  delete[] resoved_path;
  return res;
}

Status PluginManager::OpenPlugin(const string& path) {
  FFTS_LOGD("Mocker: Start to load so file[%s].", path.c_str());
  string temp_path = path;
  const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
  string ascend_install_path;
  if (ascend_custom_path_ptr != nullptr) {
    ascend_install_path = MockerRealPath(string(ascend_custom_path_ptr));
  } else {
    ascend_install_path = "/mnt/d/Ascend";
  }
  if (path.find("libgraph_tuner.so") != string::npos) {
    temp_path = ascend_install_path + "/compiler/lib64/plugin/opskernel/libgraph_tuner.so";
  } else if (path.find("libascend_sch_policy_pass.so") != string::npos) {
    temp_path = ascend_install_path + "/compiler/lib64/plugin/opskernel/libascend_sch_policy_pass.so";
  }

  std::string real_path = MockerRealPath(temp_path);
  if (real_path.empty()) {
    FFTS_LOGW("The file path [%s] is not valid", path.c_str());
    return FAILED;
  }
  FFTS_LOGD("Mocker: real so path[%s].", real_path.c_str());
  // return when dlopen is failed
  handle = mmDlopen(real_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  FFTS_CHECK(handle == nullptr, REPORT_FFTS_ERROR("[FEInit][OpPluginSo] Fail to load so file %s, error message is %s.",
                                                  real_path.c_str(), mmDlerror()),
           return FAILED);

  FFTS_LOGD("Mocker: Finish loading so file[%s].", path.c_str());
  return SUCCESS;
}

const string PluginManager::GetSoName() const { return so_name; }

}  // namespace ffts
