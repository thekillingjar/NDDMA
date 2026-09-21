/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin_manager.h"
#include <fstream>
#include <iostream>
#include <memory>
#include "common/util/json_util.h"

namespace fe {

/*
*  @ingroup fe
*  @brief   release resources needed by TbeCompilerAdapter,
*  such as dlclose so files
*           and free tbe resources etc.
*  @return  SUCCESS or FAILED
*/
Status PluginManager::CloseHandle() {
  FE_CHECK(handle == nullptr, FE_LOGW("Handle of %s is nullptr.", so_name.c_str()), return FAILED);
  if (mmDlclose(handle) != 0) {
    FE_LOGW("Failed to close handle of %s.", so_name.c_str());
    return FAILED;
  }

  return SUCCESS;
}

std::string MockerRealPath(const std::string &path) {
  if (path.empty()) {
    FE_LOGI("path string is nullptr.");
    return "";
  }
  if (path.size() >= PATH_MAX) {
    FE_LOGI("file path %s is too long! ", path.c_str());
    return "";
  }

  // PATH_MAX is the system marco，indicate the maximum length for file path
  // pclint check，one param in stack can not exceed 1K bytes
  char *resoved_path = new(std::nothrow) char[PATH_MAX];
  if (resoved_path == nullptr) {
    FE_LOGI("New resoved_path failed. ");
    return "";
  }
  (void)memset_s(resoved_path, PATH_MAX, 0, PATH_MAX);

  std::string res = "";

  // path not exists or not allowed to read，return nullptr
  // path exists and readable, return the resoved path
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    res = resoved_path;
  } else {
    FE_LOGI("Path %s is not exist.", path.c_str());
  }

  delete[] resoved_path;
  return res;
}

/*
*  @ingroup fe
*  @brief   initial resources needed by TbeCompilerAdapter,
*  such as dlopen so files
*           and load function symbols etc.
*  @return  SUCCESS or FAILED
*/
Status PluginManager::OpenPlugin(const string& path) {
  FE_LOGD("Mocker: Start to load so file[%s].", path.c_str());
  string temp_path = path;
  const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
  string ascend_install_path;
  if (ascend_custom_path_ptr != nullptr) {
    ascend_install_path = MockerRealPath(string(ascend_custom_path_ptr));
  } else {
    ascend_install_path = "/mnt/d/Ascend";
  }
  if (path.find("libte_fusion.so") != string::npos) {
    temp_path = ascend_install_path + "/compiler/lib64/libte_fusion.so";
  }

  std::string real_path = MockerRealPath(temp_path);
  if (real_path.empty()) {
    FE_LOGW("The file path [%s] is not valid", path.c_str());
    return FAILED;
  }

  // return when dlopen is failed
  handle = mmDlopen(real_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  FE_CHECK(handle == nullptr, REPORT_FE_ERROR("[FEInit][OpPluginSo] Fail to load so file %s, error message is %s.",
                                              real_path.c_str(), mmDlerror()),
           return FAILED);

  FE_LOGD("Mocker: Finish loading so file[%s].", path.c_str());
  return SUCCESS;
}

/*
*  @ingroup fe
*  @brief   return tbe so name
*  @return  tbe so name
*/
const string PluginManager::GetSoName() const { return so_name; }

}  // namespace fe
