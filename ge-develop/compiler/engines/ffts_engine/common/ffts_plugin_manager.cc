/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ffts_plugin_manager.h"
#include <fstream>
#include <iostream>
#include <memory>
#include "inc/ffts_utils.h"

namespace ffts {
Status PluginManager::CloseHandle() {
  FFTS_CHECK(handle == nullptr, FFTS_LOGW("Handle is a nullptr."), return FAILED);
  if (dlclose(handle) != 0) {
    FFTS_LOGW("Failed to close handle of %s.", so_name.c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status PluginManager::OpenPlugin(const string& path) {
  FFTS_LOGD("Start loading so file [%s].", path.c_str());

  std::string real_path = RealPath(path);
  if (real_path.empty()) {
    FFTS_LOGW("The file path [%s] is not valid.", path.c_str());
    return FAILED;
  }

  // return when dlopen is failed
  handle = dlopen(real_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  FFTS_CHECK(handle == nullptr,
             REPORT_FFTS_ERROR("[FFTS Init][OpPluginSo] Failed to load so file %s, error message is %s",
             real_path.c_str(), dlerror()), return FAILED);

  FFTS_LOGD("Finished loading so file [%s].", path.c_str());
  return SUCCESS;
}

/*
*  @ingroup ffts
*  @brief   return tbe so name
*  @return  so name
*/
const string PluginManager::GetSoName() const { return so_name; }
}  // namespace fe
