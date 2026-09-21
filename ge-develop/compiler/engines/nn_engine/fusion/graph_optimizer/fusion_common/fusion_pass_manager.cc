/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/fusion_common/fusion_pass_manager.h"
#include <dirent.h>
#include <memory>
#include <regex>
#include <string>
#include "common/configuration.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/util/constants.h"
#include "common/util/json_util.h"

namespace fe {
FusionPassManager::FusionPassManager() : init_flag_(false) {}

FusionPassManager::~FusionPassManager() {}

Status FusionPassManager::Initialize(const std::string &engine_name) {
  if (init_flag_) {
    FE_LOGW("FusionPassManager has been initialized.");
    return SUCCESS;
  }
  Status ret = InitFusionPassPlugin(engine_name);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Init passes failed.");
    CloseFusionPassPlugin();
    return ret;
  }

  FE_LOGI("Fusion pass manager initialize successfully.");
  init_flag_ = true;
  return SUCCESS;
}

Status FusionPassManager::Finalize() {
  if (!init_flag_) {
    FE_LOGW("FusionPassManager has not been initialized.");
    return SUCCESS;
  }

  CloseFusionPassPlugin();
  init_flag_ = false;
  FE_LOGD("FusionPassManager finalize successfully.");
  return SUCCESS;
}

void FusionPassManager::CloseFusionPassPlugin() {
  if (fusion_pass_plugin_manager_vec_.empty()) {
    return;
  }
  for (PluginManagerPtr &plugin_manager_ptr : fusion_pass_plugin_manager_vec_) {
    if (plugin_manager_ptr == nullptr) {
      continue;
    }
    if (plugin_manager_ptr->CloseHandle() != SUCCESS) {
      FE_LOGW("Failed to close so handle [%s].", plugin_manager_ptr->GetSoName().c_str());
    }
  }
  fusion_pass_plugin_manager_vec_.clear();
}

Status FusionPassManager::InitFusionPassPlugin(const std::string &engine_name) {
  std::string custom_pass_file_path;
  std::string builtin_pass_file_path;
  std::vector<string> custom_pass_file;
  std::vector<string> builtin_pass_file;
  // Get pass directory from configuration
  if (Configuration::Instance(engine_name).GetCustomPassFilePath(custom_pass_file_path) == SUCCESS) {
    if (custom_pass_file_path.empty()) {
      FE_LOGI("Custom pass file path is null.");
    } else {
      Status ret = OpenPassPath(custom_pass_file_path, custom_pass_file);
      if (ret != SUCCESS) {
        FE_LOGI("Unable to load custom pass file path: %s.", custom_pass_file_path.c_str());
      }
    }
  }
  if (Configuration::Instance(engine_name).GetBuiltinPassFilePath(builtin_pass_file_path) == SUCCESS) {
    if (builtin_pass_file_path.empty()) {
      FE_LOGI("Built-in pass file path is null.");
    } else {
      Status ret = OpenPassPath(builtin_pass_file_path, builtin_pass_file);
      if (ret != SUCCESS) {
        FE_LOGW("Failed to load built-in pass file path %s.", builtin_pass_file_path.c_str());
        return SUCCESS;
      }
    }
  }
  return SUCCESS;
}

Status FusionPassManager::OpenPassPath(const string &file_path, vector<string> &get_pass_files) {
  FE_LOGD("Start to load fusion pass files in path %s.", file_path.c_str());

  std::string real_path = RealPath(file_path);
  if (real_path.empty()) {
    FE_LOGI("File path[%s] is not valid.", file_path.c_str());
    return FAILED;
  }
  DIR *dir;
  struct dirent *dirp = nullptr;
  char *file_suffix = const_cast<char *>(".so");

  dir = opendir(real_path.c_str());
  FE_CHECK(dir == nullptr,
           REPORT_FE_ERROR("[GraphOpt][Init][OpenPassPath] Failed to open directory %s.", real_path.c_str()),
           return FAILED);
  while ((dirp = readdir(dir)) != nullptr) {
    if (dirp->d_name[0] == '.') {
      continue;
    }
    if (strlen(dirp->d_name) <= strlen(file_suffix)) {
      continue;
    }
    if (strcmp(&(dirp->d_name)[strlen(dirp->d_name) - strlen(file_suffix)], file_suffix) == 0) {
      get_pass_files.push_back(real_path + "/" + dirp->d_name);
    }
  }
  closedir(dir);

  if (get_pass_files.empty()) {
    FE_LOGI("No fusion pass, so files are read from path %s", real_path.c_str());
    return SUCCESS;
  }

  for (const auto &pass_file : get_pass_files) {
    FE_LOGD("Begin to open path for file [%s].", pass_file.c_str());
    PluginManagerPtr plugin_manager_ptr = nullptr;
    FE_MAKE_SHARED(plugin_manager_ptr = std::make_shared<PluginManager>(pass_file), return FAILED);
    FE_CHECK(plugin_manager_ptr == nullptr,
             REPORT_FE_ERROR("[GraphOpt][Init][OpenPassPath] Failed to create PluginManagerPtr."), return FAILED);
    if (plugin_manager_ptr->OpenPlugin(pass_file) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Init][OpenPassPath] Failed to open the fusion pass so: [%s].", pass_file.c_str());
      return FAILED;
    }
    fusion_pass_plugin_manager_vec_.push_back(plugin_manager_ptr);
    FE_LOGI("Fusion pass so [%s] loaded successfully.", pass_file.c_str());
  }

  return SUCCESS;
}
}  // namespace fe
