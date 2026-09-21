/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "cann_kb_adapter.h"
#include "common/configuration.h"
namespace fe {
namespace {
constexpr char const *CANN_KB_PLUGIN = "libcann_kb.so";
constexpr char const *CANN_KB_SEARCH_FUNC_NAME = "CannKbSearch";
constexpr char const *CANN_KB_INIT_FUNC_NAME = "CannKbInit";
constexpr char const *CANN_KB_FINALIZE_FUNC_NAME = "CannKbFinalize";
}
 
CannKBUtils::CannKBUtils() {
  init_flag_ = false;
  cann_kb_plugin_manager_ = nullptr;
  cann_kb_init_func_ = nullptr;
  cann_kb_finalize_func_ = nullptr;
  cann_kb_search_func_ = nullptr;
}
 
CannKBUtils::~CannKBUtils() {
  if (!init_flag_) {
    return;
  }
  if (cann_kb_plugin_manager_ != nullptr) {
    cann_kb_plugin_manager_->CloseHandle();
  }
}
 
CannKBUtils &CannKBUtils::Instance() {
  static CannKBUtils cann_kb_utils;
  return cann_kb_utils;
}
 
bool CannKBUtils::InitCannKb() {
  if (init_flag_) {
    return true;
  }
  // 1. open the cann kb fusion plugin
  std::string plugin_path = fe::Configuration::Instance("AIcoreEngine").GetLibPath() + CANN_KB_PLUGIN;
  FE_MAKE_SHARED(cann_kb_plugin_manager_ = std::make_shared<PluginManager>(plugin_path), return FAILED);
  FE_CHECK(cann_kb_plugin_manager_ == nullptr,
           REPORT_FE_ERROR("[GraphOpt][Init][InitLxFusPlg] Create lx fusion plugin manager ptr failed."),
           return false);
  if (cann_kb_plugin_manager_->OpenPlugin(plugin_path) != SUCCESS) {
    FE_LOGE("Failed to open %s.", plugin_path.c_str());
    return false;
  }
  FE_LOGI("Load cann_kb so[%s] success.", plugin_path.c_str());
  // 2. get the function pointer
  Status ret = cann_kb_plugin_manager_->GetFunction<CannKb::CANN_KB_STATUS, const std::string &,
      const std::map<std::string, std::string> &, std::vector<std::map<std::string, std::string>> &>(
      CANN_KB_SEARCH_FUNC_NAME, cann_kb_search_func_);
  if (ret != SUCCESS) {
    FE_LOGE("Failed to get the %s in the plugin", CANN_KB_SEARCH_FUNC_NAME);
    (void)cann_kb_plugin_manager_->CloseHandle();
    return false;
  }
  ret = cann_kb_plugin_manager_->GetFunction<CannKb::CANN_KB_STATUS,
    const std::map<std::string, std::string> &,const std::map<std::string, std::string> &>(
      CANN_KB_INIT_FUNC_NAME, cann_kb_init_func_);
  if (ret != SUCCESS) {
    FE_LOGE("Failed to get the %s in the plugin", CANN_KB_INIT_FUNC_NAME);
    (void)cann_kb_plugin_manager_->CloseHandle();
    return false;
  }
  ret = cann_kb_plugin_manager_->GetFunction<CannKb::CANN_KB_STATUS>(
      CANN_KB_FINALIZE_FUNC_NAME, cann_kb_finalize_func_);
  if (ret != SUCCESS) {
    FE_LOGE("Failed to get the %s in the plugin", CANN_KB_FINALIZE_FUNC_NAME);
    (void)cann_kb_plugin_manager_->CloseHandle();
    return false;
  }
  init_flag_ = true;
  FE_LOGI("Get the %s, %s, %s in the plugin successfully.", CANN_KB_SEARCH_FUNC_NAME,
    CANN_KB_INIT_FUNC_NAME, CANN_KB_FINALIZE_FUNC_NAME);
  return true;
}
 
CannKb::CANN_KB_STATUS CannKBUtils::RunCannKbSearch(const std::string &infoDict,
    const std::map<std::string, std::string> &searchConfig,
    std::vector<std::map<std::string, std::string>> &searchResult) const {
  CannKb::CANN_KB_STATUS ret = cann_kb_search_func_(infoDict, searchConfig, searchResult);
  return ret;
}
 
CannKb::CANN_KB_STATUS CannKBUtils::CannKbInit(const std::map<std::string, std::string> &sysConfig,
                                         const std::map<std::string, std::string> &loadConfig) {
    CannKb::CANN_KB_STATUS ret = CannKb::CANN_KB_STATUS::CANN_KB_CHECK_FAILED;
    if(cann_kb_init_func_ != nullptr)
    {
       ret = cann_kb_init_func_(sysConfig, loadConfig);
    }
    return ret;
}
 
CannKb::CANN_KB_STATUS CannKBUtils::CannKbFinalize() {
    CannKb::CANN_KB_STATUS ret = CannKb::CANN_KB_STATUS::CANN_KB_CHECK_FAILED;
    if(cann_kb_finalize_func_ != nullptr)
    {
       ret = cann_kb_finalize_func_();
    }
    return ret;
}
}