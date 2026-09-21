/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_tiling/op_compile_info_manager.h"

namespace optiling {
CompileInfoManager::CompileInfoManager() {}
CompileInfoManager::~CompileInfoManager() {}

CompileInfoManager& CompileInfoManager::Instance() {
  static CompileInfoManager compile_info_manager_instance;
  return compile_info_manager_instance;
}

bool CompileInfoManager::HasCompileInfo(const std::string &key) {
  return this->compile_info_map_.find(key) != this->compile_info_map_.end();
}

CompileInfoPtr CompileInfoManager::GetCompileInfo(const std::string &key) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  const auto iter = this->compile_info_map_.find(key);
  if (iter == this->compile_info_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

void CompileInfoManager::SetCompileInfo(const std::string &key, CompileInfoPtr compile_info_ptr) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  (void)this->compile_info_map_.emplace(key, compile_info_ptr);
}

CompileInfoCache::CompileInfoCache() {}
CompileInfoCache::~CompileInfoCache() {}

CompileInfoCache& CompileInfoCache::Instance() {
  static CompileInfoCache compile_info_cache_instance;
  return compile_info_cache_instance;
}

bool CompileInfoCache::HasCompileInfo(const std::string &key) {
  return this->compile_info_map_.find(key) != this->compile_info_map_.end();
}

void* CompileInfoCache::GetCompileInfo(const std::string &key) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  const auto iter = this->compile_info_map_.find(key);
  if (iter == this->compile_info_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

void CompileInfoCache::SetCompileInfo(const std::string &key, void *value) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  (void)this->compile_info_map_.emplace(key, value);
}
}  // namespace optiling
