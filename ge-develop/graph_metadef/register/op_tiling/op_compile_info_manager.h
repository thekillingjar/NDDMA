/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGISTER_OP_TILING_COMPILE_INFO_MANAGER_H_
#define REGISTER_OP_TILING_COMPILE_INFO_MANAGER_H_

#include <string>
#include <mutex>
#include <unordered_map>

#include "register/op_compile_info_base.h"

namespace optiling {
class CompileInfoManager {
public:
  CompileInfoManager(const CompileInfoManager &) = delete;
  CompileInfoManager &operator=(const CompileInfoManager &) = delete;
  static CompileInfoManager& Instance();
  bool HasCompileInfo(const std::string &key);
  CompileInfoPtr GetCompileInfo(const std::string &key);
  void SetCompileInfo(const std::string &key, CompileInfoPtr compile_info_ptr);

private:
  CompileInfoManager();
  ~CompileInfoManager();
  mutable std::mutex compile_info_mutex_;
  std::unordered_map<std::string, CompileInfoPtr> compile_info_map_;
};

class CompileInfoCache {
public:
  CompileInfoCache(const CompileInfoCache &) = delete;
  CompileInfoCache &operator=(const CompileInfoCache &) = delete;
  static CompileInfoCache& Instance();
  bool HasCompileInfo(const std::string &key);
  void* GetCompileInfo(const std::string &key);
  void SetCompileInfo(const std::string &key, void* value);

private:
  CompileInfoCache();
  ~CompileInfoCache();
  mutable std::mutex compile_info_mutex_;
  std::unordered_map<std::string, void *> compile_info_map_;
};
}  // namespace optiling
#endif  // REGISTER_OP_TILING_COMPILE_INFO_MANAGER_H_
