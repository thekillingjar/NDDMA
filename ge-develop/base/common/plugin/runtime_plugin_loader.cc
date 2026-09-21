/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin/runtime_plugin_loader.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/util.h"
#include "common/plugin/ge_make_unique_util.h"

namespace ge {
RuntimePluginLoader &RuntimePluginLoader::GetInstance() {
  static RuntimePluginLoader instance;
  return instance;
}

ge::graphStatus RuntimePluginLoader::Initialize(const std::string &path_base) {
  std::string lib_path = "plugin/engines/runtime";
  const std::string path = path_base + lib_path;

  plugin_manager_ = ge::MakeUnique<ge::PluginManager>();
  GE_CHECK_NOTNULL(plugin_manager_);
  // global so should be set nodelete flag in case of core dump in dlclose func at singleton destructor
  const int32_t flags = static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
      static_cast<uint32_t>(MMPA_RTLD_GLOBAL) | static_cast<uint32_t>(MMPA_RTLD_NODELETE));
  GE_CHK_STATUS_RET(plugin_manager_->LoadWithFlags(path, flags), "[RT_Plugin][Libs]Failed, lib_path=%s.", path.c_str());
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge