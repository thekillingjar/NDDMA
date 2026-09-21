/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_running_env/dir_env.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "ge_running_env/path_utils.h"
#include <fstream>
#include <sstream>
namespace ge {
namespace {
void CopyFile(const char *src, const char *dst) {
  if (IsDir(dst)) {
    auto dst_file = PathJoin(dst, BaseName(src).c_str());
    CopyFile(src, dst_file.c_str());
    return;
  }
  std::ifstream src_file(src, std::ios::binary);
  std::ofstream dst_file(dst, std::ios::binary);
  dst_file << src_file.rdbuf();
}
}
void DirEnv::InitDir() {
  InitEngineConfJson();
  InitEngineSo();
  InitOpsKernelInfoStore();
}
void DirEnv::InitEngineConfJson() {
  auto dst_engine_conf_json = engine_config_path_ + "/engine_conf.json";
  std::string src_engine_conf_json = AIR_CODE_DIR;
  src_engine_conf_json += "/compiler/engines/manager/engine_manager/engine_conf.json";
  CopyFile(src_engine_conf_json.c_str(), dst_engine_conf_json.c_str());
}
void DirEnv::InitEngineSo() {
  CopyFile(ENGINE_PATH, engine_path_.c_str());
}
DirEnv::DirEnv() {
  run_root_path_ = GetModelPath();
  engine_path_ = run_root_path_ + "/plugin/nnengine";
  engine_config_path_ = engine_path_ + "/ge_config";
  ops_kernel_info_store_path_ = run_root_path_ + "/plugin/opskernel";

  auto ret = CreateDirectory(engine_config_path_);
  if (ret != GRAPH_SUCCESS) {
    std::cout << "ERROR: Failed to crate dir " << engine_config_path_ << std::endl;
  }

  ret = CreateDirectory(ops_kernel_info_store_path_);
  if (ret != GRAPH_SUCCESS) {
    std::cout << "ERROR: Failed to crate dir " << ops_kernel_info_store_path_ << std::endl;
  }
}
DirEnv &DirEnv::GetInstance() {
  static DirEnv dir_env_ins;
  return dir_env_ins;
}
void DirEnv::InitOpsKernelInfoStore() {
  CopyFile(OPS_KERNEL_LIB_PATH, ops_kernel_info_store_path_.c_str());
}
}

