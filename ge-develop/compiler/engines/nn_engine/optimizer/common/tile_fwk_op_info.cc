/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tile_fwk_op_info.h"
#include "common/fe_log.h"

namespace fe {
TileFwkOpInfo::TileFwkOpInfo() {}
TileFwkOpInfo::~TileFwkOpInfo() {}

TileFwkOpInfo& TileFwkOpInfo::Instance() {
  static TileFwkOpInfo tile_fwk_op_info;
  return tile_fwk_op_info;
}

bool TileFwkOpInfo::CheckFatbinInfo(const std::string &op_type) const {
  std::lock_guard<std::mutex> lock_guard(fatbin_info_mutex_);
  auto iter = fatbin_info_map_.find(op_type);
  if (iter == fatbin_info_map_.end()) {
    FE_LOGD("Node[type=%s]: fatbin info is not found.", op_type.c_str());
    return false;
  }
  return true;
}

Status TileFwkOpInfo::GetFatbinInfo(const std::string &op_type, const uint64_t &config_key,
                                 SubkernelInfo &subkernel_info) const {
  std::lock_guard<std::mutex> lock_guard(fatbin_info_mutex_);
  FE_LOGD("Node[type=%s]: config key is %lu, fatbin_info_map_ size is %zu.", op_type.c_str(),
          config_key, fatbin_info_map_.size());
  auto iter = fatbin_info_map_.find(op_type);
  if (iter == fatbin_info_map_.end()) {
    FE_LOGE("Node[type=%s]: failed to match the op type from fatbin_info_map_.", op_type.c_str());
    return FAILED;
  }
  FatbinKernelInfoMap fatbin_kernel_info_map = iter->second;
  auto opiter = fatbin_kernel_info_map.find(config_key);
  if (opiter == fatbin_kernel_info_map.end()) {
    FE_LOGE("Node[type=%s]: failed to match config key %lu from fatbin_kernel_info_map.", op_type.c_str(),
            config_key);
    return FAILED;
  }
  subkernel_info = opiter->second;
  return SUCCESS;
}

void TileFwkOpInfo::SetFatbinInfo(const std::string &op_type, const FatbinKernelInfoMap &fatbin_kernel_info_map) {
  std::lock_guard<std::mutex> lock_guard(fatbin_info_mutex_);
  auto iter = fatbin_info_map_.find(op_type);
  if (iter != fatbin_info_map_.end()) {
    FE_LOGD("Node[type=%s]: fatbin_kernel_info_map has been set.", op_type.c_str());
    return;
  }
  fatbin_info_map_.emplace(op_type, fatbin_kernel_info_map);
}

bool TileFwkOpInfo::GetTileFwkOpFlag(const std::string &op_type, bool &tile_fwk_op_flag) const {
  std::lock_guard<std::mutex> lock_guard(tile_fwk_op_flag_mutex_);
  auto iter = tile_fwk_op_flag_map_.find(op_type);
  if (iter == tile_fwk_op_flag_map_.end()) {
    FE_LOGD("Node[type=%s]: failed to match the op type from tile_fwk_op_flag_map.", op_type.c_str());
    return false;
  }
  tile_fwk_op_flag = iter->second;
  return true;
}

void TileFwkOpInfo::SetTileFwkOpFlag(const std::string &op_type, const bool &tile_fwk_op_flag) {
  std::lock_guard<std::mutex> lock_guard(tile_fwk_op_flag_mutex_);
  auto iter = tile_fwk_op_flag_map_.find(op_type);
  if (iter != tile_fwk_op_flag_map_.end()) {
    FE_LOGD("Node[type=%s]: fatbin_kernel_info_map has been set.", op_type.c_str());
    return;
  }
  tile_fwk_op_flag_map_.emplace(op_type, tile_fwk_op_flag);
}
}  // namespace fe
