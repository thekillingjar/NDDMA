/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/l2_stream_info.h"
#include "common/fe_log.h"

namespace fe {
StreamL2Info::StreamL2Info() {}
StreamL2Info::~StreamL2Info() {}

StreamL2Info& StreamL2Info::Instance() {
  static StreamL2Info stream_l2_info;
  return stream_l2_info;
}

Status StreamL2Info::GetStreamL2Info(const int64_t &stream_id, const std::string &node_name, TaskL2Info *&l2_data,
                                     const std::string &batch_label) {
  std::lock_guard<std::mutex> lock_guard(stream_l2_mutex_);
  FE_LOGD("Node[name=%s]: stream_id is %d, stream_l2_map_ size is %zu.", node_name.c_str(),
          stream_id, stream_l2_map_.size());
  std::string stream_id_str = std::to_string(stream_id);
  std::string stream_l2_map_key = stream_id_str + batch_label;
  auto iter = stream_l2_map_.find(stream_l2_map_key);
  if (iter == stream_l2_map_.end()) {
    FE_LOGW("Node[name=%s]: Failed to find the key %s in stream_l2_map_.", node_name.c_str(),
            stream_l2_map_key.c_str());
    return FAILED;
  }
  TaskL2InfoMap &l2_info_map = iter->second;
  auto opiter = l2_info_map.find(node_name);
  FE_LOGD("Node[name=%s]: l2_info_map size is %zu.", node_name.c_str(), l2_info_map.size());
  if (opiter == l2_info_map.end()) {
    FE_LOGW("Node[name=%s]: Unable to find the node from l2_info_map.", node_name.c_str());
    return FAILED;
  }
  l2_data = &(opiter->second);
  return SUCCESS;
}

Status StreamL2Info::SetStreamL2Info(const int64_t &stream_id, TaskL2InfoMap &l2_alloc_res,
                                     const std::string &batch_label) {
  std::lock_guard<std::mutex> lock_guard(stream_l2_mutex_);
  std::string stream_id_str = std::to_string(stream_id);
  std::string stream_l2_map_key = stream_id_str + batch_label;
  const auto it = stream_l2_map_.find(stream_l2_map_key);
  if (it != stream_l2_map_.end()) {
    FE_LOGD("SetStreamL2InfoMap: steam_map has been set with key %s.", stream_l2_map_key.c_str());
    it->second = l2_alloc_res;
    return fe::SUCCESS;
  }

  for (TaskL2InfoMap::iterator iter = l2_alloc_res.begin(); iter != l2_alloc_res.end(); ++iter) {
    FE_LOGD("SetStreamL2InfoMap node name is %s.", iter->first.c_str());
    TaskL2Info *l2task = &(iter->second);
    l2task->isUsed = 0;
  }
  FE_LOGD("l2_alloc_res size is %zu bytes.", l2_alloc_res.size());
  stream_l2_map_[stream_l2_map_key] = l2_alloc_res;
  FE_LOGD("SetStreamL2InfoMap key is %s.", stream_l2_map_key.c_str());
  return SUCCESS;
}
}  // namespace fe
