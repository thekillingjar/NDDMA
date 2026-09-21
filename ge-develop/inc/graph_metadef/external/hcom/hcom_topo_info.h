/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_
#define METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_

#include <unordered_map>
#include <mutex>
#include "ge_common/ge_api_types.h"

namespace ge {
static constexpr uint32_t COMM_MESH = 0b1U;
static constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
static constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
static constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
class HcomTopoInfo {
 public:
  enum class TopoLevel {
    L0 = 0,
    L1,
    L2,
    MAX,
  };
  struct TopoLevelDesc {
    uint32_t comm_sets;
    uint32_t rank_size;
  };
  using TopoDescs = TopoLevelDesc[static_cast<int32_t>(TopoLevel::MAX)];
  struct TopoInfo {
    int64_t rank_size;
    void *notify_handle;
    TopoDescs topo_level_descs;
    uint64_t local_window_size = 0;
  };
  static HcomTopoInfo &Instance();
  bool TopoInfoHasBeenSet(const char_t *group);
  bool TryGetGroupTopoInfo(const char_t *group, TopoInfo &info);
  Status SetGroupTopoInfo(const char_t *group, const TopoInfo &info);
  Status GetGroupRankSize(const char_t *group, int64_t &rank_size);
  TopoDescs *GetGroupTopoDesc(const char_t *group);
  Status GetGroupNotifyHandle(const char_t *group, void *&notify_handle);
  void UnsetGroupTopoInfo(const char_t *group) {
    const std::lock_guard<std::mutex> lock(mutex_);
    (void) rank_info_.erase(group);
  }

  Status SetGroupOrderedStream(const int32_t device_id, const char_t *group, void *stream);
  Status GetGroupOrderedStream(const int32_t device_id, const char_t *group, void *&stream);
  void UnsetGroupOrderedStream(const int32_t device_id, const char_t *group);
  Status GetGroupLocalWindowSize(const char_t *group, uint64_t &local_window_size);
 private:
  HcomTopoInfo() = default;
  ~HcomTopoInfo() = default;
  std::unordered_map<std::string, TopoInfo> rank_info_;
  std::mutex mutex_;
  std::unordered_map<int32_t, std::unordered_map<std::string, void*>> device_id_to_group_to_ordered_stream_; // 通信域保序流
};
}

#endif // METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_
