/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_MEMORY_GROUP_MANAGER_H_
#define INC_MEMORY_GROUP_MANAGER_H_

#include <string>
#include <mutex>
#include "external/ge_common/ge_api_types.h"
#include "framework/common/debug/log.h"
#include "ge/ge_api_error_codes.h"
#include "common/config/configurations.h"

namespace ge {
class MemoryGroupManager {
 public:
  static MemoryGroupManager &GetInstance();

  /*
   *  @ingroup ge
   *  @brief   init memory group manager
   *  @return: SUCCESS or FAILED
   */
  Status Initialize(const NodeConfig &node_config);

  /*
   *  @ingroup ge
   *  @brief   finalize memory group manager
   *  @return: void
   */
  void Finalize();

  /*
   *  @ingroup ge
   *  @brief   is group inited
   *  @return: SUCCESS or FAILED
   */
  static Status DuplicateGrp(const int32_t pid);

  /*
 *  @ingroup ge
 *  @brief   set qs memory group name
 *  @return: void
 */
  void SetQsMemGroupName(const std::string &group_name);

  /*
   *  @ingroup ge
   *  @brief   get qs memory group name
   *  @return: std::string
   */
  const std::string &GetQsMemGroupName() const;

  /*
   *  @ingroup ge
   *  @brief   get remote memory group name
   *  @return: std::string
   */
  std::string GetRemoteMemGroupName(int32_t device_id) const;

  /*
   *  @ingroup ge
   *  @brief   create memory group
   *  @return: SUCCESS or FAILED
   */
  static Status MemGrpCreate(const std::string &group_name);

  /*
   *  @ingroup ge
   *  @brief   create memory group with pre alloc
   *  @return: SUCCESS or FAILED
   */
  static Status MemGrpCreate(const std::string &group_name, uint64_t pre_alloc_size);

  /*
   *  @ingroup ge
   *  @brief   add process into memory group
   *  @return: SUCCESS or FAILED
   */
  static Status MemGrpAddProc(const std::string &group_name, const pid_t pid, bool is_admin, bool is_alloc);

   /*
   *  @ingroup ge
   *  @brief   add process into remote memory group
   *  @return: SUCCESS or FAILED
   */
  Status RemoteMemGrpAddProc(int32_t device_id,
                             const std::string &group_name,
                             const pid_t pid,
                             bool is_admin,
                             bool is_alloc);

  /*
   *  @ingroup ge
   *  @brief   memory group init
   *  @return: SUCCESS or FAILED
   */
  Status MemGroupInit(const NodeConfig &node_config, const std::string &group_name);

  Status SetRemoteGroupCacheConfig(const std::string &remote_group_cache_config);
 private:
  MemoryGroupManager() = default;

  ~MemoryGroupManager() {
    Finalize();
  }

  std::mutex &GetDeviceMutex(int32_t device_id);

  /*
   *  @ingroup ge
   *  @brief   is memory group inited
   *  @return: SUCCESS or FAILED
   */
  bool IsGroupInited(const std::string &group_name);

  /*
   *  @ingroup ge
   *  @brief   init memory group
   *  @return: SUCCESS or FAILED
   */
  Status MemGroupInit(const std::string &group_name);

  /*
   *  @ingroup ge
   *  @brief   init remote memory group
   *  @return: SUCCESS or FAILED
   */
  Status RemoteMemGroupInit(int32_t device_id, const std::string &group_name);
  void SetDefaultRemoteGroupCacheConfig();
  Status ParseRemoteGroupCacheConfig(const std::string &remote_group_cache_config);
  std::string qs_mem_group_name_;

  std::mutex mutex_;
  std::mutex map_mutex_;
  std::map<int32_t, std::mutex> device_mutexs_;
  std::set<std::string> inited_groups_;

  // guard size and pool list
  std::mutex group_cache_mutex_;
  // remote_group_cache_alloc_size is sum of remote_group_cache_pool_list pool size
  uint64_t remote_group_cache_alloc_size_ = 0UL;
  // pair first:pool size, pair second:pool
  std::vector<std::pair<uint64_t, uint32_t>> remote_group_cache_pool_list_;
};
}
#endif // INC_EMEMORY_GROUP_MANAGER_H_
