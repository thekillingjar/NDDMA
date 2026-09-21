/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/mem_grp/memory_group_manager.h"
#include <regex>
#include "graph/ge_context.h"
#include "common/config/configurations.h"
#include "common/math/math_util.h"
#include "framework/common/util.h"
#include "mmpa/mmpa_api.h"
#include "common/data_flow/event/proxy_event_manager.h"
#include "acl/acl.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr int32_t kTimeout = 3000;
constexpr uint64_t kNpuMaxGroupMemSize = 35337011UL;  // 33.7 * 1024 * 1024 KB
constexpr uint64_t kDefaultRemoteGroupCacheAllocSize = 10 * 1024 * 1024UL;  // 10 * 1024 * 1024 KB
constexpr uint32_t kAddGroupTimeout = 10000;
}  // namespace
MemoryGroupManager &MemoryGroupManager::GetInstance() {
  static MemoryGroupManager instance;
  return instance;
}

Status MemoryGroupManager::MemGrpCreate(const std::string &group_name) {
  rtMemGrpConfig_t cfg = {};
  cfg.addGrpTimeout = kAddGroupTimeout;
  GE_CHK_RT_RET(rtMemGrpCreate(group_name.c_str(), &cfg));
  GELOGD("[Create][MemoryGrp] success, group name=%s.", group_name.c_str());
  return SUCCESS;
}

Status MemoryGroupManager::MemGrpCreate(const std::string &group_name, uint64_t pre_alloc_size) {
  rtMemGrpConfig_t cfg = {};
  cfg.maxMemSize = pre_alloc_size;
  cfg.cacheAllocFlag = 1;
  cfg.addGrpTimeout = kAddGroupTimeout;
  GE_CHK_RT_RET(rtMemGrpCreate(group_name.c_str(), &cfg));
  GELOGD("[Create][MemoryGrp] success, group name=%s.", group_name.c_str());
  return SUCCESS;
}

Status MemoryGroupManager::MemGrpAddProc(const std::string &group_name, const pid_t pid, bool is_admin, bool is_alloc) {
  rtMemGrpShareAttr_t attr = {};
  attr.admin = is_admin ? 1 : 0;
  attr.alloc = is_alloc ? 1 : 0;
  attr.read = 1;
  attr.write = 1;
  const auto ret = rtMemGrpAddProc(group_name.c_str(), pid, &attr);
  if (ret != RT_ERROR_NONE && ret != ACL_ERROR_RT_REPEATED_INIT) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemGrpAddProc fail, ret: 0x%X", static_cast<uint32_t>(ret));
    GELOGE(RT_FAILED, "[rtMemGrpAddProc] failed, rt_err = %d, group_name is[%s]", ret, group_name.c_str());
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  GELOGD("[Add][Process] success.");
  return SUCCESS;
}

Status MemoryGroupManager::MemGroupInit(const NodeConfig &node_config, const std::string &group_name) {
  for (const auto &device_config : node_config.device_list) {
    if (device_config.device_type == CPU) {
      GE_CHK_STATUS_RET(MemGroupInit(group_name),
                        "Memory group init failed, group name = %s.",
                        group_name.c_str());
    }
  }
  return SUCCESS;
}

void MemoryGroupManager::SetDefaultRemoteGroupCacheConfig() {
  std::unique_lock<std::mutex> lk(group_cache_mutex_);
  remote_group_cache_alloc_size_ = kDefaultRemoteGroupCacheAllocSize;
  remote_group_cache_pool_list_ = {{kDefaultRemoteGroupCacheAllocSize, 0}};
}

Status MemoryGroupManager::ParseRemoteGroupCacheConfig(const std::string &remote_group_cache_config) {
  constexpr int64_t kUpperLimit = 256 * 1024 * 1024 * 1024L;  // byte
  constexpr int64_t kLowerLimit = 1024L;  // byte
  constexpr uint32_t kBitShift1024 = 10U;
  std::vector<std::pair<uint64_t, uint32_t>> remote_group_cache_pool_list;
  std::vector<std::string> pool_config_list = StringUtils::Split(remote_group_cache_config, ',');
  uint64_t group_size = 0;
  for (const auto &pool_config : pool_config_list) {
    int64_t pool_size = 0;
    int32_t pool_alloc_limit = 0;
    auto find_ret = pool_config.find(':');
    if (find_ret != std::string::npos) {
      auto pool_size_str = pool_config.substr(0, find_ret);
      auto pool_alloc_limit_str = pool_config.substr(find_ret + 1);
      GE_CHK_STATUS_RET(ConvertToInt64(pool_size_str, pool_size),
                        "parse pool_size[%s] failed, remote_group_cache_config=%s.", pool_size_str.c_str(),
                        remote_group_cache_config.c_str());
      GE_CHK_STATUS_RET(ConvertToInt32(pool_alloc_limit_str, pool_alloc_limit),
                        "parse pool_alloc_limit[%s] failed, remote_group_cache_config=%s.",
                        pool_alloc_limit_str.c_str(), remote_group_cache_config.c_str());
      GE_CHK_BOOL_RET_STATUS(pool_alloc_limit <= pool_size, PARAM_INVALID,
                             "pool alloc limit[%d] must less than pool size[%ld]", pool_alloc_limit, pool_size);
      GE_CHK_BOOL_RET_STATUS((pool_alloc_limit == 0) || (pool_alloc_limit >= kLowerLimit), FAILED,
                             "The value pool alloc limit[%d] in %s is must be 0 or great or equal %ld.", pool_alloc_limit,
                             OPTION_FLOW_GRAPH_MEMORY_MAX_SIZE, kLowerLimit);
    } else {
      GE_CHK_STATUS_RET(ConvertToInt64(pool_config, pool_size),
                        "parse pool_size[%s] failed, remote_group_cache_config=%s.", pool_config.c_str(),
                        remote_group_cache_config.c_str());
    }
    GE_CHK_BOOL_RET_STATUS((pool_size >= kLowerLimit) && (pool_size <= kUpperLimit), FAILED,
                           "The value pool_size[%ld] in %s is out of range[%ld, %ld].", pool_size,
                           OPTION_FLOW_GRAPH_MEMORY_MAX_SIZE, kLowerLimit, kUpperLimit);

    GE_CHK_STATUS_RET(CheckUint64AddOverflow(group_size, static_cast<uint64_t>(pool_size)),
                      "[Check][Param] group_size:%lu add pool_size %ld is overflow", group_size, pool_size);
    group_size += static_cast<uint64_t>(pool_size);
    remote_group_cache_pool_list.emplace_back(static_cast<uint64_t>(pool_size), static_cast<uint64_t>(pool_alloc_limit));
  }
  GE_CHK_BOOL_RET_STATUS((group_size <= static_cast<uint64_t>(kUpperLimit)), FAILED,
                         "The sum of pool_size=[%lu] over limit value %ld.", group_size, kUpperLimit);
  {
    std::unique_lock<std::mutex> lk(group_cache_mutex_);
    remote_group_cache_pool_list_.clear();
    remote_group_cache_pool_list_.reserve(remote_group_cache_pool_list.size());
    for (const auto &pool : remote_group_cache_pool_list) {
      remote_group_cache_pool_list_.emplace_back(pool.first >> kBitShift1024, pool.second >> kBitShift1024);
    }
    remote_group_cache_alloc_size_ = group_size >> kBitShift1024;
  }
  return SUCCESS;
}

Status MemoryGroupManager::SetRemoteGroupCacheConfig(const std::string &remote_group_cache_config) {
  GEEVENT("Remote graph cache config is %s.", remote_group_cache_config.c_str());
  if (remote_group_cache_config.empty()) {
    SetDefaultRemoteGroupCacheConfig();
    return SUCCESS;
  }
  constexpr const char *kConfigPattern = R"(\d{1,12}(:\d{1,10})?(,\d{1,12}(:\d{1,10})?){0,127})";
  std::regex config_regex(kConfigPattern);
  if (!std::regex_match(remote_group_cache_config, config_regex)) {
    GELOGE(PARAM_INVALID, "Remote group cache config=%s is not match pattern %s.", remote_group_cache_config.c_str(),
           kConfigPattern);
    return PARAM_INVALID;
  }
  return ParseRemoteGroupCacheConfig(remote_group_cache_config);
}

Status MemoryGroupManager::Initialize(const NodeConfig &node_config) {
  qs_mem_group_name_ = std::string("DM_QS_GROUP_") + std::to_string(mmGetPid());
  if (DuplicateGrp(getpid()) == SUCCESS) {
    GELOGD("Memory group is already inited, group name = %s.", qs_mem_group_name_.c_str());
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(MemGroupInit(node_config, qs_mem_group_name_),
                    "Failed to init group, group name = %s.", qs_mem_group_name_.c_str());
  return SUCCESS;
}

void MemoryGroupManager::Finalize() {
  std::unique_lock<std::mutex> lk(mutex_);
  inited_groups_.clear();
  GELOGI("Memory group manager finalize success.");
}

bool MemoryGroupManager::IsGroupInited(const std::string &group_name) {
  const auto &it = inited_groups_.find(group_name);
  return it != inited_groups_.cend();
}

Status MemoryGroupManager::DuplicateGrp(const int32_t pid) {
  // Get all the grouping information of the current process
  rtMemGrpQueryInput_t input{};
  input.cmd = 1;  // MEM_GRP_QUERY_GROUPS_OF_PROCESS
  input.grpQueryByProc.pid = pid;
  rtMemGrpQueryOutput_t output;
  output.maxNum = RT_MEM_BUFF_MAX_CFG_NUM;
  rtMemGrpOfProc_t sub_output[RT_MEM_CACHE_MAX_NUM];
  output.groupsOfProc = sub_output;

  auto rt_ret = rtMemGrpQuery(&input, &output);
  // If the current pid has a group with admin permission, it will return directly without creating a new group
  if (rt_ret != SUCCESS) {
    return FAILED;
  }
  for (size_t i = 0U; i < output.resultNum; ++i) {
    if (output.groupsOfProc[i].attr.admin == 1U) {
      GELOGD("output.groupsOfProc[i].attr.admin == 1U.");
      return SUCCESS;
    }
  }
  GELOGD("DuplicateGrp rtMemGrpQuery resultNum:[%zu] rt_ret:[%d]", output.resultNum, rt_ret);
  return FAILED;
}

Status MemoryGroupManager::MemGroupInit(const std::string &group_name) {
  std::unique_lock<std::mutex> lk(mutex_);
  if (!IsGroupInited(group_name)) {
    GE_CHK_STATUS_RET(MemGrpCreate(group_name), "Failed to create group, group name = %s.", group_name.c_str());
    pid_t pid = getpid();
    GE_CHK_STATUS_RET(MemGrpAddProc(group_name, pid, true, true),
                      "[Add][Pid] add leader pid[%d] into memory group[%s] error.", pid, group_name.c_str());
    GE_CHK_RT_RET(rtMemGrpAttach(group_name.c_str(), kTimeout));
    GELOGD("[Attach][MemoryGrp] success.");
    inited_groups_.emplace(group_name);
    GELOGI("Memory group added success, group name = %s.", group_name.c_str());
  }
  GELOGI("Memory group init success, group name = %s.", group_name.c_str());
  return SUCCESS;
}

std::mutex &MemoryGroupManager::GetDeviceMutex(int32_t device_id) {
  std::unique_lock<std::mutex> guard(map_mutex_);
  return device_mutexs_[device_id];
}

Status MemoryGroupManager::RemoteMemGroupInit(int32_t device_id,
                                              const std::string &group_name) {
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  {
    std::unique_lock<std::mutex> lk(mutex_);
    if (IsGroupInited(group_name)) {
      return SUCCESS;
    }
  }
  std::unique_lock<std::mutex> group_cache_lk(group_cache_mutex_);
  // copy config to reduce the range of lock
  uint64_t remote_group_cache_alloc_size = remote_group_cache_alloc_size_;
  std::vector<std::pair<uint64_t, uint32_t>> remote_group_cache_pool_list(remote_group_cache_pool_list_);
  group_cache_lk.unlock();

  GE_CHK_STATUS_RET(
      ProxyEventManager::CreateGroup(device_id, group_name, remote_group_cache_alloc_size,
                                     remote_group_cache_pool_list),
      "Failed to create remote group, device_id = %d, group name = %s, "
      "pre alloc size = %lu kb, cache pool size = %zu.",
      device_id, group_name.c_str(), remote_group_cache_alloc_size, remote_group_cache_pool_list.size());

  std::unique_lock<std::mutex> lk(mutex_);
  inited_groups_.emplace(group_name);
  GELOGI("Memory group init success, group name = %s.", group_name.c_str());
  return SUCCESS;
}

Status MemoryGroupManager::RemoteMemGrpAddProc(int32_t device_id,
                                               const std::string &group_name,
                                               const pid_t pid,
                                               bool is_admin,
                                               bool is_alloc) {
  GELOGI("Remote group cache alloc size is %lu kb.", remote_group_cache_alloc_size_);
  GE_CHK_STATUS_RET(RemoteMemGroupInit(device_id, group_name),
                    "Remote memory group init failed, group name = %s, device_id = %d, alloc_size is = %lu kb.",
                    group_name.c_str(), device_id, remote_group_cache_alloc_size_);
  GE_CHK_STATUS_RET(ProxyEventManager::AddGroup(device_id, group_name, pid, is_admin, is_alloc),
                    "Failed to add remote group, device_id = %d, group name = %s.",
                    device_id, group_name.c_str());
  return SUCCESS;
}

void MemoryGroupManager::SetQsMemGroupName(const std::string &group_name) {
  qs_mem_group_name_ = group_name;
}

const std::string &MemoryGroupManager::GetQsMemGroupName() const {
  return qs_mem_group_name_;
}

std::string MemoryGroupManager::GetRemoteMemGroupName(int32_t device_id) const {
  return "GRP_" + std::to_string(mmGetPid()) + "_" + std::to_string(device_id);
}
}  // namespace ge
