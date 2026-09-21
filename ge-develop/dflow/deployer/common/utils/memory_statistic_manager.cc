/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_statistic_manager.h"
#include <fstream>
#include <unistd.h>
#include <securec.h>
#include "common/debug/ge_log.h"
#include "common/util.h"
#include "common/checker.h"
#include "acl/acl.h"

namespace ge {
MemoryStatisticManager::~MemoryStatisticManager() {
  Finalize();
}

MemoryStatisticManager &MemoryStatisticManager::Instance() {
  static MemoryStatisticManager inst;
  return inst;
}

void MemoryStatisticManager::Initialize(const std::string &xsmem_group_name) {
  if (!thread_run_flag_) {
    xsmem_group_name_ = xsmem_group_name;
    process_status_file_name_ = GetProcessStatusFile();
    xsmem_summary_file_name_ = GetXsmemSummaryFile();
    thread_run_flag_ = true;
    statistic_thread_ = std::thread(&MemoryStatisticManager::RunStatistic, this);
  }
}

void MemoryStatisticManager::Finalize() {
  if (!thread_run_flag_) {
    return;
  }
  {
    std::unique_lock<std::mutex> lock(mtx_);
    thread_run_flag_ = false;
  }
  thread_condition_.notify_all();
  if (statistic_thread_.joinable()) {
    statistic_thread_.join();
  }
  DumpMetrics();
}

void MemoryStatisticManager::RunStatistic() {
  constexpr const char *thread_name = "ge_mem_stat";
  int32_t set_ret = pthread_setname_np(pthread_self(), thread_name);
  GELOGD("set thread name to [%s], ret=%d", thread_name, set_ret);
  while (thread_run_flag_) {
    StatisticRss();
    StatisticXsmem();
    std::unique_lock<std::mutex> lock(mtx_);
    // every 2s.
    thread_condition_.wait_for(lock, std::chrono::seconds(2), [this] {
      return !thread_run_flag_;
    });
  }
}

void MemoryStatisticManager::DumpMetrics() const {
  // change kB to B, need multiply by 1024(<< 10)
  uint64_t rss_avg_in_bytes = (static_cast<uint64_t>(rss_avg_) << 10);
  // change kB to B, need multiply by 1024(<< 10)
  uint64_t rss_hwm_in_bytes = (rss_hwm_ << 10);
  GEEVENT("proc_metrics:pid=%d, rssavg=%lu B, rsshwm=%lu B, xsmemavg=%ld B, xsmemhwm=%lu B.", getpid(),
          rss_avg_in_bytes, rss_hwm_in_bytes, static_cast<int64_t>(xsmem_avg_), xsmem_hwm_);
}

void MemoryStatisticManager::StatisticRss() {
  uint64_t vm_rss = 0;
  uint64_t vm_hwm = 0;
  if (!ReadRssValue(vm_rss, vm_hwm)) {
    return;
  }
  // not possible, just prevent division by zero
  if (rss_statistic_count_ == UINT32_MAX) {
    rss_statistic_count_ = 0;
  }
  rss_hwm_ = vm_hwm;
  if (rss_statistic_count_ == 0) {
    rss_avg_ = static_cast<double>(vm_rss);
  } else {
    // to avoid multiply overflows, can't use (rss_avg_ * rss_statistic_count_ + vm_rss)/(rss_statistic_count_ + 1)
    // instead.
    rss_avg_ = rss_avg_ * (static_cast<double>(rss_statistic_count_) / (rss_statistic_count_ + 1)) +
               vm_rss * (static_cast<double>(1) / (rss_statistic_count_ + 1));
  }
  ++rss_statistic_count_;
  GELOGI("rss_statistic_count=%lu, rss_avg=%.3lf kB, rss_hwm=%lu kB.", rss_statistic_count_, rss_avg_, rss_hwm_);
}

std::string MemoryStatisticManager::GetProcessStatusFile() {
  pid_t pid = getpid();
  std::string status_file = "/proc/" + std::to_string(pid) + "/status";
  return status_file;
}

bool MemoryStatisticManager::ReadRssValue(uint64_t &vm_rss, uint64_t &vm_hwm) const {
  std::ifstream ifs(process_status_file_name_);
  if (!ifs) {
    GELOGE(FAILED, "Failed to open file:%s, errno=%d, strerror=%s.", process_status_file_name_.c_str(), errno,
           GetErrorNumStr(errno).c_str());
    return false;
  }
  std::string vm_rss_str;
  std::string vm_hwm_str;
  std::string tmp_line;
  constexpr const char *vm_rss_head = "VmRSS:";
  constexpr const char *vm_hwm_head = "VmHWM:";

  while (getline(ifs, tmp_line)) {
    if (tmp_line.compare(0, strlen(vm_rss_head), vm_rss_head) == 0) {
      vm_rss_str = std::move(tmp_line);
    } else if (tmp_line.compare(0, strlen(vm_hwm_head), vm_hwm_head) == 0) {
      vm_hwm_str = std::move(tmp_line);
    } else {
      continue;
    }
    if ((!vm_rss_str.empty()) && (!vm_hwm_str.empty())) {
      break;
    }
  }
  GE_ASSERT_TRUE(!vm_rss_str.empty(), "no line start with \"%s\" in file:%s", vm_rss_head,
                 process_status_file_name_.c_str());
  GE_ASSERT_TRUE(!vm_hwm_str.empty(), "no line start with \"%s\" in file:%s", vm_hwm_head,
                 process_status_file_name_.c_str());

  GELOGD("vm_rss_str=[%s], vm_hwm_str=[%s].", vm_rss_str.c_str(), vm_hwm_str.c_str());
  int64_t vm_rss_value = 0;
  Status status = ConvertToInt64(vm_rss_str.substr(strlen(vm_rss_head)), vm_rss_value);
  GE_ASSERT_SUCCESS(status, "parse VmRSS value from str[%s] failed.", vm_rss_str.c_str());
  GE_ASSERT_TRUE(vm_rss_value > 0, "VmRSS value in str[%s] is invalid", vm_rss_str.c_str());

  int64_t vm_hwm_value = 0;
  status = ConvertToInt64(vm_hwm_str.substr(strlen(vm_hwm_head)), vm_hwm_value);
  GE_ASSERT_SUCCESS(status, "parse VmHWM value from str[%s] failed.", vm_hwm_str.c_str());
  GE_ASSERT_TRUE(vm_hwm_value > 0, "VmHWM value in str[%s] is invalid", vm_hwm_str.c_str());

  vm_rss = static_cast<uint64_t>(vm_rss_value);
  vm_hwm = static_cast<uint64_t>(vm_hwm_value);
  GELOGI("vm_rss=%lu kB, vm_hwm=%lu kB.", vm_rss, vm_hwm);
  return true;
}

void MemoryStatisticManager::StatisticXsmem() {
  uint64_t real_size = 0;
  uint64_t peak_size = 0;
  if (!ReadXsmemValue(real_size, peak_size)) {
    return;
  }
  // not possible, just prevent division by zero
  if (xsmem_statistic_count_ == UINT32_MAX) {
    xsmem_statistic_count_ = 0;
  }
  if (peak_size < real_size) {
    peak_size = real_size;
  }
  if (xsmem_hwm_ < peak_size) {
    xsmem_hwm_ = peak_size;
  }

  if (xsmem_statistic_count_ == 0) {
    xsmem_avg_ = real_size;
  } else {
    // avoid multiply overflow, can't use (rss_avg_ * rss_statistic_count_ + vm_rss)/(rss_statistic_count_ + 1)
    // instead.
    xsmem_avg_ = xsmem_avg_ * (static_cast<double>(xsmem_statistic_count_) / (xsmem_statistic_count_ + 1)) +
                 real_size * (static_cast<double>(1) / (xsmem_statistic_count_ + 1));
  }
  ++xsmem_statistic_count_;
  GELOGI("xsmem_statistic_count=%lu, xsmem_avg=%ld B, xsmem_hwm=%lu B.", xsmem_statistic_count_,
         static_cast<int64_t>(xsmem_avg_), xsmem_hwm_);
}

std::string MemoryStatisticManager::GetXsmemSummaryFile() {
  int32_t pid = 0U;
  const aclError rt_ret = aclrtDeviceGetBareTgid(&pid);
  if (rt_ret != ACL_ERROR_NONE) {
    GELOGE(ge::RT_FAILED, "Call rt api failed, use getpid instead, ret: 0x%X", rt_ret);
    pid = static_cast<int32_t>(getpid());
  }

  std::string summary_file = "/proc/xsmem/task/" + std::to_string(pid) + "/summary";
  return summary_file;
}

bool MemoryStatisticManager::ReadXsmemValue(uint64_t &real_size, uint64_t &peak_size) const {
  std::ifstream ifs(xsmem_summary_file_name_);
  if (!ifs) {
    GELOGW("Failed to open file:%s, errno=%d, strerror=%s.", xsmem_summary_file_name_.c_str(), errno,
           GetErrorNumStr(errno).c_str());
    return false;
  }
  std::string statistic_str;
  std::string tmp_line;

  while (getline(ifs, tmp_line)) {
    if (tmp_line.find(xsmem_group_name_) != std::string::npos) {
      statistic_str = std::move(tmp_line);
      break;
    }
  }

  GE_ASSERT_TRUE(!statistic_str.empty(), "no [%s] found in file:%s", xsmem_group_name_.c_str(),
                 xsmem_summary_file_name_.c_str());

  uint64_t alloc_size = 0;
  constexpr uint32_t kMaxLen = 128;
  char pool_id_and_name[kMaxLen] = {};
  /*
   * summary file format is:
   * task pid 1234 pool cnt 1
   * pool id(name): alloc_size real_size peak_size
   * 1(DM_QS_GROUP_1234_18446614043574950784) 1023 1024 2048
   * summary: 1023 1024
   *
   */
  int32_t ret =
      sscanf_s(statistic_str.c_str(), "%s %lu %lu %lu", pool_id_and_name, kMaxLen, &alloc_size, &real_size, &peak_size);
  // must parse success 3 args(old version doen not have peak_size)
  if (ret < 3) {
    GELOGE(FAILED, "Parse [%s] failed, ret=%d, pool_id_and_name=%s, alloc_size=%lu, real_size=%lu.",
           statistic_str.c_str(), ret, pool_id_and_name, alloc_size, real_size);
    return false;
  }

  GELOGI("xsmem pool_id_and_name=%s, alloc_size=%lu B, real_size=%lu B, peak_size=%lu B, statistic_str=[%s].",
         pool_id_and_name, alloc_size, real_size, peak_size, statistic_str.c_str());
  if (real_size == 0) {
    GELOGI("xsmem real_size is 0, ignore it.");
    return false;
  }
  return true;
}
}  // namespace ge
