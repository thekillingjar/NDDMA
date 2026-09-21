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
#include <cstring>
#include <string>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include "securec.h"
#include "common/udf_log.h"
#include "common/data_utils.h"
#include "common/util.h"
#include "ascend_hal.h"

namespace FlowFunc {
MemoryStatisticManager::~MemoryStatisticManager() {
    Finalize();
}

MemoryStatisticManager &MemoryStatisticManager::Instance() {
    static MemoryStatisticManager inst;
    return inst;
}

void MemoryStatisticManager::Init(const std::string &xsmem_group_name) {
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
    // notify thread exit
    thread_condition_.notify_all();
    if (statistic_thread_.joinable()) {
        statistic_thread_.join();
    }
    DumpMetrics();
}

void MemoryStatisticManager::RunStatistic() {
    constexpr const char *thread_name = "udf_mem_stat";
    int32_t ret = pthread_setname_np(pthread_self(), thread_name);
    if (ret != 0) {
        UDF_LOG_WARN("pthread_setname_np failed, thread_name=%s, ret=%d", thread_name, ret);
    }
    while (thread_run_flag_) {
        StatisticRss();
        StatisticXsmem();
        {
            std::unique_lock<std::mutex> lock(mtx_);
            // every 2 s
            thread_condition_.wait_for(lock, std::chrono::seconds(2), [this] {
                return !thread_run_flag_;
            });
        }
    }
}

void MemoryStatisticManager::DumpMetrics() const {
    // change kB to B, need multiply by 1024(<< 10)
    uint64_t rss_avg_in_bytes = (static_cast<uint64_t>(rss_avg_) << 10);
    // change kB to B, need multiply by 1024(<< 10)
    uint64_t rss_hwm_in_bytes = (rss_hwm_ << 10);
    UDF_RUN_LOG_INFO("proc_metrics:pid=%d, rssavg=%lu B, rsshwm=%lu B, xsmemavg=%ld B, xsmemhwm=%lu B.",
        getpid(), rss_avg_in_bytes, rss_hwm_in_bytes, static_cast<int64_t>(xsmem_avg_), xsmem_hwm_);
}

void MemoryStatisticManager::StatisticRss() {
    uint64_t vm_rss = 0;
    uint64_t vm_hwm = 0;
    if (!ReadRssValue(vm_rss, vm_hwm)) {
        return;
    }
    // not possible, but to prevent division by zero
    if (rss_statistic_count_ == UINT32_MAX) {
        rss_statistic_count_ = 0;
    }
    rss_hwm_ = vm_hwm;
    if (rss_statistic_count_ == 0) {
        rss_avg_ = static_cast<double>(vm_rss);
    } else {
        rss_avg_ = rss_avg_ * (static_cast<double>(rss_statistic_count_) / (rss_statistic_count_ + 1)) +
                  vm_rss * (static_cast<double>(1) / (rss_statistic_count_ + 1));
    }
    ++rss_statistic_count_;
    UDF_LOG_INFO("rssStatisticCount=%lu, rssAvg=%.3lf kB, rssHwm=%lu kB.", rss_statistic_count_, rss_avg_, rss_hwm_);
}

std::string MemoryStatisticManager::GetProcessStatusFile() {
    pid_t pid = getpid();
    std::string status_file = "/proc/" + std::to_string(pid) + "/status";
    return status_file;
}

bool MemoryStatisticManager::ReadRssValue(uint64_t &vm_rss, uint64_t &vm_hwm) const {
    std::ifstream ifs(process_status_file_name_);
    if (!ifs) {
        int32_t error_no = errno;
        UDF_LOG_ERROR("Failed to open file:%s, errno=%d, strerror=%s.", process_status_file_name_.c_str(), error_no,
            GetErrorNoStr(error_no).c_str());
        return false;
    }

    constexpr const char *kVmRssHead = "VmRSS:";
    constexpr const char *kVmHwmHead = "VmHWM:";
    std::string vm_rss_str;
    std::string vm_hwm_str;
    std::string tmp_line;
    while (getline(ifs, tmp_line)) {
        if (tmp_line.compare(0, strlen(kVmRssHead), kVmRssHead) == 0) {
            vm_rss_str = std::move(tmp_line);
        } else if (tmp_line.compare(0, strlen(kVmHwmHead), kVmHwmHead) == 0) {
            vm_hwm_str = std::move(tmp_line);
        } else {
            continue;
        }
        if ((!vm_rss_str.empty()) && (!vm_hwm_str.empty())) {
            break;
        }
    }
    if (vm_rss_str.empty()) {
        UDF_LOG_ERROR("no \"%s\" found in file:%s.", kVmRssHead, process_status_file_name_.c_str());
        return false;
    }
    if (vm_hwm_str.empty()) {
        UDF_LOG_ERROR("no \"%s\" found in file:%s.", kVmHwmHead, process_status_file_name_.c_str());
        return false;
    }
    UDF_LOG_DEBUG("vm_rss_str=[%s], vm_hwm_str=[%s].", vm_rss_str.c_str(), vm_hwm_str.c_str());
    int64_t vmRssValue = 0;
    if ((!ConvertToInt64(vm_rss_str.substr(strlen(kVmRssHead)), vmRssValue)) || (vmRssValue <= 0)) {
        UDF_LOG_ERROR("parse VmRSS value from str[%s] failed.", vm_rss_str.c_str());
        return false;
    }
    int64_t vmHwmValue = 0;
    if ((!ConvertToInt64(vm_hwm_str.substr(strlen(kVmHwmHead)), vmHwmValue)) || (vmHwmValue <= 0)) {
        UDF_LOG_ERROR("parse VmHWM value from str[%s] failed.", vm_hwm_str.c_str());
        return false;
    }
    vm_rss = static_cast<uint64_t>(vmRssValue);
    vm_hwm = static_cast<uint64_t>(vmHwmValue);
    UDF_LOG_INFO("vm_rss=%lu kB, vm_hwm=%lu kB.", vm_rss, vm_hwm);
    return true;
}

void MemoryStatisticManager::StatisticXsmem() {
    uint64_t real_size = 0;
    uint64_t peak_size = 0;
    if (!ReadXsmemValue(real_size, peak_size)) {
        return;
    }
    // not possible, but to prevent division by zero
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
        xsmem_avg_ = static_cast<double>(real_size);
    } else {
        xsmem_avg_ = xsmem_avg_ * (static_cast<double>(xsmem_statistic_count_) / (xsmem_statistic_count_ + 1)) +
                    real_size * (static_cast<double>(1) / (xsmem_statistic_count_ + 1));
    }
    ++xsmem_statistic_count_;
    UDF_LOG_INFO("xsmemStatisticCount=%lu, xsmemAvg=%ld B, xsmemHwm=%lu B.", xsmem_statistic_count_,
        static_cast<int64_t>(xsmem_avg_), xsmem_hwm_);
}

std::string MemoryStatisticManager::GetXsmemSummaryFile() {
    pid_t drv_pid = drvDeviceGetBareTgid();
    std::string summary_file = "/proc/xsmem/task/" + std::to_string(drv_pid) + "/summary";
    return summary_file;
}

bool MemoryStatisticManager::ReadXsmemValue(uint64_t &real_size, uint64_t &peak_size) const {
    std::ifstream ifs(xsmem_summary_file_name_);
    if (!ifs) {
        int32_t error_no = errno;
        // old version driver file is no permission
        UDF_LOG_WARN("Failed to open file:%s, errno=%d, strerror=%s.", xsmem_summary_file_name_.c_str(), error_no,
            GetErrorNoStr(error_no).c_str());
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
    if (statistic_str.empty()) {
        UDF_LOG_ERROR("no [%s] found in file:%s.", xsmem_group_name_.c_str(), xsmem_summary_file_name_.c_str());
        return false;
    }

    uint64_t alloc_size = 0;
    constexpr uint32_t max_len = 128;
    char pool_id_and_name[max_len] = {};
    /*
     * summary file format is:
     * task pid 2924043 pool cnt 1
     * pool id(name): alloc_size real_size peak_size
     * 1(DM_QS_GROUP_2924043_18446614043574950784) 74071552 74866688 74866688
     * summary: 74071552 74866688
     */
    int32_t ret = sscanf_s(statistic_str.c_str(), "%s %lu %lu %lu", pool_id_and_name, max_len, &alloc_size,
        &real_size, &peak_size);
    // must parse success 3 args(old version does not have peak_size)
    if (ret < 3) {
        UDF_LOG_ERROR("parse [%s] failed, ret=%d, pool_id_and_name=%s, alloc_size=%lu, real_size=%lu.",
            statistic_str.c_str(), ret, pool_id_and_name, alloc_size, real_size);
        return false;
    }
    UDF_LOG_INFO("xsmem pool_id_and_name=%s, alloc_size=%lu B, real_size=%lu B, peak_size=%lu B, statistic_str=[%s].",
        pool_id_and_name, alloc_size, real_size, peak_size, statistic_str.c_str());
    if (real_size == 0) {
        UDF_LOG_INFO("xsmem real_size is 0 B, ignore it.");
        return false;
    }
    return true;
}
}