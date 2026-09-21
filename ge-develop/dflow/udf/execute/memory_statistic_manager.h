/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEMORY_STATISTIC_MANAGER_H
#define MEMORY_STATISTIC_MANAGER_H

#include <cstdint>
#include <mutex>
#include <thread>
#include <string>
#include <condition_variable>

namespace FlowFunc {
class MemoryStatisticManager {
public:
    static MemoryStatisticManager &Instance();

    void Init(const std::string &xsmem_group_name);

    void Finalize();

private:
    MemoryStatisticManager() = default;

    ~MemoryStatisticManager();

    void DumpMetrics() const;

    void RunStatistic();

    void StatisticRss();

    void StatisticXsmem();

    static std::string GetProcessStatusFile();

    static std::string GetXsmemSummaryFile();

    bool ReadRssValue(uint64_t &vm_rss, uint64_t &vm_hwm) const;

    bool ReadXsmemValue(uint64_t &real_size, uint64_t &peak_size) const;

    // use for notify thread exit immediately
    std::mutex mtx_;
    std::condition_variable thread_condition_;
    volatile bool thread_run_flag_ = false;
    std::thread statistic_thread_;
    std::string xsmem_group_name_;

    std::string process_status_file_name_;
    std::string xsmem_summary_file_name_;

    volatile double rss_avg_{0};
    volatile uint64_t rss_hwm_{0};
    volatile uint32_t rss_statistic_count_{0};

    volatile double xsmem_avg_{0};
    volatile uint64_t xsmem_hwm_{0};
    volatile uint32_t xsmem_statistic_count_{0};
};
}
#endif // MEMORY_STATISTIC_MANAGER_H
