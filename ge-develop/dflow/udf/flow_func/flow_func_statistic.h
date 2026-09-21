/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_STATISTIC_H
#define FLOW_FUNC_STATISTIC_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <atomic>
#include <vector>
#include <memory>
#include <chrono>
#include "flow_func/flow_msg.h"

namespace FlowFunc {
struct IOStatisticInfo {
    volatile uint64_t min_size{0};
    volatile uint64_t max_size{0};
    std::vector<int64_t> min_shape;
    std::vector<int64_t> max_shape;
};
struct FlowFuncStatisticInfo {
    volatile uint64_t min_exec_time{0};
    volatile uint64_t max_exec_time{0};
    volatile uint64_t sub_max_exec_time{0};
    volatile uint64_t total_exec_time{0};
    volatile uint64_t total_exec_num{0};
    std::vector<IOStatisticInfo> inputs;
    std::vector<IOStatisticInfo> outputs;
};

// attention: not support dump when running, just dump when exit.
class FlowFuncStatistic {
public:
    FlowFuncStatistic() = default;

    ~FlowFuncStatistic() = default;

    void Init(const std::string &flow_func_info, size_t input_num, size_t output_num);

    void ExecuteStart();

    void ExecuteFinish();

    void RecordInput(size_t input_idx, const std::shared_ptr<FlowMsg> &flow_msg);

    void RecordOutput(size_t output_idx, const std::shared_ptr<FlowMsg> &flow_msg);

    void DumpMetrics(bool with_queue_info = false) const;

private:
    static void UpdateIOStatisticInfo(const Tensor *tensor, IOStatisticInfo &io_statistic_info);

    std::string flow_func_info_;
    size_t input_num_ = 0;
    size_t output_num_ = 0;
    FlowFuncStatisticInfo statistic_info_;
    std::chrono::steady_clock::time_point execute_start_time_;
};
}

#endif // FLOW_FUNC_STATISTIC_H
