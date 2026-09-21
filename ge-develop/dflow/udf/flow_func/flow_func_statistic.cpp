/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_statistic.h"
#include <algorithm>
#include <sstream>
#include "common/udf_log.h"
#include "common/util.h"

namespace FlowFunc {
namespace {
std::string ToStringRange(const std::vector<IOStatisticInfo> &statistic_infos, size_t start_idx, size_t end_idx) {
    std::stringstream min_size_stream;
    std::stringstream max_size_stream;
    std::stringstream min_shape_stream;
    std::stringstream max_shape_stream;
    for (size_t idx = start_idx; idx <= end_idx; ++idx) {
        const auto &statisticInfo = statistic_infos[idx];
        if (idx != start_idx) {
            min_size_stream << ", ";
            max_size_stream << ", ";
            min_shape_stream << ", ";
            max_shape_stream << ", ";
        }
        min_size_stream << statisticInfo.min_size;
        max_size_stream << statisticInfo.max_size;
        min_shape_stream << ToString(statisticInfo.min_shape);
        max_shape_stream << ToString(statisticInfo.max_shape);
    }
    std::stringstream str_stream;
    str_stream << "min_size=[" << min_size_stream.str() << "], max_size=[" << max_size_stream.str() << "], min_shape=[" <<
              min_shape_stream.str() << "], max_shape=[" << max_shape_stream.str() << "]";
    return str_stream.str();
}
}

void FlowFuncStatistic::Init(const std::string &flow_func_info, size_t input_num, size_t output_num) {
    flow_func_info_ = flow_func_info;
    input_num_ = input_num;
    output_num_ = output_num;
    statistic_info_.inputs.resize(input_num_);
    statistic_info_.outputs.resize(output_num_);
}

void FlowFuncStatistic::ExecuteStart() {
    execute_start_time_ = std::chrono::steady_clock::now();
}

void FlowFuncStatistic::ExecuteFinish() {
    ++statistic_info_.total_exec_num;
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = current_time - execute_start_time_;

    int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    if (elapsed_us < 0) {
        UDF_LOG_WARN("elapsed microseconds=%ld is less than 0, ignore it, flow_func_info=%s.", elapsed_us,
            flow_func_info_.c_str());
        return;
    }
    uint64_t elapsed_us_u64 = static_cast<uint64_t>(elapsed_us);
    if ((statistic_info_.min_exec_time == 0) || (statistic_info_.min_exec_time > elapsed_us_u64)) {
        statistic_info_.min_exec_time = elapsed_us_u64;
    }

    if (elapsed_us_u64 > statistic_info_.max_exec_time) {
        statistic_info_.sub_max_exec_time = statistic_info_.max_exec_time;
        statistic_info_.max_exec_time = elapsed_us_u64;
    } else if (elapsed_us_u64 > statistic_info_.sub_max_exec_time) {
        statistic_info_.sub_max_exec_time = elapsed_us_u64;
    } else {
        // do nothing
    }
    statistic_info_.total_exec_time += elapsed_us_u64;
}

void FlowFuncStatistic::RecordInput(size_t input_idx, const std::shared_ptr<FlowMsg> &flow_msg) {
    if (input_idx >= input_num_) {
        UDF_LOG_ERROR("input_idx=%zu must be less than input_num=%zu, ignore it, flow_func_info=%s.", input_idx, input_num_,
            flow_func_info_.c_str());
        return;
    }
    auto &input_statistic_info = statistic_info_.inputs[input_idx];
    UpdateIOStatisticInfo(flow_msg->GetTensor(), input_statistic_info);
}

void FlowFuncStatistic::RecordOutput(size_t output_idx, const std::shared_ptr<FlowMsg> &flow_msg) {
    if (output_idx >= output_num_) {
        UDF_LOG_ERROR("output_idx=%zu must be less than output_num=%zu, ignore it, flow_func_info=%s.", output_idx,
            output_num_, flow_func_info_.c_str());
        return;
    }
    auto &output_statistic_info = statistic_info_.outputs[output_idx];
    UpdateIOStatisticInfo(flow_msg->GetTensor(), output_statistic_info);
}

void FlowFuncStatistic::UpdateIOStatisticInfo(const Tensor *tensor, IOStatisticInfo &io_statistic_info) {
    if (tensor == nullptr) {
        UDF_LOG_DEBUG("tensor is null, ignore it");
        return;
    }
    uint64_t data_size = tensor->GetDataSize();
    const auto &shape = tensor->GetShape();
    if ((io_statistic_info.min_size == 0) || (data_size < io_statistic_info.min_size)) {
        io_statistic_info.min_size = data_size;
        io_statistic_info.min_shape.assign(shape.cbegin(), shape.cend());
    }
    if (data_size > io_statistic_info.max_size) {
        io_statistic_info.max_size = data_size;
        io_statistic_info.max_shape.assign(shape.cbegin(), shape.cend());
    }
}

void FlowFuncStatistic::DumpMetrics(bool with_queue_info) const {
    UDF_RUN_LOG_INFO("model_metrics:name=%s, min_exec_time=%lu us, max_exec_time=%lu us, sub_max_exec_time=%lu us"
                     ", total_exec_time=%lu us, total_exec_num=%lu.",
        flow_func_info_.c_str(), statistic_info_.min_exec_time, statistic_info_.max_exec_time, statistic_info_.sub_max_exec_time,
        statistic_info_.total_exec_time, statistic_info_.total_exec_num);
    if (!with_queue_info || (statistic_info_.total_exec_num == 0)) {
        return;
    }
    constexpr size_t print_num_per_log = 10;
    for (size_t input_idx = 0; input_idx < input_num_; input_idx += print_num_per_log) {
        size_t start_idx = input_idx;
        size_t end_idx = (input_idx + print_num_per_log) > input_num_ ? (input_num_ - 1) : (input_idx + print_num_per_log - 1);
        UDF_RUN_LOG_INFO("model_metrics:name=%s, input[%zu-%zu], %s.", flow_func_info_.c_str(), start_idx, end_idx,
            ToStringRange(statistic_info_.inputs, start_idx, end_idx).c_str());
    }

    for (size_t output_idx = 0; output_idx < output_num_; output_idx += print_num_per_log) {
        size_t start_idx = output_idx;
        size_t end_idx = (output_idx + print_num_per_log) > output_num_ ? (output_num_ - 1) : (output_idx + print_num_per_log - 1);
        UDF_RUN_LOG_INFO("model_metrics:name=%s, output[%zu-%zu], %s.", flow_func_info_.c_str(), start_idx, end_idx,
            ToStringRange(statistic_info_.outputs, start_idx, end_idx).c_str());
    }
}
}
