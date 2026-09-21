/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_COMMON_DEFINE_H
#define COMMON_COMMON_DEFINE_H

#include <cstdint>
#include <ostream>
#include <vector>
#include <map>
#include "ascend_hal_define.h"

namespace FlowFunc {
namespace UdfEvent {
constexpr uint32_t kEventIdUdfStart = static_cast<uint32_t>(EVENT_USR_START);
constexpr uint32_t kEventIdProcessorInit = kEventIdUdfStart;
constexpr uint32_t kEventIdFlowFuncInit = kEventIdUdfStart + 1U;
constexpr uint32_t kEventIdFlowFuncExecute = kEventIdUdfStart + 2U;
constexpr uint32_t kEventIdTimer = kEventIdUdfStart + 3U;
constexpr uint32_t kEventIdFlowFuncReportStatus = kEventIdUdfStart + 6U;
constexpr uint32_t kEventIdNotifyThreadExit = kEventIdUdfStart + 7U;
constexpr uint32_t kEventIdFlowFuncSuspendFinished = kEventIdUdfStart + 8U;
constexpr uint32_t kEventIdFlowFuncRecoverFinished = kEventIdUdfStart + 9U;
constexpr uint32_t kEventIdSwitchToSoftSchedMode = kEventIdUdfStart + 10U;
constexpr uint32_t kEventIdRaiseException = kEventIdUdfStart + 11U;
constexpr uint32_t kEventIdSingleFlowFuncInit = kEventIdUdfStart + 12U;
constexpr uint32_t kEventIdWakeUp = kEventIdUdfStart + 13U;
}
namespace Common {
// ignore q result when output not use.
constexpr uint32_t kDummyQId = UINT32_MAX;

constexpr uint32_t kDefaultMbufAlignSize = 64U;

constexpr int32_t kDeviceTypeNpu = 0;
constexpr int32_t kDeviceTypeCpu = 1;

// drv max value is 10s
constexpr int32_t kAttachMemGrpWaitTimeout = 10 * 1000;
}

struct QueueDevInfo {
    uint32_t queue_id;
    int32_t device_type;
    int32_t device_id;
    // device id in deployer, every device os device is start by 0, but deployer device id is[0-7]
    int32_t deploy_device_id;
    uint32_t logic_queue_id;
    bool is_proxy_queue;

    // use for print IO info, can't be too long
    friend std::ostream &operator<<(std::ostream &os, const QueueDevInfo &queue_info) {
        os << queue_info.queue_id << "(type:" << queue_info.device_type <<
           ",devId:" << queue_info.deploy_device_id << ")";
        return os;
    }

    bool operator<(const QueueDevInfo &other) const {
        if (queue_id != other.queue_id) {
            return queue_id < other.queue_id;
        } else if (device_id != other.device_id) {
            return device_id < other.device_id;
        } else {
            return device_type < other.device_type;
        }
    }
};

struct ModelQueueInfos {
    std::vector<QueueDevInfo> input_queues;
    std::map<std::string, std::vector<uint32_t>> func_input_maps;
    std::vector<QueueDevInfo> output_queues;
    std::map<std::string, std::vector<uint32_t>> func_output_maps;
};

struct BufferConfigItem {
    uint32_t total_size;
    uint32_t blk_size;
    uint32_t max_buf_size;
    std::string page_type;
};
}
#endif // COMMON_COMMON_DEFINE_H
