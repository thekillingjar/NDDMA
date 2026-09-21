/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEFUSION_DXF_INFO_MANAGER_H
#define TEFUSION_DXF_INFO_MANAGER_H

#include <memory>
#include <unordered_map>
#include <array>
#include <vector>
#include <mutex>

namespace te {
namespace fusion {
enum class StatisticsType {
    DISK_CACHE = 0,
    ONLINE_COMPILE,
    BINARY_REUSE,
    BOTTOM
};

enum class RecordEventType {
    MATCH = 0,
    REUSE_SUCC,
    REUSE_FAIL,
    RESUE_CHECK_FAIL,
    COPY,
    COPY_SUCC,
    COPY_FAIL,
    JSON_INVALID,
    UPDATE_ACCESS_FAIL,
    FILE_LOCK_FAIL,
    CACHE_NOT_EXIST,
    SHA256_FAIL,
    SIMKEY_MISMATCH,
    STATICKEY_MISMATCH,
    OM_MISMATCH,
    VERSION_MISMATCH,
    TASK_SUBMIT,
    TASK_SUCC,
    TASK_FAIL,
    BOTTOM
};

using StatisticsRecorder = std::array<uint64_t, static_cast<size_t>(RecordEventType::BOTTOM)>;

class DfxInfoManager {
public:
    static DfxInfoManager &Instance()
    {
        static DfxInfoManager instance;
        return instance;
    }
    DfxInfoManager() : isInit_(false) {}
    ~DfxInfoManager() = default;
    void Initialize();
    void Finalize();
    void RecordStatistics(const StatisticsType statType, const RecordEventType recordType);
    void GetStatisticsMsgs(std::vector<std::string> &statisticsMsg) const;
private:
    void ResetStatisticsRecorders();
    static void AddStatisticsMsgOfOneRecorder(const StatisticsType statisticsType,
                                              const StatisticsRecorder &statisticsRecorder,
                                              std::vector<std::string> &statisticsMsg);
private:
    bool isInit_;
    std::array<StatisticsRecorder, static_cast<size_t>(StatisticsType::BOTTOM)> statisticsRecorders_;
    mutable std::mutex statisticsMutex_;
};
}  // namespace fusion
}  // namespace te
#endif