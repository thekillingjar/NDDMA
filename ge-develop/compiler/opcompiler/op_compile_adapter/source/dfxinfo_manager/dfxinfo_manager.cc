/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfxinfo_manager/dfxinfo_manager.h"
#include <map>
#include <sstream>
#include "inc/te_fusion_log.h"

namespace te {
namespace fusion {
namespace {
const size_t kMaxMsgSize = 111;
const std::map<StatisticsType, std::string> kStatisticsTypeMap {
    {StatisticsType::DISK_CACHE, "Disk cache"},
    {StatisticsType::BINARY_REUSE, "Binary reuse"},
    {StatisticsType::ONLINE_COMPILE, "Online compile"}
};

const std::map<RecordEventType, std::string> kRecordEventTypeMap {
    {RecordEventType::MATCH, "match times"},
    {RecordEventType::REUSE_SUCC, "reused times"},
    {RecordEventType::REUSE_FAIL, "reuse fail times"},
    {RecordEventType::RESUE_CHECK_FAIL, "reuse check fail times"},
    {RecordEventType::COPY, "copy times"},
    {RecordEventType::COPY_SUCC, "copy success times"},
    {RecordEventType::COPY_FAIL, "copy fail times"},
    {RecordEventType::JSON_INVALID, "json invalid times"},
    {RecordEventType::UPDATE_ACCESS_FAIL, "update access time fail times"},
    {RecordEventType::FILE_LOCK_FAIL, "fail to lock file times"},
    {RecordEventType::CACHE_NOT_EXIST, "cache not existed times"},
    {RecordEventType::SHA256_FAIL, "fail to verify sha256 times"},
    {RecordEventType::SIMKEY_MISMATCH, "simple key mismatch times"},
    {RecordEventType::STATICKEY_MISMATCH, "static key mismatch times"},
    {RecordEventType::VERSION_MISMATCH, "version mismatch times"},
    {RecordEventType::OM_MISMATCH, "om mismatch times"},
    {RecordEventType::TASK_SUBMIT, "compile task submit times"},
    {RecordEventType::TASK_SUCC, "compile success times"},
    {RecordEventType::TASK_FAIL, "compile fail times"}
};
}

void DfxInfoManager::Initialize()
{
    if (isInit_) {
        return;
    }
    TE_DBGLOG("Start init dfx info manager");

    ResetStatisticsRecorders();
    isInit_ = true;
}

void DfxInfoManager::Finalize()
{
    if (!isInit_) {
        return;
    }
    isInit_ = false;
}

void DfxInfoManager::ResetStatisticsRecorders()
{
    std::lock_guard<std::mutex> lock_guard(statisticsMutex_);
    for (size_t i = 0; i < static_cast<size_t>(StatisticsType::BOTTOM); i++) {
        statisticsRecorders_[i].fill(0);
    }
}

void DfxInfoManager::RecordStatistics(const StatisticsType statType, const RecordEventType recordType)
{
    if (statType == StatisticsType::BOTTOM || recordType == RecordEventType::BOTTOM) {
        return;
    }
    std::lock_guard<std::mutex> lock_guard(statisticsMutex_);
    statisticsRecorders_[static_cast<size_t>(statType)][static_cast<size_t>(recordType)]++;
}

void DfxInfoManager::AddStatisticsMsgOfOneRecorder(const StatisticsType statisticsType,
                                                   const StatisticsRecorder &statisticsRecorder,
                                                   std::vector<std::string> &statisticsMsg)
{
    auto iter = kStatisticsTypeMap.find(statisticsType);
    if (iter == kStatisticsTypeMap.end()) {
        return;
    }
    std::string prefix = iter->second + " statistics:";
    std::string statMsg;
    bool isFirst = true;
    for (auto& item : kRecordEventTypeMap) {
        uint64_t itemCount = statisticsRecorder[static_cast<size_t>(item.first)];
        if (itemCount == 0) {
            continue;
        }
        if (isFirst) {
            statMsg += prefix;
        }
        std::string itemMsg = item.second + ":" + std::to_string(itemCount);
        if (statMsg.size() + itemMsg.size() < kMaxMsgSize) {
            if (isFirst) {
                isFirst = false;
            } else {
                itemMsg = "|" + itemMsg;
            }
            statMsg += itemMsg;
        } else {
            statisticsMsg.push_back(statMsg);
            statMsg = prefix + itemMsg;
        }
    }
    if (!statMsg.empty() && statMsg != prefix) {
        statisticsMsg.push_back(statMsg);
    }
    return;
}

void DfxInfoManager::GetStatisticsMsgs(std::vector<std::string> &statisticsMsg) const
{
    std::lock_guard<std::mutex> lock_guard(statisticsMutex_);
    for (size_t i = 0; i < static_cast<size_t>(StatisticsType::BOTTOM); i++) {
        AddStatisticsMsgOfOneRecorder(static_cast<StatisticsType>(i), statisticsRecorders_.at(i), statisticsMsg);
    }
}
}  // namespace fusion
}  // namespace te