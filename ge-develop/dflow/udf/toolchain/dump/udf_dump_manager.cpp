/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_dump_manager.h"
#include <regex>
#include <sstream>
#include <thread>
#include "flow_func/flow_func_dumper.h"
#include "common/udf_log.h"
#include "common/util.h"
#include "udf_dump.h"

namespace FlowFunc {
namespace {
class UdfDumper : public FlowFuncDumper {
public:
    UdfDumper() = default;

    ~UdfDumper() override = default;

    int32_t DumpInput(const std::string &flow_func_info, const std::vector<MbufTensor *> &tensors,
        const uint32_t step_id) override
    {
        return UdfDump::DumpInput(flow_func_info, tensors, step_id);
    }

    int32_t DumpOutput(const std::string &flow_func_info, uint32_t index, const MbufTensor *tensor,
                       const uint32_t step_id) override
    {
        return UdfDump::DumpOutput(flow_func_info, index, tensor, step_id);
    }

    bool IsInDumpStep(const uint32_t step_id) override
    {
        return UdfDumpManager::Instance().IsInDumpStep(step_id);
    }
};

constexpr int32_t kDecimal = 10;
// DATA_DUMP_GRUOP_ID， DATA_DUMP_SUBEVENT_ID， DATA_DUMP_MSG_LEN固定，与aicpu保持一致
constexpr uint32_t kDataDumpGroupId = 30U;
constexpr uint32_t kDataDumpSubeventId = 42U;
constexpr uint32_t kDataDumpMsgLen = 128U;
constexpr uint32_t kDataDumpInitTimeoutInterval = 28000U;

void Split(const std::string& str, std::vector<std::string>& str_vec, const char delim) {
    str_vec.clear();
    std::istringstream iss(str);
    std::string temp;

    while (std::getline(iss, temp, delim)) {
        str_vec.emplace_back(std::move(temp));
    }
}
} // namespace

UdfDumpManager &UdfDumpManager::Instance() {
    static UdfDumpManager dump_properties;
    return dump_properties;
}

int32_t UdfDumpManager::Init() {
    dumper_ = MakeShared<UdfDumper>();
    if (dumper_ != nullptr) {
        FlowFuncDumpManager::SetDumper(dumper_);
        return FLOW_FUNC_SUCCESS;
    }
    UDF_LOG_ERROR("malloc UdfDumper failed");
    return FLOW_FUNC_FAILED;
}

void UdfDumpManager::ClearDumpInfo() {
    enable_dump_ = false;
    host_pid_ = 0;
    dump_path_.clear();
    dump_step_.clear();
    dump_mode_.clear();
    step_set_.clear();
    step_range_.clear();
    FlowFuncDumpManager::SetDumper(nullptr);
}

void UdfDumpManager::SetDumpPath(const std::string &path) {
    dump_path_ = path;
}

const std::string &UdfDumpManager::GetDumpPath() const {
    return dump_path_;
}

void UdfDumpManager::SetDumpStep(const std::string &step) {
    dump_step_ = step;
    if (!step.empty()) {
        std::vector<std::string> match_vecs;
        Split(step, match_vecs, '_');
        const std::regex single_step_patten("(\\d{1,10})");
        const std::regex interval_step_patten("(\\d{1,10})-(\\d{1,10})");
        std::smatch result;
        for (const auto &match_vec : match_vecs) {
            if (regex_match(match_vec, result, single_step_patten)) {
                const auto single_step = std::strtol(match_vec.c_str(), nullptr, kDecimal);
                step_set_.insert(static_cast<uint32_t>(single_step));
                UDF_LOG_DEBUG("Insert one step:%u", single_step);
            } else if (regex_match(match_vec, result, interval_step_patten)) {
                std::vector<std::string> vec_split;
                Split(match_vec, vec_split, '-');
                const auto lower_range = static_cast<uint32_t>(std::strtol(vec_split[0UL].c_str(), nullptr, kDecimal));
                const auto higher_range = static_cast<uint32_t>(std::strtol(vec_split[1UL].c_str(), nullptr, kDecimal));
                step_range_.emplace_back(lower_range, higher_range);
                UDF_LOG_DEBUG("Insert step range from %u to %u", lower_range, higher_range);
            } else {
                UDF_LOG_WARN("Invalid dump step %s, disenable dump.", step.c_str());
                enable_dump_ = false;
            }
        }
    }
}

bool UdfDumpManager::IsInDumpStep(const uint32_t step_id) const {
    if (!dump_step_.empty()) {
        const auto step = step_set_.find(step_id);
        if (step != step_set_.end()) {
            return true;
        }
        for (size_t i = 0UL; i < step_range_.size(); i++) {
            if ((step_id >= step_range_[i].first) && (step_id <= step_range_[i].second)) {
                return true;
            }
        }
        return false;
    }
    return true;
}

int32_t UdfDumpManager::InitAicpuDump() {
    struct halQueryDevpidInfo info = {};
    info.proc_type = DEVDRV_PROCESS_CP1;
    info.hostpid = host_pid_;
    info.devid = logic_dev_id_;
    info.vfid = 0;
    auto ret = halQueryDevpid(info, &aicpu_pid_);
    if (ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halQueryDevpid failed, ret:%d", ret);
        return FLOW_FUNC_FAILED;
    }

    char value[kDataDumpMsgLen] = {0};
    event_summary drv_event = {};
    drv_event.dst_engine = CCPU_DEVICE;
    drv_event.policy = ONLY;
    drv_event.pid = aicpu_pid_;
    drv_event.grp_id = kDataDumpGroupId;
    drv_event.event_id = static_cast<EVENT_ID>(EVENT_CCPU_CTRL_MSG);
    drv_event.subevent_id = kDataDumpSubeventId;
    drv_event.msg_len = kDataDumpMsgLen;
    drv_event.msg = value;

    struct event_reply reply = {};
    struct event_proc_result result = {};
    reply.buf = reinterpret_cast<char *>(&result);
    reply.buf_len = sizeof(struct event_proc_result);

    ret = halEschedSubmitEventSync(logic_dev_id_, &drv_event, kDataDumpInitTimeoutInterval, &reply);
    if (ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halEschedSubmitEventSync failed, ret:%d, devid:%u", ret, device_id_);
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

const std::string &UdfDumpManager::GetDumpStep() const {
    return dump_step_;
}

void UdfDumpManager::SetDumpMode(const std::string &mode) {
    dump_mode_ = mode;
}

const std::string &UdfDumpManager::GetDumpMode() const {
    return dump_mode_;
}

void UdfDumpManager::EnableDump() {
    enable_dump_ = true;
}

bool UdfDumpManager::IsEnableDump() const {
    return enable_dump_;
}

void UdfDumpManager::SetDeviceId(uint32_t dev_id) {
    device_id_ = dev_id;
}

uint32_t UdfDumpManager::GetDeviceId() const {
    return device_id_;
}

void UdfDumpManager::SetHostPid(int32_t host_pid) {
    host_pid_ = host_pid;
}

int32_t UdfDumpManager::GetHostPid() const {
    return host_pid_;
}

int32_t UdfDumpManager::GetAicpuPid() const {
    return aicpu_pid_;
}

void UdfDumpManager::SetLogicDeviceId(uint32_t dev_id) {
    logic_dev_id_ = dev_id;
}

uint32_t UdfDumpManager::GetLogicDeviceId() const {
    return logic_dev_id_;
}
}  // namespace ge
