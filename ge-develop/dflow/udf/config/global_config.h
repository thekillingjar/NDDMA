/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_CONFIG_GLOBAL_CONFIG_H
#define UDF_CONFIG_GLOBAL_CONFIG_H

#include <cstdint>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include "common/common_define.h"
#include "flow_func/flow_func_config.h"

namespace FlowFunc {
class GlobalConfig : public FlowFuncConfig {
public:
    static GlobalConfig &Instance();

    bool IsOnDevice() const override {
        return on_device_;
    }

    bool IsRunOnAiCpu() const override {
        return run_on_aicpu_;
    }

    void SetRunOnAiCpu(bool run_on_aicpu) {
        run_on_aicpu_ = run_on_aicpu;
    }

    uint32_t GetDeviceId() const override {
        return device_id_;
    }

    void SetDeviceId(uint32_t device_id) {
        device_id_ = device_id;
    }

    bool IsNpuSched() const {
        return npu_sched_;
    }

    void SetNpuSched(bool npu_sched) {
        npu_sched_ = npu_sched;
    }

    void SetMessageQueueIds(const uint32_t request_queue_id, const uint32_t response_queue_id) {
        request_queue_id_ = request_queue_id;
        response_queue_id_ = response_queue_id;
    }

    uint32_t GetReqQueueId() const {
        return request_queue_id_;
    }

    uint32_t GetRspQueueId() const {
        return response_queue_id_;
    }

    const std::string &GetMemGroupName() const {
        return mem_group_name_;
    }

    void SetMemGroupName(const std::string &mem_group_name) {
        mem_group_name_ = mem_group_name;
    }

    void SetBaseDir(const std::string &base_dir) {
        base_dir_ = base_dir;
    }

    const std::string &GetBaseDir() const {
        return base_dir_;
    }

    int32_t GetPhyDeviceId() const override {
        return phy_device_id_;
    }

    void SetPhyDeviceId(int32_t phy_device_id) {
        phy_device_id_ = phy_device_id;
    }

    int32_t GetRunningDeviceId() const {
        return running_device_id_;
    }

    void SetRunningDeviceId(int32_t running_device_id) {
        running_device_id_ = running_device_id;
    }

    uint32_t GetMainSchedGroupId() const override {
        return main_sched_group_id_;
    }

    void SetMainSchedGroupId(uint32_t main_sched_group_id)
    {
        main_sched_group_id_ = main_sched_group_id;
    }

    uint32_t GetInvokeModelSchedGroupId() const {
        return invoke_model_sched_group_id_;
    }

    void SetInvokeModelSchedGroupId(uint32_t invoke_model_sched_group_id) {
        invoke_model_sched_group_id_ = invoke_model_sched_group_id;
    }

    int32_t GetMemGroupId() const override {
        return mem_group_id_;
    }

    void SetMemGroupId(int32_t mem_group_id) {
        mem_group_id_ = mem_group_id;
    }

    void SetLimitRunBuiltinUdf(bool limit_run_builtin_udf) {
        limit_run_builtin_udf_ = limit_run_builtin_udf;
    }

    bool IsLimitRunBuiltinUdf() const {
        return limit_run_builtin_udf_;
    }
 
    void SetAbnormalStatus(const bool status) {
        abnormal_status_.store(status);
    }

    void SetBufCfg(const std::vector<BufferConfigItem> &buf_cfg) {
        buf_cfg_ = buf_cfg;
    }

    const std::vector<BufferConfigItem> &GetBufCfg() const {
        return buf_cfg_;
    }

    void SetLimitThreadInitFunc(const std::function<int32_t()> &limit_thread_init_func) {
        limit_thread_init_func_ = limit_thread_init_func;
    }

    std::function<int32_t()> GetLimitThreadInitFunc() const override {
        return limit_thread_init_func_;
    }

    uint32_t GetWorkerSchedGroupId() const override {
        return worker_sched_group_id_;
    }

    void SetWorkerSchedGroupId(uint32_t worker_sched_group_id) {
        worker_sched_group_id_ = worker_sched_group_id;
    }

    bool GetAbnormalStatus() const override {
        return abnormal_status_.load();
    }

    void SetCurrentSchedThreadIdx(uint32_t current_sched_thread_idx) const {
        current_sched_thread_idx_ = current_sched_thread_idx;
    }

    uint32_t GetCurrentSchedThreadIdx() const override {
        return current_sched_thread_idx_;
    }

    void SetCurrentSchedGroupId(uint32_t current_sched_group_id) override {
        current_sched_group_id_ = current_sched_group_id;
    }

    uint32_t GetCurrentSchedGroupId() const override {
        return current_sched_group_id_;
    }

    void SetFlowMsgQueueSchedGroupId(uint32_t flow_msg_queue_sched_group_id) {
        flow_msg_queue_sched_group_id_ = flow_msg_queue_sched_group_id;
    }

    uint32_t GetFlowMsgQueueSchedGroupId() const override {
        return flow_msg_queue_sched_group_id_;
    }

    void SetWorkerNum(uint32_t worker_num) {
        worker_num_ = worker_num;
    }

    uint32_t GetWorkerNum() const override {
        return worker_num_;
    }

    void SetExitFlag(const bool flag) {
        exit_flag_.store(flag);
    }

    bool GetExitFlag() const override {
        return exit_flag_.load();
    }

private:
    GlobalConfig() = default;

    ~GlobalConfig() override = default;

    static bool on_device_;
    bool run_on_aicpu_ = false;
    bool limit_run_builtin_udf_ = false;
    std::string mem_group_name_;
    thread_local static uint32_t current_sched_thread_idx_;
    thread_local static uint32_t current_sched_group_id_;
    std::string base_dir_;
    uint32_t device_id_ = 0U;
    int32_t phy_device_id_ = -1;
    int32_t running_device_id_ = -1;
    uint32_t main_sched_group_id_ = 0U;
    uint32_t worker_sched_group_id_ = 0U;
    uint32_t invoke_model_sched_group_id_ = 1U;
    uint32_t flow_msg_queue_sched_group_id_ = 2U;
    int32_t mem_group_id_ = 0;
    uint32_t request_queue_id_ = UINT32_MAX;
    uint32_t response_queue_id_ = UINT32_MAX;
    std::atomic<bool> abnormal_status_{false};
    std::atomic<bool> exit_flag_{false};
    std::vector<BufferConfigItem> buf_cfg_;
    std::function<int32_t()> limit_thread_init_func_;
    uint32_t worker_num_ = 0U;
    bool npu_sched_ = false;
};
}

#endif // UDF_CONFIG_GLOBAL_CONFIG_H
