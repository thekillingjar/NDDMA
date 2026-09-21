/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_LLM_COMMON_STATISTIC_MANAGER_H
#define BUILT_IN_LLM_COMMON_STATISTIC_MANAGER_H

#include <atomic>
#include <cstdint>
#include <ctime>

namespace FlowFunc {
// print entity statistic info every 1000 sync kv times
constexpr uint64_t kPrintInterval = 1000UL;
// max record entity sync kv 10000000000 times
constexpr uint64_t kRecordMaxLimit = 10000000000UL;
// ignore first n record
constexpr uint64_t kIgnoreFirstRecordCount = 0UL;

struct EntityStatisticInfo {
// reset times
    uint64_t reset_times = 0UL;
    // call hccl api statistic
    uint64_t call_probe_total_times = 0UL;
    uint64_t call_probe_success_times = 0UL;
    uint64_t call_probe_fail_times = 0UL;
    uint64_t call_recv_total_times = 0UL;
    uint64_t call_recv_success_times = 0UL;
    uint64_t call_recv_fail_times = 0UL;
    uint64_t test_recv_success_times = 0UL;
    uint64_t call_send_total_times = 0UL;
    uint64_t call_send_success_times = 0UL;
    uint64_t call_send_fail_times = 0UL;
    uint64_t call_send_full_times = 0UL;
    uint64_t test_send_success_times = 0UL;

    // transfer kv statistic for server
    uint64_t transfer_kv_total_times = 0UL;
    uint64_t transfer_kv_total_tick_cost = 0UL;
    uint64_t transfer_kv_max_tick_cost = 0UL;
    uint64_t transfer_kv_min_tick_cost = UINT64_MAX;
    // send transfer req statistic for server
    uint64_t send_transfer_req_total_times = 0UL;
    uint64_t send_transfer_req_total_tick_cost = 0UL;
    uint64_t send_transfer_req_max_tick_cost = 0UL;
    uint64_t send_transfer_req_min_tick_cost = UINT64_MAX;
    // receive transfer kv statistic for client
    uint64_t recv_transfer_kv_total_times = 0UL;
    uint64_t recv_transfer_kv_total_tick_cost = 0UL;
    uint64_t recv_transfer_kv_max_tick_cost = 0UL;
    uint64_t recv_transfer_kv_min_tick_cost = UINT64_MAX;
    // receive transfer req statistic for client
    uint64_t recv_transfer_req_total_times = 0UL;
    uint64_t recv_transfer_req_total_tick_cost = 0UL;
    uint64_t recv_transfer_req_max_tick_cost = 0UL;
    uint64_t recv_transfer_req_min_tick_cost = UINT64_MAX;

    // send req statistic for client
    uint64_t send_req_total_times = 0UL;
    uint64_t send_req_total_tick_cost = 0UL;
    uint64_t send_req_max_tick_cost = 0UL;
    uint64_t send_req_min_tick_cost = UINT64_MAX;
    // receive kv statistic for client
    uint64_t recv_kv_total_times = 0UL;
    uint64_t recv_kv_total_tick_cost = 0UL;
    uint64_t recv_kv_max_tick_cost = 0UL;
    uint64_t recv_kv_min_tick_cost = UINT64_MAX;
    // receive req statistic for server
    uint64_t recv_req_total_times = 0UL;
    uint64_t recv_req_total_tick_cost = 0UL;
    uint64_t recv_req_max_tick_cost = 0UL;
    uint64_t recv_req_min_tick_cost = UINT64_MAX;
    // send kv statistic for server
    uint64_t send_kv_total_times = 0UL;
    uint64_t send_kv_total_tick_cost = 0UL;
    uint64_t send_kv_max_tick_cost = 0UL;
    uint64_t send_kv_min_tick_cost = UINT64_MAX;
    // sync kv cache total cost for client and server
    uint64_t sync_kv_total_times = 0UL;
    uint64_t sync_kv_total_tick_cost = 0UL;
    uint64_t sync_kv_max_tick_cost = 0UL;
    uint64_t sync_kv_min_tick_cost = UINT64_MAX;
    // link and unlink tick cost for client and server
    uint64_t link_tick_cost = 0UL;
    uint64_t unlink_tick_cost = 0UL;
    void Reset() {
        reset_times++;
        call_probe_total_times = 0UL;
        call_probe_success_times = 0UL;
        call_probe_fail_times = 0UL;
        call_recv_total_times = 0UL;
        call_recv_success_times = 0UL;
        call_recv_fail_times = 0UL;
        test_recv_success_times = 0UL;
        call_send_total_times = 0UL;
        call_send_success_times = 0UL;
        call_send_fail_times = 0UL;
        call_send_full_times = 0UL;
        test_send_success_times = 0UL;
        recv_req_total_times = 0UL;
        recv_req_total_tick_cost = 0UL;
        recv_req_max_tick_cost = 0UL;
        recv_req_min_tick_cost = UINT64_MAX;
        send_kv_total_times = 0UL;
        send_kv_total_tick_cost = 0UL;
        send_kv_max_tick_cost = 0UL;
        send_kv_min_tick_cost = UINT64_MAX;
        send_req_total_times = 0UL;
        send_req_total_tick_cost = 0UL;
        send_req_max_tick_cost = 0UL;
        send_req_min_tick_cost = UINT64_MAX;
        recv_kv_total_times = 0UL;
        recv_kv_total_tick_cost = 0UL;
        recv_kv_max_tick_cost = 0UL;
        recv_kv_min_tick_cost = UINT64_MAX;
        sync_kv_total_times = 0UL;
        sync_kv_total_tick_cost = 0UL;
        sync_kv_max_tick_cost = 0UL;
        sync_kv_min_tick_cost = UINT64_MAX;
        link_tick_cost = 0UL;
        unlink_tick_cost = 0UL;

        transfer_kv_total_times = 0UL;
        transfer_kv_total_tick_cost = 0UL;
        transfer_kv_max_tick_cost = 0UL;
        transfer_kv_min_tick_cost = UINT64_MAX;
        send_transfer_req_total_times = 0UL;
        send_transfer_req_total_tick_cost = 0UL;
        send_transfer_req_max_tick_cost = 0UL;
        send_transfer_req_min_tick_cost = UINT64_MAX;
        recv_transfer_kv_total_times = 0UL;
        recv_transfer_kv_total_tick_cost = 0UL;
        recv_transfer_kv_max_tick_cost = 0UL;
        recv_transfer_kv_min_tick_cost = UINT64_MAX;
        recv_transfer_req_total_times = 0UL;
        recv_transfer_req_total_tick_cost = 0UL;
        recv_transfer_req_max_tick_cost = 0UL;
        recv_transfer_req_min_tick_cost = UINT64_MAX;
    }
};

struct FuncStatisticInfo {
    std::atomic<bool> is_server{false};
    // reset times
    std::atomic<uint64_t> reset_times{0UL};
    std::atomic<uint64_t> call_allocate_suc_times = {0UL};
    std::atomic<uint64_t> call_allocate_fail_times = {0UL};
    std::atomic<uint64_t> call_deallocate_suc_times = {0UL};
    std::atomic<uint64_t> call_deallocate_fail_times = {0UL};
    std::atomic<uint64_t> call_remove_key_suc_times = {0UL};
    std::atomic<uint64_t> call_remove_key_fail_times = {0UL};
    // sync kv
    std::atomic<uint64_t> sync_kv_total_times{0UL};
    std::atomic<uint64_t> sync_kv_succ_times{0UL};
    std::atomic<uint64_t> sync_kv_fail_times{0UL};
    std::atomic<uint64_t> sync_kv_max_tick_cost{0UL};
    std::atomic<uint64_t> sync_kv_min_tick_cost{UINT64_MAX};
    std::atomic<uint64_t> sync_kv_total_tick_cost{0UL};
    // transfer kv
    std::atomic<uint64_t> transfer_kv_total_times{0UL};
    std::atomic<uint64_t> transfer_kv_succ_times{0UL};
    std::atomic<uint64_t> transfer_kv_fail_times{0UL};
    std::atomic<uint64_t> transfer_kv_max_tick_cost{0UL};
    std::atomic<uint64_t> transfer_kv_min_tick_cost{UINT64_MAX};
    std::atomic<uint64_t> transfer_kv_total_tick_cost{0UL};
    // merge kv
    std::atomic<uint64_t> merge_kv_total_times{0UL};
    std::atomic<uint64_t> merge_kv_succ_times{0UL};
    std::atomic<uint64_t> merge_kv_fail_times{0UL};
    std::atomic<uint64_t> merge_kv_max_tick_cost{0UL};
    std::atomic<uint64_t> merge_kv_min_tick_cost{UINT64_MAX};
    std::atomic<uint64_t> merge_kv_total_tick_cost{0UL};
    // execute model
    std::atomic<uint64_t> execute_model_total_times{0UL};
    std::atomic<uint64_t> execute_model_succ_times{0UL};
    std::atomic<uint64_t> execute_model_fail_times{0UL};
    std::atomic<uint64_t> execute_model_max_tick_cost{0UL};
    std::atomic<uint64_t> execute_model_min_tick_cost{UINT64_MAX};
    std::atomic<uint64_t> execute_model_total_tick_cost{0UL};
    // copy kv tensors
    std::atomic<uint64_t> copy_kv_total_times{0UL};
    std::atomic<uint64_t> copy_kv_max_tick_cost{0UL};
    std::atomic<uint64_t> copy_kv_min_tick_cost{UINT64_MAX};
    std::atomic<uint64_t> copy_kv_total_tick_cost{0UL};
    // link/unlink
    std::atomic<uint64_t> update_link_total_times{0UL};
    std::atomic<uint64_t> link_succ_times{0UL};
    std::atomic<uint64_t> link_fail_times{0UL};
    std::atomic<uint64_t> link_max_tick_cost{0UL};
    std::atomic<uint64_t> link_min_tick_cost{UINT64_MAX};
    std::atomic<uint64_t> link_total_tick_cost{0UL};
    std::atomic<uint64_t> unlink_succ_times{0UL};
    std::atomic<uint64_t> unlink_fail_times{0UL};
    std::atomic<uint64_t> unlink_max_tick_cost{0UL};
    std::atomic<uint64_t> unlink_min_tick_cost{UINT64_MAX};
    std::atomic<uint64_t> unlink_total_tick_cost{0UL};
    std::atomic<uint64_t> risk_link_total_times{0UL};
    void Reset() {
        reset_times++;
        call_allocate_suc_times.store(0UL);
        call_allocate_fail_times.store(0UL);
        call_deallocate_suc_times.store(0UL);
        call_deallocate_fail_times.store(0UL);
        call_remove_key_suc_times.store(0UL);
        call_remove_key_fail_times.store(0UL);
        sync_kv_total_times.store(0UL);
        sync_kv_succ_times.store(0UL);
        sync_kv_fail_times.store(0UL);
        sync_kv_max_tick_cost.store(0UL);
        sync_kv_min_tick_cost.store(UINT64_MAX);
        sync_kv_total_tick_cost.store(0UL);
        merge_kv_total_times.store(0UL);
        merge_kv_succ_times.store(0UL);
        merge_kv_fail_times.store(0UL);
        merge_kv_max_tick_cost.store(0UL);
        merge_kv_min_tick_cost.store(UINT64_MAX);
        merge_kv_total_tick_cost.store(0UL);
        execute_model_total_times.store(0UL);
        execute_model_succ_times.store(0UL);
        execute_model_fail_times.store(0UL);
        execute_model_max_tick_cost.store(0UL);
        execute_model_min_tick_cost.store(UINT64_MAX);
        execute_model_total_tick_cost.store(0UL);
        copy_kv_total_times.store(0UL);
        copy_kv_max_tick_cost.store(0UL);
        copy_kv_min_tick_cost.store(UINT64_MAX);
        copy_kv_total_tick_cost.store(0UL);
        update_link_total_times.store(0UL);
        link_succ_times.store(0UL);
        link_fail_times.store(0UL);
        link_max_tick_cost.store(0UL);
        link_min_tick_cost.store(UINT64_MAX);
        link_total_tick_cost.store(0UL);
        unlink_succ_times.store(0UL);
        unlink_fail_times.store(0UL);
        unlink_max_tick_cost.store(0UL);
        unlink_min_tick_cost.store(UINT64_MAX);
        unlink_total_tick_cost.store(0UL);
        risk_link_total_times.store(0UL);
    }
};

struct RecvProfilingTrack {
    // reset times
    uint64_t reset_times{0UL};
    uint64_t improbe_num = 0UL;
    uint64_t total_improbe_cost = 0UL;
    uint64_t max_improbe_cost = 0UL;
    uint64_t min_improbe_cost = UINT64_MAX;
    uint64_t get_count_num = 0UL;
    uint64_t total_get_count_cost = 0UL;
    uint64_t max_get_count_cost = 0UL;
    uint64_t min_get_count_cost = UINT64_MAX;
    uint64_t imrecv_num = 0UL;
    uint64_t imrecv_scatter_num = 0UL;
    uint64_t total_imrecv_cost = 0UL;
    uint64_t total_imrecv_scatter_cost = 0UL;
    uint64_t max_imrecv_cost = 0UL;
    uint64_t max_imrecv_scatter_cost = 0UL;
    uint64_t min_imrecv_cost = UINT64_MAX;
    uint64_t min_imrecv_scatter_cost = UINT64_MAX;

    void Reset() {
        reset_times++;
        improbe_num = 0UL;
        total_improbe_cost = 0UL;
        max_improbe_cost = 0UL;
        min_improbe_cost = UINT64_MAX;
        get_count_num = 0UL;
        total_get_count_cost = 0UL;
        max_get_count_cost = 0UL;
        min_get_count_cost = UINT64_MAX;
        imrecv_num = 0UL;
        imrecv_scatter_num = 0UL;
        total_imrecv_cost = 0UL;
        max_imrecv_cost = 0UL;
        min_imrecv_cost = UINT64_MAX;
        total_imrecv_scatter_cost = 0UL;
        max_imrecv_scatter_cost = 0UL;
        min_imrecv_scatter_cost = UINT64_MAX;
    }
};

struct SendProfilingTrack {
    uint64_t reset_times = 0UL;
    uint64_t isend_num = 0UL;
    uint64_t total_isend_cost = 0UL;
    uint64_t max_isend_cost = 0UL;
    uint64_t min_isend_cost = UINT64_MAX;

    void Reset() {
        reset_times++;
        isend_num = 0UL;
        total_isend_cost = 0UL;
        max_isend_cost = 0UL;
        min_isend_cost = UINT64_MAX;
    }
};

struct TestSomeProfilingTrack {
    uint64_t reset_times = 0UL;
    uint64_t test_some_num = 0UL;
    uint64_t total_test_some_cost = 0UL;
    uint64_t max_test_some_cost = 0UL;
    uint64_t min_test_some_cost = UINT64_MAX;

    void Reset() {
        reset_times++;
        test_some_num = 0UL;
        total_test_some_cost = 0UL;
        max_test_some_cost = 0UL;
        min_test_some_cost = UINT64_MAX;
    }
};

struct LinkProfilingTrack {
    uint64_t raw_accept_empty_num = 0UL;
    uint64_t total_raw_accept_empty_cost = 0UL;
    uint64_t max_raw_accept_empty_cost = 0UL;
    uint64_t min_raw_accept_empty_cost = UINT64_MAX;
    uint64_t raw_accept_succ_num = 0UL;
    uint64_t total_raw_accept_succ_cost = 0UL;
    uint64_t max_raw_accept_succ_cost = 0UL;
    uint64_t min_raw_accept_succ_cost = UINT64_MAX;
    uint64_t raw_connect_empty_num = 0UL;
    uint64_t total_raw_connect_empty_cost = 0UL;
    uint64_t max_raw_connect_empty_cost = 0UL;
    uint64_t min_raw_connect_empty_cost = UINT64_MAX;
    uint64_t raw_connect_succ_num = 0UL;
    uint64_t total_raw_connect_succ_cost = 0UL;
    uint64_t max_raw_connect_succ_cost = 0UL;
    uint64_t min_raw_connect_succ_cost = UINT64_MAX;
    uint64_t raw_open_num = 0UL;
    uint64_t total_raw_open_cost = 0UL;
    uint64_t max_raw_open_cost = 0UL;
    uint64_t min_raw_open_cost = UINT64_MAX;
    uint64_t raw_close_num = 0UL;
    uint64_t total_raw_close_cost = 0UL;
    uint64_t max_raw_close_cost = 0UL;
    uint64_t min_raw_close_cost = UINT64_MAX;

    void Reset() {
        raw_accept_empty_num = 0UL;
        total_raw_accept_empty_cost = 0UL;
        max_raw_accept_empty_cost = 0UL;
        min_raw_accept_empty_cost = UINT64_MAX;
        raw_accept_succ_num = 0UL;
        total_raw_accept_succ_cost = 0UL;
        max_raw_accept_succ_cost = 0UL;
        min_raw_accept_succ_cost = UINT64_MAX;
        raw_connect_empty_num = 0UL;
        total_raw_connect_empty_cost = 0UL;
        max_raw_connect_empty_cost = 0UL;
        min_raw_connect_empty_cost = UINT64_MAX;
        raw_connect_succ_num = 0UL;
        total_raw_connect_succ_cost = 0UL;
        max_raw_connect_succ_cost = 0UL;
        min_raw_connect_succ_cost = UINT64_MAX;
        raw_open_num = 0UL;
        total_raw_open_cost = 0UL;
        max_raw_open_cost = 0UL;
        min_raw_open_cost = UINT64_MAX;
        raw_close_num = 0UL;
        total_raw_close_cost = 0UL;
        max_raw_close_cost = 0UL;
        min_raw_close_cost = UINT64_MAX;
    }
};

class StatisticManager {
public:
    static StatisticManager &GetInstance();
    virtual ~StatisticManager() = default;
    StatisticManager(const StatisticManager &) = delete;
    StatisticManager(const StatisticManager &&) = delete;
    StatisticManager &operator=(const StatisticManager &) = delete;
    StatisticManager &operator=(const StatisticManager &&) = delete;

    inline uint64_t GetCpuTick() const {
        uint64_t tick = 0UL;
#if (defined __aarch64__ || defined __arm__)
        GetAsmCpuTick(tick);
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        tick = (ts.tv_sec * 1000000UL + ts.tv_nsec / 1000UL);
#endif
        return tick;
    }
    inline uint64_t GetSystemFreq() const {
        uint64_t freq = 1UL;
#if (defined __aarch64__ || defined __arm__)
        GetAsmSysFreq(freq);
#else
        freq = 1000000UL;
#endif
        return freq;
    }
    inline double GetTimeCost(uint64_t tick) const {
        if (tick == UINT64_MAX) {
            return 0;
        }
        return static_cast<double>(tick) / one_us_for_tick_;
    }

    inline void SetForcePrintFlag(bool force_print_flag) {
        force_print_flag_.store(force_print_flag);
    }
    FuncStatisticInfo &GetStatisticInfo();
    void Dump();
    void PrintStatisticInfo();

    void AddImprobeCost(uint64_t cost);
    void AddGetCountCost(uint64_t cost);
    void AddImrecvCost(uint64_t cost);
    void AddImrecvScatterCost(uint64_t cost);
    void AddIsendCost(uint64_t cost);
    void AddTestSomeCost(uint64_t cost);
    void AddRawAcceptCost(uint64_t cost, bool succ_flag);
    void AddRawConnectCost(uint64_t cost, bool succ_flag);
    void AddRawOpenCost(uint64_t cost);
    void AddRawCloseCost(uint64_t cost);
    void DumpServerProfilingTrack() const;
    void DumpClientProfilingTrack() const;
    void ResetProfilingTrack();
private:
    StatisticManager();
#if (defined __aarch64__ || defined __arm__)
    inline void GetAsmCpuTick(uint64_t &tick) const {
        asm volatile("mrs %0, CNTVCT_EL0" : "=r"(tick) :);
    }
    inline void GetAsmSysFreq(uint64_t &freq) const {
        asm volatile("mrs %0, CNTFRQ_EL0" : "=r"(freq) :);
    }
#endif

private:
    double one_us_for_tick_ = 0.0f;
    FuncStatisticInfo stat_info_;
    std::atomic<bool> force_print_flag_{false};
    RecvProfilingTrack rec_info_track_;
    SendProfilingTrack send_info_track_;
    TestSomeProfilingTrack test_some_info_track_;
    LinkProfilingTrack link_info_track_;
};
}  // namespace FlowFunc

#endif  // BUILT_IN_LLM_COMMON_STATISTIC_MANAGER_H