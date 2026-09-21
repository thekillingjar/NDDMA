/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_common/statistic_manager.h"
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/llm_common.h"

namespace FlowFunc {
StatisticManager &StatisticManager::GetInstance() {
    static StatisticManager manager;
    return manager;
}

StatisticManager::StatisticManager() {
    constexpr double kOneUs = 1000.0;            // ns of one us
    constexpr double kOneSecond = 1000000000.0;  // ns of one second
    const uint64_t freq = GetSystemFreq();
    one_us_for_tick_ = kOneUs / (kOneSecond / static_cast<double>(freq));
    UDF_LOG_INFO("one_us_for_tick_: %lf", one_us_for_tick_);
}

FuncStatisticInfo &StatisticManager::GetStatisticInfo() {
    return stat_info_;
}

void StatisticManager::Dump() {
    // push kv time cost
    const double kTransferKvMaxTimeCost = GetTimeCost(stat_info_.transfer_kv_max_tick_cost.load());
    const double kTransferKvMinTimeCost = GetTimeCost(stat_info_.transfer_kv_min_tick_cost.load());
    const double kTransferKvAvgTimeCost =
            CalcAverageTimeCost(stat_info_.transfer_kv_succ_times.load(), stat_info_.transfer_kv_total_tick_cost.load());
    // sync kv time cost
    const double kSyncKvMaxTimeCost = GetTimeCost(stat_info_.sync_kv_max_tick_cost.load());
    const double kSyncKvMinTimeCost = GetTimeCost(stat_info_.sync_kv_min_tick_cost.load());
    const double kSyncKvAvgTimeCost =
        CalcAverageTimeCost(stat_info_.sync_kv_succ_times.load(), stat_info_.sync_kv_total_tick_cost.load());
    // merge kv time cost
    const double kMergeKvMaxTimeCost = GetTimeCost(stat_info_.merge_kv_max_tick_cost.load());
    const double kMergeKvMinTimeCost = GetTimeCost(stat_info_.merge_kv_min_tick_cost.load());
    const double kMergeKvAvgTimeCost =
        CalcAverageTimeCost(stat_info_.merge_kv_succ_times.load(), stat_info_.merge_kv_total_tick_cost.load());
    // execute model time cost
    const double kExecuteModelMaxTimeCost = GetTimeCost(stat_info_.execute_model_max_tick_cost.load());
    const double kExecuteModelMinTimeCost = GetTimeCost(stat_info_.execute_model_min_tick_cost.load());
    const double kExecuteModelAvgTimeCost =
        CalcAverageTimeCost(stat_info_.execute_model_succ_times.load(),
                            stat_info_.execute_model_total_tick_cost.load());
    // update link time cost
    const double kLinkMaxTime = GetTimeCost(stat_info_.link_max_tick_cost.load());
    const double kLinkMinTime = GetTimeCost(stat_info_.link_min_tick_cost);
    const double kLinkAvgTime =
        CalcAverageTimeCost(stat_info_.link_succ_times.load(), stat_info_.link_total_tick_cost);
    const double kUnlinkMaxTime = GetTimeCost(stat_info_.unlink_max_tick_cost.load());
    const double kUnlinkMinTime = GetTimeCost(stat_info_.unlink_min_tick_cost);
    const double kUnlinkAvgTime =
        CalcAverageTimeCost(stat_info_.unlink_succ_times.load(), stat_info_.unlink_total_tick_cost);
    const auto kUnlinkSucTimes = stat_info_.unlink_succ_times.load();
    const auto kUnlinkFailTimes = stat_info_.unlink_fail_times.load();
    const double kCopyKvMaxTimeCost = GetTimeCost(stat_info_.copy_kv_max_tick_cost.load());
    const double kCopyKvMinTimeCost = GetTimeCost(stat_info_.copy_kv_min_tick_cost.load());
    const double kCopyKvAvgTimeCost = CalcAverageTimeCost(stat_info_.copy_kv_total_times,
                                                         stat_info_.copy_kv_total_tick_cost);
    if (stat_info_.is_server) {
        UDF_RUN_LOG_INFO(
            "Func statistic info: current entity num:%zu, "
            "Allocate=[succ:%lu, fail:%lu], "
            "Deallocate=[succ:%lu, fail:%lu], "
            "Remove key=[succ:%lu, fail:%lu], "
            "copy kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "transfer kv stat=[succ:%lu, fail:%lu, total:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "execute model func=[succ:%lu, fail:%lu, total:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "link stat=[succ:%lu, fail:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "unlink stat=[succ:%lu, fail:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
            "risk stat=[total:%lu].",
            LlmCommEntityMgr::GetInstance().GetEntityMapSize(),
            stat_info_.call_allocate_suc_times.load(), stat_info_.call_allocate_fail_times.load(),
            stat_info_.call_deallocate_suc_times.load(), stat_info_.call_deallocate_fail_times.load(),
            stat_info_.call_remove_key_suc_times.load(), stat_info_.call_remove_key_fail_times.load(),
            stat_info_.copy_kv_total_times.load(), kCopyKvMaxTimeCost, kCopyKvMinTimeCost, kCopyKvAvgTimeCost,
            stat_info_.transfer_kv_succ_times.load(), stat_info_.transfer_kv_fail_times.load(),
            stat_info_.transfer_kv_total_times.load(), kTransferKvMaxTimeCost, kTransferKvMinTimeCost, kTransferKvAvgTimeCost,
            stat_info_.execute_model_succ_times.load(), stat_info_.execute_model_fail_times.load(),
            stat_info_.execute_model_total_times.load(), kExecuteModelMaxTimeCost, kExecuteModelMinTimeCost,
            kExecuteModelAvgTimeCost,
            stat_info_.link_succ_times.load(), stat_info_.link_fail_times.load(),
            kLinkMaxTime, kLinkMinTime, kLinkAvgTime,
            kUnlinkSucTimes, kUnlinkFailTimes, kUnlinkMaxTime,
            kUnlinkMinTime, kUnlinkAvgTime,
            stat_info_.risk_link_total_times.load());
        return;
    }
    UDF_RUN_LOG_INFO(
        "Func statistic info: "
        "Allocate=[succ:%lu, fail:%lu], "
        "Deallocate=[succ:%lu, fail:%lu], "
        "Remove key=[succ:%lu, fail:%lu], "
        "copy kv=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "sync kv stat=[succ:%lu, fail:%lu, total:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "merge kv func=[succ:%lu, fail:%lu, total:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "execute model func=[succ:%lu, fail:%lu, total:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "update link func=[total:%lu], "
        "link stat=[succ:%lu, fail:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "unlink stat=[succ:%lu, fail:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "reset times=[%lu].",
        stat_info_.call_allocate_suc_times.load(), stat_info_.call_allocate_fail_times.load(),
        stat_info_.call_deallocate_suc_times.load(), stat_info_.call_deallocate_fail_times.load(),
        stat_info_.call_remove_key_suc_times.load(), stat_info_.call_remove_key_fail_times.load(),
        stat_info_.copy_kv_total_times.load(), kCopyKvMaxTimeCost, kCopyKvMinTimeCost, kCopyKvAvgTimeCost,
        stat_info_.sync_kv_succ_times.load(), stat_info_.sync_kv_fail_times.load(), stat_info_.sync_kv_total_times.load(),
        kSyncKvMaxTimeCost, kSyncKvMinTimeCost, kSyncKvAvgTimeCost,
        stat_info_.merge_kv_succ_times.load(), stat_info_.merge_kv_fail_times.load(), stat_info_.merge_kv_total_times.load(),
        kMergeKvMaxTimeCost, kMergeKvMinTimeCost, kMergeKvAvgTimeCost,
        stat_info_.execute_model_succ_times.load(), stat_info_.execute_model_fail_times.load(),
        stat_info_.execute_model_total_times.load(),
        kExecuteModelMaxTimeCost, kExecuteModelMinTimeCost, kExecuteModelAvgTimeCost,
        stat_info_.update_link_total_times.load(),
        stat_info_.link_succ_times.load(), stat_info_.link_fail_times.load(), kLinkMaxTime, kLinkMinTime, kLinkAvgTime,
        kUnlinkSucTimes, kUnlinkFailTimes, kUnlinkMaxTime, kUnlinkMinTime, kUnlinkAvgTime,
        stat_info_.reset_times.load());
}

void StatisticManager::PrintStatisticInfo() {
    // ignore first n record
    if (stat_info_.execute_model_total_times.load() <= kIgnoreFirstRecordCount) {
        return;
    }
    if (force_print_flag_.load() || stat_info_.execute_model_total_times.load() % kPrintInterval == 1UL) {
        Dump();
        force_print_flag_.store(false);
        if (stat_info_.execute_model_total_times.load() > kRecordMaxLimit) {
            stat_info_.Reset();
        }
    }
}

void StatisticManager::AddImprobeCost(uint64_t cost) {
    rec_info_track_.improbe_num++;
    rec_info_track_.total_improbe_cost += cost;
    rec_info_track_.max_improbe_cost = (rec_info_track_.max_improbe_cost > cost) ?
        rec_info_track_.max_improbe_cost : cost;
    rec_info_track_.min_improbe_cost = (rec_info_track_.min_improbe_cost < cost) ?
        rec_info_track_.min_improbe_cost : cost;
}

void StatisticManager::AddGetCountCost(uint64_t cost) {
    rec_info_track_.get_count_num++;
    rec_info_track_.total_get_count_cost += cost;
    rec_info_track_.max_get_count_cost = (rec_info_track_.max_get_count_cost > cost) ? rec_info_track_.max_get_count_cost : cost;
    rec_info_track_.min_get_count_cost = (rec_info_track_.min_get_count_cost < cost) ? rec_info_track_.min_get_count_cost : cost;
}

void StatisticManager::AddImrecvCost(uint64_t cost) {
    rec_info_track_.imrecv_num++;
    rec_info_track_.total_imrecv_cost += cost;
    rec_info_track_.max_imrecv_cost = (rec_info_track_.max_imrecv_cost > cost) ? rec_info_track_.max_imrecv_cost : cost;
    rec_info_track_.min_imrecv_cost = (rec_info_track_.min_imrecv_cost < cost) ? rec_info_track_.min_imrecv_cost : cost;
}

void StatisticManager::AddImrecvScatterCost(uint64_t cost) {
    rec_info_track_.imrecv_scatter_num++;
    rec_info_track_.total_imrecv_scatter_cost += cost;
    rec_info_track_.max_imrecv_scatter_cost =
        (rec_info_track_.max_imrecv_scatter_cost > cost) ? rec_info_track_.max_imrecv_scatter_cost : cost;
    rec_info_track_.min_imrecv_scatter_cost =
        (rec_info_track_.min_imrecv_scatter_cost < cost) ? rec_info_track_.min_imrecv_scatter_cost : cost;
}

void StatisticManager::AddIsendCost(uint64_t cost) {
    send_info_track_.isend_num++;
    send_info_track_.total_isend_cost += cost;
    send_info_track_.max_isend_cost = (send_info_track_.max_isend_cost > cost) ? send_info_track_.max_isend_cost : cost;
    send_info_track_.min_isend_cost = (send_info_track_.min_isend_cost < cost) ? send_info_track_.min_isend_cost : cost;
}

void StatisticManager::AddTestSomeCost(uint64_t cost) {
    test_some_info_track_.test_some_num++;
    test_some_info_track_.total_test_some_cost += cost;
    test_some_info_track_.max_test_some_cost =
        (test_some_info_track_.max_test_some_cost > cost) ? test_some_info_track_.max_test_some_cost : cost;
    test_some_info_track_.min_test_some_cost =
        (test_some_info_track_.min_test_some_cost < cost) ? test_some_info_track_.min_test_some_cost : cost;
}

void StatisticManager::AddRawAcceptCost(uint64_t cost, bool succ_flag) {
    if (succ_flag) {
        link_info_track_.raw_accept_succ_num++;
        link_info_track_.total_raw_accept_succ_cost += cost;
        link_info_track_.max_raw_accept_succ_cost =
            (link_info_track_.max_raw_accept_succ_cost > cost) ? link_info_track_.max_raw_accept_succ_cost : cost;
        link_info_track_.min_raw_accept_succ_cost =
            (link_info_track_.min_raw_accept_succ_cost < cost) ? link_info_track_.min_raw_accept_succ_cost : cost;
    } else {
        link_info_track_.raw_accept_empty_num++;
        link_info_track_.total_raw_accept_empty_cost += cost;
        link_info_track_.max_raw_accept_empty_cost =
            (link_info_track_.max_raw_accept_empty_cost > cost) ? link_info_track_.max_raw_accept_empty_cost : cost;
        link_info_track_.min_raw_accept_empty_cost =
            (link_info_track_.min_raw_accept_empty_cost < cost) ? link_info_track_.min_raw_accept_empty_cost : cost;
    }
}

void StatisticManager::AddRawConnectCost(uint64_t cost, bool succ_flag) {
    if (succ_flag) {
        link_info_track_.raw_connect_succ_num++;
        link_info_track_.total_raw_connect_succ_cost += cost;
        link_info_track_.max_raw_connect_succ_cost =
            (link_info_track_.max_raw_connect_succ_cost > cost) ? link_info_track_.max_raw_connect_succ_cost : cost;
        link_info_track_.min_raw_connect_succ_cost =
            (link_info_track_.min_raw_connect_succ_cost < cost) ? link_info_track_.min_raw_connect_succ_cost : cost;
    } else {
        link_info_track_.raw_connect_empty_num++;
        link_info_track_.total_raw_connect_empty_cost += cost;
        link_info_track_.max_raw_connect_empty_cost =
            (link_info_track_.max_raw_connect_empty_cost > cost) ? link_info_track_.max_raw_connect_empty_cost : cost;
        link_info_track_.min_raw_connect_empty_cost =
            (link_info_track_.min_raw_connect_empty_cost < cost) ? link_info_track_.min_raw_connect_empty_cost : cost;
    }
}

void StatisticManager::AddRawCloseCost(uint64_t cost) {
    link_info_track_.raw_close_num++;
    link_info_track_.total_raw_close_cost += cost;
    link_info_track_.max_raw_close_cost = (link_info_track_.max_raw_close_cost > cost) ? link_info_track_.max_raw_close_cost : cost;
    link_info_track_.min_raw_close_cost = (link_info_track_.min_raw_close_cost < cost) ? link_info_track_.min_raw_close_cost : cost;
}

void StatisticManager::AddRawOpenCost(uint64_t cost) {
    link_info_track_.raw_open_num++;
    link_info_track_.total_raw_open_cost += cost;
    link_info_track_.max_raw_open_cost = (link_info_track_.max_raw_open_cost > cost) ? link_info_track_.max_raw_open_cost : cost;
    link_info_track_.min_raw_open_cost = (link_info_track_.min_raw_open_cost < cost) ? link_info_track_.min_raw_open_cost : cost;
}

void StatisticManager::DumpServerProfilingTrack() const {
    // probe time cost
    const double kMaxImprobeTimeCost = GetTimeCost(rec_info_track_.max_improbe_cost);
    const double kMinImprobeTimeCost = GetTimeCost(rec_info_track_.min_improbe_cost);
    const double kAvgImprobeTimeCost = CalcAverageTimeCost(rec_info_track_.improbe_num, rec_info_track_.total_improbe_cost);
    // get count time cost
    const double kMaxGetCountTimeCost = GetTimeCost(rec_info_track_.max_get_count_cost);
    const double kMinGetCountTimeCost = GetTimeCost(rec_info_track_.min_get_count_cost);
    const double kAvgGetCountTimeCost = CalcAverageTimeCost(rec_info_track_.get_count_num,
                                                           rec_info_track_.total_get_count_cost);
    // imrecv time cost
    const double kMaxImrecvTimeCost = GetTimeCost(rec_info_track_.max_imrecv_cost);
    const double kMinImrecvTimeCost = GetTimeCost(rec_info_track_.min_imrecv_cost);
    const double kAvgImrecvTimeCost = CalcAverageTimeCost(rec_info_track_.imrecv_num, rec_info_track_.total_imrecv_cost);
    // send time cost
    const double kMaxIsendTimeCost = GetTimeCost(send_info_track_.max_isend_cost);
    const double kMinIsendTimeCost = GetTimeCost(send_info_track_.min_isend_cost);
    const double kAvgIsendTimeCost = CalcAverageTimeCost(send_info_track_.isend_num, send_info_track_.total_isend_cost);
    // test some cost
    const double kMaxTestSomeTimeCost = GetTimeCost(test_some_info_track_.max_test_some_cost);
    const double kMinTestSomeTimeCost = GetTimeCost(test_some_info_track_.min_test_some_cost);
    const double kAvgTestSomeTimeCost = CalcAverageTimeCost(test_some_info_track_.test_some_num,
                                                           test_some_info_track_.total_test_some_cost);
    // link open cost
    const double kMaxRawOpenTimeCost = GetTimeCost(link_info_track_.max_raw_open_cost);
    const double kMinRawOpenTimeCost = GetTimeCost(link_info_track_.min_raw_open_cost);
    const double kAvgRawOpenTimeCost = CalcAverageTimeCost(link_info_track_.raw_open_num, link_info_track_.total_raw_open_cost);
    // link close cost
    const double kMaxRawCloseTimeCost = GetTimeCost(link_info_track_.max_raw_close_cost);
    const double kMinRawCloseTimeCost = GetTimeCost(link_info_track_.min_raw_close_cost);
    const double kAvgRawCloseTimeCost = CalcAverageTimeCost(link_info_track_.raw_close_num,
                                                           link_info_track_.total_raw_close_cost);
    // link accept cost
    const double kMaxRawAcceptTimeCost = GetTimeCost(link_info_track_.max_raw_accept_succ_cost);
    const double kMinRawAcceptTimeCost = GetTimeCost(link_info_track_.min_raw_accept_succ_cost);
    const double kAvgRawAcceptTimeCost = CalcAverageTimeCost(link_info_track_.raw_accept_succ_num,
                                                            link_info_track_.total_raw_accept_succ_cost);
    const double kMaxRawAcceptEmptyTimeCost = GetTimeCost(link_info_track_.max_raw_accept_empty_cost);
    const double kMinRawAcceptEmptyTimeCost = GetTimeCost(link_info_track_.min_raw_accept_empty_cost);
    const double kAvgRawAcceptEmptyTimeCost = CalcAverageTimeCost(link_info_track_.raw_accept_empty_num,
                                                                 link_info_track_.total_raw_accept_empty_cost);
    UDF_RUN_LOG_INFO(
        "Hccl API statistic info: HcclRawAccept=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawAccept=[empty:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawOpen=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawClose=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawImprobe=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawGetCount=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawImrecv=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawIsend=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawTestSome=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us].",
        link_info_track_.raw_accept_succ_num, kMaxRawAcceptTimeCost, kMinRawAcceptTimeCost, kAvgRawAcceptTimeCost,
        link_info_track_.raw_accept_empty_num, kMaxRawAcceptEmptyTimeCost, kMinRawAcceptEmptyTimeCost,
        kAvgRawAcceptEmptyTimeCost,
        link_info_track_.raw_open_num, kMaxRawOpenTimeCost, kMinRawOpenTimeCost, kAvgRawOpenTimeCost,
        link_info_track_.raw_close_num, kMaxRawCloseTimeCost, kMinRawCloseTimeCost, kAvgRawCloseTimeCost,
        rec_info_track_.improbe_num, kMaxImprobeTimeCost, kMinImprobeTimeCost, kAvgImprobeTimeCost,
        rec_info_track_.get_count_num, kMaxGetCountTimeCost, kMinGetCountTimeCost, kAvgGetCountTimeCost,
        rec_info_track_.imrecv_num, kMaxImrecvTimeCost, kMinImrecvTimeCost, kAvgImrecvTimeCost,
        send_info_track_.isend_num, kMaxIsendTimeCost, kMinIsendTimeCost, kAvgIsendTimeCost,
        test_some_info_track_.test_some_num, kMaxTestSomeTimeCost, kMinTestSomeTimeCost, kAvgTestSomeTimeCost);
}

void StatisticManager::DumpClientProfilingTrack() const {
    // probe time cost
    const double kMaxImprobeTimeCost = GetTimeCost(rec_info_track_.max_improbe_cost);
    const double kMinImprobeTimeCost = GetTimeCost(rec_info_track_.min_improbe_cost);
    const double kAvgImprobeTimeCost = CalcAverageTimeCost(rec_info_track_.improbe_num, rec_info_track_.total_improbe_cost);
    // get count time cost
    const double kMaxGetCountTimeCost = GetTimeCost(rec_info_track_.max_get_count_cost);
    const double kMinGetCountTimeCost = GetTimeCost(rec_info_track_.min_get_count_cost);
    const double kAvgGetCountTimeCost = CalcAverageTimeCost(rec_info_track_.get_count_num,
                                                           rec_info_track_.total_get_count_cost);
    // imrecv time cost
    const double kMaxImrecvTimeCost = GetTimeCost(rec_info_track_.max_imrecv_cost);
    const double kMinImrecvTimeCost = GetTimeCost(rec_info_track_.min_imrecv_cost);
    const double kAvgImrecvTimeCost = CalcAverageTimeCost(rec_info_track_.imrecv_num, rec_info_track_.total_imrecv_cost);
    // imrecvScatter time cost
    const double kMaxImrecvScatterTimeCost = GetTimeCost(rec_info_track_.max_imrecv_scatter_cost);
    const double kMinImrecvScatterTimeCost = GetTimeCost(rec_info_track_.min_imrecv_scatter_cost);
    const double kAvgImrecvScatterTimeCost =
        CalcAverageTimeCost(rec_info_track_.imrecv_scatter_num, rec_info_track_.total_imrecv_scatter_cost);
    // send time cost
    const double kMaxIsendTimeCost = GetTimeCost(send_info_track_.max_isend_cost);
    const double kMinIsendTimeCost = GetTimeCost(send_info_track_.min_isend_cost);
    const double kAvgIsendTimeCost = CalcAverageTimeCost(send_info_track_.isend_num, send_info_track_.total_isend_cost);
    // test some cost
    const double kMaxTestSomeTimeCost = GetTimeCost(test_some_info_track_.max_test_some_cost);
    const double kMinTestSomeTimeCost = GetTimeCost(test_some_info_track_.min_test_some_cost);
    const double kAvgTestSomeTimeCost = CalcAverageTimeCost(test_some_info_track_.test_some_num,
                                                           test_some_info_track_.total_test_some_cost);
    // link open cost
    const double kMaxRawOpenTimeCost = GetTimeCost(link_info_track_.max_raw_open_cost);
    const double kMinRawOpenTimeCost = GetTimeCost(link_info_track_.min_raw_open_cost);
    const double kAvgRawOpenTimeCost = CalcAverageTimeCost(link_info_track_.raw_open_num, link_info_track_.total_raw_open_cost);
    // link close cost
    const double kMaxRawCloseTimeCost = GetTimeCost(link_info_track_.max_raw_close_cost);
    const double kMinRawCloseTimeCost = GetTimeCost(link_info_track_.min_raw_close_cost);
    const double kAvgRawCloseTimeCost = CalcAverageTimeCost(link_info_track_.raw_close_num,
                                                           link_info_track_.total_raw_close_cost);
    // link connect cost
    const double kMaxRawConnectTimeCost = GetTimeCost(link_info_track_.max_raw_connect_succ_cost);
    const double kMinRawConnectTimeCost = GetTimeCost(link_info_track_.min_raw_connect_succ_cost);
    const double kAvgRawConnectTimeCost = CalcAverageTimeCost(link_info_track_.raw_connect_succ_num,
                                                             link_info_track_.total_raw_connect_succ_cost);
    const double kMaxRawConnectEmptyTimeCost = GetTimeCost(link_info_track_.max_raw_connect_empty_cost);
    const double kMinRawConnectEmptyTimeCost = GetTimeCost(link_info_track_.min_raw_connect_empty_cost);
    const double kAvgRawConnectEmptyTimeCost = CalcAverageTimeCost(link_info_track_.raw_connect_empty_num,
                                                                  link_info_track_.total_raw_connect_empty_cost);
    UDF_RUN_LOG_INFO(
        "Hccl API statistic info: HcclRawConnect=[succ:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawConnect=[empty:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawOpen=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawClose=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawImprobe=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawGetCount=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawImrecv=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawImrecvScatter=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawIsend=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us], "
        "HcclRawTestSome=[num:%lu, max:%.2f us, min:%.2f us, avg:%.2f us].",
        link_info_track_.raw_connect_succ_num, kMaxRawConnectTimeCost, kMinRawConnectTimeCost, kAvgRawConnectTimeCost,
        link_info_track_.raw_connect_empty_num, kMaxRawConnectEmptyTimeCost, kMinRawConnectEmptyTimeCost,
        kAvgRawConnectEmptyTimeCost,
        link_info_track_.raw_open_num, kMaxRawOpenTimeCost, kMinRawOpenTimeCost, kAvgRawOpenTimeCost,
        link_info_track_.raw_close_num, kMaxRawCloseTimeCost, kMinRawCloseTimeCost, kAvgRawCloseTimeCost,
        rec_info_track_.improbe_num, kMaxImprobeTimeCost, kMinImprobeTimeCost, kAvgImprobeTimeCost,
        rec_info_track_.get_count_num, kMaxGetCountTimeCost, kMinGetCountTimeCost, kAvgGetCountTimeCost,
        rec_info_track_.imrecv_num, kMaxImrecvTimeCost, kMinImrecvTimeCost, kAvgImrecvTimeCost,
        rec_info_track_.imrecv_scatter_num, kMaxImrecvScatterTimeCost, kMinImrecvScatterTimeCost, kAvgImrecvScatterTimeCost,
        send_info_track_.isend_num, kMaxIsendTimeCost, kMinIsendTimeCost, kAvgIsendTimeCost,
        test_some_info_track_.test_some_num, kMaxTestSomeTimeCost, kMinTestSomeTimeCost, kAvgTestSomeTimeCost);
}

void StatisticManager::ResetProfilingTrack() {
    link_info_track_.Reset();
    rec_info_track_.Reset();
    send_info_track_.Reset();
    test_some_info_track_.Reset();
}
}  // namespace FlowFunc