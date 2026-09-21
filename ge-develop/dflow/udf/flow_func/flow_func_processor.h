/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_PROCESSOR_H
#define FLOW_FUNC_PROCESSOR_H

#include <map>
#include <set>
#include <atomic>
#include <chrono>
#include <mutex>
#include <functional>
#include "common/common_define.h"
#include "flow_func/flow_func_run_context.h"
#include "func_wrapper.h"
#include "async_executor.h"
#include "mbuf_flow_msg.h"
#include "mbuf_flow_msg_queue.h"
#include "flow_func_statistic.h"
#include "reader_writer/mbuf_reader.h"
#include "reader_writer/mbuf_writer.h"

namespace FlowFunc {
using ReportStatusMbufGenFunc =
    std::function<int32_t(const std::vector<QueueDevInfo> &input_queue_infos, const uint32_t model_uuid,
                          const uint32_t input_consume_num, Mbuf *&mbuf_to_generate)>;
using ReportExceptionMbufGenFunc =
    std::function<int32_t(const std::string &current_scope, const UdfExceptionInfo &exception_info,
                  Mbuf *&mbuf_to_generate)>;

enum class FLOW_FUNC_VISIBILITY FlowFuncProcessorStatus : int32_t {
    kInit = 0,
    kInitFlowFunc,
    kReady,
    kPrepareInputData,
    kCallFlowFunc,
    kRepublishOutputData,
    kScheduleFinish,
    kSuspend,
    kProcError,
    kCallFlowFuncProcExp
};

enum class FLOW_FUNC_VISIBILITY StatusQueueMsgType : uint32_t {
    kReportStatusMsgType = 0,
    kRaiseExceptionMsgType = 1,
};

class FLOW_FUNC_VISIBILITY FlowFuncProcessor {
public:
    FlowFuncProcessor(const std::shared_ptr<FlowFuncParams> &params, const std::string &flow_func_name,
        const std::vector<QueueDevInfo> &input_queue_infos, const std::vector<QueueDevInfo> &all_out_queue_infos,
        const std::vector<uint32_t> &usable_output_indexes,
        const std::shared_ptr<AsyncExecutor> &async_executor = nullptr);

    ~FlowFuncProcessor();

    int32_t Init(uint32_t device_id);

    int32_t InitFlowFunc();

    /**
     * @brief empty to not empty event.
     * @return true:need reschedule
     */
    bool EmptyToNotEmpty();

    /**
     * @brief full to not full event.
     * @return true:need reschedule
     */
    bool FullToNotFull();

    bool IsOk() const {
        return status_ != FlowFuncProcessorStatus::kProcError;
    }

    /**
     * @brief schedule flow func
     * @return true:need reschedule again
     */
    bool Schedule(uint32_t thread_idx);

    const std::string &GetFlowFuncInfo() const {
        return flow_func_info_;
    }

    void DumpFlowFuncInfo(bool with_queue_info = false) const;

    void DumpModelMetrics(bool with_queue_info = false) const;

    void SetProcessorIdx(const uint32_t processor_idx) {
        processor_idx_ = processor_idx;
    }

    uint32_t GetProcessorIdx() const {
        return processor_idx_;
    }

    void ExecuteFunc();

    int32_t WriteStatusOutputQueue(const ReportStatusMbufGenFunc &report_status_mbuf_gen_func);
    int32_t WriteStatusOutputQueue(uint64_t trans_id,
                                   const ReportExceptionMbufGenFunc &report_exception_mbuf_gen_func);

    void SetClearAndSuspend();

    void SetClearAndRecover();

    bool NeedReplenishSchedule() const;

    void DumpInputData(const std::string &op_name,
                       const std::vector<MbufTensor *> &tensors,
                       uint32_t step_id) const;

    void DumpOutputData(const std::string &op_name, uint32_t out_idx,
                        const MbufTensor *tensor, uint32_t step_id) const;

    void SetInputAlignAttrs(uint32_t align_max_cache_num, int32_t align_timeout, bool drop_when_not_align) {
        align_max_cache_num_ = align_max_cache_num;
        align_timeout_ = align_timeout;
        drop_when_not_align_ = drop_when_not_align;
    }

    void RecordExceptionInfo(const UdfExceptionInfo &exp_info);

    bool CheckSameScope(const std::string &scope) const;

    void RecordDeleteException(uint64_t trans_id);

    void ReleaseFuncWrapper();

private:
    void ScheduleCallback(bool &need_reschedule);

    void SetInputData(std::vector<Mbuf *> &data);

    int32_t SetOutput(uint32_t out_idx, const std::shared_ptr<FlowMsg> &out_msg);

    bool DoSchedule(uint32_t thread_idx);
    /**
     * @brief prepare inputs.
     * @return true:inputs ready. false:not ready
     */
    bool PrepareInputs();

    /**
     * @brief call flow func.
     */
    void CallFlowFunc();

    void CallFlowFuncProcExp();
    /**
     * @brief republish outputs.
     * @return true:publish finish, false:not finish
     */
    bool RepublishOutputs(bool &need_retry);

    bool RepublishOutputs(size_t out_idx, bool &need_retry);

    void ClearNotEmptyEventFlag();

    void ClearNotFullEventFlag();

    bool CheckAndSetWaitNotEmpty();

    bool CheckAndSetWaitNotFull();

    void InitScheduleVar();

    int32_t CreateRunContext(uint32_t device_id);

    int32_t InitReader();

    int32_t InitWriter();

    int32_t InitInputFlowMsgQueues();

    int32_t CreateStatusWriter();

    static void CreateQueueWrapper(const QueueDevInfo &queue_dev_info,
        std::shared_ptr<QueueWrapper> &queue_wrapper);

    void ResetProcessor();

    int32_t CreateFuncWrapper();

    void DiscardAllInputData();

    // return: whether return directly. true: return directly.
    bool PreCheckSpecialStatus();

    void TryClearAndSuspend();

    void TryClearAndRecover();

    template <typename T>
    void SendEventToExecutor(uint32_t event_id, T msg);

    bool NeedProcException();

    void CachedBufferClear();

    void DumpFlowFuncQueueInfo() const;

    void ProcCallFlowFuncFailed();

    void OnScheduleReady();

    bool OnPrepareInput(bool &need_retry);

    void OnScheduleFinish();

    const std::string flow_func_name_;
    const std::string pp_name_;

    std::vector<std::vector<Mbuf *>> cache_output_data_;

    volatile FlowFuncProcessorStatus status_ = FlowFuncProcessorStatus::kInit;

    std::unique_ptr<MbufReader> reader_;

    std::vector<std::unique_ptr<MbufWriter>> writers_;

    std::unique_ptr<MbufWriter> status_writer_;

    std::vector<std::shared_ptr<FlowMsgQueue>> flow_msg_queues_;

    std::vector<std::shared_ptr<FlowMsg>> input_data_;
    uint64_t current_trans_id_ = UINT64_MAX;

    std::atomic_flag schedule_control_lock_{ATOMIC_FLAG_INIT};
    std::atomic_flag wait_schedule_lock_{ATOMIC_FLAG_INIT};
    volatile bool wait_schedule_flag_ = false;

    std::mutex queue_event_guard_;
    // std::mutex writerGuard_;
    volatile bool not_empty_event_ = false;
    volatile bool wait_not_empty_event_ = false;
    volatile bool not_full_event_ = false;
    volatile bool wait_not_full_event_ = false;
    volatile bool try_clear_and_suspend_ = false;
    volatile bool try_clear_and_recover_ = false;

    std::shared_ptr<FlowFuncRunContext> run_context_;
    std::shared_ptr<FlowFuncParams> params_;
    std::shared_ptr<FuncWrapper> func_wrapper_;
    int32_t proc_ret_ = FLOW_FUNC_SUCCESS;
    uint32_t processor_idx_ = 0U;

    const std::vector<QueueDevInfo> input_queue_infos_;
    const std::vector<QueueDevInfo> all_output_queue_infos;
    const std::vector<uint32_t> usable_output_indexes_;

    // statistic
    FlowFuncStatistic flow_func_statistic_;
    std::atomic<uint64_t> input_ready_times_{0};
    std::atomic<uint64_t> call_flow_func_times_{0};
    std::atomic<uint64_t> schedule_finish_times_{0};
    std::vector<std::atomic<uint64_t>> set_output_times_;
    std::vector<std::atomic<uint64_t>> cached_nums_;
    // set output times in a round
    std::vector<std::atomic<uint64_t>> set_output_times_round_;
    // used for log flow func
    std::string flow_func_info_;

    uint32_t input_consume_num_ = 0U;

    std::chrono::steady_clock::time_point last_schedule_time_;
    volatile uint32_t dump_step_id_ = 0U;

    std::shared_ptr<AsyncExecutor> async_executor_ = nullptr;

    uint32_t align_max_cache_num_ = 0; // 0 means align not enable
    int32_t align_timeout_ = -1;  // -1 means never timeout
    bool drop_when_not_align_ = false;
    std::mutex exp_mutex_;
    std::map<uint64_t, UdfExceptionInfo> trans_id_to_exception_infos_;
    std::mutex clear_exp_mutex_;
    std::set<uint64_t> delete_exp_trans_ids_;
    bool is_stream_input_ = false;
};
}  // namespace FlowFunc

#endif  // FLOW_FUNC_PROCESSOR_H
