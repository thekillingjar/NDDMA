/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_EXECUTOR_H
#define FLOW_FUNC_EXECUTOR_H

#include <mutex>
#include <condition_variable>
#include <memory>
#include "common/common_define.h"
#include "flow_func/flow_func_context.h"
#include "flow_func/flow_func_params.h"
#include "flow_func/flow_func_processor.h"
#include "flow_func/async_executor.h"
#include "model/flow_func_model.h"
#include "flow_func_thread_pool.h"
#include "reader_writer/mbuf_writer.h"
#include "reader_writer/proxy_queue_wrapper.h"
#include "ff_dynamic_sched.pb.h"
#include "ff_deployer.pb.h"
#include "npu_sched_processor.h"

namespace FlowFunc {
using FillFunc = std::function<int32_t(void *const buffer, const size_t size)>;
enum class ControlMessageType : int32_t {
    kInit = -1,
    kSuspend = 1,
    kRecover = 2,
    kException = 3,
    kUnknow = 99,
};

enum class RequestMsgType : int32_t {
    kControlMsg = 0,
    kExceptionMsg = 1,
    kNotify = 2,
};

enum class ExceptionType : uint32_t {
    kAddException = 0,
    kDeleteException = 1,
};

class FlowFuncExecutor {
public:
    FlowFuncExecutor();

    ~FlowFuncExecutor() = default;

    int32_t Init(const std::vector<std::unique_ptr<FlowFuncModel>> &models);

    void Destroy();

    int32_t Start();

    void Stop(bool recv_term_signal = false);

    void WaitForStop();

    void ThreadLoop(uint32_t thread_idx);

private:
    void ProcessEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessFullToNoFullEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessEmptyToNotEmptyEvent(const struct event_info &event, uint32_t thread_idx);

    int32_t ProcessRequestMessageQueue(ControlMessageType &ctrl_msg_type);

    int32_t DequeueAndParseRequestMessage(ff::deployer::ExecutorRequest &req_msg, RequestMsgType &msg_type) const;

    int32_t ParseRequestMessage(Mbuf *control_mbuf, ff::deployer::ExecutorRequest &req_msg,
        RequestMsgType &msg_type) const;

    int32_t ProcessExceptionMsg(const ff::deployer::ExecutorRequest_DataflowExceptionNotify &exp_msg);

    int32_t ProcessControlMsg(ff::deployer::ExecutorRequest_ClearModelRequest &ctrl_msg);

    int32_t SendMessageByResponseQueue(const ControlMessageType &msg_type, const int32_t result);

    void ProcessProcessorInitEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessFlowFuncInitEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessFlowFuncExecuteEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessTimerEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessNotifyThreadExitEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessReportStatusEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessSingleFlowFuncInitEvent(const struct event_info &event, uint32_t thread_idx);

    int32_t ReportStatus(const size_t flow_func_processor_idx);

    int32_t CheckProcessorEventParams(const struct event_info &event);

    void ProcessReportSuspendEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessReportRecoverEvent(const struct event_info &event, uint32_t thread_idx);

    static int32_t GenerateMbuf(const size_t req_size, const FillFunc &fill_func, Mbuf *&mbuf_to_generate);

    int32_t SubscribeInvokeModelEvent(uint32_t thread_idx) const;

    int32_t SubscribeFlowMsgQueueEvent(uint32_t thread_idx) const;

    int32_t SubscribeInputQueue() const;

    int32_t SubscribeStatusOutputQueue() const;

    int32_t SubscribeOutputQueue() const;

    void UnsubscribeInputQueue() const;

    void UnsubscribeOutputQueue() const;

    int32_t ScheduleFlowFunc(size_t flow_func_processor_idx) const;

    int32_t ScheduleFlowFuncInit() const;

    int32_t InitDrv() const;

    int32_t InitQueue() const;

    int32_t InitMessageQueue();

    int32_t SetExecutorEschedPriority() const;

    void DumpMetrics(bool with_queue_info = false) const;

    void UpdatePriority(int32_t user_priority, int32_t &priority) const;

    void DoScheduleFlowFunc(bool is_need_sched, uint32_t flow_func_processor_idx);

    void MonitorParentExit();

    void MonitorTermSignal();

    void CheckReplenishSchedule();

    int32_t SendSwitchSoftModeEvent() const;

    int32_t SendProcessorInitEvent() const;

    void ProcessSwitchSoftModeEvent(const struct event_info &event, uint32_t thread_idx);

    void ProcessRaiseExceptionEvent(const struct event_info &event, uint32_t thread_idx);

    template<typename T>
    static int32_t SerializeProtoToMbuf(const T &proto_msg, Mbuf *&mbuf_to_generate);

    static void ConstructException(const std::string &current_scope, const UdfExceptionInfo &exception_info,
        ff::deployer::SubmodelStatus &exception_msg);

    int32_t CreateFlowFuncProcessor(const FlowFuncModel &model,
        const std::map<std::string, std::vector<QueueDevInfo>> &input_maps,
        const std::map<std::string, std::vector<uint32_t>> &output_index_maps,
        const std::vector<QueueDevInfo> &all_outputs_queue_infos,
        const std::shared_ptr<FlowFuncParams> &params);

    int32_t GetModelQueueInfos(const FlowFuncModel &model, ModelQueueInfos &model_queue_infos);

    int32_t InitNpuSchedProcessor();

    static int32_t CheckInputOutputMapsValid(const std::map<std::string, std::vector<uint32_t>> &input_maps,
        uint32_t inputs_num, const std::map<std::string, std::vector<uint32_t>> &output_maps, uint32_t outputs_num);

    static void GetRealQueueInfos(const std::string &flow_func_name,
                                  const std::vector<QueueDevInfo> &input_queue_infos,
                                  const std::map<std::string, std::vector<uint32_t>> &multi_func_input_maps,
                                  std::map<std::string, std::vector<QueueDevInfo>> &real_input_queue_maps);

    static int32_t GetUsrInvokedModelKey(const std::string &name_with_scope, const std::string &scope,
                                         std::string &invoke_name);

    static int32_t SubmitEvent(uint32_t group_id, uint32_t event_id, uint32_t sub_event_id);

    int32_t CreateFlowModel(const std::shared_ptr<FlowFuncParams> &params, const FlowFuncModel &model);

    std::vector<std::shared_ptr<FlowFuncProcessor>> func_processors_;
    std::vector<std::shared_ptr<FlowFuncParams>> func_params_;

    // key: input queue id, value : funcProcessors id;
    std::map<uint32_t, size_t> input_to_flow_func_processor_idx_;
    // for multi func, one output queue may have multi flowfuncProcessors
    std::map<uint32_t, std::vector<size_t>> output_to_flow_func_processor_idx_;
    // status output queue, for subscribe
    std::map<int32_t, std::vector<uint32_t>> status_output_queue_map_;
    volatile bool running_ = false;
    volatile bool with_proxy_queue_ = false;
    volatile bool recv_term_signal_ = false;

    std::set<uint32_t> flow_msg_queues_;
    std::map<int32_t, std::map<uint32_t, bool>> dev_input_queue_map_;
    std::map<int32_t, std::map<uint32_t, bool>> dev_output_queue_map_;
    std::set<int32_t> queue_dev_set_;

    FlowFuncThreadPool udf_thread_pool_;
    std::shared_ptr<AsyncExecutor> async_executor_ = nullptr;
    // event process function type
    using EventProcFunc = void (FlowFuncExecutor::*)(const event_info &, const uint32_t);

    // event function map
    const std::map<uint32_t, EventProcFunc> event_proc_func_map_;

    std::shared_ptr<NpuSchedProcessor> npu_sched_processor_ = nullptr;

    void *statistic_timer_handle_ = nullptr;
    void *monitor_parent_timer_handle_ = nullptr;
    void *monitor_term_signal_timer_handle_ = nullptr;
    // esched priority
    int32_t esched_process_priority_;
    int32_t esched_event_priority_;
    uint32_t cpu_num_ = 1U;

    std::mutex suspend_mutex_;
    std::set<uint32_t> suspend_process_ids_;
    std::mutex recover_mutex_;
    std::set<uint32_t> recover_process_ids_;

    std::unique_ptr<QueueWrapper> request_queue_wrapper_;
    std::unique_ptr<QueueWrapper> response_queue_wrapper_;
};
}

#endif // FLOW_FUNC_EXECUTOR_H
