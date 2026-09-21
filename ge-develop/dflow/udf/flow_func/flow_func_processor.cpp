/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_processor.h"
#include <unistd.h>
#include <algorithm>
#include "flow_func_manager.h"
#include "mbuf_flow_msg_queue.h"
#include "flow_func_config_manager.h"
#include "common/inner_error_codes.h"
#include "common/udf_log.h"
#include "common/util.h"
#include "common/common_define.h"
#include "common/scope_guard.h"
#include "reader_writer/queue_wrapper.h"
#include "reader_writer/proxy_queue_wrapper.h"
#include "flow_func_dumper.h"

namespace FlowFunc {
FlowFuncProcessor::FlowFuncProcessor(const std::shared_ptr<FlowFuncParams> &params, const std::string &flow_func_name,
    const std::vector<QueueDevInfo> &input_queue_infos, const std::vector<QueueDevInfo> &all_out_queue_infos,
    const std::vector<uint32_t> &usable_output_indexes, const std::shared_ptr<AsyncExecutor> &async_executor)
    : flow_func_name_(flow_func_name), pp_name_(params->GetName()), cache_output_data_(all_out_queue_infos.size()),
    writers_(all_out_queue_infos.size()), params_(params), input_queue_infos_(input_queue_infos),
    all_output_queue_infos(all_out_queue_infos), usable_output_indexes_(usable_output_indexes),
    set_output_times_(all_out_queue_infos.size()),
    cached_nums_(all_out_queue_infos.size()),
    set_output_times_round_(all_out_queue_infos.size()),
    async_executor_(async_executor) {
    (void)flow_func_info_.append(flow_func_name_).append("[").append(params_->GetInstanceName()).append("]");
}

FlowFuncProcessor::~FlowFuncProcessor() {
    // must reset before handle close.
    (void)FlowFuncManager::Instance().ExecuteByAsyncThread([this]() {
        func_wrapper_.reset();
        return FLOW_FUNC_SUCCESS;
    });
    input_data_.clear();
    CachedBufferClear();
}

int32_t FlowFuncProcessor::SetOutput(uint32_t out_idx, const std::shared_ptr<FlowMsg> &out_msg) {
    if (out_idx >= writers_.size()) {
        UDF_LOG_ERROR("out_idx is over output num, flow_func_info=%s, out_idx=%u, output num=%zu.",
            flow_func_info_.c_str(),
            out_idx,
            writers_.size());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (std::find(usable_output_indexes_.begin(), usable_output_indexes_.end(), out_idx) ==
        usable_output_indexes_.end()) {
        UDF_LOG_ERROR("out_idx[%u] is inconsistent with compile config json file, flow_func_info=%s.", out_idx,
            flow_func_info_.c_str());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (all_output_queue_infos[out_idx].queue_id == Common::kDummyQId) {
        UDF_LOG_INFO("outIdx[%u] queue id is dummy, ignore it", out_idx);
        return FLOW_FUNC_SUCCESS;
    }
    if (out_msg == nullptr) {
        UDF_LOG_WARN(
            "output msg is null, no need publish, flow_func_info=%s, out_idx=%u.", flow_func_info_.c_str(), out_idx);
        return FLOW_FUNC_SUCCESS;
    }

    auto mbuf_flow_msg = std::dynamic_pointer_cast<MbufFlowMsg>(out_msg);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR(
            "not support custom define flow msg now, flow_func_info=%s, out_idx=%u.", flow_func_info_.c_str(), out_idx);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }

    if (mbuf_flow_msg->GetTransactionId() == 0UL) {
        if (is_stream_input_) {
            mbuf_flow_msg->SetTransactionIdInner(set_output_times_[out_idx] + 1);
        } else {
            mbuf_flow_msg->SetTransactionIdInner(current_trans_id_);
        }
    }

    UDF_LOG_DEBUG("output msg[%u]=%s", out_idx, mbuf_flow_msg->DebugString().c_str());
    Mbuf *publish_mbuf = nullptr;
    auto drv_ret = halMbufCopyRef(mbuf_flow_msg->GetMbuf(), &publish_mbuf);
    if ((drv_ret != static_cast<int32_t>(DRV_ERROR_NONE)) || (publish_mbuf == nullptr)) {
        UDF_LOG_ERROR("copy ref mbuf failed, flow_func_info=%s, out_idx=%u.", flow_func_info_.c_str(), out_idx);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }

    // dump output
    auto tensor = dynamic_cast<MbufTensor *>(mbuf_flow_msg->GetTensor());
    const std::string op_name = flow_func_name_ + "_" + pp_name_;
    DumpOutputData(op_name, out_idx, tensor, mbuf_flow_msg->GetStepId());
    flow_func_statistic_.RecordOutput(out_idx, mbuf_flow_msg);
    std::unique_lock<std::mutex> guard(params_->GetOutputQueueLocks(out_idx));
    // if has cache, need cache this data to prevent output out-of-order
    if (cache_output_data_[out_idx].empty()) {
        auto ret = writers_[out_idx]->WriteData(publish_mbuf);
        if (ret != HICAID_SUCCESS) {
            UDF_LOG_WARN("write data failed, cache for republish later, flow_func_info=%s, out_idx=%u, mbuf_len=%lu.",
                flow_func_info_.c_str(), out_idx, mbuf_flow_msg->GetMbufInfo().mbuf_len);
            cache_output_data_[out_idx].emplace_back(publish_mbuf);
            ++cached_nums_[out_idx];
        } else {
            UDF_LOG_INFO("write data success, flow_func_info=%s, out_idx=%u, mbuf_len=%lu.", flow_func_info_.c_str(), out_idx,
                mbuf_flow_msg->GetMbufInfo().mbuf_len);
        }
    } else {
        cache_output_data_[out_idx].emplace_back(publish_mbuf);
        ++cached_nums_[out_idx];
        UDF_LOG_INFO("cache for republish later, flow_func_info=%s, out_idx=%u, mbuf_len=%lu.", flow_func_info_.c_str(),
            out_idx, mbuf_flow_msg->GetMbufInfo().mbuf_len);
    }
    ++set_output_times_round_[out_idx];
    ++set_output_times_[out_idx];
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncProcessor::CreateRunContext(uint32_t device_id) {
    // run_context_ is not null when reinit.
    if (run_context_ != nullptr) {
        return FLOW_FUNC_SUCCESS;
    }

    auto callBack = [this](uint32_t out_idx, const std::shared_ptr<FlowMsg> &out_msg) {
        return SetOutput(out_idx, out_msg);
    };
    try {
        run_context_ = std::make_shared<FlowFuncRunContext>(device_id, params_, callBack, processor_idx_);
    } catch (std::exception &e) {
        UDF_LOG_ERROR("new FlowFuncRunContext failed, flow_func_info=%s, error=%s.", flow_func_info_.c_str(), e.what());
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncProcessor::CreateQueueWrapper(const QueueDevInfo &queue_dev_info,
    std::shared_ptr<QueueWrapper> &queue_wrapper) {
    if (queue_dev_info.is_proxy_queue) {
        queue_wrapper.reset(new(std::nothrow) ProxyQueueWrapper(static_cast<uint32_t>(queue_dev_info.device_id),
            queue_dev_info.queue_id));
    } else {
        queue_wrapper.reset(
            new(std::nothrow) QueueWrapper(static_cast<uint32_t>(queue_dev_info.device_id), queue_dev_info.queue_id));
    }
}

int32_t FlowFuncProcessor::InitReader() {
    if (is_stream_input_) {
        UDF_LOG_DEBUG("FlowFunc has stream inputs, no need create reader, flow_func_info=%s.", flow_func_info_.c_str());
        return FLOW_FUNC_SUCCESS;
    }
    UDF_LOG_DEBUG("begin to create reader for FlowFunc, flow_func_info=%s.", flow_func_info_.c_str());
    std::vector<std::shared_ptr<QueueWrapper>> input_queue_wrappers(input_queue_infos_.size());
    for (size_t input_idx = 0; input_idx < input_queue_infos_.size(); ++input_idx) {
        const auto &input_queue_info = input_queue_infos_[input_idx];
        CreateQueueWrapper(input_queue_info, input_queue_wrappers[input_idx]);
        if (input_queue_wrappers[input_idx] == nullptr) {
            UDF_LOG_ERROR("alloc queue_wrapper failed, flow_func_info=%s, is_proxy_queue=%d, queue_id=%u.",
                flow_func_info_.c_str(),
                static_cast<int32_t>(input_queue_info.is_proxy_queue), input_queue_info.queue_id);
            return FLOW_FUNC_FAILED;
        }
    }

    std::unique_ptr<DataAligner> data_aligner;
    if (align_max_cache_num_ > 0) {
        data_aligner.reset(new(std::nothrow) DataAligner(input_queue_infos_.size(), align_max_cache_num_,
            align_timeout_, drop_when_not_align_));
        if (data_aligner == nullptr) {
            UDF_LOG_ERROR("alloc DataAligner failed, flow_func_info=%s.", flow_func_info_.c_str());
            return FLOW_FUNC_FAILED;
        }
    }
    auto data_callback = [this](std::vector<Mbuf *> &data) { SetInputData(data); };
    reader_.reset(
        new(std::nothrow) MbufReader(flow_func_info_, input_queue_wrappers, data_callback, std::move(data_aligner)));
    if (reader_ == nullptr) {
        UDF_LOG_ERROR("alloc MbufReader failed, flow_func_info=%s.", flow_func_info_.c_str());
        return FLOW_FUNC_FAILED;
    }
    (void)reader_->Init({});
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncProcessor::InitWriter() {
    UDF_LOG_DEBUG("begin to create writer for FlowFunc, flow_func_info=%s, all output queue num=%zu.",
        flow_func_info_.c_str(), all_output_queue_infos.size());
    for (size_t idx = 0UL; idx < all_output_queue_infos.size(); ++idx) {
        const auto &output_queue_info = all_output_queue_infos[idx];
        std::shared_ptr<QueueWrapper> output_queue_wrapper;
        CreateQueueWrapper(output_queue_info, output_queue_wrapper);
        if (output_queue_wrapper == nullptr) {
            UDF_LOG_ERROR("alloc output queue_wrapper failed, flow_func_info=%s, is_proxy_queue=%d, queue_id=%u.",
                flow_func_info_.c_str(),
                static_cast<int32_t>(output_queue_info.is_proxy_queue), output_queue_info.queue_id);
            return FLOW_FUNC_FAILED;
        }
        writers_[idx].reset(new(std::nothrow) MbufWriter(output_queue_wrapper));
        if (writers_[idx].get() == nullptr) {
            UDF_LOG_ERROR(
                "alloc MbufWriter failed, flow_func_info=%s, qid=%u.", flow_func_info_.c_str(), output_queue_info.queue_id);
            return FLOW_FUNC_FAILED;
        }
        (void)writers_[idx]->Init({});
    }

    const auto ret = CreateStatusWriter();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("create status writer failed, flow_func_info=%s.", flow_func_info_.c_str());
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncProcessor::CreateStatusWriter() {
    if (params_->GetNeedReportStatusFlag() || params_->GetEnableRaiseException()) {
        std::shared_ptr<QueueWrapper> status_output_queue_wrapper;
        const auto &status_output_queue = params_->GetStatusOutputQueue();
        CreateQueueWrapper(status_output_queue, status_output_queue_wrapper);
        if (status_output_queue_wrapper == nullptr) {
            UDF_LOG_ERROR("alloc status output queue_wrapper failed, flow_func_info=%s, is_proxy_queue=%d, queue_id=%u.",
                flow_func_info_.c_str(),
                static_cast<int32_t>(status_output_queue.is_proxy_queue), status_output_queue.queue_id);
            return FLOW_FUNC_FAILED;
        }
        status_writer_.reset(new(std::nothrow) MbufWriter(status_output_queue_wrapper));
        if (status_writer_ == nullptr) {
            UDF_LOG_ERROR("alloc status MbufWriter failed, flowFuncInfo=%s, qid=%u.", flow_func_info_.c_str(),
                status_output_queue.queue_id);
            return FLOW_FUNC_FAILED;
        }
        (void)status_writer_->Init({});
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncProcessor::CreateFuncWrapper() {
    return FlowFuncManager::Instance().ExecuteByAsyncThread([this]() {
        func_wrapper_ = FlowFuncManager::Instance().GetFlowFuncWrapper(flow_func_name_, params_->GetInstanceName());
        if (func_wrapper_ == nullptr) {
            UDF_LOG_ERROR("FlowFunc is not found, flow_func_info=%s.", flow_func_info_.c_str());
            return FLOW_FUNC_FAILED;
        }
        return FLOW_FUNC_SUCCESS;
    });
}

int32_t FlowFuncProcessor::InitInputFlowMsgQueues() {
    if (!is_stream_input_) {
        UDF_LOG_DEBUG("FlowFunc has no stream input, no need create flow msg queues, flow_func_info=%s.", 
            flow_func_info_.c_str());
        return FLOW_FUNC_SUCCESS;
    }
    UDF_LOG_DEBUG("begin to create flow msg queues for FlowFunc, flow_func_info=%s.", flow_func_info_.c_str());
    std::vector<std::shared_ptr<QueueWrapper>> input_queue_wrappers(input_queue_infos_.size());
    for (size_t input_idx = 0; input_idx < input_queue_infos_.size(); ++input_idx) {
        const auto &input_queue_info = input_queue_infos_[input_idx];
        CreateQueueWrapper(input_queue_info, input_queue_wrappers[input_idx]);
        if (input_queue_wrappers[input_idx] == nullptr) {
            UDF_LOG_ERROR("alloc queue_wrapper failed, flow_func_info=%s, is_proxy_queue=%d, queue_id=%u.",
                flow_func_info_.c_str(),
                static_cast<int32_t>(input_queue_info.is_proxy_queue), input_queue_info.queue_id);
            return FLOW_FUNC_FAILED;
        }
        std::shared_ptr<MbufFlowMsgQueue> flow_msg_queue =
            MakeShared<MbufFlowMsgQueue>(input_queue_wrappers[input_idx], input_queue_info);
        if (flow_msg_queue == nullptr) {
            UDF_LOG_ERROR("create flow_msg_queue failed, flowFuncInfo=%s, is_proxy_queue=%d, queue_id=%u.",
                flow_func_info_.c_str(),
                static_cast<int32_t>(input_queue_info.is_proxy_queue), input_queue_info.queue_id);
            return FLOW_FUNC_FAILED;
        }
        flow_msg_queues_.emplace_back(flow_msg_queue);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncProcessor::Init(uint32_t device_id) {
    UDF_LOG_DEBUG("begin to get FlowFunc, flow_func_info=%s.", flow_func_info_.c_str());
    int32_t ret = params_->Init();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("flow func params init failed, flow_func_info=%s", flow_func_info_.c_str());
        return ret;
    }
    const auto &stream_input_func_names = params_->GetStreamInputFuncNames();
    if (stream_input_func_names.count(flow_func_name_) > 0) {
        is_stream_input_ = true;
        UDF_LOG_INFO("FlowFunc processor has stream input, flow_func_info=%s.", flow_func_info_.c_str());
    }

    ret = CreateFuncWrapper();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("create func wrapper failed, flow_func_info=%s.", flow_func_info_.c_str());
        return ret;
    }

    if (CreateRunContext(device_id) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("create run context failed, flow_func_info=%s", flow_func_info_.c_str());
        return FLOW_FUNC_FAILED;
    }

    if ((InitReader() != FLOW_FUNC_SUCCESS) || (InitInputFlowMsgQueues() != FLOW_FUNC_SUCCESS)) {
        UDF_LOG_ERROR("init reader or input flow msg queues failed, flow_func_info=%s", flow_func_info_.c_str());
        return FLOW_FUNC_FAILED;
    }
    
    if (InitWriter() != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("init reader and writer failed, flow_func_info=%s", flow_func_info_.c_str());
        return FLOW_FUNC_FAILED;
    }
    flow_func_statistic_.Init(flow_func_info_, input_queue_infos_.size(), all_output_queue_infos.size());
    status_ = FlowFuncProcessorStatus::kInitFlowFunc;
    UDF_RUN_LOG_INFO("end to init FlowFunc processor, flow_func_info=%s.", flow_func_info_.c_str());
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncProcessor::InitFlowFunc() {
    if (status_ == FlowFuncProcessorStatus::kInitFlowFunc) {
        UDF_LOG_DEBUG("FlowFunc init start, flow_func_info=%s.", flow_func_info_.c_str());
        int32_t ret = FlowFuncManager::Instance().ExecuteByAsyncThread(
            [this]() { return func_wrapper_->Init(params_, run_context_); });
        if (ret == FLOW_FUNC_ERR_INIT_AGAIN) {
            UDF_LOG_INFO("FlowFunc need init again, flow_func_info=%s.", flow_func_info_.c_str());
            return FLOW_FUNC_ERR_INIT_AGAIN;
        }
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("FlowFunc init failed, flow_func_info=%s, ret=%d.", flow_func_info_.c_str(), ret);
            return ret;
        }
        UDF_LOG_DEBUG("FlowFunc init success, flow_func_info=%s.", flow_func_info_.c_str());
        status_ = FlowFuncProcessorStatus::kReady;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncProcessor::DumpInputData(const std::string &op_name,
                                      const std::vector<MbufTensor *> &tensors,
                                      uint32_t step_id) const {
    if (!FlowFuncDumpManager::IsEnable()) {
        return;
    }
    if (async_executor_ == nullptr) {
        UDF_LOG_WARN("Async executor should not be null while dump is enable.");
        return;
    }
    auto fut = async_executor_->Commit([&op_name, &tensors, &step_id]() {
        return FlowFuncDumpManager::DumpInput(op_name, tensors, step_id);
    });
    if (fut.get() != FLOW_FUNC_SUCCESS) {
        UDF_LOG_WARN("DumpInput failed, flow_func_info=%s.", flow_func_info_.c_str());
    }
}

void FlowFuncProcessor::DumpOutputData(const std::string &op_name, uint32_t out_idx,
                                       const MbufTensor *tensor, uint32_t step_id) const {
    if (tensor == nullptr) {
        UDF_LOG_INFO("output tensor null, no need dump, flow_func_info=%s, out_idx=%u.", flow_func_info_.c_str(), out_idx);
        return;
    }
    if (!FlowFuncDumpManager::IsEnable()) {
        return;
    }
    if (async_executor_ == nullptr) {
        UDF_LOG_WARN("Async executor should not be null while dump is enable.");
        return;
    }
    auto fut = async_executor_->Commit([&op_name, &out_idx, &tensor, &step_id]() {
        return FlowFuncDumpManager::DumpOutput(op_name, out_idx, tensor, step_id);
    });
    if (fut.get() != FLOW_FUNC_SUCCESS) {
        UDF_LOG_WARN("DumpOutput failed, flow_func_info=%s, out_idx=%u.", flow_func_info_.c_str(), out_idx);
    }
}

void FlowFuncProcessor::SetInputData(std::vector<Mbuf *> &data) {
    UDF_LOG_DEBUG("SetInputData, flow_func_info=%s, input num=%zu.", flow_func_info_.c_str(), data.size());
    if (data.size() != input_queue_infos_.size()) {
        UDF_LOG_ERROR("call back data vector size=%zu is not match inputQueueIds_ size=%zu, flow_func_info=%s.",
            data.size(),
            input_queue_infos_.size(),
            flow_func_info_.c_str());
        return;
    }

    input_data_.resize(input_queue_infos_.size());
    bool is_copy_head_msg = false;
    std::vector<size_t> empty_inputs;
    std::vector<MbufTensor *> input_flow_tensors;
    input_flow_tensors.resize(input_queue_infos_.size());
    for (size_t idx = 0UL; idx < data.size(); ++idx) {
        if (data[idx] == nullptr) {
            empty_inputs.emplace_back(idx);
            input_flow_tensors[idx] = nullptr;
            continue;
        }
        auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
        // mbuf_flow_msg take ownership of mbuf
        std::shared_ptr<Mbuf> mbuf_ptr(data[idx], mbuf_deleter);
        auto mbuf_flow_msg = new (std::nothrow) MbufFlowMsg(mbuf_ptr);
        if (mbuf_flow_msg == nullptr) {
            UDF_LOG_ERROR("new MbufFlowMsg failed, flow_func_info=%s.", flow_func_info_.c_str());
            return;
        }
        // MbufFlowMsg take ownership
        data[idx] = nullptr;
        input_data_[idx].reset(mbuf_flow_msg);
        auto init_ret = mbuf_flow_msg->Init();
        if (init_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("input data[%zu] init failed, ret=%d, flow_func_info=%s.", idx, init_ret, flow_func_info_.c_str());
            return;
        }
        UDF_LOG_INFO("input msg[%zu] mbuf_len=%lu, flow_func_info=%s", idx, mbuf_flow_msg->GetMbufInfo().mbuf_len,
            flow_func_info_.c_str());
        if (params_->IsHead()) {
            mbuf_flow_msg->SetStepId(dump_step_id_);
        }
        dump_step_id_ = mbuf_flow_msg->GetStepId();
        if (!is_copy_head_msg) {
            current_trans_id_ = mbuf_flow_msg->GetTransactionId();
            const auto ret = run_context_->SetInputMbufHead(mbuf_flow_msg->GetMbufInfo());
            if (ret != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Failed to set mbuf head, ret=%d, flow_func_info=%s.", ret, flow_func_info_.c_str());
                return;
            }
            is_copy_head_msg = true;
        }
        flow_func_statistic_.RecordInput(idx, input_data_[idx]);
        UDF_LOG_DEBUG("input msg[%zu]=%s", idx, mbuf_flow_msg->DebugString().c_str());
        auto tensor = dynamic_cast<MbufTensor *>(mbuf_flow_msg->GetTensor());
        if (tensor != nullptr) {
            input_flow_tensors[idx] = tensor;
        } else {
            input_flow_tensors[idx] = nullptr;
            UDF_LOG_INFO("input tensor[%zu] null, flow_func_info=%s.", idx, flow_func_info_.c_str());
        }
    }
    const std::string op_name = flow_func_name_ + "_" + pp_name_;
    DumpInputData(op_name, input_flow_tensors, dump_step_id_);
    if (params_->IsHead()) {
        dump_step_id_++;
    }
    if (!empty_inputs.empty()) {
        if (empty_inputs.size() == input_queue_infos_.size()) {
            UDF_LOG_ERROR("all input msg is null, flow_func_info=%s.", flow_func_info_.c_str());
            return;
        }
        auto empty_data_msg = run_context_->AllocMbufEmptyDataMsg(MsgType::MSG_TYPE_TENSOR_DATA);
        if (empty_data_msg == nullptr) {
            UDF_LOG_ERROR("input %s is empty, but all empty msg failed, flow_func_info=%s.",
                ToString(empty_inputs).c_str(), flow_func_info_.c_str());
            return;
        }
        empty_data_msg->SetRetCodeInner(FLOW_FUNC_ERR_DATA_ALIGN_FAILED);
        for (size_t idx : empty_inputs) {
            input_data_[idx] = empty_data_msg;
        }
        UDF_LOG_WARN("input %s is empty, set to empty msg, retCode=%d, flow_func_info=%s.",
            ToString(empty_inputs).c_str(), FLOW_FUNC_ERR_DATA_ALIGN_FAILED, flow_func_info_.c_str());
    }
    UDF_LOG_INFO("input is ready, change state to CALL_FLOW_FUNC, flow_func_info=%s, trans_id=%lu, input num=%zu.",
        flow_func_info_.c_str(), current_trans_id_, input_data_.size());
    status_ = FlowFuncProcessorStatus::kCallFlowFunc;
}

bool FlowFuncProcessor::PrepareInputs() {
    UDF_LOG_DEBUG("prepare inputs.");
    do {
        ClearNotEmptyEventFlag();
        // ReadMessage callback may change status.
        reader_->ReadMessage();
        if (!reader_->StatusOk()) {
            UDF_LOG_ERROR("reader status is error, flow_func_info=%s.", flow_func_info_.c_str());
            status_ = FlowFuncProcessorStatus::kProcError;
            return false;
        }
        if (status_ == FlowFuncProcessorStatus::kPrepareInputData) {
            UDF_LOG_DEBUG("after ReadMessage, status is still PREPARE_INPUT_DATA, flow_func_info=%s.",
                flow_func_info_.c_str());
            if (CheckAndSetWaitNotEmpty()) {
                // do read again.
                continue;
            }
            return false;
        }
        break;
    } while (true);
    return true;
}

void FlowFuncProcessor::ProcCallFlowFuncFailed() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    if (try_clear_and_suspend_) {
        TryClearAndSuspend();
    } else {
        UDF_LOG_ERROR("call flow func failed, flow_func_info=%s, trans_id=%lu, ret=%d.",
            flow_func_info_.c_str(), current_trans_id_, proc_ret_);
        status_ = FlowFuncProcessorStatus::kProcError;
    }
}

void FlowFuncProcessor::RecordDeleteException(uint64_t trans_id) {
    std::unique_lock<std::mutex> guard(clear_exp_mutex_);
    UDF_LOG_DEBUG("Record delete exception trans_id=%lu.", trans_id);
    (void)delete_exp_trans_ids_.insert(trans_id);
}

void FlowFuncProcessor::CallFlowFuncProcExp() {
    UDF_LOG_DEBUG("Start to call flow func to process exception.");
    std::unique_lock<std::mutex> guard(exp_mutex_);
    {
        std::unique_lock<std::mutex> clear_guard(clear_exp_mutex_);
        for (uint64_t trans_id : delete_exp_trans_ids_) {
            const auto iter = trans_id_to_exception_infos_.find(trans_id);
            if (iter != trans_id_to_exception_infos_.cend()) {
                trans_id_to_exception_infos_.erase(iter);
            }
            reader_->DeleteExceptionTransId(trans_id);
        }
        delete_exp_trans_ids_.clear();
    }

    if (trans_id_to_exception_infos_.empty()) {
        UDF_LOG_DEBUG("There is no valid exception need to be notified.");
        status_ = FlowFuncProcessorStatus::kRepublishOutputData;
        return;
    }
    const auto iter = trans_id_to_exception_infos_.begin();
    // 1.notify trans id to align module
    reader_->AddExceptionTransId(iter->first);

    // 2.set mbuf info to context header
    UdfExceptionInfo &exception_info = iter->second;
    MbufInfo mbuf_info = {};
    mbuf_info.head_buf = static_cast<void *>(exception_info.exp_context);
    mbuf_info.head_buf_len = exception_info.exp_context_size;
    auto ret = run_context_->SetInputMbufHead(mbuf_info);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to set mbuf head, ret=%d, flow_func_info=%s.", ret, flow_func_info_.c_str());
        ProcCallFlowFuncFailed();
        return;
    }
    // 3.record exception to context
    run_context_->RecordException(exception_info);

    flow_func_statistic_.ExecuteStart();
    std::vector<std::shared_ptr<FlowMsg>> input_data;
    proc_ret_ = func_wrapper_->Proc(input_data);
    if (proc_ret_ != FLOW_FUNC_SUCCESS) {
        bool need_retry = false;
        (void)RepublishOutputs(need_retry);
        UDF_LOG_ERROR("Call flow func to process exception failed.");
        ProcCallFlowFuncFailed();
        return;
    }

    flow_func_statistic_.ExecuteFinish();
    UDF_LOG_INFO("Call flow func to proc exception end, flow_func_info=%s, exception trans_id=%lu.",
        flow_func_info_.c_str(), iter->first);
    trans_id_to_exception_infos_.erase(iter);
    status_ = FlowFuncProcessorStatus::kRepublishOutputData;
    UDF_LOG_DEBUG("Finish to call flow func to process exception.");
}

void FlowFuncProcessor::CallFlowFunc() {
    ExecuteFunc();
    if (proc_ret_ == FLOW_FUNC_ERR_PROC_PENDING) {
        UDF_LOG_INFO("flow func return pending, flow_func_info=%s, trans_id=%lu, ret=%d.", flow_func_info_.c_str(),
            current_trans_id_, proc_ret_);
        return;
    }
    if (proc_ret_ != FLOW_FUNC_SUCCESS) {
        ProcCallFlowFuncFailed();
        return;
    }

    flow_func_statistic_.ExecuteFinish();
    UDF_LOG_INFO("call flow func end, flow_func_info=%s, trans_id=%lu.", flow_func_info_.c_str(), current_trans_id_);
    input_data_.clear();
    status_ = FlowFuncProcessorStatus::kRepublishOutputData;
    return;
}

bool FlowFuncProcessor::RepublishOutputs(bool &need_retry) {
    bool all_republish_success = true;
    bool has_full_event = false;
    do {
        all_republish_success = true;
        ClearNotFullEventFlag();
        for (size_t i = 0UL; i < cache_output_data_.size(); i++) {
            if (!RepublishOutputs(i, need_retry)) {
                all_republish_success = false;
            }
        }
        has_full_event = false;
        // if not finish, check full event
        if (!all_republish_success) {
            has_full_event = CheckAndSetWaitNotFull();
        }
    } while (has_full_event);

    if (all_republish_success) {
        UDF_LOG_INFO("all output is published, flow_func_info=%s.", flow_func_info_.c_str());
        status_ = FlowFuncProcessorStatus::kScheduleFinish;
    }
    return all_republish_success;
}

bool FlowFuncProcessor::RepublishOutputs(size_t out_idx, bool &need_retry) {
    std::unique_lock<std::mutex> guard(params_->GetOutputQueueLocks(out_idx));
    auto &out_list = cache_output_data_[out_idx];
    if (out_list.empty()) {
        return true;
    }

    for (auto &out_mbuf : out_list) {
        if (out_mbuf == nullptr) {
            continue;
        }
        int32_t ret = writers_[out_idx]->WriteData(out_mbuf);
        if (ret != HICAID_SUCCESS) {
            if (ret == HICAID_ERR_QUEUE_FULL) {
                UDF_LOG_INFO("Retry to write data while queue is full, flow_func_info=%s, out_idx=%zu.",
                             flow_func_info_.c_str(), out_idx);
                need_retry = need_retry || writers_[out_idx]->NeedRetry();
            } else {
                status_ = FlowFuncProcessorStatus::kProcError;
                UDF_LOG_ERROR("WriteData failed, flow_func_info=%s, ret=%d, out_idx=%zu.", flow_func_info_.c_str(), ret,
                    out_idx);
            }
            // republish next time.
            return false;
        }
        --cached_nums_[out_idx];
        // if publish success, reset to null, otherwise may be republish later
        out_mbuf = nullptr;
    }
    out_list.clear();
    UDF_LOG_DEBUG("output[%zu] republish finished, flow_func_info=%s.", out_idx, flow_func_info_.c_str());
    return true;
}

bool FlowFuncProcessor::NeedReplenishSchedule() const {
    if ((status_ != FlowFuncProcessorStatus::kPrepareInputData) &&
        (status_ != FlowFuncProcessorStatus::kRepublishOutputData)) {
        return false;
    }

    if (last_schedule_time_ == std::chrono::steady_clock::time_point()) {
        return false;
    }

    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = current_time - last_schedule_time_;
    int64_t elapsed_second = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    // if schedule over 1500 ms. need replenish schedule
    constexpr const int64_t kNeedReplenishMilliSec = 1500;
    if (elapsed_second > kNeedReplenishMilliSec) {
        UDF_LOG_INFO("last schedule %ld ms passed, over need replenish %ld ms, flow_func_info=%s", elapsed_second,
            kNeedReplenishMilliSec, flow_func_info_.c_str());
        return true;
    }
    return false;
}

bool FlowFuncProcessor::Schedule(uint32_t thread_idx) {
    bool need_reschedule = false;
    // if in schedule, set wait schedule flag, otherwise do schedule
    if (!schedule_control_lock_.test_and_set(std::memory_order_acquire)) {
        // clear wait_schedule_flag_ before schedule
        while (wait_schedule_lock_.test_and_set(std::memory_order_acquire)) {
            // spin lock
        }
        wait_schedule_flag_ = false;
        wait_schedule_lock_.clear(std::memory_order_release);

        need_reschedule = DoSchedule(thread_idx);
        ScheduleCallback(need_reschedule);
    } else {
        while (wait_schedule_lock_.test_and_set(std::memory_order_acquire)) {
            // spin lock
        }
        wait_schedule_flag_ = true;
        wait_schedule_lock_.clear(std::memory_order_release);
        // if schedule_control_lock_ release, need re schedule
        if (!schedule_control_lock_.test_and_set(std::memory_order_acquire)) {
            schedule_control_lock_.clear(std::memory_order_release);
            need_reschedule = true;
        }
    }
    return need_reschedule;
}

void FlowFuncProcessor::ScheduleCallback(bool &need_reschedule) {
    schedule_control_lock_.clear(std::memory_order_release);
    if (!need_reschedule) {
        // check whether the schedule come
        if (!wait_schedule_lock_.test_and_set(std::memory_order_acquire)) {
            need_reschedule = wait_schedule_flag_;
            wait_schedule_lock_.clear(std::memory_order_release);
        }
    }
}

bool FlowFuncProcessor::NeedProcException() {
    {
        std::unique_lock<std::mutex> guard(exp_mutex_);
        if (!delete_exp_trans_ids_.empty()) {
            return true;
        }
    }
    {
        std::unique_lock<std::mutex> clear_guard(clear_exp_mutex_);
        if (!trans_id_to_exception_infos_.empty()) {
            return true;
        }
    }
    return false;
}

bool FlowFuncProcessor::DoSchedule(uint32_t thread_idx) {
    UDF_LOG_DEBUG("Schedule start, flow_func_info=%s, status=%d, thread_idx=%u.", flow_func_info_.c_str(),
        static_cast<int32_t>(status_), thread_idx);
    bool need_reschedule = false;
    last_schedule_time_ = std::chrono::steady_clock::now();
    if (status_ < FlowFuncProcessorStatus::kReady) {
        UDF_LOG_WARN("processor is not ready, need reschedule later, flow_func_info=%s, status=%d.",
            flow_func_info_.c_str(), static_cast<int32_t>(status_));
        return true;
    }
    if (PreCheckSpecialStatus()) {
        UDF_LOG_DEBUG("Current do schedule loop is suspend or recover procedure.");
        return false;
    }

    if (status_ == FlowFuncProcessorStatus::kSuspend) {
        DiscardAllInputData();
    }

    if (status_ == FlowFuncProcessorStatus::kReady) {
        OnScheduleReady();
    }

    if (status_ == FlowFuncProcessorStatus::kPrepareInputData) {
        if (!OnPrepareInput(need_reschedule)) {
            return need_reschedule;
        }
    }

    if (status_ == FlowFuncProcessorStatus::kCallFlowFuncProcExp) {
        CallFlowFuncProcExp();
    }

    if (status_ == FlowFuncProcessorStatus::kCallFlowFunc) {
        CallFlowFunc();
        ++call_flow_func_times_;
    }

    if (status_ == FlowFuncProcessorStatus::kProcError) {
        UDF_LOG_ERROR("flow func status is proc error, no need schedule, flow_func_info=%s.", flow_func_info_.c_str());
        return false;
    }

    if (status_ == FlowFuncProcessorStatus::kRepublishOutputData) {
        if (!RepublishOutputs(need_reschedule)) {
            UDF_LOG_DEBUG("republish output data not finish, flow_func_info=%s.", flow_func_info_.c_str());
            return need_reschedule;
        }
    }

    if (status_ == FlowFuncProcessorStatus::kScheduleFinish) {
        OnScheduleFinish();
        need_reschedule = true;
    }
    UDF_LOG_DEBUG("Schedule end, flow_func_info=%s, status=%d.", flow_func_info_.c_str(), static_cast<int32_t>(status_));
    return need_reschedule;
}

void FlowFuncProcessor::OnScheduleReady() {
    InitScheduleVar();
    status_ = FlowFuncProcessorStatus::kPrepareInputData;
}

bool FlowFuncProcessor::OnPrepareInput(bool &need_retry) {
    // if has exception need priority handle it
    if (NeedProcException()) {
        if (is_stream_input_) {
            UDF_LOG_ERROR("Report or proc exception is not supported when func has stream input", flow_func_info_.c_str());
            status_ = FlowFuncProcessorStatus::kProcError;
            return false;
        }
        status_ = FlowFuncProcessorStatus::kCallFlowFuncProcExp;
        UDF_LOG_INFO("There are some exceptions need to be processed");
        return true;
    }

    if (input_queue_infos_.empty()) {
        status_ = FlowFuncProcessorStatus::kCallFlowFunc;
        UDF_LOG_INFO("input is empty, no need prepare input, flow_func_info=%s", flow_func_info_.c_str());
    } else if (is_stream_input_) {
        status_ = FlowFuncProcessorStatus::kCallFlowFunc;
        UDF_LOG_INFO("func has stream inputs, no need prepare input, flow_func_info=%s", flow_func_info_.c_str());
        return true;
    } else {
        if (!PrepareInputs()) {
            UDF_LOG_DEBUG("inputs not ready, flow_func_info=%s.", flow_func_info_.c_str());
            need_retry = reader_->NeedRetry();
            return false;
        }
    }
    ++input_ready_times_;
    return true;
}

void FlowFuncProcessor::OnScheduleFinish() {
    UDF_LOG_INFO("schedule finished, flow_func_info=%s, trans_id=%lu, set output in schedule times=%s, out queues=%s.",
        flow_func_info_.c_str(), current_trans_id_, ToString(set_output_times_round_).c_str(),
        ToString(all_output_queue_infos).c_str());
    ++schedule_finish_times_;
    if (params_->GetNeedReportStatusFlag()) {
        SendEventToExecutor<int32_t>(UdfEvent::kEventIdFlowFuncReportStatus, FLOW_FUNC_SUCCESS);
    }
    status_ = FlowFuncProcessorStatus::kReady;
}

bool FlowFuncProcessor::CheckAndSetWaitNotEmpty() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    if (not_empty_event_) {
        wait_not_empty_event_ = false;
        not_empty_event_ = false;
        return true;
    }
    wait_not_empty_event_ = true;
    return false;
}

bool FlowFuncProcessor::CheckAndSetWaitNotFull() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    if (not_full_event_) {
        wait_not_full_event_ = false;
        not_full_event_ = false;
        return true;
    }
    wait_not_full_event_ = true;
    return false;
}

bool FlowFuncProcessor::EmptyToNotEmpty() {
    bool need_reschedule = false;
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    if (wait_not_empty_event_) {
        wait_not_empty_event_ = false;
        need_reschedule = true;
    } else {
        not_empty_event_ = true;
    }
    return need_reschedule;
}

bool FlowFuncProcessor::FullToNotFull() {
    bool need_reschedule = false;
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    if (wait_not_full_event_) {
        wait_not_full_event_ = false;
        need_reschedule = true;
    } else {
        not_full_event_ = true;
    }
    return need_reschedule;
}

void FlowFuncProcessor::ClearNotEmptyEventFlag() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    not_empty_event_ = false;
}

void FlowFuncProcessor::ClearNotFullEventFlag() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    not_full_event_ = false;
}

void FlowFuncProcessor::InitScheduleVar() {
    for (auto &times : set_output_times_round_) {
        times.store(0);
    }
    for (auto &cache_num : cached_nums_) {
        cache_num.store(0);
    }
}

void FlowFuncProcessor::DumpFlowFuncInfo(bool with_queue_info) const {
    UDF_RUN_LOG_INFO("flow_func_info=%s, status=%d, last trans_id=%lu, statistic info: input ready times=%lu,"
                     " call flow func times=%lu, schedule finish times=%lu, set output times=%s, cached nums=%s.",
        flow_func_info_.c_str(), static_cast<int32_t>(status_), current_trans_id_, input_ready_times_.load(),
        call_flow_func_times_.load(), schedule_finish_times_.load(), ToString(set_output_times_).c_str(),
        ToString(cached_nums_).c_str());

    if (with_queue_info) {
        if (reader_ != nullptr) {
            reader_->DumpReaderStatus();
        }
        DumpFlowFuncQueueInfo();
    }
}

void FlowFuncProcessor::DumpFlowFuncQueueInfo() const {
    for (size_t input_idx = 0; input_idx < input_queue_infos_.size(); ++input_idx) {
        const auto &input_queue = input_queue_infos_[input_idx];
        UDF_RUN_LOG_INFO(
            "flowFuncInfo=%s input[%zu] queue info=[queue:%u, device_id:%u, device_type:%u], input ready times=%lu",
            flow_func_info_.c_str(), input_idx, input_queue.queue_id, input_queue.deploy_device_id, input_queue.device_type,
            input_ready_times_.load());
    }

    for (size_t output_index = 0; output_index < all_output_queue_infos.size(); ++output_index) {
        const auto &output_queue = all_output_queue_infos[output_index];
        UDF_RUN_LOG_INFO("flow_func_info=%s output[%zu] queue info=[queue:%u, device_id:%u, device_type:%u], "
                         "set output times=%lu, cached nums=%lu",
            flow_func_info_.c_str(), output_index, output_queue.queue_id, output_queue.deploy_device_id, output_queue.device_type,
            set_output_times_[output_index].load(), cached_nums_[output_index].load());
    }
}

void FlowFuncProcessor::DumpModelMetrics(bool with_queue_info) const {
    flow_func_statistic_.DumpMetrics(with_queue_info);
}

void FlowFuncProcessor::ExecuteFunc() {
    UDF_LOG_INFO("call flow func start, flow_func_info=%s, trans_id=%lu, input num=%zu.", flow_func_info_.c_str(),
        current_trans_id_, input_data_.size());
    flow_func_statistic_.ExecuteStart();
    // if worker num is 1, means flow func init and proc in the same thread
    if (FlowFuncConfigManager::GetConfig()->GetWorkerNum() == 1U) {
        const uint32_t sched_group_id = FlowFuncConfigManager::GetConfig()->GetCurrentSchedGroupId();
        proc_ret_ = FlowFuncManager::Instance().ExecuteByAsyncThread([this, sched_group_id]() {
            FlowFuncConfigManager::GetConfig()->SetCurrentSchedGroupId(sched_group_id);
            return is_stream_input_ ? func_wrapper_->Proc(flow_msg_queues_) : func_wrapper_->Proc(input_data_);
        });
    } else {
        proc_ret_ = is_stream_input_ ? func_wrapper_->Proc(flow_msg_queues_) : func_wrapper_->Proc(input_data_);
    }
}

template <typename T>
void FlowFuncProcessor::SendEventToExecutor(uint32_t event_id, T msg)
{
    UDF_LOG_INFO("Send report status event begin, processorIdx=%u.", processor_idx_);
    event_summary event_info_summary = {};
    event_info_summary.pid = getpid();
    event_info_summary.event_id = static_cast<EVENT_ID>(event_id);
    event_info_summary.subevent_id = processor_idx_;
    event_info_summary.msg = reinterpret_cast<char *>(&msg);
    event_info_summary.msg_len = static_cast<uint32_t>(sizeof(T));
    event_info_summary.dst_engine = FlowFuncConfigManager::GetConfig()->IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
    uint32_t group_id =
        (event_id == UdfEvent::kEventIdFlowFuncExecute ? FlowFuncConfigManager::GetConfig()->GetWorkerSchedGroupId()
                                                         : FlowFuncConfigManager::GetConfig()->GetMainSchedGroupId());
    event_info_summary.grp_id = group_id;
    drvError_t ret = halEschedSubmitEvent(params_->GetDeviceId(), &event_info_summary);
    if (ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        UDF_LOG_WARN("Send report status event failed, processor_idx=%u, ret=%d.",
            processor_idx_, static_cast<int32_t>(ret));
    } else {
        UDF_LOG_INFO("Send report status event success, processor_idx=%u.", processor_idx_);
    }
}

void FlowFuncProcessor::RecordExceptionInfo(const UdfExceptionInfo &exp_info) {
    if (run_context_->SelfRaiseChkAndDel(exp_info.trans_id)) {
        UDF_LOG_INFO("TransId[%lu] is raised by current udf.", exp_info.trans_id);
        return;
    }
    std::unique_lock<std::mutex> guard(exp_mutex_);
    trans_id_to_exception_infos_[exp_info.trans_id] = exp_info;
    UDF_LOG_DEBUG("Record exception finished, trans_id=%lu.", exp_info.trans_id);
}

bool FlowFuncProcessor::CheckSameScope(const std::string &scope) const {
    if (scope != params_->GetScope()) {
        UDF_LOG_INFO("Current udf scope[%s] and exception scope[%s] are different. skip to record exception.",
                     scope.c_str(), params_->GetScope().c_str());
        return false;
    }
    return true;
}

int32_t FlowFuncProcessor::WriteStatusOutputQueue(const ReportStatusMbufGenFunc &report_status_mbuf_gen_func) {
    if (params_->GetNeedReportStatusFlag()) {
        const std::lock_guard<std::mutex> lk(params_->GetReportStatusMutexRef());
        input_consume_num_++;
        Mbuf *mbuf = nullptr;
        const auto gen_mbuf_ret = report_status_mbuf_gen_func(input_queue_infos_,
            params_->GetModelUuid(), input_consume_num_, mbuf);
        if ((gen_mbuf_ret != FLOW_FUNC_SUCCESS) || (mbuf == nullptr)) {
            UDF_LOG_ERROR("execute report status mbuf func failed.");
            return FLOW_FUNC_FAILED;
        }
        auto mbuf_deleter = [mbuf]() { (void)halMbufFree(mbuf); };
        ScopeGuard mbuf_guard(mbuf_deleter);
        // enqueue
        const auto ret = status_writer_->WriteData(mbuf);
        if (ret == HICAID_ERR_QUEUE_FULL) {
            UDF_LOG_DEBUG("Status queue is full.");
            return FLOW_FUNC_SUCCESS;
        } else if (ret != HICAID_SUCCESS) {
            UDF_LOG_ERROR("Failed to enqueue by hicaid writer, ret[%d].", ret);
            return FLOW_FUNC_FAILED;
        }
        input_consume_num_ = 0U;
        mbuf_guard.ReleaseGuard();
    } else {
        UDF_LOG_ERROR("flowFuncProcessor can't report status.");
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncProcessor::WriteStatusOutputQueue(uint64_t trans_id,
    const ReportExceptionMbufGenFunc &report_exception_mbuf_gen_func) {
    if (!params_->GetEnableRaiseException()) {
        UDF_LOG_ERROR("flowFuncProcessor can't report exception result of exception catch is disable.");
        return FLOW_FUNC_FAILED;
    }
    {
        std::unique_lock<std::mutex> guard(exp_mutex_);
        if (trans_id_to_exception_infos_.find(trans_id) != trans_id_to_exception_infos_.cend()) {
            UDF_LOG_INFO("TransId[%lu] has already been raised by other UDF. Skip to raise again.", trans_id);
            return FLOW_FUNC_SUCCESS;
        }
    }
    // mbuf header can not be got while input is empty
    if (input_queue_infos_.empty()) {
        UDF_LOG_ERROR("Current udf is not allowed raising exception result of inputs is empty.");
        return FLOW_FUNC_FAILED;
    }
    UdfExceptionInfo exception_info{};
    if (run_context_->GetExceptionByTransId(trans_id, exception_info) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Waiting to report exception is empty.");
        return FLOW_FUNC_FAILED;
    }
    Mbuf *mbuf = nullptr;
    const auto gen_mbuf_ret = report_exception_mbuf_gen_func(params_->GetScope(), exception_info, mbuf);
    if ((gen_mbuf_ret != FLOW_FUNC_SUCCESS) || (mbuf == nullptr)) {
        UDF_LOG_ERROR("execute report status mbuf func failed.");
        return FLOW_FUNC_FAILED;
    }
    auto mbuf_deleter = [mbuf]() { (void)halMbufFree(mbuf); };
    ScopeGuard mbuf_guard(mbuf_deleter);

    const std::lock_guard<std::mutex> lk(params_->GetReportStatusMutexRef());
    const auto ret = status_writer_->WriteData(mbuf);
    if (ret == HICAID_ERR_QUEUE_FULL) {
        UDF_LOG_INFO("Try to send raise exception event again while status queue is full. trans id[%lu]", trans_id);
        SendEventToExecutor<uint64_t>(UdfEvent::kEventIdRaiseException, trans_id);
        return FLOW_FUNC_SUCCESS;
    } else if (ret != HICAID_SUCCESS) {
        UDF_LOG_ERROR("Failed to enqueue status queue writer, ret[%d].", ret);
        return FLOW_FUNC_FAILED;
    }
    mbuf_guard.ReleaseGuard();
    if (!is_stream_input_) {
        reader_->AddExceptionTransId(trans_id);
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncProcessor::CachedBufferClear() {
    for (auto &out_list : cache_output_data_) {
        if (out_list.empty()) {
            continue;
        }

        for (auto &out_mbuf : out_list) {
            if (out_mbuf == nullptr) {
                continue;
            }
            (void)halMbufFree(out_mbuf);
            out_mbuf = nullptr;
        }
        out_list.clear();
    }
    cache_output_data_.clear();
}

void FlowFuncProcessor::ResetProcessor() {
    CachedBufferClear();
    input_data_.clear();
    current_trans_id_ = UINT64_MAX;
    wait_schedule_flag_ = false;
    input_ready_times_  = 0UL;
    call_flow_func_times_  = 0UL;
    schedule_finish_times_  = 0UL;
    for (auto &set_times : set_output_times_) {
        set_times = 0UL;
    }
    for (auto &cache_num : cached_nums_) {
        cache_num = 0UL;
    }
    for (auto &set_times_round : set_output_times_round_) {
        set_times_round = 0UL;
    }
    input_consume_num_ = 0U;
    if (reader_ != nullptr) {
        reader_->Reset();
    }
}

void FlowFuncProcessor::ReleaseFuncWrapper() {
    (void)FlowFuncManager::Instance().ExecuteByAsyncThread([this]() {
        func_wrapper_.reset();
        return FLOW_FUNC_SUCCESS;
    });
}

void FlowFuncProcessor::DiscardAllInputData() {
    if (!is_stream_input_) {
        reader_->DiscardAllInputData();
        if (!reader_->StatusOk()) {
            UDF_LOG_ERROR("reader clear all input data error, flow_func_info=%s.", flow_func_info_.c_str());
            status_ = FlowFuncProcessorStatus::kProcError;
            return;
        }
    } else {
        for (const auto &flow_msg_queue : flow_msg_queues_) {
            auto mbuf_flow_msg_queue = std::dynamic_pointer_cast<MbufFlowMsgQueue>(flow_msg_queue);
            mbuf_flow_msg_queue->DiscardAllInputData();
            if (!mbuf_flow_msg_queue->StatusOk()) {
                UDF_LOG_ERROR("flow_msg_queue clear all input data error, flow_func_info=%s.", flow_func_info_.c_str());
                status_ = FlowFuncProcessorStatus::kProcError;
                return; 
            }
        }
    }
    status_ = FlowFuncProcessorStatus::kSuspend;
}

void FlowFuncProcessor::SetClearAndSuspend() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    try_clear_and_suspend_ = true;
}

void FlowFuncProcessor::SetClearAndRecover() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    try_clear_and_recover_ = true;
}

void FlowFuncProcessor::TryClearAndSuspend() {
    UDF_LOG_INFO("Process try to suspend start.");
    ResetProcessor();
    DiscardAllInputData();
    uint32_t ret = FLOW_FUNC_SUCCESS;
    if (IsOk()) {
        status_ = FlowFuncProcessorStatus::kSuspend;
    } else {
        ret = FLOW_FUNC_SUSPEND_FAILED;
    }
    try_clear_and_suspend_ = false;
    SendEventToExecutor<int32_t>(UdfEvent::kEventIdFlowFuncSuspendFinished, ret);
    UDF_LOG_INFO("Process try to suspend finished.");
}

void FlowFuncProcessor::TryClearAndRecover() {
    UDF_LOG_INFO("Process try to recover start.");
    DiscardAllInputData();
    int32_t ret = FLOW_FUNC_SUCCESS;
    if (IsOk()) {
        // when func support reset state, func_wrapper_ no need recreate.
        if (func_wrapper_ == nullptr) {
            ret = CreateFuncWrapper();
            if (ret != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Create func wrapper failed, flow_func_info=%s.", flow_func_info_.c_str());
                ret = FLOW_FUNC_RECOVER_FAILED;
                status_ = FlowFuncProcessorStatus::kProcError;
            } else {
                status_ = FlowFuncProcessorStatus::kInitFlowFunc;
                SendEventToExecutor<int32_t>(UdfEvent::kEventIdSingleFlowFuncInit, ret);
            }
        } else {
            status_ = FlowFuncProcessorStatus::kReady;
            SendEventToExecutor<int32_t>(UdfEvent::kEventIdFlowFuncExecute, ret);
        }
    } else {
        ret = FLOW_FUNC_RECOVER_FAILED;
    }
    try_clear_and_recover_ = false;
    SendEventToExecutor<int32_t>(UdfEvent::kEventIdFlowFuncRecoverFinished, ret);
    UDF_LOG_INFO("Process try to recover finished.");
}

bool FlowFuncProcessor::PreCheckSpecialStatus() {
    std::unique_lock<std::mutex> guard(queue_event_guard_);
    if ((!try_clear_and_suspend_) && (!try_clear_and_recover_)) {
        return false;
    } else if (try_clear_and_suspend_ && (!try_clear_and_recover_)) {
        TryClearAndSuspend();
        return true;
    } else if (try_clear_and_recover_ && (!try_clear_and_suspend_)) {
        TryClearAndRecover();
        return true;
    } else {
        UDF_LOG_ERROR("Suspend status and recover status can not be both true. Recover should be done after suspend.");
        status_ = FlowFuncProcessorStatus::kProcError;
        return false;
    }
    return false;
}
}  // namespace FlowFunc
