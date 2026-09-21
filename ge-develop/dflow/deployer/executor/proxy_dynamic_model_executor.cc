/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/proxy_dynamic_model_executor.h"
#include "ge/ge_data_flow_api.h"
#include "common/utils/heterogeneous_profiler.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/scope_guard.h"
#include "dflow/base/deploy/exchange_service.h"
#include "cpu_tasks.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "dflow/inc/data_flow/model/graph_model.h"

namespace ge {
namespace {
constexpr size_t kDefaultMbufLen = 8UL;
constexpr uint32_t kDynamicFlag = 2U;
constexpr int64_t kDynamicTensorSize = 0;
constexpr int32_t kMsgQueueDequeueTimeout = -1; // no timeout
constexpr int32_t kMsgQueueEnqueueTimeout = 10 * 60 * 1000; // 10 min
constexpr int32_t kRetryInterval = 1000; // 1000 ms
constexpr int32_t kPrintRetryInterval = 1000;
constexpr uint32_t kDummyQId = UINT32_MAX;
const std::string kPostProcessV2 = "postprocessDynamicOutputV2";
}

ProxyDynamicModelExecutor::ProxyDynamicModelExecutor() : DynamicModelExecutor(false) {
}

Status ProxyDynamicModelExecutor::Initialize() {
  HeterogeneousProfiler::Instance().InitHeterogeneousPoriler();
  return SUCCESS;
}

void ProxyDynamicModelExecutor::Finalize() {
  return;
}

Status ProxyDynamicModelExecutor::SetNpuModelLoaderOutputInfo() {
  std::vector<int64_t> filtered_tensor_sizes;
  std::vector<uint32_t> filtered_output_dynamic_flags;
  std::vector<uint32_t> filtered_output_queue_ids;
  bool is_need_check = false;
  for (size_t i = 0; i < output_queue_attrs_.size(); ++i) {
    // range is checked before.
    if ((output_queue_attrs_[i].queue_id != kDummyQId) && (is_output_dynamic_[i]) && (output_tensor_sizes_[i] == -1)) {
      is_need_check = true;
      break;
    }
  }
  bool post_v2_support = false;
  if (is_need_check) {
    if (CpuTasks::ExecuteCheckSupported(kPostProcessV2, post_v2_support) != SUCCESS) {
      GELOGW("CheckKernelSupported kernel fail maybe result of kernel not supported.");
    } else {
      GELOGI("Current version support postprocessDynamicOutputV2 flag is [%d].", static_cast<int32_t>(post_v2_support));
    }
    loader_.SetEnablePostProcessV2Flag(post_v2_support);
  }

  for (size_t i = 0; i < output_queue_attrs_.size(); ++i) {
    // range is checked before.
    if (output_queue_attrs_[i].queue_id != kDummyQId) {
      if ((!post_v2_support) && (is_output_dynamic_[i]) && (output_tensor_sizes_[i] == -1)) {
        GELOGE(FAILED, "Max output size should be set while current version does not support aicpu post V2 task.");
        return FAILED;
      }
      if (post_v2_support && (is_output_dynamic_[i])) {
        // aicpu kernel support v2
        filtered_tensor_sizes.emplace_back(kDynamicTensorSize);
        filtered_output_dynamic_flags.emplace_back(kDynamicFlag);
        GELOGD("Set output[%zu] tesor size[%d] and dynamic flag[%u] while current version support aicpu post V2 task.",
               i, kDynamicTensorSize, kDynamicFlag);
      } else {
        filtered_tensor_sizes.emplace_back(output_tensor_sizes_[i]);
        filtered_output_dynamic_flags.emplace_back(static_cast<uint32_t>(is_output_dynamic_[i]));
        GELOGD("Set output[%zu] tensor size[%d] and dynamic flag[%u].",
               i, output_tensor_sizes_[i], static_cast<uint32_t>(is_output_dynamic_[i]));
      }
      filtered_output_queue_ids.emplace_back(output_queue_attrs_[i].queue_id);
    }
  }
  loader_.SetOutputTensorSizes(filtered_tensor_sizes);
  loader_.SetOutputDynamicFlags(filtered_output_dynamic_flags);
  loader_.SetOutputQueueIds(filtered_output_queue_ids);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::LoadWithAicpuSd(const ComputeGraphPtr &root_graph,
                                                  const ModelQueueParam &model_queue_param) {
  GE_CHK_BOOL_RET_STATUS(num_outputs_ == output_queue_attrs_.size(), FAILED,
                         "Invalid! num_outputs[%zu] != output_queues_num[%zu].", num_outputs_, output_queues_num_);
  model_name_ = root_graph->GetName();
  GELOGD("Begin to load model with aicpu sd, device_id = %d, model_id = %u, model_name = %s.",
         device_id_, model_id_, model_name_.c_str());
  loader_.SetModelId(model_id_);
  loader_.SetDeviceId(device_id_);
  if (!align_attrs_.empty()) {
    uint32_t align_interval;
    std::vector<uint32_t> align_offsets;
    GE_CHK_STATUS_RET_NOLOG(CheckAndGetAlignAttr(align_interval, align_offsets));
    loader_.SetAlignAttributes(align_interval, align_offsets);
  }
  loader_.SetInputDynamicFlags(is_input_dynamic_);
  GE_CHK_STATUS_RET(SetNpuModelLoaderOutputInfo(), "Set npu model loader output info failed.");
  loader_.SetOutputStaticTensorDescs(output_static_runtime_tensor_descs_);
  loader_.SetGlobalStep(global_step_);
  loader_.SetInputAlignAttrs(input_align_attrs_);
  GE_CHK_STATUS_RET(loader_.LoadModel(model_queue_param, runtime_model_id_),
                    "Fail to load model, device_id = %d, model_id = %u.", device_id_, model_id_);
  req_msg_queue_id_ = static_cast<int32_t>(loader_.GetReqMsgQueueId());
  resp_msg_queue_id_ = static_cast<int32_t>(loader_.GetRespMsgQueueId());
  // calculate expected outputs and inputs len in req mbuf
  const size_t dynamic_inputs_num = std::count_if(is_input_dynamic_.begin(), is_input_dynamic_.end(),
                                                  [](uint32_t flag) { return flag;});
  req_inputs_len_ = dynamic_inputs_num * sizeof(RuntimeTensorDesc) +
                    (num_inputs_ - dynamic_inputs_num) * sizeof(uint64_t);
  const size_t valid_output_num = std::count_if(output_queue_attrs_.begin(), output_queue_attrs_.end(),
      [](const QueueAttrs &queue_attrs) { return queue_attrs.queue_id != UINT32_MAX; });
  req_outputs_len_ = valid_output_num * sizeof(uint64_t);
  const size_t dynamic_outputs_num = std::count_if(is_output_dynamic_.begin(), is_output_dynamic_.end(),
                                                   [](uint32_t flag) { return flag;});
  resp_total_len_ = (dynamic_outputs_num == 0UL) ? kDefaultMbufLen : sizeof(RuntimeTensorDesc) * dynamic_outputs_num;
  GE_CHK_STATUS_RET(StartDispatcherThread(), "Fail to start dispatcher thread, "
                    "device_id = %d, model_id = %u, runtime_model_id = %u.", device_id_, model_id_, runtime_model_id_);
  GEEVENT("Success to load model with aicpu sd, req_inputs_len = %lu, dynamic_inputs_num = %zu, inputs_num = %zu, "
          "resp_total_len_ = %lu, dynamic_outputs_num = %zu, outputs_num = %zu, valid outputs num = %zu, "
          "device_id = %d, model_id = %u, runtime_model_id = %u, model_name = %s.",
          req_inputs_len_, dynamic_inputs_num, num_inputs_, resp_total_len_, dynamic_outputs_num, num_outputs_,
          valid_output_num, device_id_, model_id_, runtime_model_id_, model_name_.c_str());
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::UnloadFromAicpuSd() {
  GELOGD("Begin to unload model from aicpu sd, device_id = %d, model_id = %u, runtime_model_id = %u.",
         device_id_, model_id_, runtime_model_id_);
  dispatcher_running_flag_ = false;
  if (dispatcher_thread_.joinable()) {
    dispatcher_thread_.join();
  }
  (void) loader_.UnloadModel();
  GELOGD("Success to unload model from aicpu sd, device_id = %d, model_id = %u, runtime_model_id = %u.",
         device_id_, model_id_, runtime_model_id_);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::CheckInputs() {
  is_need_execute_model_ = true;
  data_ret_code_ = 0;
  void *req_mbuf = model_execute_param_.req_mbuf;
  GE_CHECK_NOTNULL(req_mbuf);
  void *head_buf = nullptr;
  uint64_t head_size = 0U;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(reinterpret_cast<rtMbufPtr_t>(req_mbuf), &head_buf, &head_size));
  GE_CHECK_NOTNULL(head_buf);
  GE_CHECK_GE(head_size, sizeof(ExchangeService::MsgInfo));
  ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
      static_cast<char_t *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
  if (msg_info->ret_code != 0) {
    is_need_execute_model_ = false;
    data_ret_code_ = msg_info->ret_code;
    GEEVENT(
        "No need to execute model, trans_id = %lu, data ret code = %d, device_id = %d, "
        "model_id = %u, runtime_model_id = %u.",
        msg_info->trans_id, data_ret_code_, device_id_, model_id_, runtime_model_id_);
    return SUCCESS;
  }
  // is null data with eos
  const bool is_null_data_input = ((msg_info->data_flag & kNullDataFlagBit) != 0U);
  if (is_null_data_input) {
    is_need_execute_model_ = false;
    data_ret_code_ = 0;
    GELOGI("Input is null data, trans_id = %lu, device_id = %d, model_id = %u, runtime_model_id = %u.",
           msg_info->trans_id, device_id_, model_id_, runtime_model_id_);
    return SUCCESS;
  }
  is_need_execute_model_ = true;
  GELOGD("Input is ok, trans_id = %lu, device_id = %d, model_id = %u, runtime_model_id = %u.",
         msg_info->trans_id, device_id_, model_id_, runtime_model_id_);
  void *data_buff = nullptr;
  uint64_t buff_size = 0UL;
  GE_CHK_RT_RET(rtMbufGetBuffAddr(req_mbuf, &data_buff));
  GE_CHK_RT_RET(rtMbufGetBuffSize(req_mbuf, &buff_size));
  const size_t expected_req_len = req_outputs_len_ + req_inputs_len_;
  GE_CHK_BOOL_RET_STATUS(buff_size == expected_req_len, FAILED, "Fail to check inputs, expected_req_len = %zu, "
                         "buff_size = %lu, device_id = %d, model_id = %u, runtime_model_id = %u.",
                         expected_req_len, buff_size, device_id_, model_id_, runtime_model_id_);
  GELOGD("Success to Check inputs, expected_req_len = %zu, buff_size = %lu, device_id = %d, model_id = %u, "
         "runtime_model_id = %u.", expected_req_len, buff_size, device_id_, model_id_, runtime_model_id_);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::PrepareInputs(std::vector<DataBuffer> &model_inputs) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kPrepareInputs,
                                                                     device_id_);
  void *req_mbuf = model_execute_param_.req_mbuf;
  void *req_data_buff = nullptr;
  GE_CHK_RT_RET(rtMbufGetBuffAddr(req_mbuf, &req_data_buff));
  uint64_t req_data_offset = 0UL;
  for (size_t i = 0UL; i < num_inputs_; ++i) {
    DataBuffer data_buffer;
    auto &tensor_desc = input_tensor_descs_[i];
    if (is_input_dynamic_[i]) {
      const RuntimeTensorDesc *runtime_tensor_desc =
          PtrToPtr<void, const RuntimeTensorDesc>(ValueToPtr((PtrToValue(req_data_buff) + req_data_offset)));
      GE_CHK_STATUS_RET(UpdateTensorDesc(*runtime_tensor_desc, tensor_desc),
                        "Failed to update tensor desc, input index = %zu, device_id = %d, model_id = %u, "
                        "runtime_model_id = %u.", i, device_id_, model_id_, runtime_model_id_);
      int64_t tensor_size = 0L;
      GE_CHK_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc.GetShape(), tensor_desc.GetFormat(),
                                                       tensor_desc.GetDataType(), tensor_size),
                        "Failed to calc output size, shape = [%s]", tensor_desc.GetShape().ToString().c_str());
      GELOGD("Inputs[%zu] is dynamic, shape = [%s], original shape = [%s], tensor size = %ld, tensor addr = 0x%lx, "
             "device_id = %d, model_id = %u, runtime_model_id = %u.",
             i, tensor_desc.GetShape().ToString().c_str(), tensor_desc.GetOriginShape().ToString().c_str(),
             tensor_size, runtime_tensor_desc->data_addr, device_id_, model_id_, runtime_model_id_);
      data_buffer.data = ValueToPtr(runtime_tensor_desc->data_addr);
      data_buffer.length = runtime_tensor_desc->data_size;
      req_data_offset += sizeof(RuntimeTensorDesc);
    } else {
      const uint64_t *addr = PtrToPtr<void, const uint64_t>(ValueToPtr(PtrToValue(req_data_buff) + req_data_offset));
      data_buffer.data = ValueToPtr(*addr);
      data_buffer.length = input_tensor_sizes_[i];
      GELOGI("Inputs[%zu] is not dynamic, shape = [%s], original shape = [%s], tensor_size = %ld, tensor addr = 0x%lx, "
             "device_id = %d, model_id = %u, runtime_model_id = %u.",
             i, tensor_desc.GetShape().ToString().c_str(), tensor_desc.GetOriginShape().ToString().c_str(),
             input_tensor_sizes_[i], *addr, device_id_, model_id_, runtime_model_id_);
      req_data_offset += sizeof(uint64_t);
    }
    GE_CHECK_NOTNULL(data_buffer.data);
    data_buffer.placement = kPlacementDevice;
    model_inputs.emplace_back(data_buffer);
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kPrepareInputs,
                                                                     device_id_);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::PrepareOutputs(std::vector<DataBuffer> &model_outputs) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kPrepareOutputs,
                                                                     device_id_);
  void *req_mbuf = model_execute_param_.req_mbuf;
  void *req_data_buff = nullptr;
  GE_CHK_RT_RET(rtMbufGetBuffAddr(req_mbuf, &req_data_buff));
  uint64_t req_data_offset = req_inputs_len_;
  for (size_t i = 0UL; i < num_outputs_; ++i) {
    auto tensor_size = output_tensor_sizes_[i];
    if (tensor_size < 0) { // no valid range
      GELOGD("Output[%zu] is dynamic and cannot get a valid size by range.", i);
      model_outputs.emplace_back(DataBuffer{});
      continue;
    }
    DataBuffer data_buffer;
    data_buffer.length = tensor_size;
    data_buffer.placement = kPlacementDevice;
    if (output_queue_attrs_[i].queue_id != kDummyQId) {
      const uint64_t *addr = PtrToPtr<void, const uint64_t>(ValueToPtr(PtrToValue(req_data_buff) + req_data_offset));
      data_buffer.data = ValueToPtr(*addr);
      req_data_offset += sizeof(uint64_t);
      GELOGD(
          "Output[%zu], is dynamic = %d, tensor size = %zu, tensor addr = 0x%lx, device_id = %d, model_id = %u, "
          "runtime_model_id = %u.",
          i, static_cast<int32_t>(is_output_dynamic_[i]), tensor_size, *addr, device_id_, model_id_, runtime_model_id_);
    }
    model_outputs.emplace_back(data_buffer);
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kPrepareOutputs,
                                                                     device_id_);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::UpdateOutputs(std::vector<DataBuffer> &model_outputs) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kUpdateOutputs,
                                                                     device_id_);
  std::vector<RuntimeTensorDesc> dynamic_output_tensor_descs;
  for (size_t i = 0UL; i < num_outputs_; ++i) {
    if (!is_output_dynamic_[i]) {
      continue;
    }
    if ((i < output_queue_attrs_.size()) && (output_queue_attrs_[i].queue_id == kDummyQId)) {
      continue;
    }
    auto &tensor_desc = output_tensor_descs_[i];
    GE_CHK_STATUS_RET_NOLOG(UpdateRuntimeTensorDesc(tensor_desc, output_runtime_tensor_descs_[i]));
    int64_t tensor_size = 0L;
    GE_CHK_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc.GetShape(), tensor_desc.GetFormat(),
                                                     tensor_desc.GetDataType(), tensor_size),
                      "Fail to calculate output size, shape = [%s]", tensor_desc.GetShape().ToString().c_str());
    output_runtime_tensor_descs_[i].data_size = static_cast<uint64_t>(tensor_size);
    output_runtime_tensor_descs_[i].data_addr = PtrToValue(model_outputs[i].data);
    dynamic_output_tensor_descs.emplace_back(output_runtime_tensor_descs_[i]);
    GELOGI("Output[%zu] is dynamic, tensor size = %ld, device_id = %d, model_id = %u, runtime_model_id = %u,"
           "model_name = %s, output data addr = %p.", i, tensor_size, device_id_, model_id_,
           runtime_model_id_, model_name_.c_str(), model_outputs[i].data);
  }
  size_t buffer_size = resp_total_len_;
  if (!dynamic_output_tensor_descs.empty()) {
    void *buffer_addr = nullptr;
    GE_CHK_RT_RET(rtMbufGetBuffAddr(model_execute_param_.resp_mbuf, &buffer_addr));
    GE_CHECK_NOTNULL(buffer_addr);
    GE_CHK_BOOL_RET_STATUS(memcpy_s(static_cast<uint8_t *>(buffer_addr), buffer_size,
                                    dynamic_output_tensor_descs.data(),
                                    dynamic_output_tensor_descs.size() * sizeof(RuntimeTensorDesc)) == EOK,
                           FAILED, "Failed to copy output tensor descs.");
  }
  GELOGD("Success to copy dynamic_output_tensor_descs, size = %zu, device_id = %d, model_id = %u, "
         "runtime_model_id = %u.", buffer_size, device_id_, model_id_, runtime_model_id_);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kUpdateOutputs);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::PublishOutputWithoutExecute() {
  const size_t dynamic_output_num =
      std::count_if(is_output_dynamic_.begin(), is_output_dynamic_.end(), [](uint32_t flag) { return flag; });
  void *resp_head_buff = nullptr;
  uint64_t resp_head_size = 0UL;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(model_execute_param_.resp_mbuf, &resp_head_buff, &resp_head_size));
  // resp_head_buff/resp_head_size have been checked in CopyMbufHead
  ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
      static_cast<char_t *>(resp_head_buff) + resp_head_size - sizeof(ExchangeService::MsgInfo));
  msg_info->ret_code = data_ret_code_;

  void *buffer_addr = nullptr;
  GE_CHK_STATUS_RET(rtMbufGetBuffAddr(model_execute_param_.resp_mbuf, &buffer_addr));
  GE_CHECK_NOTNULL(buffer_addr);
  for (size_t i = 0UL; i < dynamic_output_num; i++) {
    RuntimeTensorDesc *runtime_tensor_desc = reinterpret_cast<RuntimeTensorDesc *>(buffer_addr);
    // empty tensor shape: dim_num = 1; dim_value = 0;
    runtime_tensor_desc->shape[0] = 1L;
    runtime_tensor_desc->shape[1] = 0L;
    runtime_tensor_desc->original_shape[0] = 1L;
    runtime_tensor_desc->original_shape[1] = 0L;
    runtime_tensor_desc->data_size = 0;
  }
  GELOGI(
      "Success to publish output without execute, trans_id = %lu, device_id = %d, model_id = %u, "
      "runtime_model_id = %u, ret_code=%d.",
      msg_info->trans_id, device_id_, model_id_, runtime_model_id_, data_ret_code_);
  return SUCCESS;
}

void ProxyDynamicModelExecutor::PublishErrorOutput(Status ret) {
  data_ret_code_ = ret;
  (void)PublishOutputWithoutExecute();
}

void ProxyDynamicModelExecutor::UpdateFusionInputsAddr() {
  return;
}

Status ProxyDynamicModelExecutor::StartDispatcherThread() {
  dispatcher_running_flag_ = true;
  dispatcher_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_dpl_pdisp");
    Dispatcher();
  });
  return SUCCESS;
}

void ProxyDynamicModelExecutor::Dispatcher() {
  GELOGI("Dispatcher thread started, device_id = %d, model_id = %u, runtime_model_id = %u.",
         device_id_, model_id_, runtime_model_id_);
  while (dispatcher_running_flag_) {
    rtMbufPtr_t req_msg_mbuf = nullptr;
    rtMbufPtr_t resp_msg_mbuf = nullptr;
    const auto ret = PrepareMsgMbuf(req_msg_mbuf, resp_msg_mbuf);
    if (ret != SUCCESS) {
      if (!dispatcher_running_flag_) {
        GEEVENT("Exit dispatcher thread, device_id = %d, model_id = %u, runtime_model_id = %u.",
                device_id_, model_id_, runtime_model_id_);
      } else {
        dispatcher_running_flag_ = false;
        GELOGE(FAILED, "Fail to prepare msg buf, ret = %d, device_id = %d, model_id = %u, runtime_model_id = %u.",
               ret, device_id_, model_id_, runtime_model_id_);
      }
      return;
    }
    if (OnInputsReady(req_msg_mbuf, resp_msg_mbuf) != SUCCESS) {
      (void) rtMbufFree(req_msg_mbuf);
      (void) rtMbufFree(resp_msg_mbuf);
      dispatcher_running_flag_ = false;
      GELOGE(FAILED, "Fail to ready inputs, device_id = %d, model_id = %u, runtime_model_id = %u.",
             device_id_, model_id_, runtime_model_id_);
      return;
    }
  }
  GELOGI("Dispatcher thread stopped, device_id = %d, model_id = %u, runtime_model_id = %u.",
         device_id_, model_id_, runtime_model_id_);
}

Status ProxyDynamicModelExecutor::PrepareMsgMbuf(rtMbufPtr_t &req_msg_mbuf, rtMbufPtr_t &resp_msg_mbuf) const {
  GE_CHK_STATUS_RET_NOLOG(DoDequeue(device_id_, static_cast<uint32_t>(req_msg_queue_id_),
                                    req_msg_mbuf, kMsgQueueDequeueTimeout));
  GE_CHECK_NOTNULL(req_msg_mbuf);
  GE_DISMISSABLE_GUARD(req_msg_mbuf, [req_msg_mbuf]() {rtMbufFree(req_msg_mbuf);});
  GE_CHK_RT_RET(rtMbufAlloc(&resp_msg_mbuf, resp_total_len_));
  GE_CHECK_NOTNULL(resp_msg_mbuf);
  GE_DISMISSABLE_GUARD(resp_msg_mbuf, [resp_msg_mbuf]() {rtMbufFree(resp_msg_mbuf);});
  GE_CHK_RT_RET(rtMbufSetDataLen(resp_msg_mbuf, resp_total_len_));
  GE_CHK_RT_RET(CopyMbufHead(req_msg_mbuf, resp_msg_mbuf));
  GE_DISMISS_GUARD(req_msg_mbuf);
  GE_DISMISS_GUARD(resp_msg_mbuf);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::DoDequeue(const int32_t device_id, const uint32_t queue_id,
                                            rtMbufPtr_t &mbuf, const int32_t timeout) const {
  int32_t retry_num = 0;
  const int32_t total_timeout = timeout;
  int32_t left_wait_time = total_timeout;
  GELOGD("Begin to do dequeue req msg mbuf, device_id = %d, queue_id = %u, timeout = %dms",
         device_id, queue_id, total_timeout);
  Status status = SUCCESS;
  while (dispatcher_running_flag_) {
    // -1 means always wait when queue is empty.
    const int32_t wait_time = ((total_timeout == -1) || (left_wait_time >= kRetryInterval)) ?
                              kRetryInterval : left_wait_time;
    status = DequeueMbuf(device_id, queue_id, mbuf, wait_time);
    if (status == SUCCESS) {
      break;
    }
    if (status == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
      left_wait_time = (left_wait_time >= wait_time) ? (left_wait_time - wait_time) : left_wait_time;
      if ((total_timeout == -1) || (left_wait_time > 0)) {
        if (retry_num++ % kPrintRetryInterval == 0) {
          GELOGD("Retry to dequeue req msg queue, model_id = %u, runtime_model_id = %u, queue_id = %u, retry_num = %d.",
                 model_id_, runtime_model_id_, queue_id, retry_num);
        }
        continue;
      }
      GELOGE(FAILED, "Dequeue req msg queue timeout, model_id = %u, runtime_model_id = %u, queue_id = %u, "
             "total_timeout = %dms, left_wait_time = %dms",
             model_id_, runtime_model_id_, queue_id, total_timeout, left_wait_time);
      status = FAILED;
      break;
    } else {
      GELOGE(FAILED, "Dequeue req msg queue failed, model_id = %u, runtime_model_id = %u, queue_id = %u, ret = %u",
             model_id_, runtime_model_id_, queue_id, status);
      status = FAILED;
      break;
    }
  }
  GELOGD("Finish to do dequeue req msg queue, model_id = %u, runtime_model_id = %u, queue_id = %u, device_id = %d, "
         "retry_num = %d, status = %d, running flag = %d.",
         model_id_, runtime_model_id_, queue_id, device_id, retry_num, status, dispatcher_running_flag_.load());
  return status;
}

Status ProxyDynamicModelExecutor::DequeueMbuf(const int32_t device_id, const uint32_t queue_id,
                                              rtMbufPtr_t &mbuf, const int32_t timeout) const {
  GELOGD("Begin to dequeue mbuf, device_id = %d, queue_id = %u, timeout = %dms.", device_id, queue_id, timeout);
  uint64_t data_buffer_size = 0U;
  const auto ret = rtMemQueuePeek(device_id, queue_id, &data_buffer_size, timeout);
  if (ret != RT_ERROR_NONE) {
    GELOGD("Finish to call rtMemQueuePeek, device_id = %d, queue_id = %u, timeout = %dms, ret = %d.",
           device_id, queue_id, timeout, ret);
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  GE_CHK_RT_RET(rtMbufAlloc(&mbuf, data_buffer_size));
  GE_CHECK_NOTNULL(mbuf);
  GE_DISMISSABLE_GUARD(mbuf, [mbuf]() {rtMbufFree(mbuf);});
  uint64_t head_size = 0U;
  void *control_data = nullptr;
  void *data_buffer = nullptr;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(mbuf, &control_data, &head_size));
  GE_CHK_RT_RET(rtMbufGetBuffAddr(mbuf, &data_buffer));
  rtMemQueueBuffInfo queue_buff_info = {data_buffer, data_buffer_size};
  rtMemQueueBuff_t queue_buff = {control_data, head_size, &queue_buff_info, 1U};
  GE_CHK_RT_RET(rtMemQueueDeQueueBuff(device_id, queue_id, &queue_buff, timeout));
  GE_DISMISS_GUARD(mbuf);
  GELOGD("Success to dequeue mbuf, device_id = %d, queue_id = %u, timeout = %dms.", device_id, queue_id, timeout);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::EnqueueMbuf(const int32_t device_id, const uint32_t queue_id,
                                              rtMbufPtr_t mbuf, const int32_t timeout) const {
  GELOGD("Begin to enqueue mbuf, device_id = %d, queue_id = %u, timeout = %dms.", device_id, queue_id, timeout);
  void *control_data = nullptr;
  void *data_buffer = nullptr;
  uint64_t head_size = 0UL;
  uint64_t data_buffer_size = 0UL;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(mbuf, &control_data, &head_size));
  GE_CHK_RT_RET(rtMbufGetBuffAddr(mbuf, &data_buffer));
  GE_CHK_RT_RET(rtMbufGetBuffSize(mbuf, &data_buffer_size));
  rtMemQueueBuffInfo queue_buf_info = {data_buffer, data_buffer_size};
  rtMemQueueBuff_t queue_buf = {control_data, head_size, &queue_buf_info, 1U};
  GE_CHK_RT_RET(rtMemQueueEnQueueBuff(device_id, queue_id, &queue_buf, timeout));
  GELOGD("Success to enqueue mbuf, device_id = %d, queue_id = %u, timeout = %dms.", device_id, queue_id, timeout);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::OnInputsReady(rtMbufPtr_t req_msg_mbuf, rtMbufPtr_t resp_msg_mbuf) {
  GELOGD("Begin to activate model, device_id = %d, model_id = %u, runtime_model_id = %u.",
         device_id_, model_id_, runtime_model_id_);
  auto callback = [this](Status status, void *req, void *resp) {
    OnModelExecuted(status, req, resp);
    if (status != SUCCESS) {
      GELOGE(FAILED, "Execute model failed, model_id = %u.", model_id_);
    }
  };
  GE_CHK_STATUS_RET(ExecuteAsync(callback, req_msg_mbuf, resp_msg_mbuf),
                    "Failed to submit inner model execute task, device_id = %d, model id = %u, runtime_model_id = %u.",
                    device_id_, model_id_, runtime_model_id_);
  GELOGD("Success to activate model, device_id = %d, model_id = %u, runtime_model_id = %u.",
         device_id_, model_id_, runtime_model_id_);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::OnModelExecuted(const Status status, rtMbufPtr_t req_msg_mbuf,
                                                  rtMbufPtr_t resp_msg_mbuf) const {
  (void) status;
  const auto ret = EnqueueMbuf(device_id_, static_cast<uint32_t>(resp_msg_queue_id_),
                               resp_msg_mbuf, kMsgQueueEnqueueTimeout);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Enqueue failed, ret = %d, device_id = %d, queue_id = %d, model_id = %u, runtime_model_id = %u.",
           ret, device_id_, resp_msg_queue_id_, model_id_, runtime_model_id_);
  } else {
    GELOGD("Success to notify model execute result, device_id = %d, model_id = %u, runtime_model_id = %d, "
           "model_name = %s.", device_id_, model_id_, runtime_model_id_, model_name_.c_str());
  }
  (void) rtMbufFree(req_msg_mbuf);
  (void) rtMbufFree(resp_msg_mbuf);
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::ClearModel(const int32_t clear_type) {
  GE_CHK_STATUS_RET_NOLOG(ClearModelInner(clear_type));
  return SUCCESS;
}

Status ProxyDynamicModelExecutor::ExceptionNotify(uint32_t type, uint64_t trans_id) {
  (void)type;
  (void)trans_id;
  // ProxyDynamicModelExecutor send exception notify with multi model ids by CpuTasks
  return SUCCESS;
}

uint32_t ProxyDynamicModelExecutor::GetRuntimeModelId() const {
  return runtime_model_id_;
}
}
